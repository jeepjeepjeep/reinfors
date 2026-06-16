//! The `Planner` trait — the swappable *algorithm* seam. A planner turns the live value network into
//! (a) per-decision action values for the rollout and (b) the training targets for executed
//! trajectories. `SelectiveTreeStrap` (selective expectimax + z-mixing) is the first impl; a
//! model-free planner (e.g. ensemble DQN) is another, slotting into the same rollout `Engine` without
//! touching it. The Engine owns the game-agnostic framework (parallel rollout, ensemble Thompson +
//! epsilon action choice, bootstrap masks, replay, telemetry); the Planner owns what is
//! algorithm-specific (here: the search, the interior TreeStrap targets, and the z-mix).

use crate::engine::blend_outcome_targets;
use crate::game::Game;
use crate::search::{search_many, SearchConfig, SearchParams, SearchResult};

/// How an algorithm evaluates states and builds training targets, driving the rollout `Engine`. The
/// trait is game-agnostic: `evaluate` is generic over the game, and the target methods don't touch it.
pub trait Planner {
    /// Pooled evaluation of a batch of `(state, agent)` requests with the live net (`infer`): per
    /// request, the per-head root action values `[K][A]`, any extra training records to emit
    /// immediately (TreeStrap interior nodes), and search diagnostics. (Thompson-head/epsilon action
    /// choice on top of the returned values is the Engine's job, not the planner's.)
    fn evaluate<G, F>(
        &self,
        game: &G,
        requests: Vec<(G::State, usize)>,
        infer: &mut F,
    ) -> Vec<SearchResult>
    where
        G: Game + Sync,
        G::State: Send,
        F: FnMut(Vec<f32>, usize) -> Vec<f64>;

    /// Per-step training targets for one executed trajectory at episode end. `trajectory` is
    /// time-ordered `(searched values [K][A], executed action, realized reward)`; `tail` (len K) seeds
    /// the bootstrap past the last step (0 at a terminal, the net's per-head state value at a
    /// truncation). Returns one `[K][A]` target per step, in time order. For TreeStrap this is z-mixing.
    fn targets(
        &self,
        trajectory: &[(Vec<Vec<f64>>, usize, f64)],
        tail: &[f64],
    ) -> Vec<Vec<Vec<f64>>>;

    /// Whether `targets` consumes the per-head bootstrap value of the final state (the z-tail). When
    /// false the Engine skips computing it (a forward). Default: false.
    fn uses_episode_tail(&self) -> bool {
        false
    }
}

/// Selective expectimax + TreeStrap — today's algorithm. Holds the (game-agnostic) search config, the
/// z-mix `outcome_weight`, and whether to collect interior TreeStrap targets.
pub struct SelectiveTreeStrap {
    cfg: SearchConfig,
    outcome_weight: f64,
    collect_interior: bool,
}

impl SelectiveTreeStrap {
    pub fn new(search: &SearchParams, outcome_weight: f64, collect_interior: bool) -> Self {
        SelectiveTreeStrap {
            cfg: SearchConfig::from_params(search),
            outcome_weight,
            collect_interior,
        }
    }
}

impl Planner for SelectiveTreeStrap {
    fn evaluate<G, F>(
        &self,
        game: &G,
        requests: Vec<(G::State, usize)>,
        infer: &mut F,
    ) -> Vec<SearchResult>
    where
        G: Game + Sync,
        G::State: Send,
        F: FnMut(Vec<f32>, usize) -> Vec<f64>,
    {
        search_many(
            game,
            &self.cfg,
            requests,
            self.collect_interior,
            &mut *infer,
        )
    }

    fn targets(
        &self,
        trajectory: &[(Vec<Vec<f64>>, usize, f64)],
        tail: &[f64],
    ) -> Vec<Vec<Vec<f64>>> {
        blend_outcome_targets(trajectory, self.cfg.gamma, self.outcome_weight, tail)
    }

    fn uses_episode_tail(&self) -> bool {
        self.outcome_weight > 0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::{SnakeGame, SnakeState};
    use crate::reward::Reward;
    use crate::search::{selective_search, Opponent};
    use crate::snake::SnakeEnv;

    const G: i32 = 8;

    fn params() -> SearchParams {
        SearchParams {
            grid_size: G,
            initial_length: 3,
            play_to_last: false,
            win_food_lead: None,
            gamma: 0.99,
            beta: 1.0,
            expansion_budget: 24,
            top_k: 4,
            max_depth: 6,
            food_samples: 1,
            reward: Reward {
                step: 0.0,
                food: 1.0,
                loss: -10.0,
                draw: -5.0,
                kill: 5.0,
                win: 10.0,
                survival: 0.0,
            },
            opponent: Opponent::Uniform,
        }
    }

    fn snake_game() -> SnakeGame {
        SnakeGame {
            grid_size: G,
            initial_length: 3,
            play_to_last: false,
            win_food_lead: None,
            initial_food_count: 0,
            reward: params().reward,
        }
    }

    // Two disagreeing heads, sum-dependent — flat (obs[n*dim], n) -> values[n*2*3], head-major.
    fn infer(obs: Vec<f32>, n: usize) -> Vec<f64> {
        let dim = obs.len() / n;
        let mut out = Vec::with_capacity(n * 2 * 3);
        for i in 0..n {
            let s = obs[i * dim..(i + 1) * dim].iter().sum::<f32>() as f64;
            out.extend_from_slice(&[
                s.sin(),
                s.cos(),
                (s * 0.5).sin(),
                (s + 1.0).sin(),
                (s * 0.3).cos(),
                (s * 0.2).sin(),
            ]);
        }
        out
    }

    fn state() -> SnakeState {
        let env = SnakeEnv::new(G, 3, false, None);
        SnakeState {
            snakes: env.snakes,
            food: std::collections::HashSet::new(),
        }
    }

    #[test]
    fn evaluate_wraps_the_pooled_search() {
        let planner = SelectiveTreeStrap::new(&params(), 0.3, false);
        let st = state();
        let results = planner.evaluate(&snake_game(), vec![(st.clone(), 0)], &mut infer);
        let (values, _i, _s) = selective_search(
            &params(),
            st.snakes.clone(),
            st.food.clone(),
            0,
            false,
            &mut infer,
        );
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, values);
    }

    #[test]
    fn targets_z_mix_and_tail_usage_track_outcome_weight() {
        // targets == blend_outcome_targets at the planner's outcome_weight; uses_episode_tail iff > 0.
        let traj = vec![(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]], 1usize, 10.0)];
        let tail = [0.0, 0.0];
        let p = SelectiveTreeStrap::new(&params(), 0.25, false);
        assert_eq!(
            p.targets(&traj, &tail),
            blend_outcome_targets(&traj, 0.99, 0.25, &tail)
        );
        assert!(p.uses_episode_tail());
        let p0 = SelectiveTreeStrap::new(&params(), 0.0, false);
        assert!(!p0.uses_episode_tail());
    }
}
