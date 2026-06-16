//! The `Planner` trait — the swappable *algorithm* seam. A planner turns the live value network into
//! (a) per-decision action values for the rollout and (b) the training targets for executed
//! trajectories. `SelectiveTreeStrap` (selective expectimax + z-mixing) is the first impl; a
//! model-free planner (e.g. ensemble DQN) is another, slotting into the same rollout `Engine` without
//! touching it. The Engine owns the game-agnostic framework (parallel rollout, ensemble Thompson +
//! epsilon action choice, bootstrap masks, replay, telemetry); the Planner owns what is
//! algorithm-specific (here: the search, the interior TreeStrap targets, and the z-mix).

use crate::engine::blend_outcome_targets;
use crate::game::Game;
use crate::search::{search_many, SearchConfig, SearchResult};

/// How an algorithm evaluates states and builds training targets, driving the rollout `Engine`. The
/// trait is game-agnostic: `evaluate` is generic over the game, and the target methods don't touch it.
pub trait Planner {
    /// Pooled evaluation of a batch of `(state, agent)` requests with the live net (`infer`): per
    /// request, the per-head root action values `[K][A]`, any extra training records to emit
    /// immediately (TreeStrap interior nodes), and search diagnostics. (Thompson-head/epsilon action
    /// choice on top of the returned values is the Engine's job, not the planner's.) `seed` seeds any
    /// stochastic chance sampling the search does, so a caller controls reproducibility.
    fn evaluate<G, F>(
        &self,
        game: &G,
        requests: Vec<(G::State, usize)>,
        seed: u64,
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
    pub fn new(cfg: SearchConfig, outcome_weight: f64, collect_interior: bool) -> Self {
        SelectiveTreeStrap {
            cfg,
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
        seed: u64,
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
            seed,
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
    use crate::search::Opponent;

    fn cfg() -> SearchConfig {
        SearchConfig {
            gamma: 0.99,
            beta: 1.0,
            expansion_budget: 24,
            top_k: 4,
            max_depth: 6,
            food_samples: 1,
            opponent: Opponent::Uniform,
        }
    }

    #[test]
    fn targets_z_mix_and_tail_usage_track_outcome_weight() {
        // targets == blend_outcome_targets at the planner's outcome_weight; uses_episode_tail iff > 0.
        let traj = vec![(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]], 1usize, 10.0)];
        let tail = [0.0, 0.0];
        let p = SelectiveTreeStrap::new(cfg(), 0.25, false);
        assert_eq!(
            p.targets(&traj, &tail),
            blend_outcome_targets(&traj, 0.99, 0.25, &tail)
        );
        assert!(p.uses_episode_tail());
        let p0 = SelectiveTreeStrap::new(cfg(), 0.0, false);
        assert!(!p0.uses_episode_tail());
    }
}
