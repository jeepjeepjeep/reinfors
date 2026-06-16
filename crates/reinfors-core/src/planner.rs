//! The `Planner` trait — how an agent *acts* (and, later, *learns*) given a `Game` and the value
//! network. It is the seam that lets the search algorithm be swapped: `SelectiveTreeStrap` (the
//! selective expectimax used today) is the first impl; a model-free planner (e.g. ensemble DQN) will
//! be another, slotting into the same rollout/training loop without touching it.
//!
//! Step 1 covers *action selection* only — the half cleanly available before the search is made
//! generic over `Game`. The training-target half (z-mixing, interior nodes) joins the trait when the
//! rollout `Engine` is genericized (migration step 3), so it is not designed prematurely here.

use crate::game::{Game, SnakeGame, SnakeState};
use crate::search::{selective_search, SearchParams};

/// A planner that selects actions for a `Game` using the value network. Exploration (epsilon,
/// Thompson head choice) is the rollout engine's concern, not the planner's — `act` is greedy.
pub trait Planner<G: Game> {
    /// Greedily select an action index for `agent` at `state` under ensemble `head`, via `infer`.
    fn act<F>(&self, game: &G, state: &G::State, agent: usize, head: usize, infer: &mut F) -> usize
    where
        F: FnMut(Vec<f32>, usize) -> Vec<f64>;
}

/// Selective expectimax + TreeStrap — today's algorithm. For now it wraps the snake-concrete
/// `selective_search`; migration step 2 makes the search generic over `Game` and this generic over G.
pub struct SelectiveTreeStrap {
    pub params: SearchParams,
}

impl Planner<SnakeGame> for SelectiveTreeStrap {
    fn act<F>(
        &self,
        _game: &SnakeGame,
        state: &SnakeState,
        agent: usize,
        head: usize,
        infer: &mut F,
    ) -> usize
    where
        F: FnMut(Vec<f32>, usize) -> Vec<f64>,
    {
        let (values, _interior, _stats) = selective_search(
            &self.params,
            state.snakes.clone(),
            state.food.clone(),
            agent,
            false,
            infer,
        );
        let h = head.min(values.len().saturating_sub(1));
        argmax(&values[h])
    }
}

/// First-argmax (ties resolve to the lowest index), matching the rollout engine's action choice.
fn argmax(values: &[f64]) -> usize {
    let mut best = 0;
    for (i, &v) in values.iter().enumerate() {
        if v > values[best] {
            best = i;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::Game;
    use crate::reward::Reward;
    use crate::search::Opponent;
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
    fn act_returns_a_legal_action_matching_a_direct_search() {
        let game = SnakeGame {
            grid_size: G,
            initial_length: 3,
            play_to_last: false,
            win_food_lead: None,
            initial_food_count: 0,
            reward: params().reward,
        };
        let planner = SelectiveTreeStrap { params: params() };
        let st = state();
        let a = planner.act(&game, &st, 0, 1, &mut infer);
        assert!(game.legal_actions(&st, 0).contains(&a));
        // Wraps the search faithfully: same head argmax as calling the search directly.
        let (values, _i, _s) = selective_search(
            &params(),
            st.snakes.clone(),
            st.food.clone(),
            0,
            false,
            &mut infer,
        );
        assert_eq!(a, argmax(&values[1]));
    }
}
