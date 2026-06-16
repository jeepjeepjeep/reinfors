//! `SnakeGame` — the two-player simultaneous-move snake implementing `reinfors_core::Game`. It
//! reproduces the concrete `SnakeEnv`/search behavior exactly (the snake↔snake_RL differential suite
//! is the guard). The framework consumes a game through the `Game` trait only.

use std::collections::HashSet;

use reinfors_core::game::{Actor, Game, Rng, Transition};

use crate::action::{relative_to_absolute, Action, RELATIVE_ACTIONS};
use crate::obs::{egocentric_parts, N_CHANNELS};
use crate::reward::Reward;
use crate::snake::{empty_cells, first_empty_cell, Cell, Snake, SnakeEnv};

/// Snake's dynamic state: the two snakes and the food. Static config (grid size, rules, reward) lives
/// on `SnakeGame`, so the search/engine can carry just this around per node.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SnakeState {
    pub snakes: [Snake; 2],
    pub food: HashSet<Cell>,
}

/// Two-player simultaneous-move snake with environment chance (apple respawn) — the concrete `SnakeEnv`
/// dynamics behind the `Game` trait.
pub struct SnakeGame {
    pub grid_size: i32,
    pub initial_length: usize,
    pub play_to_last: bool,
    pub win_food_lead: Option<usize>,
    pub initial_food_count: usize,
    pub reward: Reward,
}

impl SnakeGame {
    fn env(&self, state: &SnakeState) -> SnakeEnv {
        SnakeEnv::from_parts(
            self.grid_size,
            self.initial_length,
            self.play_to_last,
            self.win_food_lead,
            state.snakes.clone(),
            state.food.clone(),
        )
    }

    /// Spawn one apple at a uniform-random empty cell (the env's true spawn), or nothing if the grid
    /// is full — matching the rollout engine's `spawn_food`.
    fn spawn_one(&self, snakes: &[Snake; 2], food: &mut HashSet<Cell>, rng: &mut dyn Rng) {
        let empty = empty_cells(snakes, food, self.grid_size);
        if !empty.is_empty() {
            food.insert(empty[rng.below(empty.len())]);
        }
    }
}

impl Game for SnakeGame {
    type State = SnakeState;

    fn num_agents(&self) -> usize {
        2
    }

    fn action_count(&self) -> usize {
        RELATIVE_ACTIONS.len()
    }

    fn obs_shape(&self) -> (usize, usize, usize) {
        (N_CHANNELS, self.grid_size as usize, self.grid_size as usize)
    }

    fn actor(&self, _state: &SnakeState) -> Actor {
        Actor::Simultaneous
    }

    fn legal_actions(&self, state: &SnakeState, agent: usize) -> Vec<usize> {
        if state.snakes[agent].alive {
            (0..RELATIVE_ACTIONS.len()).collect()
        } else {
            Vec::new()
        }
    }

    fn step(&self, state: &SnakeState, actions: &[usize]) -> Transition<SnakeState> {
        // Relative action index -> absolute heading per (living) snake; `advance` coasts dead ones.
        let mut moves: [Option<Action>; 2] = [None, None];
        for (i, (slot, snake)) in moves.iter_mut().zip(state.snakes.iter()).enumerate() {
            if snake.alive {
                *slot = Some(relative_to_absolute(
                    snake.direction,
                    RELATIVE_ACTIONS[actions[i]],
                ));
            }
        }
        // `|| None` = no in-advance respawn; the respawn is the chance step (`chance_outcomes`), so the
        // deterministic part and the (belief/RNG) spawn stay separable — matching the search.
        let mut env = self.env(state);
        let events = env.advance(moves, || None);
        let rewards = vec![self.reward.eval(&events[0]), self.reward.eval(&events[1])];
        Transition {
            next_state: SnakeState {
                snakes: env.snakes,
                food: env.food,
            },
            rewards,
            terminal: env.done,
        }
    }

    fn chance_outcomes(
        &self,
        state: &SnakeState,
        transition: &Transition<SnakeState>,
    ) -> Vec<(f64, SnakeState)> {
        // An eaten apple is the only stochastic event: `step` removed it without respawning, so the
        // count drop = apples eaten. None eaten -> deterministic (empty). Otherwise the believed
        // outcome is a first-empty respawn per eaten apple (the search's bit-reproducible spawn belief;
        // the env's true spawn is uniform-RNG, injected by the rollout engine). Single believed branch;
        // the `food_samples` Monte-Carlo fan-out of it is a planner concern, not the game's.
        let next = &transition.next_state;
        let eaten = state.food.len().saturating_sub(next.food.len());
        if eaten == 0 {
            return Vec::new();
        }
        let mut food = next.food.clone();
        for _ in 0..eaten {
            match first_empty_cell(&next.snakes, &food, self.grid_size) {
                Some(cell) => {
                    food.insert(cell);
                }
                None => break,
            }
        }
        vec![(
            1.0,
            SnakeState {
                snakes: next.snakes.clone(),
                food,
            },
        )]
    }

    fn observe(&self, state: &SnakeState, agent: usize) -> Vec<f32> {
        egocentric_parts(&state.snakes, &state.food, self.grid_size, agent)
    }

    fn initial_state(&self, rng: &mut dyn Rng) -> SnakeState {
        let env = SnakeEnv::new(
            self.grid_size,
            self.initial_length,
            self.play_to_last,
            self.win_food_lead,
        );
        let mut food = HashSet::new();
        for _ in 0..self.initial_food_count {
            self.spawn_one(&env.snakes, &mut food, rng);
        }
        SnakeState {
            snakes: env.snakes,
            food,
        }
    }

    fn step_env(
        &self,
        state: &SnakeState,
        actions: &[usize],
        rng: &mut dyn Rng,
    ) -> Transition<SnakeState> {
        // Same deterministic move as `step` (no in-advance respawn), then the env's TRUE respawn: one
        // apple at a uniform-random empty cell per eaten apple, in snake order. Reward excludes the
        // survival/truncation bonus (the engine adds that via `truncation_bonus`).
        let mut moves: [Option<Action>; 2] = [None, None];
        for (i, (slot, snake)) in moves.iter_mut().zip(state.snakes.iter()).enumerate() {
            if snake.alive {
                *slot = Some(relative_to_absolute(
                    snake.direction,
                    RELATIVE_ACTIONS[actions[i]],
                ));
            }
        }
        let mut env = self.env(state);
        let events = env.advance(moves, || None);
        for ev in events.iter() {
            if ev.ate_food {
                self.spawn_one(&env.snakes, &mut env.food, rng);
            }
        }
        let rewards = vec![self.reward.eval(&events[0]), self.reward.eval(&events[1])];
        Transition {
            next_state: SnakeState {
                snakes: env.snakes,
                food: env.food,
            },
            rewards,
            terminal: env.done,
        }
    }

    fn truncation_bonus(&self, state: &SnakeState, agent: usize) -> f64 {
        if state.snakes[agent].alive {
            self.reward.survival
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::obs::egocentric_parts;
    use crate::snake::SnakeEnv;

    const G: i32 = 8;

    fn reward() -> Reward {
        Reward {
            step: 0.0,
            food: 1.0,
            loss: -10.0,
            draw: -5.0,
            kill: 5.0,
            win: 10.0,
            survival: 0.0,
        }
    }

    fn game() -> SnakeGame {
        SnakeGame {
            grid_size: G,
            initial_length: 3,
            play_to_last: false,
            win_food_lead: None,
            initial_food_count: 1,
            reward: reward(),
        }
    }

    struct TestRng(u64);
    impl Rng for TestRng {
        fn below(&mut self, n: usize) -> usize {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 33) as usize) % n.max(1)
        }
        fn unit(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    fn initial_state(food: &[Cell]) -> SnakeState {
        let env = SnakeEnv::new(G, 3, false, None);
        SnakeState {
            snakes: env.snakes,
            food: food.iter().copied().collect(),
        }
    }

    /// The reference successor exactly as the search builds it: `advance` with no in-step spawn, then
    /// a first-empty respawn per eaten apple. `step` + `chance_outcomes` must reproduce this.
    fn search_successor(
        g: &SnakeGame,
        state: &SnakeState,
        actions: [usize; 2],
    ) -> (SnakeState, [f64; 2], bool) {
        let mut moves: [Option<Action>; 2] = [None, None];
        for (i, (slot, snake)) in moves.iter_mut().zip(state.snakes.iter()).enumerate() {
            if snake.alive {
                *slot = Some(relative_to_absolute(
                    snake.direction,
                    RELATIVE_ACTIONS[actions[i]],
                ));
            }
        }
        let mut sim = SnakeEnv::from_parts(
            g.grid_size,
            g.initial_length,
            g.play_to_last,
            g.win_food_lead,
            state.snakes.clone(),
            state.food.clone(),
        );
        let events = sim.advance(moves, || None);
        for ev in events.iter() {
            if ev.ate_food {
                if let Some(cell) = first_empty_cell(&sim.snakes, &sim.food, g.grid_size) {
                    sim.food.insert(cell);
                }
            }
        }
        let rewards = [g.reward.eval(&events[0]), g.reward.eval(&events[1])];
        (
            SnakeState {
                snakes: sim.snakes,
                food: sim.food,
            },
            rewards,
            sim.done,
        )
    }

    fn apply(
        g: &SnakeGame,
        state: &SnakeState,
        actions: [usize; 2],
    ) -> (SnakeState, Vec<f64>, bool) {
        let t = g.step(state, &actions);
        let outcomes = g.chance_outcomes(state, &t);
        let after = if outcomes.is_empty() {
            t.next_state.clone() // empty == deterministic
        } else {
            let total: f64 = outcomes.iter().map(|(p, _)| p).sum();
            assert!((total - 1.0).abs() < 1e-12, "chance probs must sum to 1");
            outcomes[0].1.clone()
        };
        (after, t.rewards, t.terminal)
    }

    #[test]
    fn step_then_chance_matches_search_successor_with_an_eat() {
        // Food directly in front of A (faces Right, head (4,2)): Forward eats it, triggering a respawn.
        let g = game();
        let st = initial_state(&[(4, 3)]);
        let (after, rewards, terminal) = apply(&g, &st, [0, 0]);
        let (exp_state, exp_rewards, exp_terminal) = search_successor(&g, &st, [0, 0]);
        assert_eq!(after, exp_state);
        assert_eq!(rewards, exp_rewards.to_vec());
        assert_eq!(terminal, exp_terminal);
        assert!(
            (rewards[0] - 1.0).abs() < 1e-12,
            "A ate one apple -> food reward"
        );
        assert_eq!(after.food.len(), 1, "respawn restored the apple count");
    }

    #[test]
    fn step_then_chance_matches_search_successor_no_eat() {
        let g = game();
        let st = initial_state(&[(0, 0)]); // far corner, untouched
        for actions in [[0usize, 0], [1, 2], [2, 1], [0, 2]] {
            let (after, rewards, terminal) = apply(&g, &st, actions);
            let (exp_state, exp_rewards, exp_terminal) = search_successor(&g, &st, actions);
            assert_eq!(after, exp_state, "actions {actions:?}");
            assert_eq!(rewards, exp_rewards.to_vec());
            assert_eq!(terminal, exp_terminal);
        }
        // No apple eaten -> chance is deterministic (empty outcomes).
        let t = g.step(&st, &[1, 2]);
        assert!(g.chance_outcomes(&st, &t).is_empty());
    }

    #[test]
    fn observe_matches_egocentric() {
        let g = game();
        let st = initial_state(&[(4, 3)]);
        for agent in 0..2 {
            assert_eq!(
                g.observe(&st, agent),
                egocentric_parts(&st.snakes, &st.food, G, agent)
            );
        }
    }

    #[test]
    fn legal_actions_and_metadata() {
        let g = game();
        let st = initial_state(&[(4, 3)]);
        assert_eq!(g.num_agents(), 2);
        assert_eq!(g.action_count(), 3);
        assert_eq!(g.obs_shape(), (N_CHANNELS, G as usize, G as usize));
        assert_eq!(g.actor(&st), Actor::Simultaneous);
        assert_eq!(g.legal_actions(&st, 0), vec![0, 1, 2]);
        // A dead snake has no legal actions (the planner reads activeness from this).
        let mut dead = st.clone();
        dead.snakes[1].alive = false;
        assert!(g.legal_actions(&dead, 1).is_empty());
    }

    #[test]
    fn initial_state_spawns_the_configured_food_count_deterministically() {
        let mut g = game();
        g.initial_food_count = 3;
        let a = g.initial_state(&mut TestRng(7));
        let b = g.initial_state(&mut TestRng(7));
        assert_eq!(a, b, "same seed -> same initial state");
        assert_eq!(a.food.len(), 3);
        // Snakes match the env's initial placement; food sits on empty cells.
        let env = SnakeEnv::new(G, 3, false, None);
        assert_eq!(a.snakes, env.snakes);
        let occupied: std::collections::HashSet<Cell> = a
            .snakes
            .iter()
            .flat_map(|s| s.body.iter().copied())
            .collect();
        assert!(a.food.iter().all(|c| !occupied.contains(c)));
    }

    #[test]
    fn step_env_realizes_the_move_plus_rng_respawn() {
        // Realized transition: same move as `step`, but the eaten apple respawns at an RNG empty cell
        // (not the first-empty belief). Reward carries the food bonus; count is restored.
        let g = game();
        let st = initial_state(&[(4, 3)]);
        let t = g.step_env(&st, &[0, 0], &mut TestRng(1));
        assert!((t.rewards[0] - 1.0).abs() < 1e-12, "A ate -> food reward");
        assert!(!t.terminal);
        assert_eq!(
            t.next_state.food.len(),
            1,
            "respawn restored the apple count"
        );
        // A coast with no food/death scores the bare step reward, never the survival bonus.
        let empty = initial_state(&[]);
        let t2 = g.step_env(&empty, &[0, 0], &mut TestRng(1));
        assert_eq!(t2.rewards, vec![0.0, 0.0]);
    }

    #[test]
    fn truncation_bonus_is_survival_for_the_living_only() {
        let mut g = game();
        g.reward.survival = 0.25;
        let st = initial_state(&[]);
        assert!((g.truncation_bonus(&st, 0) - 0.25).abs() < 1e-12);
        let mut dead = st.clone();
        dead.snakes[0].alive = false;
        assert_eq!(g.truncation_bonus(&dead, 0), 0.0);
    }
}
