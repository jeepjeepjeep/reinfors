//! `Snake` — the two-player simultaneous-move snake implementing `reinfors_core::Game`. It
//! reproduces the concrete `SnakeEnv`/search behavior exactly (the snake↔snake_RL differential suite
//! is the guard). The framework consumes a game through the `Game` trait only.

use std::collections::HashSet;

use reinfors_core::game::{Actor, Game, Rng, Transition};
use reinfors_core::StateEncoder;

use crate::action::{relative_to_absolute, Action, RELATIVE_ACTIONS};
use crate::obs::{egocentric_parts, N_CHANNELS};
use crate::reward::SnakeReward;
use crate::snake::{Cell, SnakeBody, SnakeEnv};

/// Snake's dynamic state: the two snakes and the food. Static config (grid size, rules, reward) lives
/// on `Snake`, so the search/engine can carry just this around per node.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SnakeState {
    pub snakes: [SnakeBody; 2],
    pub food: HashSet<Cell>,
}

/// The default snake observation: an egocentric 5-channel grid, the searching snake always facing up
/// (see [`egocentric_parts`]). Carries `grid_size` (which lives on `Snake`, not in `SnakeState`).
pub struct EgocentricSnake {
    pub grid_size: i32,
}

impl StateEncoder for EgocentricSnake {
    type State = SnakeState;

    fn encode(&self, state: &SnakeState, agent: usize) -> Vec<f32> {
        egocentric_parts(&state.snakes, &state.food, self.grid_size, agent)
    }

    fn obs_shape(&self) -> (usize, usize, usize) {
        (N_CHANNELS, self.grid_size as usize, self.grid_size as usize)
    }
}

/// Two-player simultaneous-move snake with environment chance (apple respawn) — the concrete `SnakeEnv`
/// dynamics behind the `Game` trait.
pub struct Snake {
    pub grid_size: i32,
    pub initial_length: usize,
    pub play_to_last: bool,
    pub win_food_lead: Option<usize>,
    pub initial_food_count: usize,
    pub reward: SnakeReward,
}

impl Snake {
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

    /// Spawn one apple at a uniform-random empty cell (the env's true spawn), or nothing if the grid is
    /// full. Build the occupancy set once (food + both bodies, deduped), so the empty count is
    /// `g² − occupied.len()` (no count pass) and lookups are O(1); then walk to the k-th empty cell in
    /// row-major order. A single `rng.below(n)` indexing the row-major empties is identical to
    /// materializing the empties `Vec` and indexing it — same cell, same RNG — but without that `Vec`.
    fn spawn_one(&self, snakes: &[SnakeBody; 2], food: &mut HashSet<Cell>, rng: &mut dyn Rng) {
        let g = self.grid_size;
        let mut occupied: HashSet<Cell> = food.clone();
        for s in snakes {
            occupied.extend(s.body.iter().copied());
        }
        let n = (g * g) as usize - occupied.len();
        if n == 0 {
            return;
        }
        let mut k = rng.below(n);
        for r in 0..g {
            for c in 0..g {
                let cell = (r, c);
                if occupied.contains(&cell) {
                    continue;
                }
                if k == 0 {
                    food.insert(cell);
                    return;
                }
                k -= 1;
            }
        }
    }
}

impl Game for Snake {
    type State = SnakeState;

    fn num_agents(&self) -> usize {
        2
    }

    fn action_count(&self) -> usize {
        RELATIVE_ACTIONS.len()
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
        // `|| None` = no in-advance respawn; the respawn is the chance step (`sample_chance`), so the
        // deterministic part and the sampled spawn stay separable and are shared by the env and search.
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

    fn sample_chance(
        &self,
        state: &SnakeState,
        transition: &Transition<SnakeState>,
        rng: &mut dyn Rng,
        n: usize,
    ) -> Vec<SnakeState> {
        // An eaten apple is the only stochastic event: `step` removed it without respawning, so the
        // count drop = apples eaten. None eaten -> deterministic (empty). Otherwise draw `n` independent
        // realizations, each respawning one uniform-random apple per eaten apple via `spawn_one` — the
        // same spawn the env rollout uses, so search and env share one chance model.
        let next = &transition.next_state;
        let eaten = state.food.len().saturating_sub(next.food.len());
        if eaten == 0 {
            return Vec::new();
        }
        (0..n)
            .map(|_| {
                let mut food = next.food.clone();
                for _ in 0..eaten {
                    self.spawn_one(&next.snakes, &mut food, rng);
                }
                SnakeState {
                    snakes: next.snakes.clone(),
                    food,
                }
            })
            .collect()
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
    use crate::snake::{empty_cells, SnakeEnv};

    const G: i32 = 8;

    fn reward() -> SnakeReward {
        SnakeReward {
            step: 0.0,
            food: 1.0,
            loss: -10.0,
            draw: -5.0,
            kill: 5.0,
            win: 10.0,
            survival: 0.0,
        }
    }

    fn game() -> Snake {
        Snake {
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

    #[test]
    fn step_env_equals_step_then_sample_chance() {
        // The unification invariant: the realized env step and the search's chance sampler are the
        // SAME draw. `step_env` must equal `step` then `sample_chance(.., 1)` under the same RNG seed,
        // so the rollout and the search can never use different chance dynamics.
        // Food directly in front of A (faces Right, head (4,2)): Forward eats it, triggering a respawn.
        let g = game();
        let st = initial_state(&[(4, 3)]);
        let actions = [0usize, 0];
        let realized = g.step_env(&st, &actions, &mut TestRng(42));
        let t = g.step(&st, &actions);
        let mut sampled = g.sample_chance(&st, &t, &mut TestRng(42), 1);
        assert_eq!(sampled.len(), 1, "an eaten apple is a chance node");
        assert_eq!(realized.next_state, sampled.swap_remove(0));
        assert_eq!(realized.rewards, t.rewards);
        assert_eq!(realized.terminal, t.terminal);
        assert!((realized.rewards[0] - 1.0).abs() < 1e-12, "A ate one apple");
        assert_eq!(
            realized.next_state.food.len(),
            1,
            "respawn restored the count"
        );
    }

    #[test]
    fn no_eat_is_deterministic() {
        let g = game();
        let st = initial_state(&[(0, 0)]); // far corner, untouched
        for actions in [[0usize, 0], [1, 2], [2, 1], [0, 2]] {
            let t = g.step(&st, &actions);
            // Nothing eaten -> no chance node, regardless of how many samples are requested.
            assert!(
                g.sample_chance(&st, &t, &mut TestRng(1), 4).is_empty(),
                "actions {actions:?}"
            );
            // ...so the realized env step is exactly the deterministic step.
            let realized = g.step_env(&st, &actions, &mut TestRng(1));
            assert_eq!(realized.next_state, t.next_state, "actions {actions:?}");
            assert_eq!(realized.rewards, t.rewards);
            assert_eq!(realized.terminal, t.terminal);
        }
    }

    #[test]
    fn sample_chance_draws_independent_valid_respawns() {
        // food_samples > 1 fans the chance node into that many independent draws, each a uniform-random
        // apple on a previously empty cell — not a single deterministic belief.
        let g = game();
        let st = initial_state(&[(4, 3)]);
        let t = g.step(&st, &[0, 0]);
        let samples = g.sample_chance(&st, &t, &mut TestRng(7), 20);
        assert_eq!(samples.len(), 20);
        let occupied: std::collections::HashSet<Cell> = t
            .next_state
            .snakes
            .iter()
            .flat_map(|s| s.body.iter().copied())
            .collect();
        for s in &samples {
            assert_eq!(s.food.len(), 1, "respawn restored the apple count");
            let cell = *s.food.iter().next().unwrap();
            assert!(!occupied.contains(&cell), "apple spawns on an empty cell");
        }
        let distinct: std::collections::HashSet<Cell> = samples
            .iter()
            .map(|s| *s.food.iter().next().unwrap())
            .collect();
        assert!(
            distinct.len() > 1,
            "uniform sampling should vary across draws"
        );
    }

    #[test]
    fn sample_chance_is_uniform_over_empty_cells() {
        // The in-tree respawn must be uniform over the empty cells — the same draw the env makes, not a
        // bias toward any cell (e.g. the old first-empty belief). Over many single-apple respawns,
        // assert full coverage of the empty cells and a balanced hit frequency.
        let g = game();
        let st = initial_state(&[(4, 3)]);
        let t = g.step(&st, &[0, 0]); // A eats the only apple -> a respawn chance node
        let n = 20_000;
        let samples = g.sample_chance(&st, &t, &mut TestRng(12345), n);
        let empties = empty_cells(&t.next_state.snakes, &t.next_state.food, G);
        let mut counts: std::collections::HashMap<Cell, usize> = std::collections::HashMap::new();
        for s in &samples {
            let new: Vec<Cell> = s.food.difference(&t.next_state.food).copied().collect();
            assert_eq!(new.len(), 1, "exactly one apple respawns");
            *counts.entry(new[0]).or_default() += 1;
        }
        assert_eq!(
            counts.len(),
            empties.len(),
            "every empty cell must be reachable (full coverage)"
        );
        let min = *counts.values().min().unwrap();
        let max = *counts.values().max().unwrap();
        assert!(
            max <= 2 * min,
            "hit frequency should be balanced for a uniform draw: min={min} max={max}"
        );
    }

    #[test]
    fn encoder_matches_egocentric() {
        let enc = EgocentricSnake { grid_size: G };
        let st = initial_state(&[(4, 3)]);
        for agent in 0..2 {
            assert_eq!(
                enc.encode(&st, agent),
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
        assert_eq!(
            EgocentricSnake { grid_size: G }.obs_shape(),
            (N_CHANNELS, G as usize, G as usize)
        );
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
