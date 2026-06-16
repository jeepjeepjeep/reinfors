//! The `Game` trait — the rules an environment exposes so the framework (search, rollout, training)
//! can drive it without knowing the game. Snake is the first implementation; `SnakeGame` reproduces
//! the concrete `SnakeEnv`/search behavior exactly (the snake↔snake_RL differential suite is the
//! guard). The framework consumes a game through this trait only; nothing here is snake-specific
//! except the `SnakeGame` impl, which will move to a `games` crate once the core is generic.

use std::collections::HashSet;

use crate::action::{relative_to_absolute, Action, RELATIVE_ACTIONS};
use crate::obs::{egocentric_parts, N_CHANNELS};
use crate::reward::Reward;
use crate::snake::{first_empty_cell, Cell, Snake, SnakeEnv};

/// Who chooses at a node: one agent (a sequential turn), all agents at once (a simultaneous move), or
/// nature (a chance node). The game only declares the shape; the planner decides how to expand each.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Actor {
    Agent(usize),
    Simultaneous,
    Chance,
}

/// One transition's deterministic outcome: the resulting state, the per-agent reward vector, and
/// whether the game ended. Per-agent activeness is read from `legal_actions` being empty, not here.
pub struct Transition<S> {
    pub next_state: S,
    pub rewards: Vec<f64>,
    pub terminal: bool,
}

/// A finite-action, perfect-information game. Single-agent, sequential or simultaneous multi-agent,
/// and N-player general-sum are all expressible via `actor` + the per-agent reward vector.
pub trait Game {
    type State: Clone;

    fn num_agents(&self) -> usize;
    /// Net head width: the (homogeneous) per-agent action-space size.
    fn action_count(&self) -> usize;
    /// Observation tensor shape `(C, H, W)` the value network consumes.
    fn obs_shape(&self) -> (usize, usize, usize);

    /// Who acts at `state`.
    fn actor(&self, state: &Self::State) -> Actor;
    /// Action indices (into `0..action_count`) legal for `agent`; empty when the agent is out of play.
    fn legal_actions(&self, state: &Self::State, agent: usize) -> Vec<usize>;
    /// Apply a joint action (one index per agent) — the deterministic part of the transition, before
    /// any chance resolution. The entry for an agent with no legal moves is ignored.
    fn step(&self, state: &Self::State, actions: &[usize]) -> Transition<Self::State>;
    /// The believed chance outcomes of the transition from `state` to `transition.next_state` —
    /// environment stochasticity the planner expands as a chance node: `(probability, state)` summing
    /// to 1. An **empty** result means the transition was deterministic (no chance node). The default
    /// is deterministic. Takes the source `state` so a game can derive what happened (e.g. how many
    /// apples were eaten) by comparing it to `transition.next_state`.
    fn chance_outcomes(
        &self,
        state: &Self::State,
        transition: &Transition<Self::State>,
    ) -> Vec<(f64, Self::State)> {
        let _ = (state, transition);
        Vec::new()
    }
    /// Egocentric observation for `agent`, a flat `[C*H*W]` f32 buffer.
    fn observe(&self, state: &Self::State, agent: usize) -> Vec<f32>;
}

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
            reward: reward(),
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
}
