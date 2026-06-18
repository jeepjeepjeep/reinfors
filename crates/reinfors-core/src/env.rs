//! A single, caller-driven game instance — the inverse of the autonomous [`Engine`](crate::Engine).
//! `Env` holds one live episode (a `Game`'s rules + the current `State` + an RNG + an encoder) and
//! advances it with externally supplied actions, so the action source is the caller's: a trained net,
//! a baseline, a search, or a human. The `Engine` drives N games with its own policy for training;
//! `Env` drives one game move-by-move for play, evaluation, and debugging.

use crate::encoder::StateEncoder;
use crate::game::Game;
use crate::rng::SplitMix64;
use crate::space::Space;

pub struct Env<G: Game> {
    game: G,
    encoder: Box<dyn StateEncoder<State = G::State>>,
    rng: SplitMix64,
    state: G::State,
    done: bool,
}

impl<G: Game> Env<G> {
    /// Start a fresh episode of `game` (initial chance drawn from `seed`), observed through `encoder`.
    pub fn new(game: G, encoder: Box<dyn StateEncoder<State = G::State>>, seed: u64) -> Self {
        let mut rng = SplitMix64::new(seed);
        let state = game.initial_state(&mut rng);
        Env {
            game,
            encoder,
            rng,
            state,
            done: false,
        }
    }

    /// Begin a new episode, drawing fresh initial chance from the (continuing) RNG stream.
    pub fn reset(&mut self) {
        self.state = self.game.initial_state(&mut self.rng);
        self.done = false;
    }

    pub fn num_agents(&self) -> usize {
        self.game.num_agents()
    }

    pub fn action_count(&self) -> usize {
        self.game.action_count()
    }

    /// The native game state, for rendering/inspection (the encoder is for the net's view).
    pub fn state(&self) -> &G::State {
        &self.state
    }

    pub fn done(&self) -> bool {
        self.done
    }

    /// Agents that must supply an action this tick: a single mover for a sequential game, all live
    /// agents for a simultaneous one. Empty once the episode is over.
    pub fn active_agents(&self) -> Vec<usize> {
        (0..self.game.num_agents())
            .filter(|&a| !self.game.legal_actions(&self.state, a).is_empty())
            .collect()
    }

    pub fn legal_actions(&self, agent: usize) -> Vec<usize> {
        self.game.legal_actions(&self.state, agent)
    }

    /// The encoded observation for `agent` (the value-network view of the current state).
    pub fn observe(&self, agent: usize) -> Vec<f32> {
        self.encoder.encode(&self.state, agent)
    }

    /// The observation `Space` (from the encoder) — so a caller can size/validate a network from the
    /// `Env` alone.
    pub fn observation_space(&self) -> Space {
        self.encoder.observation_space()
    }

    /// Apply a joint action (one index per agent; entries for inactive agents are ignored), advancing
    /// the episode through the env transition. Returns this tick's per-agent reward vector. Stepping a
    /// finished episode is a misuse — `active_agents()` is empty once `done()`; `reset()` first.
    pub fn step(&mut self, actions: &[usize]) -> Vec<f64> {
        debug_assert!(!self.done, "step() after done — call reset() first");
        debug_assert_eq!(
            actions.len(),
            self.game.num_agents(),
            "step() expects one action per agent"
        );
        let t = self.game.step_env(&self.state, actions, &mut self.rng);
        self.state = t.next_state;
        self.done = t.terminal;
        t.rewards
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoder::StateEncoder;
    use crate::game::{Actor, Game, Rng, Transition};

    // A 1-agent walk to `goal`: action 1 steps right, 0 stays; terminal (reward 1) at the goal.
    struct Walk {
        goal: i32,
    }
    impl Game for Walk {
        type State = i32;
        fn num_agents(&self) -> usize {
            1
        }
        fn action_count(&self) -> usize {
            2
        }
        fn actor(&self, _: &i32) -> Actor {
            Actor::Agent(0)
        }
        fn legal_actions(&self, pos: &i32, agent: usize) -> Vec<usize> {
            if agent == 0 && *pos < self.goal {
                vec![0, 1]
            } else {
                Vec::new()
            }
        }
        fn step(&self, pos: &i32, actions: &[usize]) -> Transition<i32> {
            let next = pos + actions[0] as i32;
            let terminal = next >= self.goal;
            Transition {
                next_state: next,
                rewards: vec![if terminal { 1.0 } else { 0.0 }],
                terminal,
            }
        }
        fn initial_state(&self, _: &mut dyn Rng) -> i32 {
            0
        }
    }

    struct PosEncoder;
    impl StateEncoder for PosEncoder {
        type State = i32;
        fn encode(&self, pos: &i32, _: usize) -> Vec<f32> {
            vec![*pos as f32]
        }
        fn obs_shape(&self) -> (usize, usize, usize) {
            (1, 1, 1)
        }
    }

    fn env() -> Env<Walk> {
        Env::new(Walk { goal: 3 }, Box::new(PosEncoder), 0)
    }

    #[test]
    fn steps_to_terminal_and_reports_done_and_reward() {
        let mut e = env();
        assert!(!e.done() && e.active_agents() == vec![0]);
        assert_eq!(e.observe(0), vec![0.0]);
        assert_eq!(e.step(&[1]), vec![0.0]); // 0 -> 1
        assert_eq!(e.step(&[1]), vec![0.0]); // 1 -> 2
        assert_eq!(e.step(&[1]), vec![1.0]); // 2 -> 3: goal
        assert!(e.done());
        assert!(e.active_agents().is_empty()); // no legal actions once finished
        assert_eq!(*e.state(), 3);
    }

    #[test]
    fn reset_starts_a_fresh_episode() {
        let mut e = env();
        e.step(&[1]);
        e.step(&[1]);
        e.reset();
        assert!(!e.done() && *e.state() == 0 && e.observe(0) == vec![0.0]);
    }

    #[test]
    fn surfaces_the_encoder_observation_space() {
        assert_eq!(
            env().observation_space(),
            Space::Box {
                shape: vec![1, 1, 1],
                low: f32::NEG_INFINITY,
                high: f32::INFINITY,
            }
        );
    }

    #[test]
    #[should_panic(expected = "step() after done")]
    fn stepping_after_done_is_a_misuse() {
        let mut e = env();
        for _ in 0..3 {
            e.step(&[1]); // reaches the goal -> done
        }
        e.step(&[1]); // misuse: should trip the debug assert
    }
}
