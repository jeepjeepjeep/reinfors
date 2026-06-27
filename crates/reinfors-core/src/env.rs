//! A single, caller-driven game instance — the inverse of the autonomous [`Engine`](crate::Engine).
//! `Env` holds one live episode (a `Game`'s rules + the current `State` + an RNG + an encoder) and
//! advances it with externally supplied actions, so the action source is the caller's: a trained net,
//! a baseline, a search, or a human. The `Engine` drives N games with its own policy for training;
//! `Env` drives one game move-by-move for play, evaluation, and debugging.

use crate::encoder::StateEncoder;
use crate::episode::Episode;
use crate::game::Game;
use crate::space::Space;

pub struct Env<G: Game> {
    game: G,
    encoder: Box<dyn StateEncoder<State = G::State>>,
    episode: Episode<G>,
    done: bool,
}

impl<G: Game> Env<G> {
    pub fn new(game: G, encoder: Box<dyn StateEncoder<State = G::State>>, seed: u64) -> Self {
        let episode = Episode::new(&game, seed);
        Env {
            game,
            encoder,
            episode,
            done: false,
        }
    }

    pub fn reset(&mut self) {
        self.episode.reset(&self.game);
        self.done = false;
    }

    pub fn num_agents(&self) -> usize {
        self.game.num_agents()
    }

    pub fn action_count(&self) -> usize {
        self.game.action_count()
    }

    pub fn state(&self) -> &G::State {
        &self.episode.state
    }

    pub fn done(&self) -> bool {
        self.done
    }

    pub fn active_agents(&self) -> Vec<usize> {
        self.episode.active_agents(&self.game)
    }

    pub fn legal_actions(&self, agent: usize) -> Vec<usize> {
        self.game.legal_actions(&self.episode.state, agent)
    }

    /// The encoded observation for `agent` (the value-network view of the current state).
    pub fn observe(&self, agent: usize) -> Vec<f32> {
        self.episode.observe(&*self.encoder, agent)
    }

    pub fn observation_space(&self) -> Space {
        self.encoder.observation_space()
    }

    /// Apply a joint action (one index per agent; entries for inactive agents are ignored), advancing
    /// the episode through the env transition. Returns this tick's per-agent events — what happened to
    /// each agent (`Env` holds no reward; a game-aware caller reads the outcome from these or `state()`).
    pub fn step(&mut self, actions: &[usize]) -> Vec<G::Event> {
        debug_assert!(!self.done, "step() after done — call reset() first");
        debug_assert_eq!(
            actions.len(),
            self.game.num_agents(),
            "step() expects one action per agent"
        );
        let (events, terminal) = self.episode.advance(&self.game, actions);
        self.done = terminal;
        events
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoder::StateEncoder;
    use crate::game::{Actor, Game, Rng, Transition};

    // A 1-agent walk to `goal`: action 1 steps right, 0 stays; the event is 1.0 at the goal, else 0.0.
    struct Walk {
        goal: i32,
    }
    impl Game for Walk {
        type State = i32;
        type Event = f64;
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
        fn step(&self, pos: &i32, actions: &[usize]) -> Transition<i32, f64> {
            let next = pos + actions[0] as i32;
            let terminal = next >= self.goal;
            Transition {
                next_state: next,
                events: vec![if terminal { 1.0 } else { 0.0 }],
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
    fn steps_to_terminal_and_reports_done_and_event() {
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
