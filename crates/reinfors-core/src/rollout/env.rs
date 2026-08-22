//! Caller-driven game environment.

use crate::encoder::StateEncoder;
use crate::game::Game;
use crate::rng::SplitMix64;
use crate::rollout::episode::Episode;
use crate::space::Space;

pub struct Env<G: Game> {
    game: G,
    encoder: Box<dyn StateEncoder<State = G::State>>,
    episode: Episode<G>,
    done: bool,
    ticks: usize,
}

impl<G: Game> Env<G> {
    pub fn new(game: G, encoder: Box<dyn StateEncoder<State = G::State>>, seed: u64) -> Self {
        let episode = Episode::new(&game, crate::rng::SplitMix64::new(seed));
        Env {
            game,
            encoder,
            episode,
            done: false,
            ticks: 0,
        }
    }

    pub fn reset(&mut self) {
        self.episode.reset(&self.game);
        self.done = false;
        self.ticks = 0;
    }

    /// Completed steps this episode (the current decision's ply in sequential games).
    pub fn ticks(&self) -> usize {
        self.ticks
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

    pub fn encoder(&self) -> &dyn StateEncoder<State = G::State> {
        &*self.encoder
    }

    pub fn game(&self) -> &G {
        &self.game
    }

    pub fn done(&self) -> bool {
        self.done
    }

    /// Clone the mutable environment state for snapshots or forks.
    pub fn parts(&self) -> (G::State, u64, bool, usize) {
        (
            self.episode.state.clone(),
            self.episode.rng.state(),
            self.done,
            self.ticks,
        )
    }

    /// Restore mutable state at a step boundary.
    pub fn set_parts(&mut self, state: G::State, rng_state: u64, done: bool, ticks: usize) {
        self.episode.state = state;
        self.episode.rng = SplitMix64::from_state(rng_state);
        self.done = done;
        self.ticks = ticks;
    }

    /// Active agents, or an empty list after the episode ends.
    pub fn active_agents(&self) -> Vec<usize> {
        if self.done {
            return Vec::new();
        }
        self.episode.active_agents(&self.game)
    }

    pub fn legal_actions(&self, agent: usize) -> Vec<usize> {
        if self.done {
            return Vec::new();
        }
        self.game.legal_actions(&self.episode.state, agent)
    }

    pub fn observe(&self, agent: usize) -> Vec<f32> {
        self.episode.observe(&*self.encoder, agent)
    }

    pub fn observation_space(&self) -> Space {
        self.encoder.observation_space()
    }

    /// Apply a joint action and return the tick's ordered event trace.
    pub fn step(&mut self, actions: &[usize]) -> Vec<(usize, G::Event)> {
        debug_assert!(!self.done, "step() after done — call reset() first");
        debug_assert_eq!(
            actions.len(),
            self.game.num_agents(),
            "step() expects one action per agent"
        );
        let (trace, terminal) = self.episode.advance(&self.game, actions);
        self.done = terminal;
        self.ticks += 1;
        trace
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoder::{ActionView, StateEncoder};
    use crate::game::{Actor, Game, Transition};

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
                events: vec![if terminal { Some(1.0) } else { None }],
                terminal,
            }
        }
        fn initial_state(&self) -> i32 {
            0
        }
    }

    struct PosEncoder;
    impl ActionView for PosEncoder {}
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
        assert_eq!(e.step(&[1]), vec![]);
        assert_eq!(e.step(&[1]), vec![]);
        assert_eq!(e.step(&[1]), vec![(0, 1.0)]);
        assert!(e.done());
        assert!(e.active_agents().is_empty());
        assert_eq!(*e.state(), 3);
    }

    #[test]
    fn reset_starts_a_fresh_episode() {
        let mut e = env();
        e.step(&[1]);
        e.step(&[1]);
        assert_eq!(e.ticks(), 2);
        e.reset();
        assert!(!e.done() && *e.state() == 0 && e.observe(0) == vec![0.0]);
        assert_eq!(e.ticks(), 0);
    }

    #[test]
    fn parts_round_trip_carries_ticks() {
        let mut e = env();
        e.step(&[0]);
        e.step(&[1]);
        let (state, rng, done, ticks) = e.parts();
        assert_eq!(ticks, 2);
        let mut restored = env();
        restored.set_parts(state, rng, done, ticks);
        assert_eq!(restored.ticks(), 2);
        assert_eq!(*restored.state(), 1);
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
            e.step(&[1]);
        }
        e.step(&[1]);
    }
}
