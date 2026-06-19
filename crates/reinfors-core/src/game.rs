//! The `Game` trait — the rules an environment exposes so the framework (search, rollout, training)
//! can drive it without knowing the game. The framework consumes a game through this trait only;
//! nothing here is game-specific. Concrete games (e.g. snake) live in the `reinfors-games` crate.

use crate::space::Space;

/// Minimal random source the rollout passes to a game's *realized* (non-belief) transitions.
pub trait Rng {
    fn below(&mut self, n: usize) -> usize;
    fn unit(&mut self) -> f64;
}

/// Who chooses at a node: one agent (a sequential turn), all agents at once (a simultaneous move), or
/// nature (a chance node).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Actor {
    Agent(usize),
    Simultaneous,
    Chance,
}

/// One transition's deterministic outcome: the resulting state, the per-agent reward vector, and
/// whether the game ended.
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

    fn action_count(&self) -> usize;

    fn action_space(&self) -> Space {
        Space::Discrete {
            n: self.action_count(),
        }
    }

    fn actor(&self, state: &Self::State) -> Actor;

    fn legal_actions(&self, state: &Self::State, agent: usize) -> Vec<usize>;

    fn step(&self, state: &Self::State, actions: &[usize]) -> Transition<Self::State>;

    /// Sample `n` independent realizations of the transition's environment chance. An **empty**
    /// result means the transition is deterministic (no chance node); callers then use
    /// `transition.next_state` directly. The default is deterministic.
    fn sample_chance(
        &self,
        state: &Self::State,
        transition: &Transition<Self::State>,
        rng: &mut dyn Rng,
        n: usize,
    ) -> Vec<Self::State> {
        let _ = (state, transition, rng, n);
        Vec::new()
    }

    fn initial_state(&self, rng: &mut dyn Rng) -> Self::State;

    fn step_env(
        &self,
        state: &Self::State,
        actions: &[usize],
        rng: &mut dyn Rng,
    ) -> Transition<Self::State> {
        let t = self.step(state, actions);
        let mut outcomes = self.sample_chance(state, &t, rng, 1);
        let next_state = if outcomes.is_empty() {
            t.next_state
        } else {
            outcomes.swap_remove(0)
        };
        Transition {
            next_state,
            rewards: t.rewards,
            terminal: t.terminal,
        }
    }

    fn truncation_bonus(&self, state: &Self::State, agent: usize) -> f64 {
        let _ = (state, agent);
        0.0
    }
}
