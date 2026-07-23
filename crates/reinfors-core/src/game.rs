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

/// One transition's deterministic outcome: the resulting state, the per-agent `Event` (what happened
/// to each agent — a [`Reward`](crate::Reward) maps these to scalars), and whether the game ended.
pub struct Transition<S, E> {
    pub next_state: S,
    pub events: Vec<E>,
    pub terminal: bool,
}

/// A finite-action, perfect-information game. Single-agent, sequential or simultaneous multi-agent,
/// and N-player general-sum are all expressible via `actor` + the per-agent `Event`. The game owns
/// only dynamics + outcomes; turning an `Event` into a scalar reward is the [`Reward`](crate::Reward)'s
/// job, decoupled like the encoder.
pub trait Game {
    type State: Clone;

    /// The per-agent outcome of a tick (e.g. snake's `StepEvent`), consumed by a `Reward`.
    type Event;

    fn num_agents(&self) -> usize;

    fn action_count(&self) -> usize;

    fn action_space(&self) -> Space {
        Space::Discrete {
            n: self.action_count(),
        }
    }

    fn actor(&self, state: &Self::State) -> Actor;

    fn legal_actions(&self, state: &Self::State, agent: usize) -> Vec<usize>;

    fn step(&self, state: &Self::State, actions: &[usize]) -> Transition<Self::State, Self::Event>;

    /// Draw one realization of the transition's environment chance. `None` means the transition is
    /// deterministic (no chance node), so callers use `transition.next_state` directly. Determinism is
    /// a property of the transition, not the draw, so a stochastic transition always returns `Some`.
    /// Callers wanting several independent realizations (e.g. the search's `food_samples` Monte-Carlo
    /// fan-out) call this `n` times. The default is deterministic.
    fn sample_chance(
        &self,
        state: &Self::State,
        transition: &Transition<Self::State, Self::Event>,
        rng: &mut dyn Rng,
    ) -> Option<Self::State> {
        let _ = (state, transition, rng);
        None
    }

    /// The transition's chance distribution, *declared*: probabilities over the outcome indices
    /// that [`apply_chance`](Self::apply_chance) accepts. `None` means the transition is
    /// deterministic — the same condition under which `sample_chance` returns `None`, and the two
    /// must agree (a correct sampler is constructive proof the game knows this distribution; this
    /// is its declarative form, and tree searches consume it per their configured
    /// [`ChanceMode`](crate::ChanceMode)). Contract: probabilities are positive and sum to 1;
    /// terminal transitions return `None`; outcomes only vary the chance element — they share the
    /// transition's `terminal` flag and next actor. The default declares every transition
    /// deterministic.
    fn chance_outcomes(
        &self,
        state: &Self::State,
        transition: &Transition<Self::State, Self::Event>,
    ) -> Option<Vec<f64>> {
        let _ = (state, transition);
        None
    }

    /// Materialize one outcome of the transition's chance distribution — the state `sample_chance`
    /// would produce had it drawn `outcome` (an index into the `chance_outcomes` probabilities).
    /// Only called with indices of a `Some` distribution; games that declare no chance never see
    /// it.
    fn apply_chance(
        &self,
        state: &Self::State,
        transition: &Transition<Self::State, Self::Event>,
        outcome: usize,
    ) -> Self::State {
        let _ = (state, transition, outcome);
        unreachable!("apply_chance called on a game that declares no chance_outcomes")
    }

    fn initial_state(&self, rng: &mut dyn Rng) -> Self::State;

    fn step_env(
        &self,
        state: &Self::State,
        actions: &[usize],
        rng: &mut dyn Rng,
    ) -> Transition<Self::State, Self::Event> {
        let t = self.step(state, actions);
        let next_state = self.sample_chance(state, &t, rng).unwrap_or(t.next_state);
        Transition {
            next_state,
            events: t.events,
            terminal: t.terminal,
        }
    }

    /// The episode-length cap after which the rollout truncates a still-running game, or `None` for a
    /// game that always ends on its own (e.g. Connect-4). This is a property the game *declares* — the
    /// `Engine` does the tick-counting and enforces it, so the horizon never enters `State` or the
    /// search. Truncation is thus wholly a game concern (when *and*, via `mark_truncation`, what).
    fn truncation_horizon(&self) -> Option<usize> {
        None
    }

    /// Stamp the truncation outcome onto `events` when the rollout cuts the episode off at the horizon
    /// (the `Engine` calls this on that tick, before the reward evaluates the events). A game encodes
    /// "survived to the cutoff" here — e.g. snake flags its still-alive agents so their `Reward` pays
    /// the survival bonus. Default: no truncation-specific outcome.
    fn mark_truncation(&self, state: &Self::State, events: &mut [Self::Event]) {
        let _ = (state, events);
    }
}
