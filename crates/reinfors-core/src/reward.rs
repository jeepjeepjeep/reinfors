//! The reward seam: a `Reward` turns a game's per-agent `Event`s — the outcome of a tick that
//! [`Game::step`](crate::Game) produces — into scalar rewards. It is decoupled from `Game` (which owns
//! only dynamics + outcomes), mirroring how [`StateEncoder`](crate::StateEncoder) decouples the
//! observation: the reward is the agent's / training objective, not a rule of the game, so it is a
//! separate handle threaded into the `Engine` and the search.
//!
//! Object-safe (no generic methods, no `Self` by value), so it is held as
//! `Box<dyn Reward<Event = G::Event, State = G::State>>` — the reward equivalent of the boxed encoder.
//! Only the training path needs it: the `Engine` (to fill training-record rewards + the episode-end
//! z-mix) and the search (immediate rewards in the backup). The caller-driven `Env` holds none — it
//! surfaces game-specific events, and a game-aware consumer reads the outcome from those.

pub trait Reward: Send + Sync {
    type Event;
    type State;

    /// The scalar reward `agent` earns from its per-agent `event` this tick.
    fn step_reward(&self, event: &Self::Event, agent: usize) -> f64;

    /// A bonus paid to `agent` on a truncation tick it reached alive (e.g. a survival reward). Read
    /// from a state snapshot, since truncation is the rollout hitting `max_ticks`, not a transition.
    /// Defaults to none.
    fn truncation_bonus(&self, state: &Self::State, agent: usize) -> f64 {
        let _ = (state, agent);
        0.0
    }
}
