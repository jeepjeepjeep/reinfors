//! The reward seam: a `Reward` turns a game's `Event`s — what each EDGE causally decided for an
//! agent (a tick's outcome is the ordered trace of its edges' emissions) — into scalar rewards. It is decoupled from `Game` (which owns
//! only dynamics + outcomes), mirroring how [`StateEncoder`](crate::StateEncoder) decouples the
//! observation: the reward is the agent's / training objective, not a rule of the game, so it is a
//! separate handle threaded into the `Engine` and the search.
//!
//! `Reward` is purely `Event -> scalar` — the reward-relevant mirror of the encoder's `State -> obs`,
//! so the two halves of a transition never cross over (`State` is observational, `Event` is
//! reward-relevant). Even truncation flows through here: the rollout lets the game amend the
//! tick's trace (via [`Game::mark_truncation`](crate::Game::mark_truncation)), so this is the one
//! path that maps outcomes to scalars.
//!
//! Object-safe (no generic methods, no `Self` by value), so it is held as
//! `Box<dyn Reward<Event = G::Event>>` — the reward equivalent of the boxed encoder. Only the training
//! path needs it: the `Engine` (training-record rewards + the episode-end z-mix) and the search
//! (immediate rewards in the backup). The caller-driven `Env` holds none — it surfaces game-specific
//! events, and a game-aware consumer reads the outcome from those.

pub trait Reward: Send + Sync {
    type Event;

    /// The scalar reward `agent` earns from one emitted `event`. Events are per-EDGE and
    /// incremental (see [`Transition`](crate::Transition)); a tick's reward is the sum over its
    /// emitted events.
    fn step_reward(&self, event: &Self::Event, agent: usize) -> f64;
}

/// One edge's reward for `agent`: 0.0 where the edge emitted nothing for it. The uniform read
/// used by every consumer that scores a single [`Transition`](crate::Transition) edge (searches,
/// solvers) — rollout consumers fold the whole tick trace instead.
pub fn edge_reward<E>(reward: &dyn Reward<Event = E>, events: &[Option<E>], agent: usize) -> f64 {
    events[agent]
        .as_ref()
        .map_or(0.0, |e| reward.step_reward(e, agent))
}
