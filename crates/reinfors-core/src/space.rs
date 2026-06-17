//! Observation/action space descriptors a `Game` advertises, so the framework and bindings can size
//! value networks without hard-coding any game's dimensions. Mirrors Gymnasium's `Box`/`Discrete`.
//!
//! `shape` is the contract — it is what sizes networks and what the framework relies on. Bounds are
//! a single scalar `low`/`high` for the whole tensor: uniform and advisory, not per-element. That is
//! exact for every game today (all observations are one-hot/binary planes, range `[0, 1]`) and keeps
//! the type simple. A per-element form (Gymnasium-style array bounds, needed only to faithfully
//! describe a future game whose channels span different ranges) is intentionally deferred until a
//! concrete consumer — the ecosystem adapter — exists; that is the moment to widen the bounds.

/// A typed space: a continuous `Box` (an N-dimensional `f32` tensor with uniform scalar bounds) or a
/// `Discrete` set of `n` choices. Games derive these from their observation tensor shape and action
/// count (see `Game::observation_space` / `Game::action_space`).
#[derive(Clone, Debug, PartialEq)]
pub enum Space {
    /// An N-dimensional `f32` tensor of `shape`. `low`/`high` are a single range shared by every
    /// element (advisory; `shape` is the contract — see the module docs); use ±∞ for unbounded.
    Box {
        shape: Vec<usize>,
        low: f32,
        high: f32,
    },
    /// A choice from `0..n`.
    Discrete { n: usize },
}
