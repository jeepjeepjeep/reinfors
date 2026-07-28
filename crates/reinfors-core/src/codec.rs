//! The persistence seam: a `StateCodec` serializes a game's native `State` to opaque bytes.
//! Optional capability, deliberately separate from [`Game`](crate::Game): serialization is never a
//! bound on `Game::State`, and downstream games implement it only if they want persistent
//! snapshots.
//!
//! The contract is deliberately narrow: decoded states are structurally valid and **safe for all
//! game operations**; snapshots are opaque, and only snapshots produced by reinfors have
//! meaningful gameplay semantics. Codecs do NOT prove a decoded state is reachable through legal
//! play — proving reachability would mean mirroring the game's rules in a second place, which is
//! exactly the duplication (and drift risk) this seam avoids. Where a lifecycle flag is derivable
//! from the state, `decode` recomputes it from the same functions `step` uses rather than
//! transporting a second copy.

/// Encode/decode one game's `State`. Encoding is infallible (a live state is always encodable);
/// decoding returns a message for every malformed input, never panics. Implementations version
/// their own byte layout (lead with a version byte) — the envelope above carries only the
/// snapshot schema, not per-game layouts. `decode` is structural (built-in games derive it — no
/// hand-written byte plumbing) plus recomputation of derived fields; semantic safety checks live
/// in [`validate_decoded_state`](StateCodec::validate_decoded_state).
pub trait StateCodec: Send + Sync {
    type State;

    fn encode(&self, state: &Self::State) -> Vec<u8>;

    fn decode(&self, bytes: &[u8]) -> Result<Self::State, String>;

    /// Safety validation of a decoded state — callers installing decoded states must run this.
    /// Its explicit responsibilities: prevent panics and invalid indexing, prevent arithmetic
    /// failures, enforce representation invariants required by game methods, and keep environment
    /// lifecycle state (the independently transported `done` flag) coherent with the state. It
    /// explicitly does NOT prove reachability: states impossible under legal play are accepted as
    /// long as every game operation on them is safe.
    fn validate_decoded_state(&self, state: &Self::State, done: bool) -> Result<(), String> {
        let _ = (state, done);
        Ok(())
    }
}
