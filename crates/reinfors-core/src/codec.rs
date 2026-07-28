//! The persistence seam: a `StateCodec` serializes a game's native `State` to opaque bytes and —
//! the important direction — validates while decoding. Optional capability, deliberately separate
//! from [`Game`](crate::Game): serialization is never a bound on `Game::State`, and downstream
//! games implement it only if they want persistent snapshots. `decode` is STRUCTURAL (built-in
//! games derive it — no hand-written byte plumbing); [`validate_state`](StateCodec::validate_state)
//! is the single semantic boundary where untrusted input becomes trusted state (out-of-grid
//! cells, impossible checker counts, terminal inconsistencies) — callers run it on every decoded
//! state before installing it, and downstream invariants may then assume well-formedness.

/// Encode/decode one game's `State`. Encoding is infallible (a live state is always encodable);
/// decoding returns a message for every malformed input, never panics. Implementations version
/// their own byte layout (lead with a version byte) — the envelope above carries only the
/// snapshot schema, not per-game layouts.
pub trait StateCodec: Send + Sync {
    type State;

    fn encode(&self, state: &Self::State) -> Vec<u8>;

    fn decode(&self, bytes: &[u8]) -> Result<Self::State, String>;

    /// SEMANTIC validation of a decoded state against the game's invariants and the
    /// independently transported `done` flag — the untrusted-input boundary now that `decode` is
    /// purely structural (derived deserialization). Callers installing decoded states must run
    /// this; games express their invariants once, over typed fields, byte-layout-free.
    fn validate_state(&self, state: &Self::State, done: bool) -> Result<(), String> {
        let _ = (state, done);
        Ok(())
    }
}
