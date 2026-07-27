//! The persistence seam: a `StateCodec` serializes a game's native `State` to opaque bytes and —
//! the important direction — validates while decoding. Optional capability, deliberately separate
//! from [`Game`](crate::Game): serialization is never a bound on `Game::State`, and downstream
//! games implement it only if they want persistent snapshots. Decode is the single boundary where
//! untrusted bytes become trusted state, so it must reject anything malformed (out-of-grid cells,
//! impossible checker counts, unparseable positions) — internal invariants downstream may assume
//! decoded states are well-formed.

/// Encode/decode one game's `State`. Encoding is infallible (a live state is always encodable);
/// decoding returns a message for every malformed input, never panics. Implementations version
/// their own byte layout (lead with a version byte) — the envelope above carries only the
/// snapshot schema, not per-game layouts.
pub trait StateCodec: Send + Sync {
    type State;

    fn encode(&self, state: &Self::State) -> Vec<u8>;

    fn decode(&self, bytes: &[u8]) -> Result<Self::State, String>;

    /// Consistency of a decoded state with an independently transported `done` flag (snapshot
    /// envelopes carry both). Games with a state-level flag require equality; games whose
    /// terminality is derivable require "definitely-terminal state implies done". Default: no
    /// check (a game with neither).
    fn check_done(&self, state: &Self::State, done: bool) -> Result<(), String> {
        let _ = (state, done);
        Ok(())
    }
}
