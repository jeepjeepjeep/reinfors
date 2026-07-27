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
    ///
    /// Required rather than defaulted: the contract carries safety obligations, so a codec cannot
    /// claim it by accident. A game with no invariants beyond structure returns `Ok(())`
    /// explicitly.
    fn validate_decoded_state(&self, state: &Self::State, done: bool) -> Result<(), String>;
}

/// Bounds-checked little-endian byte plumbing for snapshot payloads (public: the binding and
/// per-family evaluation codecs build on it). Reads error on truncation — never panic.
pub mod bytes {
    pub struct Reader<'a> {
        bytes: &'a [u8],
        pos: usize,
    }

    impl<'a> Reader<'a> {
        pub fn new(bytes: &'a [u8]) -> Self {
            Reader { bytes, pos: 0 }
        }

        pub fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
            let end = self
                .pos
                .checked_add(n)
                .filter(|&e| e <= self.bytes.len())
                .ok_or_else(|| format!("truncated snapshot: needed {n} bytes at {}", self.pos))?;
            let out = &self.bytes[self.pos..end];
            self.pos = end;
            Ok(out)
        }

        pub fn u8(&mut self) -> Result<u8, String> {
            Ok(self.take(1)?[0])
        }

        pub fn u32(&mut self) -> Result<u32, String> {
            Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
        }

        pub fn u64(&mut self) -> Result<u64, String> {
            Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
        }

        pub fn i64(&mut self) -> Result<i64, String> {
            Ok(i64::from_le_bytes(self.take(8)?.try_into().unwrap()))
        }

        pub fn f64(&mut self) -> Result<f64, String> {
            Ok(f64::from_le_bytes(self.take(8)?.try_into().unwrap()))
        }

        pub fn f32(&mut self) -> Result<f32, String> {
            Ok(f32::from_le_bytes(self.take(4)?.try_into().unwrap()))
        }

        pub fn blob(&mut self) -> Result<&'a [u8], String> {
            let n = self.u32()? as usize;
            self.take(n)
        }

        pub fn done(self) -> Result<(), String> {
            if self.pos == self.bytes.len() {
                Ok(())
            } else {
                Err(format!("{} trailing bytes", self.bytes.len() - self.pos))
            }
        }
    }

    pub fn put_u8(out: &mut Vec<u8>, v: u8) {
        out.push(v);
    }
    pub fn put_u32(out: &mut Vec<u8>, v: u32) {
        out.extend_from_slice(&v.to_le_bytes());
    }
    pub fn put_u64(out: &mut Vec<u8>, v: u64) {
        out.extend_from_slice(&v.to_le_bytes());
    }
    pub fn put_i64(out: &mut Vec<u8>, v: i64) {
        out.extend_from_slice(&v.to_le_bytes());
    }
    pub fn put_f64(out: &mut Vec<u8>, v: f64) {
        out.extend_from_slice(&v.to_le_bytes());
    }
    pub fn put_f32(out: &mut Vec<u8>, v: f32) {
        out.extend_from_slice(&v.to_le_bytes());
    }
    pub fn put_blob(out: &mut Vec<u8>, b: &[u8]) {
        put_u32(out, b.len() as u32);
        out.extend_from_slice(b);
    }

    pub fn put_usizes(out: &mut Vec<u8>, xs: &[usize]) {
        put_u32(out, xs.len() as u32);
        for &x in xs {
            put_u64(out, x as u64);
        }
    }
    pub fn usizes(r: &mut Reader) -> Result<Vec<usize>, String> {
        let n = r.u32()? as usize;
        (0..n).map(|_| Ok(r.u64()? as usize)).collect()
    }
    pub fn put_f64s(out: &mut Vec<u8>, xs: &[f64]) {
        put_u32(out, xs.len() as u32);
        for &x in xs {
            put_f64(out, x);
        }
    }
    pub fn f64s(r: &mut Reader) -> Result<Vec<f64>, String> {
        let n = r.u32()? as usize;
        (0..n).map(|_| r.f64()).collect()
    }
    pub fn put_f32s(out: &mut Vec<u8>, xs: &[f32]) {
        put_u32(out, xs.len() as u32);
        for &x in xs {
            put_f32(out, x);
        }
    }
    pub fn f32s(r: &mut Reader) -> Result<Vec<f32>, String> {
        let n = r.u32()? as usize;
        (0..n).map(|_| r.f32()).collect()
    }
}
