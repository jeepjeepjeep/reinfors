//! Optional net-evaluation cache at the infer seam — position-keyed reuse of raw net output rows.
//!
//! **Why**: transposition-rich sequential games (connect4, chess) re-reach positions across moves
//! and games; every re-evaluation is a wasted forward. The peer benchmark measured this reuse at
//! ~1.34× effective throughput for OpenSpiel's equivalent cache.
//!
//! **What is cached**: the *raw* net output row for an observation (per-head Q values, or logits +
//! value for the AlphaZero family) — pre-softmax, pre-noise — so consumers re-derive priors and
//! per-tree root noise identically to a fresh forward. Cache-on and cache-off searches are
//! bit-identical given the same weights (guarded by tests).
//!
//! **Efficiency** (the point of the feature): keys are 128-bit hashes of the observation bytes
//! (~0.1 µs vs a 100+ µs forward; no stored-obs comparison and no obs memory), values are f32 rows
//! (net precision anyway), and eviction is generational — two
//! rotating half-capacity maps with promote-on-old-hit — O(1) with zero per-hit bookkeeping,
//! unlike a strict LRU's list splice on every hit. Single-threaded by design (it lives inside the
//! engine's collect loop); no locks.
//!
//! **Invalidation**: entries are only valid for the weights that produced them, and the core
//! cannot observe the user's net. The contract is explicit: the trainer calls
//! `engine.weights_updated()` after installing new weights, which bumps a shared generation
//! counter (an atomic, so it works from the consumer thread while a `collect_stream` worker owns
//! the engine); the cache clears at the next round boundary. Never calling it asserts "weights
//! never changed" — which makes it correct, not stale.
//!
//! **The collision bet, stated for future maintainers**: no observation is stored, so a key
//! collision would silently return the WRONG row — a corrupted value backpropagated into training
//! data, with no crash. The 128-bit key width is what makes that negligible (collision-safe at
//! cache scale: even pessimistic effective-entropy estimates put the probability around 1e-12 or
//! below at realistic cache sizes — the same bet chess engines make with 64-bit transposition
//! keys, with far more margin). Do not shrink the key or weaken the mixer without weighing that
//! failure mode.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// splitmix64 finalizer: full-avalanche mixing of a 64-bit accumulator (each input bit flips each
/// output bit with ~50% probability) — repairs the weak diffusion of multiply-xor stream hashes.
fn avalanche(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// 128-bit content hash: two multiply-xor streams with *distinct odd multipliers* (FNV-1a prime;
/// the golden-ratio constant) and distinct offsets, each finalized with a splitmix64 avalanche and
/// salted with the length. Deterministic, dependency-free, and collision-safe at cache scale —
/// deliberately NOT claimed to be a full 2⁻¹²⁸ guarantee (the streams are strong but not proven
/// independent); see the module doc for what a collision would cost and why the margin suffices.
fn hash128(bytes: &[u8]) -> u128 {
    const PRIME_A: u64 = 0x0000_0100_0000_01B3; // FNV-1a 64
    const PRIME_B: u64 = 0x9E37_79B9_7F4A_7C15; // 2^64 / golden ratio (odd)
    let mut a: u64 = 0xcbf2_9ce4_8422_2325;
    let mut b: u64 = 0x6a09_e667_f3bc_c909; // sqrt(2) fractional bits
    for &x in bytes {
        a = (a ^ u64::from(x)).wrapping_mul(PRIME_A);
        b = (b ^ u64::from(x)).wrapping_mul(PRIME_B);
    }
    let a = avalanche(a ^ bytes.len() as u64);
    let b = avalanche(b.rotate_left(32) ^ bytes.len() as u64);
    (u128::from(a) << 64) | u128::from(b)
}

fn obs_key(obs: &[f32]) -> u128 {
    // Observations are deterministic encoder output; bit-exact keying over the raw f32 bytes.
    let bytes = unsafe { std::slice::from_raw_parts(obs.as_ptr().cast::<u8>(), obs.len() * 4) };
    hash128(bytes)
}

pub struct InferCache {
    current: HashMap<u128, Box<[f32]>>,
    prev: HashMap<u128, Box<[f32]>>,
    half_capacity: usize,
    generation: Arc<AtomicU64>,
    seen_generation: u64,
    // Per-collect counters (reset by `begin_collect`, read into telemetry).
    pub lookups: usize,
    pub hits: usize,
}

impl InferCache {
    /// `capacity` = max cached positions (split across the two generations); `generation` is the
    /// shared weights-version counter `engine.weights_updated()` bumps.
    pub fn new(capacity: usize, generation: Arc<AtomicU64>) -> Self {
        let half = (capacity / 2).max(1);
        InferCache {
            // Maps allocate on first insert, so constructed-but-unused caches (the engine's
            // inactive routing-mode slots) cost nothing.
            current: HashMap::new(),
            prev: HashMap::new(),
            half_capacity: half,
            seen_generation: generation.load(Ordering::Relaxed),
            generation,
            lookups: 0,
            hits: 0,
        }
    }

    /// Reset per-collect telemetry counters and pick up any pending weights bump.
    pub fn begin_collect(&mut self) {
        self.lookups = 0;
        self.hits = 0;
        self.sync_generation();
    }

    /// Clear everything unconditionally — restore installs state that may pair with different
    /// net weights than whatever warmed this cache, even at an equal generation NUMBER (the
    /// counter is engine-local, not a weights identity).
    pub fn force_clear(&mut self) {
        self.current.clear();
        self.prev.clear();
    }

    /// Clear everything if the trainer bumped the weights generation. Called at round boundaries
    /// (one relaxed atomic load), so a mid-collect sync under `collect_stream` pipelining takes
    /// effect within one search round.
    pub fn sync_generation(&mut self) {
        let now = self.generation.load(Ordering::Relaxed);
        if now != self.seen_generation {
            self.current.clear();
            self.prev.clear();
            self.seen_generation = now;
        }
    }

    /// The key for an observation — exposed so callers staging batch rows can reuse it for
    /// within-batch dedup without hashing twice.
    /// A key salted with the PLAYER whose network the row belongs to — per-player routing must
    /// never let two nets share an observation-keyed entry. (Shared-network mode keeps the
    /// untagged [`key`](Self::key), preserving its hit behavior byte-for-byte.)
    pub fn key_for_player(player: usize, obs: &[f32]) -> u128 {
        Self::key(obs)
            ^ u128::from(avalanche(player as u64 ^ 0x9E37_79B9_7F4A_7C15)).rotate_left(64)
    }

    pub fn key(obs: &[f32]) -> u128 {
        obs_key(obs)
    }

    /// Look up a cached row by key, promoting old-generation hits. Returns the row upcast to f64
    /// (the infer contract's element type).
    pub fn lookup(&mut self, key: u128) -> Option<Vec<f64>> {
        self.lookups += 1;
        if let Some(row) = self.current.get(&key) {
            self.hits += 1;
            return Some(row.iter().map(|&v| f64::from(v)).collect());
        }
        if let Some(row) = self.prev.remove(&key) {
            self.hits += 1;
            let out = row.iter().map(|&v| f64::from(v)).collect();
            self.insert_raw(key, row);
            return Some(out);
        }
        None
    }

    /// Insert a freshly computed row (stored as f32 — net precision).
    pub fn insert(&mut self, key: u128, row: &[f64]) {
        let stored: Box<[f32]> = row.iter().map(|&v| v as f32).collect();
        self.insert_raw(key, stored);
    }

    fn insert_raw(&mut self, key: u128, row: Box<[f32]>) {
        if self.current.len() >= self.half_capacity {
            self.prev = std::mem::take(&mut self.current);
        }
        self.current.insert(key, row);
    }

    pub fn len(&self) -> usize {
        self.current.len() + self.prev.len()
    }

    pub fn is_empty(&self) -> bool {
        self.current.is_empty() && self.prev.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cache(cap: usize) -> (InferCache, Arc<AtomicU64>) {
        let generation = Arc::new(AtomicU64::new(0));
        (InferCache::new(cap, generation.clone()), generation)
    }

    #[test]
    fn roundtrip_and_counters() {
        let (mut c, _) = cache(8);
        let obs = [0.5f32, 1.0, 0.0];
        let key = InferCache::key(&obs);
        assert!(c.lookup(key).is_none());
        c.insert(key, &[1.5, -2.0]);
        assert_eq!(c.lookup(key), Some(vec![1.5, -2.0]));
        assert_eq!((c.lookups, c.hits), (2, 1));
    }

    #[test]
    fn near_identical_observations_diverge_in_both_key_halves() {
        // The failure mode a weak mixer invites: related inputs (one plane bit flipped — the
        // typical single-piece board delta) yielding related keys. Both 64-bit halves must differ.
        let base = vec![0.0f32; 64];
        let base_key = InferCache::key(&base);
        for i in 0..64 {
            let mut flipped = base.clone();
            flipped[i] = 1.0;
            let k = InferCache::key(&flipped);
            assert_ne!(k as u64, base_key as u64, "low half collided on flip {i}");
            assert_ne!(k >> 64, base_key >> 64, "high half collided on flip {i}");
        }
    }

    #[test]
    fn distinct_observations_get_distinct_keys() {
        let a = InferCache::key(&[0.0f32, 1.0]);
        let b = InferCache::key(&[1.0f32, 0.0]);
        let c = InferCache::key(&[0.0f32, 1.0]);
        assert_ne!(a, b);
        assert_eq!(a, c);
    }

    #[test]
    fn generational_eviction_bounds_size_and_promotes_hot_entries() {
        let (mut c, _) = cache(4); // half = 2
        for i in 0..2 {
            c.insert(InferCache::key(&[i as f32]), &[f64::from(i)]);
        }
        // rotation: inserting a 3rd moves the first two to `prev`
        c.insert(InferCache::key(&[9.0f32]), &[9.0]);
        assert!(c.len() <= 4);
        // an old-generation hit survives the NEXT rotation via promotion
        let hot = InferCache::key(&[0.0f32]);
        assert!(c.lookup(hot).is_some());
        c.insert(InferCache::key(&[10.0f32]), &[10.0]);
        c.insert(InferCache::key(&[11.0f32]), &[11.0]);
        assert!(c.lookup(hot).is_some(), "promoted entry evicted too early");
    }

    #[test]
    fn weights_generation_clears_at_sync() {
        let (mut c, generation) = cache(8);
        let key = InferCache::key(&[1.0f32]);
        c.insert(key, &[2.0]);
        c.sync_generation();
        assert!(c.lookup(key).is_some(), "no bump -> entries persist");
        generation.fetch_add(1, Ordering::Relaxed);
        c.sync_generation();
        assert!(c.is_empty());
        assert!(c.lookup(key).is_none());
    }

    #[test]
    fn f32_storage_roundtrips_net_precision() {
        let (mut c, _) = cache(4);
        let key = InferCache::key(&[3.0f32]);
        let row = [0.123456789f64, -1e-3];
        c.insert(key, &row);
        let got = c.lookup(key).unwrap();
        for (g, r) in got.iter().zip(row) {
            assert!((g - r).abs() < 1e-6); // f32 quantization only
        }
    }
}
