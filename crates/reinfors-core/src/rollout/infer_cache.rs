//! Generational cache for raw network-output rows.
//!
//! Each cache is confined to one engine collection worker. It is deliberately mutable and
//! lock-free; the atomic generation signal does not make the cache itself thread-safe.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

fn avalanche(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

// Keys omit the original observation, so weakening this mixer risks silent false hits.
fn hash128(bytes: &[u8]) -> u128 {
    const PRIME_A: u64 = 0x0000_0100_0000_01B3;
    const PRIME_B: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut a: u64 = 0xcbf2_9ce4_8422_2325;
    let mut b: u64 = 0x6a09_e667_f3bc_c909;
    for &x in bytes {
        a = (a ^ u64::from(x)).wrapping_mul(PRIME_A);
        b = (b ^ u64::from(x)).wrapping_mul(PRIME_B);
    }
    let a = avalanche(a ^ bytes.len() as u64);
    let b = avalanche(b.rotate_left(32) ^ bytes.len() as u64);
    (u128::from(a) << 64) | u128::from(b)
}

/// Streaming variant of the row hash for encoder-derived cache keys. The seed
/// is salted so encoder-key and observation-hash key spaces cannot collide.
pub struct CacheHasher {
    a: u64,
    b: u64,
    len: u64,
}

impl CacheHasher {
    /// Engine-seeded: the tagged seed and player salt keep encoder keys and
    /// observation-hash keys in disjoint spaces.
    pub fn seeded(player: usize) -> Self {
        let mut h = CacheHasher {
            a: 0xcbf2_9ce4_8422_2325 ^ 0xE5C0_DE4B_5EED_0001,
            b: 0x6a09_e667_f3bc_c909,
            len: 0,
        };
        h.write_u64(player as u64);
        h
    }

    fn absorb(&mut self, bytes: &[u8]) {
        const PRIME_A: u64 = 0x0000_0100_0000_01B3;
        const PRIME_B: u64 = 0x9E37_79B9_7F4A_7C15;
        for &x in bytes {
            self.a = (self.a ^ u64::from(x)).wrapping_mul(PRIME_A);
            self.b = (self.b ^ u64::from(x)).wrapping_mul(PRIME_B);
        }
        self.len += bytes.len() as u64;
    }

    /// Variable-length fields are length-prefixed, so `([1], [2,3])` and
    /// `([1,2], [3])` produce distinct streams.
    pub fn write(&mut self, bytes: &[u8]) {
        self.absorb(&(bytes.len() as u64).to_le_bytes());
        self.absorb(bytes);
    }

    pub fn write_u8(&mut self, v: u8) {
        self.absorb(&[v]);
    }

    pub fn write_u64(&mut self, v: u64) {
        self.absorb(&v.to_le_bytes());
    }

    pub fn write_usize(&mut self, v: usize) {
        self.write_u64(v as u64);
    }

    pub fn write_f32s(&mut self, v: &[f32]) {
        self.absorb(&(v.len() as u64).to_le_bytes());
        let bytes = unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), v.len() * 4) };
        self.absorb(bytes);
    }

    pub fn finish(&self) -> u128 {
        let a = avalanche(self.a ^ self.len);
        let b = avalanche(self.b.rotate_left(32) ^ self.len);
        (u128::from(a) << 64) | u128::from(b)
    }
}

fn obs_key(obs: &[f32]) -> u128 {
    let bytes = unsafe { std::slice::from_raw_parts(obs.as_ptr().cast::<u8>(), obs.len() * 4) };
    hash128(bytes)
}

pub struct InferCache {
    current: HashMap<u128, Box<[f32]>>,
    prev: HashMap<u128, Box<[f32]>>,
    half_capacity: usize,
    generation: Arc<AtomicU64>,
    seen_generation: u64,
    pub lookups: usize,
    pub hits: usize,
}

impl InferCache {
    pub fn new(capacity: usize, generation: Arc<AtomicU64>) -> Self {
        let half = (capacity / 2).max(1);
        InferCache {
            current: HashMap::new(),
            prev: HashMap::new(),
            half_capacity: half,
            seen_generation: generation.load(Ordering::Relaxed),
            generation,
            lookups: 0,
            hits: 0,
        }
    }

    pub fn begin_collect(&mut self) {
        self.lookups = 0;
        self.hits = 0;
        self.sync_generation();
    }

    /// Clear entries even when the numeric generation is unchanged.
    pub fn force_clear(&mut self) {
        self.current.clear();
        self.prev.clear();
    }

    /// The generation this cache last synced to.
    pub fn seen_generation(&self) -> u64 {
        self.seen_generation
    }

    pub(crate) fn params(&self) -> (usize, Arc<AtomicU64>) {
        (self.half_capacity * 2, self.generation.clone())
    }

    /// Clear entries when the shared weights generation changes.
    pub fn sync_generation(&mut self) {
        let now = self.generation.load(Ordering::Relaxed);
        if now != self.seen_generation {
            self.current.clear();
            self.prev.clear();
            self.seen_generation = now;
        }
    }

    /// Salt an observation key by player to isolate routed networks.
    pub fn key_for_player(player: usize, obs: &[f32]) -> u128 {
        Self::key(obs)
            ^ u128::from(avalanche(player as u64 ^ 0x9E37_79B9_7F4A_7C15)).rotate_left(64)
    }

    pub fn key(obs: &[f32]) -> u128 {
        obs_key(obs)
    }

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

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.current.len() + self.prev.len()
    }

    #[cfg(test)]
    pub(crate) fn is_empty(&self) -> bool {
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
        let (mut c, _) = cache(4);
        for i in 0..2 {
            c.insert(InferCache::key(&[i as f32]), &[f64::from(i)]);
        }
        c.insert(InferCache::key(&[9.0f32]), &[9.0]);
        assert!(c.len() <= 4);
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
            assert!((g - r).abs() < 1e-6);
        }
    }
}
