//! Generational cache for raw network-output rows.
//!
//! Each cache is confined to one engine collection worker. It is deliberately mutable and
//! lock-free; the atomic generation signal does not make the cache itself thread-safe.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

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

    pub fn len(&self) -> usize {
        self.current.len() + self.prev.len()
    }

    pub fn is_empty(&self) -> bool {
        self.current.is_empty() && self.prev.is_empty()
    }
}

/// Concurrent cache for grouped collection: the key space is partitioned across
/// independently locked shards, so accessors collide only within a shard. Not a replica
/// set — an entry lives in exactly one shard, chosen by key hash, so there is no
/// cross-shard coherence. Generations synchronize lazily per access; the serve guarantee
/// is per-read (stronger than the exclusive cache's batch-boundary window). Total
/// configured capacity is divided across shards, so sharded eviction differs from the
/// exclusive cache — grouped digests are a distinct composition by design.
///
/// Lock discipline: every operation acquires exactly one shard lock, never holds two,
/// and never spans inference or channel work.
pub struct ShardedInferCache {
    shards: Vec<Mutex<InferCache>>,
    generation: Arc<AtomicU64>,
    mask: u64,
}

impl ShardedInferCache {
    /// `capacity` is the TOTAL entry budget, divided across `shards` (a power of two).
    pub fn new(capacity: usize, shards: usize, generation: Arc<AtomicU64>) -> Self {
        assert!(
            shards.is_power_of_two(),
            "shard count must be a power of two"
        );
        let per_shard = (capacity / shards).max(2);
        ShardedInferCache {
            shards: (0..shards)
                .map(|_| Mutex::new(InferCache::new(per_shard, generation.clone())))
                .collect(),
            mask: shards as u64 - 1,
            generation,
        }
    }

    fn shard(&self, key: u128) -> &Mutex<InferCache> {
        &self.shards[(key as u64 & self.mask) as usize]
    }

    /// The current weights generation (batch staging captures this).
    pub fn generation(&self) -> u64 {
        self.generation.load(Ordering::Relaxed)
    }

    pub fn lookup(&self, key: u128) -> Option<Vec<f64>> {
        let mut shard = self.shard(key).lock().expect("cache shard poisoned");
        shard.sync_generation();
        shard.lookup(key)
    }

    /// Insert a row computed under weights generation `staged`. Rejected when the shard
    /// has advanced past `staged` — rows from superseded weights never enter.
    pub fn insert(&self, key: u128, row: &[f64], staged: u64) {
        let mut shard = self.shard(key).lock().expect("cache shard poisoned");
        shard.sync_generation();
        debug_assert!(
            shard.seen_generation() >= staged,
            "generations only advance"
        );
        if shard.seen_generation() == staged {
            shard.insert(key, row);
        }
    }

    pub fn begin_collect(&self) {
        for shard in &self.shards {
            shard.lock().expect("cache shard poisoned").begin_collect();
        }
    }

    pub fn lookups(&self) -> usize {
        self.shards
            .iter()
            .map(|s| s.lock().expect("cache shard poisoned").lookups)
            .sum()
    }

    pub fn hits(&self) -> usize {
        self.shards
            .iter()
            .map(|s| s.lock().expect("cache shard poisoned").hits)
            .sum()
    }

    pub fn len(&self) -> usize {
        self.shards
            .iter()
            .map(|s| s.lock().expect("cache shard poisoned").len())
            .sum()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
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

#[cfg(test)]
mod sharded_tests {
    use super::*;
    use std::sync::atomic::Ordering;

    fn cache(capacity: usize) -> (ShardedInferCache, Arc<AtomicU64>) {
        let generation = Arc::new(AtomicU64::new(0));
        (
            ShardedInferCache::new(capacity, 16, generation.clone()),
            generation,
        )
    }

    #[test]
    fn lookup_after_insert_round_trips() {
        let (c, _g) = cache(1024);
        let key = InferCache::key(&[1.0, 2.0]);
        assert!(c.lookup(key).is_none());
        c.insert(key, &[0.5, 0.25], 0);
        assert_eq!(c.lookup(key).unwrap(), vec![0.5, 0.25]);
        assert_eq!((c.lookups(), c.hits()), (2, 1));
    }

    #[test]
    fn superseded_generation_rows_never_enter() {
        let (c, g) = cache(1024);
        let key = InferCache::key(&[3.0, 4.0]);
        let staged = c.generation();
        g.fetch_add(1, Ordering::Relaxed);
        c.insert(key, &[9.0, 9.0], staged);
        assert!(c.lookup(key).is_none(), "stale insert must be rejected");
        c.insert(key, &[7.0, 7.0], c.generation());
        assert_eq!(c.lookup(key).unwrap(), vec![7.0, 7.0]);
    }

    #[test]
    fn generation_bump_clears_lazily_per_shard() {
        let (c, g) = cache(1024);
        let keys: Vec<u128> = (0..64).map(|i| InferCache::key(&[i as f32, 1.0])).collect();
        for &k in &keys {
            c.insert(k, &[1.0], c.generation());
        }
        g.fetch_add(1, Ordering::Relaxed);
        for &k in &keys {
            assert!(
                c.lookup(k).is_none(),
                "pre-bump entries must never be served"
            );
        }
    }

    #[test]
    fn total_capacity_is_divided_not_multiplied() {
        let (c, _g) = cache(256);
        for i in 0..4096u32 {
            let k = InferCache::key(&[i as f32, 0.5]);
            c.insert(k, &[1.0], 0);
        }
        assert!(c.len() <= 256, "stored {} > configured total 256", c.len());
    }

    #[test]
    fn begin_collect_resets_counters() {
        let (c, _g) = cache(1024);
        let key = InferCache::key(&[1.0, 1.0]);
        c.insert(key, &[1.0], 0);
        let _ = c.lookup(key);
        c.begin_collect();
        assert_eq!((c.lookups(), c.hits()), (0, 0));
    }

    #[test]
    fn concurrent_hammer_with_generation_churn() {
        let (c, g) = cache(4096);
        std::thread::scope(|scope| {
            for t in 0..4u32 {
                let c = &c;
                scope.spawn(move || {
                    for i in 0..2000u32 {
                        let k = InferCache::key(&[(i % 257) as f32, t as f32]);
                        let staged = c.generation();
                        if i % 3 == 0 {
                            c.insert(k, &[f64::from(i)], staged);
                        } else {
                            let _ = c.lookup(k);
                        }
                    }
                });
            }
            let g = &g;
            scope.spawn(move || {
                for _ in 0..20 {
                    g.fetch_add(1, Ordering::Relaxed);
                    std::thread::yield_now();
                }
            });
        });
        assert!(c.lookups() > 0);
        let _ = c.len();
    }
}
