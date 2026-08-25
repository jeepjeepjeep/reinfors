//! Worker-read, scheduler-write cache for the zero-copy path: immutable copy-on-write
//! views published per fire. Design: `docs/development/scheduler-zero-copy.md`, element 7.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

#[derive(Default, Clone)]
struct Shard {
    current: HashMap<u128, Arc<[f32]>>,
    prev: HashMap<u128, Arc<[f32]>>,
}

/// An immutable snapshot; workers read only through the view pinned at spawn.
pub(crate) struct CacheView {
    pub(crate) generation: u64,
    mask: usize,
    shards: Box<[Arc<Shard>]>,
}

impl CacheView {
    fn empty(n_shards: usize, generation: u64) -> Arc<CacheView> {
        Arc::new(CacheView {
            generation,
            mask: n_shards - 1,
            shards: (0..n_shards).map(|_| Arc::new(Shard::default())).collect(),
        })
    }

    pub(crate) fn lookup(&self, key: u128) -> Option<Arc<[f32]>> {
        let shard = &self.shards[(key as usize) & self.mask];
        shard
            .current
            .get(&key)
            .or_else(|| shard.prev.get(&key))
            .cloned()
    }
}

pub(crate) struct ZcCache {
    view: Arc<CacheView>,
    shard_half: usize,
    generation: Arc<AtomicU64>,
    staged: Vec<(u128, Arc<[f32]>)>,
    promotes: Vec<u128>,
}

impl ZcCache {
    pub(crate) fn new(capacity: usize, generation: Arc<AtomicU64>) -> Self {
        let n_shards = (capacity / 128).max(1).next_power_of_two().min(256);
        let shard_half = (capacity / n_shards / 2).max(1);
        let seen = generation.load(Ordering::Relaxed);
        ZcCache {
            view: CacheView::empty(n_shards, seen),
            shard_half,
            generation,
            staged: Vec::new(),
            promotes: Vec::new(),
        }
    }

    pub(crate) fn view(&self) -> Arc<CacheView> {
        self.view.clone()
    }

    pub(crate) fn seen_generation(&self) -> u64 {
        self.view.generation
    }

    pub(crate) fn sync_generation(&mut self) {
        let now = self.generation.load(Ordering::Relaxed);
        if now != self.view.generation {
            self.view = CacheView::empty(self.view.shards.len(), now);
            self.staged.clear();
            self.promotes.clear();
        }
    }

    pub(crate) fn force_clear(&mut self) {
        let now = self.generation.load(Ordering::Relaxed);
        self.view = CacheView::empty(self.view.shards.len(), now);
        self.staged.clear();
        self.promotes.clear();
    }

    pub(crate) fn stage_insert(&mut self, key: u128, row: &[f64]) {
        self.staged
            .push((key, row.iter().map(|&v| v as f32).collect()));
    }

    /// Recency is scheduler-private: released hit keys promote at publication.
    pub(crate) fn stage_promote(&mut self, key: u128) {
        self.promotes.push(key);
    }

    /// Batched publication: clone each touched shard once shallowly, apply its
    /// mutations, publish one new view. Inserts staged before a generation
    /// boundary must not seed the post-boundary cache.
    pub(crate) fn publish(&mut self, generation_at_fire: u64) {
        self.sync_generation();
        if self.view.generation != generation_at_fire {
            self.staged.clear();
            self.promotes.clear();
            return;
        }
        if self.staged.is_empty() && self.promotes.is_empty() {
            return;
        }
        let mask = self.view.mask;
        let mut touched: HashMap<usize, Shard> = HashMap::new();
        for key in self.promotes.drain(..) {
            let si = (key as usize) & mask;
            // Prefer current: promote only prev-half keys absent from current,
            // and never clone a shard for the others.
            let (in_current, in_prev) = match touched.get(&si) {
                Some(t) => (t.current.contains_key(&key), t.prev.contains_key(&key)),
                None => {
                    let shard = &self.view.shards[si];
                    (
                        shard.current.contains_key(&key),
                        shard.prev.contains_key(&key),
                    )
                }
            };
            if in_current || !in_prev {
                continue;
            }
            let shard = touched
                .entry(si)
                .or_insert_with(|| (*self.view.shards[si]).clone());
            let row = shard.prev.remove(&key).expect("membership checked");
            if shard.current.len() >= self.shard_half {
                shard.prev = std::mem::take(&mut shard.current);
            }
            shard.current.insert(key, row);
        }
        let shard_half = self.shard_half;
        for (key, row) in self.staged.drain(..) {
            let si = (key as usize) & mask;
            let shard = touched
                .entry(si)
                .or_insert_with(|| (*self.view.shards[si]).clone());
            // The halves stay disjoint: replace in place without rotation, and a
            // key rotated to prev moves back rather than duplicating.
            if let Some(slot) = shard.current.get_mut(&key) {
                *slot = row;
                continue;
            }
            shard.prev.remove(&key);
            if shard.current.len() >= shard_half {
                shard.prev = std::mem::take(&mut shard.current);
            }
            shard.current.insert(key, row);
        }
        if touched.is_empty() {
            return;
        }
        let shards: Box<[Arc<Shard>]> = self
            .view
            .shards
            .iter()
            .enumerate()
            .map(|(i, s)| match touched.remove(&i) {
                Some(t) => Arc::new(t),
                None => s.clone(),
            })
            .collect();
        self.view = Arc::new(CacheView {
            generation: self.view.generation,
            mask,
            shards,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cache(cap: usize) -> (ZcCache, Arc<AtomicU64>) {
        let generation = Arc::new(AtomicU64::new(0));
        (ZcCache::new(cap, generation.clone()), generation)
    }

    #[test]
    fn published_entries_are_visible_only_in_new_views() {
        let (mut c, _) = cache(256);
        let pinned = c.view();
        c.stage_insert(7, &[1.5, -2.0]);
        c.publish(0);
        assert!(pinned.lookup(7).is_none(), "pinned views are immutable");
        let row = c.view().lookup(7).expect("published entry");
        assert_eq!(&row[..], &[1.5, -2.0]);
    }

    #[test]
    fn generation_bump_clears_and_blocks_stale_inserts() {
        let (mut c, generation) = cache(256);
        c.stage_insert(1, &[1.0]);
        c.publish(0);
        generation.fetch_add(1, Ordering::Relaxed);
        c.stage_insert(2, &[2.0]);
        c.publish(0);
        let view = c.view();
        assert_eq!(view.generation, 1);
        assert!(view.lookup(1).is_none(), "pre-boundary entries survive");
        assert!(
            view.lookup(2).is_none(),
            "pre-boundary fire seeded the cache"
        );
    }

    #[test]
    fn current_half_promotions_leave_the_view_untouched() {
        let (mut c, _) = cache(256);
        c.stage_insert(1, &[1.0]);
        c.publish(0);
        let before = c.view();
        c.stage_promote(1);
        c.publish(0);
        assert!(
            Arc::ptr_eq(&before, &c.view()),
            "a current-half promotion must not publish a new view"
        );
    }

    #[test]
    fn duplicate_keys_across_halves_cannot_resurrect_stale_values() {
        let (mut c, _) = cache(4);
        c.stage_insert(1, &[1.0]);
        c.stage_insert(2, &[2.0]);
        c.publish(0);
        c.stage_insert(3, &[3.0]);
        c.publish(0);
        assert!(c.view().lookup(1).is_some(), "key 1 rotated to prev");
        c.stage_insert(1, &[10.0]);
        c.publish(0);
        assert_eq!(&c.view().lookup(1).expect("reinserted")[..], &[10.0]);
        c.stage_promote(1);
        c.stage_insert(4, &[4.0]);
        c.publish(0);
        assert_eq!(
            &c.view().lookup(1).expect("still present")[..],
            &[10.0],
            "promotion resurrected the stale prev-half value"
        );
    }

    #[test]
    fn eviction_flips_halves_and_promotion_rescues() {
        let (mut c, _) = cache(4);
        c.stage_insert(1, &[1.0]);
        c.stage_insert(2, &[2.0]);
        c.publish(0);
        c.stage_insert(3, &[3.0]);
        c.publish(0);
        assert!(c.view().lookup(1).is_some(), "prev half still serves");
        c.stage_promote(1);
        c.stage_insert(4, &[4.0]);
        c.publish(0);
        assert!(c.view().lookup(1).is_some(), "promoted entry evicted");
    }
}
