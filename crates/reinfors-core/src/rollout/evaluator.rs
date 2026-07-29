//! The evaluation service: THE single path from any consumer (tree searches, episode-tail
//! bootstraps, plain forwards) to the net. Wraps the user's `infer` callback together with the
//! optional [`InferCache`] and all throughput telemetry, so no consumer can bypass caching or
//! account inconsistently — the layering the cache's first design got wrong (the cache was a peer
//! parameter each call site had to remember to consult).
//!
//! Consumers use a two-phase batch protocol:
//!
//! ```ignore
//! let mut batch = eval.batch();                 // syncs the weights generation once
//! match batch.resolve_or_stage(&obs) {
//!     Resolve::Resolved(row) => consume(row),   // cache hit — no forward will be paid
//!     Resolve::Staged(ticket) => wait(ticket),  // deduped: identical obs ⇒ identical ticket
//! }
//! let rows = batch.commit();                    // ONE pooled call of pure misses + cache insert
//! consume(rows.row(ticket));
//! ```
//!
//! Accounting is layered by where knowledge lives: the Evaluator counts *global throughput*
//! (calls, rows, seconds, cache lookups/hits — purpose-blind), while search-simulation bookkeeping
//! (the sim-fate identity) lives in the trees that know what a simulation is. A tail bootstrap is
//! therefore not a category here — just another batch.

use crate::rollout::infer_cache::InferCache;

pub struct Evaluator<'a, F> {
    infer: &'a mut F,
    cache: Option<&'a mut InferCache>,
    /// Global throughput telemetry (all consumers): forwarded rows, pooled calls, callback seconds.
    pub rows: usize,
    pub calls: usize,
    pub seconds: f64,
}

/// The outcome of offering one observation to a batch.
pub enum Resolve {
    /// Served from the cache — the row is available immediately, no forward will run for it.
    Resolved(Vec<f64>),
    /// Queued for the pooled call; redeem the ticket against [`CommittedRows`] after `commit`.
    /// Identical observations within one batch share a ticket (within-batch dedup).
    Staged(usize),
}

pub struct EvalBatch<'e, 'a, F> {
    eval: &'e mut Evaluator<'a, F>,
    obs_flat: Vec<f32>,
    keys: Vec<u128>,
    staged: std::collections::HashMap<u128, usize>,
    n: usize,
}

/// The committed rows of one batch, indexed by ticket.
pub struct CommittedRows {
    out: Vec<f64>,
    stride: usize,
}

impl CommittedRows {
    pub fn row(&self, ticket: usize) -> &[f64] {
        &self.out[ticket * self.stride..(ticket + 1) * self.stride]
    }
}

impl<'a, F> Evaluator<'a, F>
where
    F: FnMut(Vec<f32>, usize) -> Vec<f64>,
{
    pub fn new(infer: &'a mut F, cache: Option<&'a mut InferCache>) -> Self {
        Evaluator {
            infer,
            cache,
            rows: 0,
            calls: 0,
            seconds: 0.0,
        }
    }

    /// Start a batch. Picks up any pending `weights_updated` bump before any lookup can be served.
    pub fn batch<'e>(&'e mut self) -> EvalBatch<'e, 'a, F> {
        if let Some(cache) = self.cache.as_deref_mut() {
            cache.sync_generation();
        }
        EvalBatch {
            eval: self,
            obs_flat: Vec::new(),
            keys: Vec::new(),
            staged: std::collections::HashMap::new(),
            n: 0,
        }
    }

    /// Convenience single-phase forward for consumers whose whole request set is one batch (the
    /// non-tree policies, per-round expectimax pools, episode-tail bootstraps): every row goes
    /// through the same resolve/stage/commit path, so they get caching and dedup identically to the
    /// tree searches.
    pub fn forward(&mut self, obs: Vec<f32>, n: usize) -> Vec<f64> {
        let dim = obs.len() / n.max(1);
        let mut batch = self.batch();
        let mut tickets: Vec<Resolve> = Vec::with_capacity(n);
        for i in 0..n {
            tickets.push(batch.resolve_or_stage(&obs[i * dim..(i + 1) * dim]));
        }
        let stride_hint = tickets.iter().find_map(|t| match t {
            Resolve::Resolved(row) => Some(row.len()),
            Resolve::Staged(_) => None,
        });
        let rows = batch.commit();
        let stride = if rows.stride > 0 {
            rows.stride
        } else {
            stride_hint.unwrap_or(0)
        };
        let mut out = Vec::with_capacity(n * stride);
        for t in &tickets {
            match t {
                Resolve::Resolved(row) => out.extend_from_slice(row),
                Resolve::Staged(ticket) => out.extend_from_slice(rows.row(*ticket)),
            }
        }
        out
    }

    pub fn cache_lookups(&self) -> usize {
        self.cache.as_deref().map_or(0, |c| c.lookups)
    }

    pub fn cache_hits(&self) -> usize {
        self.cache.as_deref().map_or(0, |c| c.hits)
    }
}

impl<'e, 'a, F> EvalBatch<'e, 'a, F>
where
    F: FnMut(Vec<f32>, usize) -> Vec<f64>,
{
    /// Offer one observation: a cache hit resolves immediately; otherwise it is staged (deduped)
    /// for the pooled call.
    pub fn resolve_or_stage(&mut self, obs: &[f32]) -> Resolve {
        if let Some(cache) = self.eval.cache.as_deref_mut() {
            let key = InferCache::key(obs);
            if let Some(row) = cache.lookup(key) {
                return Resolve::Resolved(row);
            }
            if let Some(&ticket) = self.staged.get(&key) {
                return Resolve::Staged(ticket);
            }
            self.staged.insert(key, self.n);
            self.keys.push(key);
        }
        self.obs_flat.extend_from_slice(obs);
        let ticket = self.n;
        self.n += 1;
        Resolve::Staged(ticket)
    }

    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Run the single pooled forward over the staged misses, insert them into the cache, and
    /// return the rows for ticket redemption. An empty batch performs no call.
    pub fn commit(self) -> CommittedRows {
        if self.n == 0 {
            return CommittedRows {
                out: Vec::new(),
                stride: 0,
            };
        }
        let t = std::time::Instant::now();
        let out = (self.eval.infer)(self.obs_flat, self.n);
        self.eval.seconds += t.elapsed().as_secs_f64();
        self.eval.calls += 1;
        self.eval.rows += self.n;
        let stride = out.len() / self.n;
        if let Some(cache) = self.eval.cache.as_deref_mut() {
            for (i, &key) in self.keys.iter().enumerate() {
                cache.insert(key, &out[i * stride..(i + 1) * stride]);
            }
        }
        CommittedRows { out, stride }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU64;
    use std::sync::Arc;

    fn double_infer(obs: Vec<f32>, n: usize) -> Vec<f64> {
        // row = [2·obs] per staged observation, dim 2
        assert_eq!(obs.len(), n * 2);
        obs.iter().map(|&v| f64::from(v) * 2.0).collect()
    }

    #[test]
    fn batch_stages_dedupes_and_commits() {
        let mut infer = double_infer;
        let generation = Arc::new(AtomicU64::new(0));
        let mut cache = InferCache::new(64, generation);
        let mut eval = Evaluator::new(&mut infer, Some(&mut cache));

        let mut batch = eval.batch();
        let a = batch.resolve_or_stage(&[1.0, 2.0]);
        let b = batch.resolve_or_stage(&[3.0, 4.0]);
        let c = batch.resolve_or_stage(&[1.0, 2.0]); // duplicate of a
        let (Resolve::Staged(ta), Resolve::Staged(tb), Resolve::Staged(tc)) = (a, b, c) else {
            panic!("cold cache must stage everything");
        };
        assert_eq!(ta, tc, "within-batch dedup must share tickets");
        assert_ne!(ta, tb);
        let rows = batch.commit();
        assert_eq!(rows.row(ta), &[2.0, 4.0]);
        assert_eq!(rows.row(tb), &[6.0, 8.0]);
        assert_eq!((eval.rows, eval.calls), (2, 1)); // dedup: 2 rows forwarded, not 3

        // second batch: both positions now resolve from cache, commit is a no-op call-wise
        let mut batch = eval.batch();
        let Resolve::Resolved(row) = batch.resolve_or_stage(&[1.0, 2.0]) else {
            panic!("warm cache must resolve");
        };
        assert_eq!(row, vec![2.0, 4.0]);
        assert!(batch.is_empty());
        batch.commit();
        assert_eq!(eval.calls, 1, "empty commit must not call the net");
    }

    #[test]
    fn forward_convenience_matches_direct_call_and_uses_cache() {
        let mut infer = double_infer;
        let generation = Arc::new(AtomicU64::new(0));
        let mut cache = InferCache::new(64, generation);
        let mut eval = Evaluator::new(&mut infer, Some(&mut cache));
        let obs = vec![1.0f32, 2.0, 3.0, 4.0];
        let out = eval.forward(obs.clone(), 2);
        assert_eq!(out, vec![2.0, 4.0, 6.0, 8.0]);
        let again = eval.forward(obs, 2);
        assert_eq!(again, vec![2.0, 4.0, 6.0, 8.0]);
        assert_eq!(eval.rows, 2, "second forward should be fully cache-served");
    }

    #[test]
    fn cacheless_evaluator_still_batches() {
        let mut infer = double_infer;
        let mut eval: Evaluator<'_, _> = Evaluator::new(&mut infer, None);
        let mut batch = eval.batch();
        let Resolve::Staged(t0) = batch.resolve_or_stage(&[5.0, 6.0]) else {
            panic!()
        };
        let rows = batch.commit();
        assert_eq!(rows.row(t0), &[10.0, 12.0]);
        assert_eq!((eval.rows, eval.calls), (1, 1));
    }
}
