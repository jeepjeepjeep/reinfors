//! Batched network evaluation, caching, and throughput accounting. Every inference consumer must
//! pass through this service; the earlier peer-parameter design let call sites bypass caching and
//! telemetry accidentally.

use crate::rollout::infer_cache::InferCache;

/// Whether rows route to one shared network or one network per player.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum InferMode {
    Shared,
    PerPlayer,
}

pub struct Evaluator<'a, F> {
    infer: &'a mut F,
    mode: InferMode,
    caches: Option<&'a mut [InferCache]>,
    pub rows: usize,
    pub calls: usize,
    pub seconds: f64,
}

pub enum Resolve {
    Resolved(Vec<f64>),
    Staged(usize),
}

pub struct EvalBatch<'e, 'a, F> {
    eval: &'e mut Evaluator<'a, F>,
    obs_flat: Vec<f32>,
    keys: Vec<u128>,
    players: Vec<usize>,
    staged: std::collections::HashMap<u128, usize>,
    n: usize,
    dim: usize,
    generation: u64,
}

/// A batch detached from its evaluator: rows to forward elsewhere (e.g. on a submitter
/// thread), then hand back through [`Evaluator::ingest`].
pub struct StagedBatch {
    pub players: Vec<usize>,
    pub obs_flat: Vec<f32>,
    pub n: usize,
    pub dim: usize,
    keys: Vec<u128>,
    generation: u64,
}

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
    F: FnMut(usize, Vec<f32>, usize) -> Vec<f64>,
{
    pub fn new(infer: &'a mut F, mode: InferMode, caches: Option<&'a mut [InferCache]>) -> Self {
        Evaluator {
            infer,
            mode,
            caches,
            rows: 0,
            calls: 0,
            seconds: 0.0,
        }
    }

    fn cache_slot(&mut self, player: usize) -> Option<&mut InferCache> {
        let idx = match self.mode {
            InferMode::Shared => 0,
            InferMode::PerPlayer => player,
        };
        self.caches.as_deref_mut().map(|c| &mut c[idx])
    }

    fn row_key(&self, player: usize, obs: &[f32]) -> u128 {
        match self.mode {
            InferMode::Shared => InferCache::key(obs),
            InferMode::PerPlayer => InferCache::key_for_player(player, obs),
        }
    }

    pub fn batch<'e>(&'e mut self) -> EvalBatch<'e, 'a, F> {
        if let Some(caches) = self.caches.as_deref_mut() {
            for cache in caches.iter_mut() {
                cache.sync_generation();
            }
        }
        let generation = match self.caches.as_deref() {
            Some(caches) => caches.first().map_or(0, InferCache::seen_generation),
            None => 0,
        };
        EvalBatch {
            eval: self,
            obs_flat: Vec::new(),
            keys: Vec::new(),
            players: Vec::new(),
            staged: std::collections::HashMap::new(),
            n: 0,
            dim: 0,
            generation,
        }
    }

    pub fn forward(&mut self, players: &[usize], obs: Vec<f32>, n: usize) -> Vec<f64> {
        let dim = obs.len() / n.max(1);
        let mut batch = self.batch();
        let mut tickets: Vec<Resolve> = Vec::with_capacity(n);
        for i in 0..n {
            tickets.push(batch.resolve_or_stage(players[i], &obs[i * dim..(i + 1) * dim]));
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

    /// Account and cache rows forwarded elsewhere for a batch detached via `into_staged`.
    /// The freshness check compares the batch's staging generation to the cache's last-synced
    /// one, so an update landing after that sync can still admit a stale insert — but every
    /// batch synchronizes generations before its first lookup, clearing such entries first.
    /// The observable guarantee: the cache never SERVES rows from superseded weights.
    pub fn ingest(
        &mut self,
        staged: StagedBatch,
        out: Vec<f64>,
        seconds: f64,
        calls: usize,
    ) -> CommittedRows {
        if staged.n == 0 {
            return CommittedRows {
                out: Vec::new(),
                stride: 0,
            };
        }
        self.seconds += seconds;
        self.calls += calls;
        self.rows += staged.n;
        let stride = out.len() / staged.n;
        let fresh = match self.caches.as_deref() {
            Some(caches) => caches
                .first()
                .is_some_and(|c| c.seen_generation() == staged.generation),
            None => false,
        };
        if fresh {
            for i in 0..staged.keys.len() {
                let (key, player) = (staged.keys[i], staged.players[i]);
                let row = out[i * stride..(i + 1) * stride].to_vec();
                self.cache_slot(player)
                    .expect("caches present")
                    .insert(key, &row);
            }
        }
        CommittedRows { out, stride }
    }

    pub fn cache_lookups(&self) -> usize {
        self.caches
            .as_deref()
            .map_or(0, |c| c.iter().map(|x| x.lookups).sum())
    }

    pub fn cache_hits(&self) -> usize {
        self.caches
            .as_deref()
            .map_or(0, |c| c.iter().map(|x| x.hits).sum())
    }
}

/// Forward `n` staged rows through `infer`, honoring the routing mode; returns rows in
/// ticket order plus the number of callback invocations made.
pub fn run_infer<F>(
    infer: &mut F,
    mode: InferMode,
    players: &[usize],
    obs_flat: Vec<f32>,
    n: usize,
    dim: usize,
) -> (Vec<f64>, usize)
where
    F: FnMut(usize, Vec<f32>, usize) -> Vec<f64>,
{
    match mode {
        InferMode::Shared => (infer(0, obs_flat, n), 1),
        InferMode::PerPlayer => {
            // Preserve first-seen player order so routed calls remain deterministic.
            let mut order: Vec<usize> = Vec::new();
            let mut groups: std::collections::HashMap<usize, Vec<usize>> =
                std::collections::HashMap::new();
            for (ticket, &player) in players.iter().enumerate() {
                groups.entry(player).or_insert_with(|| {
                    order.push(player);
                    Vec::new()
                });
                groups.get_mut(&player).expect("just inserted").push(ticket);
            }
            let mut calls = 0;
            let mut out: Vec<f64> = Vec::new();
            let mut scattered: Vec<(usize, Vec<f64>)> = Vec::with_capacity(n);
            let mut group_stride: Option<usize> = None;
            for player in order {
                let tickets = &groups[&player];
                let mut obs: Vec<f32> = Vec::with_capacity(tickets.len() * dim);
                for &t in tickets {
                    obs.extend_from_slice(&obs_flat[t * dim..(t + 1) * dim]);
                }
                let rows = infer(player, obs, tickets.len());
                calls += 1;
                // A single stride is used below, so routed row widths must agree.
                assert!(
                    rows.len().is_multiple_of(tickets.len()),
                    "player {player} infer returned {} values for {} rows (not divisible)",
                    rows.len(),
                    tickets.len()
                );
                let stride = rows.len() / tickets.len();
                if let Some(expected) = group_stride {
                    assert_eq!(
                        stride, expected,
                        "player {player} infer row width differs from another player's"
                    );
                }
                group_stride = Some(stride);
                for (i, &t) in tickets.iter().enumerate() {
                    scattered.push((t, rows[i * stride..(i + 1) * stride].to_vec()));
                }
            }
            scattered.sort_by_key(|(t, _)| *t);
            for (_, row) in scattered {
                out.extend(row);
            }
            (out, calls)
        }
    }
}

impl<'e, 'a, F> EvalBatch<'e, 'a, F>
where
    F: FnMut(usize, Vec<f32>, usize) -> Vec<f64>,
{
    pub fn resolve_or_stage(&mut self, player: usize, obs: &[f32]) -> Resolve {
        if self.eval.caches.is_some() {
            let key = self.eval.row_key(player, obs);
            if let Some(row) = self
                .eval
                .cache_slot(player)
                .expect("caches present")
                .lookup(key)
            {
                return Resolve::Resolved(row);
            }
            if let Some(&ticket) = self.staged.get(&key) {
                return Resolve::Staged(ticket);
            }
            self.staged.insert(key, self.n);
            self.keys.push(key);
        }
        self.dim = obs.len();
        self.players.push(player);
        self.obs_flat.extend_from_slice(obs);
        let ticket = self.n;
        self.n += 1;
        Resolve::Staged(ticket)
    }

    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Forward staged misses and return rows in ticket order.
    pub fn commit(self) -> CommittedRows {
        if self.n == 0 {
            return CommittedRows {
                out: Vec::new(),
                stride: 0,
            };
        }
        let t = std::time::Instant::now();
        let (out, calls) = run_infer(
            self.eval.infer,
            self.eval.mode,
            &self.players,
            self.obs_flat,
            self.n,
            self.dim,
        );
        self.eval.seconds += t.elapsed().as_secs_f64();
        self.eval.calls += calls;
        self.eval.rows += self.n;
        let stride = out.len() / self.n;
        if self.eval.caches.is_some() {
            for i in 0..self.keys.len() {
                let (key, player) = (self.keys[i], self.players[i]);
                let row = out[i * stride..(i + 1) * stride].to_vec();
                self.eval
                    .cache_slot(player)
                    .expect("caches present")
                    .insert(key, &row);
            }
        }
        CommittedRows { out, stride }
    }

    /// Detach the staged rows from the evaluator without forwarding them.
    pub fn into_staged(self) -> StagedBatch {
        StagedBatch {
            players: self.players,
            obs_flat: self.obs_flat,
            n: self.n,
            dim: self.dim,
            keys: self.keys,
            generation: self.generation,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU64;
    use std::sync::Arc;

    fn double_infer(_player: usize, obs: Vec<f32>, n: usize) -> Vec<f64> {
        assert_eq!(obs.len(), n * 2);
        obs.iter().map(|&v| f64::from(v) * 2.0).collect()
    }

    #[test]
    fn batch_stages_dedupes_and_commits() {
        let mut infer = double_infer;
        let generation = Arc::new(AtomicU64::new(0));
        let mut cache = InferCache::new(64, generation);
        let mut eval = Evaluator::new(
            &mut infer,
            InferMode::Shared,
            Some(std::slice::from_mut(&mut cache)),
        );

        let mut batch = eval.batch();
        let a = batch.resolve_or_stage(0, &[1.0, 2.0]);
        let b = batch.resolve_or_stage(0, &[3.0, 4.0]);
        let c = batch.resolve_or_stage(0, &[1.0, 2.0]);
        let (Resolve::Staged(ta), Resolve::Staged(tb), Resolve::Staged(tc)) = (a, b, c) else {
            panic!("cold cache must stage everything");
        };
        assert_eq!(ta, tc, "within-batch dedup must share tickets");
        assert_ne!(ta, tb);
        let rows = batch.commit();
        assert_eq!(rows.row(ta), &[2.0, 4.0]);
        assert_eq!(rows.row(tb), &[6.0, 8.0]);
        assert_eq!((eval.rows, eval.calls), (2, 1));

        let mut batch = eval.batch();
        let Resolve::Resolved(row) = batch.resolve_or_stage(0, &[1.0, 2.0]) else {
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
        let mut eval = Evaluator::new(
            &mut infer,
            InferMode::Shared,
            Some(std::slice::from_mut(&mut cache)),
        );
        let obs = vec![1.0f32, 2.0, 3.0, 4.0];
        let out = eval.forward(&[0, 0], obs.clone(), 2);
        assert_eq!(out, vec![2.0, 4.0, 6.0, 8.0]);
        let again = eval.forward(&[0, 0], obs, 2);
        assert_eq!(again, vec![2.0, 4.0, 6.0, 8.0]);
        assert_eq!(eval.rows, 2, "second forward should be fully cache-served");
    }

    #[test]
    fn ingest_caches_fresh_generation_rows() {
        let mut infer = double_infer;
        let generation = Arc::new(AtomicU64::new(0));
        let mut cache = InferCache::new(64, generation);
        let mut eval = Evaluator::new(
            &mut infer,
            InferMode::Shared,
            Some(std::slice::from_mut(&mut cache)),
        );
        let mut batch = eval.batch();
        let _ = batch.resolve_or_stage(0, &[1.0, 2.0]);
        let staged = batch.into_staged();
        let rows = eval.ingest(staged, vec![2.0, 4.0], 0.5, 1);
        assert_eq!(rows.row(0), &[2.0, 4.0]);
        assert_eq!((eval.rows, eval.calls), (1, 1));
        let mut batch = eval.batch();
        let Resolve::Resolved(row) = batch.resolve_or_stage(0, &[1.0, 2.0]) else {
            panic!("fresh-generation ingest must populate the cache");
        };
        assert_eq!(row, vec![2.0, 4.0]);
    }

    #[test]
    fn ingest_skips_cache_for_superseded_generation() {
        // weights_updated lands while the batch's rows are in flight on the submitter: the
        // cache syncs (clears) before the reply is ingested; the stale rows must not enter.
        let mut infer = double_infer;
        let generation = Arc::new(AtomicU64::new(0));
        let shared = generation.clone();
        let mut cache = InferCache::new(64, generation);
        let mut eval = Evaluator::new(
            &mut infer,
            InferMode::Shared,
            Some(std::slice::from_mut(&mut cache)),
        );
        let mut batch = eval.batch();
        let _ = batch.resolve_or_stage(0, &[1.0, 2.0]);
        let staged = batch.into_staged();
        shared.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        // the other group's next round syncs the cache before this reply returns
        let empty = eval.batch();
        drop(empty);
        let rows = eval.ingest(staged, vec![2.0, 4.0], 0.5, 1);
        assert_eq!(
            rows.row(0),
            &[2.0, 4.0],
            "the round still consumes the rows"
        );
        let mut batch = eval.batch();
        let Resolve::Staged(_) = batch.resolve_or_stage(0, &[1.0, 2.0]) else {
            panic!("superseded-generation rows must not be served from the cache");
        };
    }

    #[test]
    fn cacheless_evaluator_still_batches() {
        let mut infer = double_infer;
        let mut eval: Evaluator<'_, _> = Evaluator::new(&mut infer, InferMode::Shared, None);
        let mut batch = eval.batch();
        let Resolve::Staged(t0) = batch.resolve_or_stage(0, &[5.0, 6.0]) else {
            panic!()
        };
        let rows = batch.commit();
        assert_eq!(rows.row(t0), &[10.0, 12.0]);
        assert_eq!((eval.rows, eval.calls), (1, 1));
    }
}
