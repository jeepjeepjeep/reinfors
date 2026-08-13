//! Batched network evaluation, caching, and throughput accounting. Every inference consumer must
//! pass through this service; the earlier peer-parameter design let call sites bypass caching and
//! telemetry accidentally.

use crate::rollout::infer_cache::{InferCache, ShardedInferCache};

/// Whether rows route to one shared network or one network per player.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum InferMode {
    Shared,
    PerPlayer,
}

/// How an evaluator reaches its cache: exclusively owned, shared sharded slots, or none.
pub enum CacheAccess<'a> {
    None,
    /// Within-batch key deduplication with no store: identical rows in one batch resolve
    /// to one inference row, and nothing survives the batch.
    BatchDedup,
    Exclusive(&'a mut [InferCache]),
    Shared(&'a [ShardedInferCache]),
}

pub struct Evaluator<'a, F> {
    infer: &'a mut F,
    mode: InferMode,
    cache: CacheAccess<'a>,
    // The sharded cache's own counters are global across accessors; per-evaluator
    // telemetry must be local or folding double-counts.
    shared_lookups: usize,
    shared_hits: usize,
    pad_rows_to: Option<usize>,
    pub rows: usize,
    pub padded_rows: usize,
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
    // per-player slots advance generations independently, so staging generations are per row
    row_generations: Vec<u64>,
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
            cache: match caches {
                Some(c) => CacheAccess::Exclusive(c),
                None => CacheAccess::None,
            },
            shared_lookups: 0,
            shared_hits: 0,
            pad_rows_to: None,
            rows: 0,
            padded_rows: 0,
            calls: 0,
            seconds: 0.0,
        }
    }

    /// Grouped-collection constructor: shared sharded cache slots (one per routing slot).
    /// Batch-deduplicating cacheless evaluator (the `choose` composition).
    pub fn with_batch_dedup(infer: &'a mut F, mode: InferMode) -> Self {
        Evaluator {
            infer,
            mode,
            cache: CacheAccess::BatchDedup,
            shared_lookups: 0,
            shared_hits: 0,
            pad_rows_to: None,
            rows: 0,
            padded_rows: 0,
            calls: 0,
            seconds: 0.0,
        }
    }

    pub fn with_shared_cache(
        infer: &'a mut F,
        mode: InferMode,
        slots: &'a [ShardedInferCache],
    ) -> Self {
        Evaluator {
            infer,
            mode,
            cache: CacheAccess::Shared(slots),
            shared_lookups: 0,
            shared_hits: 0,
            pad_rows_to: None,
            rows: 0,
            padded_rows: 0,
            calls: 0,
            seconds: 0.0,
        }
    }

    /// Fix shared-mode call shapes at exactly `pad` rows: short batches are padded
    /// with zero rows (outputs discarded), oversized batches split into `pad`-row
    /// chunks. Assumes row-independent callback outputs (evaluation-mode networks).
    pub fn with_pad_rows_to(mut self, pad: Option<usize>) -> Self {
        assert!(
            pad.is_none() || self.mode == InferMode::Shared,
            "pad_rows_to supports shared-mode inference only"
        );
        self.pad_rows_to = pad;
        self
    }

    fn slot_index(&self, player: usize) -> usize {
        match self.mode {
            InferMode::Shared => 0,
            InferMode::PerPlayer => player,
        }
    }

    fn cache_slot(&mut self, player: usize) -> Option<&mut InferCache> {
        let idx = self.slot_index(player);
        match &mut self.cache {
            CacheAccess::Exclusive(c) => Some(&mut c[idx]),
            _ => None,
        }
    }

    fn row_key(&self, player: usize, obs: &[f32]) -> u128 {
        match self.mode {
            InferMode::Shared => InferCache::key(obs),
            InferMode::PerPlayer => InferCache::key_for_player(player, obs),
        }
    }

    pub fn batch<'e>(&'e mut self) -> EvalBatch<'e, 'a, F> {
        // Shared shards sync lazily per access; locking all per round would serialize.
        if let CacheAccess::Exclusive(caches) = &mut self.cache {
            for cache in caches.iter_mut() {
                cache.sync_generation();
            }
        }
        EvalBatch {
            eval: self,
            obs_flat: Vec::new(),
            keys: Vec::new(),
            players: Vec::new(),
            staged: std::collections::HashMap::new(),
            n: 0,
            dim: 0,
            row_generations: Vec::new(),
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

    pub fn cache_lookups(&self) -> usize {
        match &self.cache {
            CacheAccess::Exclusive(c) => c.iter().map(|x| x.lookups).sum(),
            CacheAccess::Shared(_) => self.shared_lookups,
            CacheAccess::None | CacheAccess::BatchDedup => 0,
        }
    }

    pub fn cache_hits(&self) -> usize {
        match &self.cache {
            CacheAccess::Exclusive(c) => c.iter().map(|x| x.hits).sum(),
            CacheAccess::Shared(_) => self.shared_hits,
            CacheAccess::None | CacheAccess::BatchDedup => 0,
        }
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

/// Forward `n` staged rows through shared-mode `infer` in calls of EXACTLY `pad` rows:
/// the final short chunk is zero-padded (pad outputs discarded), earlier chunks are
/// full. Returns rows in ticket order, the call count, and the pad rows forwarded.
fn run_infer_fixed_shape<F>(
    infer: &mut F,
    obs_flat: Vec<f32>,
    n: usize,
    dim: usize,
    pad: usize,
) -> (Vec<f64>, usize, usize)
where
    F: FnMut(usize, Vec<f32>, usize) -> Vec<f64>,
{
    let chunks = n.div_ceil(pad);
    if chunks == 1 {
        let mut obs = obs_flat;
        obs.resize(pad * dim, 0.0);
        let mut rows = infer(0, obs, pad);
        let stride = rows.len() / pad;
        rows.truncate(n * stride);
        return (rows, 1, pad - n);
    }
    let mut out = Vec::new();
    let mut stride: Option<usize> = None;
    for c in 0..chunks {
        let lo = c * pad;
        let hi = ((c + 1) * pad).min(n);
        let mut obs = obs_flat[lo * dim..hi * dim].to_vec();
        obs.resize(pad * dim, 0.0);
        let rows = infer(0, obs, pad);
        let s = rows.len() / pad;
        if let Some(expected) = stride {
            assert_eq!(s, expected, "padded chunk row width changed between chunks");
        }
        stride = Some(s);
        out.extend_from_slice(&rows[..(hi - lo) * s]);
    }
    (out, chunks, chunks * pad - n)
}

impl<'e, 'a, F> EvalBatch<'e, 'a, F>
where
    F: FnMut(usize, Vec<f32>, usize) -> Vec<f64>,
{
    pub fn resolve_or_stage(&mut self, player: usize, obs: &[f32]) -> Resolve {
        if !matches!(self.eval.cache, CacheAccess::None) {
            let key = self.eval.row_key(player, obs);
            let mut staging_generation = None;
            let hit = match &mut self.eval.cache {
                CacheAccess::BatchDedup => None,
                CacheAccess::Exclusive(_) => self
                    .eval
                    .cache_slot(player)
                    .expect("caches present")
                    .lookup(key),
                CacheAccess::Shared(slots) => {
                    let idx = match self.eval.mode {
                        InferMode::Shared => 0,
                        InferMode::PerPlayer => player,
                    };
                    self.eval.shared_lookups += 1;
                    let hit = slots[idx].lookup(key);
                    if hit.is_some() {
                        self.eval.shared_hits += 1;
                    }
                    staging_generation = Some(slots[idx].generation());
                    hit
                }
                CacheAccess::None => unreachable!(),
            };
            if let Some(row) = hit {
                return Resolve::Resolved(row);
            }
            if let Some(&ticket) = self.staged.get(&key) {
                return Resolve::Staged(ticket);
            }
            self.staged.insert(key, self.n);
            self.keys.push(key);
            if let Some(generation) = staging_generation {
                self.row_generations.push(generation);
            }
        }
        self.dim = obs.len();
        if self.n == 0 {
            if let Some(pad) = self.eval.pad_rows_to {
                self.obs_flat.reserve(pad * obs.len());
            }
        }
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
        let n_real = self.n;
        let (out, calls, padded) = match self.eval.pad_rows_to {
            Some(pad) => {
                run_infer_fixed_shape(self.eval.infer, self.obs_flat, n_real, self.dim, pad)
            }
            None => {
                let (out, calls) = run_infer(
                    self.eval.infer,
                    self.eval.mode,
                    &self.players,
                    self.obs_flat,
                    n_real,
                    self.dim,
                );
                (out, calls, 0)
            }
        };
        self.eval.seconds += t.elapsed().as_secs_f64();
        self.eval.calls += calls;
        self.eval.rows += n_real;
        self.eval.padded_rows += padded;
        let stride = out.len() / n_real;
        match &mut self.eval.cache {
            CacheAccess::Exclusive(_) => {
                for i in 0..self.keys.len() {
                    let (key, player) = (self.keys[i], self.players[i]);
                    let row = out[i * stride..(i + 1) * stride].to_vec();
                    self.eval
                        .cache_slot(player)
                        .expect("caches present")
                        .insert(key, &row);
                }
            }
            CacheAccess::Shared(slots) => {
                for i in 0..self.keys.len() {
                    let (key, player) = (self.keys[i], self.players[i]);
                    let idx = match self.eval.mode {
                        InferMode::Shared => 0,
                        InferMode::PerPlayer => player,
                    };
                    slots[idx].insert(
                        key,
                        &out[i * stride..(i + 1) * stride],
                        self.row_generations[i],
                    );
                }
            }
            CacheAccess::None | CacheAccess::BatchDedup => {}
        }
        CommittedRows { out, stride }
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
    fn padding_fixes_call_shape_and_discards_pad_outputs() {
        let seen_n = Arc::new(AtomicU64::new(0));
        let seen = seen_n.clone();
        let mut infer = move |_p: usize, obs: Vec<f32>, n: usize| -> Vec<f64> {
            seen.store(n as u64, std::sync::atomic::Ordering::Relaxed);
            assert_eq!(obs.len(), n * 2);
            (0..n).flat_map(|i| [f64::from(obs[i * 2]), 0.5]).collect()
        };
        let mut eval =
            Evaluator::new(&mut infer, InferMode::Shared, None).with_pad_rows_to(Some(8));
        let mut batch = eval.batch();
        for i in 0..3 {
            batch.resolve_or_stage(0, &[i as f32 + 1.0, 1.0]);
        }
        let rows = batch.commit();
        assert_eq!(seen_n.load(std::sync::atomic::Ordering::Relaxed), 8);
        for i in 0..3 {
            assert_eq!(rows.row(i), &[f64::from(i as u32) + 1.0, 0.5]);
        }
        assert_eq!(eval.rows, 3);
        assert_eq!(eval.padded_rows, 5);
    }

    #[test]
    fn oversize_batches_chunk_to_the_exact_shape() {
        let sizes = Arc::new(std::sync::Mutex::new(Vec::new()));
        let seen = sizes.clone();
        let mut infer = move |_p: usize, obs: Vec<f32>, n: usize| -> Vec<f64> {
            seen.lock().unwrap().push(n);
            (0..n).flat_map(|i| [f64::from(obs[i * 2]), 0.5]).collect()
        };
        let mut eval =
            Evaluator::new(&mut infer, InferMode::Shared, None).with_pad_rows_to(Some(2));
        let mut batch = eval.batch();
        for i in 0..5 {
            batch.resolve_or_stage(0, &[i as f32 + 1.0, 1.0]);
        }
        let rows = batch.commit();
        // 5 rows at pad 2: three calls of exactly 2 rows, one zero pad row in the last
        assert_eq!(*sizes.lock().unwrap(), vec![2, 2, 2]);
        assert_eq!(eval.calls, 3);
        assert_eq!(eval.rows, 5);
        assert_eq!(eval.padded_rows, 1);
        // ticket order preserved across chunk boundaries
        for i in 0..5 {
            assert_eq!(rows.row(i), &[f64::from(i as u32) + 1.0, 0.5]);
        }
    }

    #[test]
    #[should_panic(expected = "shared-mode inference only")]
    fn padding_rejects_per_player_mode_at_configuration() {
        let mut infer = |_p: usize, _obs: Vec<f32>, n: usize| -> Vec<f64> { vec![0.5; n * 2] };
        let _ = Evaluator::new(&mut infer, InferMode::PerPlayer, None).with_pad_rows_to(Some(4));
    }

    #[test]
    fn pad_rows_never_reach_the_cache() {
        // a real observation that HAPPENS to equal the zero padding must still miss
        let mut infer = |_p: usize, _obs: Vec<f32>, n: usize| -> Vec<f64> { vec![0.5; n * 2] };
        let generation = Arc::new(AtomicU64::new(0));
        let mut caches = vec![InferCache::new(64, generation)];
        let mut eval = Evaluator::new(&mut infer, InferMode::Shared, Some(&mut caches))
            .with_pad_rows_to(Some(4));
        let mut batch = eval.batch();
        batch.resolve_or_stage(0, &[7.0, 7.0]);
        batch.commit(); // padded with three [0.0, 0.0] rows
        let mut batch = eval.batch();
        assert!(matches!(
            batch.resolve_or_stage(0, &[7.0, 7.0]),
            Resolve::Resolved(_)
        ));
        assert!(matches!(
            batch.resolve_or_stage(0, &[0.0, 0.0]),
            Resolve::Staged(_)
        ));
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
    fn batch_dedup_dedupes_within_a_batch_and_stores_nothing() {
        let mut infer = double_infer;
        let mut eval = Evaluator::with_batch_dedup(&mut infer, InferMode::Shared);
        let mut batch = eval.batch();
        let a = batch.resolve_or_stage(0, &[1.0, 2.0]);
        let b = batch.resolve_or_stage(0, &[1.0, 2.0]);
        let (Resolve::Staged(ta), Resolve::Staged(tb)) = (a, b) else {
            panic!("both stage");
        };
        assert_eq!(ta, tb, "identical rows must share one ticket");
        let rows = batch.commit();
        assert_eq!(rows.row(ta), &[2.0, 4.0]);
        assert_eq!(eval.rows, 1, "one inference row for the duplicate pair");

        let mut batch = eval.batch();
        assert!(
            matches!(batch.resolve_or_stage(0, &[1.0, 2.0]), Resolve::Staged(_)),
            "nothing survives the batch: the next round re-infers"
        );
        drop(batch);
        assert_eq!(eval.cache_lookups(), 0);
    }

    #[test]
    fn shared_cache_evaluator_stages_dedupes_and_serves() {
        use crate::rollout::infer_cache::ShardedInferCache;
        let mut infer = double_infer;
        let generation = Arc::new(AtomicU64::new(0));
        let slots = [ShardedInferCache::new(1024, 16, generation)];
        let mut eval = Evaluator::with_shared_cache(&mut infer, InferMode::Shared, &slots);

        let mut batch = eval.batch();
        let a = batch.resolve_or_stage(0, &[1.0, 2.0]);
        let b = batch.resolve_or_stage(0, &[1.0, 2.0]);
        let (Resolve::Staged(ta), Resolve::Staged(tb)) = (a, b) else {
            panic!("cold shared cache must stage");
        };
        assert_eq!(
            ta, tb,
            "within-batch dedup applies over the shared cache too"
        );
        let rows = batch.commit();
        assert_eq!(rows.row(ta), &[2.0, 4.0]);

        let mut batch = eval.batch();
        let Resolve::Resolved(row) = batch.resolve_or_stage(0, &[1.0, 2.0]) else {
            panic!("warm shared cache must resolve");
        };
        assert_eq!(row, vec![2.0, 4.0]);
        assert_eq!(eval.cache_hits(), 1);
    }

    #[test]
    fn two_evaluators_report_their_own_shared_traffic() {
        use crate::rollout::infer_cache::ShardedInferCache;
        let generation = Arc::new(AtomicU64::new(0));
        let slots = [ShardedInferCache::new(1024, 16, generation)];

        let mut infer_a = double_infer;
        let mut a = Evaluator::with_shared_cache(&mut infer_a, InferMode::Shared, &slots);
        let mut batch = a.batch();
        let _ = batch.resolve_or_stage(0, &[1.0, 2.0]);
        batch.commit();
        let mut batch = a.batch();
        let _ = batch.resolve_or_stage(0, &[1.0, 2.0]); // hit
        batch.commit();

        let mut infer_b = double_infer;
        let mut b = Evaluator::with_shared_cache(&mut infer_b, InferMode::Shared, &slots);
        let mut batch = b.batch();
        let _ = batch.resolve_or_stage(0, &[1.0, 2.0]); // hit, b's only lookup
        batch.commit();

        assert_eq!(
            (a.cache_lookups(), a.cache_hits()),
            (2, 1),
            "a's own traffic only"
        );
        assert_eq!(
            (b.cache_lookups(), b.cache_hits()),
            (1, 1),
            "b's own traffic only"
        );
        assert_eq!(
            slots[0].lookups(),
            3,
            "the cache itself carries the global totals"
        );
    }

    #[test]
    fn per_player_slots_stage_under_their_own_generations() {
        use crate::rollout::infer_cache::ShardedInferCache;
        let gen0 = Arc::new(AtomicU64::new(0));
        let gen1 = Arc::new(AtomicU64::new(0));
        let slots = [
            ShardedInferCache::new(1024, 16, gen0),
            ShardedInferCache::new(1024, 16, gen1.clone()),
        ];
        let mut infer = double_infer;
        let mut eval = Evaluator::with_shared_cache(&mut infer, InferMode::PerPlayer, &slots);

        let mut batch = eval.batch();
        let _ = batch.resolve_or_stage(0, &[1.0, 2.0]);
        let _ = batch.resolve_or_stage(1, &[3.0, 4.0]);
        // player 1's slot advances while the batch is in flight; player 0's does not
        gen1.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        batch.commit();

        let mut batch = eval.batch();
        assert!(
            matches!(batch.resolve_or_stage(0, &[1.0, 2.0]), Resolve::Resolved(_)),
            "player 0's row staged under its own (unchanged) generation must be cached"
        );
        drop(batch);
        let mut batch = eval.batch();
        assert!(
            matches!(batch.resolve_or_stage(1, &[3.0, 4.0]), Resolve::Staged(_)),
            "player 1's row staged under a superseded generation must be rejected"
        );
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
