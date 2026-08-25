# Proposal: zero-copy observation marshalling

Status: proposed, not scheduled. Companion to
[engine collection internals](../concepts/engine-collection.md), which documents the
bound this proposal removes.

## Problem

An observation row born in the encoder is copied four times before inference —
once on the worker, three times on the single scheduler thread:

```text
encoder Vec ──▶ RequestSink::obs        (worker: sink append)
            ──▶ RequestQueue::obs       (scheduler: queue append)
            ──▶ fired slice `.to_vec()` (scheduler: fire)
            ──▶ EvalBatch::obs_flat     (scheduler: cache/dedup staging)
            ──▶ NumPy                   (ownership move — already zero-copy)
```

Per-player routing adds one more regrouping copy. For pixel-scale rows (~110 KB)
that is ~330 KB of single-threaded memory traffic per environment step, and it
caps `car_racing` pixel collection near 8,000 steps/s regardless of `n_threads`
— while the render work itself scales near-linearly across workers. Small rows
never notice; large rows turn the scheduler into the pipeline.

## Proposal

Make the callback batch the primary storage and have workers write into it
directly; the scheduler stops touching observation bytes entirely.

```text
TODAY                                    PROPOSED
worker: encode → sink → emit ──mpsc──▶   worker: encode → reserve a row span in
scheduler: copy into queue                       the open pooled buffer (bounded
scheduler: copy out at fire                      CAS reservation) → copy rows in
scheduler: copy into eval batch                  → commit → notify
scheduler: GIL + infer(batch)            scheduler: count reservations; when the
                                                 buffer fills: GIL + infer(batch
                                                 wrapping the buffer, no copy)
```

1. **Pooled write buffers, in two stages.** Each request queue owns a
   fixed-capacity observation buffer (one per inference route: shared, or per
   player). A worker emitting `k` rows reserves a span through the protocol in
   (2), fills it, and commits; the mpsc message carries only metadata:
   `(slot, span, players)` — bytes never ride the channel.

   *Stage 1 (policy-agnostic, no encoder/policy API changes):* the policy encodes
   into worker-local `Vec`s exactly as today, and the worker copies them into its
   span. The copy inventory becomes two worker-side copies and **zero** scheduler
   copies — the surviving copy is parallel and scales with `n_threads`, which is
   the point.

   *Stage 2 (opt-in, per encoder):* add `StateEncoder::encode_into(state, agent,
   dst: &mut [f32])` with a default that delegates to `encode()`, plus a sink
   `push_with(player, dim, |dst| ...)` that reserves the span and hands the
   encoder the destination slice. This removes the last copy for the rows where
   it compounds — high-volume, never-retained search leaf rows. `car_racing`'s
   pixel encoder is already shaped for it (its downsample writes into a
   caller-provided buffer).

   Handing encoders `&mut [f32]` requires the storage to be initialized —
   constructing a reference over uninitialized memory is unsound regardless of
   the element type. Arenas are therefore allocated zeroed
   (`vec![0f32; capacity * dim]`), keeping `MaybeUninit` out of every API.
   Large zeroed allocations are *commonly* satisfied with lazily mapped zero
   pages, making the zeroing near-free in practice — but neither Rust's
   allocator nor the OS guarantees this, so it is an expected cost profile,
   not an invariant; a recycled dirty allocation would require a real zeroing
   pass. Zero initialization also solves only value validity: the unsafe arena
   wrapper must separately guarantee that concurrently writing workers receive
   strictly disjoint `&mut` slices (see the reservation protocol).

   Canonical root rows (`push_root`) keep an owned copy in both stages, by
   design: the training record needs a row that outlives the batch, and the
   arena's storage is destined for Python ownership. One parallel copy is the
   floor for that one row per decision.
2. **The reservation protocol is more than an atomic bump.** Disjoint concurrent
   writes into shared storage force an `unsafe` interior-mutability wrapper, so
   these invariants must be documented beside that wrapper and tested
   independently of the engine:

   - *Bounded reservation.* Spans are claimed by CAS on the write cursor that
     fails at capacity — a plain `fetch_add(k)` can overrun the buffer.
   - *Disjointness.* The wrapper's core aliasing invariant: every handed-out
     span is a strictly disjoint `&mut [f32]`, guaranteed by the reservation
     arithmetic (non-overlapping ranges from a monotone cursor) and tested
     independently — zeroed allocation makes the values valid, but only
     disjointness makes the mutable references sound.
   - *Splitting.* A round emitting more rows than the open buffer's remaining
     space splits across buffers; more than a whole batch, across several fires.
     Spans address `(buffer, range)` pairs, and a search's rows may span fires —
     `absorb` order is preserved by the routing table, not by contiguity.
   - *Commit tracking.* Reservation and completion are distinct states: workers
     finish out of order, so span commits publish with Release ordering that the
     sealing check Acquires before any byte reaches inference.
   - *Atomic close, linearized with aliases.* "Every reserved span committed" is
     not a sufficient sealing predicate: aliases reserve nothing, so an alias
     registered while the scheduler freezes routing would wait forever for a
     prediction row no routing entry delivers. Each buffer therefore carries one
     packed atomic — `(closed, row_cursor, alias_count)` — and reservations AND
     alias registrations are both CASes on it, both failing once closed. Close
     is itself a CAS: performed by the reservation that fills the buffer to
     capacity, or by the scheduler at quiescence. The closing CAS freezes exact
     final counts `(k rows, m aliases)`, and the scheduler seals routing only
     after processing exactly `k` commit and `m` alias messages — every CAS that
     beat the close is guaranteed a routing entry. A loser (CAS after close)
     retries against the next generation: cache first (its result may have
     landed — gated hit), then the new buffer's staged map, else an ordinary
     reservation; termination is structural since generations advance
     monotonically. Cost: the reserve path keeps its single CAS (the word was
     already there), aliases gain one CAS, reconciliation is two integer
     compares per message, and close-race retries reuse their computed hash —
     bookkeeping noise against fire cadence. The alias-vs-close interleaving
     joins the independently tested invariants.
   - *Panic safety.* Reservation is held by an RAII guard that commits or poisons
     its span on drop: a worker panicking after reserving must not leave an
     uncommitted hole that stalls sealing forever. The scheduler's existing
     panic-drain path must stay deadlock-free with an arena in flight.
   - *Lifecycle.* Each buffer moves `Filling → Sealed → InInference → Released`
     (released = moved into NumPy), with transitions owned by the scheduler.

3. **Capacity is fixed; shape is not.** The buffer's capacity is a
   protocol-internal constant — spans must be reserved against a fixed limit —
   and is NOT a callback-shape contract. A short fire truncates the `Vec` to its
   initialized prefix (O(1) for `f32`, capacity slack rides along until Python
   frees it) and moves it into NumPy as `(n, dim)`, exactly today's
   variable-shape default, zero copies. Padding stays the `pad=True` opt-in for
   callers who want `torch.compile`-stable shapes: with it enabled, a short
   fire's tail is already zero (arenas are allocated zeroed for soundness — see
   element 1), so padding costs nothing at seal time. Zero-copy must never
   require padding.
4. **Memory is bounded, checked, and lazy.** Arena memory is
   `pool_depth x capacity x dim x 4` bytes per inference route, and pixel rows
   make that real money: two 1,024-row `car_racing` buffers are ~216 MiB, and
   per-player routing multiplies by the player count. Therefore: routes allocate
   their buffers lazily on first request (per-player mode with `learn_players`
   filtering means some routes never fire at all); size arithmetic is checked at
   construction (`capacity * dim * 4` via `checked_mul`, rejected loudly on
   overflow); pool depth (buffers concurrently alive inside the engine: filling
   plus sealed-awaiting-inference) is bounded and configurable; and reservation
   applies explicit backpressure — a worker that cannot reserve because the pool
   is exhausted parks until a fire completes, which is safe because the
   scheduler can always drain a sealed buffer through inference. Buffers already
   moved into NumPy are the caller's memory, not the pool's.

5. **Ownership per fire.** Each fired buffer is a distinct allocation, moved into
   its NumPy array exactly as today (Python owns the storage; its GC frees it).
   Nothing is recycled when the callback returns: callbacks legitimately retain
   arrays, wrap them in `torch.from_numpy`, or start asynchronous device
   transfers, and reuse on return would mutate retained tensors or race an
   in-flight copy. Workers simply open a freshly allocated arena while the
   callback runs on the previous one — the allocator hands back same-sized blocks
   essentially for free, and the pages are touched by worker writes either way.
   If profiling ever shows allocation itself mattering, storage can be returned
   to a pool through the NumPy owner's deallocator capsule — reuse gated on the
   array's actual death, so retained arrays delay reuse safely instead of being
   corrupted by it. Note the coupled cost: recycled buffers come back dirty, so
   pooling reopens the zeroing question (a real memset per reuse, or
   `MaybeUninit` internally) — a deferred cost the pooling path carries, not a
   free optimization.
6. **The evaluator is replaced, not layered over.** A pooled buffer placed
   underneath the current `Evaluator::forward` would still be staged row-by-row
   into `EvalBatch::obs_flat` — the copy would move, not disappear. The arena IS
   the batch, and this is the only path: there is no fallback evaluator to fork
   behavior on, so caching and deduplication must work inside it.
7. **Caching moves to a read/write split; the cache stays an overlay.** Today the
   evaluator hashes and compacts rows at fire time — inherently O(bytes) on the
   scheduler. Instead:

   - *Worker-read, scheduler-write.* Workers hash the row they just encoded
     (cache-hot, parallel; ~10us for a 110 KB row against a ~350us encode) and
     look the key up in a read-mostly shared cache. The scheduler remains the
     cache's ONLY writer: inserts, eviction, and resizing happen at routing time,
     in message order, so today's single-threaded cache logic (including the
     per-player `Exclusive` slots) survives unchanged. Readers see shards behind
     atomic pointer swaps; evicted rows stay alive through `Arc` clones held by
     in-flight readers; recency updates ride the span metadata workers already
     send ("keys I touched"), applied by the writer in order.
   - *A hit never reserves a span.* The batch contains only genuine misses by
     construction — no dead rows, no fire-time compaction, no wasted callback
     compute.
   - *Hits are gated on fire cadence.* A hit resolves at the worker (the row is
     already in hand) but the slot parks until the next full batch has fired;
     the scheduler releases all gated slots after each fire. Ungated hits would
     let cache-hot slots free-run: they would advance many decisions per fire,
     starve rare misses of batch-mates, and let the record floor close on a
     batch that overrepresents cache-hot (i.e. recurring) states. Gating
     restores today's pacing — hits resolve at fire cadence — so the sampling
     mix does not shift, and unlike counting hits toward the firing threshold it
     never fires a partial batch: the GPU always sees `batch_size` rows in
     steady state. The latency cost is absorbed by slot count, not throughput —
     see the sizing rule below.
   - *In-flight duplicates alias.* A per-buffer staged map (fixed capacity,
     insert-or-read-only, CAS on empty slots, cleared by the scheduler at each
     fire) lets a second requester of an already-reserved key record an alias
     ticket instead of reserving; the scheduler routes the one prediction row to
     every claimant.
   - *Invalidation is validated at the gate.* `weights_updated()` clears the
     cache at a safe boundary, but a worker can take an old shard snapshot,
     resolve a hit from it, and hold that row through a retained `Arc` past the
     boundary — no worker-side generation check can close this (the window
     between check and use is unbounded under preemption). The gate closes it
     structurally: every hit already waits for a scheduler release, and the
     scheduler is also the single thread that applies invalidation, so hit
     notifications carry their generation-at-lookup and the scheduler validates
     it at release. Current generation → deliver. Stale → demote: the scheduler
     spawns the slot's resume task with a recompute verdict, and a worker
     re-emits the retained rows through the completely ordinary reservation path
     into the fresh open buffer — indistinguishable from a first-time miss. Two
     supporting rules: a gated slot's search state keeps its encoded rows until
     the release confirms the hit (so demotion never re-encodes), and each fired
     batch carries its fire-time generation so routing skips cache INSERTS from
     pre-boundary fires (delivering their values to waiting slots is fine — that
     is the documented in-flight-work semantics — but they must not seed the
     post-boundary cache). Net: no post-boundary decision ever consumes a
     pre-boundary cached value, at the cost of a few demoted hits in the one
     round straddling the update.
   - *Two planes, never reconciled.* Cache reads and buffer reservations are
     separate communication paths with one monotone invariant: a reserved span
     is inference work, unconditionally. If a key completes and is inserted
     after a worker's miss decision (the lookup-to-fire staleness window), that
     worker's row rides to inference anyway — one redundant row, and the
     routing-time insert is idempotent because a deterministic callback returns
     identical values for identical bytes. The scheduler never re-checks the
     cache for buffered rows; correctness never depends on cache state. The
     staged map covers duplicates within a buffer, the two-plane rule covers
     staleness across fires, and both failure costs are compute, not
     correctness.
8. **The quiescence flush guarantees liveness.** Gating creates one stall the
   ungated design could not have: if every slot is parked — hit-gated or
   contributing to a miss buffer below capacity — no fire can happen and the
   gated hits wait forever. The scheduler already has the exact observation
   point: its idle path blocks on the message channel, and blocking with
   `in_flight == 0` and no admissible slot IS the stall. The guard there
   triggers the flush: close each open buffer (the same atomic close as a
   capacity fill), seal at current fill (truncate-to-prefix; race-free because
   zero in-flight tasks implies every reserved span is committed AND every
   registered alias message already processed — the `(k, m)` reconciliation is
   trivially satisfied at true quiescence), fire non-empty buffers,
   then release ALL gated hits — including on routes whose buffer is empty,
   because the gate is a pacing device and at quiescence there is no cadence
   left to pace against. Released slots respawn into the fresh buffer and the
   system exits quiescence. Detection is an O(1) counter check at a point the
   loop already visits; determinism at `n_threads=1` holds because quiescence
   occurs at structurally fixed points.

   Sizing rule: full batches in steady state need roughly
   `n_games >= 2 x batch_size / (1 - hit_rate)` — one batch's worth of misses
   accumulating while the previous batch is in the callback, with hit-gated
   slots not contributing. A well-sized run only flushes at window boundaries,
   so a quiescence-flush counter in telemetry directly exposes undersized
   configurations: if it ticks in steady state, the config violates the sizing
   rule and the operator sees small batches coming.
9. **Tail bootstraps become worker tasks.** Today `tail_requests` encodes the
   truncation/fragment bootstrap observations on the scheduler thread — for
   pixel games that is a full render each, serialized. Under the arena, a cut
   spawns a tail task onto the worker pool that encodes, reserves, and commits
   like any other emission, with a tail marker routing its prediction rows to
   the existing `AwaitingTail` bookkeeping.
10. **The scheduler's remaining job** is control flow only: reservation accounting,
   firing, key lookups' insert/evict side, routing the (small, f64) prediction
   rows back to blocked slots, floor and record bookkeeping. All of it is
   O(rows), none of it is O(bytes).

## What must not change

- **`n_threads=1` reproducibility.** With one worker, reservation order equals
  emission order; batch contents and firing points must remain byte-identical to
  today's. The multi-worker engine is already declared nondeterministic in
  composition, and span reservation order falls under that existing contract.
- **Root-row reuse.** Retained canonical rows (`push_root`) currently move an
  owned `Vec` through the blocked slot. Workers writing into pooled buffers write
  the same bytes twice (buffer + retained row) unless the record path learns to
  reference the span; keeping the owned copy is acceptable for v1 — it is worker-
  side and parallel, unlike the copies this proposal removes.
- **Backpressure and floor semantics**, tail bootstraps riding the same queues,
  and per-player routing all keep their observable behavior.

## Expected effect

Removing the O(bytes) scheduler work leaves workers bounded by encoding itself.
Pure render scaling measured 3.7x at 4 threads and 7.1x at 8 on the reference
machine; pixel collection should approach that curve from its current flat
~8,000 steps/s — a projected 2-3x at 8-10 threads, more on wider machines. The
vec-encoder path (already ~260,000 steps/s, scheduling-bound not copy-bound)
should be unchanged, which doubles as the no-regression control.

## Validation

- Determinism at `n_threads=1` under the new semantics: fixed seed produces
  byte-identical batches run-to-run (equality against the OLD scheduler is not a
  goal — hit gating and fire cadence legitimately reorder collection).
- Telemetry assertions: per-slot record counts and hit rates exposed; the
  quiescence-flush counter stays at zero in a correctly sized steady-state run.
- The existing scheduler test suite (firing thresholds, per-player routing,
  fragment cuts, panic drains) unchanged.
- Throughput gate: pixel collection at 8 threads must beat the current plateau by
  a stated margin on the benchmark machine before the old path is removed.
