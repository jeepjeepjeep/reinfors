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
   - *Splitting.* A round emitting more rows than the open buffer's remaining
     space splits across buffers; more than a whole batch, across several fires.
     Spans address `(buffer, range)` pairs, and a search's rows may span fires —
     `absorb` order is preserved by the routing table, not by contiguity.
   - *Commit tracking.* Reservation and completion are distinct states: workers
     finish out of order, so a buffer seals on "every reserved span committed"
     (a committed count, not the cursor), and span commits publish with Release
     ordering that the sealing check Acquires before any byte reaches inference.
   - *Panic safety.* Reservation is held by an RAII guard that commits or poisons
     its span on drop: a worker panicking after reserving must not leave an
     uncommitted hole that stalls sealing forever. The scheduler's existing
     panic-drain path must stay deadlock-free with an arena in flight.
   - *Lifecycle.* Each buffer moves `Filling → Sealed → InInference → Released`
     (released = moved into NumPy), with transitions owned by the scheduler.

3. **Fixed batch shape.** The buffer's capacity IS the callback batch size, padded
   on the final short fire of a window (`pad_rows_to` semantics). Stable shape is
   what callers need to `torch.compile` against; the array object itself is fresh
   per fire.
4. **Ownership per fire.** Each fired buffer is a distinct allocation, moved into
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
   corrupted by it.
5. **The evaluator is replaced for arena batches, not layered over.** A pooled
   buffer placed underneath the current `Evaluator::forward` would still be
   staged row-by-row into `EvalBatch::obs_flat` — the copy would move, not
   disappear. Arena batches need a specialized forward path where the arena IS
   the batch. The infer cache and batch dedup are the design risk here: they key
   and compact rows during staging. Keys can be hashed from arena spans in place;
   dedup either works through ticket indirection into the arena or is disabled in
   arena mode for v1 (it exists for search workloads with repeated leaves, which
   are not the large-row workloads this proposal targets).
6. **The scheduler's remaining job** is control flow only: reservation accounting,
   firing, routing the (small, f64) prediction rows back to blocked slots, floor
   and record bookkeeping. All of it is O(rows), none of it is O(bytes).

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

- A/B byte-equality of collected batches at `n_threads=1` against the current
  scheduler on a cheap game and on `car_racing` pixels.
- The existing scheduler test suite (firing thresholds, per-player routing,
  fragment cuts, panic drains) unchanged.
- Throughput gate: pixel collection at 8 threads must beat the current plateau by
  a stated margin on the benchmark machine before the old path is removed.
