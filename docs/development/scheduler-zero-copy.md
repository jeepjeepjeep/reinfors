# Proposal: zero-copy observation marshalling

Status: proposed, not scheduled. Companion to
[engine collection internals](../concepts/engine-collection.md), which documents the
bound this proposal removes.

## Problem

Every observation row a worker produces crosses the single scheduler thread twice:
appended into the request queue's flat buffer when the slot's rows are emitted, then
assembled into the NumPy array handed to the infer callback. For pixel-scale rows
(~110 KB) that is ~220 KB of single-threaded memory traffic per environment step,
and it caps `car_racing` pixel collection near 8,000 steps/s regardless of
`n_threads` — while the render work itself scales near-linearly across workers.
Small rows never notice; large rows turn the scheduler into the pipeline.

## Proposal

Make the callback batch the primary storage and have workers write into it
directly; the scheduler stops touching observation bytes entirely.

```text
TODAY                                    PROPOSED
worker: encode → emit rows ──mpsc──▶     worker: encode → reserve a row span in
scheduler: copy into queue (A)                   the open pooled buffer (atomic
scheduler: copy into NumPy (B)                   offset bump) → write rows in
scheduler: GIL + infer(batch)                    place → notify
                                         scheduler: count reservations; when the
                                                 buffer fills: GIL + infer(batch
                                                 wrapping the buffer, no copy)
```

1. **Pooled write buffers.** Each request queue owns a preallocated, fixed-capacity
   observation buffer (one per inference route: shared, or per player). A worker
   emitting `k` rows reserves a contiguous span with an atomic offset add and
   encodes/copies its rows straight into the span. The mpsc message carries only
   metadata: `(slot, span, players)` — bytes never ride the channel.
2. **Fixed batch shape.** The buffer's capacity IS the callback batch size, padded
   on the final short fire of a window (`pad_rows_to` semantics). Stable shape is
   what callers need to `torch.compile` against; the array object itself is fresh
   per fire.
3. **Ownership per fire.** Each fired buffer is a distinct allocation, moved into
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
4. **The scheduler's remaining job** is control flow only: reservation accounting,
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
