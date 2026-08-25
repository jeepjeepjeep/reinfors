# Engine collection internals

How one `Engine.collect(...)` window executes across threads, and where its time
goes. [Architecture](architecture.md) shows the component seams; this page shows
the runtime: the thread roles, the message flow, and the serialization point that
bounds pixel-scale throughput today.

## Thread roles

```text
CALLER-OWNED PYTHON
┌──────────────────────────────────────────────────────────────────────────┐
│ Training loop calls Engine.collect(...) — the calling thread enters Rust │
│ and becomes the SCHEDULER for the whole window (GIL released while it    │
│ waits; reacquired only to run the infer callback)                        │
└────────────────────────────────────┬─────────────────────────────────────┘
═══════════════════════ RUST / PYTHON BOUNDARY ═════════════════════════════
                                     ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ SCHEDULER — one thread                                                   │
│ admission · request queues (shared or per-player) · batch firing ·       │
│ row routing · floor accounting · record collection · respawns            │
└───────┬──────────────────────────────────────────────────▲───────────────┘
        │ Work::Begin / Work::Resume                       │ TaskOut::Emitted /
        │ (spawn onto the pool)                            │ TaskOut::Completed
        ▼                                                  │ (mpsc channel)
┌──────────────────────────────────────────────────────────┴───────────────┐
│ WORKER POOL — `n_threads` rayon threads                                  │
│ a task locks ONE of the `n_games` slots (Mutex<SlotCtx>) and runs the    │
│ policy's search rounds, game advance, and record assembly for that slot  │
└──────────────────────────────────────────────────────────────────────────┘
```

`n_games` is the number of episode slots, not the parallelism: slots hold state,
threads do work, and the scheduler keeps every idle thread fed with whichever
slot is ready next.

## One decision, step by step

```text
 WORKER (slot gi)                          SCHEDULER
 ────────────────                          ─────────
 1. policy round: encode observation(s),
    emit request rows into the sink
    (model-free policies mark their row
    canonical — `push_root` — so step 6
    reuses it instead of re-encoding)
 2. TaskOut::Emitted ─────────────────────▶ 3. COPY B: append each row to the
    slot parks as Blocked                      request queue's flat buffer;
    (search + retained rows carried)           count it toward the batch
    (COPY A already happened in step 1:
    each row into the sink's flat buffer)
                                            4. queue reaches `batch_size`:
                                               COPY C: slice the fired rows out
                                               of the queue; COPY D: stage them
                                               into the evaluator's batch (the
                                               cache/dedup layer); move that
                                               buffer into NumPy (no copy),
                                               take the GIL, call the caller's
                                               infer callback, route prediction
                                               rows back to each blocked slot
 5. Work::Resume ◀───────────────────────── freed slots respawn on the pool
    absorb predictions; either another
    round (tree search: many leaf rows
    per action) or the search is Done
 6. finish: select actions, advance the
    game, build learner records (the
    retained row IS the record's obs),
    flush finished episodes
 7. TaskOut::Completed ───────────────────▶ 8. apply records/stats in message
                                               order; admit the slot's next
                                               decision; stop at the floor
```

Rounds 1–5 repeat per inference round: a model-free policy uses exactly one, a
tree search uses as many as its expansion schedule needs, pooling every slot's
leaf rows into the same shared batches.

## Where the time goes

Steps 3 and 4 are the serialization point: after the worker-side sink copy (A),
**every observation row is copied three more times on the single scheduler
thread** — into the request queue (B), out of the queue at fire (C), and into
the evaluator's cache/dedup batch (D) — while the workers that produced the rows
sit blocked. The final hand-off to NumPy is an ownership move, not a copy.
Per-player inference routing adds one more regrouping copy. Everything else
scales with `n_threads`.

Whether this matters is a function of row size:

| observation row | scheduler traffic | measured ceiling (Apple M1 Max)          |
| --- | --- | --- |
| `car_racing` pixels, ~110 KB | ~3 × 110 KB per step | ~8,000 steps/s, flat from ~8 threads |
| `car_racing` vec, 84 B | negligible | ~260,000 steps/s (scheduling-bound, not copy-bound) |

The pure render path scales near-linearly on the same machine (3.7x at 4
threads, 7.1x at 8), so the plateau is this marshalling, not the simulation.
Small-row games never notice it; pixel-scale games hit it once a few workers can
out-produce one thread's memory bandwidth. Reducing it is the current known
scaling limit of the scheduler design; the proposed fix is documented in
[zero-copy observation marshalling](../development/scheduler-zero-copy.md).
