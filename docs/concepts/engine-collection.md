# Engine collection internals

How one `Engine.collect(...)` window executes across threads, and where its time
goes. [Architecture](architecture.md) shows the component seams; this page shows
the runtime: the thread roles, the zero-copy observation path, and the
mechanisms that keep it correct and live.

## Definitions

- **Slot** — one of the `n_games` episode holders; it stores game and search
  state and is locked by whichever worker task is advancing it.
- **Round** — one batch of evaluation requests a search emits before blocking on
  their predictions; model-free policies use one per decision, tree searches many.
- **Route** — one inference destination: a single queue for a shared callback, or
  one per player for per-player callbacks.
- **Arena** — a route's fixed-capacity shared write buffer; workers copy or
  encode observation rows directly into it.
- **Span** — a worker's exclusive reservation of contiguous arena rows, filled
  without locks and then committed (or poisoned by a panic).
- **Seal** — converting a closed, fully committed arena into an owned `Vec<f32>`
  without copying, ready to hand to the callback.
- **Fire** — sending one sealed batch through the inference callback and routing
  its prediction rows back to the blocked slots.
- **Root row** — the canonical current-state observation a policy marks for
  reuse, so the training record carries it without re-encoding.
- **Gated ticket** — a cache hit parked until its route's next fire (or
  quiescence), so hits obey the same weights-generation discipline as real
  inference.
- **Alias ticket** — a claim that a row identical to one already in flight will
  reuse that row's prediction instead of reserving its own.
- **Generation** — the counter bumped by `weights_updated()`; cached values from
  an older generation are never served.
- **Demotion** — re-emitting a gated hit as an ordinary arena row because its
  generation went stale before release.
- **View** — the immutable cache snapshot a worker task reads, pinned when the
  task spawns and replaced only by scheduler publication.
- **Liveness ladder** — the escalation from capacity fires to stall-triggered
  partial fires and releases that keeps a slow or hit-only route from waiting
  forever on a batch that will never fill.
- **Quiescence** — the state with no worker tasks in flight, at which every open
  buffer is closed, fired, and released.
- **Floor** — the `n_records` target that ends the collection window once
  reached.

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
│ SCHEDULER — one thread, METADATA ONLY (it never touches row bytes)       │
│ span accounting · batch firing · row routing · cache publication ·       │
│ liveness ladder · floor accounting · record collection · respawns        │
└───────┬────────────────────────────▲─────────────────────▲───────────────┘
        │ Work::Begin / Work::Resume │ TaskOut::Emitted:   │ shared arenas:
        │ (spawn onto the pool)      │ span/hit/alias      │ one open write
        ▼                            │ METADATA (mpsc)     ▼ buffer per route
┌────────────────────────────────────┴─────────────────────────────────────┐
│ WORKER POOL — `n_threads` rayon threads                                  │
│ a task locks ONE of the `n_games` slots (Mutex<SlotCtx>), runs the       │
│ policy's search rounds, and writes each observation row it emits         │
│ DIRECTLY into the route's shared arena                                   │
└──────────────────────────────────────────────────────────────────────────┘
```

`n_games` is the number of episode slots, not the parallelism: slots hold state,
threads do work, and the scheduler keeps every idle thread fed with whichever
slot is ready next. A route is one inference destination — one queue for a
shared callback, one per player for per-player callbacks — and each route owns
a chain of arenas: fixed-capacity, contiguous `f32` write buffers sized by
`batch_size`.

## One decision, step by step

```text
 WORKER (slot gi)                          SCHEDULER
 ────────────────                          ─────────
 1. policy round: for each request row,
    in order —
      cache HIT?    park a gated ticket
                    (value known; no row)
      key already   claim an ALIAS ticket
      in flight?    (the twin's prediction
                    will be reused)
      otherwise     reserve a span in the
                    route's open arena and
                    encode INTO it (root
                    rows: one copy; state
                    rows: zero copies)
 2. TaskOut::Emitted ─────────────────────▶ 3. account spans/hits/aliases; the
    span positions, hit tickets, alias         message carries NO row bytes. A
    claims; slot parks as Blocked              reservation that filled the
                                               arena already closed it in the
                                               same atomic step
                                            4. closed arena fully accounted →
                                               FIRE: seal its committed rows
                                               as an owned Vec<f32> (no copy),
                                               move it into NumPy (no copy),
                                               take the GIL, call infer
                                            5. route prediction rows back to
                                               each blocked slot by position;
                                               alias rows resolve from their
                                               twin; the route's gated hits
                                               release; cache inserts publish
                                               as ONE new immutable view
 6. Work::Resume ◀───────────────────────── freed slots respawn on the pool
    absorb predictions; another round
    (tree search) or the search is Done
 7. finish: select actions, advance the
    game, build learner records (the root
    row Vec moves into the record — it is
    never re-encoded), flush episodes
 8. TaskOut::Completed ───────────────────▶ 9. apply records/stats in message
                                               order; admit the slot's next
                                               decision; stop at the floor
```

Rounds 1–6 repeat per inference round: a model-free policy uses exactly one, a
tree search as many as its expansion schedule needs, pooling every slot's leaf
rows into the same shared arenas. A sealed arena's replacement is allocated by
the next worker that needs it — allocation is worker work, never scheduler work.

## The observation arena

Each arena packs its whole reservation state into one `AtomicU64`
(`closed:1 | row cursor:31 | alias count:32`), so claiming rows, claiming an
alias, and closing are single compare-and-swap transitions:

```text
   reserve(k): cursor += k        ── spans are DISJOINT: each worker owns its
   fill closes in the same CAS       rows exclusively and copies/encodes with
   close(): freeze (rows, aliases)   no lock held
        │
        ▼
 OPEN ──────────▶ CLOSED ──────────▶ SEALED (owned Vec<f32>, zero copies)
   writers may      no new claims;     only after EVERY reserved span has
   still be         frozen counts      committed (or poisoned) — commits
   committing       must reconcile     publish with Release, the seal
                                       checks with Acquire
```

A span commits or, if its task panics mid-write, poisons the arena — sealing
never waits on a row that cannot arrive, and a poisoned buffer is discarded
rather than served. Alias tickets follow the same discipline: an abandoned
claim is counted, so the fire condition (`rows and alias claims all accounted`)
reconciles instead of deadlocking. The protocol's interleavings are model-checked
with Loom and the unsafe storage is tested under Miri.

## The inference cache

With `infer_cache` enabled, workers consult the cache before reserving arena
rows. The cache is split by role: workers only ever read immutable snapshot
views pinned at task spawn; the scheduler owns all mutation and publishes a new
copy-on-write view once per fire. There are no locks on the lookup path.

A hit does not answer immediately — it parks as a *gated ticket* that releases
at its route's next fire (or at quiescence). Gating pins each hit to the same
weights-generation discipline as real inference: a ticket that outlives a
`weights_updated()` boundary is *demoted* — re-emitted as an ordinary arena row
and re-inferred — rather than served stale (`cache_demotions` in telemetry).

The cache pays for itself in proportion to callback cost: with a real network
it more than doubles AlphaZero-style throughput, while with a near-free callback
its pipeline overhead exceeds the saved work — see the
[`infer_cache` guidance](../reference/python-api.md) before enabling it.

## Liveness

Batches normally fire at capacity, so a route that fills slower than its peers
— or whose every lookup hits — could otherwise wait on rows that are never
coming. Three rungs guarantee progress:

- **Fire at capacity** — the normal case.
- **The stall ladder** — the scheduler tracks per-route progress from processed
  messages only; a route with unresolved demand that has not advanced across 8
  consecutive real fires engine-wide is closed and fired as a partial batch, and
  a hit-only route releases its gated tickets without an inference call
  (`stall_closes` / `stall_releases` in telemetry — both zero in balanced
  workloads).
- **Quiescence** — with nothing in flight, every open buffer is closed, fired,
  and released; this is also the terminal drain that ends the window.

## Where the time goes

The scheduler performs no per-byte work — every observation row is written once
on a worker and read next by the inference callback. Root observation rows cost
one worker-side copy (the row also becomes the training record); state-keyed
rows encode directly into arena storage and cost zero. What remains on the
scheduler is per-row constant work (span accounting, routing) and the
callback itself.

Measured against the previous copying scheduler (Apple M1 Max, interleaved
same-session pairs, decisions/s):

| workload | 1 thread | 10 threads |
| --- | --- | --- |
| `car_racing` pixels, PPO | 1.02x | 1.70x |
| `snake` vec, PPO | 1.08x | 1.57x |
| `connect4` AlphaZero + cache, 2 ms callback | — | 2.06x |

Pixel rows no longer serialize on the scheduler (the old design's known scaling
limit); at one thread the encoder itself dominates, so the gain appears as
thread count grows. Small-row games gain from the per-row fixed costs the
streaming path removed.
