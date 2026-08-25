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
   dst: &mut [f32])` with a default that delegates to `encode()`, plus a
   state-backed sink operation `push_state(player, state)` under which the SINK
   calls the encoder: cache key first (see the caching element), then, on a
   miss, span reservation and `encode_into`. This removes the last copy for the
   rows where it compounds — high-volume, never-retained search leaf rows — and
   guarantees key/row coherence structurally, because the key and the row come
   from the same encoder and state; a closure-based push cannot promise that,
   so legacy arbitrary-row pushes keep observation hashing. `car_racing`'s
   pixel encoder is already shaped for it (its downsample writes into a
   caller-provided buffer).

   Arenas are allocated uninitialized (`MaybeUninit<f32>` internal to the
   wrapper; nothing zeroes whole arenas). Stage 1 needs no zeroing at all:
   copying a complete source row into the raw span initializes it. Stage 2
   hands encoders `&mut [f32]`, and constructing a reference over
   uninitialized memory is unsound regardless of element type — so the
   wrapper enforces this sequence: reserve a disjoint raw range → zero the
   range → only now construct the `&mut [f32]` → encode → commit with
   Release. The per-span zero is one row-equivalent of work, parallel across
   the workers producing observations, and warm — it touches exactly the
   lines the encoder writes next. Uninitialized memory is never reachable
   through any reference, and an encoder that underwrites its span produces
   zeros, not undefined behavior. At seal, after the Acquire-ordered
   reconciliation proves every prefix span committed (each committed span is
   copy- or zero-then-write-initialized), the prefix is converted from the
   internal `MaybeUninit<f32>` allocation into an owned `Vec<f32>`. Zeroing
   solves value validity only: the wrapper must separately guarantee that
   concurrently writing workers receive strictly disjoint ranges (see the
   reservation protocol).

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
     independently — the per-span zero makes the values valid, but only
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
     packed atomic — a single `AtomicU64`: `closed:1`, `row_cursor:31`,
     `alias_count:32`, with buffer capacity checked `< 2^31` at construction
     and the alias count bounded below its field by the staged map's fixed
     capacity (checked, never assumed) — and reservations AND alias
     registrations are both CASes on it, both failing once closed. A saturated
     alias count or full staged map falls back to an ordinary reservation — a
     duplicate row rides to inference, compute not correctness; alias pressure
     never closes a buffer, closing stays owned by capacity and fire cadence. Close
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
     uncommitted hole that stalls sealing forever. Aliases get the same
     treatment: the alias CAS hands back a ticket whose `Drop`, if uncommitted,
     delivers an abandonment notification through the panic drain, and
     reconciliation counts abandonments toward `m` (the slot is dead — the
     scheduler accounts, routes nothing), so `(k, m)` never waits on a
     notification that cannot arrive. The scheduler's existing panic-drain path
     must stay deadlock-free with an arena in flight.
   - *Lifecycle.* Each buffer moves `Filling → Sealed → InInference → Released`
     (released = moved into NumPy), with transitions owned by the scheduler.
   - *An explicit `RoundMailbox`, installed at spawn.* Stage 2 commits spans
     while `Policy::round(...)` is still executing, so a full arena can fire
     — and its predictions return — before the worker sends
     `TaskOut::Emitted`, i.e. before the scheduler owns the search state.
     Deferring commits until the round returns is not safe: a round larger
     than the pool's remaining capacity would hold uncommitted spans and
     deadlock the fire. Instead the scheduler installs the round's mailbox
     BEFORE spawning it; commits and returned predictions resolve tickets
     into it even while the worker still owns the search; `Emitted` transfers
     the search state and marks the ticket set complete; and the slot resumes
     only when BOTH have happened — state transferred and every ticket
     resolved. Either arrival order works, and it is the same join the
     mailbox already performs for rounds straddling fires. One cleanup
     obligation: a worker panicking mid-round after committing spans fires
     normally but never sends `Emitted` — the panic drain must also discard
     the orphaned mailbox and its parked predictions.

3. **Capacity is fixed; shape is not.** The buffer's capacity is a
   protocol-internal constant — spans must be reserved against a fixed limit —
   and is NOT a callback-shape contract. A short fire truncates the `Vec` to its
   initialized prefix (O(1) for `f32`, capacity slack rides along until Python
   frees it) and moves it into NumPy as `(n, dim)`, exactly today's
   variable-shape default, zero copies. Padding stays the `pad=True` opt-in for
   callers who want `torch.compile`-stable shapes: with it enabled, a short
   fire's suffix is initialized by a spawned worker task (see the fire ladder
   in element 8) — full fires have no suffix and cost nothing. Zero-copy must
   never require padding.
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

5. **Ownership per fire; allocation is worker work.** Each fired buffer is a
   distinct allocation, moved into its NumPy array exactly as today (Python owns
   the storage; its GC frees it). Replacement arenas are allocated OFF the
   scheduler, by rule, and uninitialized — allocation is a capacity request,
   not a memset, so its cost does not depend on allocator or page state
   (zeroing is per span, at reservation — see (1)). The worker whose
   reservation CAS closed the previous buffer commits and notifies the
   scheduler FIRST, unblocking sealing and the fire, and only then allocates
   the replacement. The scheduler never allocates observation storage.
   Nothing is recycled when the callback returns: callbacks legitimately retain
   arrays, wrap them in `torch.from_numpy`, or start asynchronous device
   transfers, and reuse on return would mutate retained tensors or race an
   in-flight copy. Workers simply open a freshly allocated arena while the
   callback runs on the previous one — the allocator hands back same-sized blocks
   essentially for free, and the pages are touched by worker writes either way.
   If profiling ever shows allocation itself mattering, storage can be returned
   to a pool through the NumPy owner's deallocator capsule — reuse gated on the
   array's actual death, so retained arrays delay reuse safely instead of being
   corrupted by it. Recycled buffers come back dirty, which this design
   tolerates by construction — storage is uninitialized anyway and zeroing is
   per span — so pooling carries no hidden arena-scale re-zeroing cost.
6. **The evaluator is replaced, not layered over.** A pooled buffer placed
   underneath the current `Evaluator::forward` would still be staged row-by-row
   into `EvalBatch::obs_flat` — the copy would move, not disappear. The arena IS
   the batch, and this is the only path: there is no fallback evaluator to fork
   behavior on, so caching and deduplication must work inside it.
7. **Caching moves to a read/write split; the cache stays an overlay.** Today the
   evaluator hashes and compacts rows at fire time — inherently O(bytes) on the
   scheduler. Instead:

   - *Cache identity is an encoder capability.* The lookup must precede the
     reservation — a hit reserves nothing, and there is no un-reserving from a
     monotone cursor — so the key must exist BEFORE the row does. Only the
     encoder knows what the observation depends on, so it names its own key:

     ```rust
     fn cache_key(
         &self,
         state: &Self::State,
         perspective: usize,
         hasher: &mut CacheHasher,
     ) -> bool;
     ```

     Built-in encoders stream the minimal identity that guarantees identical
     observations; imperfect-information encoders stream their
     information-state representation (recovering the hits that a full-state
     key would lose); a conservative encoder hashes the full state; `false`
     means no pre-encoding key exists, and the request takes a worker-local
     scratch path instead — encode into scratch, hash the bytes, copy to a
     span only on a miss (stage-1 economics for that row, per request, not
     per route). The contract is an invariant with teeth: equal `cache_key`
     streams MUST imply byte-identical encoded rows — a too-narrow key only
     costs hits, a too-wide key silently serves wrong predictions. Encoder
     keys and observation-hash keys share one cache, domain-separated by a
     leading tag byte so the two key spaces cannot collide. `CacheHasher` is
     an opaque engine-owned streamer (the engine picks function and seed, and
     can salt per route). On a hit, the encoder never runs — for expensive
     encoders that saving dwarfs the GPU trip itself.
   - *Worker-read, scheduler-write.* Workers compute the key (cache-hot,
     parallel) and look it up in a read-mostly shared cache. The scheduler
     remains the cache's ONLY writer: inserts and eviction happen at routing
     time, in message order, keyed by the hash the worker sends with its span
     metadata — the scheduler never touches bytes. The write-side logic
     (insert, evict, the per-player `Exclusive` slots) survives; read-side
     promotion does NOT — today's cache promotes entries during lookup, and
     under this split recency moves to the writer (next bullet).
   - *Publication is shallow copy-on-write; recency lives outside it.* A shard
     is an `Arc<HashMap<Key, Arc<Row>>>` behind an atomic pointer; an insert or
     evict clones the shard map SHALLOWLY (`C/S` pointer clones for capacity
     `C` over `S` shards — size shards so `C/S <~ 64` and maintenance is
     sub-microsecond, O(rows)-scale), mutates, and swaps the pointer. Evicted
     rows stay alive through `Arc` clones held by in-flight readers and pinned
     views. Recency metadata is deliberately NOT in the shared shards: workers
     never read it, so it needs no snapshot semantics — the scheduler keeps a
     private LRU fed by the span metadata workers already send ("keys I
     touched"), and promotions mutate only that private structure. Only
     insert/evict touch the shared shards; a COW map that also carried recency
     would clone a shard per LOOKUP and cost more than the copies it removes.
     An in-place `RwLock` shard is not an alternative: pinned views (below)
     require immutable snapshots, and retrofitting those onto in-place mutation
     (sequence-gated lookups, deferred eviction) rebuilds copy-on-write with
     epoch machinery on top. If shard-clone cost still measures high at target
     capacity, the escalation path is a per-shard persistent map (structural
     sharing, O(log n) insert) — same snapshot semantics, cheaper writes.
   - *Cache views are pinned at spawn.* A lookup that dereferences the shared
     shard pointers at read time sees an insert published in the SAME generation
     — or misses it — by pure pointer-load timing, and hit-vs-miss changes
     whether an arena row is occupied and hence fire cadence: nondeterministic
     even at `n_threads=1`, because scheduler and worker are still two threads.
     So the scheduler attaches the snapshot current at spawn (an `Arc` clone of
     the shard-pointer array) to each `Work` item, and worker lookups read only
     through that pinned view. Which snapshot an item carries is then a pure
     function of the serial spawn sequence: read outcomes depend on
     deterministic spawn order, not timing. Pinning applies at every thread
     count — the cost is an `Arc` clone per item and a microseconds-stale view
     (marginal hit-rate loss), and old views die with their in-flight items, so
     retained memory stays bounded by in-flight work. The generation gate below
     is unchanged and complementary: pinning gives same-generation read
     determinism; the gate gives cross-generation validity. A state inserted
     after a pin can still be recomputed by that item as a miss — the same
     duplicate-compute window as the two-plane rule below, covered the same way.
   - *A hit never reserves a span.* The batch contains only genuine misses by
     construction — no dead rows, no fire-time compaction, no wasted callback
     compute.
   - *Hits are gated on fire cadence — per ticket, per route.* Every emitted
     row is a ticket in its slot's `RoundMailbox` (installed at spawn — see
     the reservation protocol), and the slot resumes only when the round's
     search state has transferred AND all tickets resolve. Hits, aliases,
     misses, and rows spanning multiple fires are all just entries there,
     resolved by different paths. A
     ticket resolves by its fire returning (miss), by aliasing a reserved row
     (alias), or — for a hit — at its release point: the row's value is already
     in hand at the worker, but the ticket parks until the next completed fire
     OF ITS OWN inference route. The gate exists for two per-route properties —
     sampling cadence and generation validation — so parking a player-A hit on
     player B's callback would pace nothing and validate nothing. Ungated hits
     would let cache-hot slots free-run: many decisions per fire, rare misses
     starved of batch-mates, the record floor closing on a batch that
     overrepresents recurring states. Gating restores today's pacing, and
     unlike counting hits toward the firing threshold it never fires a partial
     batch: the GPU always sees `batch_size` rows in steady state. Ticket
     granularity makes the mixed cases free: a round with misses on route R
     releases its R hits at the same fire that resolves those misses — zero
     added latency for any round with at least one same-route miss; only
     pure-hit rounds wait for a fire driven by other slots (or the ladder).
     Rounds split across buffers need no new rule: tickets resolve fire-by-fire
     and the mailbox joins them, exactly as when a round's requests straddle
     fires today. The latency cost is absorbed by slot count, not throughput —
     see the sizing rule below.
   - *In-flight duplicates alias.* A per-buffer staged map (fixed capacity,
     insert-or-read-only, CAS on empty slots, cleared by the scheduler at each
     fire) lets a second requester of an already-reserved key record an alias
     ticket instead of reserving; the scheduler routes the one prediction row to
     every claimant. A full staged map falls back to an ordinary reservation —
     a duplicate row rides to inference; compute cost, not correctness (bounds
     and abandonment handling live with the reservation protocol's packed
     atomic).
   - *Invalidation is validated at the gate.* `weights_updated()` clears the
     cache at a safe boundary, but a worker can take an old shard snapshot,
     resolve a hit from it, and hold that row through a retained `Arc` past the
     boundary — no worker-side generation check can close this (the window
     between check and use is unbounded under preemption). The gate closes it
     structurally: every hit already waits for a scheduler release, and the
     scheduler is also the single thread that applies invalidation, so hit
     notifications carry their generation-at-lookup and the scheduler validates
     it at release. Current generation → deliver. Stale → demote that TICKET:
     the scheduler spawns a recompute task and a worker re-emits the retained
     row through the completely ordinary reservation path into the fresh open
     buffer — indistinguishable from a first-time miss; the round's other
     tickets are untouched and the slot keeps waiting on its mailbox. Two
     supporting rules: a slot keeps a gated ticket's encoded row until the
     release confirms the hit (so demotion never re-encodes), and each fired
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
8. **A fire ladder guarantees liveness.** Gating and partial buffers create
   stalls the ungated design could not have, and a GLOBAL quiescence guard
   misses them: an unrelated route can keep `in_flight` nonzero indefinitely
   while a stalled route starves — whether hit-bound (every lookup hits, no
   misses accumulate, no fire ever comes) or stuck on a partial buffer whose
   producers are all blocked on that very buffer. No snapshot condition can
   detect the latter — a slot blocked on route B may or may not emit A rows
   after it resumes — so the ladder detects lack of PROGRESS instead:

   - *Fire at capacity* — the normal case.
   - *Progress-counted partial close.* Per route, the scheduler records
     buffer fill at each completed fire. A route with unresolved demand —
     pending rows or gated tickets — whose fill has not advanced across K
     consecutive completed fires engine-wide is closed and fired as a partial
     batch, its gated tickets released under normal fire semantics; a stalled
     route with an EMPTY buffer "fires" as a no-op and just releases its
     tickets, generation-validated against the route's current generation
     directly. Only real, non-empty inference fires advance this clock — an
     empty release does not count, or stalled empty routes would tick one
     another into release cascades. A healthy route's buffer grows between
     fires, so the counter never reaches K in balanced steady state; it
     triggers exactly when starvation is real, and over-eagerness is tunable
     via K. Scope honestly: this bounds ZERO-PROGRESS time, not total
     latency — a route receiving one row every K-1 foreign fires keeps
     resetting its counter and can take a long time to fill, a deliberate
     utilization tradeoff; a separate larger maximum-age threshold can be
     added later if measurements show trickling routes are a problem.
   - *Global quiescence* — every slot parked, `in_flight == 0` — remains the
     terminal drain, and the only detector for a single-route engine (no
     foreign fires to count): close each open buffer (the same atomic close
     as a capacity fill), seal at current fill (truncate-to-prefix;
     race-free because zero in-flight tasks implies every reserved span is
     committed AND every registered alias message already processed — the
     `(k, m)` reconciliation is trivially satisfied at true quiescence),
     fire non-empty buffers, release every route's gated tickets.

   The counter is deterministic, but closing can still race a worker
   reservation: even at `n_threads=1` the scheduler and worker are
   concurrent threads, and at the Kth fire a worker mid-reservation lands in
   the old or the next buffer by CAS timing. The threshold therefore
   triggers a brief scheduling barrier — stop spawning/resuming worker
   tasks, drain already-running tasks and their commit messages, close and
   fire the stalled routes on that stable prefix, resume normal admission —
   paid only on an actual starvation event, where a partial fire is already
   accepted. Without the barrier the protocol stays memory-safe and live,
   but exact one-thread batch ordering is not formally deterministic. On a
   `pad=True` route, a partial close is where the padded suffix comes from,
   and the scheduler must not zero it itself: it atomically closes the
   buffer, spawns a small worker task that initializes rows `k..capacity`,
   and fires only after that task publishes completion; unpadded closes
   truncate and need no suffix work. Telemetry distinguishes partial
   miss-buffer closes, empty hit-only releases, and forced global drains.

   Today's engine has this hole in miniature — partial queues fire only at
   global settle, so a hot route can starve a cold route's staged rows for
   the whole window. The ladder is strictly stronger than today's behaviour.

   Sizing rule (per route): full batches in steady state need roughly
   `n_games >= 2 x batch_size / (1 - hit_rate)` — one batch's worth of misses
   accumulating while the previous batch is in the callback, with hit-gated
   tickets not contributing. A well-sized run only closes partially at
   window boundaries, so the ladder's counters directly expose undersized
   configurations: if one ticks in steady state, the config violates the
   sizing rule and the operator sees small batches coming.
9. **Tail bootstraps become worker tasks.** Today `tail_requests` encodes the
   truncation/fragment bootstrap observations on the scheduler thread — for
   pixel games that is a full render each, serialized. Under the arena, a cut
   spawns a tail task onto the worker pool that encodes, reserves, and commits
   like any other emission, with a tail marker routing its prediction rows to
   the existing `AwaitingTail` bookkeeping.
10. **The scheduler performs no O(bytes) operations — by construction, with the
   residue ledger stated.** Its work is reservation accounting, firing, key
   lookups' insert/evict side, routing the (small, f64) prediction rows back to
   blocked slots, and floor/record bookkeeping — all O(rows). The system's
   remaining byte work lives elsewhere, priced: worker-side row copies (two in
   stage 1, approaching zero for stage-2 encoders, one owned copy for canonical
   root rows); per-span zeroing on the stage-2 path (one row-equivalent per
   row, parallel across workers, warm — see (1)); and suffix initialization
   for `pad=True` partial closes, run as a worker task the fire waits on.
   Nothing in that ledger is serialized: every entry is per-row work spread
   across workers, or a bounded task off the arena hand-off path.

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

Engine-level gates:

- Determinism at `n_threads=1` under the new semantics: fixed seed produces
  byte-identical batches run-to-run (equality against the OLD scheduler is not a
  goal — hit gating and fire cadence legitimately reorder collection).
- Telemetry assertions: per-slot record counts and hit rates exposed; the
  stall-close, empty-release, and global-drain counters stay at zero in a
  correctly sized steady-state run.
- The existing scheduler test suite (firing thresholds, per-player routing,
  fragment cuts, panic drains) unchanged.
- Throughput gate: pixel collection at 8 threads must beat the current plateau by
  a stated margin on the benchmark machine before the old path is removed.

Protocol-level coverage, tested against the arena wrapper independently of the
engine:

- Out-of-order commits: spans committed in permuted orders still seal exactly
  once, with correct `(k, m)` reconciliation.
- Panic between reserve and commit: the RAII guard poisons the span; sealing and
  the scheduler's panic drain both terminate.
- Mid-round fires: a round exceeding the open buffer's remaining capacity fires
  and returns predictions before `Emitted`; the slot resumes correctly in
  either arrival order, and a mid-round panic after committed spans drains the
  orphaned mailbox.
- Requests crossing and exceeding capacity: emissions split across buffers and
  across multiple fires, preserving `absorb` order through the routing table.
- Alias-versus-seal races: the atomic close either admits the alias into the
  fire's counts or fails its CAS; no interleaving orphans a claimant.
- Alias overflow and abandonment: a full staged map falls back to an ordinary
  reservation; an alias ticket dropped without commit delivers an abandonment
  counted toward `m`.
- Generation invalidation during lookup: hits resolved from a pre-invalidation
  shard are demoted at the gate; no post-boundary decision consumes a
  pre-boundary row; pre-boundary fires never insert.
- Pinned-view determinism: a scheduler insert racing a lookup never changes
  outcomes — each Work item's hits and misses are a function of its spawn-time
  snapshot; fixed-seed `n_threads=1` runs are byte-identical with the cache
  enabled.
- All-hit and sparse-miss batches: gated tickets release on their route's fire
  or its stall release; a window of pure hits terminates.
- Mixed hit/miss rounds and cross-buffer rounds: the slot resumes only when
  every ticket resolves; same-route hits release with the fire that resolves
  the round's misses.
- Starved-route close: a route with a partial buffer and gated tickets whose
  producers are all blocked on it closes and fires after K foreign fires while
  an unrelated route stays hot; an empty stalled route releases as a no-op;
  empty releases never advance the stall clock (no release cascades).
- Stall-barrier determinism: the Kth-fire close drains running tasks and their
  commit messages before closing; fixed-seed `n_threads=1` runs stay
  byte-identical with induced stalls.
- High-hit partial-buffer starvation: configs below the sizing rule make
  progress through the ladder and increment its counters.
- Cache-maintenance microbenchmark: insert/evict/promotion measured at target
  capacity before the COW sharding is accepted; escalation to a per-shard
  persistent map if shard clones measure high.
- Cache-key soundness: a property test per built-in encoder — states with
  equal `cache_key` streams must encode byte-identically — alongside the
  existing parity suites.
- Key-space separation: encoder-derived and observation-hash keys never
  collide (tag-byte domain separation exercised on a route using both push
  paths).
- Retained Python arrays: a callback that stores every batch it receives —
  arrays stay valid and unchanged after arbitrary further collection.
- Padded-suffix initialization: with `pad=True`, the suffix reads as zeros on
  every platform/allocator — written by the spawned padding task, the fire
  blocked on its completion — including after buffer splits and quiescence
  seals.
- Allocation-path benchmark: measured with recycled allocations, not fresh
  ones — lazily mapped fresh pages understate what dirty buffers pay.
- Pool backpressure and cancellation: reservation parks at pool exhaustion and
  wakes on fire; dropping the engine mid-window releases parked workers and
  leaks no arena.
- Per-player and tail ordering: per-player routes and worker tail tasks preserve
  today's routing and `AwaitingTail` semantics under the arena.
- Tooling: the unsafe storage wrapper runs under Miri; the reservation /
  close / alias protocol gets a small Loom model checking the CAS
  interleavings exhaustively.
