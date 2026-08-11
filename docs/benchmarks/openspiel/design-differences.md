# Design differences

Where the two stacks' numbers differ, the causes are identifiable structural choices — each
sensible for its system's goals. This page describes both architectures and traces each
measured difference to its design origin, in both directions: several of OpenSpiel's
choices are the right ones for a reference research library, and reinfors' choices cost
it elsewhere (generality, per-game latency).

## The two architectures

**OpenSpiel (C++ libtorch AlphaZero)** runs *independent actor threads*, each playing its
own game with its own MCTS. Actors submit inference requests to a central service whose
batcher gathers up to `inference_batch_size` requests per forward. The MCTS evaluator asks
two questions per expanded node — a value query and a policy-prior query, as separate
calls — and an LRU cache in front of the network merges the pair into one forward and
serves cross-search repeats. The learner runs periodically: when enough new states arrive
it takes the device and performs a full sweep of buffer-sized minibatches, then actors
resume against refreshed weights.

**reinfors** runs a *lockstep pooled search*: all `n_games` games advance together, and
each simulation round gathers their uncached leaves into one callback that returns policy
and value from one forward. The inference cache is position-keyed and independent of call
structure. The learner is caller-owned Python running concurrently (records stream out per
collected batch); weight refreshes are explicit and clear the cache at round boundaries.
Optionally, games split into two groups, each collecting on its own worker thread with
inference forwarded to a service thread that owns the callback, so tree work overlaps
inference ([grouped collection](../internal/throughput-levers.md)).

## Consequence 1: batch formation

OpenSpiel's asynchronous batcher needs enough independently progressing actors to fill a
large inference batch. reinfors instead stages up to one fresh leaf per active game in a
synchronized round, making callback sizes more predictable. Neither batch size is simply
the configured game count: cache hits, terminal simulations, and deduplication remove rows,
while larger game or actor counts can increase per-game latency and leave more work in
flight at the deadline. The tuning grids measure whether fuller batches outweigh those
completion costs.

## Consequence 2: one inference question per node, or two

reinfors' callback contract returns both heads in one forward. OpenSpiel's evaluator
interface separates value and prior queries — a clean interface for a library that also
serves rollout-based evaluators without networks — and relies on its cache to merge the
pair. With the cache on (its intended condition) the merge works and the difference mostly
vanishes; the structural residue is cache-shaped rather than compute-shaped (entries,
lookups, and eviction pressure per node differ). This is also why cache-off cross-stack
comparisons are invalid rather than "clean" — see the [comparison protocol](protocol.md).

## Consequence 3: continuous versus burst training

The reinfors learner trains continuously beside collection; self-play runs against
near-current weights, and GPU time interleaves at fine grain. OpenSpiel trains in periodic
sweeps, so actors play against weights up to one sweep stale. Burst training simplifies
synchronization but introduces device contention and staleness; continuous training keeps
weights fresher while sharing the process with a Python training loop. Telemetry verifies
matched gradient-samples per state so this scheduling difference is not mistaken for a
difference in training intensity.

## Consequence 4: what a deadline does to in-flight work

Both stacks lose in-flight games at the hard kill; how much is in flight is a design
consequence — proportional to concurrent games and per-game latency. See the shared
[methodology](../methodology.md) for why completed-game states/s is the selection metric.

## Trade-offs

| dimension | OpenSpiel | reinfors |
|---|---|---|
| heterogeneous play | Per-seat `Bot` composition inside its run infrastructure, including evaluation actors and rollout baselines. Batching still requires evaluator integration. | Per-player networks and frozen opponents stay inside `Engine`; arbitrary bot or search mixes use `Env`, outside Engine-provided batching, caching, and telemetry. |
| new search integration | A new bot can use the actor loop directly; batched network service requires its evaluator queue. | The standard policy seam uses normal collection, including under `n_groups=2` (group workers run any policy's search opaquely). |
| per-game latency | Small actor counts can favor individual-game latency. | Throughput-oriented lockstep batches can increase individual-game latency. |
| training ownership | The C++ learner is self-contained. | The caller-owned Python learner is flexible but shares the process during concurrent collection. |

## Scope of the claim

The benchmark tests whether reinfors preserves throughput across its modular Rust/Python
boundary on this workload. The published results bound that claim; they do not rank the
libraries generally.
