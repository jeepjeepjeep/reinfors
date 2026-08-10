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
each simulation round gathers one leaf per game into a single batched callback that returns
policy and value from one forward. The inference cache is position-keyed and independent of
call structure. The learner is caller-owned Python running concurrently (records stream out
per collected batch); weight refreshes are explicit and clear the cache at round
boundaries. Optionally, games split into two groups whose rounds alternate so tree work
overlaps inference ([grouped collection](../internal/throughput-levers.md)).

## Consequence 1: batch size coupled to — or decoupled from — game count

In the OpenSpiel design, inference batch size is bounded by actor count: big batches
require many actors. But many concurrent games each progress slower, and under a deadline,
slower games mean fewer *completed* games — fewer training states. Its states/s-optimal
configuration therefore sits at a modest actor count with correspondingly small inference
batches, paying the small-batch region of the kernel curve. reinfors' lockstep makes batch
size *equal* to game count while all games advance at one shared rate, so it can sit at
the device's kernel sweet spot without a completion penalty. This coupling difference is
hypothesized to be the largest contributor to any measured throughput gap (the published
ablations quantify it) — and it is a scheduling choice, not an implementation-quality
difference: per-actor threading is the natural design
for a library whose algorithms and games vary widely; lockstep pooling is available to
reinfors because its search contract is narrower.

## Consequence 2: one inference question per node, or two

reinfors' callback contract returns both heads in one forward. OpenSpiel's evaluator
interface separates value and prior queries — a clean interface for a library that also
serves rollout-based evaluators without networks — and relies on its cache to merge the
pair. With the cache on (its intended condition) the merge works and the difference mostly
vanishes; the structural residue is cache-shaped rather than compute-shaped (entries,
lookups, and eviction pressure per node differ). This is also why cache-off cross-stack
comparisons are invalid rather than "clean" — see the [comparison protocol](protocol.md).

## Consequence 3: continuous versus burst training

The reinfors learner trains continuously beside collection; self-play always runs against
near-current weights, and GPU time interleaves at fine grain. OpenSpiel's learner trains in
periodic sweeps during which the device prioritizes training; its actors play against
weights up to one sweep stale, then jump. Neither profile is free — and neither is
more "canonical" (the original AlphaZero trained asynchronously and continuously; both
stacks approximate that differently): burst training buys simple synchronization and a
self-contained learner at some device contention and staleness; continuous training buys
freshness at the cost of a Python-side training loop sharing the process. The matched training-intensity check (gradient-samples per state,
verified from telemetry on both sides) exists so this scheduling difference is not
mistaken for a difference in how much learning happens.

## Consequence 4: what a deadline does to in-flight work

Both stacks lose in-flight games at the hard kill; how much is in flight is a design
consequence — proportional to concurrent games and per-game latency. See the shared
[methodology](../methodology.md) for why completed-game states/s is the selection metric.

## What reinfors pays for its choices

Symmetry requires the reverse list, stated at the same precision as the forward one.

**Heterogeneous compositions are first-class inside their run infrastructure.** Any object
satisfying their `Bot` interface slots into an actor loop per seat: their trainer runs
concurrent *evaluation actors* (current net vs. reference baselines, producing live
strength curves) during training; mixed-bot matches reuse the identical loop; and a
no-network rollout evaluator ships in-library. reinfors handles heterogeneity at a
different layer — the caller-driven `Env` expresses arbitrary bot mixes (this benchmark's
own head-to-head bridge and referee bots are built on it), and per-player inference with
`learn_players` covers frozen opponents and per-seat *networks* inside the engine — but
anything beyond homogeneous self-play forfeits the engine's batching, cache, and telemetry
machinery. Concretely missing in-engine: concurrent evaluation actors, per-seat *search*
heterogeneity, and an in-library rollout baseline.

**New search families pay an integration cost for throughput.** In their model, a new bot
type never touches the run loop; in reinfors, a search that wants the engine's machinery
must integrate with the pooled round staging, and features like grouped collection extend
to search families individually. The symmetric caveat: their free
scheduling is not free *batching* — a custom OpenSpiel bot wanting their inference
service's batches must integrate with the evaluator queue just the same. Generality with
free scheduling is not generality with free throughput on either side.

**Per-game latency.** At its throughput-optimal configuration, reinfors' games share each
search round, so individual games progress slower than under a small-actor OpenSpiel
configuration that finishes single games faster — a real difference for interactive or
latency-sensitive uses, immaterial for bulk collection.

**The learner shares a Python runtime.** reinfors' caller-owned training loop lives in
Python beside collection; OpenSpiel's all-C++ learner never shares a runtime with user
code. The flip side is flexibility — the training loop being caller-owned is a reinfors
design goal, not an accident — but the cost is real in workloads with heavy Python-side
work between collects.

## Scope of the claim

The measured gap is the sum of scheduling and interface choices, each of which OpenSpiel
could adopt at the cost of generality it deliberately keeps, and each of which reinfors
could lose by broadening scope. The benchmark's claim is correspondingly narrow: on this
workload, a modular Rust/Python boundary does not have to forfeit throughput to a fused
C++ design — not that one library outranks the other.
