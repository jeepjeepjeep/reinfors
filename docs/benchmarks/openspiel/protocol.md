# Comparison protocol

The fairness rules specific to benchmarking two frameworks against each other, on top
of the shared [measurement discipline](../methodology.md).

## Caches are architecture, not a matched knob

Each stack runs its own cache design at its own capacity (their default; ours set equal —
hit rate is monotone in capacity). Cross-stack comparison with
caches *disabled* is not a "clean" condition: OpenSpiel's evaluator issues two requests per
expanded node (policy and value are separate calls) and relies on its LRU cache to merge
them, so cache-off roughly doubles its forwards per node while leaving reinfors (a single
both-heads call) untouched — the row stops meaning the same unit of work on the two sides.
The same principle applies to capacity: sizing their cache by a constraint measured on ours
(or vice versa) imports one architecture's costs into the other.

## Matched knobs

What *is* matched, exactly, on both sides: network architecture (layer-for-layer, verified
by parameter count at startup), search budget per move — including the convention that the
root expansion counts against it — exploration constants, Dirichlet noise parameters,
temperature schedule, replay-buffer size, minibatch size, optimizer hyperparameters, and
the effective training intensity (gradient-samples per collected state; the two learners
amortize this differently — one minibatch per fixed state count vs. periodic full-buffer
sweeps — but both resolve to the same reuse factor, which is verified from telemetry rather
than assumed). Loss definitions are aligned: both sides train a masked policy
cross-entropy over legal actions and value MSE on outcomes in [−1, 1], with weight decay
applied to the same parameter-name set.

## Each side its best measured configuration

Parallelism topology (actor/game counts, inference batch sizes) is **not** matched: each
stack runs the configuration that maximized *its own* states/s among the tested candidates
under the round workload — cache on, learner and checkpoint writes sharing the GPU,
windows long enough to contain several learn steps and game lengths (see
[tuning](tuning.md), which also records the grid's untested edges). Selecting a rival's
topology for it, or measuring topology under a lighter workload than the round, both bias
the comparison.

## Artifact selection under the deadline

Neither stack gets to write a "final" checkpoint — the deadline is a kill, not a request.
The head-to-head loads each side's last *complete periodic* checkpoint before the deadline;
checkpoint cadences are configured so worst-case staleness is comparable (on the order of
two minutes on both sides). Alias/"latest" checkpoint files are never used: a kill can tear
them mid-write.

## Head-to-head scoring

Training runs are single runs per (seed, side) with the seed recorded; the headline
results are replicated across two independent training seeds, and per-cell precision is
carried explicitly as an [open item](matched-round.md#open-items). Head-to-head matches
use paired openings
(each opening played once per color) so opening imbalance cancels within pairs, and are
reported with standard errors computed over pairs, not games.
