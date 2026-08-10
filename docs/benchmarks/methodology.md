# Methodology

The protocol below was not designed in one sitting: several rules exist because an earlier
version of the benchmark violated them and produced numbers that flattered one side. Each
rule states what it prevents.

## Termination: hard kill, interior windows

Every timed leg is terminated by SIGKILL at its deadline, on **both** stacks, and rates are
computed from counter deltas between timestamped samples strictly inside the window — never
by dividing a total by the nominal duration.

*Why:* graceful shutdown lets a stack keep producing while it drains in-flight work. In an
early sweep, one side was stopped with SIGINT plus a grace period and internally-timed
totals, inflating its rates by 18–31% relative to the hard-killed side. Discovered when its
single inference thread logged more busy-seconds than the leg's nominal length. If interior
samples are missing, the result is reported as failed — never reconstructed.

## The training-relevant rate is states/s, not rows/s

A *row* is one position forwarded through the net: search effort. A *state* is one training
example delivered to the learner — and in AlphaZero-style training a position only becomes
an example when its game **finishes** (the value target is the realized outcome). Under a
hard deadline, in-flight games count for nothing. Configurations routinely trade these
against each other (more parallel games → more rows/s, slower per-game progress → fewer
completed states/s), so topology selection and headline comparisons use states/s at matched
search budget; rows/s is recorded as diagnosis.

## Caches are architecture, not a matched knob

Each stack runs its own cache design at its own best capacity. Cross-stack comparison with
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

## Each side its own best configuration

Parallelism topology (actor/game counts, inference batch sizes) is **not** matched: each
stack runs the configuration that empirically maximizes *its own* states/s under the round
workload — cache on, learner and checkpoint writes sharing the GPU, windows long enough to
contain several learn steps and game lengths (see [tuning](tuning.md)). Selecting a rival's
topology for it, or measuring topology under a lighter workload than the round, are both
ways of quietly handicapping one side.

## Artifact selection under the deadline

Neither stack gets to write a "final" checkpoint — the deadline is a kill, not a request.
The head-to-head loads each side's last *complete periodic* checkpoint before the deadline;
checkpoint cadences are configured so worst-case staleness is comparable (on the order of
two minutes on both sides). Alias/"latest" checkpoint files are never used: a kill can tear
them mid-write.

## Determinism and repeats

Collection is seeded and bit-reproducible per stack under fixed weights and deterministic
inference; training runs are single-seed per round with the seed recorded, and the
single-seed caveat is carried explicitly until multi-seed rounds land (see the
[open items](matched-round.md#open-items)). Head-to-head matches use paired openings
(each opening played once per color) so opening imbalance cancels within pairs, and are
reported with standard errors computed over pairs, not games.
