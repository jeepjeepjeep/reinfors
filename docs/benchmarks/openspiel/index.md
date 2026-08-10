# reinfors vs. OpenSpiel

A controlled comparison against OpenSpiel's C++ libtorch AlphaZero on a fixed cloud GPU
instance: each stack trains chess from scratch for the same wall-clock at its own measured
best configuration, and the resulting models play each other.

## What this comparison is — and is not

OpenSpiel is a research library whose goals are breadth (many games, many algorithms) and
reference clarity; high throughput is not one of its stated objectives. reinfors narrows
its scope to a modular game/search/training boundary and asks a narrower question: **how
much throughput does that modularity preserve against a mature C++ implementation, on a
workload both systems support well?**

The headline shape is therefore not "reinfors is faster" but: comparable — in places
somewhat better — throughput, while keeping the pluggable game/encoder/search/learner
seams that are reinfors' design goal. Where the numbers differ, the
differences trace to identifiable structural design choices on each side, not to one
implementation being "better engineered" — see [design differences](design-differences.md).
Both stacks were measured at their own best configuration, on the same hardware, under the
same protocol, with every mismatch we found treated as a bug in the benchmark rather than
a result.

## The comparison target

The benchmark targets OpenSpiel **as maintained today**: a pinned master snapshot built
from source with CUDA libtorch, including upstream's own performance fixes. The pin policy:
the snapshot is taken from the upstream tip, and before any publication run it is
re-verified against current master — if commits have landed touching the measured
subsystems (`algorithms/alpha_zero_torch`, the game, the evaluator/bot surfaces), the
benchmark re-pins and rebuilds rather than publishing against a superseded target. The pin
in force is recorded with every result. Two documented,
content-preserving interventions were required — restoring build glue that master deletes
while still referencing (the libtorch path does not build as shipped), and measurement
instrumentation in its evaluator plus a device flag in its example game runner. All patches
ship with the benchmark harness and are applied idempotently. Torch/libtorch version pinning is
covered in [setup](../setup.md).

## Contents

- [Comparison protocol](protocol.md) — the fairness rules layered on the shared
  methodology: matched knobs, each side its own best configuration, cache semantics,
  deadline artifacts, paired scoring.
- [Operating-point tuning](tuning.md) — how each side's configuration was selected
  empirically, under the full round workload, before any head-to-head.
- [The matched round](matched-round.md) — 2h training throughput, the head-to-head
  protocol and results, strength-over-time, and the open items.
- [Design differences](design-differences.md) — both architectures, and each measured
  difference traced to its structural origin, in both directions.

The shared [environment](../setup.md) and [methodology](../methodology.md) pages govern
every number in this family; the [comparison protocol](protocol.md) adds the rules that
only exist because there are two frameworks in the room.
