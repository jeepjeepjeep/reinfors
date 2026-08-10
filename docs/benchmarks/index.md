# Benchmarks

This section documents a controlled, like-for-like comparison between reinfors and
[OpenSpiel](https://github.com/google-deepmind/open_spiel)'s C++ libtorch AlphaZero on a fixed
cloud GPU instance, together with the measurements used to configure it and the throughput
levers reinfors exposes.

## What this comparison is — and is not

OpenSpiel is a research library whose goals are breadth (many games, many algorithms) and
reference clarity; high throughput is not one of its stated objectives. reinfors narrows its
scope to a modular game/search/training boundary and asks a narrower question: **how much
throughput does that modularity preserve against a mature C++ implementation, on a workload
both systems support well?**

The honest headline shape is therefore not "reinfors is faster" but: comparable — in places
somewhat better — throughput, while keeping the pluggable game/encoder/search/learner seams
that are reinfors' actual design goal. Where the numbers differ, the differences trace to
identifiable structural design choices on each side, not to one implementation being "better
engineered" — see [design differences](design-differences.md) for that analysis. Both stacks
were measured at their own best configuration, on the same hardware, under the same
protocol, with every mismatch we found treated as a bug in the benchmark rather than a
result.

## Contents

- [Environment and setup](setup.md) — the instance, isolation, and software both stacks ran on.
- [Methodology](methodology.md) — the measurement protocol and the fairness rules, including
  the ones we learned the hard way.
- [Operating-point tuning](tuning.md) — how each side's best configuration was determined
  empirically before any head-to-head.
- [The matched round](matched-round.md) — 2h wall-clock training on each stack plus a
  head-to-head strength evaluation of the resulting models.
- [Design differences](design-differences.md) — the structural analysis of *why* the numbers
  come out the way they do.
- [Throughput levers](throughput-levers.md) — measured, opt-in reinfors features that move
  collection throughput (f32 outputs, inference cache, grouped collection).
- [Reproducing](reproducing.md) — exact commands, seeds, and artifacts.

Numeric results are being populated as the final long-form runs complete; tables marked
_TBD_ are structural placeholders whose protocol is already frozen. Every published number
links to the commit, resolved configuration, command line, and raw logs that produced it.
