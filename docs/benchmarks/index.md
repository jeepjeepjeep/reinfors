# Benchmarks

Two distinct families of measurement live here, sharing one environment and one
measurement discipline:

**[reinfors vs. OpenSpiel](openspiel/index.md)** — a controlled, like-for-like comparison
against [OpenSpiel](https://github.com/google-deepmind/open_spiel)'s C++ libtorch
AlphaZero: matched 2-hour training rounds, head-to-head strength evaluation, and the
structural analysis of why the numbers differ.

**[reinfors internals](internal/index.md)** — reinfors measured against itself:
device characterization (kernel batch curves, cache behavior) and the throughput effect of
individual configuration levers (native f32 outputs, inference caching, grouped
collection). These are the numbers that guide *your* configuration choices, independent of
any other framework.

Shared foundations, applying to both families:

- [Environment and setup](setup.md) — the instance, isolation invariants, and version pins.
- [Methodology](methodology.md) — the measurement protocol, including the rules that exist
  because an earlier version of the benchmark broke them.

Each family carries its own reproduction guide —
[internal](internal/reproducing.md) and [comparison](openspiel/reproducing.md); both
families' harnesses live in the companion benchmark repository, so one checkout carries
every measurement.

Every table states its own provenance inline — hardware, measurement window, and run
type (a repeat-derived spread, a repeat-median label, or an explicit single-run label). The commands that produce each
family's numbers are in the reproduction guides; raw learner telemetry for the published
runs is archived in the benchmark workspace.
