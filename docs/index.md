# Overview

Reinfors is a reinforcement-learning library for experiments involving games, planning, and
neural networks. Game dynamics, search, episode orchestration, and batch assembly run in Rust.
Model inference and training are caller-owned and connected through Python.

## Execution model

In `Engine` workflows, policies request model evaluations while games and searches are running.
The engine pools these requests into NumPy arrays, calls Python for inference, and routes the
returned values to the requesting policies. Learners then convert completed decisions or
trajectories into training records.

```text
Rust engine and policies ── pooled NumPy observations ──▶  Python inference callback
Rust engine and policies ◀───── NumPy model outputs ───  Python inference callback
Rust learners            ──── typed training records ────▶  Python training loop
```

`Engine.collect` performs synchronous collection. `Engine.collect_stream` runs collection
concurrently in Rust and passes batches to the Python training loop through a bounded queue that
provides backpressure. Inference callbacks may use PyTorch, JAX, another local runtime, or a remote
service; they may also differ by player.

Other execution surfaces cover workflows that do not use learner-shaped engine collection:
[`Env`](guides/evaluation.md) exposes caller-controlled play, [`Arena`](guides/arena.md) runs paired
evaluations, and standalone solvers own algorithm-specific traversals such as CFR.

## Documentation map

- [Getting started](getting-started.md) covers installation and a first collection call.
- [Architecture](concepts/architecture.md) describes the execution surfaces and component
  responsibilities.
- [Training](guides/training.md) gives a complete Python training loop; the
  [streaming guide](guides/streaming.md) covers concurrent collection.
- The [game](catalogue/games.md), [algorithm](catalogue/algorithms.md), and
  [compatibility](catalogue/compatibility.md) catalogues describe the built-in components.
- [Examples](examples/index.md) links runnable scripts for collection, training, evaluation,
  adapters, direct play, and solvers.
- [Extending reinfors](extending/index.md) documents the Rust traits and Python registration path
  for new components.
- [Reference](reference/index.md) specifies inference shapes, batch fields, telemetry, terminology,
  and current boundaries.

## Scope

Reinfors uses native simulation and search within a reusable experimental interface. It is not
intended to maximize throughput for one fixed workload; a fully fused implementation specialized
for a particular model, game, and device topology may be faster.

The current scope includes:

- native game simulation, planning, episode collection, and inference batching;
- synchronous and concurrent collection with caller-owned models, optimization, and replay;
- sequential and simultaneous decisions, explicit chance, multiple players, and imperfect
  information where supported by the selected algorithm;
- resolved configurations, snapshots, telemetry, per-player inference, and evaluation surfaces;
- [Gymnasium and PettingZoo adapters](guides/adapters.md) for compatible games.

Games currently use
[fixed discrete action and observation spaces](reference/limits.md#fixed-observation-and-action-spaces),
one [decision phase](reference/limits.md#decision-phases), and native Rust
[component implementations](reference/limits.md#native-component-boundary). Reinfors is pre-1.0;
the [current boundaries](reference/limits.md) page records the applicable API and modeling
constraints.
