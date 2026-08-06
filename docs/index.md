# Search in Rust. Train your network your way.

Reinfors is a reinforcement-learning sampling library for experiments that combine games,
planning, and neural networks. It keeps the throughput-sensitive work—game dynamics,
parallel search, episode orchestration, and batch assembly—in Rust. It leaves the model,
optimizer, replay system, and hardware topology in Python.

The boundary is one inference callback:

```text
Rust games + search  ── pooled NumPy observations ──▶  your inference code
Rust batch assembly  ◀────── NumPy model outputs ───  PyTorch / JAX / RPC / …
```

That design supports a laptop training loop, CPU search feeding one GPU, concurrent
actor–learner collection, per-player networks, or a callback backed by a remote inference
service without changing the engine.

## Start at the level you need

<div class="grid cards" markdown>

-   **Try it**

    ---

    Install reinfors and collect your first batch with a small synchronous example.

    [Getting started →](getting-started.md)

-   **Build a training loop**

    ---

    Connect an arbitrary network, choose synchronous or concurrent collection, and consume
    learner-shaped training records (rows).

    [Architecture →](concepts/architecture.md) · [Training guide →](guides/training.md)

-   **Choose components**

    ---

    Browse the supported games and algorithm compositions.

    [Games →](catalogue/games.md) · [Algorithms →](catalogue/algorithms.md)

-   **Extend the library**

    ---

    Implement games, encoders, rewards, policies, learners, or solvers against the Rust
    traits and expose them through Python.

    [Extension guide →](extending/index.md)

</div>

## Core capabilities

- **Batched search and sampling.** Inference requests are pooled across parallel games and
  search leaves before crossing into Python.
- **Injectable training.** The callback exchanges NumPy arrays; reinfors does not own a
  framework, module, optimizer, replay buffer, or device.
- **Two collection modes.** `Engine.collect` is a direct request/response loop.
  `Engine.collect_stream` overlaps a background Rust collector with Python training and
  provides bounded backpressure.
- **Broad game semantics.** The core represents explicit chance, sequential or simultaneous
  decisions, N-player rewards, and imperfect information.
- **Native simulation, flexible models.** Games and search components run in Rust while Python
  supplies networks, optimization, replay, and deployment logic.
- **Direct play and standalone solving.** `Env` supports caller-driven evaluation and interactive
  play; solvers own algorithm-specific traversals outside the policy/learner collection model
  (currently including tabular and deep CFR workflows).
- **Experiment lifecycle.** Resolved configurations, configuration fingerprints, exact engine
  snapshots, environment forks, structured telemetry, and per-player record routing are
  part of the public surface.
- **Ecosystem adapters.** Supported games expose Gymnasium or PettingZoo adapters where
  those standards fit their interaction model; see the [adapter guide](guides/adapters.md).

## Scope

Reinfors owns native simulation, search, and batched data generation; the caller owns model training
and deployment. Games currently use
[fixed discrete action and observation spaces](reference/limits.md#fixed-observation-and-action-spaces),
one [decision phase](reference/limits.md#decision-phases), and native Rust
[component implementations](reference/limits.md#native-component-boundary). The project is pre-1.0,
so review the canonical [current boundaries](reference/limits.md) before committing to an experiment
design.
