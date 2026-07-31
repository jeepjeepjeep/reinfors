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
    learner-shaped records.

    [Sampling and training →](how-it-works/sampling-and-training.md)

-   **Choose components**

    ---

    Compare the currently registered games and algorithm compositions without relying on a
    hard-coded count.

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
- **Broad game semantics.** The core represents deterministic and explicit chance nodes,
  sequential and simultaneous decisions, N-player rewards, and information-state keys for
  imperfect-information algorithms.
- **Experiment lifecycle.** Resolved configurations, composition fingerprints, exact engine
  snapshots, environment forks, structured telemetry, and per-player record routing are
  part of the public surface.
- **Ecosystem adapters.** Supported games expose Gymnasium or PettingZoo adapters where
  those standards fit their interaction model.

## Scope

Reinfors is an orchestration and data-generation library, not a prescribed training stack.
It intentionally does not provide a distributed cluster manager, a universal replay
service, or framework-specific model classes. Those systems can sit behind or around the
inference seam while the Rust engine remains unchanged.

The project is pre-1.0. See [current boundaries](reference/limits.md) before committing to
an experiment design.
