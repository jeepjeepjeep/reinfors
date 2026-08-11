# Guides

These guides move from a first synchronous update to production experiment concerns. Start with the
training loop, then follow the path relevant to your run:

1. [Build a training loop](training.md).
2. Add [telemetry and TensorBoard](telemetry.md).
3. Overlap sampling and optimization with [concurrent collection](streaming.md).
4. Persist the complete run with [configuration and checkpoints](configuration-and-checkpoints.md).
5. Compare trained agents through [evaluation](evaluation.md) and run concurrent searched matches
   with [Arena](arena.md).

Use the [Gymnasium and PettingZoo adapters](adapters.md) when downstream tooling expects an ecosystem
environment API instead of reinfors' native `Env` or batched `Engine`.

Before tuning parallelism or search width, read the five-step
[collection round](../concepts/sampling-and-training.md#a-collection-round), then use
[sizing a run](../concepts/sampling-and-training.md#sizing-a-run) to choose `n_games`, record floors,
and search budgets.
