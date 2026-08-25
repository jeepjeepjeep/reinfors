# Concepts

Concept pages explain why reinfors divides responsibilities as it does. They are optional context
for the first example and useful when choosing an experiment architecture.

- [Architecture](architecture.md) maps games, encoders, policies, learners, solvers, `Engine`, and
  `Env` onto their Rust and Python responsibilities.
- [Sampling and training](sampling-and-training.md) follows one collection round, explains pooled
  inference, and connects `n_games` and search budgets to hardware utilization.
- [Engine collection internals](engine-collection.md) shows the runtime: thread roles, the
  scheduler/worker message flow, and where throughput is bound today.

When the model is clear, put it into practice with the [training guide](../guides/training.md).
