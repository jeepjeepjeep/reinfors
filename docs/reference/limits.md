# Current boundaries

These are deliberate v0 boundaries or algorithm-specific constraints, not hidden fallback behavior.

## Fixed observation and action spaces

Observations are fixed-shape `float32` arrays and every game exposes one fixed global discrete action
vocabulary. Variable entities and text observations require an encoding into that tensor contract.
Continuous and parameterized actions are not currently expressible without a core extension.

## Decision phases

A game is sequential or simultaneous for its entire episode. Mixed-phase games that switch between
sequential and simultaneous decisions are not currently expressible.

## Native component boundary

Games and engine components run in Rust. Python owns inference and training, but cannot implement
the hot-path `Game`, `Policy`, or `Learner` traits. Adding a game that participates in reinfors search
and collection therefore requires a Rust implementation; see [extending reinfors](../extending/index.md).

## Out-of-tree native composition

Out-of-tree composition is not a supported workflow pre-1.0. The intended stable boundary
is published `reinfors-core` and `reinfors-games` crates plus a documented PyO3 registration
mechanism. Until that boundary is published, contributors should use the in-tree extension path and
expect native contracts and registration glue to change.

## Distributed orchestration

Reinfors provides no built-in cluster runtime. Remote callbacks and external actor/trainer
orchestration are supported composition patterns, but provisioning, retry, discovery, and fault
tolerance are caller responsibilities.

## Algorithm-specific compatibility

Player-count, dynamics, information, and chance constraints vary by algorithm. Consult the generated
[algorithm catalogue](../catalogue/algorithms.md) and
[built-in compatibility matrix](../catalogue/compatibility.md); construction rejects unsupported
compositions.

## Enumeration limits

Exact chance expansion rejects more than 1,048,576 outcomes, simultaneous search rejects more than
1,048,576 joint-action slots, and exact best-response metrics reject trees above 4,000,000 nodes.
Large games require sampling or approximate evaluation.

## Snapshot compatibility

Snapshots are opaque continuation artifacts tied to a compatible composition and schema, not a
permanent cross-project state format.

If one of these boundaries blocks a well-defined research workload, open an issue describing the
game semantics, algorithm, required record/inference shape, and expected scale.
