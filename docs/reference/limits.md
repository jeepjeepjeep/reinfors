# Current boundaries

These are deliberate v0 boundaries or algorithm-specific constraints, not hidden fallback
behavior.

- Observations are fixed-shape `float32` arrays and action heads use a fixed global discrete
  vocabulary per game. Variable entities, text observations, continuous actions, and
  parameterized actions need an encoding into that contract or a core extension.
- A game has either sequential decisions or simultaneous decisions throughout. Mixed-phase
  games that switch between both are not currently expressible.
- Native games and engine components are Rust-defined. Python owns inference and training but
  does not implement hot-path `Game`, `Policy`, or `Learner` traits.
- Reinfors provides no built-in distributed cluster runtime. Remote callbacks and external
  actor/trainer orchestration are supported composition patterns, but provisioning, retry,
  discovery, and fault tolerance are caller responsibilities.
- Plain UCT MCTS supports sequential games up to two players and simultaneous N-player games.
  AlphaZero supports sequential N-player backup and simultaneous DUCT. Consult the generated
  [algorithm catalogue](../catalogue/algorithms.md) for the current matrix.
- Engine tree-search policies require perfect information. Imperfect-information planning is
  provided by CFR, external-sampling MCCFR, and Deep CFR; observation-only DQN can also train
  in imperfect-information games without belief-state search.
- CFR-family exact metrics and exact chance enumeration have safety caps. Large games require
  sampling or approximate evaluation.
- Snapshots are opaque continuation artifacts tied to a compatible composition and schema,
  not a permanent cross-project state format.

If one of these boundaries blocks a well-defined research workload, open an issue describing
the game semantics, algorithm, required record/inference shape, and expected scale.
