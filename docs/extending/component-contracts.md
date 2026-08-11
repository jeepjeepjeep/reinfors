# Native component contracts

The extension traits live in `crates/reinfors-core/src`; built-in implementations live in
`crates/reinfors-games/src` and the corresponding core algorithm modules.

| Component | Contract | Good first implementation |
| --- | --- | --- |
| Game | `crates/reinfors-core/src/game.rs` | `crates/reinfors-games/src/gridworld.rs` |
| Encoder/action view | `crates/reinfors-core/src/encoder.rs` | `GridWorldPlanes` in `gridworld.rs` |
| Reward | `crates/reinfors-core/src/reward.rs` | `GridWorldReward` in `gridworld.rs` |
| Policy | `crates/reinfors-core/src/policy.rs` | `crates/reinfors-core/src/policies/modelfree/epsilon_greedy_q.rs` |
| Learner | `crates/reinfors-core/src/learner.rs` | `crates/reinfors-core/src/learners/dqn.rs` |
| Solver | No common trait | `crates/reinfors-core/src/solvers/cfr.rs` |

## `Game`

`Game` is the authoritative rules state machine: state, actors, legal actions, transitions, explicit
chance, terminal status, and information-state identity. Accepted states must be safe for every game
method, and each transition emits only the events causally determined on that edge. Prefer chains of
narrow chance nodes when outcomes are naturally sequential.

Use the [worked game walkthrough](rust-components.md) for the complete implementation and test path.

## `StateEncoder` and `ActionView`

`StateEncoder` maps `(state, agent)` to a fixed-shape `float32` observation. Its `ActionView`
supertrait maps canonical game-action ids into the network-head frame. Perspective-relative
implementations may transform both; test the action mapping with `reinfors_core::check_action_view`.
The [worked encoder walkthrough](encoders.md) adds an alternative observation and non-identity
action view to an existing game.

## `Reward`

`Reward` maps ordered transition events to one scalar per agent. Keep rule outcomes in events and
experimental weighting in reward configuration so an objective can change without duplicating game
dynamics.

## `Policy`

`Policy` owns evaluation and action selection: exploration, search state, backups, chance handling,
legal-action masking, and inference requests. Its capability methods declare supported player
counts, information models, and decision dynamics.

## `Learner`

`Learner<E>` converts policy evaluations and finished trajectories into typed records. Document a
new record's shape, dtype, legal-action treatment, and terminal/truncation behavior.

## `Engine` and `Env`

`Engine` composes games, encoders, rewards, policies, and learners; owns active episodes and RNGs;
batches inference; and produces training records. `Env` exposes one caller-driven game and resolves
chance between decision boundaries. New components should satisfy these generic contracts rather
than add algorithm-specific runtime branches.

## Solvers

A solver owns a traversal that does not fit policy-driven episode collection. Current
implementations are CFR-based and live under `crates/reinfors-core/src/solvers`; there is no shared
`Solver` trait or generic Python registry. Follow the
[solver binding path](python-bindings.md#solvers-are-different).

## Validation boundaries

Validate user configuration in constructors and Python factories. Use debug assertions for
invariants guaranteed by valid internal transitions. Snapshot decoding checks schema, lengths,
discriminants, and safety preconditions, but does not replay a game to prove reachability.

## Next steps

- Add a game with the [end-to-end walkthrough](rust-components.md).
- Add an alternative view with the [encoder walkthrough](encoders.md).
- Expose policies, learners, encoders, rewards, chance modes, and noise through the
  [handle-based binding path](python-bindings.md#other-handle-based-components).
- Run the repository gates from [development setup](../development/setup.md).
