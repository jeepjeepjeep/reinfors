# Extending reinfors

Reinfors is modular at the Rust composition boundary. Native components implement the contracts in
`reinfors-core`, and PyO3 constructors expose compositions to Python. See the
[native component boundary](../reference/limits.md#native-component-boundary) for the deliberate v0
split between Rust components and Python-owned inference and training.

## Before you start

Complete the [development setup](../development/setup.md) first. These guides assume basic Rust
familiarity and a working native build; they are not an introduction to Rust or PyO3.

## Choose the smallest extension

- A different observation or action-head view of an existing game is usually an
  [`Encoder`](encoders.md).
- Different scalar objectives over existing events are usually a `Reward`.
- A new action-selection/search procedure is a `Policy`.
- A new record schema or trajectory target is a `Learner`.
- An algorithm that owns its traversal and persistent tables/buffers is usually a solver. This is
  an architectural category rather than a shared Rust trait or Python registry.
- New rules, state, legal actions, actors, or chance distributions require a `Game`.

For an alternative view of an existing game, follow [Add an encoder](encoders.md). For a new game,
follow [Add a game](rust-components.md), then
[register its Python binding](python-bindings.md#register-a-built-in-game). For a new policy or
learner, follow [Add a policy or learner](policies-and-learners.md). Reward, chance-mode, and
noise authors should continue through the
[native component contracts](component-contracts.md), then the
[handle-based binding path](python-bindings.md#other-handle-based-components). Solver authors should
start with the [solver contract](component-contracts.md#solvers).

## Compatibility questions to answer first

Every contribution should state:

- player-count range;
- sequential or simultaneous decision dynamics;
- perfect or imperfect information;
- chance-node behavior and fan-out;
- observation shape and action vocabulary;
- compatible policies, learners, and solvers;
- truncation behavior and reward events;
- serialization/snapshot support;
- adapter support, if a standard API matches.

These facts belong in the machine-readable catalogue or constructor configuration rather
than in prose counts scattered across the site.

## Definition of done

A new component should include constructor validation, focused Rust tests, Python registry
and type-stub updates, a rebuilt native extension, composition tests, catalogue metadata, and one
concise usage example.
Games should additionally test legality, terminal behavior, chance distributions, state
codec round trips, information-state identity where applicable, and adapter compliance.
