# Extending reinfors

Reinfors is modular at the Rust composition boundary. Downstream Rust crates can implement
new components against `reinfors-core`, reuse games from `reinfors-games`, and register PyO3
constructors for Python composition.

Python-defined game, policy, or learner implementations are not a v0 goal. A user who needs
Python-side rules can build an environment directly; reinfors' value is keeping the hot
simulation and search path native.

## Choose the smallest extension

- A different observation or action-head view of an existing game is usually an `Encoder`.
- Different scalar objectives over existing events are usually a `Reward`.
- A new action-selection/search procedure is a `Policy`.
- A new record schema or trajectory target is a `Learner`.
- An algorithm that owns its traversal and persistent tables/buffers is usually a `Solver`.
- New rules, state, legal actions, actors, or chance distributions require a `Game`.

Read the [Rust component guide](rust-components.md), then the [Python binding guide](python-bindings.md).

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
and type-stub updates, composition tests, catalogue metadata, and one concise usage example.
Games should additionally test legality, terminal behavior, chance distributions, state
codec round trips, information-state identity where applicable, and adapter compliance.
