# Rust components

The public native composition traits live in `reinfors-core`; built-in implementations live
in `reinfors-games` and the policy, learner, engine, or solver modules. Read the trait source
and a small existing implementation together—the contracts below explain ownership, not
every method signature.

## `Game`

`Game` is the authoritative rules state machine. It defines initial state, player count,
decision dynamics, active agents, legal actions, action transitions, explicit chance
distributions and chance transitions, terminal state, and information-state keys.

Important invariants:

- every accepted state must be safe for all game methods;
- legal actions and active agents must agree with the current node kind;
- chance probabilities must be finite, non-negative, and normalized as required by the
  chance distribution type;
- each transition emits only the events causally determined on that edge;
- initialization resolves chance until it reaches a non-terminal decision root;
- the state codec is a trusted round-trip boundary with structural decoding checks, not a
  second implementation of all game rules.

Chance should be represented as a chain of narrow nodes when outcomes are naturally
sequential. Avoid eagerly combining independent draws into a combinatorial fan unless the
algorithm genuinely needs that joint distribution.

## `StateEncoder` and `ActionView`

The Python surface calls this composition an encoder. In Rust, `StateEncoder` maps
`(state, agent)` to a fixed-shape `f32` observation and its `ActionView` supertrait maps game
action ids to network-head indices. Perspective-relative implementations may transform both
observations and actions. The game enables any history bookkeeping required by the selected
encoder.

Encoding must not reconstruct hidden authoritative state that transitions depend on. It may
derive observation features from state, including private information visible to the agent.

## `Reward`

A reward maps the ordered per-edge events emitted during a tick to one scalar per agent.
Keep rule outcomes in events and experimental weighting in reward configuration. Validate
unknown keys and non-finite weights at construction.

## `Policy`

A policy advances a decision using evaluations supplied through the engine. It owns search
state and selection semantics: exploration, backups, chance treatment, legal-action masking,
and action choice. It must declare composition constraints early—for example, whether it
supports imperfect information, simultaneous nodes, player counts, or a chance mode.

## `Learner`

A learner observes policy/episode state and emits its typed training records. It defines the
semantic connection between search and loss: DQN transitions, TreeStrap action targets, or
AlphaZero visit and return targets. Record rows carry the player perspective.

Keep the inference output contract associated with the policy/learner family. New records
must have a documented shape, dtype, legal-action rule, and terminal/truncation rule.

## `Engine`

The engine composes the above components, owns active episodes and RNGs, pools evaluation
requests, routes per-player callbacks, maintains the optional inference cache, and assembles
Python batches. A new component should normally integrate through existing engine traits,
not add an algorithm-specific branch at the Python boundary.

## `Env`

`Env` exposes one game to a caller. It resolves explicit chance internally between caller
decision boundaries and returns the ordered event trace from the full tick. It is also the
foundation for Gymnasium and PettingZoo adapters.

## `Solver`

A solver owns a traversal that does not fit policy-driven episode collection. CFR owns
regret and average-strategy tables; Deep CFR owns traversal iteration while inference and
training buffers remain external. A solver should expose resolved configuration, iteration
state, metrics with clear enumeration limits, and a serializable continuation boundary when
applicable.

## Validation layers

Validate user configuration at constructors and Python boundaries. Use debug assertions for
internal invariants that valid transitions guarantee. Snapshot decoding should verify schema,
composition, lengths, discriminants, and memory-safety preconditions; it need not prove
reachability by replaying the game.
