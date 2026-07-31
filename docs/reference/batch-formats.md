# Batch formats

`Engine.collect` and `CollectStream.next` return a learner-specific object. Use named fields;
positional unpacking exists only for compatibility.

All observations are flattened `float32` rows. Targets are `float64` unless noted.

## `TreeStrapBatch`

| Field | Shape | Meaning |
| --- | --- | --- |
| `obs` | `(records, observation_size)` | Acting/player-perspective observation |
| `players` | `(records,)` | Perspective/player id |
| `targets` | `(records, heads, actions)` | Searched per-head action targets |
| `masks` | `(records, heads)` | Bootstrap membership, `float32` |
| `telemetry` | dictionary | Collection measurements |

Compute a per-head action loss and weight each head by `masks`.

## `DqnBatch`

| Field | Shape | Meaning |
| --- | --- | --- |
| `obs`, `next_obs` | `(records, observation_size)` | Transition endpoints |
| `players` | `(records,)` | Record perspective |
| `actions` | `(records,)` | Chosen action ids |
| `rewards` | `(records,)` | Immediate scalar rewards |
| `dones` | `(records,)` | Episode-ended flag, not the bootstrap rule |
| `masks` | `(records, heads)` | Bootstrap membership |
| `legal_ids`, `legal_offsets` | CSR | Current legal actions |
| `next_legal_ids`, `next_legal_offsets` | CSR | Next legal actions |

Bootstrap a row exactly when its next-legal CSR slice is non-empty. Empty means terminal or
an alternating-game truncation tail, so its target is the immediate reward. Do not compute
`(1 - done) * max(masked_q)`; multiplying zero by negative infinity can produce `NaN`, and
`dones` does not encode every non-bootstrapping tail.

## `AlphaZeroBatch`

| Field | Shape | Meaning |
| --- | --- | --- |
| `obs` | `(records, observation_size)` | Player-perspective observation |
| `players` | `(records,)` | Perspective/player id |
| `policy_targets` | `(records, actions)` | Root visit distribution |
| `value_targets` | `(records,)` | Discounted realized return |
| `policy_weights` | `(records,)` | Whether the row contributes policy loss |

Sequential N-player search emits value-only rows for non-moving perspectives. Their policy
target is inert and `policy_weights` is zero; every row contributes to value loss.

## `DeepCfrBatch`

Deep CFR returns separate advantage and average-strategy samples. Each side includes
observations, iteration numbers, CSR legal-action ids/offsets, and packed targets or
probabilities. Strategy samples additionally carry `strategy_players`. Buffers, reservoir
sampling, iteration weighting, networks, and optimization remain caller-owned.
