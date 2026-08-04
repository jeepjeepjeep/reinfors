# Batch formats

`Engine.collect` and `CollectStream.next` return `TreeStrapBatch`, `DqnBatch`, or `AlphaZeroBatch`,
according to the engine's learner. Use named fields; positional unpacking exists only for
compatibility.

`n_records` and streaming `collect_size` are record floors, not exact sizes. Completed episode or
search work can make a returned batch larger than requested; see the
[record-floor definition](glossary.md#collection).

All observations are flattened `float32` rows. Targets are `float64` unless noted.
Shape terms and sparse encodings are defined in the [glossary](glossary.md).

Every action-indexed Engine field uses the selected encoder's network-head frame. See
[action frames](glossary.md#action-frames) before moving actions between batches, networks, and
`Env`.

## Engine collection

### `TreeStrapBatch`

| Field | Shape | Meaning |
| --- | --- | --- |
| `obs` | `(records, observation_size)` | Acting/player-perspective observation |
| `players` | `(records,)` | Perspective/player id |
| `targets` | `(records, heads, actions)` | Searched action targets for each ensemble head |
| `masks` | `(records, heads)` | Record/head bootstrap membership, `float32` |
| `telemetry` | dictionary | Collection measurements |

Compute an action loss for each record/head pair, multiply it by `masks`, and reduce over the
included pairs. A zero entry means that record is outside that head's bootstrap sample.

### `DqnBatch`

| Field | Shape | Meaning |
| --- | --- | --- |
| `obs`, `next_obs` | `(records, observation_size)` | Transition endpoints |
| `players` | `(records,)` | Record perspective |
| `actions` | `(records,)` | Chosen action ids |
| `rewards` | `(records,)` | Immediate scalar rewards |
| `dones` | `(records,)` | Episode-ended flag, not the bootstrap rule |
| `can_bootstrap` | `(records,)` | Whether the TD target includes a next-state value |
| `masks` | `(records, heads)` | Record/head bootstrap membership, `float32` |
| `legal_ids`, `legal_offsets` | CSR (compressed sparse row) | Current legal actions |
| `next_legal_ids`, `next_legal_offsets` | CSR (compressed sparse row) | Next legal actions |

For record `i`, its current legal ids are
`legal_ids[legal_offsets[i]:legal_offsets[i + 1]]`; use the corresponding `next_` arrays for its
successor. This is CSR storage without a dense action mask.

For Q output shaped `(records, heads, actions)`, gather each record's chosen action, compute one TD
loss per head, multiply by `masks`, and reduce over non-zero entries.

`can_bootstrap` is true exactly when the row's next-legal CSR slice is non-empty. Empty means
terminal or an alternating-game truncation tail, so its target is the immediate reward. Do not compute
`(1 - done) * max(masked_q)`; multiplying zero by negative infinity can produce `NaN`, and
`dones` does not encode every non-bootstrapping tail.

### `AlphaZeroBatch`

| Field | Shape | Meaning |
| --- | --- | --- |
| `obs` | `(records, observation_size)` | Player-perspective observation |
| `players` | `(records,)` | Perspective/player id |
| `policy_targets` | `(records, actions)` | Root visit distribution |
| `value_targets` | `(records,)` | Realized return-to-go, discounted by `rf.learners.AlphaZero(gamma=...)` (default `1.0`) |
| `policy_weights` | `(records,)` | Whether the row contributes policy loss |

Sequential N-player search emits value-only rows for non-moving perspectives. Their policy
target is inert and `policy_weights` is zero; every row contributes to value loss.

## Solver collection

### `DeepCfrBatch`

`DeepCfrBatch` is returned only by `rf.solvers.DeepCfr.collect`, not by `Engine` or `CollectStream`.
It contains two independently sized sample streams:

| Field | Shape | Meaning |
| --- | --- | --- |
| `advantage_obs` | `(advantage_records, observation_size)` | Information-state observations |
| `advantage_iterations` | `(advantage_records,)` | CFR iteration used for loss weighting |
| `advantage_legal_offsets` | `(advantage_records + 1,)` | CSR row offsets |
| `advantage_legal_ids` | `(advantage_nnz,)` | Legal action ids |
| `advantage_targets` | `(advantage_nnz,)` | Advantages aligned with legal ids |
| `strategy_obs` | `(strategy_records, observation_size)` | Average-policy observations |
| `strategy_iterations` | `(strategy_records,)` | CFR iteration used for sampling/weighting |
| `strategy_players` | `(strategy_records,)` | Acting player for each strategy row |
| `strategy_legal_offsets` | `(strategy_records + 1,)` | CSR row offsets |
| `strategy_legal_ids` | `(strategy_nnz,)` | Legal action ids |
| `strategy_probs` | `(strategy_nnz,)` | Action probabilities aligned with legal ids |
| `telemetry` | dictionary | Traversal and inference measurements |

Targets and probabilities are packed rather than dense: slice row `i` from
`offsets[i]:offsets[i + 1]`. Buffers, reservoir sampling, iteration weighting, networks, and
optimization remain caller-owned.

## Next steps

- Implement the corresponding loss in the [training guide](../guides/training.md).
- Match model outputs to the [inference contract](inference-contract.md).
- Diagnose rejected arrays with [troubleshooting](troubleshooting.md).
