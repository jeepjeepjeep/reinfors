# Batch formats

`Engine.collect` and `CollectStream.next` return `TreeStrapBatch`, `DqnBatch`, `AlphaZeroBatch`, or `PpoBatch`,
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

Target-rule and replay variants are caller-side: the engine records transitions, never target
values or buffer positions, so neither Double DQN nor prioritized replay needs engine support.
For Double DQN, take the masked argmax over the row's `next_` legal slice with the online
network and evaluate that action with the target network. For prioritized replay, keep a
priority per stored row, insert fresh rows at the running max, and write |TD error| back after
each minibatch — the TD errors are already computed caller-side at training time. The
[Hold'em example](../examples/index.md#dqn-holdem)'s `--double` and `--per` flags show both in
ensemble form. Deterministic architecture variants such as dueling heads never touch the seam
at all: the callback contract is Q rows, however they were produced — the example's `--dueling`
flag splits value and advantage streams inside the network. Distributional heads (C51) also
stay caller-side with one adapter: the callback collapses each action's atom distribution to
its expected Q, so the engine selects on scalars while training regresses full distributions —
the example's `--c51` flag shows the projection loss. Stochastic layers (noisy networks)
preserve the callback shape but interact with inference caching: a cached row freezes one noise
realization, so construct with `infer_cache=0` or call `Engine.weights_updated` after
resampling noise, exactly as after a weight sync.

`can_bootstrap` is true exactly when the row's next-legal CSR slice is non-empty. Empty means
terminal or an alternating-game truncation tail, so its target is the immediate reward. Do not compute
`(1 - done) * max(masked_q)`; multiplying zero by negative infinity can produce `NaN`, and
`dones` does not encode every non-bootstrapping tail.

### `PpoBatch`

| Field | Shape | Meaning |
| --- | --- | --- |
| `obs` | `(records, observation_size)` | Acting-player observation |
| `players` | `(records,)` | Acting player id |
| `actions` | `(records,)` | Sampled action ids, encoder head frame |
| `behavior_log_probs` | `(records,)` | Log-probability of the sampled action under the collection-time masked softmax |
| `advantages` | `(records,)` | GAE(`lam`) advantages, seeded by the terminal/truncation tail |
| `returns` | `(records,)` | `advantages + values` — the value-function regression target |
| `values` | `(records,)` | Collection-time critic values, for PPO's clipped value loss |
| `legal_ids`, `legal_offsets` | CSR (compressed sparse row) | Legal actions at each decision, head frame |
| `telemetry` | dictionary | Collection measurements |

Mask the training-time softmax with the same legal CSR before computing new log-probabilities:
an unmasked distribution silently mismatches the recorded behavior log-probs and corrupts the
clipped ratio. Normalize `advantages` per batch or minibatch in the loss, not in the data.

PPO collection is windowed: `collect(n_records)` advances complete rounds under frozen
weights until the usual record floor is met (overshoot is at most one decision per learning
agent per game), then bootstraps every unfinished trajectory from the critic and emits the
fragment, so no record ever spans two collect calls and every batch is internally
single-version. Episodes persist across windows; each window's GAE recursion restarts from its
own bootstrap (the standard truncated-GAE estimator). Sequential non-mover bootstraps query
the critic off-turn — the same approximation the DQN tail accepts. Data is on-policy: run a
few clipped epochs on a batch, discard it, and collect with the updated weights. The recorded
ratio corrects the action-likelihood factor only — not state-visitation staleness — so strict
PPO uses synchronous `collect`; a streamed window can additionally straddle a collector weight
sync, and streamed PPO should be treated as approximately on-policy with coarse syncs.

### `AlphaZeroBatch`

| Field | Shape | Meaning |
| --- | --- | --- |
| `obs` | `(records, observation_size)` | Player-perspective observation |
| `players` | `(records,)` | Perspective/player id |
| `policy_targets` | `(records, actions)` | Root visit distribution |
| `value_targets` | `(records,)` | Realized return-to-go, discounted by `rf.learners.AlphaZero(gamma=...)` (default `1.0`) |
| `policy_weights` | `(records,)` | Whether the row contributes policy loss |
| `legal_ids` | `(legal_nnz,)` | Legal action ids in the encoder's head frame, packed as CSR |
| `legal_offsets` | `(records + 1,)` | CSR offsets delimiting each row's legal ids |

Sequential N-player search emits value-only rows for non-moving perspectives. Their policy
target is inert, `policy_weights` is zero, and their legal slice is empty; every row contributes to
value loss.

For a direct update on the returned batch, densify its CSR legality and mask logits before the
policy softmax:

```python
counts = np.diff(batch.legal_offsets)
rows = np.repeat(np.arange(len(batch.obs)), counts)
legal = np.zeros(batch.policy_targets.shape, dtype=bool)
legal[rows, batch.legal_ids] = True

legal = torch.from_numpy(legal).to(logits.device)
masked_logits = logits.masked_fill(~legal, torch.finfo(logits.dtype).min)
```

Use `torch.finfo(logits.dtype).min`, not a fixed large negative constant: a value such as `-1e9`
overflows `float16`. Multiply the per-row policy loss by `policy_weights`; rows with empty legality
slices are value-only and must not contribute to the policy reduction. When sampling from replay,
gather the selected rows' CSR slices and allocate only `(minibatch_size, actions)` rather than a
dense mask for the full buffer.

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
