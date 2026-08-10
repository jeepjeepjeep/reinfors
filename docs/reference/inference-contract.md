# Inference contract

`Engine` policies and network-backed solvers can both call caller-owned Python networks, but each
execution surface defines its own contract. Engine inference is synchronous for each pooled search
round. The current Deep CFR solver performs synchronous inference within one traversal batch.

## Input

Every callback receives:

```text
obs: float32 NumPy array, shape (rows, flattened_observation_size)
```

Reshape trailing dimensions using `game.observation_space().shape`. Do not assume a
particular contiguity on input; make it contiguous if your framework requires it.

## Engine outputs by policy family

| Policy | Required output |
| --- | --- |
| `EpsilonGreedyQ` | Q values, `(rows, heads, actions)` |
| `Mcts` with TreeStrap | Q values, `(rows, 1, actions)`—one ensemble head |
| `SelectiveExpectimax` | Ensemble Q values, `(rows, heads, actions)` |
| `AlphaZero` | Tuple of policy logits `(rows, width)` where `width >= actions`, and values `(rows,)`; logit columns after `actions` are ignored |

Every output array may be `float32` or `float64`. Returning the network's native `float32` is the
recommended path: reinfors widens it exactly after crossing the boundary and avoids a caller-side
conversion. Outputs must have exactly the requested row count and contain finite values. Q-family
action widths are exact; AlphaZero alone permits a padded policy head and consumes its first
`actions` columns. Constructor settings determine `n_heads` and the action count; see
[head terminology](glossary.md#network-outputs-and-ensembles).

## Deep CFR

`DeepCfr.collect(player=..., traversals=..., infer=...)` queries the current advantage networks.
Each callback has this exact contract:

```text
input:  float32 NumPy array, shape (rows, flattened_observation_size)
output: float32 or float64 NumPy array, shape (rows, actions)
```

The output contains one advantage per action id, with no ensemble-head dimension. It must have the
exact two-dimensional shape and contain finite values; legality is applied inside the solver.

Pass one callable to share an advantage network between players, or a sequence indexed by player.
Networks must remain frozen for the duration of each `collect` call and may be retrained between
calls. `DeepCfr.exploitability(policy_infer=...)` separately expects one `(rows, actions)` policy
probability array, also `float32` or `float64`.

## Legal actions

The callback receives no legal-action mask. Legality belongs to the game and is applied
around the network:

- search normalizes priors or selects values over legal actions;
- environment-side agents must select from `env.legal_actions(agent)`;
- DQN loss code masks `next` maxima using sparse legal ids in the batch.

The model is free to learn low logits for illegal actions, but correctness never depends on
it doing so.

Inference action columns use the encoder's network-head frame. The
[action-frame contract](glossary.md#action-frames) defines how this differs from the game ids used by
`Env`.

## Per-player routing

Pass one callable to share a network, or a sequence indexed by player:

```python
engine.collect(n_records=4096, infer=[infer_0, infer_1, infer_2])
```

Each callback receives only rows for its player perspective. The sequence length must equal
the game player count. Deep CFR uses the same shared-callable or per-player-sequence convention,
with the two-dimensional output described above.

## Concurrency

During `collect`, the callback runs in the calling thread. During `collect_stream`, the
background collector invokes it while Python may train concurrently. Protect mutable model
state and use a stable collector copy. A callback can perform RPC, but it must return before
the pooled search round advances.

## Cache lifetime

Cached outputs are valid only while their network weights are unchanged. The
[cache lifecycle guide](../guides/configuration-and-checkpoints.md#inference-cache-lifecycle) covers
capacity, player routing, concurrent invalidation, telemetry, and snapshot behavior.

For common callback failures and lifecycle errors, see [troubleshooting](troubleshooting.md).
