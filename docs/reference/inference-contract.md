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

Row counts vary call to call: the scheduler fires at `batch_size` rows but drains
smaller batches when no search can progress, and cache hits, in-batch deduplication
and terminal simulations all remove rows. Fixed-shape consumers (compiled or
graph-captured forwards, XLA) can set `rf.Engine(..., pad=True)`: every call then
carries exactly `batch_size` rows — short batches are padded with zero rows, and
pad outputs are discarded. Telemetry reports pad rows as `padded_rows`;
`infer_rows` keeps counting real rows. Padding requires a single shared callback,
and the no-op guarantee assumes the callback's outputs are row-independent —
evaluation-mode networks, no batch-coupled statistics — as the contract already
requires for caching.

## Engine outputs by policy family

| Policy | Required output |
| --- | --- |
| `EpsilonGreedyQ` | Q values, `(rows, heads, actions)` |
| `Mcts` with TreeStrap | Q values, `(rows, 1, actions)`—one ensemble head |
| `SelectiveExpectimax` | Ensemble Q values, `(rows, heads, actions)` |
| `Minimax` | Q values, `(rows, 1, actions)`—one ensemble head. Every frontier row is requested for the leaf's own mover (that player id appears in per-player routing) and collapsed by the masked max over its legal set; opponent horizons negate back to the searcher under the zero-sum contract, so a self-play network only ever serves on-turn inputs — the same distribution the paired TreeStrap learner trains |
| `AlphaZero` | Tuple of policy logits `(rows, width)` where `width >= actions`, and values `(rows,)`; logit columns after `actions` are ignored |
| `Ppo` | Same tuple contract as `AlphaZero`: policy logits `(rows, width)` with `width >= actions` (tail ignored) and values `(rows,)` — one actor-critic forward serves both |

Every output array may be `float32` or `float64`. Returning the network's native `float32` is the
recommended path: reinfors widens it exactly after crossing the boundary and avoids a caller-side
conversion. Outputs must have exactly the requested row count and contain finite values. Q-family
action widths are exact; AlphaZero and PPO permit a padded policy head and consume its first
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
calls. `DeepCfr.exploitability(policy_infer=...)` separately expects one `(rows, actions)` score
array, also `float32` or `float64`. It clamps negative scores to zero and renormalizes over legal
actions, using a uniform legal policy when a row has no positive mass.

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

During `collect`, the callback runs in the calling thread — search rounds may fan
across `n_threads` workers, but the callback itself is always invoked from the
scheduler's thread. During `collect_stream`, the background collector invokes it while Python may train
concurrently, and the invoking thread is **fixed for the stream's lifetime**: every call across
every batch arrives on one persistent thread, so thread-affine callbacks (a
`torch.compile(mode="reduce-overhead")` forward, whose cudagraph state is thread-local) work
under a stream. The guarantee covers only invocations made *through* the stream — state a
callback captured on another thread beforehand is not repaired. Warm a compiled callable via
the stream itself (e.g. discard the first batch) or construct it lazily inside the callback;
never invoke it on the caller thread first. One-shot `collect` calls give no cross-call
thread guarantee.
Protect mutable model state and use a stable collector copy. A callback can
perform RPC, but the requesting group's search blocks until it returns.

## Cache lifetime

Cached outputs are valid only while their network weights are unchanged. The
[cache lifecycle guide](../guides/configuration-and-checkpoints.md#inference-cache-lifecycle) covers
capacity, player routing, concurrent invalidation, telemetry, and snapshot behavior.

For common callback failures and lifecycle errors, see [troubleshooting](troubleshooting.md).
