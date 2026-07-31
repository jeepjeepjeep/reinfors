# Inference contract

Inference is the only Python callback used by engine search. It is synchronous for each
pooled round and may be a callable shared by every player or a sequence of per-player
callables.

## Input

Every callback receives:

```text
obs: float32 NumPy array, shape (rows, flattened_observation_size)
```

Reshape trailing dimensions using `game.observation_space().shape`. Do not assume a
particular contiguity on input; make it contiguous if your framework requires it.

## Outputs by policy family

| Policy | Required output |
| --- | --- |
| `EpsilonGreedyQ` | Q values, `float64`, `(rows, heads, actions)` |
| `Mcts` with TreeStrap | Q values, `float64`, `(rows, 1, actions)` |
| `SelectiveExpectimax` | Ensemble Q values, `float64`, `(rows, heads, actions)` |
| `AlphaZero` | Tuple of policy logits `(rows, actions)` and values `(rows,)`, both `float64` |
| `DeepCfr` | Per-player advantage predictions in the solver-documented action shape |

Outputs must have exactly the requested row count and contain finite values. Constructor
settings determine the number of heads and actions.

## Legal actions

The callback receives no legal-action mask. Legality belongs to the game and is applied
around the network:

- search normalizes priors or selects values over legal actions;
- environment-side agents must select from `env.legal_actions(agent)`;
- DQN loss code masks `next` maxima using sparse legal ids in the batch.

The model is free to learn low logits for illegal actions, but correctness never depends on
it doing so.

## Per-player routing

Pass one callable to share a network, or a sequence indexed by player:

```python
engine.collect(n_records=4096, infer=[infer_0, infer_1, infer_2])
```

Each callback receives only rows for its player perspective. The sequence length must equal
the game player count. The same form applies to Deep CFR's player-specific advantage models.

## Concurrency

During `collect`, the callback runs in the calling thread. During `collect_stream`, the
background collector invokes it while Python may train concurrently. Protect mutable model
state and use a stable collector copy. A callback can perform RPC, but it must return before
the pooled search round advances.

If an inference cache is enabled, call `engine.weights_updated()` after changing weights.
