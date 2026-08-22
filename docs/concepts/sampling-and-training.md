# Sampling and training

The central reinfors operation is not a single environment step. It is a request for a
learner-shaped batch of training [records](../reference/glossary.md#collection) produced by many
games and, where applicable, many search leaves.

## A collection round

1. Rust advances each eligible game or search until network evaluation is needed.
2. Requests from different games, players, and leaves are pooled into one observation array.
3. The engine calls your inference function once for that pool.
4. Rust routes the returned rows back into their searches and continues them.
5. The learner emits records until the requested record floor is reached.

The callback overhead is paid per pooled round rather than per environment or tree node.
Increasing the number of parallel episode slots (`n_games`), search width, or concurrent work can
therefore increase accelerator utilization, subject to the algorithm and game.

## Sizing a run

Start with `n_games=8`, a record floor matching one training update, and—when the policy performs
search—its default search budget. Collect a short run, inspect telemetry, and change one axis at a
time.

`n_games`, the record floor, and search budgets affect different parts of the experiment:

- `n_games` adds independent episode slots. Increase it when callback batches underfill the
  accelerator and CPU or memory headroom remains.
- [Minimax](../catalogue/algorithms.md#minimax)'s `depth` grows the tree with the game's
  *realized* legal branching — roughly `b^depth` leaves for typical branching `b`, evaluated in one
  pooled callback round per ply. Connect 4 sustains full width at the default depth; wide games
  such as chess need `top_k` or a shallower depth, and a run that outgrows the node bound stops
  with `search tree exceeds` rather than allocating without limit.
- `n_records` for `collect`, or `collect_size` for streaming, is the minimum returned training-row
  count. Raise it for fewer, larger training handoffs; completed work may overshoot the floor.
- MCTS and AlphaZero each default `num_simulations` to `64`. Larger values spend more search work
  per decision and may improve action quality at the cost of more tree work and inference.
- [SelectiveExpectimax](../catalogue/algorithms.md#treestrap-selective-expectimax) defaults to `64`
  total node expansions per decision. `top_k=8` expands up to eight highest-priority frontier nodes
  per pooled inference round, while constructor `max_depth=12` stops deeper branches. These are
  search controls; telemetry `max_depth` is the deepest branch actually observed.

For DQN, only `n_games` and the record floor apply; it performs no tree search. Measure mean
callback batch size as `infer_rows / infer_calls` (add `padded_rows` to the numerator when
`pad` is set): if it is small relative to the device's efficient batch size, increase
`n_games` first. Then sweep the search budget against both throughput and task
performance; more simulations are not automatically useful once model quality, latency, or memory
is limiting.

Use `mean_leaves` to confirm how much of the requested search work reaches leaves. Compare
`terminal_sims` and `depthcap_sims` to see whether simulations finish at rules-terminal states or at
the configured depth limit. The telemetry field `max_depth` reports the deepest search actually
observed. Persistent depth-cap pressure is a reason to inspect the horizon and value estimates, not
simply to raise the simulation count. The complete units and policy coverage are in
[telemetry fields](../reference/telemetry-fields.md).

## The network remains injectable

The callback may close over any local model or make a remote request. Different players may use
different callbacks by passing a sequence in player order:

```python
callbacks = [infer_player_0, infer_player_1, infer_player_2]
batch = engine.collect(n_records=4096, infer=callbacks)
```

Records include a `players` field, and `learn_players` can suppress records for frozen
opponents while they continue to act.

## Synchronous and concurrent modes

`collect` blocks until the batch is complete. It is the simplest mode, gives a clear point between
batches at which to update model weights, and is often right for debugging or when collection
dominates.

```text
collect batch 0 → train 0 → collect batch 1 → train 1
```

`collect_stream` runs collection on a background Rust worker and queues completed batches.
Python can train batch 0 while Rust collects batch 1.

```text
Rust:    collect 0 ─ collect 1 ─ collect 2 ─ …
Python:             train 0   ─ train 1   ─ …
```

The caller controls the queue depth and collector-weight synchronization. This makes
staleness a visible experimental choice rather than a hidden runtime policy. See
[concurrent collection](../guides/streaming.md).

## Deployment topologies

The same callback contract supports:

- one process on a CPU-only laptop;
- multithreaded CPU search feeding a local GPU;
- a background collector using a stable model copy while another copy trains;
- per-player models on different devices;
- inference hidden behind RPC, with training and search on separate hosts.

These are caller-composed deployment patterns. Provisioning and failure handling are described by
the [distributed-orchestration boundary](../reference/limits.md#distributed-orchestration); the exact
callback and cache rules live in the [inference contract](../reference/inference-contract.md).

## Next steps

- Build the synchronous path in the [training-loop guide](../guides/training.md).
- Overlap actors and learning with the [concurrent-collection guide](../guides/streaming.md).
- Instrument either path with [telemetry and TensorBoard](../guides/telemetry.md).
