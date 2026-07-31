# Sampling and training

The central reinfors operation is not a single environment step. It is a request for a
learner-shaped batch produced by many games and, where applicable, many search leaves.

## A collection round

1. Rust advances each eligible game or search until network evaluation is needed.
2. Requests from different games, players, and leaves are pooled into one observation array.
3. The engine calls your inference function once for that pool.
4. Rust routes the returned rows back into their searches and continues them.
5. The learner emits records until the requested batch floor is reached.

The callback overhead is paid per pooled round rather than per environment or tree node.
Increasing `n_games`, search width, or concurrent work can therefore increase accelerator
utilization, subject to the algorithm and game.

## The network remains injectable

Reinfors does not construct, serialize, optimize, or distribute a neural network. The
callback may close over any local model or make a remote request. Different players may use
different callbacks by passing a sequence in player order:

```python
callbacks = [infer_player_0, infer_player_1, infer_player_2]
batch = engine.collect(4096, callbacks)
```

Records include a `players` field, and `learn_players` can suppress records for frozen
opponents while they continue to act.

## Synchronous and concurrent modes

`collect` blocks until the batch is complete. It is the simplest mode, gives an obvious
weight-version boundary, and is often right for debugging or when collection dominates.

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

Reinfors deliberately does not provision processes, services, queues, or cluster membership.
It provides the parallel search and batching primitive; the caller owns deployment and
failure policy.

## Caching and changing weights

`infer_cache` can reuse position evaluations within and across search work. After changing
the corresponding model weights, call:

```python
engine.weights_updated()          # shared callback
engine.weights_updated(player=1)  # one per-player callback
```

The engine invalidates affected cache entries at a safe round boundary. With caching off,
the call is a no-op.
