# Concurrent collection

`Engine.collect_stream` overlaps Rust data generation with Python training. Use it when
collection and optimization can make useful progress concurrently.

The [collection-round model](../concepts/sampling-and-training.md#a-collection-round) explains why
parallel games and search leaves pool into callback batches; streaming overlaps repeated rounds
with optimization.

The stream borrows the engine for its lifetime. Calling `engine.collect()` or starting another
stream while it is active raises an error. `stop()`, `pause()`, or context-manager exit returns the
engine.

> **Warning:** Never abandon a stream without stopping it. The engine then remains unavailable
> for the rest of the process: it cannot collect, snapshot, restore, or start another stream.
> Always use the `with` form unless lifecycle ownership is handled explicitly.

`collect_size` names the requested [record floor](../reference/glossary.md#collection) for each
worker-produced batch; the synchronous spelling is `n_records`.

## Run the minimal example

The maintained example uses a NumPy callback and placeholder Python-side work so the stream
lifecycle is visible without installing a training framework:

```bash
python examples/streaming.py --updates 3
```

It demonstrates bounded background collection, consuming completed batches, inspecting the queue,
and context-managed shutdown. See the [streaming example entry](../examples/index.md#streaming) for its
dependencies and the synchronous counterpart.

## The two-network pattern

For real concurrent training, keep a collector copy stable while the learner copy is updated.
Synchronize only between batches and protect the copy because framework weight loading may release
the GIL. The following focuses on that pattern; `make_network`, `run_inference`, and `train` are
application-specific placeholders.

```python
import copy
import threading

learner_net = make_network()
collector_net = copy.deepcopy(learner_net)
weights_lock = threading.Lock()

def infer(obs):
    with weights_lock:
        return run_inference(collector_net, obs)

with engine.collect_stream(collect_size=2048, infer=infer, depth=1) as stream:
    for update in range(num_updates):
        batch = next(stream)
        train(learner_net, batch)

        with weights_lock:
            collector_net.load_state_dict(learner_net.state_dict())
            engine.weights_updated()  # Invalidate cached rows for the shared callback.
```

`weights_updated()` is a no-op when inference caching is disabled, so keeping it beside every weight
sync makes the loop safe if `infer_cache` is enabled later. With per-player callbacks, use
`weights_updated(player=...)` for the changed network. See the
[cache lifecycle guide](configuration-and-checkpoints.md#inference-cache-lifecycle).

## Queue depth and staleness

`depth` is the maximum number of completed batches waiting for Python:

- `depth=1` allows one finished batch ahead and bounds both memory and policy lag;
- larger values tolerate uneven training time but can train on older policy data;
- `depth=None` is an unbounded continuous-actor topology and requires explicit memory and
  staleness monitoring.

`stream.pending()` reports the number of completed batches currently waiting in the queue. It is an
advisory snapshot—the worker may have another batch in flight—but is the direct signal for queue
growth and sustained producer/consumer imbalance.

The stream has one consumer. The thread calling `next()` should also own shutdown; use the
context-manager form so exceptions stop the worker and return the engine.

## Checkpointing without losing records

Calling `stream.pause()` stops new collection, finishes in-flight work, and returns every
queued batch. Consume or persist those batches, then snapshot the engine:

```python
remaining = stream.pause()
for batch in remaining:
    replay.add(batch)

snapshot = engine.snapshot(policy_version=model_version)
```

This is a lossless barrier: the returned engine state follows exactly from the batches you
have received. `pause()` permanently ends that stream handle; a second `pause()` is an error. Resume
collection by creating a new stream from the returned engine:

```python
with engine.collect_stream(collect_size=2048, infer=infer, depth=1) as resumed_stream:
    next_batch = next(resumed_stream)
```

`stop()` is appropriate when queued work may be discarded. See
[configuration and checkpoints](configuration-and-checkpoints.md) for persistence and restoration.

## Remote inference

RPC can be implemented inside `infer`, but the callback is synchronous for each pooled
round: it must return before that round can advance. Use the engine's large pooled requests,
multiple collectors, or service-side dynamic batching to hide network latency. Reinfors does
not retry failed remote calls or define cluster membership; those policies belong to the
caller.

## Next steps

- Inspect queue pressure with `stream.pending()` and callback throughput with
  [telemetry](telemetry.md). Reinfors does not emit a policy-lag field; record the model version used
  for each collected batch in your training loop when exact staleness matters.
- Pause and persist a run with [configuration and checkpoints](configuration-and-checkpoints.md).
- Evaluate synchronized model versions with the [evaluation guide](evaluation.md).
