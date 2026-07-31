# Concurrent collection

`Engine.collect_stream` overlaps Rust data generation with Python training. Use it when
collection and optimization can make useful progress concurrently.

## The two-network pattern

Keep a collector copy stable while the learner copy is updated. Synchronize only between
batches and protect the copy because framework weight loading may release the GIL.

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
        engine.weights_updated()
```

The engine call after the copy invalidates any cached evaluations. With per-player callbacks,
pass the changed `player` to invalidate only that cache partition.

## Queue depth and staleness

`depth` is the maximum number of completed batches waiting for Python:

- `depth=1` allows one finished batch ahead and bounds both memory and policy lag;
- larger values tolerate uneven training time but can train on older policy data;
- `depth=None` is an unbounded continuous-actor topology and requires explicit memory and
  staleness monitoring.

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
have received. `stop()` is appropriate when queued work may be discarded.

## Remote inference

RPC can be implemented inside `infer`, but the callback is synchronous for each pooled
round: it must return before that round can advance. Use the engine's large pooled requests,
multiple collectors, or service-side dynamic batching to hide network latency. Reinfors does
not retry failed remote calls or define cluster membership; those policies belong to the
caller.
