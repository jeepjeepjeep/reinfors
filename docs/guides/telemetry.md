# Telemetry and TensorBoard

Every collected batch carries a plain Python `telemetry` dictionary. It has no dependency on
a logging framework, so you can send it to TensorBoard, Weights & Biases, MLflow, a database,
or structured logs.

## Read a batch

```python
batch = engine.collect(2048, infer)
t = batch.telemetry

print("inference rows:", t["infer_rows"])
print("inference calls:", t["infer_calls"])
print("completed episodes:", len(t["episodes"]))
```

Counters and timings describe that collection call, not the engine's lifetime. This makes
them safe to aggregate under the experiment step you choose.

## TensorBoard integration

TensorBoard's writer is optional and remains outside reinfors:

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("runs/experiment-001")

for update in range(num_updates):
    batch = engine.collect(2048, infer)
    loss = train(batch)
    t = batch.telemetry

    writer.add_scalar("train/loss", loss, update)
    writer.add_scalar("sampling/infer_rows", t["infer_rows"], update)
    writer.add_scalar("sampling/infer_calls", t["infer_calls"], update)
    writer.add_scalar("sampling/infer_seconds", t["infer_seconds"], update)

    if t["infer_calls"]:
        writer.add_scalar(
            "sampling/rows_per_infer_call",
            t["infer_rows"] / t["infer_calls"],
            update,
        )

    episodes = t["episodes"]
    if episodes:
        mean_length = sum(length for _returns, length, _seeded in episodes) / len(episodes)
        writer.add_scalar("episodes/mean_length", mean_length, update)
```

The complete runnable version is `examples/telemetry_tensorboard.py`.

## Useful derived measurements

- `infer_rows / infer_calls`: mean callback batch size;
- `infer_rows / infer_seconds`: callback row throughput (includes the callback as observed by
  Rust, not training time);
- `cache_hits / cache_lookups`: evaluation-cache hit rate;
- episode return and length distributions from `episodes`;
- `fresh_rows`, `hit_rows`, and `shared_rows`: where evaluated rows came from;
- `terminal_sims / (terminal_sims + depthcap_sims)`: how searches ended, when applicable.

Do not compare search configurations on a single throughput number alone. Record resolved
configuration, hardware, build profile, model, callback batch sizes, and episode/search
quality metrics together.

See the [telemetry field reference](../reference/telemetry-fields.md) for definitions.
