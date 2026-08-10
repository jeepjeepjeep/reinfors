# reinfors internals

reinfors measured against itself: what the device can do, and what each configuration
lever buys on top. Nothing here compares frameworks — these are the measurements that
guide configuration choices for your own workload, and the calibration curves the
[OpenSpiel comparison](../openspiel/index.md) leans on when explaining its results.

## Device characterization

The pure-forward kernel ceiling of the benchmark network, measured outside any engine —
the ceiling every configuration approaches and the batch-size curve that drives sizing
decisions everywhere else (engine batch, grouped-collection group size):

| batch | rows/s (A10G, chess net w256 d8) | µs/row |
|---|---|---|
| 32 | _TBD_ | _TBD_ |
| 64 | _TBD_ | _TBD_ |
| 128 | _TBD_ | _TBD_ |

The A10G's sweet spot for this net sits at batch 64, with a measurable per-row
*regression* at batch 128. Batch curves are strongly device- and net-dependent (the same
sweep on Apple silicon keeps improving well past 100 rows) — measure yours before sizing
anything against it.

Inference-cache hit rate versus capacity (chess self-play):

| capacity | hit rate |
|---|---|
| 4,096 | _TBD_ |
| 32,768 | _TBD_ |
| 262,144 | _TBD_ |
| 2M | _TBD_ |

Hit rate is monotone in capacity but flattens sharply, and — worth knowing — *rises as
training progresses*: a stronger net concentrates its own play. Capacity is a throughput
choice unless host memory binds.

## Configuration levers

[Throughput levers](throughput-levers.md) measures the individual opt-in features — native
f32 callback outputs, the inference cache, and grouped collection (`n_groups`) — each with
its effect at a reference operating point and the model predicting where it transfers.
