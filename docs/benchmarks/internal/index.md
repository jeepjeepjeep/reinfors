# reinfors internals

reinfors measured against itself: what the device can do, and what each configuration
lever buys on top. Nothing here compares frameworks — these are the measurements that
guide configuration choices for your own workload, and the calibration curves the
[OpenSpiel comparison](../openspiel/index.md) leans on when explaining its results.

## Device characterization

The pure-forward kernel ceiling of the benchmark network (w256 d8, A10G), measured
outside any engine at the operating-point batch, is the ceiling every configuration
approaches:

| | rows/s | µs/row |
|---|---|---|
| pure forward, batch 64 | 15.2k | 65.8 |

The batch-size curve that drives sizing decisions (engine batch, grouped-collection
group size) was measured end-to-end through the engine (120-second internally timed
legs, cache off — a within-stack comparison) — same shape, lower absolute:

| batch (`n_games`) | rows/s (engine, A10G, w256 d8) |
|---|---|
| 32 | 10,952 |
| 64 | **12,367** |
| 128 | 11,620 |

The A10G's sweet spot for this net sits at batch 64, with a measurable per-row
*regression* at batch 128. Batch curves are strongly device- and net-dependent (the same
sweep on Apple silicon keeps improving well past 100 rows) — measure yours before sizing
anything against it.

Inference-cache hit rate versus capacity (chess self-play, early-training net):

| capacity | hit rate |
|---|---|
| 4,096 | 6.6% |
| 32,768 | 13.6% |
| 262,144 | 14.1% |
| 2M | 14.5% |

Hit rate is monotone in capacity but flattens sharply, and — worth knowing — *rises as
training progresses*: a stronger net concentrates its own play (the same 262,144-entry
configuration reaches 26–27% after two hours of training in the
[matched round](../openspiel/matched-round.md)). Capacity is a throughput choice unless
host memory binds.

## Configuration levers

[Throughput levers](throughput-levers.md) measures the individual opt-in features — native
f32 callback outputs, the inference cache, and grouped collection (`n_groups`) — each with
its effect at a reference operating point and the model predicting where it transfers.
