# Operating-point tuning

No head-to-head ran until each side's configuration was selected by measurement on the
target hardware. This page records the selection experiments; the numbers populate as the
final runs are re-executed at publication commit.

## Workload

Chess, AlphaZero-style self-play: their `--nn_width/--nn_depth` resnet family mirrored
layer-for-layer in torch (parameter counts asserted equal at startup), observation encoding
parity-checked position-by-position against pyspiel, 64 simulations per move, matched noise
and temperature. Chess is used because both implementations support it natively and its
action space (4,672 encoded moves) exercises the wide-policy path where boundary costs show.

## Device inputs

Sizing decisions below lean on the [device characterization](../internal/index.md)
measured separately: the kernel batch curve (A10G sweet spot at batch 64 for this net,
regression at 128) and the cache capacity curve.

## Topology selection: states/s under the round workload

The decisive measurement. Each candidate topology runs the **full round workload** — cache
on at its own capacity, learner training and checkpoint writes sharing the GPU — for
windows long enough to contain several learn steps and several game-lengths (20-minute
interior windows after 5-minute warmups). Selection is by completed-game states/s.

OpenSpiel (actors × inference batch):

| config | states/s | rows/s | rows/call | learn steps in window |
|---|---|---|---|---|
| 16 actors | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| 32 actors | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| 64 actors | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| 64 actors, batch 32 (decoupling probe) | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

reinfors (parallel games):

| config | states/s | net rows/s | rows/call | learn steps in window |
|---|---|---|---|---|
| n_games 64 | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| n_games 128 | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

The grids test how actor or game count changes realized rows per call, per-game progress,
and completed states/s. The [design differences](design-differences.md) explain why the two
schedulers may make different trade-offs; the published tables determine whether they do.

## Selected operating points

| | OpenSpiel | reinfors |
|---|---|---|
| topology | _TBD_ | _TBD_ |
| inference batch | _TBD_ | _TBD_ |
| cache capacity | own default | own measured optimum |
| net | w256 d8 (identical) | w256 d8 (identical) |
