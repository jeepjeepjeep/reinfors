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

## Kernel ceiling

Pure-forward throughput of the *identical* network, outside either engine — the ceiling any
configuration can approach, and the source of the batch-size curve used everywhere below.

| batch | rows/s (A10G, w256 d8) | µs/row |
|---|---|---|
| 32 | _TBD_ | _TBD_ |
| 64 | _TBD_ | _TBD_ |
| 128 | _TBD_ | _TBD_ |

The A10G's sweet spot for this net sits at batch 64, with a measurable per-row *regression*
at batch 128 — this single curve explains several otherwise-surprising results downstream
(engine batch sizing, grouped-collection sizing).

## Inference-cache capacity (reinfors)

Position-keyed cache hit rate versus capacity, chess self-play:

| capacity | hit rate |
|---|---|
| 4,096 | _TBD_ |
| 32,768 | _TBD_ |
| 262,144 | _TBD_ |
| 2M | _TBD_ |

Hit rate is monotone in capacity but flattens sharply; capacity is chosen for throughput
(RAM was not a binding constraint on this instance). A secondary observation worth keeping:
hit rate *rises as training progresses* — a stronger net concentrates its own play.

## Topology selection: states/s under the round workload

The decisive measurement. Each candidate topology runs the **full round workload** — cache
on at its own capacity, learner training and checkpoint writes sharing the GPU — for
windows long enough to contain several learn steps and several game-lengths (20-minute
interior windows after 5-minute warmups). Selection is by completed-game states/s.

OpenSpiel (actors × inference batch):

| config | states/s | rows/s | learn steps in window |
|---|---|---|---|
| 16 actors | _TBD_ | _TBD_ | _TBD_ |
| 32 actors | _TBD_ | _TBD_ | _TBD_ |
| 64 actors | _TBD_ | _TBD_ | _TBD_ |
| 64 actors, batch 32 (decoupling probe) | _TBD_ | _TBD_ | _TBD_ |

reinfors (parallel games):

| config | states/s | net rows/s | learn steps in window |
|---|---|---|---|
| n_games 64 | _TBD_ | _TBD_ | _TBD_ |
| n_games 128 | _TBD_ | _TBD_ | _TBD_ |

Two findings from these grids shape the interpretation and are discussed in
[design differences](design-differences.md): OpenSpiel's completed-game rate falls as its
actor count rises even as its rows/s climbs (its batch size is coupled to actor count, so
GPU efficiency and game completion pull against each other), while reinfors' grid is
decided almost entirely by the kernel batch curve (its batch size equals its game count by
construction, decoupled from per-game progress).

## Selected operating points

| | OpenSpiel | reinfors |
|---|---|---|
| topology | _TBD_ | _TBD_ |
| inference batch | _TBD_ | _TBD_ |
| cache capacity | own default | own measured optimum |
| net | w256 d8 (identical) | w256 d8 (identical) |
