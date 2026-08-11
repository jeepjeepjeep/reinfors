# Operating-point tuning

No head-to-head ran until each side's configuration was selected by measurement on the
target hardware. Selection cells were single 20-minute legs under the full round workload,
reduced over the interior window (minutes 5 to 19.5, counter deltas) — August 2026,
decision-grade rather than repeat-derived; the margins between candidates were far outside
window noise.

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
legs; the reduced interior window spans minutes 5 to 19.5). Selection is by
completed-game states/s.

OpenSpiel (actors × inference batch):

| config | states/s | achieved batch |
|---|---|---|
| 16 actors | **146.3** | 16 (full fill) |
| 32 actors | 111.6 | 32 |
| 64 actors | 62.9 | ~64 |
| 64 actors, batch 32 (decoupling probe) | 63.2 | 32 |

Their states/s falls monotonically with actor count on this 4-core box: per-game progress
decelerates faster than batching gains (per-game rates 9.2 / 6.4 / 3.5 across the column),
and the decoupling probe shows the batch size is not the cause. The bracketing rules out
both a8 (deceleration) and a24 (interpolation).

reinfors (parallel games, ungrouped — grouped collection was adopted later via the
[lever grid](../internal/throughput-levers.md)):

| config | states/s | rows/call |
|---|---|---|
| n_games 64 | **164.6** | ~56 |
| n_games 128 | 158.8 | ~110 |

Identical rows-per-state across the two cells (53.3) isolates the difference as the pure
batch-128 kernel regression.

The grids answered the mechanism question directly: at each side's optimum, OpenSpiel pays
128.4 µs/row (full-fill batch-16 calls — its batching-vs-completion coupling forces small
batches at its states/s optimum, with its single inference thread 89% saturated) against
reinfors' 89.8 µs/row at the batch-64 sweet spot. The
[design differences](design-differences.md) trace why the two schedulers land on different
trade-offs.

## Selected operating points

| | OpenSpiel | reinfors |
|---|---|---|
| topology | 16 actors | 128 games × 2 groups |
| inference batch | 16 | ~56-row calls (64-row groups) |
| cache capacity | 262,144 (own default) | 262,144 (matched; hit rate is monotone in capacity) |
| net | w256 d8 (identical) | w256 d8 (identical) |

reinfors' selection proceeded in two steps: `n_games=64` won the ungrouped grid above,
then grouped collection's [matched-rows comparison](../internal/throughput-levers.md)
moved the operating point to `n_games=128, n_groups=2` (same ~56-row calls, tree work
overlapped with inference). The seed-0 round ran both points; the seed-1 round and the
published headline use the grouped one.
