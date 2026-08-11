# The matched round

The headline experiment: each stack trains chess from scratch for **2 hours of wall-clock**
on the whole machine (legs run sequentially, never sharing the box), at its own measured
operating point, under the full matched-knob protocol — then the two resulting models play
each other.

## Provenance

Figures below are from single 2-hour runs per (seed, side) on the shared
[environment](../setup.md) (g5.2xlarge, August 2026), with the headline quantities
replicated across two independent training seeds. Sustained rates use the interior
window (minutes 5–115, counter deltas). OpenSpiel pin `112b7770` (source-built, CUDA
libtorch 2.3.0); the seed-0 reinfors leg ran the v1 grouped scheduler, the seed-1 leg the
rewritten v2 scheduler at the merge commit — the two agree to 0.4%. These runs predate
the run-manifest tooling.

## Training throughput

Cells read seed 0 / seed 1. reinfors ran its selected operating point
(`n_games=128, n_groups=2`); the seed-1 leg ran the rewritten grouped scheduler (v2),
which reproduced the v1 numbers.

| | OpenSpiel | reinfors |
|---|---|---|
| states collected (2h) | 1.06M / 1.08M | 1.45M / 1.44M |
| sustained states/s | 147.2 / 150.7 | 204.1 / 203.2 |
| learn steps (1024-sample minibatch-equivalents) | 3,072 / 3,136 | 4,209 / 4,208 |
| gradient-samples per state (verified reuse, target 3.0) | 2.98 / 2.99 | 2.97 / 2.99 |
| final cache hit rate¹ | 46.8% / 46.8% | 26.4% / 27.3% |

A **+39% / +35%** sustained-throughput edge at matched wall-clock, cadence, and net
architecture. Seed 0 also ran the pre-grouping operating point (`n_games=64`,
ungrouped): 172.4 states/s, a +17% edge — grouped collection accounts for the
difference between the two rows, as measured in the
[lever grid](../internal/throughput-levers.md).

¹ Not comparable across the columns: OpenSpiel's figure is run-cumulative and its
evaluator issues Prior and Evaluate as two cache queries per node (pairs merge in the
cache), while the reinfors figure is interior-window with one query per row. Both rise
as the nets strengthen.

Learner telemetry panels (losses vs wall-clock and vs states collected, throughput
trajectories, cache-hit evolution, game-length trends) are rendered from both learners'
structured logs by one script; the loss curves are definitionally aligned but each is
measured on its own self-play distribution — they show per-system learning progress, not
head-to-head quality.

## Head-to-head

Protocol: the two final models play a match at 64 simulations per move on both sides,
their side running its own unmodified game runner with its native chess solver disabled
(search-plus-network against search-plus-network at matched simulation budgets, with no
solver assist), our side driven through a bridge that mirrors the game state and submits moves
over its human-input interface. Openings are seeded uniform-random fixed lines, each played
once per color (paired scoring); artifacts are each side's last complete checkpoint before
the deadline kill; every game is exported as PGN with full run metadata.

| | result (pooled) |
|---|---|
| games (opening pairs) | 250 (125) |
| W / D / L (reinfors perspective) | 74 / 130 / 46 |
| score ± SE (paired) | 0.556 ± 0.022 |
| implied Elo difference (95% CI) | +39 (+9 to +70) |

The pool combines three protocol-identical matches; the edge holds in each and
replicates across the independent seed-1 training draw:

| match | nets | games | W / D / L | score ± SE |
|---|---|---|---|---|
| seed 0, match 1 | rf `n64` leg vs OS | 50 | 14 / 28 / 8 | 0.560 ± 0.051 |
| seed 0, match 2 | rf `n128×2` leg vs OS | 100 | 34 / 44 / 22 | 0.560 ± 0.037 |
| seed 1 | rf `n128×2` leg vs OS | 100 | 26 / 58 / 16 | 0.550 ± 0.034 |

Interpretation uses the pair-level standard error and confidence interval above; no result
is treated as more precise than the match supports. The interval quantifies
match-sampling uncertainty for these fixed trained checkpoints — robustness across
training draws is evidenced by the seed-1 replication, not by the interval.

## Open items

- **single run per (seed, configuration)** — the two-seed replicate addresses
  training-draw variance for the headline claims; per-cell precision is bounded by one
  run each;
- one game (chess) at one net size.
