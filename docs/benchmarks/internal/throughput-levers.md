# Throughput levers

Opt-in reinfors features that move collection throughput, each with its measured effect at
the benchmark operating point and the model that predicts where it helps elsewhere.

## Native f32 inference outputs

The engine accepts `float32` callback outputs at every ingestion surface and widens
exactly in Rust (bit-identical to the caller converting to `float64`, without the
conversion and double-width transfer), including padded policy widths so torch heads sized
past the action space return whole tensors without a device-side slice.

| config (chess, CUDA) | f64 path rows/s | f32 path rows/s | gain |
|---|---|---|---|
| w256 d8, batch 64 | _TBD_ | _TBD_ | _TBD_ |
| w128 d8, batch 64 | _TBD_ | _TBD_ | _TBD_ |

## Inference cache

Position-keyed reuse of network rows across the search, cleared on weight refresh. Effect
depends on the game's transposition structure (measured hit rates: chess low-teens percent
and *rising* as the net strengthens; small-board connect-style games far higher) and on
capacity only up to a saturation point — see the [capacity curve](index.md).

## Grouped collection (`n_groups=2`)

Splits the games into two fixed groups whose search rounds alternate, so one group's tree
work overlaps the other's inference (which runs on a dedicated submitter thread). Full
semantics in the [training guide](../../guides/training.md#overlapping-search-and-inference-n_groups).

The model, validated empirically: with per-round search time `S` and inference time `I`,
throughput improves by up to `(S + I) / max(S, I)` — maximized when the stages are
*balanced*, small when either dominates — **minus** whatever the configuration pays on the
kernel batch curve for its per-call row count. Two comparisons matter:

- **matched game count** (does splitting my current games help?): usually not — halving
  rows per call costs kernel efficiency;
- **matched rows per call** (should I double my games into two groups?): the deployment
  question; with inference share `p`, gains approach the `1 / max(p, 1 - p)` ceiling when
  the scheduler is saturated.

Mechanism validation (local Apple-silicon grid; illustrative of the model, not of absolute
throughput): at a near-balanced operating point the matched-rows comparison realized ×1.70
of a theoretical ×1.72 ceiling, and at an inference-dominated point with a steep batch
curve, splitting below the sweet spot made grouping net-negative — both signs predicted by
the model.

A10G grid at the benchmark operating point (n64×1 / n128×1 / n128×2 / n64×2, reporting
throughput, realized rows per call, inference share, and decision latency):

| config | states/s | rows/s | rows/call | infer share |
|---|---|---|---|---|
| n64 × 1 group | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| n128 × 1 group | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| n128 × 2 groups | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| n64 × 2 groups | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

Predicted for this operating point: the measured inference share ≈ 0.92 puts the ceiling
at ≈ +9% (`1/0.92`), with a favorable batch term: two 64-row groups sit at the A10G
sweet spot, while a single 128-row batch pays a measured kernel regression. Boundary costs
(submitter spawn per collect, one discarded in-flight batch at the record floor) are
amortized at round-scale collection and measured separately at small collection sizes
before any general recommendation.
