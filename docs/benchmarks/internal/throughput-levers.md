# Throughput levers

Opt-in reinfors features that move collection throughput, each with its measured effect at
the benchmark operating point and the model that predicts where it helps elsewhere.

## Native f32 inference outputs

The engine accepts `float32` callback outputs at every ingestion surface and widens
exactly in Rust (bit-identical to the caller converting to `float64`, without the
conversion and double-width transfer), including padded policy widths so torch heads sized
past the action space return whole tensors without a device-side slice.

A10G, 3-cycle medians, within-stack comparison (the callback drops its `.double()` and
returns native f32; everything else identical):

| config (chess, CUDA) | f64 path rows/s | f32 path rows/s | gain |
|---|---|---|---|
| w256 d8, batch 64 | 11,380 | 12,493 | +9.8% |
| w128 d8, batch 64 | 15,591 | 19,483 | +25.0% |

The gain grows as the net shrinks because the boundary cost is a larger share of a
smaller forward. Measured null worth knowing: layering pinned host transfers, packed
single-copy returns, and no-slice padded heads on top of the f32 path moved nothing on
CUDA — the remaining per-call overhead is Python/torch op dispatch in the callback, not
transfer, so further boundary tuning is not the lever there.

## Inference cache

Position-keyed reuse of network rows across the search, cleared on weight refresh. Effect
depends on the game's transposition structure (measured at the chess benchmark operating
point: mid-teens early, 26–27% by two hours of training, *rising* as the net strengthens;
small-board connect-style games far higher) and on capacity only up to a saturation
point — see the [capacity curve](index.md).

## Grouped collection (`n_groups=2`)

Splits the games into two fixed groups, each collecting on its own worker thread with
inference forwarded to a service thread that owns the callback, so one group's tree work
overlaps the other's inference. Full
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

A10G grid at the benchmark operating point (v2 scheduler at the merged tip, 12-minute
interior windows on the round-true workload — early-training games make the absolute
rates read below the 2-hour round numbers; the grid measures the levers' *relative*
effects):

| config | states/s | rows/s | rows/call | infer share |
|---|---|---|---|---|
| n64 × 1 group | 148.3 | 8,307 | 55.8 | 0.78 |
| n128 × 1 group | 151.6 | 8,105 | 109.9 | 0.76 |
| n128 × 2 groups | **185.8** | 10,200 | 56.4 | 0.98 |
| n64 × 2 groups | 149.7 | 8,078 | 28.3 | 0.94 |

The matched-rows comparison (n64×1 → n128×2) realized ×1.25 against the ×1.28 ceiling
that the ungrouped inference share (0.78) predicts, with the grouped service near
saturation (share 0.98); the matched-games comparison (n64×1 → n64×2) gained little, as
the batch-curve term predicts for 28-row calls.

The ceiling prediction uses the inference share measured **in the target condition**,
from the same telemetry the grid reports (per-round wall against callback time). Shares
derived from isolated component probes understate the overlappable engine-side work —
record emission, batch packing, and runtime interplay with a concurrent learner all sit on
the engine side of the cycle — and therefore understate the gain. The batch term at this
operating point is favorable: two 64-row groups sit at the A10G sweet spot, while a single
128-row batch pays a measured kernel regression. Boundary costs differ in scale: worker and service thread spawns amortize over the length
of a collect, while the channel round-trip on every inference call amortizes only over
that call's rows. Both are measured at small collection sizes before any general
recommendation.
