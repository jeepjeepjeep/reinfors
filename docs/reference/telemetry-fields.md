# Telemetry fields

Engine batch telemetry is a plain dictionary scoped to one collection call. The keys are uniform
across policies; **Applies to** identifies which policies populate a mechanism rather than leaving
its counters at zero. This distinction prevents, for example, interpreting zero ensemble
disagreement from `EpsilonGreedyQ` as measured agreement.

| Field | Type / unit | Applies to | Meaning |
| --- | --- | --- | --- |
| `episodes` | `list[tuple[list[float], int, bool]]` | All engine policies | Completed `(returns, length, seeded)` tuples. Returns contain one scalar per player; length is in ticks. |
| `decisions` | `int` decisions | All engine policies | Policy decisions completed during collection. |
| `max_depth` | `int` tree depth | `SelectiveExpectimax`, `Minimax`, `Mcts`, `AlphaZero` | Maximum search depth observed. |
| `mean_leaves` | `float` leaves / decision | `SelectiveExpectimax`, `Minimax`, `Mcts`, `AlphaZero` | Mean leaf count over completed searches. |
| `mean_rounds` | `float` rounds / decision | `SelectiveExpectimax`, `Minimax`, `Mcts`, `AlphaZero` | Mean pooled inference/search rounds. |
| `mean_expansions` | `float` nodes / decision | `SelectiveExpectimax`, `Minimax`, `Mcts`, `AlphaZero` | Mean expanded-node count. |
| `mean_sigma` | `float` value units | `SelectiveExpectimax`, `n_heads >= 2` | Mean epistemic uncertainty across searched leaves; reported as `0.0` with one head. |
| `mean_disagreement` | `float` value units | `SelectiveExpectimax`, `n_heads >= 2` | Mean root action-value disagreement across ensemble heads; reported as `0.0` with one head. |
| `infer_seconds` | `float` seconds | All engine policies | Wall time observed inside inference callbacks. |
| `infer_calls` | `int` calls | All engine policies | Pooled callback invocations. |
| `infer_rows` | `int` rows | All engine policies | Real observation rows evaluated, excluding `pad` padding. Mean physical call size is `(infer_rows + padded_rows) / infer_calls`. |
| `padded_rows` | `int` rows | All engine policies | Zero pad rows forwarded by `pad` (0 when disabled). |
| `cache_lookups` | `int` rows | Any policy with `infer_cache` enabled | Persistent evaluation-cache lookups. |
| `cache_hits` | `int` rows | Any policy with `infer_cache` enabled | Successful persistent evaluation-cache lookups. |
| `cache_demotions` | `int` rows | Any policy with `infer_cache` enabled | Gated hits invalidated by a `weights_updated()` boundary at release and re-inferred instead of delivered. |
| `terminal_sims` | `int` simulations | `SelectiveExpectimax`, `Mcts`, `AlphaZero` | Simulations ending at a terminal state. |
| `depthcap_sims` | `int` simulations | `SelectiveExpectimax`, `Mcts`, `AlphaZero` | Simulations ending at the configured depth cap. |
| `requested_rows` | `int` rows | All engine policies | Rows entering the inference queue (decision evaluations and tail bootstraps), pre-dedup and pre-cache, counted once at the queue seam; dedup and cache savings are the gap to `infer_rows`. Replaces 0.2.x's `fresh_rows`/`hit_rows`/`shared_rows` split — row provenance is an Evaluator fact. |
| `tail_rows` | `int` rows | All engine policies | The tail-bootstrap share of `requested_rows`; `requested_rows - tail_rows` is the decision-evaluation term of the simulation identity. |
| `extra_eval_rows` | `int` rows | `Mcts`, `AlphaZero` | Rows beyond one per simulation, from multi-perspective leaf evaluation or an `ExpandAll` chance fan. |

The preceding Engine cache counters apply only when the engine was constructed with `infer_cache`;
they remain zero when it is disabled. See the
[cache lifecycle guide](../guides/configuration-and-checkpoints.md#inference-cache-lifecycle).
The `seeded` episode flag is relevant only when
[reached-state starts](../guides/configuration-and-checkpoints.md#reached-state-starts) are enabled.

## Deep CFR

`rf.solvers.DeepCfr.collect` returns a separate telemetry dictionary:

| Field | Type / unit | Applies to | Meaning |
| --- | --- | --- | --- |
| `player` | `int` player id | Every Deep CFR collect | Traversing player for this collection. |
| `traversals` | `int` traversals | Every Deep CFR collect | External-sampling traversals completed. |
| `advantage_samples` | `int` records | Every Deep CFR collect | Advantage records produced. |
| `strategy_samples` | `int` records | Every Deep CFR collect | Average-strategy records produced. |
| `infer_calls` | `int` calls | Every Deep CFR collect | Pooled advantage-network callback invocations. |
| `infer_rows` | `int` rows | Every Deep CFR collect | Information-state rows sent to callbacks. |
| `infer_seconds` | `float` seconds | Every Deep CFR collect | Wall time observed inside inference callbacks. |
| `collect_seconds` | `float` seconds | Every Deep CFR collect | Total collection wall time. |
| `cache_lookups` | `int` rows | Every Deep CFR collect | Advantage-cache lookups within this collection. |
| `cache_hits` | `int` rows | Every Deep CFR collect | Successful advantage-cache lookups within this collection. |

Deep CFR's per-player caches are unconditional implementation machinery rather than a constructor
option. They reuse rows within one `collect` call and are force-cleared at the start of the next
call, allowing networks to be retrained between calls without explicit invalidation.

Treat the dictionary as extensible. Read known keys and preserve unknown keys when forwarding
structured logs. See [telemetry and TensorBoard](../guides/telemetry.md) for aggregation.
