# Telemetry fields

Engine batch telemetry is a plain dictionary scoped to one collection call. A field may be
zero when the selected policy does not use that mechanism.

| Field | Meaning |
| --- | --- |
| `episodes` | Completed `(returns, length, seeded)` tuples. `returns` is one scalar per player; `seeded` marks reached-state-buffer starts. |
| `decisions` | Policy decisions completed during collection. |
| `max_depth` | Maximum search depth observed. |
| `mean_leaves` | Mean leaf count over measured searches. |
| `mean_rounds` | Mean pooled inference/search rounds. |
| `mean_expansions` | Mean expanded nodes. |
| `mean_sigma` | Mean ensemble uncertainty statistic where defined. |
| `mean_disagreement` | Mean ensemble action disagreement where defined. |
| `infer_seconds` | Wall time observed inside inference callbacks. |
| `infer_calls` | Number of pooled callback invocations. |
| `infer_rows` | Total observation rows sent to callbacks. |
| `cache_lookups` | Evaluation-cache lookup count. |
| `cache_hits` | Successful evaluation-cache lookups. |
| `terminal_sims` | Simulations ending at terminal state. |
| `depthcap_sims` | Simulations ending at the configured depth cap. |
| `shared_rows` | Rows shared by more than one request within batching/search reuse. |
| `fresh_rows` | Newly evaluated rows. |
| `hit_rows` | Rows served from the persistent inference cache. |
| `extra_eval_rows` | Additional perspectives evaluated for an algorithmic backup. |

Deep CFR batch telemetry includes `player`, `traversals`, advantage and strategy sample
counts, inference calls/rows/seconds, total collection seconds, and cache statistics.

Treat the dictionary as extensible. Read known keys and preserve unknown keys when forwarding
structured logs. See [telemetry and TensorBoard](../guides/telemetry.md) for aggregation.
