# Benchmarks

This section is intentionally a skeleton while the benchmark suite is rerun on controlled
hardware. No historical headline numbers are carried forward without their complete
configuration and reproducibility artifacts.

These benchmarks are not intended to claim universal throughput leadership. Specialized, fully
fused implementations may be faster on the workload they were built around; the relevant question
for reinfors is how much throughput its modular game/search/training boundary preserves, and where
that flexibility becomes the limiting cost.

The published results will evaluate the claims reinfors is designed around:

- native environment and search throughput;
- effective inference batch sizes at the Rust/Python boundary;
- synchronous versus overlapped actor–learner utilization;
- scaling with parallel games, CPU threads, and search budget;
- comparison with relevant OpenSpiel implementations where semantics align;
- end-to-end time-to-training-data, not only isolated environment steps.

Results will link to the exact commit, resolved configurations, command lines, raw data, and
analysis code. Until those artifacts are ready, treat performance as an implementation goal,
not a documented guarantee.

## Next steps

- Read the required [methodology and publication checklist](methodology.md).
- Run the existing harnesses from the [reproduction guide](reproducing.md).
