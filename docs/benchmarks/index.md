# Benchmarks

This section is intentionally a skeleton while the benchmark suite is rerun on controlled
hardware. No historical headline numbers are carried forward without their complete
configuration and reproducibility artifacts.

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

See [methodology](methodology.md) and [reproducing](reproducing.md) for the planned reporting
contract.
