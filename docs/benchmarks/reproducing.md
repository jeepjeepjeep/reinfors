# Reproducing benchmarks

The repository already contains two benchmark harnesses. Published A10G results and downloadable
artifacts are still pending.

## Native throughput

`scripts/benchmark.py` measures native environment stepping, records per second from synchronous
collection, inference-cost sweeps, and scaling across parallel games:

```bash
# Build or install reinfors in release mode first.
python scripts/benchmark.py --quick
python scripts/benchmark.py
```

The harness warns when `rf.core_build_profile()` reports a debug build. Use `--help` for workload,
repeat, and scaling controls. `tests/test_benchmark.py` smoke-tests the harness with a tiny workload.

## Cross-framework comparison

`scripts/benchmark_vs.py` is the manually run Connect 4 comparison with OpenSpiel and Pgx/Mctx. Its
optional dependencies are intentionally not part of the normal development environment; unavailable
backends are reported and skipped.

```bash
python scripts/benchmark_vs.py --help
```

Read the script's module documentation before running it: raw stepping, Python-network MCTS,
OpenSpiel's native rollout reference, and accelerator-resident Pgx/Mctx are separate tracks rather
than one interchangeable throughput number. `tests/test_benchmark_vs.py` smoke-tests the reinfors
backend.

## Publication artifacts

The pending benchmark publication will pin the commit, release wheel, dependencies, machine
metadata, resolved configurations, seeds, command lines, and raw JSON/CSV. This page will then link
to those artifacts and exact reproduction commands; until then, the harnesses are available but no
headline result is documented.

## Next steps

- Check every run against the [benchmark methodology](methodology.md).
- Return to [benchmark status](index.md) for published-artifact availability.
