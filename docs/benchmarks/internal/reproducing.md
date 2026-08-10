# Reproducing the internal benchmarks

The internal family makes claims about reinfors against itself, so its reproduction
surface lives in this repository.

## In-repo today

```bash
# CPU / parallel-scaling sweeps (records/sec vs n_games, engine modes, stepping ceiling)
uv run python scripts/benchmark.py

# engine-level configuration through the maintained examples, e.g. grouped collection
uv run --with torch python examples/train_alphazero_example.py --n-games 128 --n-groups 2
```

## GPU lever grids

The CUDA grids behind the [throughput levers](throughput-levers.md) tables (f32 outputs,
cache capacity, the `n_groups` grid) currently run through the companion benchmark
repository's trainer, on the same instance and under the same protocol as the
cross-framework comparison. A self-contained in-repo runner (torch example net + engine +
the grid loop) is planned so this family never requires the companion repo; until it
lands, the exact invocations are:

```bash
# in the companion repository
CORES=0-3 WIDTH=256 DEPTH=8 NGAMES="64 128" NGROUPS=1 MINUTES=20 bash scripts/measure_states_rf.sh
CORES=0-3 WIDTH=256 DEPTH=8 NGAMES="64 128" NGROUPS=2 MINUTES=20 bash scripts/measure_states_rf.sh
```

## Publication artifacts

Per published number: the reinfors commit, `resolved_config()` / `config_fingerprint()` of
every engine, full command line, and raw telemetry.
