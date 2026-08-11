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
cache capacity, the `n_groups` grid) run through the companion benchmark repository's
trainer, on the same instance and under the same protocol as the cross-framework
comparison. As on the [comparison reproduction page](../openspiel/reproducing.md), these
are command templates; raw telemetry for the published cells is public under the
companion repository's `published/` directory:

```bash
# from the companion repository root
# n_groups grid (the published cells ran MINUTES=12)
CORES=0-3 WIDTH=256 DEPTH=8 NGAMES="64 128" NGROUPS=1 MINUTES=12 bash scripts/measure_states_rf.sh
CORES=0-3 WIDTH=256 DEPTH=8 NGAMES="128 64" NGROUPS=2 MINUTES=12 bash scripts/measure_states_rf.sh

# kernel ceiling + f32-vs-f64 callback arms (pure-forward and engine modes)
python benchmarks/openspiel/phase0_gpu_sweep.py  # modes, widths, batch grids per --help

# cache capacity curve: the trainer at fixed conditions across --infer-cache values
for CAP in 4096 32768 262144 2097152; do
    taskset -c 0-3 .venv23/bin/python benchmarks/openspiel/train_reinfors_az.py \
        --minutes 15 --device cuda --game chess --out "results/cap_$CAP" \
        --n-games 64 --sims 64 --width 256 --depth 8 --infer-cache "$CAP"
done
```

## Publication artifacts

Per published number: the reinfors commit, `resolved_config()` / `config_fingerprint()` of
every engine, full command line, and raw telemetry.
