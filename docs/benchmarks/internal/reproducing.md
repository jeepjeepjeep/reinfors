# Reproducing the internal benchmarks

The internal family makes claims about reinfors against itself; its harnesses live in
the companion benchmark repository so one checkout carries every measurement.

## Harness location

All benchmarking scripts live in the companion repository — including the CPU /
parallel-scaling sweeps (`benchmarks/internal/benchmark.py`) and the cross-framework
connect4 tracks (`benchmarks/internal/benchmark_vs.py`) — so one checkout carries every
measurement. The maintained reinfors examples remain the reference for engine-level
configuration, e.g.:

```bash
# in this repository
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

# kernel ceiling (pure forward at the operating shape)
python benchmarks/openspiel/phase0_gpu_sweep.py --mode net --game chess --devices cuda \
    --widths 256 --depths 8 --batches 64

# f32-vs-f64 callback arms (engine mode at the published cells): one leg per arm per
# cycle, three alternating cycles, medians published
python benchmarks/openspiel/phase0_gpu_sweep.py --mode engine --game chess --devices cuda \
    --widths 256,128 --depths 8 --n-games 64 --infer-dtype f64 --engine-leg-seconds 120
python benchmarks/openspiel/phase0_gpu_sweep.py --mode engine --game chess --devices cuda \
    --widths 256,128 --depths 8 --n-games 64 --infer-dtype f32 --engine-leg-seconds 120

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
