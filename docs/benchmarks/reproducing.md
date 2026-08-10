# Reproducing

All harnesses, patches, raw logs, and analysis scripts live in the companion benchmark
repository (`reinfors-benchmarks`); this page summarizes the run surface so results can be
audited against exact commands. Every published table links to the commit and resolved
configuration that produced it.

## One-time setup (per instance)

```bash
# OpenSpiel from source with CUDA libtorch (restores build glue, applies the
# instrumentation + device patches, builds trainer and head-to-head binaries)
bash scripts/setup_openspiel_cpp.sh

# reinfors: release wheel into the measurement venv (a debug build is refused at runtime)
maturin develop --release -m crates/reinfors-py/Cargo.toml
```

Per-boot: disable SMT (guarded by every script), pull, rebuild the wheel.

## Operating-point measurements

```bash
# kernel ceiling + engine sweeps (batch curve, per-row costs)
python benchmarks/openspiel/phase0_gpu_sweep.py ...

# topology selection under the round workload (cache on, learner + checkpoints active,
# 20-minute interior windows, hard-kill protocol)
CORES=0-3 GAME=chess WIDTH=256 DEPTH=8 ACTORS="16 32 64 64:32" bash scripts/measure_states.sh
CORES=0-3 WIDTH=256 DEPTH=8 NGAMES="64 128" MINUTES=20 bash scripts/measure_states_rf.sh
```

## The matched round and head-to-head

```bash
# two sequential 2h legs (OpenSpiel then reinfors), hard-killed at the deadline,
# post-run listing of each side's checkpoint artifacts
MINUTES=120 OS_ACTORS=<selected> bash scripts/run_round_chess_gpu.sh

# head-to-head: paired openings, both colors, matched simulations, solver off,
# PGN export with run metadata for every game
python benchmarks/openspiel/eval_h2h_chess.py <rf_ckpt> <os_dir> --os-checkpoint <N> \
    --games 50 --sims 64 --device cuda --az-device /cuda:0

# telemetry comparison panels from both learners' structured logs
python scripts/plot_round.py <rf_round_dir> <os_round_dir> -o round_panels.png
```

## Grouped-collection grid

```bash
CORES=0-3 WIDTH=256 DEPTH=8 NGAMES="64 128" NGROUPS=1 MINUTES=20 bash scripts/measure_states_rf.sh
CORES=0-3 WIDTH=256 DEPTH=8 NGAMES="64 128" NGROUPS=2 MINUTES=20 bash scripts/measure_states_rf.sh
```

## Publication artifacts

Per published number: reinfors and benchmark-repo commits, `resolved_config()` /
`config_fingerprint()` of every engine, full command line, raw interior-window samples,
learner logs from both stacks, head-to-head PGNs, and the analysis notebook or script that
reduced them.
