# Reproducing the comparison

The cross-framework harness lives in the companion benchmark repository
(`reinfors-benchmarks`), published alongside this library: the OpenSpiel source-build
machinery and its documented patches, the measurement and round-orchestration scripts, the
head-to-head bridge, and the raw logs and artifacts of every published run. That material
is deliberately not vendored into reinfors — third-party patches, training loops, and
cloud tooling are outside the library's scope, and a drifting copy would be worse than a
pointer.

The blocks below are **command templates**, not verbatim invocations: they run from the
companion repository's root (except where noted), and angle-bracket values come from the
tuning tables. Every published result links a pinned companion-repo commit whose run
manifest contains the exact commands, resolved configurations, and seeds — audit against
the manifest, not this page.

## One-time setup (per instance)

```bash
# OpenSpiel from source with CUDA libtorch (restores build glue, applies the
# instrumentation + device patches, builds trainer and head-to-head binaries)
bash scripts/setup_openspiel_cpp.sh

# from the reinfors checkout: release wheel into the measurement venv (a debug build is
# refused at runtime)
maturin develop --release -m crates/reinfors-py/Cargo.toml
```

Per-boot: disable SMT (guarded by every script), pull, rebuild the wheel.

## Operating-point measurements

```bash
# kernel ceiling + engine sweeps (batch curve, per-row costs)
python benchmarks/openspiel/phase0_gpu_sweep.py  # modes and grids per --help

# topology selection under the round workload (cache on, learner + checkpoints active,
# 20-minute interior windows, hard-kill protocol)
CORES=0-3 GAME=chess WIDTH=256 DEPTH=8 ACTORS="16 32 64 64:32" bash scripts/measure_states.sh
CORES=0-3 WIDTH=256 DEPTH=8 NGAMES="64 128" MINUTES=20 bash scripts/measure_states_rf.sh
```

## The matched round and head-to-head

```bash
# two sequential 2h legs (OpenSpiel then reinfors), hard-killed at the deadline,
# post-run listing of each side's checkpoint artifacts
MINUTES=120 OS_ACTORS=<from tuning> bash scripts/run_round_chess_gpu.sh

# head-to-head: paired openings, both colors, matched simulations, solver off,
# PGN export with run metadata for every game
python benchmarks/openspiel/eval_h2h_chess.py <rf_ckpt> <os_dir> --os-checkpoint <N> \
    --games 50 --sims 64 --device cuda --az-device /cuda:0

# telemetry comparison panels from both learners' structured logs
python scripts/plot_round.py <rf_round_dir> <os_round_dir> -o round_panels.png
```

## Publication artifacts

Per published number: reinfors and companion-repo commits, `resolved_config()` /
`config_fingerprint()` of every engine, full command line, raw interior-window samples,
learner logs from both stacks, head-to-head PGNs, and the analysis script that reduced
them.
