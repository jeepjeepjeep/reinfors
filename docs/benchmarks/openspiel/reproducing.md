# Reproducing the comparison

The cross-framework harness lives in the companion benchmark repository
([reinfors-benchmarks](https://github.com/jeepjeepjeep/reinfors-benchmarks)), published
alongside this library: the OpenSpiel source-build
machinery and its documented patches, the measurement, round-orchestration, and
head-to-head scripts, and — under its `published/` directory — the per-run artifacts of
every published number: learner telemetry, resolved configurations, head-to-head logs and
PGNs, each with a provenance record (dates, commit pins, command). Deadline checkpoints
distribute as release assets. That material
is deliberately not vendored into reinfors — third-party patches, training loops, and
cloud tooling are outside the library's scope, and a drifting copy would be worse than a
pointer.

The blocks below are **command templates**, not verbatim invocations: they run from the
companion repository's root (except where noted), and uppercase shell variables come from
the tuning tables or artifact paths. The published training rounds predate the
run-manifest tooling; their raw learner telemetry is archived in the companion workspace,
and every run surface from protocol v1 onward appends machine-readable manifests
(source identity for both repositories, checkpoint hashes, seeds, concurrency, versions)
automatically.

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
MINUTES=120 OS_ACTORS=16 RF_NGAMES=128 RF_NGROUPS=2 ROUND_SEED="$SEED" \
    bash scripts/run_round_chess_gpu.sh

# head-to-head (protocol v1): paired openings, both colors, matched simulations,
# solver off, PGN export and a run manifest for every match.
# The currently published matches predate protocol v1 (they ran the earlier bridge,
# companion commit 7ed1120 — see published/ provenance); reproduce those at that commit.
python benchmarks/openspiel/eval_h2h_chess.py "$RF_CKPT" "$OS_DIR" --os-checkpoint "$OS_CHECKPOINT" \
    --games 100 --sims 64 --seed "$MATCH_SEED" --device cuda --az-device /cuda:0

# telemetry comparison panels from both learners' structured logs
python scripts/plot_round.py "$RF_ROUND_DIR" "$OS_ROUND_DIR" -o round_panels.png
```

## Publication artifacts

Public per published run (companion `published/`): both learners' structured telemetry
(the interior-window counters every table reduces), resolved configurations, run stdout,
head-to-head logs and PGNs, and a provenance record; Arena-protocol matches additionally
append machine-readable manifests. Tables state their provenance inline; commands here
plus the [tuning tables](tuning.md) reconstruct any cell.
