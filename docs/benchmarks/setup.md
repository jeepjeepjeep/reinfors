# Environment and setup

Every cross-stack number in this section was measured on one machine, in one session per
comparison, with both stacks built against the same kernel generation.

## Instance

| item | value |
|---|---|
| instance | AWS g5.2xlarge |
| GPU | 1× NVIDIA A10G (24 GB) |
| CPU | 8 vCPU (4 physical cores; SMT **disabled** for every measurement) |
| memory | 32 GiB |
| OS | Ubuntu 22.04 |
| storage | gp3 (baseline 125 MB/s — relevant to checkpoint-write costs) |

Isolation invariants, enforced by the measurement scripts (which refuse to run otherwise):

- SMT off (`/sys/devices/system/cpu/smt/control`) — this resets on every instance
  stop/start and is guarded, not assumed;
- all benchmark processes pinned to cores 0–3 (`taskset`), one measurement at a time;
- `OMP_NUM_THREADS=1` for the OpenSpiel side, per its own documentation (the libtorch
  intra-op pool otherwise competes with its actor threads).

## Software

| component | version / provenance |
|---|---|
| reinfors | release build (`core_build_profile()` guard aborts on debug), commit recorded per run |
| Python torch | 2.3.0 + cu121 (pinned to the libtorch generation below) |
| OpenSpiel | master snapshot `112b7770` built from source with CUDA libtorch |
| libtorch | 2.3.0 cu121 (their pinned generation) |

Torch/libtorch are pinned to the **same kernel generation** on both sides: a 2.13-vs-2.3
skew was measured at 1.29–1.4× on this workload, large enough to dominate any real
difference.

### The OpenSpiel build

The comparison targets OpenSpiel **as maintained today**, which required two documented
interventions, both content-preserving:

- current master cannot build the libtorch path as shipped (the libtorch/libnop CMake glue
  was deleted upstream while still referenced); the build restores that glue verbatim from
  the parent commit — build wiring only, zero behavior change;
- the as-shipped release era carried a long-standing GPU tensor-staging inefficiency that
  upstream has since fixed on master; benchmarking the fixed master (rather than the slower
  pinned release) is deliberate — measuring a known-fixed bug would flatter reinfors and
  say nothing.

One additional patch adds instrumentation counters to their evaluator (requests, cache
hits, forwards, forward time) and a device flag to their example game runner — measurement
and evaluation surface only, no algorithmic change. All patches ship in the benchmark
repository and are applied idempotently by its setup script.

## Per-boot checklist

Settings that silently reset on instance stop/start, re-verified before every session: SMT
off; branches pulled; reinfors release wheel rebuilt into the measurement venv; OpenSpiel
binaries rebuilt if any patch changed. The public IP also changes per boot (no elastic IP).
