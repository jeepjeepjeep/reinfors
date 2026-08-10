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

The comparison targets a recorded master snapshot built from source with CUDA libtorch;
the required build interventions and patches are documented with the
[comparison itself](openspiel/index.md#the-comparison-target).

## Per-boot checklist

Settings that silently reset on instance stop/start, re-verified before every session: SMT
off; branches pulled; reinfors release wheel rebuilt into the measurement venv; OpenSpiel
binaries rebuilt if any patch changed. The public IP also changes per boot (no elastic IP).
