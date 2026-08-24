"""Reproduce the README's CarRacing throughput comparison against Gymnasium.

Two measurements, stepping the same game with pixel observations under random
actions: one `rf.Env` (step + observe per tick) and one Gymnasium `CarRacing-v3`
env (its `step` renders the obs). Single-threaded, bare user-stepped loops on
both sides — the symmetric comparison. Parallel collection and collect_stream
overlap multiply reinfors' side further but have no gym-primitive equivalent,
so they are not benchmarked here.

Methodology: one discarded warm-up per implementation, then `--trials` repeated
measurements in alternating order; medians and ranges are reported, with machine
and software provenance. The script does not set CPU affinity — pin externally
(`taskset -c 0-3 ...` on Linux) and say so when reporting results.

Build the wheel with `RUSTFLAGS="-C target-cpu=native"` for the quoted numbers;
Gymnasium's stack selects SIMD at runtime, so a native build is the like-for-like
configuration.

    uv run --no-project --with "reinfors,gymnasium[box2d]" \
        python scripts/bench_carracing_throughput.py
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import reinfors as rf


def bench_reinfors_single(seconds: float) -> float:
    env = rf.Env(rf.games.CarRacing(), rf.Reward(), seed=0)
    env.reset()
    rng = np.random.default_rng(0)
    steps = 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        if env.done():
            env.reset()
        env.step({0: int(rng.integers(5))})
        env.observe(0)
        steps += 1
    return steps / (time.perf_counter() - t0)


def bench_gym_single(seconds: float) -> float:
    import gymnasium as gym

    env = gym.make("CarRacing-v3", continuous=False)
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    steps = 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        _, _, term, trunc, _ = env.step(int(rng.integers(5)))
        steps += 1
        if term or trunc:
            env.reset()
    env.close()
    return steps / (time.perf_counter() - t0)


def provenance() -> None:
    import os
    import pathlib
    import platform
    import subprocess

    import gymnasium

    cpu = platform.processor() or platform.machine()
    if platform.system() == "Darwin":
        cpu = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
        ).stdout.strip()
    else:
        cpuinfo = pathlib.Path("/proc/cpuinfo")
        if cpuinfo.exists():
            for line in cpuinfo.read_text().splitlines():
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    sched_getaffinity = getattr(os, "sched_getaffinity", None)
    if sched_getaffinity is not None:
        affinity = f"{len(sched_getaffinity(0))} cpus in affinity mask"
    else:
        affinity = "not enforceable via this OS (report external pinning separately)"
    print(f"platform:  {platform.platform()}")
    print(f"cpu:       {cpu}")
    print(f"affinity:  {affinity}")
    print(
        f"python:    {platform.python_version()}  numpy: {np.__version__}  "
        f"gymnasium: {gymnasium.__version__}  reinfors: {rf.build_info()['version']}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=30.0, help="wall time per trial")
    ap.add_argument("--trials", type=int, default=3, help="repeated trials per side")
    args = ap.parse_args()

    provenance()
    bench_reinfors_single(3.0)  # warm-up, discarded
    bench_gym_single(3.0)
    rf_rates: list[float] = []
    gym_rates: list[float] = []
    for t in range(args.trials):
        order = [("rf", bench_reinfors_single), ("gym", bench_gym_single)]
        if t % 2 == 1:
            order.reverse()
        for name, fn in order:
            (rf_rates if name == "rf" else gym_rates).append(fn(args.seconds))

    def show(label: str, rates: list[float]) -> float:
        rates = sorted(rates)
        med = rates[len(rates) // 2]
        print(f"{label:<44}{med:>10,.0f}   [{rates[0]:,.0f} .. {rates[-1]:,.0f}]")
        return med

    print()
    print(f"{'configuration (median of trials)':<44}{'steps/s':>10}   range")
    print("-" * 72)
    r1 = show("reinfors  1 env, single-threaded", rf_rates)
    g1 = show("Gymnasium 1 env, single-threaded", gym_rates)
    print("-" * 72)
    print(f"single-threaded: reinfors / Gymnasium = {r1 / g1:5.1f}x")


if __name__ == "__main__":
    main()
