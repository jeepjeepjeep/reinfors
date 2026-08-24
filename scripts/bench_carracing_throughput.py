"""Reproduce the README's CarRacing throughput comparison against Gymnasium.

Two measurements, stepping the same game with pixel observations under random
actions: one `rf.Env` (step + observe per tick) and one Gymnasium `CarRacing-v3`
env (its `step` renders the obs). Single core, bare user-stepped loops on both
sides — the symmetric comparison. Parallel collection and collect_stream overlap
multiply reinfors' side further but have no gym-primitive equivalent, so they are
not benchmarked here.

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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=30.0, help="wall time per measurement")
    args = ap.parse_args()

    r1 = bench_reinfors_single(args.seconds)
    g1 = bench_gym_single(args.seconds)

    print()
    print(f"{'configuration':<44}{'steps/s':>10}")
    print("-" * 54)
    print(f"{'reinfors  1 env, 1 core':<44}{r1:>10,.0f}")
    print(f"{'Gymnasium 1 env, 1 core':<44}{g1:>10,.0f}")
    print("-" * 54)
    print(f"single core: reinfors / Gymnasium = {r1 / g1:5.1f}x")


if __name__ == "__main__":
    main()
