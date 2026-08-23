"""Reproduce the README's CarRacing throughput comparison against Gymnasium.

Four measurements, all stepping the same game with pixel observations:

  1. reinfors, single core: one `rf.Env`, step + observe per tick.
  2. Gymnasium, single core: one `CarRacing-v3` env (its `step` renders the obs).
  3. reinfors, parallel: `Engine.collect` with `n_threads` worker threads driving
     `n_games` episode slots through the PPO policy with a trivial inference
     callback (uniform logits) — the library's real collection path.
  4. Gymnasium, parallel: `AsyncVectorEnv` stepped with random actions, swept
     over several worker counts — its per-step barrier means more processes can
     hurt, so the BEST worker count is reported (strongest baseline).

The comparison is deliberately end-to-end rather than perfectly symmetric: the
reinfors side pays its Python inference-callback round trip and training-batch
assembly, the Gymnasium side pays vectorization IPC; each is the overhead its
users actually experience.

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


def bench_reinfors_parallel(seconds: float, n_threads: int, n_games: int) -> float:
    game = rf.games.CarRacing()
    n_actions = game.action_space().n
    engine = rf.Engine(
        game=game,
        reward=rf.Reward(),
        policy=rf.policies.Ppo(),
        learner=rf.learners.Ppo(),
        n_games=n_games,
        seed=0,
        n_threads=n_threads,
    )

    def infer(obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rows = obs.shape[0]
        return (
            np.zeros((rows, n_actions), dtype=np.float32),
            np.zeros(rows, dtype=np.float32),
        )

    engine.collect(n_records=2 * n_games, infer=infer)  # warm-up
    steps = 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        batch = engine.collect(n_records=4096, infer=infer)
        steps += len(batch.obs)
    return steps / (time.perf_counter() - t0)


def bench_gym_parallel(seconds: float, n_workers: int) -> float:
    import gymnasium as gym

    envs = gym.vector.AsyncVectorEnv([(lambda: gym.make("CarRacing-v3", continuous=False)) for _ in range(n_workers)])
    envs.reset(seed=0)
    rng = np.random.default_rng(0)
    envs.step(rng.integers(5, size=n_workers))  # warm-up
    steps = 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        envs.step(rng.integers(5, size=n_workers))  # autoreset handles episode ends
        steps += n_workers
    rate = steps / (time.perf_counter() - t0)
    envs.close()
    return rate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=10.0, help="wall time per measurement")
    ap.add_argument("--n-threads", type=int, default=10, help="workers per parallel side")
    ap.add_argument("--n-games", type=int, default=64, help="reinfors episode slots")
    args = ap.parse_args()

    r1 = bench_reinfors_single(args.seconds)
    g1 = bench_gym_single(args.seconds)
    rn = bench_reinfors_parallel(args.seconds, args.n_threads, args.n_games)
    gym_workers = sorted({2, 4, args.n_threads})
    gym_runs = {w: bench_gym_parallel(args.seconds, w) for w in gym_workers}
    gw, gn = max(gym_runs.items(), key=lambda kv: kv[1])

    w = args.n_threads
    print()
    print(f"{'configuration':<44}{'steps/s':>10}")
    print("-" * 54)
    print(f"{'reinfors  1 env, 1 core':<44}{r1:>10,.0f}")
    print(f"{'Gymnasium 1 env, 1 core':<44}{g1:>10,.0f}")
    print(f"{f'reinfors  n_threads={w}, n_games={args.n_games}':<44}{rn:>10,.0f}")
    for wk, rate in gym_runs.items():
        best = "  <- best" if wk == gw else ""
        print(f"{f'Gymnasium AsyncVectorEnv, {wk} processes':<44}{rate:>10,.0f}{best}")
    print("-" * 54)
    print(f"single core:            reinfors / Gymnasium = {r1 / g1:5.1f}x")
    print(f"parallel, best-vs-best: reinfors / Gymnasium = {rn / gn:5.1f}x")
    print(f"{w}-thread reinfors vs 1-core Gymnasium       = {rn / g1:5.1f}x")


if __name__ == "__main__":
    main()
