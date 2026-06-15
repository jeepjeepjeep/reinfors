"""Throughput benchmark for the reinfors rollout + search.

Reports ms per searched decision for (1) the end-to-end `Engine.collect` loop and (2) an isolated
pooled search on a fixed state, and — with `--baseline`, where a sibling snake_RL checkout is
importable — the same isolated search through snake_RL's planner on an identical (vectorised) value
function, as a rough cross-implementation comparison.

The per-search Rust work is rayon-parallel across the pooled requests; the controlled measurement of
that is the single- vs multi-threaded run — set `RAYON_NUM_THREADS=1` and compare (the script prints
the active setting). On a CPU value function the per-round work is dominated by the serial path (obs
marshalling across PyO3 + the infer call), so parallel scaling is modest; the design target for the
Rust + pooling win is GPU inference, where one large batched forward per round dominates.

    python scripts/benchmark.py                 # reinfors only
    python scripts/benchmark.py --baseline       # + snake_RL planner comparison
    RAYON_NUM_THREADS=1 python scripts/benchmark.py   # single-threaded reinfors
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import reinfors

_REWARD = (0.0, 0.0, -10.0, -6.0, 20.0, 20.0, 0.0)  # step, food, loss, draw, kill, win, survival


def q_batch(arr: np.ndarray, k: int) -> np.ndarray:
    # A vectorised (N, K, 3) value function — one batched op, a fair proxy for a real net forward
    # (not a per-row Python loop, which would make the workload Python-bound rather than search-bound).
    s = np.asarray(arr, dtype=np.float64).sum(axis=1)[:, None]  # (N, 1)
    h = np.arange(k)[None, :]  # (1, K)
    a0 = np.sin(s + 0.5 * h)
    a1 = np.cos(0.5 * s + 0.3 * h)
    a2 = np.sin(0.2 * s - 0.7 * h)
    return np.stack([a0, a1, a2], axis=-1)  # (N, K, 3)


def q_heads(obs: object, k: int) -> np.ndarray:
    return q_batch(np.asarray(obs).ravel()[None, :], k)[0]


def make_infer(k: int) -> object:
    def infer(arr: np.ndarray) -> np.ndarray:
        return q_batch(arr, k)

    return infer


def bench_collect(args: argparse.Namespace) -> None:
    engine = reinfors._reinfors.Engine(
        args.games,
        args.grid,
        3,
        False,
        None,
        3,  # n_games, grid, initial_length, play_to_last, win_food_lead, initial_food_count
        0.99,
        1.0,
        args.budget,
        4,
        args.max_depth,  # gamma, beta, expansion_budget, top_k, max_depth
        _REWARD,
        "uniform",
        1.0,
        0.1,  # reward, opponent, opp_temperature, opp_floor
        0.1,
        200,
        args.heads,  # epsilon, max_ticks, n_heads
        0.5,
        False,
        1.0,  # outcome_weight, interior_targets (off so records == decisions), bootstrap_p
        0,
    )
    infer = make_infer(args.heads)
    engine.collect(args.games * 4, infer)  # warm up
    t0 = time.perf_counter()
    obs, _tgt, _mask = engine.collect(args.records, infer)
    dt = time.perf_counter() - t0
    _report("Engine.collect", obs.shape[0], dt)


def _report(label: str, n: int, dt: float) -> None:
    print(f"  {label:16s}: {n:5d} decisions, {dt / n * 1e3:7.3f} ms/decision, {n / dt:7.0f} dec/s")


def _fixed_state() -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    # A clear mid-board state where both snakes have room to search several plies deep.
    return [(6, 5), (6, 4), (6, 3)], [(6, 16), (6, 17), (6, 18)]


def bench_pooled_search(args: argparse.Namespace) -> float:
    a, b = _fixed_state()
    infer = make_infer(args.heads)
    search_args = (0.99, 1.0, args.budget, 4, args.max_depth, _REWARD, "uniform", 1.0, 0.1)

    def fresh_envs() -> list[object]:
        envs = []
        for _ in range(args.games):
            e = reinfors._reinfors.SnakeEnv(args.grid, 3, True, None)
            e.set_snakes(a, 3, True, b, 2, True)  # dirs: Right (3), Left (2)
            envs.append(e)
        return envs

    agents = [i % 2 for i in range(args.games)]
    reinfors._reinfors.selective_search_many(fresh_envs(), agents, *search_args, infer)  # warm up
    t0 = time.perf_counter()
    for _ in range(args.reps):
        reinfors._reinfors.selective_search_many(fresh_envs(), agents, *search_args, infer)
    dt = time.perf_counter() - t0
    n = args.reps * args.games
    _report("reinfors search", n, dt)
    return dt / n * 1e3


def bench_baseline(args: argparse.Namespace) -> float | None:
    src = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "snake_RL", "src"))
    if os.path.isdir(src) and src not in sys.path:
        sys.path.insert(0, src)
    try:
        from snake_rl.agent.model_based.expectimax import UniformOpponent
        from snake_rl.agent.model_based.selective_expectimax import SelectiveExpectimaxPlanner
        from snake_rl.agent.shared.observation import EgocentricGridObservation
        from snake_rl.agent.shared.reward import MinimalReward
        from snake_rl.environment.base import RELATIVE_ACTIONS, Action, BaseSnakeEnv, Snake, WorldState
        from snake_rl.environment.clean import CleanSnakeEnv
    except ImportError:
        print("  snake_RL not importable — skipping baseline.")
        return None

    obs_builder = EgocentricGridObservation(grid_size=args.grid)
    k = args.heads

    def qf(o: object) -> np.ndarray:
        return q_heads(o, k)

    a, b = _fixed_state()
    from collections import deque

    state = WorldState(
        snakes={
            BaseSnakeEnv.PLAYER_A: Snake(deque(a), Action.RIGHT),
            BaseSnakeEnv.PLAYER_B: Snake(deque(b), Action.LEFT),
        },
        food=set(),
        grid_size=args.grid,
    )
    planner = SelectiveExpectimaxPlanner(
        rules=CleanSnakeEnv(grid_size=args.grid, initial_food_count=0, play_to_last=True, seed=0),
        reward_fn=MinimalReward(food=0.0, loss=-10.0, draw=-6.0, kill=20.0, win=20.0, step=0.0, survival=0.0),
        obs_builder=obs_builder,
        q_values=qf,
        opp_model=UniformOpponent(RELATIVE_ACTIONS),
        actions=RELATIVE_ACTIONS,
        gamma=0.99,
        q_value_batch=lambda obss: q_batch(np.asarray(obss).reshape(len(obss), -1), k),
        seed=0,
        expansion_budget=args.budget,
        top_k=4,
        max_depth=args.max_depth,
        beta=1.0,
    )
    requests = [(BaseSnakeEnv.PLAYER_A if i % 2 == 0 else BaseSnakeEnv.PLAYER_B, False) for i in range(args.games)]
    planner.search_many(state, requests)  # warm up
    t0 = time.perf_counter()
    for _ in range(args.reps):
        planner.search_many(state, requests)
    dt = time.perf_counter() - t0
    n = args.reps * args.games
    _report("snake_RL search", n, dt)
    return dt / n * 1e3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=int, default=20)
    parser.add_argument("--games", type=int, default=16, help="pooled requests per search round")
    parser.add_argument("--heads", type=int, default=10)
    parser.add_argument("--budget", type=int, default=64, help="expansion_budget")
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--records", type=int, default=1000, help="decisions for the collect benchmark")
    parser.add_argument("--reps", type=int, default=30, help="search-round reps for the pooled benchmark")
    parser.add_argument("--baseline", action="store_true", help="compare against snake_RL's planner")
    args = parser.parse_args()

    threads = os.environ.get("RAYON_NUM_THREADS", "all cores (default)")
    print(
        f"reinfors benchmark — grid={args.grid} games={args.games} heads={args.heads} "
        f"budget={args.budget} depth={args.max_depth}, RAYON_NUM_THREADS={threads}\n"
    )
    bench_collect(args)
    rein_ms = bench_pooled_search(args)
    if args.baseline:
        base_ms = bench_baseline(args)
        if base_ms is not None:
            print(
                f"\n  reinfors / snake_RL per-decision ratio: {rein_ms / base_ms:.2f}x "
                f"(>1 means reinfors is slower; same value function and search params)."
            )


if __name__ == "__main__":
    main()
