"""Throughput benchmarks for reinfors' Rust core (Phase 1: reinfors' own performance, reproducible).

reinfors is a *data-generation* engine: its product is training records per second, produced by driving
many games in parallel through the batched search. So the headline is **records/sec from
`Engine.collect`**, reported two ways:

* **Inference-cost sweep.** The search calls a Python `infer` once per pooled round — on the critical
  path. We sweep a synthetic net from free (zeros) to heavy, so you can see where throughput is
  engine-bound vs inference-bound. The synthetic forward does representative matmul FLOPs but returns a
  *fixed* value, so the search's dynamics (hence how many records a collect yields) stay constant across
  levels — only inference wall-time varies. Numpy's BLAS releases the GIL during the matmul, the same
  way a real CPU/GPU net does, so the parallel picture is representative.
* **Parallel scaling.** records/sec vs `n_games`, with the record budget scaled so every point reaches
  steady state (the naive fixed-budget version under-measures large `n_games`, which finish in a round
  or two). Shows the rayon speedup across cores.

Plus a raw single-`rf.Env` stepping ceiling. `search` = SelectiveExpectimax + TreeStrap (the product);
`reactive` = EpsilonGreedyQ + DQN (minimal search ~ the rollout ceiling with a cheap net).

    uv run --with numpy python scripts/benchmark.py            # full run (~minutes)
    uv run --with numpy python scripts/benchmark.py --quick    # fast smoke

reinfors MUST be a release build for any of this to mean anything — a debug build (`maturin develop`
without `--release`) runs the Rust core ~10x slower. The harness checks `core_build_profile()` and warns
loudly on a debug build; install a wheel (`pip install reinfors`) or `maturin develop --release`.

Phase 2 — true cross-framework head-to-heads (Pgx on GPU, OpenSpiel MCTS) — needs those environments and
a pinned machine, so it lives in a separate, manually-run script rather than here.
"""

from __future__ import annotations

import argparse
import math
import os
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from statistics import median
from time import perf_counter
from typing import Any

import numpy as np
import reinfors as rf

_N_HEADS = 4
Infer = Callable[[np.ndarray], np.ndarray]

# Synthetic-net cost levels: (label, hidden units). 0 = zeros (pure engine+search floor). Cost per
# forward is ~ N*dim*hidden (a one-hidden-layer MLP), so these span a light to a heavy value net.
INFER_LEVELS: list[tuple[str, int]] = [("zeros", 0), ("small", 256), ("medium", 1024), ("large", 4096)]


@dataclass
class GameCfg:
    make: Callable[[int], Any]  # grid_size -> game handle (grid ignored by fixed-board games)
    reward: rf.Reward
    grid: int
    single_agent: bool


GAMES: dict[str, GameCfg] = {
    "snake": GameCfg(
        make=lambda g: rf.games.Snake(grid_size=g, max_ticks=200),
        reward=rf.Reward(food=1.0, loss=-1.0, win=1.0, draw=-0.5),
        grid=12,
        single_agent=False,
    ),
    "connect4": GameCfg(
        make=lambda _g: rf.games.Connect4(),
        reward=rf.Reward(win=1.0, loss=-1.0, draw=0.0),
        grid=0,
        single_agent=False,
    ),
    "gridworld": GameCfg(
        make=lambda g: rf.games.GridWorld(size=g, max_ticks=200),
        reward=rf.Reward(goal=1.0),
        grid=8,
        single_agent=True,
    ),
}


def _obs_dim(name: str, grid: int) -> int:
    return int(math.prod(GAMES[name].make(grid).observation_space().shape))


def _infer(dim: int, action_count: int, hidden: int) -> Infer:
    """A `(N, dim) -> (N, K, A)` forward whose cost is set by `hidden`. It returns a *fixed* zero output
    (so search behaviour is identical across cost levels) but does `hidden`-sized matmul FLOPs first, so
    only the inference wall-time — the thing we're sweeping — changes."""
    if hidden == 0:
        return lambda arr: np.zeros((arr.shape[0], _N_HEADS, action_count), dtype=np.float64)
    rng = np.random.default_rng(0)
    w1 = rng.standard_normal((dim, hidden), dtype=np.float32)  # dominant cost: N*dim*hidden
    w2 = rng.standard_normal((hidden, _N_HEADS * action_count), dtype=np.float32)

    def infer(arr: np.ndarray) -> np.ndarray:
        h = np.maximum(arr @ w1, 0.0)  # (N, hidden) — GIL released during the BLAS call
        _ = h @ w2  # project to the output width; result discarded (a fixed value is returned)
        return np.zeros((arr.shape[0], _N_HEADS, action_count), dtype=np.float64)

    return infer


def _engine(name: str, grid: int, mode: str, n_games: int, seed: int = 0) -> rf.Engine:
    cfg = GAMES[name]
    if mode == "search":
        policy = rf.policies.SelectiveExpectimax(
            expansion_budget=32,
            top_k=4,
            max_depth=6,
            beta=1.0,
            food_samples=1,
            n_heads=_N_HEADS,
            epsilon=0.1,
            opponent="uniform",
            opp_temperature=1.0,
            opp_floor=0.1,
        )
        learner = rf.learners.TreeStrap(gamma=0.99, outcome_weight=0.3, bootstrap_p=1.0, interior_targets=False)
    else:  # reactive
        policy = rf.policies.EpsilonGreedyQ(n_heads=_N_HEADS, epsilon=0.1)
        learner = rf.learners.Dqn(bootstrap_p=1.0)
    return rf.Engine(cfg.make(grid), cfg.reward, policy, learner, n_games=n_games, seed=seed)


def _throughput(work: Callable[[], int], repeats: int) -> float:
    """Median of `units / seconds` over `repeats` timed runs (one untimed warm-up first)."""
    work()
    rates = []
    for _ in range(repeats):
        t0 = perf_counter()
        units = work()
        rates.append(units / (perf_counter() - t0))
    return median(rates)


def bench_collect(name: str, *, grid: int, mode: str, n_games: int, records: int, repeats: int, hidden: int) -> float:
    engine = _engine(name, grid, mode, n_games)
    infer = _infer(_obs_dim(name, grid), GAMES[name].make(grid).action_space().n, hidden)
    return _throughput(lambda: int(engine.collect(records, infer).obs.shape[0]), repeats)


def bench_env_steps(name: str, *, grid: int, steps: int, repeats: int, seed: int = 0) -> float:
    env = rf.Env(GAMES[name].make(grid), seed=seed)
    rng = random.Random(seed)

    def work() -> int:
        env.reset()
        for _ in range(steps):
            if env.done():
                env.reset()
            env.step({a: rng.choice(env.legal_actions(a)) for a in env.active_agents()})
        return steps

    return _throughput(work, repeats)


def _table(title: str, header: tuple[str, ...], rows: Sequence[tuple[str, ...]], note: str = "") -> None:
    widths = [max(len(header[i]), *(len(r[i]) for r in rows)) for i in range(len(header))]

    def fmt(cells: tuple[str, ...]) -> str:
        return "  ".join(c.rjust(widths[i]) if i else c.ljust(widths[i]) for i, c in enumerate(cells))

    print(f"\n{title}")
    if note:
        print(note)
    print(fmt(header))
    print("-" * (sum(widths) + 2 * (len(header) - 1)))
    for r in rows:
        print(fmt(r))


def _fmt(x: float) -> str:
    return f"{x:,.0f}"


def _warn_if_not_release() -> None:
    profile = rf.core_build_profile()
    if profile == "release":
        return
    bar = "!" * 78
    print(
        f"\n{bar}\n"
        f"!! reinfors was built in '{profile}' mode — the Rust core runs ~10x slower, so EVERY\n"
        f"!! number below is meaningless. Rebuild release: `maturin develop --release`, or\n"
        f"!! install a wheel (`maturin build` / `pip install reinfors`).\n"
        f"{bar}"
    )


def run(args: argparse.Namespace) -> None:
    _warn_if_not_release()
    cores = os.cpu_count() or 1
    n_games = args.n_games or max(1, cores - 2)
    scaling = [n for n in ([1, 2, 4] if args.quick else [1, 2, 4, 8, 16, 32]) if n <= 4 * cores]
    scale_per_game = 16 if args.quick else 128  # budget scales with n_games -> comparable rounds per point
    levels = INFER_LEVELS[:2] if args.quick else INFER_LEVELS
    print(
        f"reinfors {rf.__version__} ({rf.core_build_profile()}) — {cores} cores | "
        f"n_games={n_games} records={args.records} repeats={args.repeats}"
    )

    def collect(name: str, mode: str, hidden: int, n: int = n_games, budget: int = args.records) -> str:
        grid = GAMES[name].grid
        return _fmt(
            bench_collect(name, grid=grid, mode=mode, n_games=n, records=budget, repeats=args.repeats, hidden=hidden)
        )

    def steps(name: str) -> str:
        return _fmt(bench_env_steps(name, grid=GAMES[name].grid, steps=args.steps, repeats=args.repeats))

    _table(
        "Data generation — snake records/sec across inference cost (higher is better)",
        ("infer", "reactive", "search"),
        [(f"{label} ({h})", collect("snake", "reactive", h), collect("snake", "search", h)) for label, h in levels],
        note="rows = synthetic net (label + hidden units); 'zeros' is the engine+search floor.",
    )

    _table(
        "Data generation — records/sec per game (zeros infer)",
        ("game", "reactive", "search"),
        [(name, collect(name, "reactive", 0), collect(name, "search", 0)) for name in GAMES],
    )

    base = 0.0
    rows = []
    for n in scaling:
        rate = bench_collect(
            "snake",
            grid=GAMES["snake"].grid,
            mode="search",
            n_games=n,
            records=scale_per_game * n,
            repeats=args.repeats,
            hidden=0,
        )
        base = base or rate
        rows.append((str(n), _fmt(rate), f"{rate / base:.2f}x"))
    _table(
        "Parallel scaling — snake, search, zeros infer (records/sec vs n_games)",
        ("n_games", "records/sec", "speedup"),
        rows,
        note=f"budget = {scale_per_game} records/game so every point runs ~{scale_per_game} rounds; "
        "the GIL-holding Python infer caps the speedup — a GIL-releasing net scales further.",
    )

    _table(
        "Raw env stepping — env-ticks/sec (single rf.Env, random actions)",
        ("game", "ticks/sec"),
        [(name, steps(name)) for name in GAMES],
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--records", type=int, default=512, help="record floor per collect (sweep tables)")
    p.add_argument("--steps", type=int, default=50_000, help="env ticks per raw-stepping run")
    p.add_argument("--n-games", type=int, default=0, help="parallel games (0 = cpu_count - 2)")
    p.add_argument("--repeats", type=int, default=3, help="timed runs per measurement (median reported)")
    p.add_argument("--quick", action="store_true", help="tiny/fast run for a smoke check")
    args = p.parse_args()
    if args.quick:
        args.records, args.steps, args.repeats = 64, 2_000, 1
    run(args)


if __name__ == "__main__":
    main()
