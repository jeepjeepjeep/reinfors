"""Throughput benchmark for the reinfors rollout + search: ms per searched decision through the
end-to-end `Engine.collect` loop.

IMPORTANT: build the extension in RELEASE, or these numbers are meaningless. A debug build leaves
Rust's iterator and allocation paths unoptimised and inflates the per-search cost several-fold. Use:

    uvx maturin@1.14.0 build --release --out dist-release && \\
        uv run --with dist-release/*.whl --with numpy python scripts/benchmark.py

The per-search Rust work is rayon-parallel across the pooled requests; the controlled measurement of
that is the single- vs multi-threaded run — set `RAYON_NUM_THREADS=1` and compare (the script prints
the active setting). The pooling advantage compounds with GPU inference (`--net`), where one large
batched forward per round dominates.
"""

from __future__ import annotations

import argparse
import os
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


def make_infer(k: int) -> object:
    def infer(arr: np.ndarray) -> np.ndarray:
        return q_batch(arr, k)

    return infer


def make_net_infer(grid: int, k: int, device: str) -> object:
    """A real `BootstrappedQNetwork` forward on `device` (e.g. "mps") as the value function, via the
    production `reinfors.training.make_infer`. Lets the benchmark measure the GPU-inference regime the
    pooling design targets, instead of the cheap CPU proxy."""
    import torch
    from reinfors.training import BootstrappedQNetwork
    from reinfors.training import make_infer as net_infer

    torch.manual_seed(0)
    net = BootstrappedQNetwork((5, grid, grid), 3, k)
    return net_infer(net, device)


def build_infer(args: argparse.Namespace) -> object:
    return make_net_infer(args.grid, args.heads, args.device) if args.net else make_infer(args.heads)


def bench_collect(args: argparse.Namespace) -> None:
    engine = reinfors.Engine(
        reinfors.games.Snake(
            grid_size=args.grid,
            initial_length=3,
            food=3,
            play_to_last=False,
            win_food_lead=None,
            reward=reinfors.Reward(
                step=_REWARD[0],
                food=_REWARD[1],
                loss=_REWARD[2],
                draw=_REWARD[3],
                kill=_REWARD[4],
                win=_REWARD[5],
                survival=_REWARD[6],
            ),
        ),
        reinfors.policies.SelectiveExpectimax(
            expansion_budget=args.budget,
            top_k=4,
            max_depth=args.max_depth,
            beta=1.0,
            food_samples=1,
            n_heads=args.heads,
            epsilon=0.1,
            opponent="uniform",
            opp_temperature=1.0,
            opp_floor=0.1,
        ),
        # interior off so records == decisions (the quantity the benchmark times)
        reinfors.learners.TreeStrap(gamma=0.99, outcome_weight=0.5, bootstrap_p=1.0, interior_targets=False),
        n_games=args.games,
        max_ticks=200,
        seed=0,
    )
    infer = build_infer(args)
    engine.collect(args.games * 4, infer)  # warm up
    t0 = time.perf_counter()
    obs, _tgt, _mask, _stats = engine.collect(args.records, infer)
    dt = time.perf_counter() - t0
    n = obs.shape[0]
    print(f"  Engine.collect : {n:5d} decisions, {dt / n * 1e3:7.3f} ms/decision, {n / dt:7.0f} dec/s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=int, default=20)
    parser.add_argument("--games", type=int, default=16, help="parallel games (pooled requests per round)")
    parser.add_argument("--heads", type=int, default=10)
    parser.add_argument("--budget", type=int, default=64, help="expansion_budget")
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--records", type=int, default=1000, help="decisions to time")
    parser.add_argument("--net", action="store_true", help="use a real BootstrappedQNetwork as the value function")
    parser.add_argument("--device", default="cpu", help="torch device for --net (e.g. cpu, mps, cuda)")
    args = parser.parse_args()

    threads = os.environ.get("RAYON_NUM_THREADS", "all cores (default)")
    print(
        f"reinfors benchmark — grid={args.grid} games={args.games} heads={args.heads} "
        f"budget={args.budget} depth={args.max_depth}, "
        f"value_fn={'net@' + args.device if args.net else 'numpy'}, RAYON_NUM_THREADS={threads}"
    )
    print("  (numbers are only meaningful for a RELEASE extension build — see module docstring)\n")
    bench_collect(args)


if __name__ == "__main__":
    main()
