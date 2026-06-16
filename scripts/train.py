"""Config-driven reinfors training run with TensorBoard logging, for a like-for-like comparison
against snake_RL.

The hyperparameters below mirror snake_RL's `configs/ensemble_treestrap.yaml` (grid 20 and 3 apples
are snake_RL's env defaults, not in that file). It logs the same TensorBoard scalars snake_RL's
`EnsembleTreeStrapRunner` does — `train/loss`, `train/mean_q`, `train/mean_target_q`,
`episode/reward_{A,B}`, `episode/length`, `search/{max_depth,mean_leaves,mean_rounds,mean_sigma,
root_disagreement}` — plus `throughput/*` (records/s, collect seconds) so both the per-step and the
wall-clock (TensorBoard's "Relative"/"Wall" x-axis) learning curves can be read off directly.

    python scripts/train.py --device mps --log-dir runs/reinfors_ensemble --output ckpts/reinfors
    # snake_RL side, same config:
    #   python scripts/train.py configs/ensemble_treestrap.yaml --device mps --log-dir runs/snake_rl
    tensorboard --logdir runs    # both runs side by side

Caveats for reading the *quality* comparison fairly (the speed comparison is unaffected):
  * Spawn belief — reinfors' search assumes food respawns at the first empty cell (deterministic,
    for cross-impl parity); snake_RL's assumes uniform-random. The env is uniform-random in both, so
    the two *agents* differ. Phase 2 (a stochastic spawn belief) removes this; until then the quality
    curves compare different agents.
  * Parallelism — `--n-games` is a reinfors-only knob (snake_RL runs one self-play env). It changes
    the replay mix, so it affects the per-step curve; it is the source of reinfors' wall-clock win.
  * Cadence — snake_RL trains once per `train_every` env ticks. reinfors trains `--grad-steps` per
    `collect(--collect-size)`; since `collect` counts every record (roots + interior), the fresh
    records-per-gradient-step is ~`collect_size / grad_steps`. Tune these to match snake_RL's measured
    records/step (visible from `throughput/*`) for a tight per-step comparison.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, cast

import reinfors
import torch
from reinfors.training import BootstrappedQNetwork, CollectReport, StepMetrics, train

# --- snake_RL configs/ensemble_treestrap.yaml (grid_size / initial_food_count are env defaults) ---
GRID = 20
INITIAL_FOOD = 3
INITIAL_LENGTH = 3
PLAY_TO_LAST = False
WIN_FOOD_LEAD = 10
MAX_TICKS = 750
GAMMA = 0.99
# reward: (step, food, loss, draw, kill, win, survival)
REWARD = (0.0, 1.0, -10.0, -5.0, 5.0, 10.0, 0.0)
N_HEADS = 10
BOOTSTRAP_P = 0.6
PRIOR_SCALE = 2.5
OPPONENT = "distributional"
OPP_TEMPERATURE = 1.0
OPP_FLOOR = 0.01
EXPANSION_BUDGET = 64
TOP_K = 8
MAX_DEPTH = 12
BETA = 1.0
FOOD_SAMPLES = 1
OUTCOME_WEIGHT = 0.3
INTERIOR_TARGETS = True
EPSILON = 0.0
LR = 2.5e-4
BATCH_SIZE = 512
BUFFER_CAPACITY = 100_000
MIN_BUFFER_SIZE = 2_000
# snake_RL trains once per `train_every` env ticks and logs against the env-tick count, so its "step"
# advances by `train_every` per gradient update. We scale reinfors' gradient-step index by the same
# factor when logging, so the TensorBoard step axis is directly comparable (no 4x offset to undo).
TRAIN_EVERY = 4


def build_engine(n_games: int, seed: int) -> object:
    return reinfors._reinfors.Engine(
        n_games,
        GRID,
        INITIAL_LENGTH,
        PLAY_TO_LAST,
        WIN_FOOD_LEAD,
        INITIAL_FOOD,
        GAMMA,
        BETA,
        EXPANSION_BUDGET,
        TOP_K,
        MAX_DEPTH,
        REWARD,
        OPPONENT,
        OPP_TEMPERATURE,
        OPP_FLOOR,
        EPSILON,
        MAX_TICKS,
        N_HEADS,
        OUTCOME_WEIGHT,
        INTERIOR_TARGETS,
        BOOTSTRAP_P,
        seed,
        FOOD_SAMPLES,
    )


def default_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-dir", default="runs/reinfors_ensemble")
    parser.add_argument("--output", default=None, help="checkpoint dir (default: <log-dir>/ckpts)")
    parser.add_argument(
        "--episodes", type=int, default=20_000, help="self-play episodes to train (snake_RL num_episodes)"
    )
    parser.add_argument("--max-iterations", type=int, default=None, help="optional cap on collect+train cycles")
    parser.add_argument("--n-games", type=int, default=16, help="parallel games (reinfors-only knob)")
    parser.add_argument("--collect-size", type=int, default=256, help="record floor per collect")
    parser.add_argument("--grad-steps", type=int, default=1, help="gradient steps per collect")
    parser.add_argument("--checkpoint-every", type=int, default=500, help="episodes between checkpoints")
    args = parser.parse_args()

    from torch.utils.tensorboard.writer import SummaryWriter

    out_dir = Path(args.output) if args.output else Path(args.log_dir) / "ckpts"
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=args.log_dir)
    print(
        f"reinfors training — device={args.device} grid={GRID} heads={N_HEADS} "
        f"episodes={args.episodes} log_dir={args.log_dir}"
    )

    torch.manual_seed(args.seed)
    net = BootstrappedQNetwork((5, GRID, GRID), 3, N_HEADS, prior_scale=PRIOR_SCALE)
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)
    engine = build_engine(args.n_games, args.seed)

    state = {"episode": 0, "step": 0, "next_ckpt": args.checkpoint_every}

    def on_step(step: int, m: StepMetrics) -> None:
        s = step * TRAIN_EVERY  # scale to snake_RL's env-tick step axis (see TRAIN_EVERY)
        state["step"] = s
        writer.add_scalar("train/loss", m.loss, s)
        writer.add_scalar("train/mean_q", m.mean_q, s)
        writer.add_scalar("train/mean_target_q", m.mean_target_q, s)

    def on_collect(it: int, r: CollectReport) -> None:
        t = cast("dict[str, Any]", r.telemetry)  # heterogeneous telemetry dict from the Rust binding
        for reward_a, reward_b, length in t["episodes"]:
            writer.add_scalar("episode/reward_A", reward_a, state["episode"])
            writer.add_scalar("episode/reward_B", reward_b, state["episode"])
            writer.add_scalar("episode/length", length, state["episode"])
            state["episode"] += 1
        s = state["step"]  # search/throughput share the scaled (env-tick) step axis with train/*
        writer.add_scalar("search/max_depth", t["max_depth"], s)
        writer.add_scalar("search/mean_leaves", t["mean_leaves"], s)
        writer.add_scalar("search/mean_rounds", t["mean_rounds"], s)
        writer.add_scalar("search/mean_sigma", t["mean_sigma"], s)
        writer.add_scalar("search/root_disagreement", t["mean_disagreement"], s)
        writer.add_scalar("throughput/records_per_s", r.records / max(r.seconds, 1e-9), s)
        writer.add_scalar("throughput/collect_seconds", r.seconds, s)
        if state["episode"] >= state["next_ckpt"]:
            path = out_dir / f"ckpt_ep{state['episode']}.pt"
            grad_steps = s // TRAIN_EVERY
            torch.save({"net": net.state_dict(), "episode": state["episode"], "step": grad_steps}, path)
            print(f"  {state['episode']} episodes, {grad_steps} gradient steps -> {path}")
            while state["next_ckpt"] <= state["episode"]:
                state["next_ckpt"] += args.checkpoint_every

    train(
        engine,
        net,
        optimizer,
        max_episodes=args.episodes,
        iterations=args.max_iterations,
        collect_size=args.collect_size,
        batch_size=BATCH_SIZE,
        grad_steps_per_collect=args.grad_steps,
        buffer_capacity=BUFFER_CAPACITY,
        min_buffer_size=MIN_BUFFER_SIZE,
        device=args.device,
        seed=args.seed,
        on_step=on_step,
        on_collect=on_collect,
    )
    torch.save({"net": net.state_dict(), "episode": state["episode"], "step": state["step"]}, out_dir / "final.pt")
    writer.close()
    print(f"done — {state['episode']} episodes, {state['step'] // TRAIN_EVERY} gradient steps")


if __name__ == "__main__":
    main()
