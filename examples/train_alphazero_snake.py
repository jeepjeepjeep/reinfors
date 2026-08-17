"""AlphaZero on snake — the simultaneous + stochastic composition, end to end.

Snake exercises both tree capabilities the sequential connect4 example
(`train_alphazero_example.py`, the minimal reference — read that first) cannot: simultaneous moves
(decoupled/DUCT per-agent search statistics) and declared environment chance (the food respawn,
consumed per `--chance-mode`). The wiring is the same one-seam story: reinfors runs self-play +
PUCT in Rust and calls the `infer` callback — `(N, C*H*W) float32 -> (logits (N, A), values (N,))
float32` — once per pooled round; the net and the gradient step live here.

Differences from the connect4 reference, all snake-driven:
  - the value head is LINEAR, not tanh: snake's z is a discounted *return* (food rewards
    accumulate), not a bounded game outcome;
  - both agents' decisions become records (the engine keeps per-agent trajectories), so one
    self-play game yields two perspectives;
  - `--chance-mode` picks how the search treats the respawn distribution — `committed` (freeze
    `--chance-samples` futures per edge and plan deeply inside them: expectimax's `food_samples`
    treatment, the wide-fan default here) vs `always_resample` (unbiased per-descent draws; thin
    on snake's ~free-cell-wide fan at small sim budgets) — so the modes can be compared by
    measurement on the same budget;
  - eval plays the raw policy head against a uniform-random opponent in an `rf.Env` and reports
    the net side's mean episode reward (food - death), the analogue of the reference's win rate.
    Known limitation, kept as instruction: on snake the searchless head is a weak probe — the
    game's reactive safety lives in the search, so the self-play reward/length curve is the
    primary training signal, and a search-backed eval (net + small-sim search vs random) is the
    right future probe.

    uv run --with torch python examples/train_alphazero_snake.py --iterations 30
    uv run --with torch python examples/train_alphazero_snake.py --chance-mode always_resample
"""

from __future__ import annotations

import argparse
import random
import time
from collections.abc import Callable

import numpy as np
import reinfors as rf
import torch
from torch import nn
from torch.nn import functional as F


class SnakeAzNet(nn.Module):
    """Shared conv trunk -> policy logits (B, 3) + LINEAR value (B,) — snake's z is an unbounded
    discounted return, so no tanh squash on the value head."""

    def __init__(self, obs_shape: tuple[int, ...], n_actions: int, width: int = 32) -> None:
        super().__init__()
        c, h, w = obs_shape
        self.obs_shape = (c, h, w)
        self.trunk = nn.Sequential(
            nn.Conv2d(c, width, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(width, width, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(width * h * w, 2 * width),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(2 * width, n_actions)
        self.value_head = nn.Linear(2 * width, 1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.trunk(obs)
        return self.policy_head(z), self.value_head(z).squeeze(-1)


def make_infer(net: SnakeAzNet, device: str) -> Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]:
    c, h, w = net.obs_shape
    # Default-mode compile is the pattern the V1 benchmark favored (measure it on your
    # workload); CPU runs skip the first-call compile cost, which would dominate a short example.
    forward = torch.compile(net) if device.startswith("cuda") else net

    def infer(obs_batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        with torch.no_grad():
            x = torch.from_numpy(np.ascontiguousarray(obs_batch)).reshape(-1, c, h, w).to(device)
            logits, values = forward(x)
        # Native f32 out: the engine widens exactly; skips the f64 conversion (GPU fast path).
        return logits.cpu().numpy(), values.cpu().numpy()

    return infer


def build_engine(args: argparse.Namespace) -> rf.Engine:
    return rf.Engine(
        rf.games.Snake(grid_size=args.grid, max_ticks=args.max_ticks, num_snakes=args.num_snakes),
        rf.Reward(food=1.0, loss=-1.0),
        rf.policies.AlphaZero(
            num_simulations=args.sims,
            c_puct=2.0,
            temperature=1.0,
            temperature_drop=args.temperature_drop,
            # CLI strings stay ergonomic; the typed handles are the API
            chance=rf.chance_modes.make(
                args.chance_mode,
                **({"samples": args.chance_samples} if args.chance_mode == "committed" else {}),
            ),
        ),
        rf.learners.AlphaZero(gamma=args.gamma),
        n_games=args.n_games,
        seed=args.seed,
        infer_cache=args.infer_cache,
    )


def train_pass(
    net: SnakeAzNet,
    optimizer: torch.optim.Optimizer,
    batch: rf.AlphaZeroBatch,
    batch_size: int,
    device: str,
    rng: np.random.Generator,
) -> tuple[float, float]:
    c, h, w = net.obs_shape
    obs = torch.from_numpy(batch.obs).reshape(-1, c, h, w).to(device)
    pi = torch.from_numpy(batch.policy_targets).float().to(device)
    z = torch.from_numpy(batch.value_targets).float().to(device)
    order = torch.from_numpy(rng.permutation(obs.shape[0]))
    policy_loss_total, value_loss_total, batches = 0.0, 0.0, 0
    for start in range(0, obs.shape[0], batch_size):
        idx = order[start : start + batch_size]
        logits, values = net(obs[idx])
        policy_loss = -(pi[idx] * F.log_softmax(logits, dim=-1)).sum(-1).mean()
        value_loss = F.mse_loss(values, z[idx])
        optimizer.zero_grad()
        (policy_loss + value_loss).backward()  # type: ignore[no-untyped-call]  # torch stub gap
        optimizer.step()
        policy_loss_total += float(policy_loss.item())
        value_loss_total += float(value_loss.item())
        batches += 1
    return policy_loss_total / batches, value_loss_total / batches


def eval_vs_random(net: SnakeAzNet, args: argparse.Namespace, games: int, seed: int) -> float:
    """Search-free probe: the raw policy head (argmax logits) drives one snake against
    uniform-random opponents; returns the net side's mean episode reward (food - death). The
    net seat rotates through every position (placement is not seat-identical, so evaluating
    only seats 0/1 would bias multi-snake results)."""
    c, h, w = net.obs_shape
    rng = random.Random(seed)
    was_training = net.training
    net.eval()
    total = 0.0
    for g in range(games):
        env = rf.Env(
            rf.games.Snake(grid_size=args.grid, max_ticks=args.max_ticks, num_snakes=args.num_snakes),
            rf.Reward(food=1.0, loss=-1.0),
            seed=rng.randrange(2**31),
        )
        net_side = g % args.num_snakes
        episode = 0.0
        ticks = 0
        # rf.Env never truncates (that is an Engine concern), so cap the episode here — otherwise
        # a lone surviving snake under play_to_last plays forever.
        while not env.done() and ticks < args.max_ticks:
            ticks += 1
            actions: dict[int, int] = {}
            for agent in env.active_agents():
                if agent == net_side:
                    with torch.no_grad():
                        x = torch.from_numpy(env.observe(agent)).reshape(1, c, h, w)
                        logits, _ = net(x)
                    actions[agent] = int(logits[0].argmax().item())
                else:
                    actions[agent] = rng.choice(env.legal_actions(agent))
            env.step(actions)
            rewards = env.rewards
            if rewards is not None:
                episode += rewards[net_side]
        total += episode
    net.train(was_training)
    return total / games


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--iterations", type=int, default=30, help="collect+train cycles")
    parser.add_argument("--grid", type=int, default=8)
    parser.add_argument("--num-snakes", type=int, default=2, help="2-8 simultaneous snakes")
    parser.add_argument("--max-ticks", type=int, default=120)
    parser.add_argument("--n-games", type=int, default=8)
    parser.add_argument("--collect-size", type=int, default=768, help="record floor per collect")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--sims", type=int, default=48)
    parser.add_argument("--temperature-drop", type=int, default=20)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--chance-mode", default="committed", choices=["committed", "always_resample", "expand_all"])
    parser.add_argument("--chance-samples", type=int, default=2)
    parser.add_argument("--infer-cache", type=int, default=0)
    parser.add_argument("--eval-games", type=int, default=0, help="0 disables the eval probe")
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    engine = build_engine(args)
    game = rf.games.Snake(grid_size=args.grid, max_ticks=args.max_ticks, num_snakes=args.num_snakes)
    obs_shape = tuple(game.observation_space().shape)
    net = SnakeAzNet(obs_shape, game.action_space().n).to(args.device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    infer = make_infer(net, args.device)

    if args.eval_games:
        print(f"eval[start] mean reward vs random: {eval_vs_random(net, args, args.eval_games, args.seed):+.2f}")
    t0 = time.perf_counter()
    for it in range(1, args.iterations + 1):
        batch = engine.collect(n_records=args.collect_size, infer=infer)
        engine.weights_updated()  # about to train: cached rows from the old weights must not serve
        assert isinstance(batch, rf.AlphaZeroBatch)
        policy_loss, value_loss = train_pass(net, optimizer, batch, args.batch_size, args.device, rng)
        episodes = batch.telemetry["episodes"]
        mean_reward = float(np.mean([np.mean(r) for r, _len, _seeded in episodes])) if episodes else float("nan")
        mean_len = float(np.mean([length for _r, length, _s in episodes])) if episodes else float("nan")
        print(
            f"iter {it:3d}  wall {time.perf_counter() - t0:6.0f}s  records {batch.obs.shape[0]:5d}  "
            f"policy_loss {policy_loss:.3f}  value_loss {value_loss:.3f}  "
            f"ep_reward {mean_reward:+.2f}  ep_len {mean_len:5.1f}"
        )
        if args.eval_games and it % args.eval_every == 0:
            print(
                f"eval[{it}] mean reward vs random: {eval_vs_random(net, args, args.eval_games, args.seed + it):+.2f}"
            )


if __name__ == "__main__":
    main()
