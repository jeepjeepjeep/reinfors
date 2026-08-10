"""Minimal example: train a value network on reinfors' Rust engine.

reinfors owns *data generation* — env dynamics, observations, and the batched selective search all run
in Rust. *You* own *learning* — the network and the gradient step are plain PyTorch here. The two meet
at one seam: the `infer` callback, an `(N, C*H*W) float32` batch -> `(N, K, A) float64` forward that
the search calls once per pooled round. Because it closes over the live network, each `collect`
automatically searches with the current weights — the actor-learner weight sync is implicit.

This is a deliberately tiny, self-contained reference, not a production trainer: no replay buffer, no
checkpoints, no logging. It exists to show the wiring. For a real run (config, replay, TensorBoard,
resume) see snake_RL's `scripts/train_reinfors.py`, which uses this same `Engine` + `infer` contract.

    uv run --with torch python examples/train_treestrap_snake.py --iterations 20
"""

from __future__ import annotations

import argparse
from collections.abc import Callable

import numpy as np
import reinfors as rf
import torch
from torch import nn
from torch.nn import functional as F


class ExampleNet(nn.Module):
    """A small ensemble value net: shared conv trunk + K linear heads -> (B, K, A). Just enough to plug
    a torch model into reinfors; the K heads give the selective search its disagreement signal."""

    def __init__(self, obs_shape: tuple[int, ...], n_actions: int, n_heads: int) -> None:
        super().__init__()
        # `tuple[int, ...]` matches what `game.observation_space().shape` advertises; these games are
        # all planar, so it unpacks to (channels, height, width).
        c, h, w = obs_shape
        self.obs_shape = (c, h, w)
        self.n_actions = n_actions
        self.n_heads = n_heads
        self.trunk = nn.Sequential(nn.Conv2d(c, 16, kernel_size=3, padding=1), nn.ReLU(), nn.Flatten())
        self.head = nn.Linear(16 * h * w, n_heads * n_actions)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.head(self.trunk(obs)).view(-1, self.n_heads, self.n_actions)
        return out


def make_infer(net: ExampleNet, device: str = "cpu") -> Callable[[np.ndarray], np.ndarray]:
    """The search's per-round callback: a flat `(N, C*H*W)` float32 batch -> `(N, K, A)` float64
    forward of `net`. A no-grad forward, run in eval mode and restored afterwards so it's side-effect
    free; the device->host copy moves float32, with the upcast to float64 on the host."""
    net.to(device)
    c, h, w = net.obs_shape

    def infer(obs_batch: np.ndarray) -> np.ndarray:
        was_training = net.training
        net.eval()
        with torch.no_grad():
            x = torch.from_numpy(np.ascontiguousarray(obs_batch)).reshape(-1, c, h, w).to(device)
            q = net(x)
        net.train(was_training)
        out: np.ndarray = q.cpu().numpy()  # native f32; the engine widens exactly
        return out

    return infer


def treestrap_loss(q: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per-head masked Huber: each head regresses its own searched targets, on the records its
    bootstrap mask selects. `q`/`target` are (B, K, A); `mask` is (B, K)."""
    huber = F.smooth_l1_loss(q, target, reduction="none")
    return (mask.unsqueeze(-1) * huber).sum() / (mask.sum().clamp(min=1.0) * q.shape[-1])


def build_engine(
    grid: int, n_heads: int, n_games: int, seed: int, num_snakes: int = 2, policy_name: str = "expectimax"
) -> rf.Engine:
    game = rf.games.Snake(grid_size=grid, max_ticks=200, num_snakes=num_snakes)
    reward = rf.Reward(food=1.0, loss=-10.0, win=10.0, draw=-5.0)
    if policy_name == "mcts":
        # The UCT family (DUCT at simultaneous nodes, any N); TreeStrap consumes the same
        # [K][A] evaluations, so only the policy handle changes.
        policy = rf.policies.Mcts(num_simulations=48)
    else:
        policy = rf.policies.SelectiveExpectimax(
            expansion_budget=32,
            top_k=4,
            max_depth=6,
            beta=1.0,
            chance=rf.chance_modes.Committed(samples=1),
            n_heads=n_heads,
            epsilon=0.1,
        )
    learner = rf.learners.TreeStrap(gamma=0.99, outcome_weight=0.3, bootstrap_p=1.0, interior_targets=False)
    return rf.Engine(game, reward, policy, learner, n_games=n_games, seed=seed)


def train_step(
    net: ExampleNet,
    optimizer: torch.optim.Optimizer,
    obs: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    device: str,
) -> float:
    """One gradient step on a `collect`ed batch; returns the loss."""
    c, h, w = net.obs_shape
    o = torch.from_numpy(obs).reshape(-1, c, h, w).to(device)
    t = torch.from_numpy(target).float().to(device)
    m = torch.from_numpy(mask).to(device)
    loss = treestrap_loss(net(o), t, m)
    optimizer.zero_grad()
    loss.backward()  # type: ignore[no-untyped-call]  # torch stubs leave Tensor.backward untyped
    optimizer.step()
    return float(loss.item())


def default_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--iterations", type=int, default=20, help="collect+train cycles")
    parser.add_argument("--grid", type=int, default=12)
    parser.add_argument("--num-snakes", type=int, default=2, help="2-8 simultaneous snakes")
    parser.add_argument("--policy", choices=["expectimax", "mcts"], default="expectimax")
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--n-games", type=int, default=8, help="parallel games per collect")
    parser.add_argument("--collect-size", type=int, default=256, help="record floor per collect")
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.policy == "mcts" and args.heads != 1:
        # The Mcts composition is single-head (its [1][A] evaluations pair with a K=1 net);
        # ensemble heads are the expectimax family's axis.
        print(f"--policy mcts is single-head: overriding --heads {args.heads} -> 1")
        args.heads = 1

    torch.manual_seed(args.seed)
    game = rf.games.Snake(grid_size=args.grid, num_snakes=args.num_snakes)
    net = ExampleNet(game.observation_space().shape, game.action_space().n, args.heads).to(args.device)
    optimizer = torch.optim.Adam(net.parameters(), lr=2.5e-4)
    engine = build_engine(args.grid, args.heads, args.n_games, args.seed, args.num_snakes, args.policy)
    infer = make_infer(net, args.device)

    print(f"training on {args.device} — grid {args.grid}, {args.heads} heads, {args.n_games} games/collect")
    for it in range(args.iterations):
        obs, target, mask, telemetry = engine.collect(
            n_records=args.collect_size, infer=infer
        )  # search with live weights
        loss = train_step(net, optimizer, obs, target, mask, args.device)
        eps = telemetry["episodes"]
        mean_r = sum(sum(r) / len(r) for r, _len, _s in eps) / len(eps) if eps else float("nan")
        mean_len = sum(_len for _r, _len, _s in eps) / len(eps) if eps else float("nan")
        print(
            f"  iter {it:3d}  records {obs.shape[0]:4d}  loss {loss:.4f}  "
            f"episodes {len(eps)}  ep_reward {mean_r:+.2f}  ep_len {mean_len:5.1f}"
        )


if __name__ == "__main__":
    main()
