"""Minimal example: AlphaZero on connect4 via reinfors' Rust engine.

reinfors owns *data generation* — self-play, the PUCT search (root Dirichlet noise + opening
temperature), and record assembly all run in Rust. *You* own *learning* — the two-headed network and
the gradient step are plain PyTorch here. The two meet at one seam: the `infer` callback, an
`(N, C*H*W) float32` batch -> `(policy_logits (N, A), values (N,)) float64` tuple, one forward for
both heads, called once per pooled search round with the live weights.

Each collect yields `(obs, pi, z)`: pi is the root visit distribution (the policy head's
cross-entropy target — the search improves on the prior, the prior distills the search), z the
realized game outcome (the value head's MSE target). The optional eval plays the *raw policy head*
(no search) against a uniform-random opponent — a cheap probe that the distilled prior is learning.

This is a deliberately tiny, self-contained reference, not a production trainer: no replay buffer,
no checkpoints, no logging. It exists to show the wiring.

    uv run --with torch python scripts/train_alphazero_example.py --iterations 40
"""

from __future__ import annotations

import argparse
import random
from collections.abc import Callable

import numpy as np
import reinfors as rf
import torch
from torch import nn
from torch.nn import functional as F


class AlphaZeroNet(nn.Module):
    """Two-headed net: shared conv trunk -> policy logits (B, A) + tanh value (B,). Just enough to
    plug a torch model into the AlphaZero family; both heads come from one trunk pass."""

    def __init__(self, obs_shape: tuple[int, ...], n_actions: int) -> None:
        super().__init__()
        c, h, w = obs_shape
        self.obs_shape = (c, h, w)
        self.trunk = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * h * w, 64),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(64, n_actions)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.trunk(obs)
        return self.policy_head(z), torch.tanh(self.value_head(z)).squeeze(-1)


def make_infer(net: AlphaZeroNet, device: str) -> Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """The search's per-round callback: a flat `(N, C*H*W)` float32 batch -> the AlphaZero tuple
    `(policy_logits (N, A) f64, values (N,) f64)`. No-grad, eval-mode, side-effect free."""
    net.to(device)
    c, h, w = net.obs_shape

    def infer(obs_batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        was_training = net.training
        net.eval()
        with torch.no_grad():
            x = torch.from_numpy(np.ascontiguousarray(obs_batch)).reshape(-1, c, h, w).to(device)
            logits, values = net(x)
        net.train(was_training)
        return logits.cpu().double().numpy(), values.cpu().double().numpy()

    return infer


def alphazero_loss(
    logits: torch.Tensor, values: torch.Tensor, pi: torch.Tensor, z: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """The paper's two terms: cross-entropy of the policy head against the search's visit
    distribution, MSE of the value head against the realized outcome."""
    policy_loss = -(pi * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
    value_loss = F.mse_loss(values, z)
    return policy_loss, value_loss


def build_engine(n_games: int, seed: int) -> rf.Engine:
    # Handle defaults are the AlphaZero conventions: noise_epsilon 0.25, noise_alpha 0.3,
    # temperature 1.0 for the first 8 plies. gamma=1 + win/loss=±1 gives the paper's z.
    return rf.Engine(
        rf.games.Connect4(),
        rf.Reward(win=1.0, loss=-1.0),
        rf.policies.AlphaZero(num_simulations=48),
        rf.learners.AlphaZero(gamma=1.0),
        n_games=n_games,
        seed=seed,
    )


def train_pass(
    net: AlphaZeroNet,
    optimizer: torch.optim.Optimizer,
    obs: np.ndarray,
    pi: np.ndarray,
    z: np.ndarray,
    batch_size: int,
    device: str,
) -> tuple[float, float]:
    """One shuffled minibatch pass over a collected batch; returns mean (policy, value) losses."""
    c, h, w = net.obs_shape
    o = torch.from_numpy(obs).reshape(-1, c, h, w).to(device)
    p = torch.from_numpy(pi).float().to(device)
    v = torch.from_numpy(z).float().to(device)
    perm = torch.randperm(o.shape[0])
    p_sum, v_sum, batches = 0.0, 0.0, 0
    for start in range(0, o.shape[0], batch_size):
        idx = perm[start : start + batch_size]
        logits, values = net(o[idx])
        policy_loss, value_loss = alphazero_loss(logits, values, p[idx], v[idx])
        optimizer.zero_grad()
        (policy_loss + value_loss).backward()
        optimizer.step()
        p_sum += float(policy_loss.item())
        v_sum += float(value_loss.item())
        batches += 1
    return p_sum / batches, v_sum / batches


def eval_vs_random(net: AlphaZeroNet, device: str, games: int, seed: int) -> float:
    """Search-free probe: the raw policy head (argmax logits) vs a uniform-random opponent,
    alternating sides. Returns the net's win rate (draws count half)."""
    c, h, w = net.obs_shape
    rng = random.Random(seed)
    was_training = net.training  # restored below — leaving eval mode set would silently
    net.eval()  # break a later train_pass if the trunk ever grows BN/dropout
    score = 0.0
    for g in range(games):
        env = rf.Env(rf.games.Connect4(), seed=rng.randrange(2**31))
        net_side = g % 2
        events = ["", ""]
        while not env.done():
            agent = env.active_agents()[0]
            if agent == net_side:
                with torch.no_grad():
                    x = torch.from_numpy(env.observe(agent)).reshape(1, c, h, w).to(device)
                    logits, _ = net(x)
                action = int(logits[0].argmax().item())
            else:
                action = rng.choice(env.legal_actions(agent))
            events = env.step({agent: action})
        if events[net_side] == "win":
            score += 1.0
        elif events[net_side] == "draw":
            score += 0.5
    net.train(was_training)
    return score / games


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--iterations", type=int, default=40, help="collect+train cycles")
    parser.add_argument("--n-games", type=int, default=8, help="parallel self-play games per collect")
    parser.add_argument("--collect-size", type=int, default=512, help="record floor per collect")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-every", type=int, default=10, help="iterations between eval probes (0 = off)")
    parser.add_argument("--eval-games", type=int, default=50)
    parser.add_argument("--device", default="cpu", help="cpu is fastest at these tiny batch sizes")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    game = rf.games.Connect4()
    net = AlphaZeroNet(game.observation_space().shape, game.action_space().n).to(args.device)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    engine = build_engine(args.n_games, args.seed)
    infer = make_infer(net, args.device)

    print(f"training on {args.device} — connect4, {args.n_games} games/collect, 48 sims/move")
    if args.eval_every:
        rate = eval_vs_random(net, args.device, args.eval_games, args.seed)
        print(f"  eval (policy head vs random): {rate:.2f} win rate (untrained)")
    for it in range(1, args.iterations + 1):
        obs, pi, z, telemetry = engine.collect(args.collect_size, infer)  # search with live weights
        policy_loss, value_loss = train_pass(net, optimizer, obs, pi, z, args.batch_size, args.device)
        print(
            f"  iter {it:3d}  records {obs.shape[0]:4d}  policy_loss {policy_loss:.4f}  "
            f"value_loss {value_loss:.4f}  episodes {len(telemetry['episodes'])}"
        )
        if args.eval_every and it % args.eval_every == 0:
            rate = eval_vs_random(net, args.device, args.eval_games, args.seed + it)
            print(f"  eval (policy head vs random): {rate:.2f} win rate")


if __name__ == "__main__":
    main()
