"""Minimal example: ensemble DQN on fixed-limit Texas hold'em via reinfors' Rust engine.

Hold'em is reinfors' first HIDDEN-information game: the tree-search families reject it (their
values would be clairvoyant about hole cards), so training runs through the observation-only DQN
family — `EpsilonGreedyQ` acts on each seat's own egocentric observation, `Dqn` emits off-policy
transitions. reinfors owns the data generation; this script owns the learning: a small torch MLP
with K ensemble heads, a uniform replay buffer (legality densified from the batch's CSR fields at
insert), 1-step Q targets from a periodically synced target network, and per-head bootstrap
masks. Rewards are chip deltas scaled to big
blinds; one episode = one hand, so gamma multiplies across the streets of a single hand.

The eval probe seats the GREEDY policy head at a rotating position against scripted opponents
(uniform-random or always-call) and reports mean big blinds per hand — the standard sanity that
self-play DQN learned to fold junk and bet made hands. The default epsilon is deliberately high
(0.6): independent Q-learning in an imperfect-info game chases a best response to itself and
cycles (aggro -> fold-to-aggro -> exploit-folders), collapsing to fold-heavy play at low
exploration; heavy mixing keeps the learned policy a best response to mixed opponents, which is
what the probes measure (measured: eps 0.1-0.3 drift to ~0 bb/hand vs random, eps 0.6 holds
~+5.8 across seeds). Principled self-play convergence for poker (NFSP/CFR-style) is out of scope
for this example.

    uv run --with torch python scripts/train_dqn_holdem.py --iterations 60
"""

from __future__ import annotations

import argparse
import random
import time

import numpy as np
import reinfors as rf
import torch
from torch import nn


class QNet(nn.Module):
    """Flattened-obs MLP -> (B, K, A) ensemble Q values."""

    def __init__(self, dim: int, n_heads: int, n_actions: int, width: int = 256) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.n_actions = n_actions
        self.trunk = nn.Sequential(
            nn.Linear(dim, width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, n_heads * n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.trunk(x).view(-1, self.n_heads, self.n_actions)


def dense_mask(offsets: np.ndarray, ids: np.ndarray, m: int, a: int) -> np.ndarray:
    counts = np.diff(offsets)
    rows = np.repeat(np.arange(m), counts)
    mask = np.zeros((m, a), dtype=bool)
    mask[rows, ids] = True
    return mask


class Replay:
    """Uniform ring buffer over the DqnBatch columns, next-state legality densified at insert."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.cols: list[np.ndarray] | None = None
        self.size = 0
        self.head = 0

    def push(self, batch: object, n_actions: int) -> None:
        m = batch.obs.shape[0]
        legal = dense_mask(np.asarray(batch.next_legal_offsets), np.asarray(batch.next_legal_ids), m, n_actions)
        cols = [batch.obs, batch.actions, batch.rewards, batch.next_obs, batch.dones, batch.masks, legal]
        if self.cols is None:
            self.cols = [np.empty((self.capacity, *c.shape[1:]), dtype=c.dtype) for c in cols]
        idx = (self.head + np.arange(m)) % self.capacity
        for buf, c in zip(self.cols, cols, strict=True):
            buf[idx] = c
        self.head = (self.head + m) % self.capacity
        self.size = min(self.size + m, self.capacity)

    def sample(self, n: int, rng: np.random.Generator) -> list[np.ndarray]:
        assert self.cols is not None
        idx = rng.integers(self.size, size=n)
        return [c[idx] for c in self.cols]


def train_step(
    net: QNet,
    target: QNet,
    optimizer: torch.optim.Optimizer,
    replay: Replay,
    rng: np.random.Generator,
    updates: int,
    gamma: float,
    batch_size: int,
    device: str,
) -> float:
    total, batches = 0.0, 0
    for _ in range(updates):
        obs_n, actions_n, rewards_n, next_obs_n, dones_n, masks_n, legal_n = replay.sample(batch_size, rng)
        obs = torch.from_numpy(obs_n).to(device)
        next_obs = torch.from_numpy(next_obs_n).to(device)
        actions = torch.from_numpy(actions_n).long().to(device)
        rewards = torch.from_numpy(rewards_n).float().to(device)
        dones = torch.from_numpy(dones_n).to(device)
        masks = torch.from_numpy(masks_n).float().to(device)  # (B, K) bootstrap masks
        legal = torch.from_numpy(legal_n).to(device)
        q = net(obs)  # (B, K, A)
        q_sa = q.gather(2, actions.view(-1, 1, 1).expand(-1, net.n_heads, 1)).squeeze(2)
        with torch.no_grad():
            q_next = target(next_obs)  # (B, K, A)
            q_next = q_next.masked_fill(~legal.unsqueeze(1), -1e9)
            boot = q_next.max(dim=2).values
            boot = torch.where(legal.any(dim=1, keepdim=True), boot, torch.zeros_like(boot))
            td = rewards.unsqueeze(1) + gamma * boot * (~dones).unsqueeze(1).float()
        huber = nn.functional.smooth_l1_loss(q_sa, td, reduction="none")
        loss = (masks * huber).sum() / masks.sum().clamp(min=1.0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += float(loss.item())
        batches += 1
    return total / max(batches, 1)


def eval_vs(net: QNet, args: argparse.Namespace, opponent: str, games: int, seed: int) -> float:
    """Greedy policy head at a rotating seat vs scripted opponents; mean big blinds per hand."""
    rng = random.Random(seed)
    was_training = net.training
    net.eval()
    total_bb = 0.0
    for g in range(games):
        env = rf.Env(
            rf.games.TexasHoldem(
                num_players=args.num_players,
                stack=args.stack,
                small_blind=args.small_blind,
                big_blind=args.big_blind,
            ),
            rf.Reward(),
            seed=rng.randrange(2**31),
        )
        env.reset()
        net_seat = g % args.num_players
        while not env.done():
            (agent,) = env.active_agents()
            legal = env.legal_actions(agent)
            if agent == net_seat:
                with torch.no_grad():
                    x = torch.from_numpy(env.observe(agent).reshape(1, -1))
                    q = net(x)[0].mean(dim=0).numpy()  # head-mean Q
                action = max(legal, key=lambda a: q[a])
            elif opponent == "random":
                action = rng.choice(legal)
            else:  # always-call
                action = 1
            env.step({agent: action})
        rewards = env.rewards
        assert rewards is not None
        total_bb += rewards[net_seat] / args.big_blind
    net.train(was_training)
    return total_bb / games


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--num-players", type=int, default=3)
    parser.add_argument("--stack", type=int, default=200)
    parser.add_argument("--small-blind", type=int, default=5)
    parser.add_argument("--big-blind", type=int, default=10)
    parser.add_argument("--n-games", type=int, default=8)
    parser.add_argument("--collect-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--epsilon", type=float, default=0.6)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--target-sync", type=int, default=10, help="iterations between target-net syncs")
    parser.add_argument("--replay-capacity", type=int, default=100_000)
    parser.add_argument("--updates-per-iter", type=int, default=4)
    parser.add_argument("--eval-every", type=int, default=100, help="iterations between probes (0 = off)")
    parser.add_argument("--eval-games", type=int, default=500)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    game = rf.games.TexasHoldem(
        num_players=args.num_players,
        stack=args.stack,
        small_blind=args.small_blind,
        big_blind=args.big_blind,
    )
    c, h, w = game.observation_space().shape
    dim = c * h * w
    net = QNet(dim, args.heads, 3, args.width).to(args.device)
    target = QNet(dim, args.heads, 3, args.width).to(args.device)
    target.load_state_dict(net.state_dict())
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    replay = Replay(args.replay_capacity)
    rng = np.random.default_rng(args.seed)
    engine = rf.Engine(
        game,
        rf.Reward(scale=1.0 / args.big_blind),  # rewards in big blinds
        rf.policies.EpsilonGreedyQ(n_heads=args.heads, epsilon=args.epsilon),
        rf.learners.Dqn(),
        n_games=args.n_games,
        seed=args.seed,
    )

    def infer(arr: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x = torch.from_numpy(arr).to(args.device)
            return net(x).double().cpu().numpy()

    t0 = time.perf_counter()
    print(f"DQN hold'em: {args.num_players} seats, {args.heads} heads, {args.n_games} games/collect")
    for it in range(1, args.iterations + 1):
        batch = engine.collect(args.collect_size, infer)
        replay.push(batch, 3)
        loss = train_step(
            net, target, optimizer, replay, rng, args.updates_per_iter, args.gamma, args.batch_size, args.device
        )
        if it % args.target_sync == 0:
            target.load_state_dict(net.state_dict())
        eps = batch.telemetry["episodes"]
        mean_len = sum(ep_len for _r, ep_len, _s in eps) / max(len(eps), 1)
        print(
            f"iter {it:3d}  wall {time.perf_counter() - t0:5.0f}s  records {batch.obs.shape[0]:5d}  "
            f"loss {loss:.4f}  hands {len(eps):3d}  hand_len {mean_len:4.1f}"
        )
        if args.eval_every and it % args.eval_every == 0:
            vs_rand = eval_vs(net, args, "random", args.eval_games, seed=1)
            vs_call = eval_vs(net, args, "call", args.eval_games, seed=2)
            print(f"  eval (bb/hand): vs random {vs_rand:+.2f}   vs always-call {vs_call:+.2f}")


if __name__ == "__main__":
    main()
