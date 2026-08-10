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

`--depth` switches to overlapped collection (`engine.collect_stream`): a Rust worker collects batch
t+1 while this process trains on batch t (depth 1 = one batch ahead; `--depth none` = unbounded,
OpenSpiel's continuous-actor topology). Overlap needs the *two-net pattern*: the worker's callback
reads a collector net, training updates the learner net, and weights sync between batches under a
lock — `load_state_dict` releases the GIL mid-copy, so an unlocked swap could expose half-updated
weights to a concurrent search round. Python locks are GIL-aware; there is no deadlock.

This is a deliberately tiny, self-contained reference, not a production trainer: no replay buffer,
no checkpoints, no logging. It exists to show the wiring.

    uv run --with torch python examples/train_alphazero_example.py --iterations 40
    uv run --with torch python examples/train_alphazero_example.py --iterations 40 --depth 1
"""

from __future__ import annotations

import argparse
import copy
import random
import threading
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import reinfors as rf
import torch
from torch import nn
from torch.nn import functional as F


class AlphaZeroNet(nn.Module):
    """Two-headed net: shared conv trunk -> policy logits (B, A) + tanh value (B,). Just enough to
    plug a torch model into the AlphaZero family; both heads come from one trunk pass."""

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
        # Native f32 out: the engine widens exactly; skips the f64 conversion (GPU fast path).
        return logits.cpu().numpy(), values.cpu().numpy()

    return infer


def alphazero_loss(
    logits: torch.Tensor,
    values: torch.Tensor,
    pi: torch.Tensor,
    z: torch.Tensor,
    w: torch.Tensor,
    legal: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The paper's two terms, policy-weighted: cross-entropy of the policy head against the
    search's visit distribution on acting rows (`w` = batch.policy_weights, 0 on value-only
    non-mover rows — their pi is inert zeros), MSE of the value head against the realized
    outcome on every row. Illegal logits are masked out before the softmax (`legal` densified
    from batch.legal_ids/legal_offsets) so the policy distributes over legal actions only —
    unmasked, early training spends its gradient suppressing illegal actions instead."""
    masked = logits.masked_fill(~legal, torch.finfo(logits.dtype).min)
    ce = -(pi * F.log_softmax(masked, dim=-1)).sum(dim=-1)
    policy_loss = (w * ce).sum() / w.sum().clamp(min=1.0)
    value_loss = F.mse_loss(values, z)
    return policy_loss, value_loss


def build_engine(n_games: int, seed: int, sequential_backup: str = "auto") -> rf.Engine:
    # Handle defaults are the AlphaZero conventions: noise_epsilon 0.25, noise_alpha 0.3,
    # temperature 1.0 for the first 8 plies. gamma=1 + win/loss=±1 gives the paper's z.
    # `sequential_backup="maxn"` forces the Max^N vector backup (per-perspective leaf values +
    # value-only rows for the non-mover) — the negamax-deletion measurement seam.
    return rf.Engine(
        rf.games.Connect4(),
        rf.Reward(win=1.0, loss=-1.0),
        rf.policies.AlphaZero(num_simulations=48, sequential_backup=sequential_backup),
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
    weights: np.ndarray,
    legal: np.ndarray,
    batch_size: int,
    device: str,
) -> tuple[float, float]:
    """One shuffled minibatch pass over a collected batch; returns mean (policy, value) losses."""
    c, h, w = net.obs_shape
    o = torch.from_numpy(obs).reshape(-1, c, h, w).to(device)
    p = torch.from_numpy(pi).float().to(device)
    v = torch.from_numpy(z).float().to(device)
    pw = torch.from_numpy(weights).float().to(device)
    lg = torch.from_numpy(legal).to(device)
    perm = torch.randperm(o.shape[0])
    p_sum, v_sum, batches = 0.0, 0.0, 0
    for start in range(0, o.shape[0], batch_size):
        idx = perm[start : start + batch_size]
        logits, values = net(o[idx])
        policy_loss, value_loss = alphazero_loss(logits, values, p[idx], v[idx], pw[idx], lg[idx])
        optimizer.zero_grad()
        (policy_loss + value_loss).backward()  # type: ignore[no-untyped-call]  # torch stub gap
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
        events: list[tuple[int, Any]] = []
        while not env.done():
            agent = env.active_agents()[0]
            if agent == net_side:
                with torch.no_grad():
                    x = torch.from_numpy(env.observe(agent)).reshape(1, c, h, w).to(device)
                    logits, _ = net(x)
                # Argmax over the LEGAL set: a dense argmax can pick a full column, which the Env
                # boundary rejects (a trained net rarely ranks one on top, but late-game it can).
                row = logits[0].cpu().numpy()
                action = max(env.legal_actions(agent), key=lambda a: row[a])
            else:
                action = rng.choice(env.legal_actions(agent))
            events = env.step({agent: action})
        result = next((event for player, event in events if player == net_side), None)
        if result == "win":
            score += 1.0
        elif result == "draw":
            score += 0.5
    net.train(was_training)
    return score / games


def run_iteration(
    args: argparse.Namespace,
    net: AlphaZeroNet,
    optimizer: torch.optim.Optimizer,
    it: int,
    batch: rf.AlphaZeroBatch,
) -> None:
    obs, pi, z = batch.obs, batch.policy_targets, batch.value_targets
    weights = batch.policy_weights
    # densify the legality CSR once per collected batch; rows shuffle with it in train_pass
    counts = np.diff(batch.legal_offsets)
    rows = np.repeat(np.arange(obs.shape[0]), counts)
    legal = np.zeros((obs.shape[0], pi.shape[1]), dtype=bool)
    legal[rows, batch.legal_ids] = True
    policy_loss = value_loss = float("nan")  # --train-passes 0 = collect-only; nothing to report
    for _ in range(args.train_passes):
        policy_loss, value_loss = train_pass(net, optimizer, obs, pi, z, weights, legal, args.batch_size, args.device)
    print(
        f"  iter {it:3d}  records {obs.shape[0]:4d}  policy_loss {policy_loss:.4f}  "
        f"value_loss {value_loss:.4f}  episodes {len(batch.telemetry['episodes'])}"
    )
    if args.eval_every and it % args.eval_every == 0:
        rate = eval_vs_random(net, args.device, args.eval_games, args.seed + it)
        print(f"  eval (policy head vs random): {rate:.2f} win rate")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--iterations", type=int, default=40, help="collect+train cycles")
    parser.add_argument("--n-games", type=int, default=8, help="parallel self-play games per collect")
    parser.add_argument("--collect-size", type=int, default=512, help="record floor per collect")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--train-passes", type=int, default=1, help="minibatch passes per batch (data reuse)")
    parser.add_argument("--width", type=int, default=32, help="trunk width (channels)")
    parser.add_argument(
        "--depth",
        default=None,
        help="overlapped collection: batches the worker may run ahead (int), 'none' = unbounded "
        "(continuous actors); omit for the synchronous loop",
    )
    parser.add_argument("--eval-every", type=int, default=10, help="iterations between eval probes (0 = off)")
    parser.add_argument("--eval-games", type=int, default=50)
    parser.add_argument(
        "--sequential-backup",
        choices=["auto", "maxn"],
        default="auto",
        help="auto = negamax (2p zero-sum); maxn forces the Max^N vector backup (measurement seam)",
    )
    parser.add_argument("--device", default="cpu", help="cpu is fastest at these tiny batch sizes")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", default=None, help="path to torch.save the final net state_dict")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    game = rf.games.Connect4()
    net = AlphaZeroNet(game.observation_space().shape, game.action_space().n, args.width).to(args.device)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    engine = build_engine(args.n_games, args.seed, args.sequential_backup)

    mode = "sync" if args.depth is None else f"depth={args.depth}"
    print(
        f"training on {args.device} — connect4, {args.n_games} games/collect, 48 sims/move, width {args.width}, {mode}"
    )
    if args.eval_every:
        rate = eval_vs_random(net, args.device, args.eval_games, args.seed)
        print(f"  eval (policy head vs random): {rate:.2f} win rate (untrained)")

    t0 = time.perf_counter()
    if args.depth is None:
        # Synchronous reference loop: collect and train take turns; one net, implicit weight sync.
        infer = make_infer(net, args.device)
        for it in range(1, args.iterations + 1):
            batch = engine.collect(n_records=args.collect_size, infer=infer)  # search with live weights
            assert isinstance(batch, rf.AlphaZeroBatch)  # narrows the family union
            run_iteration(args, net, optimizer, it, batch)
    else:
        # Overlapped loop: the worker collects batch t+1 (reading the collector net) while we train
        # batch t (updating the learner net). The sync point — right after next() — gives the worker
        # weights exactly one iteration stale, and the lock keeps a state_dict swap atomic relative
        # to search rounds (Python locks release the GIL while blocking, so this cannot deadlock).
        depth = None if str(args.depth).lower() in ("none", "inf") else int(args.depth)
        collector_net = copy.deepcopy(net)
        sync_lock = threading.Lock()
        base_infer = make_infer(collector_net, args.device)

        def locked_infer(obs_batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            with sync_lock:
                return base_infer(obs_batch)

        with engine.collect_stream(collect_size=args.collect_size, infer=locked_infer, depth=depth) as stream:
            for it in range(1, args.iterations + 1):
                batch = stream.next()
                assert isinstance(batch, rf.AlphaZeroBatch)  # narrows the family union
                with sync_lock:
                    collector_net.load_state_dict(net.state_dict())
                run_iteration(args, net, optimizer, it, batch)
    print(f"  total {time.perf_counter() - t0:.1f}s for {args.iterations} iterations ({mode})")
    if args.save:
        torch.save({"state_dict": net.state_dict(), "width": args.width}, args.save)
        print(f"saved {args.save}")


if __name__ == "__main__":
    main()
