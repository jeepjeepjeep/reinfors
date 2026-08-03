"""Reference Deep CFR trainer (Brown et al. 2019, external sampling) over reinfors'
`rf.solvers.DeepCfr` data generator.

Division of labor: reinfors runs the traversals and emits the two training streams; this
script owns everything neural — per-player ADVANTAGE networks (queried by the traversals
through the per-player `infer` list), the AVERAGE-POLICY network (the playable product),
reservoir buffers, and the iteration-weighted losses. Per Brown's Algorithm 1, each player's
advantage net retrains AFTER that player's traversal pass (continually by default — Brown's
from-scratch retrain is behind --from-scratch-advantage-nets; continual training measured
better at modest budgets), and the policy net trains on the accumulated strategy stream.

Convergence is measured EXACTLY on the small games: `solver.exploitability(policy_infer)`
enumerates every infoset and runs the exact best response (zero at Nash), so the curve here is
directly comparable to tabular CFR's. On Kuhn the script also prints the learned strategy at
the classic infosets — the point being the MIXED strategies (bluffing frequencies) that
equilibrium play requires and plain self-play RL does not produce. Full hold'em (any table
size via --players) runs the same loop at scale (no exploitability — the tree is not
enumerable); evaluate with --holdem-eval hands against a table of scripted opponents instead.

    uv run --with torch python scripts/train_deep_cfr.py --game kuhn_poker --iterations 60
    uv run --with torch python scripts/train_deep_cfr.py --game leduc_poker --iterations 150
    uv run --with torch python scripts/train_deep_cfr.py --game texas_holdem --iterations 3
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

InferFn = Callable[[np.ndarray], np.ndarray]


class Mlp(nn.Module):
    """Shared trunk shape for both net families: obs -> A raw outputs."""

    def __init__(self, dim: int, n_actions: int, width: int) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(dim, width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.trunk(x)  # type: ignore[no-any-return]  # torch stub gap


class Reservoir:
    """Uniform reservoir (Algorithm R) over `(obs, dense values, legality mask, iteration)`
    rows — Brown's buffer: every emitted sample has an equal chance of surviving, and the
    iteration column carries the linear-CFR loss weight."""

    def __init__(self, capacity: int, dim: int, n_actions: int, seed: int) -> None:
        self.capacity = capacity
        self.obs = np.zeros((capacity, dim), dtype=np.float32)
        self.values = np.zeros((capacity, n_actions), dtype=np.float64)
        self.mask = np.zeros((capacity, n_actions), dtype=bool)
        self.iterations = np.zeros(capacity, dtype=np.int64)
        self.size = 0
        self.seen = 0
        self.rng = random.Random(seed)

    def add_csr(
        self,
        obs: np.ndarray,
        offsets: np.ndarray,
        ids: np.ndarray,
        flat_values: np.ndarray,
        iterations: np.ndarray,
    ) -> None:
        for i in range(obs.shape[0]):
            row_ids = ids[offsets[i] : offsets[i + 1]]
            dense = np.zeros(self.values.shape[1])
            dense[row_ids] = flat_values[offsets[i] : offsets[i + 1]]
            mask = np.zeros(self.values.shape[1], dtype=bool)
            mask[row_ids] = True
            self.seen += 1
            if self.size < self.capacity:
                slot = self.size
                self.size += 1
            else:
                slot = self.rng.randrange(self.seen)
                if slot >= self.capacity:
                    continue
            self.obs[slot] = obs[i]
            self.values[slot] = dense
            self.mask[slot] = mask
            self.iterations[slot] = iterations[i]

    def sample(self, n: int, rng: np.random.Generator) -> tuple[np.ndarray, ...]:
        idx = rng.integers(self.size, size=n)  # with replacement (standard minibatching)
        return self.obs[idx], self.values[idx], self.mask[idx], self.iterations[idx]


def make_infer(net: Mlp, device: str) -> InferFn:
    def infer(obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x = torch.from_numpy(obs).to(device)
            return np.asarray(net(x).cpu().numpy())  # native f32; the solver widens exactly

    return infer


def make_policy_infer(net: Mlp, device: str) -> InferFn:
    """The policy net's exploitability/play adapter: softmax probabilities (the solver
    renormalizes over each infoset's legal actions)."""

    def infer(obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x = torch.from_numpy(obs).to(device)
            return np.asarray(torch.softmax(net(x), dim=1).cpu().numpy())  # native f32

    return infer


def train_regression(
    net: Mlp,
    optimizer: torch.optim.Optimizer,
    buffer: Reservoir,
    steps: int,
    batch_size: int,
    rng: np.random.Generator,
    device: str,
    cross_entropy: bool,
) -> float:
    """T-weighted regression on reservoir rows: MSE to advantage targets over the legality
    mask (advantage nets), or CE to the strategy probabilities (the policy net). The
    optimizer persists with its network across calls (true continual training — Adam's
    moments carry over); --from-scratch-advantage-nets replaces both together."""
    if buffer.size == 0:
        return 0.0
    total = 0.0
    for _ in range(steps):
        obs, values, mask, iterations = buffer.sample(batch_size, rng)
        x = torch.from_numpy(obs).to(device)
        target = torch.from_numpy(values).float().to(device)
        legal = torch.from_numpy(mask).to(device)
        weight = torch.from_numpy(iterations).float().to(device)
        weight = weight / weight.mean().clamp(min=1.0)
        out = net(x)
        if cross_entropy:
            logits = out.masked_fill(~legal, -1e9)
            per_row = -(target * torch.log_softmax(logits, dim=1)).sum(dim=1)
        else:
            per_row = ((out - target) ** 2 * legal.float()).sum(dim=1)
        loss = (weight * per_row).mean()
        optimizer.zero_grad()
        loss.backward()  # type: ignore[no-untyped-call]  # torch stubs leave Tensor.backward untyped
        optimizer.step()
        total += float(loss.item())
    return total / steps


def kuhn_strategy_report(policy_infer: InferFn) -> str:
    """The learned strategy at the classic Kuhn infosets — the mixed-play (bluffing) numbers
    that make equilibrium poker equilibrium poker. Observations are crafted directly against
    KuhnEncoder's layout: own-card one-hot (3) + history slots ((action + 1) / 2)."""

    def obs(card: int, history: list[int]) -> np.ndarray:
        row = np.zeros(6, dtype=np.float32)
        row[card] = 1.0
        for i, action in enumerate(history):
            row[3 + i] = (action + 1) / 2.0
        return row

    probes = [
        ("P0 bets with J (the bluff, Nash: alpha in [0, 1/3])", obs(0, []), 1),
        ("P0 bets with K (Nash: 3*alpha)", obs(2, []), 1),
        ("P1 bets J after a check (the bluff, Nash: 1/3)", obs(0, [0]), 1),
        ("P0 calls a bet with Q (Nash: alpha + 1/3)", obs(1, [0, 1]), 1),
    ]
    rows = np.stack([p[1] for p in probes])
    out = policy_infer(rows)
    probs = out / out.sum(axis=1, keepdims=True)
    return "\n".join(f"    {label}: {probs[i][action]:.3f}" for i, (label, _, action) in enumerate(probes))


def holdem_eval(policy_infer: InferFn, hands: int, seed: int, n_players: int) -> dict[str, float]:
    """bb/hand for the policy net at one rotating seat vs a table of scripted opponents."""
    results: dict[str, float] = {}
    for opponent in ("random", "call"):
        rng = random.Random(seed)
        total = 0.0
        for hand in range(hands):
            env = rf.Env(rf.games.TexasHoldem(num_players=n_players), rf.Reward(), seed=rng.randrange(2**31))
            env.reset()
            seat = hand % n_players
            while not env.done():
                (agent,) = env.active_agents()
                legal = env.legal_actions(agent)
                if agent == seat:
                    row = policy_infer(env.observe(agent).reshape(1, -1))[0]
                    sigma = np.clip(row[legal], 0.0, None)
                    total_mass = sigma.sum()
                    if total_mass <= 0:
                        sigma = np.ones(len(legal)) / len(legal)
                    else:
                        sigma = sigma / total_mass
                    action = int(rng.choices(legal, weights=sigma.tolist())[0])
                elif opponent == "random":
                    action = int(rng.choice(legal))
                else:
                    action = 1
                env.step({agent: action})
            rewards = env.rewards
            assert rewards is not None
            total += rewards[seat] / 10.0  # big blind
        results[opponent] = total / hands
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    def positive_int(value: str) -> int:
        out = int(value)
        if out <= 0:
            raise argparse.ArgumentTypeError(f"{value} is not a positive integer")
        return out

    def positive_float(value: str) -> float:
        out = float(value)
        if out <= 0:
            raise argparse.ArgumentTypeError(f"{value} is not a positive number")
        return out

    parser.add_argument("--game", choices=["kuhn_poker", "leduc_poker", "texas_holdem"], default="kuhn_poker")
    parser.add_argument("--iterations", type=positive_int, default=60)
    parser.add_argument("--traversals", type=positive_int, default=256, help="per player per iteration")
    parser.add_argument("--train-steps", type=positive_int, default=1000, help="sgd steps per advantage retrain")
    parser.add_argument("--policy-train-steps", type=positive_int, default=600)
    parser.add_argument("--batch-size", type=positive_int, default=512)
    parser.add_argument("--buffer-capacity", type=positive_int, default=200_000)
    parser.add_argument("--width", type=positive_int, default=128)
    parser.add_argument("--lr", type=positive_float, default=1e-3)
    parser.add_argument(
        "--from-scratch-advantage-nets",
        action="store_true",
        help="Brown's theory-clean variant: reinitialize each player's advantage net before its "
        "retrain. Measured WORSE at our budgets (Leduc sweep: 0.33 vs 0.23 exploitability at 60 "
        "iterations), so continual training is the default.",
    )
    parser.add_argument("--eval-every", type=int, default=10, help="iterations between probes (0 = off)")
    parser.add_argument("--holdem-eval", type=int, default=0, help="eval hands vs scripted opponents")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--players",
        type=positive_int,
        default=2,
        help="player count (kuhn_poker / texas_holdem; leduc_poker is 2-player)",
    )
    parser.add_argument(
        "--shared-advantage-net",
        action="store_true",
        help="one advantage net + buffer pooled across players (symmetric games); default is "
        "one per player (fits positional asymmetry)",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    if args.game == "leduc_poker" and args.players != 2:
        parser.error("leduc_poker is 2-player")
    handle = {
        "kuhn_poker": lambda: rf.games.KuhnPoker(players=args.players),
        "leduc_poker": lambda: rf.games.LeducPoker(),
        "texas_holdem": lambda: rf.games.TexasHoldem(num_players=args.players),
    }[args.game]()
    n_players = args.players
    solver = rf.solvers.DeepCfr(handle, seed=args.seed)
    dim = int(np.prod(rf.Env(handle, rf.Reward(), seed=0).observation_space().shape))
    n_actions = {"kuhn_poker": 2, "leduc_poker": 3, "texas_holdem": 3}[args.game]
    # Exact metrics enumerate the whole tree; Kuhn past 6 players exceeds the best-response
    # arena cap (the binding raises ValueError there rather than crashing — measured: 6p fits).
    enumerable = args.game == "leduc_poker" or (args.game == "kuhn_poker" and args.players <= 6)
    if args.game == "kuhn_poker" and not enumerable:
        print(f"exploitability unavailable at {args.players}p Kuhn: the tree exceeds the exact enumeration cap")

    if args.shared_advantage_net:
        # One net + one buffer pooled across every player: more data per parameter for
        # symmetric games; the per-player alternative learns positional asymmetry.
        shared_net = Mlp(dim, n_actions, args.width).to(args.device)
        shared_opt = torch.optim.Adam(shared_net.parameters(), lr=args.lr)
        shared_buffer = Reservoir(args.buffer_capacity, dim, n_actions, seed=args.seed)
        advantage_nets = [shared_net] * n_players
        advantage_optimizers = [shared_opt] * n_players
        advantage_buffers = [shared_buffer] * n_players
    else:
        advantage_nets = [Mlp(dim, n_actions, args.width).to(args.device) for _ in range(n_players)]
        advantage_optimizers = [torch.optim.Adam(net.parameters(), lr=args.lr) for net in advantage_nets]
        advantage_buffers = [
            Reservoir(args.buffer_capacity, dim, n_actions, seed=args.seed + p) for p in range(n_players)
        ]
    policy_net = Mlp(dim, n_actions, args.width).to(args.device)
    policy_optimizer = torch.optim.Adam(policy_net.parameters(), lr=args.lr)
    strategy_buffer = Reservoir(args.buffer_capacity, dim, n_actions, seed=args.seed + 7)

    t0 = time.perf_counter()
    infer_share = 0.0
    print(f"Deep CFR on {args.game}: {args.iterations} iterations x {args.traversals} traversals/player")
    for it in range(1, args.iterations + 1):
        solver.next_iteration()
        for player in range(n_players):
            batch = solver.collect(
                player=player,
                traversals=args.traversals,
                infer=[make_infer(net, args.device) for net in advantage_nets],
            )
            advantage_buffers[player].add_csr(
                batch.advantage_obs,
                batch.advantage_legal_offsets,
                batch.advantage_legal_ids,
                batch.advantage_targets,
                batch.advantage_iterations,
            )
            strategy_buffer.add_csr(
                batch.strategy_obs,
                batch.strategy_legal_offsets,
                batch.strategy_legal_ids,
                batch.strategy_probs,
                batch.strategy_iterations,
            )
            infer_share += batch.telemetry["infer_seconds"] / max(batch.telemetry["collect_seconds"], 1e-9)
            if args.from_scratch_advantage_nets and not args.shared_advantage_net:
                advantage_nets[player] = Mlp(dim, n_actions, args.width).to(args.device)
                advantage_optimizers[player] = torch.optim.Adam(advantage_nets[player].parameters(), lr=args.lr)
            if args.shared_advantage_net and player != n_players - 1:
                continue  # shared: everyone's samples land first, then ONE retrain per iteration
            train_regression(
                advantage_nets[player],
                advantage_optimizers[player],
                advantage_buffers[player],
                args.train_steps,
                args.batch_size,
                rng,
                args.device,
                cross_entropy=False,
            )
        if args.eval_every and it % args.eval_every == 0:
            train_regression(
                policy_net,
                policy_optimizer,
                strategy_buffer,
                args.policy_train_steps,
                args.batch_size,
                rng,
                args.device,
                cross_entropy=True,
            )
            wall = time.perf_counter() - t0
            shown = advantage_buffers[:1] if args.shared_advantage_net else advantage_buffers
            adv_sizes = "/".join(str(b.size) for b in shown)
            line = (
                f"iter {it:4d}  wall {wall:6.1f}s  adv buffers {adv_sizes}  "
                f"strategy {strategy_buffer.size}  infer share {infer_share / (n_players * it):.0%}"
            )
            if enumerable:
                e = solver.exploitability(make_policy_infer(policy_net, args.device))
                line += f"  exploitability {e:.4f}"
            print(line)

    # The final playable product: the average-policy net, trained on the full strategy stream.
    train_regression(
        policy_net,
        policy_optimizer,
        strategy_buffer,
        args.policy_train_steps,
        args.batch_size,
        rng,
        args.device,
        cross_entropy=True,
    )
    policy_infer = make_policy_infer(policy_net, args.device)
    if enumerable:
        e = solver.exploitability(policy_infer)
        print(f"final exploitability: {e:.4f}")
    if args.game == "kuhn_poker" and args.players == 2:
        # The classic alpha-parameterized infoset numbers exist only for the 2-player game.
        print("learned strategy at the classic infosets (mixed = the point):")
        print(kuhn_strategy_report(policy_infer))
    if args.game == "texas_holdem" and args.holdem_eval:
        results = holdem_eval(policy_infer, args.holdem_eval, seed=args.seed + 99, n_players=n_players)
        print(
            f"holdem eval ({args.holdem_eval} hands/opponent, bb/hand): "
            f"vs random {results['random']:+.2f}, vs always-call {results['call']:+.2f}"
        )


if __name__ == "__main__":
    main()
