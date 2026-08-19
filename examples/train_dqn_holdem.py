"""Minimal example: ensemble DQN on fixed-limit Texas hold'em via reinfors' Rust engine.

Hold'em is reinfors' first HIDDEN-information game: the tree-search families reject it (their
values would be clairvoyant about hole cards), so training runs through the observation-only DQN
family — `EpsilonGreedyQ` acts on each seat's own egocentric observation, `Dqn` emits off-policy
transitions. reinfors owns the data generation; this script owns the learning: a small torch MLP
with K ensemble heads, a uniform replay buffer (legality densified from the batch's CSR fields at
insert), 1-step Q targets from a periodically synced target network (`--double` switches to
Double DQN targets: the online net selects the bootstrap action, the target net evaluates it),
and per-head bootstrap masks. `--dueling` splits each head into state-value and zero-mean
advantage streams (Wang et al. 2016). `--c51` makes each head distributional (Bellemare et al.
2017): the net predicts categorical return distributions on a fixed support covering the reachable
bb-scaled returns (lose at most your stack, win at most the other players'), trained by
cross-entropy against the projected Bellman target, while the infer
callback hands the engine expected Q — the engine contract never changes. `--per` switches the
buffer to prioritized replay (Schaul et al.
2016): sampling proportional to |TD error|^alpha, annealed importance weights in the loss, and
priorities written back after every minibatch. `--noisy` replaces every linear layer with a
factorized-Gaussian noisy layer (Fortunato et al. 2018) and forces epsilon to 0: exploration
comes from learned weight noise, resampled per inference callback (canonical Rainbow) and per
training minibatch (online and target independently); the eval probe's `net.eval()` uses mean
weights. One callback serves the whole game pool, so each round's noise draw is shared across
games — standard for parallelised noisy nets. Perturbations still differ per game through the
states, but in a game with no chance and a fixed start, shared noise plus greedy selection
makes every parallel game play identically, so parallelism adds no exploration diversity;
hold'em's dealt cards diverge the pool. Seed torch for reproducible collection — exploration
randomness lives caller-side under this flag. Note `--noisy` alone is not
noisy-ONLY exploration: with the default 4-head ensemble, `EpsilonGreedyQ` still
Thompson-samples a head per episode, an independent per-game diversity source — `--heads 1` is
the canonical Rainbow composition. The sigma parameters anneal during training, so the
high-epsilon cycling caution above applies with extra force (measured at 4 heads and at
`--heads 1` alike: near 0 bb/hand vs random where eps 0.6 holds ~+5.8 — the documented
self-play cycling, not a defect of the layer). `--n-step`
widens TD windows to n of the seat's own decisions (Rainbow's multi-step returns): the engine
emits the discounted n-step reward sum and per-record `discounts` = gamma^k, the single
discount source for every target rule here.
Rewards are chip deltas scaled to big blinds; one episode = one hand, so gamma (now a
`Dqn` constructor argument) multiplies across the streets of a single hand.

The eval probe seats the GREEDY policy head at a rotating position against scripted opponents
(uniform-random or always-call) and reports mean big blinds per hand — the standard sanity that
self-play DQN learned to fold junk and bet made hands. The default epsilon is deliberately high
(0.6): independent Q-learning in an imperfect-info game chases a best response to itself and
cycles (aggro -> fold-to-aggro -> exploit-folders), collapsing to fold-heavy play at low
exploration; heavy mixing keeps the learned policy a best response to mixed opponents, which is
what the probes measure (measured: eps 0.1-0.3 drift to ~0 bb/hand vs random, eps 0.6 holds
~+5.8 across seeds). Principled self-play convergence for poker (NFSP/CFR-style) is out of scope
for this example.

    uv run --with torch python examples/train_dqn_holdem.py --iterations 60
"""

from __future__ import annotations

import argparse
import math
import random
import time

import numpy as np
import reinfors as rf
import torch
from reinfors import DqnBatch
from torch import nn


class NoisyLinear(nn.Module):
    """Factorized Gaussian noisy layer (Fortunato et al. 2018): weights are mu + sigma * eps
    with learned sigma, so exploration lives in weight space and anneals via the loss. Training
    mode uses the current noise draw; eval mode uses the mean weights."""

    def __init__(self, in_features: int, out_features: int, sigma0: float = 0.5) -> None:
        super().__init__()
        self.w_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.w_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.b_mu = nn.Parameter(torch.empty(out_features))
        self.b_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("eps_in", torch.zeros(in_features))
        self.register_buffer("eps_out", torch.zeros(out_features))
        bound = 1.0 / math.sqrt(in_features)
        nn.init.uniform_(self.w_mu, -bound, bound)
        nn.init.uniform_(self.b_mu, -bound, bound)
        nn.init.constant_(self.w_sigma, sigma0 / math.sqrt(in_features))
        nn.init.constant_(self.b_sigma, sigma0 / math.sqrt(in_features))

    def resample_noise(self) -> None:
        for eps in (self.eps_in, self.eps_out):
            eps.normal_()
            eps.copy_(eps.sign() * eps.abs().sqrt())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return nn.functional.linear(x, self.w_mu, self.b_mu)
        w = self.w_mu + self.w_sigma * torch.outer(self.eps_out, self.eps_in)
        b = self.b_mu + self.b_sigma * self.eps_out
        return nn.functional.linear(x, w, b)


class QNet(nn.Module):
    """Flattened-obs MLP ensemble: `forward` gives (B, K, A) Q values, or (B, K, A, atoms)
    return-distribution logits when distributional; `q_values` gives (B, K, A) expected Q in
    both modes. Optional dueling V/A heads compose with either.

    The dueling advantage mean runs over ALL actions — the callback never sees legality (it
    lives in the engine and the batch CSR), and the zero-mean constraint is an identifiability
    device, not a semantic one: any offset is constant per state, so action selection is
    unchanged.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_actions: int,
        width: int = 256,
        dueling: bool = False,
        atoms: int | None = None,
        v_bounds: tuple[float, float] = (-20.0, 20.0),
        noisy: bool = False,
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.n_actions = n_actions
        self.atoms = atoms
        linear = NoisyLinear if noisy else nn.Linear
        self.trunk = nn.Sequential(
            linear(dim, width),
            nn.ReLU(),
            linear(width, width),
            nn.ReLU(),
        )
        if atoms is not None:
            if atoms < 2:
                raise ValueError(f"C51 needs atoms >= 2, got {atoms}")
            lo, hi = v_bounds
            if not (math.isfinite(lo) and math.isfinite(hi) and lo < hi):
                raise ValueError(f"C51 support bounds must be finite with v_min < v_max, got {v_bounds}")
        per_out = atoms if atoms is not None else 1
        self.adv = linear(width, n_heads * n_actions * per_out)
        self.value = linear(width, n_heads * per_out) if dueling else None
        if atoms is not None:
            self.register_buffer("support", torch.linspace(v_bounds[0], v_bounds[1], atoms))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, K, A) Q values, or (B, K, A, Z) atom logits when distributional."""
        h = self.trunk(x)
        if self.atoms is None:
            q = self.adv(h).view(-1, self.n_heads, self.n_actions)
            if self.value is None:
                return q
            # Zero-mean advantages pin the otherwise unidentifiable V/A split (Wang et al. 2016).
            v = self.value(h).view(-1, self.n_heads, 1)
            return v + q - q.mean(dim=2, keepdim=True)
        logits = self.adv(h).view(-1, self.n_heads, self.n_actions, self.atoms)
        if self.value is not None:
            v = self.value(h).view(-1, self.n_heads, 1, self.atoms)
            logits = v + logits - logits.mean(dim=2, keepdim=True)
        return logits

    def resample_noise(self) -> None:
        """No-op without noisy layers, so call sites need no branching."""
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.resample_noise()

    def q_values(self, x: torch.Tensor) -> torch.Tensor:
        """(B, K, A) expected Q either way: the engine and eval always see scalars."""
        out = self.forward(x)
        if self.atoms is None:
            return out
        return (out.softmax(-1) * self.support).sum(-1)  # type: ignore[operator]  # buffer typing


def project_distribution(
    t_dist: torch.Tensor,
    rewards: torch.Tensor,
    discounts: torch.Tensor,
    support: torch.Tensor,
) -> torch.Tensor:
    """Re-bin the Bellman-shifted support `r + discount z` onto the fixed atoms (Bellemare et
    al. 2017).

    `t_dist` is `(B, K, Z)` next-state probabilities; `rewards`/`discounts` are `(B,)`, with
    `discounts` the batch's per-record gamma^k. A zero discount shifts every atom to r, so all
    mass projects onto r's neighbours regardless of `t_dist`.
    """
    atoms = support.shape[0]
    tz = (rewards.view(-1, 1) + discounts.view(-1, 1) * support).clamp(support[0], support[-1])
    b = ((tz - support[0]) / (support[1] - support[0])).clamp(0, atoms - 1)  # float-safe indices
    low, up = b.floor().long(), b.ceil().long()
    low_w = up.float() - b + (low == up).float()  # exact hits keep full mass
    up_w = b - low.float()
    m = torch.zeros_like(t_dist)
    m.scatter_add_(2, low.unsqueeze(1).expand_as(t_dist), t_dist * low_w.unsqueeze(1))
    m.scatter_add_(2, up.unsqueeze(1).expand_as(t_dist), t_dist * up_w.unsqueeze(1))
    return m


def dense_mask(offsets: np.ndarray, ids: np.ndarray, m: int, a: int) -> np.ndarray:
    counts = np.diff(offsets)
    rows = np.repeat(np.arange(m), counts)
    mask = np.zeros((m, a), dtype=bool)
    mask[rows, ids] = True
    return mask


class Replay:
    """Ring buffer over the DqnBatch columns, next-state legality densified at insert.

    Uniform by default. With `alpha` set, samples proportional to priority^alpha and returns
    importance weights (PER): fresh rows enter at the running max priority, favouring them
    until their first sampled TD error, which `update` writes back after training.
    """

    def __init__(self, capacity: int, alpha: float | None = None) -> None:
        self.capacity = capacity
        self.cols: list[np.ndarray] | None = None
        self.size = 0
        self.head = 0
        self.alpha = alpha
        self.priorities = np.zeros(capacity, dtype=np.float64) if alpha is not None else None
        self.max_priority = 1.0

    def push(self, batch: DqnBatch, n_actions: int) -> None:
        m = batch.obs.shape[0]
        legal = dense_mask(np.asarray(batch.next_legal_offsets), np.asarray(batch.next_legal_ids), m, n_actions)
        cols: list[np.ndarray] = [
            batch.obs,
            batch.actions,
            batch.rewards,
            batch.discounts,
            batch.next_obs,
            batch.dones,
            batch.masks,
            legal,
        ]
        if self.cols is None:
            self.cols = [np.empty((self.capacity, *c.shape[1:]), dtype=c.dtype) for c in cols]
        idx = (self.head + np.arange(m)) % self.capacity
        for buf, c in zip(self.cols, cols, strict=True):
            buf[idx] = c
        if self.priorities is not None:
            self.priorities[idx] = self.max_priority
        self.head = (self.head + m) % self.capacity
        self.size = min(self.size + m, self.capacity)

    def sample(
        self, n: int, rng: np.random.Generator, beta: float = 0.0
    ) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
        assert self.cols is not None
        if self.priorities is None:
            idx = rng.integers(self.size, size=n)
            weights = np.ones(n, dtype=np.float32)
        else:
            p = self.priorities[: self.size] ** self.alpha
            p /= p.sum()
            idx = rng.choice(self.size, size=n, p=p)
            w = (self.size * p[idx]) ** -beta
            # Normalize by the buffer-wide max weight (min probability), not the batch max:
            # batch normalization weakens the correction for whichever rows were sampled.
            max_w = (self.size * p.min()) ** -beta
            weights = (w / max_w).astype(np.float32)
        return [c[idx] for c in self.cols], idx, weights

    def update(self, idx: np.ndarray, td_abs: np.ndarray) -> None:
        assert self.priorities is not None
        self.priorities[idx] = td_abs + 1e-3
        self.max_priority = max(self.max_priority, float(self.priorities[idx].max()))


def train_step(
    net: QNet,
    target: QNet,
    optimizer: torch.optim.Optimizer,
    replay: Replay,
    rng: np.random.Generator,
    updates: int,
    batch_size: int,
    device: str,
    double: bool,
    beta: float,
) -> float:
    total, batches = 0.0, 0
    for _ in range(updates):
        net.resample_noise()
        target.resample_noise()
        cols, idx, weights_n = replay.sample(batch_size, rng, beta)
        obs_n, actions_n, rewards_n, discounts_n, next_obs_n, _dones_n, masks_n, legal_n = cols
        obs = torch.from_numpy(obs_n).to(device)
        next_obs = torch.from_numpy(next_obs_n).to(device)
        actions = torch.from_numpy(actions_n).long().to(device)
        rewards = torch.from_numpy(rewards_n).float().to(device)
        discounts = torch.from_numpy(discounts_n).float().to(device)
        masks = torch.from_numpy(masks_n).float().to(device)  # (B, K) bootstrap masks
        legal = torch.from_numpy(legal_n).to(device)
        if net.atoms is None:
            q = net(obs)  # (B, K, A)
            q_sa = q.gather(2, actions.view(-1, 1, 1).expand(-1, net.n_heads, 1)).squeeze(2)
            with torch.no_grad():
                q_next = target(next_obs)  # (B, K, A)
                if double:
                    # Decouple: the online net's masked argmax picks, the target net scores.
                    sel = net(next_obs).masked_fill(~legal.unsqueeze(1), -1e9).argmax(dim=2, keepdim=True)
                    boot = q_next.gather(2, sel).squeeze(2)
                else:
                    boot = q_next.masked_fill(~legal.unsqueeze(1), -1e9).max(dim=2).values
                boot = torch.where(legal.any(dim=1, keepdim=True), boot, torch.zeros_like(boot))
                td = rewards.unsqueeze(1) + discounts.unsqueeze(1) * boot
            elem = nn.functional.smooth_l1_loss(q_sa, td, reduction="none")  # (B, K)
            err = (q_sa - td).abs()
        else:
            z = net.support  # (Z,)
            logits = net(obs)  # (B, K, A, Z)
            chosen = logits.gather(2, actions.view(-1, 1, 1, 1).expand(-1, net.n_heads, 1, net.atoms)).squeeze(
                2
            )  # (B, K, Z)
            with torch.no_grad():
                t_dist_all = target(next_obs).softmax(-1)  # (B, K, A, Z)
                sel_q = (net(next_obs).softmax(-1) * z).sum(-1) if double else (t_dist_all * z).sum(-1)
                sel = sel_q.masked_fill(~legal.unsqueeze(1), -1e9).argmax(dim=2)  # (B, K)
                t_dist = t_dist_all.gather(2, sel.view(-1, net.n_heads, 1, 1).expand(-1, -1, 1, net.atoms)).squeeze(
                    2
                )  # (B, K, Z)
                m = project_distribution(t_dist, rewards, discounts, z)
            elem = -(m * chosen.log_softmax(-1)).sum(-1)  # (B, K) cross-entropy
            err = elem
        weights = torch.from_numpy(weights_n).to(device)
        loss = (weights.unsqueeze(1) * masks * elem).sum() / masks.sum().clamp(min=1.0)
        if replay.priorities is not None:
            included = masks.sum(dim=1).clamp(min=1.0)
            per_row = ((masks * err.abs()).sum(dim=1) / included).detach().cpu().numpy()
            replay.update(idx, per_row)
        optimizer.zero_grad()
        loss.backward()  # type: ignore[no-untyped-call]  # torch stubs leave Tensor.backward untyped
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
                    x = torch.from_numpy(env.observe(agent).reshape(1, -1)).to(args.device)
                    q = net.q_values(x)[0].mean(dim=0).cpu().numpy()  # head-mean Q
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
    parser.add_argument(
        "--noisy",
        action="store_true",
        help="noisy-net exploration: learned weight noise, epsilon forced to 0",
    )
    parser.add_argument("--gamma", type=float, default=1.0, help="engine-side discount over own decisions")
    parser.add_argument(
        "--n-step",
        type=int,
        default=1,
        help="multi-step return window over own decisions; uncorrected off-policy, keep small",
    )
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--target-sync", type=int, default=10, help="iterations between target-net syncs")
    parser.add_argument(
        "--double",
        action="store_true",
        help="Double DQN targets: the online net selects the bootstrap action, the target net evaluates it",
    )
    parser.add_argument(
        "--per",
        action="store_true",
        help="prioritized replay: sample by |TD|^alpha with annealed importance weights",
    )
    parser.add_argument("--per-alpha", type=float, default=0.6, help="priority exponent")
    parser.add_argument(
        "--dueling",
        action="store_true",
        help="dueling heads: Q = V + A - mean(A), sharing value learning across actions",
    )
    parser.add_argument(
        "--c51",
        action="store_true",
        help="distributional heads: categorical return distributions, trained by projected cross-entropy",
    )
    parser.add_argument("--atoms", type=int, default=51, help="C51 support size")
    parser.add_argument(
        "--v-min",
        type=float,
        default=None,
        help="C51 support lower bound in big blinds (default: -stack)",
    )
    parser.add_argument(
        "--v-max",
        type=float,
        default=None,
        help="C51 support upper bound in big blinds (default: (players - 1) * stack)",
    )
    parser.add_argument("--per-beta", type=float, default=0.4, help="initial importance-weight exponent")
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
    # Asymmetric support: a hand's bb-scaled chip delta is bounded below by the player's own
    # stack but above by the other (N-1) stacks they can win.
    stack_bb = args.stack / args.big_blind
    v_bounds = (
        args.v_min if args.v_min is not None else -stack_bb,
        args.v_max if args.v_max is not None else (args.num_players - 1) * stack_bb,
    )
    atoms = args.atoms if args.c51 else None
    net = QNet(
        dim, args.heads, 3, args.width, dueling=args.dueling, atoms=atoms, v_bounds=v_bounds, noisy=args.noisy
    ).to(args.device)
    target = QNet(
        dim, args.heads, 3, args.width, dueling=args.dueling, atoms=atoms, v_bounds=v_bounds, noisy=args.noisy
    ).to(args.device)
    target.load_state_dict(net.state_dict())
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    replay = Replay(args.replay_capacity, alpha=args.per_alpha if args.per else None)
    rng = np.random.default_rng(args.seed)
    engine = rf.Engine(
        game,
        rf.Reward(scale=1.0 / args.big_blind),  # rewards in big blinds
        rf.policies.EpsilonGreedyQ(n_heads=args.heads, epsilon=0.0 if args.noisy else args.epsilon),
        rf.learners.Dqn(n_step=args.n_step, gamma=args.gamma),
        n_games=args.n_games,
        seed=args.seed,
    )

    def infer(arr: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            net.resample_noise()  # per-forward, the canonical NoisyNets cadence
            x = torch.from_numpy(arr).to(args.device)
            # The engine always sees scalar Q rows: C51 atoms collapse to expectations here.
            return np.asarray(net.q_values(x).cpu().numpy())  # native f32; the engine widens exactly

    t0 = time.perf_counter()
    print(f"DQN hold'em: {args.num_players} seats, {args.heads} heads, {args.n_games} games/collect")
    for it in range(1, args.iterations + 1):
        # Anneal the IS correction from per_beta to full strength (Schaul et al. 2016).
        progress = (it - 1) / max(args.iterations - 1, 1)
        beta = args.per_beta + (1.0 - args.per_beta) * progress if args.per else 0.0
        batch = engine.collect(n_records=args.collect_size, infer=infer)
        assert isinstance(batch, DqnBatch)
        replay.push(batch, 3)
        loss = train_step(
            net,
            target,
            optimizer,
            replay,
            rng,
            args.updates_per_iter,
            args.batch_size,
            args.device,
            double=args.double,
            beta=beta,
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
