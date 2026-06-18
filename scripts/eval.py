"""Evaluate a trained snake checkpoint against a random baseline: win-rate over N games via `rf.Env`.

This is the external "is it actually learning?" signal the training telemetry can't give — the trained
net plays real games against a fixed opponent and we count wins. The net only consumes observations at
inference, so the eval env uses a clean win/loss/draw reward (the agent's behaviour is unaffected) and
we read the terminal reward to score each game.

    python scripts/eval.py --checkpoint ckpts/reinfors/final.pt --games 200

With no `--checkpoint`, a randomly-initialised net is evaluated (a ~50% sanity baseline).
"""

from __future__ import annotations

import argparse

import numpy as np
import reinfors as rf
import torch
from reinfors.training import BootstrappedQNetwork


def greedy_action(net: BootstrappedQNetwork, env: rf.Env, agent: int, device: str) -> int:
    """The trained agent: argmax over legal actions of the head-mean Q from the net."""
    obs = torch.from_numpy(env.observe(agent))[None].to(device)
    with torch.no_grad():
        q = net(obs).mean(1).squeeze(0).cpu().numpy()  # mean over heads -> (A,)
    legal = env.legal_actions(agent)
    return max(legal, key=lambda a: q[a])


def play_game(
    env: rf.Env,
    net: BootstrappedQNetwork,
    agent_idx: int,
    max_ticks: int,
    device: str,
    rng: np.random.Generator,
) -> str:
    """One agent-vs-random game; returns 'win' / 'loss' / 'draw' / 'timeout' from `agent_idx`'s view."""
    env.reset()
    last = [0.0, 0.0]
    for _ in range(max_ticks):
        if env.done():
            break
        moves = {}
        for a in env.active_agents():
            if a == agent_idx:
                moves[a] = greedy_action(net, env, a, device)
            else:
                moves[a] = int(rng.choice(env.legal_actions(a)))
        last = env.step(moves)
    if not env.done():
        return "timeout"
    other = 1 - agent_idx
    if last[agent_idx] > last[other]:
        return "win"
    if last[agent_idx] < last[other]:
        return "loss"
    return "draw"


def default_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default=None, help="net state_dict (default: a random-init net)")
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--grid", type=int, default=20)
    parser.add_argument("--heads", type=int, default=10, help="must match the checkpoint's head count")
    parser.add_argument("--prior-scale", type=float, default=2.5, help="must match training")
    parser.add_argument("--max-ticks", type=int, default=750, help="cap per game (snake can run forever)")
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # Clean win/loss/draw reward so the terminal reward vector is exactly the game outcome. The net acts
    # on observations only, so this doesn't change the agent — it just makes scoring unambiguous.
    game = rf.games.Snake(grid_size=args.grid, reward=rf.Reward(win=1.0, loss=-1.0, draw=0.0))
    net = BootstrappedQNetwork(
        game.observation_space().shape, game.action_space().n, args.heads, prior_scale=args.prior_scale
    )
    if args.checkpoint is not None:
        net.load_state_dict(torch.load(args.checkpoint, map_location=args.device)["net"])
    net.to(args.device).eval()

    env = rf.Env(game, seed=args.seed)
    rng = np.random.default_rng(args.seed)
    tally = {"win": 0, "loss": 0, "draw": 0, "timeout": 0}
    for i in range(args.games):
        # Alternate which side the agent plays, to cancel any first-position asymmetry.
        tally[play_game(env, net, i % 2, args.max_ticks, args.device, rng)] += 1

    n = args.games
    wins, losses, draws, timeouts = tally["win"], tally["loss"], tally["draw"], tally["timeout"]
    wr = wins / n
    ci = 1.96 * (wr * (1 - wr) / n) ** 0.5  # normal-approx 95% interval
    src = args.checkpoint if args.checkpoint else "random-init net"
    print(f"{src} vs random — {n} games, grid {args.grid}")
    print(f"  win-rate {wr:.3f} ± {ci:.3f}   (W {wins} / L {losses} / D {draws} / timeout {timeouts})")


if __name__ == "__main__":
    main()
