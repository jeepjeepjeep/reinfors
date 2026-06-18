"""Play snake against a trained agent in the terminal, via `rf.Env`.

A human controls one snake, the net controls the other; both move each tick (snake is simultaneous).
The board is rendered from the env's native `state()`; the agent acts greedily on its head-mean Q.

    python scripts/play.py --checkpoint ckpts/reinfors/final.pt

With no `--checkpoint`, the agent is a randomly-initialised net (a weak sparring partner).
"""

from __future__ import annotations

import argparse

import reinfors as rf
import torch
from reinfors.training import BootstrappedQNetwork

DIR_ARROW = {0: "^", 1: "v", 2: "<", 3: ">"}  # Action: Up, Down, Left, Right
RELATIVE = "0=forward  1=left  2=right"


def render(state: dict, grid: int, you: int) -> None:
    cells = [["." for _ in range(grid)] for _ in range(grid)]
    for snake, (head, body) in enumerate([("A", "a"), ("B", "b")]):
        you_mark = snake == you
        glyph_head, glyph_body = (head, body) if you_mark else (head.swapcase(), body.swapcase())
        for i, (r, c) in enumerate(state["bodies"][snake]):
            cells[r][c] = glyph_head if i == 0 else glyph_body
    for r, c in state["food"]:
        cells[r][c] = "*"
    print("\n".join(" ".join(row) for row in cells))
    facing = [DIR_ARROW[d] for d in state["directions"]]
    print(f"you = snake {'A' if you == 0 else 'B'}   facing A:{facing[0]} B:{facing[1]}\n")


def greedy_action(net: BootstrappedQNetwork, env: rf.Env, agent: int, device: str) -> int:
    obs = torch.from_numpy(env.observe(agent))[None].to(device)
    with torch.no_grad():
        q = net(obs).mean(1).squeeze(0).cpu().numpy()
    legal = env.legal_actions(agent)
    return max(legal, key=lambda a: q[a])


def human_action(env: rf.Env, agent: int) -> int:
    legal = env.legal_actions(agent)
    while True:
        try:
            move = int(input(f"your move ({RELATIVE}): "))
        except EOFError as eof:
            raise SystemExit("\naborted") from eof
        except ValueError:
            print("  enter 0, 1, or 2")
            continue
        if move in legal:
            return move
        print(f"  illegal; choose from {legal}")


def default_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default=None, help="net state_dict (default: a random-init net)")
    parser.add_argument("--grid", type=int, default=20)
    parser.add_argument("--heads", type=int, default=10, help="must match the checkpoint")
    parser.add_argument("--prior-scale", type=float, default=2.5, help="must match training")
    parser.add_argument("--you", type=int, default=0, choices=(0, 1), help="which snake you control")
    parser.add_argument("--max-ticks", type=int, default=200)
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # `play_to_last=False` ends the game the moment a snake dies (so a human crash concludes the
    # match), and a clean win/loss/draw reward makes the terminal reward vector *be* the outcome. The
    # agent acts on observations only, so neither changes its play — they just make the verdict real.
    game = rf.games.Snake(grid_size=args.grid, play_to_last=False, reward=rf.Reward(win=1.0, loss=-1.0, draw=0.0))
    net = BootstrappedQNetwork(
        game.observation_space().shape, game.action_space().n, args.heads, prior_scale=args.prior_scale
    )
    if args.checkpoint is not None:
        net.load_state_dict(torch.load(args.checkpoint, map_location=args.device)["net"])
    net.to(args.device).eval()

    env = rf.Env(game, seed=args.seed)
    last = [0.0, 0.0]
    for _ in range(args.max_ticks):
        if env.done():
            break
        render(env.state(), args.grid, args.you)
        moves = {}
        for a in env.active_agents():
            moves[a] = human_action(env, a) if a == args.you else greedy_action(net, env, a, args.device)
        last = env.step(moves)

    render(env.state(), args.grid, args.you)
    if not env.done():
        print("max ticks reached — draw")
    elif last[args.you] > last[1 - args.you]:
        print("you win!")
    elif last[args.you] < last[1 - args.you]:
        print("you lose.")
    else:
        print("draw")


if __name__ == "__main__":
    main()
