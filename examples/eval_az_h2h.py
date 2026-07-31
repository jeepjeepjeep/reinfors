"""Head-to-head referee for two saved AlphaZero connect4 nets (`train_alphazero_example.py
--save`): play `--games` connect4 games with each net's RAW policy head, alternating colors.
Connect4 is deterministic and argmax play is too — without diversity every pairing collapses to
two distinct games — so the first `--opening-plies` moves are SAMPLED from each net's softmax
over the legal moves (the AZ opening-temperature convention), argmax after. Reports A's score
(win=1, draw=0.5). A search-free probe compares what the nets distilled, uncontaminated by
search-budget noise.

    uv run --with torch python examples/eval_az_h2h.py a.pt b.pt --games 200
"""

from __future__ import annotations

import argparse
import random

import numpy as np
import reinfors as rf
import torch
from train_alphazero_example import AlphaZeroNet


def load(path: str) -> AlphaZeroNet:
    blob = torch.load(path, weights_only=True)
    game = rf.games.Connect4()
    c, h, w = game.observation_space().shape
    net = AlphaZeroNet((c, h, w), game.action_space().n, blob["width"])
    net.load_state_dict(blob["state_dict"])
    net.eval()
    return net


def pick(net: AlphaZeroNet, obs: np.ndarray, legal: list[int], sample: random.Random | None) -> int:
    with torch.no_grad():
        logits, _ = net(torch.from_numpy(obs[None].astype(np.float32)))
    row = logits[0].numpy()
    if sample is not None:
        weights = np.exp(row[legal] - row[legal].max())
        return int(sample.choices(legal, weights=weights.tolist())[0])
    return max(legal, key=lambda a: row[a])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("net_a")
    parser.add_argument("net_b")
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--opening-plies", type=int, default=4, help="softmax-sampled opening moves")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    nets = [load(args.net_a), load(args.net_b)]
    rng = random.Random(args.seed)
    score_a = 0.0
    for g in range(args.games):
        a_seat = g % 2  # alternate colors
        env = rf.Env(rf.games.Connect4(), rf.Reward(win=1.0, loss=-1.0), seed=rng.randrange(2**31))
        env.reset()
        ply = 0
        while not env.done():
            (mover,) = env.active_agents()
            net = nets[0] if mover == a_seat else nets[1]
            sample = rng if ply < args.opening_plies else None
            action = pick(net, env.observe(mover), env.legal_actions(mover), sample)
            env.step({mover: action})
            ply += 1
        rewards = env.rewards
        assert rewards is not None
        r_a = rewards[a_seat]
        score_a += 1.0 if r_a > 0 else (0.5 if r_a == 0 else 0.0)
    print(f"A={args.net_a} score {score_a}/{args.games} = {score_a / args.games:.3f} vs B={args.net_b}")


if __name__ == "__main__":
    main()
