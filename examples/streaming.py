"""Overlap bounded background collection with placeholder Python-side work."""

from __future__ import annotations

import argparse

import numpy as np
import reinfors as rf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--updates", type=int, default=3)
    parser.add_argument("--records", type=int, default=64)
    args = parser.parse_args()

    game = rf.games.Connect4()
    engine = rf.Engine(
        game=game,
        reward=rf.Reward(win=1.0, loss=-1.0),
        policy=rf.policies.Mcts(num_simulations=16),
        learner=rf.learners.TreeStrap(),
        n_games=8,
        seed=0,
    )
    n_actions = game.action_space().n

    def infer(obs: np.ndarray) -> np.ndarray:
        return np.zeros((len(obs), 1, n_actions), dtype=np.float32)

    with engine.collect_stream(collect_size=args.records, infer=infer, depth=1) as stream:
        for update in range(args.updates):
            batch = next(stream)
            # Replace this line with optimization on the main Python thread.
            print(f"update={update} records={len(batch.obs)} queued={stream.pending()}")


if __name__ == "__main__":
    main()
