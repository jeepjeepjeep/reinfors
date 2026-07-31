"""Write reinfors collection telemetry to TensorBoard.

Install the optional dependencies with: pip install torch tensorboard
"""

from __future__ import annotations

import argparse

import numpy as np
import reinfors as rf
from torch.utils.tensorboard import SummaryWriter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--updates", type=int, default=10)
    parser.add_argument("--records", type=int, default=128)
    parser.add_argument("--logdir", default="runs/reinfors-telemetry")
    args = parser.parse_args()

    game = rf.games.Connect4()
    engine = rf.Engine(
        game=game,
        reward=rf.Reward(win=1.0, loss=-1.0),
        policy=rf.policies.Mcts(num_simulations=24),
        learner=rf.learners.TreeStrap(),
        n_games=8,
        seed=0,
    )
    n_actions = game.action_space().n

    def infer(obs: np.ndarray) -> np.ndarray:
        return np.zeros((len(obs), 1, n_actions), dtype=np.float64)

    with SummaryWriter(args.logdir) as writer:
        for update in range(args.updates):
            telemetry = engine.collect(n_records=args.records, infer=infer).telemetry
            for key in ("decisions", "infer_calls", "infer_rows", "infer_seconds", "max_depth"):
                writer.add_scalar(f"sampling/{key}", telemetry[key], update)

            if telemetry["infer_calls"]:
                writer.add_scalar(
                    "sampling/rows_per_infer_call",
                    telemetry["infer_rows"] / telemetry["infer_calls"],
                    update,
                )

            episodes = telemetry["episodes"]
            if episodes:
                mean_length = sum(length for _returns, length, _seeded in episodes) / len(episodes)
                writer.add_scalar("episodes/mean_length", mean_length, update)
                for player in range(len(episodes[0][0])):
                    mean_return = sum(returns[player] for returns, _length, _seeded in episodes) / len(episodes)
                    writer.add_scalar(f"episodes/mean_return_player_{player}", mean_return, update)

            print(f"update={update} episodes={len(episodes)} infer_rows={telemetry['infer_rows']}")

    print(f"wrote TensorBoard events to {args.logdir}")


if __name__ == "__main__":
    main()
