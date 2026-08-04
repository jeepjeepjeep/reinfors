"""Collect a small AlphaZero batch without a machine-learning framework."""

from __future__ import annotations

import argparse

import numpy as np
import reinfors as rf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=int, default=128)
    parser.add_argument("--games", type=int, default=8)
    args = parser.parse_args()

    game = rf.games.Connect4()
    engine = rf.Engine(
        game=game,
        reward=rf.Reward(win=1.0, loss=-1.0, draw=0.0),
        policy=rf.policies.AlphaZero(num_simulations=24),
        learner=rf.learners.AlphaZero(),
        n_games=args.games,
        seed=0,
    )
    n_actions = game.action_space().n

    def infer(obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        logits = np.zeros((len(obs), n_actions), dtype=np.float64)
        values = np.zeros(len(obs), dtype=np.float64)
        return logits, values

    batch = engine.collect(n_records=args.records, infer=infer)
    print(
        f"records={len(batch.obs)} obs={batch.obs.shape} "
        f"policy_targets={batch.policy_targets.shape} value_targets={batch.value_targets.shape}"
    )
    print(f"infer_calls={batch.telemetry['infer_calls']} infer_rows={batch.telemetry['infer_rows']}")


if __name__ == "__main__":
    main()
