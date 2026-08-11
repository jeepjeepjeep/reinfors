"""Run a paired Arena match between searched and external Connect Four agents."""

from __future__ import annotations

import argparse

import numpy as np
import reinfors as rf


class CenterBot:
    """A stateless external agent; subprocess-backed engines use the same contract."""

    def act(self, view: rf.arena.View) -> int:
        return min(view.legal_actions, key=lambda action: (abs(action - 3), action))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=8)
    parser.add_argument("--simulations", type=int, default=8)
    parser.add_argument("--slots", type=int, default=4)
    args = parser.parse_args()

    game = rf.games.Connect4()
    reward = rf.Reward(win=1.0, loss=-1.0)
    n_actions = game.action_space().n

    def infer(obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Replace these uniform outputs with the searched contestant's network."""
        rows = len(obs)
        logits = np.zeros((rows, n_actions), dtype=np.float32)
        values = np.zeros(rows, dtype=np.float32)
        return logits, values

    searched = (
        rf.policies.AlphaZero(
            num_simulations=args.simulations,
            temperature=0.0,
            noise=None,
        ),
        infer,
        1.0,  # gamma used when search backs up rewards
    )
    external = rf.arena.External(CenterBot, workers=args.slots, timeout=10.0)
    arena = rf.Arena(
        game,
        reward,
        contestants=[searched, external],
        n_slots=args.slots,
        start=rf.starts.RandomStartingMoves(2),
        seed=0,
    )

    result = arena.play(args.games)
    mean, stderr = result.payoff(0)
    score = mean / 2.0 + 0.5
    print(f"searched score={score:.3f} +/- {stderr / 2.0:.3f} over {len(result.games) // 2} opening pairs")
    print(f"searched payoff by seat: {result.seat_payoffs(0)}")


if __name__ == "__main__":
    main()
