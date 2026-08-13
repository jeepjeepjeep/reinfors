"""Shared test helpers."""

import numpy as np
import reinfors as rf


def az_zeros_infer(n_actions: int):
    def infer(obs, n=None):
        m = obs.shape[0]
        return np.zeros((m, n_actions), dtype=np.float32), np.zeros(m, dtype=np.float32)

    return infer


def connect4_az_engine(*, num_simulations: int = 12, **kwargs) -> rf.Engine:
    kwargs.setdefault("n_games", 4)
    kwargs.setdefault("seed", 5)
    return rf.Engine(
        rf.games.Connect4(),
        rf.Reward(win=1.0, loss=-1.0),
        rf.policies.AlphaZero(num_simulations=num_simulations),
        rf.learners.AlphaZero(gamma=1.0),
        **kwargs,
    )
