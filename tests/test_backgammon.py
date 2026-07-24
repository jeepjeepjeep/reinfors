"""Backgammon through the Python surface: construction, spaces, env stepping with the state dict,
and an engine-level AZ collect over the declared dice chance."""

import numpy as np
import pytest
import reinfors as rf
from reinfors._reinfors import AlphaZeroBatch


def test_construction_and_spaces() -> None:
    g = rf.games.Backgammon()
    assert g.action_space().n == 1352
    assert tuple(g.observation_space().shape) == (200, 1, 1)
    assert g.truncation_horizon() == 1000
    rf.games.Backgammon(max_ticks=None)
    with pytest.raises(ValueError):
        rf.games.Backgammon(max_ticks=0)


def test_reward_keys() -> None:
    rf.Engine(
        rf.games.Backgammon(),
        rf.Reward(win=1.0, gammon=2.0, backgammon=3.0),
        rf.policies.Mcts(num_simulations=4),
        rf.learners.TreeStrap(),
        n_games=1,
        seed=0,
    )
    with pytest.raises(ValueError):
        rf.Engine(
            rf.games.Backgammon(),
            rf.Reward(win=1.0, draw=0.0),  # draw is not a backgammon key
            rf.policies.Mcts(num_simulations=4),
            rf.learners.TreeStrap(),
            n_games=1,
            seed=0,
        )


def test_env_steps_and_exposes_state() -> None:
    env = rf.Env(rf.games.Backgammon(), rf.Reward(win=1.0), seed=3)
    s = env.state()
    assert s["dice"][0] != s["dice"][1], "opening roll is never a double"
    assert sum(s["board"][0]) + s["bar"][0] + s["scores"][0] == 15
    mover = env.active_agents()[0]
    legal = env.legal_actions(mover)
    assert legal, "a fresh position has legal actions"
    events = env.step({mover: legal[0]})
    assert events[mover]["result"] in ("ongoing", "win", "loss")
    s2 = env.state()
    assert sum(s2["board"][0]) + s2["bar"][0] + s2["scores"][0] == 15


def test_alphazero_collects_on_backgammon() -> None:
    # The full composition: 1352-wide legal masking + the non-uniform 21-roll declared chance,
    # through the engine with the AZ tuple infer contract.
    def infer(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros((arr.shape[0], 1352)), np.zeros(arr.shape[0])

    engine = rf.Engine(
        rf.games.Backgammon(max_ticks=60),
        rf.Reward(win=1.0, gammon=2.0, backgammon=3.0),
        rf.policies.AlphaZero(num_simulations=8),
        rf.learners.AlphaZero(),
        n_games=2,
        seed=0,
    )
    batch = engine.collect(40, infer)
    assert isinstance(batch, AlphaZeroBatch)
    assert batch.obs.shape[0] >= 40
    assert batch.obs.shape[1] == 200
    assert batch.policy_targets.shape[1] == 1352
    rows = batch.policy_targets.sum(axis=1)
    assert np.allclose(rows[rows > 0], 1.0)
    assert np.all(np.isfinite(batch.value_targets))
