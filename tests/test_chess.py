"""Chess through the Python surface: spaces, the AlphaZero and Mcts pairings at the 4672-action
width (legal masking end to end), Env play, and determinism."""

from typing import Any

import numpy as np
import pytest
import reinfors as rf

_A = 4672


def _az_infer(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.zeros((arr.shape[0], _A)), np.zeros(arr.shape[0])


def _engine(seed: int = 0, **kwargs: Any) -> rf.Engine:
    return rf.Engine(
        rf.games.Chess(max_ticks=60),  # short horizon: zeros-net games otherwise shuffle for a while
        rf.Reward(win=1.0, loss=-1.0),
        rf.policies.AlphaZero(num_simulations=8, temperature_drop=8),
        rf.learners.AlphaZero(),
        n_games=2,
        seed=seed,
        **kwargs,
    )


def test_spaces() -> None:
    game = rf.games.Chess()
    assert tuple(game.observation_space().shape) == (19, 8, 8)
    assert game.action_space().n == _A
    assert game.truncation_horizon() == 512
    assert rf.games.Chess(max_ticks=None).truncation_horizon() is None
    assert tuple(rf.games.Chess(encoding="az119").observation_space().shape) == (119, 8, 8)
    with pytest.raises(ValueError, match="encoding"):
        rf.games.Chess(encoding="huge")


def test_az119_engine_collects() -> None:
    def infer(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        assert arr.shape[1] == 119 * 8 * 8  # the history-bearing view reaches the net
        return np.zeros((arr.shape[0], _A)), np.zeros(arr.shape[0])

    engine = rf.Engine(
        rf.games.Chess(max_ticks=40, encoding="az119"),
        rf.Reward(win=1.0, loss=-1.0),
        rf.policies.AlphaZero(num_simulations=8),
        rf.learners.AlphaZero(),
        n_games=1,
        seed=0,
    )
    obs, pi, _, _ = engine.collect(20, infer)
    assert obs.shape == (obs.shape[0], 119 * 8 * 8)
    np.testing.assert_allclose(pi.sum(axis=1), 1.0, atol=1e-12)


def test_az119_env_observation() -> None:
    env = rf.Env(rf.games.Chess(encoding="az119"))
    obs = env.observe(0)
    assert obs.shape == (119, 8, 8)
    assert obs[112].sum() == 64.0  # white to move
    # history steps beyond t=0 are empty at the start position
    assert obs[14:112].sum() == 0.0


def test_alphazero_collect_masks_illegal_actions() -> None:
    obs, pi, z, telemetry = _engine().collect(60, _az_infer)
    m = obs.shape[0]
    assert m >= 60
    assert obs.shape == (m, 19 * 8 * 8) and pi.shape == (m, _A)
    np.testing.assert_allclose(pi.sum(axis=1), 1.0, atol=1e-12)
    # Legal masking end to end: chess never has more than ~218 legal moves, so every π row must be
    # supported on a tiny fraction of the 4672 ids.
    assert ((pi > 0).sum(axis=1) <= 218).all()
    assert telemetry["decisions"] > 0
    # Truncated episodes bootstrap z from the (zero) value head; outcomes stay in [-1, 1].
    assert (np.abs(z) <= 1.0).all()


def test_mcts_treestrap_pairs() -> None:
    def infer(arr: np.ndarray) -> np.ndarray:
        return np.zeros((arr.shape[0], 1, _A))

    engine = rf.Engine(
        rf.games.Chess(max_ticks=40),
        None,
        rf.policies.Mcts(num_simulations=8),
        rf.learners.TreeStrap(),
        n_games=1,
        seed=0,
    )
    obs, tgt, mask, _ = engine.collect(20, infer)
    assert tgt.shape == (obs.shape[0], 1, _A)
    assert mask.shape == (obs.shape[0], 1)


def test_collect_is_deterministic_per_seed() -> None:
    b1 = _engine(7).collect(40, _az_infer)
    b2 = _engine(7).collect(40, _az_infer)
    assert isinstance(b1, rf._reinfors.AlphaZeroBatch) and isinstance(b2, rf._reinfors.AlphaZeroBatch)
    assert np.array_equal(b1.obs, b2.obs)
    assert np.array_equal(b1.policy_targets, b2.policy_targets)


def test_env_plays_uci_like_moves() -> None:
    env = rf.Env(rf.games.Chess())
    assert env.num_agents() == 2 and env.action_count() == _A
    st = env.state()
    assert st["fen"].startswith("rnbqkbnr/pppppppp") and st["turn"] == 0 and not st["done"]
    legal = env.legal_actions(0)
    assert len(legal) == 20  # the classic starting-position move count
    assert env.legal_actions(1) == []  # non-mover has no actions (sequential game)
    events = env.step({0: legal[0]})
    assert events == ["ongoing", "ongoing"]
    assert env.state()["turn"] == 1 and env.active_agents() == [1]


def test_env_illegal_action_loses() -> None:
    env = rf.Env(rf.games.Chess(), reward=rf.Reward(win=1.0, loss=-1.0))
    legal = set(env.legal_actions(0))
    bogus = next(a for a in range(_A) if a not in legal)
    events = env.step({0: bogus})
    assert events == ["loss", "win"]
    assert env.done()
    assert env.rewards == [-1.0, 1.0]


def test_start_buffer_rejected() -> None:
    with pytest.raises(ValueError, match="snake"):
        rf.Engine(
            rf.games.Chess(),
            None,
            rf.policies.AlphaZero(),
            rf.learners.AlphaZero(),
            n_games=1,
            start_buffer=True,
        )
