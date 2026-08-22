"""Minimax baseline: composition gates, collection, determinism, and config round-trip."""

from __future__ import annotations

import numpy as np
import pytest
import reinfors as rf


def zero_infer(obs: np.ndarray) -> np.ndarray:
    return np.zeros((len(obs), 1, 7), dtype=np.float32)


def connect4_engine(seed: int = 0, **policy_kwargs: object) -> rf.Engine:
    return rf.Engine(
        rf.games.Connect4(),
        rf.Reward(win=1.0, loss=-1.0),
        rf.policies.Minimax(**policy_kwargs),
        rf.learners.TreeStrap(),
        n_games=2,
        seed=seed,
        n_threads=1,
    )


def test_zero_callback_collects_treestrap_batches() -> None:
    batch = connect4_engine(depth=2).collect(n_records=8, infer=zero_infer)
    assert isinstance(batch, rf.TreeStrapBatch)
    records = len(batch.obs)
    assert records >= 8
    assert batch.targets.shape == (records, 1, 7)
    assert batch.masks.shape == (records, 1)
    assert batch.telemetry["decisions"] > 0
    assert batch.telemetry["infer_calls"] > 0
    # One pooled inference round per depth level of the lockstep frontier.
    assert batch.telemetry["mean_rounds"] <= 2.0 + 1e-9


def test_deterministic_given_the_evaluator() -> None:
    a = connect4_engine(seed=7, depth=2).collect(n_records=8, infer=zero_infer)
    b = connect4_engine(seed=7, depth=2).collect(n_records=8, infer=zero_infer)
    assert np.array_equal(a.obs, b.obs)
    assert np.array_equal(a.targets, b.targets)


def test_registry_and_resolved_config_round_trip() -> None:
    named = rf.policies.make("minimax", depth=3, top_k=5)
    engine = rf.Engine(
        rf.games.Connect4(),
        rf.Reward(win=1.0, loss=-1.0),
        named,
        rf.learners.TreeStrap(),
        n_games=1,
        seed=0,
    )
    config = engine.resolved_config()
    assert config["policy"]["name"] == "minimax"
    assert config["policy"]["depth"] == 3
    assert config["policy"]["top_k"] == 5
    rebuilt = rf.engine_from_config(config)
    assert rebuilt.config_fingerprint() == engine.config_fingerprint()


def test_an_omitted_beam_renders_null_and_keeps_fingerprints_stable() -> None:
    explicit = connect4_engine(depth=2, top_k=None)
    implicit = connect4_engine(depth=2)
    assert explicit.resolved_config()["policy"]["top_k"] is None
    assert explicit.config_fingerprint() == implicit.config_fingerprint()


def test_chance_defaults_to_the_exact_expectation() -> None:
    config = connect4_engine(depth=2).resolved_config()
    assert config["policy"]["chance"] == {"name": "expand_all"}


@pytest.mark.parametrize(
    "game",
    [
        rf.games.Snake(grid_size=8),
        rf.games.GridWorld(size=4),
    ],
)
def test_rejects_non_two_player_sequential_games(game: object) -> None:
    with pytest.raises(ValueError, match="two-player sequential"):
        rf.Engine(
            game,
            None,
            rf.policies.Minimax(depth=2),
            rf.learners.TreeStrap(),
            n_games=1,
        )


def test_rejects_hidden_information_games() -> None:
    with pytest.raises(ValueError, match="clairvoyant"):
        rf.Engine(
            rf.games.KuhnPoker(),
            None,
            rf.policies.Minimax(depth=2),
            rf.learners.TreeStrap(),
            n_games=1,
        )


def test_rejects_non_treestrap_learners() -> None:
    with pytest.raises(ValueError, match="incompatible policy/learner composition"):
        rf.Engine(
            rf.games.Connect4(),
            rf.Reward(win=1.0, loss=-1.0),
            rf.policies.Minimax(depth=2),
            rf.learners.Dqn(),
            n_games=1,
        )


def test_rejects_non_zero_sum_rewards() -> None:
    for bad in (
        rf.Reward(win=1.0, loss=0.0),
        rf.Reward(win=1.0, loss=-1.0, draw=0.5),
    ):
        with pytest.raises(ValueError, match="antisymmetric"):
            rf.Engine(
                rf.games.Connect4(),
                bad,
                rf.policies.Minimax(depth=2),
                rf.learners.TreeStrap(),
                n_games=1,
            )


def test_accepts_zero_sum_rewards_and_backgammon_margins() -> None:
    # Scaled antisymmetric weights pass; None uses the zero-sum schema defaults.
    for reward in (rf.Reward(win=2.0, loss=-2.0, draw=0.0), None):
        rf.Engine(
            rf.games.Connect4(),
            reward,
            rf.policies.Minimax(depth=2),
            rf.learners.TreeStrap(),
            n_games=1,
        )
    # Backgammon events carry their own sign, so any margin weights stay zero-sum.
    rf.Engine(
        rf.games.Backgammon(),
        rf.Reward(win=1.0, gammon=4.0, backgammon=9.0),
        rf.policies.Minimax(depth=2),
        rf.learners.TreeStrap(),
        n_games=1,
    )


def test_choose_rejects_non_zero_sum_env_rewards() -> None:
    envs = [rf.Env(rf.games.Connect4(), reward=rf.Reward(win=1.0, loss=0.0), seed=0)]
    with pytest.raises(ValueError, match="antisymmetric"):
        rf.policies.Minimax(depth=2).choose(envs, zero_infer, gamma=1.0)


def test_constructor_validation() -> None:
    with pytest.raises(ValueError, match="at least one ply"):
        rf.policies.Minimax(depth=0)
    with pytest.raises(ValueError, match="at least one move"):
        rf.policies.Minimax(depth=2, top_k=0)
    with pytest.raises(ValueError, match="per-traversal"):
        rf.policies.Minimax(depth=2, chance=rf.chance_modes.AlwaysResample())
