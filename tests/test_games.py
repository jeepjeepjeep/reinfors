"""The non-snake games (Connect-4, GridWorld) driven through the generic core: each engine collects
TreeStrap records of the game's `(K, action_count)` shape, the registry reports the right
`(obs_shape, action_count)`, and (torch-gated) end-to-end training runs on a non-snake game.

A dummy numpy `infer` (zeros of the right K/A) keeps the shape/telemetry tests torch-free, so they run
in CI; the training smoke test is gated on torch.
"""

from __future__ import annotations

import numpy as np
import pytest
import reinfors
from reinfors import games

_K = 2
_TELEMETRY_KEYS = {
    "episodes",
    "decisions",
    "max_depth",
    "mean_leaves",
    "mean_rounds",
    "mean_expansions",
    "mean_sigma",
    "mean_disagreement",
}


def _connect4_engine() -> object:
    # win, loss, draw rewards; then search + rollout knobs.
    return reinfors._reinfors.Connect4Engine(
        1.0,
        -1.0,
        0.0,  # win, loss, draw
        0.99,
        1.0,
        24,
        4,
        6,  # gamma, beta, budget, top_k, max_depth
        "uniform",
        1.0,
        0.1,  # opponent, opp_temperature, opp_floor
        0.0,
        30,
        _K,  # epsilon, max_ticks, n_heads
        0.5,
        False,
        1.0,  # outcome_weight, interior_targets, bootstrap_p
        0,
        2,  # seed, n_games
    )


def _gridworld_engine() -> object:
    return reinfors._reinfors.GridWorldEngine(
        5,
        0,
        1,  # size, goal_row, goal_col
        0.0,
        1.0,  # step_reward, goal_reward
        0.99,
        1.0,
        24,
        4,
        6,  # gamma, beta, budget, top_k, max_depth
        "uniform",
        1.0,
        0.1,  # opponent, opp_temperature, opp_floor
        0.0,
        30,
        _K,  # epsilon, max_ticks, n_heads
        0.5,
        False,
        1.0,  # outcome_weight, interior_targets, bootstrap_p
        0,
        2,  # seed, n_games
    )


def _dummy_infer(a: int) -> object:
    def infer(arr: np.ndarray) -> np.ndarray:
        return np.zeros((arr.shape[0], _K, a), dtype=np.float64)

    return infer


@pytest.mark.parametrize(
    ("make_engine", "action_count"),
    [(_connect4_engine, 7), (_gridworld_engine, 4)],
)
def test_collect_shapes_and_telemetry(make_engine, action_count: int) -> None:
    engine = make_engine()
    obs, tgt, mask, telemetry = engine.collect(40, _dummy_infer(action_count))
    m = obs.shape[0]
    assert m >= 40
    assert tgt.shape == (m, _K, action_count) and tgt.dtype == np.float64
    assert mask.shape == (m, _K)
    assert set(telemetry) >= _TELEMETRY_KEYS
    assert telemetry["decisions"] > 0


@pytest.mark.parametrize(
    ("make_engine", "action_count", "num_agents"),
    [(_connect4_engine, 7, 2), (_gridworld_engine, 4, 1)],
)
def test_episode_reward_is_per_agent(make_engine, action_count: int, num_agents: int) -> None:
    # The engine generalizes to any agent count: each finished episode's telemetry carries one reward
    # per agent — length 1 for single-agent GridWorld, 2 for Connect-4 — not a hardcoded pair.
    engine = make_engine()
    _, _, _, telemetry = engine.collect(150, _dummy_infer(action_count))
    episodes = telemetry["episodes"]
    assert len(episodes) > 0
    for rewards, length in episodes:
        assert len(rewards) == num_agents
        assert length >= 1 and all(np.isfinite(r) for r in rewards)


@pytest.mark.parametrize(
    ("name", "kwargs", "obs_shape", "action_count"),
    [
        ("snake", {"grid_size": 12}, (5, 12, 12), 3),
        ("connect4", {}, (2, 6, 7), 7),
        ("gridworld", {"size": 5}, (2, 5, 5), 4),
    ],
)
def test_registry_net_shape(name: str, kwargs: dict, obs_shape: tuple, action_count: int) -> None:
    shape, actions = games.net_shape(name, **kwargs)
    assert shape == obs_shape
    assert actions == action_count
    assert games.get(name).engine is not None


def test_registry_rejects_unknown_game() -> None:
    with pytest.raises(KeyError):
        games.get("pong")


def test_connect4_end_to_end_training() -> None:
    pytest.importorskip("torch")
    import torch
    from reinfors.training import BootstrappedQNetwork, train

    obs_shape, n_actions = games.net_shape("connect4")
    net = BootstrappedQNetwork(obs_shape, n_actions=n_actions, n_heads=_K)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    engine = _connect4_engine()
    losses = train(
        engine,
        net,
        optimizer,
        iterations=2,
        collect_size=20,
        batch_size=8,
    )
    assert len(losses) >= 1
    assert all(np.isfinite(loss) for loss in losses)
