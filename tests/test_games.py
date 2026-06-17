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


# --- Model-free DQN: a second algorithm through the same generic engine, with a different record shape
# (off-policy transitions instead of TreeStrap targets) — the seam + binding generalization. ---


def _dqn_engine() -> object:
    # size, goal_row, goal_col, step_reward, goal_reward, epsilon, n_heads, bootstrap_p, max_ticks,
    # seed, n_games
    return reinfors._reinfors.DqnGridWorldEngine(5, 0, 1, 0.0, 1.0, 0.1, _K, 1.0, 10, 0, 3)


def test_dqn_engine_emits_well_formed_transitions() -> None:
    engine = _dqn_engine()
    dim = 2 * 5 * 5
    obs, actions, rewards, next_obs, dones, masks, telemetry = engine.collect(60, _dummy_infer(4))
    m = obs.shape[0]
    assert m >= 60
    assert obs.shape == (m, dim) and obs.dtype == np.float32
    assert actions.shape == (m,) and actions.dtype == np.int64
    assert rewards.shape == (m,) and rewards.dtype == np.float64
    assert next_obs.shape == (m, dim) and next_obs.dtype == np.float32
    assert dones.shape == (m,) and dones.dtype == bool
    assert masks.shape == (m, _K) and np.isin(masks, (0.0, 1.0)).all()
    assert (actions >= 0).all() and (actions < 4).all()
    assert "episodes" in telemetry and telemetry["decisions"] > 0


def test_dqn_transitions_drive_a_td_step() -> None:
    # Smoke: the binding's transitions feed a bootstrapped-DQN TD update end to end (finite loss). Uses
    # the online net as its own target — enough to prove the transition record trains, not a full DQN.
    pytest.importorskip("torch")
    import torch
    from reinfors.training import BootstrappedQNetwork, make_infer

    net = BootstrappedQNetwork((2, 5, 5), n_actions=4, n_heads=_K)
    engine = _dqn_engine()
    obs, actions, rewards, next_obs, dones, masks, _ = engine.collect(64, make_infer(net))

    o = torch.from_numpy(obs).reshape(-1, 2, 5, 5)
    no = torch.from_numpy(next_obs).reshape(-1, 2, 5, 5)
    a = torch.from_numpy(actions).long()
    r = torch.from_numpy(rewards).float()
    d = torch.from_numpy(dones).float()
    mask = torch.from_numpy(masks)

    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    q = net(o)  # (M, K, A)
    with torch.no_grad():
        target = r[:, None] + 0.99 * (1.0 - d[:, None]) * net(no).max(dim=-1).values  # (M, K)
    chosen = q.gather(-1, a[:, None, None].expand(-1, _K, 1)).squeeze(-1)  # (M, K)
    loss = (mask * (chosen - target) ** 2).sum() / mask.sum().clamp(min=1.0)
    opt.zero_grad()
    loss.backward()
    opt.step()
    assert torch.isfinite(loss)
