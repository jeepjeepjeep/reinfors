"""The non-snake games (Connect-4, GridWorld) and the DQN family driven through the unified `Engine`:
each composition collects records of the right shape, telemetry carries one reward per agent, and the
name registries resolve. A dummy numpy `infer` (zeros of the right K/A) keeps the shape/telemetry
tests torch-free; the training smoke tests are gated on torch.
"""

from __future__ import annotations

import numpy as np
import pytest
import reinfors as rf

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


def _selective() -> object:
    return rf.policies.SelectiveExpectimax(
        expansion_budget=24,
        top_k=4,
        max_depth=6,
        beta=1.0,
        food_samples=1,
        n_heads=_K,
        epsilon=0.0,
        opponent="uniform",
        opp_temperature=1.0,
        opp_floor=0.1,
    )


def _treestrap() -> object:
    return rf.learners.TreeStrap(gamma=0.99, outcome_weight=0.5, bootstrap_p=1.0, interior_targets=False)


def _connect4_engine() -> object:
    return rf.Engine(
        rf.games.Connect4(reward=rf.Reward(win=1.0, loss=-1.0, draw=0.0)),
        _selective(),
        _treestrap(),
        n_games=2,
        max_ticks=30,
        seed=0,
    )


def _gridworld_engine() -> object:
    return rf.Engine(
        rf.games.GridWorld(size=5, goal_row=0, goal_col=1, reward=rf.Reward(step=0.0, goal=1.0)),
        _selective(),
        _treestrap(),
        n_games=2,
        max_ticks=30,
        seed=0,
    )


def _dqn_engine() -> object:
    return rf.Engine(
        rf.games.GridWorld(size=5, goal_row=0, goal_col=1, reward=rf.Reward(step=0.0, goal=1.0)),
        rf.policies.EpsilonGreedyQ(n_heads=_K, epsilon=0.1),
        rf.learners.Dqn(bootstrap_p=1.0),
        n_games=3,
        max_ticks=10,
        seed=0,
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


def test_registries_list_the_built_in_names() -> None:
    assert rf.registered_games() == ["connect4", "gridworld", "snake"]
    assert rf.registered_policies() == ["epsilon_greedy_q", "selective_expectimax"]
    assert rf.registered_learners() == ["dqn", "treestrap"]


def test_make_constructs_and_rejects_unknown() -> None:
    # The name-addressable path builds the same handles the typed constructors do.
    engine = rf.Engine(
        rf.make_game("connect4"),
        rf.make_policy("selective_expectimax", n_heads=_K),
        rf.make_learner("treestrap"),
        n_games=2,
        max_ticks=10,
    )
    _, _, _, telemetry = engine.collect(20, _dummy_infer(7))
    assert telemetry["decisions"] > 0
    with pytest.raises(KeyError):
        rf.make_game("pong")
    with pytest.raises(KeyError):
        rf.make_policy("a2c")


def test_reward_rejects_keys_not_valid_for_the_game() -> None:
    # The generic Reward is validated per game: any key the game doesn't define is an error (not
    # silently ignored). Valid keys still work, and missing ones fall back to the game's default.
    rf.games.Snake(reward=rf.Reward(food=1.0, loss=-10.0))  # snake keys: ok
    rf.games.Connect4(reward=rf.Reward(win=1.0))  # connect4 keys: ok
    with pytest.raises(ValueError, match="unknown reward key"):
        rf.games.Snake(reward=rf.Reward(goal=1.0))  # 'goal' is gridworld's, not snake's
    with pytest.raises(ValueError, match="unknown reward key"):
        rf.games.Connect4(reward=rf.Reward(food=1.0))  # 'food' is snake's, not connect4's


def test_incompatible_policy_learner_pairing_is_rejected() -> None:
    # A search learner with a Q policy (mismatched evaluation type) must fail at Engine construction.
    with pytest.raises(ValueError):
        rf.Engine(
            rf.games.GridWorld(size=5),
            rf.policies.EpsilonGreedyQ(n_heads=_K),
            rf.learners.TreeStrap(),
            n_games=1,
            max_ticks=10,
        )


def test_connect4_end_to_end_training() -> None:
    pytest.importorskip("torch")
    import torch
    from reinfors.training import BootstrappedQNetwork, train

    net = BootstrappedQNetwork((2, 6, 7), n_actions=7, n_heads=_K)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    losses = train(
        _connect4_engine(),
        net,
        optimizer,
        iterations=2,
        collect_size=20,
        batch_size=8,
    )
    assert len(losses) >= 1
    assert all(np.isfinite(loss) for loss in losses)


# --- Model-free DQN: a second algorithm through the same unified engine, with a different record shape
# (off-policy transitions instead of TreeStrap targets) — the seam + binding generalization. ---


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
