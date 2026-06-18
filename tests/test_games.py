"""The non-snake games (Connect-4, GridWorld) and the DQN family driven through the unified `Engine`:
each composition collects records of the right shape, telemetry carries one reward per agent, and the
name registries resolve. These are engine-contract tests: a dummy numpy `infer` (zeros of the right
K/A) keeps them torch-free — the model and gradient step live in the consumer, not reinfors.
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


def test_game_handles_advertise_spaces() -> None:
    # A handle reports the observation Box (whose shape sizes the network input) and action Discrete,
    # so a network can be sized from the game instead of hard-coding its dimensions.
    obs = rf.games.Snake(grid_size=12).observation_space()
    assert isinstance(obs, rf.spaces.Box) and obs.shape == (5, 12, 12)
    assert obs.low.shape == obs.shape == obs.high.shape  # bounds broadcast to the obs shape
    assert np.isneginf(obs.low).all() and np.isposinf(obs.high).all()
    act = rf.games.Snake(grid_size=12).action_space()
    assert isinstance(act, rf.spaces.Discrete) and act.n == 3
    # The non-snake games advertise their own shapes (mirrors the Rust `spaces` test).
    assert rf.games.Connect4().observation_space().shape == (2, 6, 7)
    assert rf.games.Connect4().action_space().n == 7
    assert rf.games.GridWorld(size=5).observation_space().shape == (2, 5, 5)
    assert rf.games.GridWorld(size=5).action_space().n == 4


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


def test_engine_from_config_round_trips_a_yaml_shaped_dict() -> None:
    # A config shaped like parsed YAML — a nested `reward` mapping, not a pre-built handle — builds a
    # working engine: engine_from_config wraps the reward dict into rf.Reward automatically.
    config = {
        "game": {"name": "snake", "grid_size": 8, "reward": {"food": 1.0, "loss": -10.0}},
        "policy": {"name": "selective_expectimax", "n_heads": _K, "expansion_budget": 16, "max_depth": 6},
        "learner": {"name": "treestrap", "gamma": 0.99},
        "engine": {"n_games": 2, "max_ticks": 10, "seed": 0},
    }
    engine = rf.engine_from_config(config)
    _, _, _, telemetry = engine.collect(20, _dummy_infer(3))
    assert telemetry["decisions"] > 0
    # Reward validation still fires through the config path (the wrapped dict isn't trusted blindly).
    bad = {**config, "game": {"name": "snake", "grid_size": 8, "reward": {"goal": 1.0}}}
    with pytest.raises(ValueError, match="unknown reward key"):
        rf.engine_from_config(bad)


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


@pytest.mark.parametrize(
    "bad",
    [
        {"expansion_budget": 0},
        {"top_k": 0},
        {"max_depth": 0},
        {"food_samples": 0},
        {"beta": 1.5},
    ],
)
def test_engine_rejects_degenerate_search_params(bad: dict) -> None:
    # SelectiveExpectimax search knobs are validated at Engine construction (the core does not).
    kw = {"expansion_budget": 24, "top_k": 4, "max_depth": 6, "beta": 1.0, "food_samples": 1, "n_heads": _K, **bad}
    with pytest.raises(ValueError):
        rf.Engine(
            rf.games.Snake(grid_size=8),
            rf.policies.SelectiveExpectimax(**kw),
            rf.learners.TreeStrap(),
            n_games=1,
            max_ticks=10,
        )


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
