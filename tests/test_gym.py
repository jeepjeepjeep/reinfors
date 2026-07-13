"""The standard-API adapters (`reinfors.gym`): a single-agent game conforms to the Gymnasium API and a
simultaneous multi-agent game to the PettingZoo Parallel API — each verified with that project's own
conformance checker. Skipped when the optional backends (`reinfors[gym]`) aren't installed."""

from __future__ import annotations

import pytest
import reinfors as rf
from reinfors import gym

gymnasium = pytest.importorskip("gymnasium")


def _gridworld() -> object:
    return rf.games.GridWorld(size=6, max_ticks=50)


def _snake() -> object:
    return rf.games.Snake(grid_size=8, initial_length=3, food=1, max_ticks=50)


def test_gymnasium_env_conforms_to_the_api() -> None:
    from gymnasium.utils.env_checker import check_env

    env = gym.gymnasium_env(_gridworld(), rf.Reward(goal=1.0, step=-0.01))
    check_env(env.unwrapped, skip_render_check=True)


def test_gymnasium_episode_runs_to_a_terminal_or_the_time_limit() -> None:
    env = gym.make(_gridworld(), rf.Reward(goal=1.0))
    assert isinstance(env, gymnasium.Env)
    env.reset(seed=0)
    terminated = truncated = False
    for _ in range(50):
        _, reward, terminated, truncated, _ = env.step(env.action_space.sample())
        assert isinstance(reward, float)
        if terminated or truncated:
            break
    assert terminated or truncated  # ends by reaching the goal or hitting max_ticks


def test_pettingzoo_parallel_env_conforms_to_the_api() -> None:
    pytest.importorskip("pettingzoo")
    from pettingzoo.test import parallel_api_test

    env = gym.parallel_env(_snake(), rf.Reward(food=1.0, loss=-1.0, kill=1.0))
    assert sorted(env.possible_agents) == ["player_0", "player_1"]
    parallel_api_test(env, num_cycles=200)


def test_make_dispatches_by_game_shape() -> None:
    assert isinstance(gym.make(_gridworld()), gymnasium.Env)  # single-agent -> Gymnasium

    snake = gym.make(_snake())  # simultaneous multi-agent -> PettingZoo Parallel
    assert hasattr(snake, "possible_agents") and len(snake.possible_agents) == 2

    with pytest.raises(NotImplementedError):  # turn-based multi-agent not yet exposed
        gym.make(rf.games.Connect4())
