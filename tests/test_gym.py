"""The standard-API adapters (`reinfors.gym`): a single-agent game conforms to the Gymnasium API, a
simultaneous multi-agent game to the PettingZoo Parallel API, and a turn-based multi-agent game to the
PettingZoo AEC API — each verified with that project's own conformance checker. Skipped when the
optional backends (`reinfors[gym]`) aren't installed."""

from __future__ import annotations

import numpy as np
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


def test_pettingzoo_parallel_infos_carry_the_action_mask() -> None:
    pytest.importorskip("pettingzoo")
    env = gym.parallel_env(_snake(), rf.Reward(food=1.0, loss=-1.0))
    _, infos = env.reset(seed=0)
    assert all(infos[a]["action_mask"].tolist() == [1, 1, 1] for a in env.agents)
    _, _, terminations, truncations, infos = env.step(dict.fromkeys(env.agents, 0))
    for a in infos:
        expected = [0, 0, 0] if terminations[a] or truncations[a] else [1, 1, 1]
        assert infos[a]["action_mask"].tolist() == expected


@pytest.mark.parametrize(
    ("game", "num_cycles"),
    [
        (lambda: rf.games.Connect4(), 60),
        (lambda: rf.games.Chess(max_ticks=80), 120),  # short horizon: exercises the truncation dance
        (lambda: rf.games.Backgammon(max_ticks=200), 250),
    ],
    ids=["connect4", "chess", "backgammon"],
)
def test_pettingzoo_aec_env_conforms_to_the_api(game: object, num_cycles: int) -> None:
    pytest.importorskip("pettingzoo")
    from pettingzoo.test import api_test

    env = gym.aec_env(game())  # type: ignore[operator]
    assert sorted(env.possible_agents) == ["player_0", "player_1"]
    api_test(env, num_cycles=num_cycles)


def test_aec_masks_gate_legality_and_rewards_are_zero_sum() -> None:
    pytest.importorskip("pettingzoo")
    env = gym.aec_env(rf.games.Connect4(), seed=7)
    env.reset(seed=7)
    rng = np.random.default_rng(7)
    outcomes: dict[str, float] = {}
    for agent in env.agent_iter():
        obs, reward, terminated, truncated, _ = env.last()
        mask = obs["action_mask"]
        for other in env.agents:  # only the mover's mask names actions
            if other != agent:
                assert not env.observe(other)["action_mask"].any()
        if terminated or truncated:
            outcomes[agent] = reward
            env.step(None)
            continue
        assert mask.any()
        env.step(int(rng.choice(np.flatnonzero(mask))))
    assert sorted(outcomes) == ["player_0", "player_1"]
    assert sum(outcomes.values()) == 0.0  # connect4 default reward is win/loss symmetric


def test_aec_reset_seed_reproduces_the_episode() -> None:
    pytest.importorskip("pettingzoo")

    def rollout() -> list[tuple[str, int, bytes]]:
        env = gym.aec_env(rf.games.Backgammon(max_ticks=60))
        env.reset(seed=11)
        rng = np.random.default_rng(11)
        trace = []
        for agent in env.agent_iter(max_iter=120):
            obs, _, terminated, truncated, _ = env.last()
            if terminated or truncated:
                env.step(None)
                continue
            action = int(rng.choice(np.flatnonzero(obs["action_mask"])))
            trace.append((agent, action, obs["observation"].tobytes()))
            env.step(action)
        return trace

    assert rollout() == rollout()  # dice included: seeding pins the chance stream


def test_make_dispatches_by_game_shape() -> None:
    assert isinstance(gym.make(_gridworld()), gymnasium.Env)  # single-agent -> Gymnasium

    snake = gym.make(_snake())  # simultaneous multi-agent -> PettingZoo Parallel
    assert hasattr(snake, "possible_agents") and len(snake.possible_agents) == 2

    pettingzoo = pytest.importorskip("pettingzoo")
    c4 = gym.make(rf.games.Connect4())  # turn-based multi-agent -> PettingZoo AEC
    assert isinstance(c4, pettingzoo.AECEnv)
