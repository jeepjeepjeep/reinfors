"""`rf.Env` — the caller-driven single-game instance: a sequential game (Connect-4) played to a known
win, and a simultaneous game (snake) stepped a tick. Mirrors the Rust `Env` integration tests."""

import numpy as np
import pytest
import reinfors as rf


def test_connect4_played_to_a_win() -> None:
    env = rf.Env(rf.games.Connect4(), seed=0)
    # P0 stacks column 0, P1 column 1; P0 completes four-in-a-column first. Moves alternate, matching
    # the turn order `active_agents` reports.
    moves = [(0, 0), (1, 1), (0, 0), (1, 1), (0, 0), (1, 1), (0, 0)]
    last: list[float] = []
    for agent, col in moves:
        assert env.active_agents() == [agent]  # sequential: one mover per tick
        last = env.step({agent: col})
    assert env.done()
    assert last[0] > 0 and last[1] < 0  # zero-sum: P0 wins
    assert env.active_agents() == []


def test_native_state_is_renderable_per_game() -> None:
    # The native state() exposes interpretable, game-specific structure (for rendering / human play).
    c4 = rf.Env(rf.games.Connect4(), seed=0)
    s = c4.state()
    assert s["turn"] == 0 and s["done"] is False
    assert len(s["board"]) == 6 and len(s["board"][0]) == 7
    assert all(cell == 0 for row in s["board"] for cell in row)  # empty board
    c4.step({0: 3})
    assert c4.state()["board"][0][3] == 1  # P0's piece at the bottom of column 3

    st = rf.Env(rf.games.Snake(grid_size=8, initial_length=3, food=2), seed=0).state()
    assert len(st["bodies"]) == 2 and len(st["bodies"][0]) == 3  # two snakes, length 3
    assert len(st["food"]) == 2 and st["alive"] == [True, True]
    assert len(st["directions"]) == 2


def test_snake_steps_both_agents_simultaneously() -> None:
    env = rf.Env(rf.games.Snake(grid_size=8, initial_length=3, food=1), seed=0)
    assert env.num_agents() == 2 and env.action_count() == 3
    assert env.active_agents() == [0, 1]  # simultaneous: both live agents act
    assert env.observation_space().shape == (5, 8, 8)  # net sizable from the env alone
    obs = env.observe(0)
    assert obs.shape == (5, 8, 8) and obs.dtype == np.float32
    rewards = env.step({0: 1, 1: 1})  # both move forward
    assert len(rewards) == 2


def test_step_rejects_wrong_or_missing_agents() -> None:
    # A caller-driven API must fail loudly rather than play an unintended default move.
    env = rf.Env(rf.games.Connect4(), seed=0)  # P0's turn: active == [0]
    with pytest.raises(ValueError):
        env.step({1: 0})  # P1 is not active this tick
    with pytest.raises(ValueError):
        env.step({})  # P0's action is missing
    for agent, col in [(0, 0), (1, 1), (0, 0), (1, 1), (0, 0), (1, 1), (0, 0)]:
        env.step({agent: col})
    assert env.done()
    with pytest.raises(ValueError):
        env.step({0: 0})  # episode over


def test_reset_restarts_the_episode() -> None:
    env = rf.Env(rf.games.Connect4(), seed=0)
    env.step({0: 0})
    env.step({1: 0})
    env.reset()
    assert not env.done() and env.active_agents() == [0]
