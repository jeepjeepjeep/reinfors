"""Differential parity test: reinfors' snake core vs snake_RL's CleanSnakeEnv (the oracle).

Capture-replay (the agreed Option B): drive the Python env with random actions; each tick, capture
the actions it took and the food cells it spawned, then replay the identical actions + spawns into
reinfors and assert bit-identical bodies, directions, aliveness, food, per-snake events, and the
egocentric observation. Food RNG is treated as an injected input, so we never reproduce numpy's PRNG.

This is a temporary cross-repo harness: it imports snake_RL from a sibling checkout and is skipped
(via importorskip) wherever that isn't present — e.g. reinfors CI.
"""

import os
import random
import sys

import numpy as np
import pytest

_SNAKE_RL_SRC = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "snake_RL", "src"))
if os.path.isdir(_SNAKE_RL_SRC) and _SNAKE_RL_SRC not in sys.path:
    sys.path.insert(0, _SNAKE_RL_SRC)

pytest.importorskip("snake_rl.environment.clean", reason="snake_RL oracle not available")

import reinfors  # noqa: E402
from snake_rl.agent.shared.observation import EgocentricGridObservation  # noqa: E402
from snake_rl.environment.base import ACTIONS, BaseSnakeEnv  # noqa: E402
from snake_rl.environment.clean import CleanSnakeEnv  # noqa: E402

_A = BaseSnakeEnv.PLAYER_A
_B = BaseSnakeEnv.PLAYER_B
_SIDS = (_A, _B)
# snake_RL Action enum order is UP, DOWN, LEFT, RIGHT -> matches reinfors' 0,1,2,3.
_ACT2I = {a: i for i, a in enumerate(ACTIONS)}


def _assert_state(py: CleanSnakeEnv, rein, obs_builder: EgocentricGridObservation) -> None:
    state = py.state
    rein_bodies = rein.bodies()
    for idx, sid in enumerate(_SIDS):
        assert [tuple(c) for c in state.snakes[sid].body] == list(rein_bodies[idx]), f"body mismatch for {sid}"
    assert (state.snakes[_A].alive, state.snakes[_B].alive) == rein.alive(), "aliveness mismatch"
    assert (_ACT2I[state.snakes[_A].direction], _ACT2I[state.snakes[_B].direction]) == rein.directions(), "direction"
    assert {tuple(c) for c in state.food} == {tuple(c) for c in rein.food()}, "food mismatch"
    g = obs_builder.grid_size
    for idx, sid in enumerate(_SIDS):
        py_obs = np.asarray(obs_builder.build(state, sid))
        rein_obs = np.asarray(rein.obs(idx)).reshape(5, g, g)
        assert np.array_equal(py_obs, rein_obs), f"obs mismatch for {sid}"


def _assert_events(py_events: dict, rein_events: list) -> None:
    for idx, sid in enumerate(_SIDS):
        e = py_events[sid]
        cause = e.death_cause.value if e.death_cause is not None else None
        expected = (e.ate_food, e.died, cause, e.killed_opponent, e.won, e.lost, e.drew)
        assert expected == tuple(rein_events[idx]), f"event mismatch for {sid}"


def _run_episode(seed: int, n_ticks: int, *, grid=20, init_len=3, play_to_last=False, win_food_lead=None) -> None:
    py = CleanSnakeEnv(
        grid_size=grid,
        initial_length=init_len,
        play_to_last=play_to_last,
        win_food_lead=win_food_lead,
        seed=seed,
    )
    rein = reinfors._reinfors.SnakeEnv(grid, init_len, play_to_last, win_food_lead)
    rein.set_food([tuple(c) for c in py.state.food])  # match the oracle's initial (RNG-spawned) food

    spawn_log: list = []
    original_spawn = py._spawn_cells

    def capturing_spawn(snakes, food, n):  # type: ignore[no-untyped-def]
        cells = original_spawn(snakes, food, n)
        spawn_log.extend(cells)
        return cells

    py._spawn_cells = capturing_spawn  # type: ignore[method-assign]

    obs_builder = EgocentricGridObservation(grid_size=grid)
    rng = random.Random(seed)

    _assert_state(py, rein, obs_builder)  # initial state must already agree

    for _ in range(n_ticks):
        state = py.state
        chosen: dict = {}
        for sid in _SIDS:
            if state.snakes[sid].alive:
                action = rng.choice(ACTIONS)
                chosen[sid] = action
                py.submit_action(sid, action)

        spawn_log.clear()
        py_result = py.tick()
        rein_actions = (_ACT2I.get(chosen.get(_A)), _ACT2I.get(chosen.get(_B)))
        rein_events = rein.step(rein_actions, [tuple(c) for c in spawn_log])

        _assert_events(py_result.events, rein_events)
        _assert_state(py, rein, obs_builder)
        assert py_result.done == rein.is_done(), "done flag mismatch"
        if py_result.done:
            break


@pytest.mark.parametrize("seed", range(12))
def test_parity_random_rollouts(seed: int) -> None:
    _run_episode(seed, n_ticks=300)


@pytest.mark.parametrize("seed", range(4))
def test_parity_play_to_last(seed: int) -> None:
    _run_episode(seed, n_ticks=500, play_to_last=True)


@pytest.mark.parametrize("seed", range(4))
def test_parity_win_food_lead(seed: int) -> None:
    _run_episode(seed, n_ticks=500, win_food_lead=3)
