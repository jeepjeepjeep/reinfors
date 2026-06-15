"""Differential parity test: reinfors' snake core vs snake_RL's CleanSnakeEnv (the oracle).

Capture-replay (Option B): drive the Python env with an action policy, capture the actions it took
and the food it spawned each tick, replay the identical actions + spawns into reinfors, and assert
bit-identical bodies, directions, aliveness, food, per-snake events, and the egocentric observation.

Uniform-random snakes wall-die in ~30 ticks, before eating/growth/win-lead engage, so alongside the
random rollouts we run a forward-biased policy (longer games, reaches food) and a few *directed*
scenarios that deterministically exercise the machinery that matters most: eat-along-a-row (single
and dual spawn replay + observation-with-growth), a head-on draw, and a win_food_lead win.

Temporary cross-repo harness: imports snake_RL from a sibling checkout; skipped where absent (CI).
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
from snake_rl.environment.base import ACTIONS, Action, BaseSnakeEnv  # noqa: E402
from snake_rl.environment.clean import CleanSnakeEnv  # noqa: E402

_A = BaseSnakeEnv.PLAYER_A
_B = BaseSnakeEnv.PLAYER_B
_SIDS = (_A, _B)
# snake_RL Action enum order is UP, DOWN, LEFT, RIGHT -> reinfors' 0, 1, 2, 3.
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


def _run(
    seed: int,
    n_ticks: int,
    policy,  # (sid, state, tick) -> Action | None
    *,
    grid: int = 20,
    init_len: int = 3,
    play_to_last: bool = False,
    win_food_lead: int | None = None,
    setup=None,  # (py) -> None, called after reset to override the initial state (e.g. seed food)
):
    py = CleanSnakeEnv(
        grid_size=grid, initial_length=init_len, play_to_last=play_to_last, win_food_lead=win_food_lead, seed=seed
    )
    if setup is not None:
        setup(py)
    rein = reinfors._reinfors.SnakeEnv(grid, init_len, play_to_last, win_food_lead)  # default placement matches py
    rein.set_food([tuple(c) for c in py.state.food])

    spawn_log: list = []
    original_spawn = py._spawn_cells

    def capturing_spawn(snakes, food, n):  # type: ignore[no-untyped-def]
        cells = original_spawn(snakes, food, n)
        spawn_log.extend(cells)
        return cells

    py._spawn_cells = capturing_spawn  # type: ignore[method-assign]

    obs_builder = EgocentricGridObservation(grid_size=grid)
    _assert_state(py, rein, obs_builder)

    last = None
    for tick in range(n_ticks):
        state = py.state
        chosen: dict = {}
        for sid in _SIDS:
            if state.snakes[sid].alive:
                action = policy(sid, state, tick)
                if action is not None:
                    chosen[sid] = action
                    py.submit_action(sid, action)

        spawn_log.clear()
        last = py.tick()
        rein_actions = (_ACT2I.get(chosen.get(_A)), _ACT2I.get(chosen.get(_B)))
        rein_events = rein.step(rein_actions, [tuple(c) for c in spawn_log])

        _assert_events(last.events, rein_events)
        _assert_state(py, rein, obs_builder)
        assert last.done == rein.is_done(), "done flag mismatch"
        if last.done:
            break
    return py, last


# --- random / forward-biased rollouts ---------------------------------------------------------


@pytest.mark.parametrize("seed", range(12))
def test_parity_random_rollouts(seed: int) -> None:
    rng = random.Random(seed)
    _run(seed, 300, lambda sid, state, tick: rng.choice(ACTIONS))


@pytest.mark.parametrize("seed", range(8))
def test_parity_forward_biased_rollouts(seed: int) -> None:
    # Mostly coast straight -> snakes live much longer and occasionally reach RNG-spawned food,
    # so the random harness exercises eating/growth/longer trajectories, not just quick deaths.
    rng = random.Random(seed)

    def policy(sid, state, tick):  # type: ignore[no-untyped-def]
        if rng.random() < 0.85:
            return state.snakes[sid].direction  # forward
        return rng.choice(ACTIONS)

    _run(seed, 500, policy, play_to_last=True)


# --- directed scenarios ------------------------------------------------------------------------


def _seed_food(*cells: tuple[int, int]):
    """Setup that replaces the reset's random apples with an exact set, for a deterministic scenario."""

    def setup(py: CleanSnakeEnv) -> None:
        py._food = set(cells)

    return setup


def _go(sid: str, direction: Action):
    """Policy: snake `sid` follows `direction`; the other steers up, out of the way and staying alive."""

    def policy(s, state, tick):  # type: ignore[no-untyped-def]
        return direction if s == sid else Action.UP

    return policy


def test_parity_eat_along_row_single_spawn() -> None:
    # A starts at (10,6) heading Right; seed four apples straight ahead on its row. A eats one per
    # tick (single spawn replay each time) and grows, exercising observation-with-growth. B steers up.
    py, _ = _run(0, 6, _go(_A, Action.RIGHT), setup=_seed_food((10, 7), (10, 8), (10, 9), (10, 10)))
    assert py.state.snakes[_A].length >= 3 + 4, "A should have eaten the row of apples and grown"


def test_parity_dual_spawn_same_tick() -> None:
    # Apples directly ahead of BOTH snakes -> both eat on the same tick -> two spawns replayed at once.
    def policy(sid, state, tick):  # type: ignore[no-untyped-def]
        return Action.RIGHT if sid == _A else Action.LEFT  # each moves forward onto its apple

    _, last = _run(0, 1, policy, setup=_seed_food((10, 7), (10, 13)))
    assert last is not None
    assert last.events[_A].ate_food and last.events[_B].ate_food, "both eat on the same tick (dual spawn)"


def test_parity_head_on_draw() -> None:
    # Default placement faces A and B across row 10; both coast forward and meet at (10,10) on tick 4.
    _, last = _run(0, 10, lambda sid, s, t: Action.RIGHT if sid == _A else Action.LEFT, setup=_seed_food())
    assert last is not None and last.done
    assert last.events[_A].drew and last.events[_B].drew, "head-on with no survivors is a draw"


def test_parity_win_food_lead() -> None:
    # A eats two apples (B eats none) to reach a 2-apple lead, triggering an outright win_food_lead win.
    _, last = _run(0, 4, _go(_A, Action.RIGHT), win_food_lead=2, setup=_seed_food((10, 7), (10, 8)))
    assert last is not None and last.done
    assert last.events[_A].won and last.events[_B].lost, "A should win on the food lead"
