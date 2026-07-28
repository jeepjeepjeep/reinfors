"""No public Python input reaches a Rust panic: constructors validate at the boundary and raise
ordinary Python exceptions (ValueError, or OverflowError/TypeError from the conversion layer) —
never pyo3's PanicException. Constructions that succeed must also survive actual *use* (a short
episode), because latent config bugs panic at encode/step time, far from the constructor."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import reinfors as rf

INT_EDGES = [-(2**63), -1, 0, 1, 2**31 - 1, 2**31, 2**63]  # 2**31-1 reaches Rust; 2**31 dies at pyo3 i32 conversion
FLOAT_EDGES = [float("nan"), float("inf"), float("-inf"), -1e308, -1.0, 0.0]

# (constructor, is_game, {param: edge values}) — each param is perturbed alone, others at defaults.
SWEEP: list[tuple[Any, bool, dict[str, list[Any]]]] = [
    (
        rf.games.Snake,
        True,
        {
            "grid_size": INT_EDGES,
            "initial_length": INT_EDGES,
            "food": INT_EDGES,
            "win_food_lead": INT_EDGES,
            "max_ticks": INT_EDGES,
            "num_snakes": INT_EDGES,
        },
    ),
    (
        rf.games.GridWorld,
        True,
        {"size": INT_EDGES, "goal_row": INT_EDGES, "goal_col": INT_EDGES, "max_ticks": INT_EDGES},
    ),
    (rf.games.Chess, True, {"max_ticks": INT_EDGES}),
    (rf.games.Backgammon, True, {"max_ticks": INT_EDGES}),
    (rf.Reward, False, {"win": FLOAT_EDGES}),
    (rf.chance_modes.Committed, False, {"samples": INT_EDGES}),
    (rf.noise.Dirichlet, False, {"epsilon": FLOAT_EDGES, "alpha": FLOAT_EDGES}),
    (rf.encoders.AlphaZeroChess, False, {"history_length": INT_EDGES}),
    (rf.policies.Mcts, False, {"uct_c": FLOAT_EDGES, "temperature": FLOAT_EDGES}),
    (rf.policies.AlphaZero, False, {"c_puct": FLOAT_EDGES, "temperature": FLOAT_EDGES}),
    (rf.policies.SelectiveExpectimax, False, {"beta": FLOAT_EDGES, "opp_temperature": FLOAT_EDGES}),
    (rf.learners.TreeStrap, False, {"gamma": FLOAT_EDGES, "outcome_weight": FLOAT_EDGES}),
    (rf.learners.Dqn, False, {"bootstrap_p": FLOAT_EDGES}),
]


def _play(game: Any) -> None:
    env = rf.Env(game, rf.Reward(), seed=0)
    env.reset()
    rng = np.random.default_rng(0)
    for _ in range(10):
        if env.done():
            break
        actions = {a: int(rng.choice(env.legal_actions(a))) for a in env.active_agents()}
        for a in env.active_agents():
            env.observe(a)
        env.step(actions)


@pytest.mark.parametrize(
    ("ctor", "is_game", "param", "value"),
    [
        (ctor, is_game, param, value)
        for ctor, is_game, params in SWEEP
        for param, values in params.items()
        for value in values
    ],
    ids=lambda v: getattr(v, "__name__", str(v)),
)
def test_no_public_input_reaches_a_rust_panic(ctor: Any, is_game: bool, param: str, value: Any) -> None:
    try:
        made = ctor(**{param: value})
    except BaseException as exc:  # PanicException derives from BaseException by design
        assert type(exc).__name__ != "PanicException", f"Rust panic escaped {ctor}({param}={value}): {exc}"
        return
    if is_game:
        _play(made)  # construction passing is not enough: latent panics fire at encode/step time


def test_gridworld_goal_derives_from_size() -> None:
    # The regression that motivated this file: absolute goal defaults panicked for size < 5 and sat
    # silently mid-grid for size > 5. The default is now the far corner, derived from `size`.
    for size in [4, 8]:
        env = rf.Env(rf.games.GridWorld(size=size), rf.Reward(), seed=0)
        env.reset()
        goal_plane = env.observe(0).reshape(2, size, size)[1]
        assert goal_plane[size - 1, size - 1] == 1.0 and goal_plane.sum() == 1.0


def test_gridworld_explicit_goal_is_honored_and_validated() -> None:
    env = rf.Env(rf.games.GridWorld(size=5, goal_row=2, goal_col=1), rf.Reward(), seed=0)
    env.reset()
    goal_plane = env.observe(0).reshape(2, 5, 5)[1]
    assert goal_plane[2, 1] == 1.0 and goal_plane.sum() == 1.0
    with pytest.raises(ValueError, match="outside the 4x4 grid"):
        rf.games.GridWorld(size=4, goal_row=4, goal_col=4)
    with pytest.raises(ValueError, match="size must be >= 2"):
        rf.games.GridWorld(size=1)  # 1x1 has no non-goal start cell: would hang, not panic


def test_snake_placement_is_validated_by_construction() -> None:
    with pytest.raises(ValueError, match="outside the grid"):
        rf.games.Snake(grid_size=2)  # placement puts a head out of bounds
    with pytest.raises(ValueError, match="does not fit"):
        rf.games.Snake(grid_size=8, initial_length=100)
    with pytest.raises(ValueError, match="overlap"):
        rf.games.Snake(grid_size=8, initial_length=16)  # bodies wrap the perimeter into each other
    with pytest.raises(ValueError, match="free cells"):
        rf.games.Snake(grid_size=3, food=4)
    with pytest.raises(ValueError, match="win_food_lead"):
        rf.games.Snake(win_food_lead=0)
    _play(rf.games.Snake(grid_size=3, food=3))  # smallest default-length grid really runs


def test_max_ticks_zero_rejected_everywhere() -> None:
    ctors: list[Any] = [rf.games.Snake, rf.games.GridWorld, rf.games.Chess, rf.games.Backgammon]
    for ctor in ctors:
        with pytest.raises(ValueError, match="max_ticks"):
            ctor(max_ticks=0)


def test_reward_weights_must_be_finite() -> None:
    for bad in [float("nan"), float("inf"), float("-inf")]:
        with pytest.raises(ValueError, match="finite"):
            rf.Reward(win=bad)
    rf.Reward(win=-1e308)  # large-but-finite is the user's business


def test_observation_size_ceiling() -> None:
    with pytest.raises(ValueError, match="2\\^31"):
        rf.games.GridWorld(size=40_000)
    with pytest.raises(ValueError, match="2\\^31"):
        rf.games.Snake(grid_size=46_000)
    with pytest.raises(ValueError, match="2\\^31"):
        rf.encoders.AlphaZeroChess(history_length=3_000_000)
    # Past the point where the ceiling arithmetic itself would overflow i64 (5*g^2 at g ~1.36e9):
    # the check must reject, not panic (debug) or wrap past itself (release).
    with pytest.raises(ValueError, match="2\\^31"):
        rf.games.Snake(grid_size=1_500_000_000)
    with pytest.raises(ValueError, match="2\\^31"):
        rf.games.GridWorld(size=2**31 - 1)
