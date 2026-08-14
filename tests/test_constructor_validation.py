"""No public Python input reaches a Rust panic: constructors validate at the boundary and raise
ordinary Python exceptions (ValueError, or OverflowError/TypeError from the conversion layer) —
never pyo3's PanicException. Constructions that succeed must also survive actual *use* (a short
episode), because latent config bugs panic at encode/step time, far from the constructor."""

from __future__ import annotations

import inspect
from typing import Any

import numpy as np
import pytest
import reinfors as rf

INT_EDGES = [-(2**63), -1, 0, 1, 2**31 - 1, 2**31, 2**63]  # 2**31-1 reaches Rust; 2**31 dies at pyo3 i32 conversion
FLOAT_EDGES = [float("nan"), float("inf"), float("-inf"), -1e308, -1.0, 0.0]
STR_EDGES = ["", "\x00", "not-a-registered-option", "a" * 4096]
WRONG_TYPES: list[Any] = ["wrong-type", 3.5]

# banks for params whose default (None/Ellipsis) hides the real type
NONE_BANKS: dict[str, list[Any]] = {
    "win_food_lead": INT_EDGES,
    "goal_row": INT_EDGES,
    "goal_col": INT_EDGES,
    "temperature_drop": INT_EDGES,
    "top_k": INT_EDGES,
    "learn_players": [[-1], [0, 2**31], "wrong-type"],
    "encoder": WRONG_TYPES,
    "chance": WRONG_TYPES,
    "noise": WRONG_TYPES,
    "reward": WRONG_TYPES,
}

REQUIRED: dict[str, dict[str, Any]] = {
    "Cfr": {"game": rf.games.KuhnPoker},
    "DeepCfr": {"game": rf.games.KuhnPoker},
    "Env": {"game": rf.games.Connect4},
    "Engine": {
        "game": rf.games.Connect4,
        "reward": rf.Reward,
        "policy": lambda: rf.policies.Mcts(num_simulations=2),
        "learner": rf.learners.TreeStrap,
        "n_games": lambda: 1,
    },
    "RandomStartingMoves": {"n_moves": lambda: 1},
}

MODULES: list[tuple[Any, bool]] = [
    (rf.games, True),
    (rf.encoders, False),
    (rf.policies, False),
    (rf.learners, False),
    (rf.chance_modes, False),
    (rf.noise, False),
]


def _ctors() -> list[tuple[Any, bool]]:
    out = [(mod._REGISTRY[name], is_game) for mod, is_game in MODULES for name in mod.registered()]
    out += [
        (rf.solvers.Cfr, False),
        (rf.solvers.DeepCfr, False),
        (rf.Env, False),
        (rf.Engine, False),
        (rf.starts.RandomStartingMoves, False),
    ]
    return out


def _bank(ctor_name: str, p: inspect.Parameter) -> list[Any]:
    if p.name in NONE_BANKS:
        return NONE_BANKS[p.name]
    default = p.default
    if default is inspect.Parameter.empty:
        default = REQUIRED[ctor_name][p.name]()
    if isinstance(default, bool):
        return [not default]
    if isinstance(default, int):
        return INT_EDGES
    if isinstance(default, float):
        return FLOAT_EDGES
    if isinstance(default, str):
        return STR_EDGES
    if p.name in REQUIRED.get(ctor_name, {}):
        return WRONG_TYPES
    raise AssertionError(f"{ctor_name}.{p.name}: unrecognized default — enroll it in NONE_BANKS or REQUIRED")


def _cases() -> list[tuple[Any, bool, str, Any]]:
    out = [
        (ctor, is_game, p.name, value)
        for ctor, is_game in _ctors()
        for p in inspect.signature(ctor).parameters.values()
        if p.kind is not inspect.Parameter.VAR_KEYWORD
        for value in _bank(getattr(ctor, "__name__", str(ctor)), p)
    ]
    out += [(rf.Reward, False, "win", v) for v in FLOAT_EDGES]  # Reward is **weights; no signature to derive
    return out


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
    _cases(),
    ids=lambda v: getattr(v, "__name__", repr(v)[:40]),
)
def test_no_public_input_reaches_a_rust_panic(ctor: Any, is_game: bool, param: str, value: Any) -> None:
    kwargs = {k: make() for k, make in REQUIRED.get(getattr(ctor, "__name__", ""), {}).items()}
    kwargs[param] = value
    try:
        made = ctor(**kwargs)
    except BaseException as exc:  # PanicException derives from BaseException by design
        assert type(exc).__name__ != "PanicException", f"Rust panic escaped {ctor}({param}={value!r}): {exc}"
        return
    if is_game:
        _play(made)  # construction passing is not enough: latent panics fire at encode/step time


def test_every_registered_handle_constructs_with_defaults() -> None:
    for mod, is_game in MODULES:
        for name in mod.registered():
            made = mod.make(name)
            if is_game:
                _play(made)


@pytest.mark.parametrize("game_name", rf.games.registered())
@pytest.mark.parametrize("encoder_name", rf.encoders.registered())
def test_every_encoder_attaches_or_refuses_cleanly(game_name: str, encoder_name: str) -> None:
    try:
        game = rf.games.make(game_name, encoder=rf.encoders.make(encoder_name))
    except BaseException as exc:
        assert type(exc).__name__ != "PanicException", f"Rust panic escaped {game_name}+{encoder_name}: {exc}"
        return
    _play(game)


def test_unknown_reward_weight_rejected_at_attach() -> None:
    # Reward is game-agnostic: keys validate against the game's schema when attached
    with pytest.raises(ValueError, match="unknown reward key"):
        rf.Env(rf.games.Connect4(), rf.Reward(zzz=1.0), seed=0)


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
