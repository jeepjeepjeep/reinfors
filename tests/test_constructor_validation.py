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

INT_EDGES = [
    -(2**63),
    -1,
    0,
    1,
    2**16,
    2**16 + 1,
    2**31 - 1,
    2**31,
    2**63,
]  # 2**31-1 reaches Rust; 2**31 dies at pyo3 i32 conversion; 2**16 straddles the engine flat caps
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


def _zeros_q(a: int) -> Any:
    def infer(obs: np.ndarray) -> np.ndarray:
        return np.zeros((obs.shape[0], 1, a))

    return infer


def _zeros_az(a: int) -> Any:
    def infer(obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        m = obs.shape[0]
        return np.zeros((m, a), dtype=np.float32), np.zeros(m, dtype=np.float32)

    return infer


def _collect(game: Any, policy: Any, learner: Any, infer: Any, execute: bool = True) -> None:
    engine = rf.Engine(game, rf.Reward(), policy, learner, n_games=2, seed=0)
    if execute:
        engine.collect(1, infer)


def _env_use(env: Any) -> None:
    env.reset()
    for a in env.active_agents():
        env.observe(a)


# composition hooks: handles defer some validation to attachment/use, so a successful
# construction must also survive an engine build; `execute` additionally runs one collect
USE: dict[tuple[str, str], Any] = {
    ("policies", "mcts"): lambda p, ex: _collect(rf.games.Connect4(), p, rf.learners.TreeStrap(), _zeros_q(7), ex),
    ("policies", "alphazero"): lambda p, ex: _collect(
        rf.games.Connect4(), p, rf.learners.AlphaZero(), _zeros_az(7), ex
    ),
    ("policies", "minimax"): lambda p, ex: _collect(rf.games.Connect4(), p, rf.learners.TreeStrap(), _zeros_q(7), ex),
    ("policies", "ppo"): lambda p, ex: _collect(rf.games.Connect4(), p, rf.learners.Ppo(), _zeros_az(7), ex),
    ("policies", "selective_expectimax"): lambda p, ex: _collect(
        rf.games.Snake(), p, rf.learners.TreeStrap(), _zeros_q(3), ex
    ),
    ("policies", "epsilon_greedy_q"): lambda p, ex: _collect(
        rf.games.GridWorld(), p, rf.learners.Dqn(), _zeros_q(4), ex
    ),
    ("learners", "treestrap"): lambda le, ex: _collect(
        rf.games.Connect4(), rf.policies.Mcts(num_simulations=2), le, _zeros_q(7), ex
    ),
    ("learners", "dqn"): lambda le, ex: _collect(
        rf.games.GridWorld(), rf.policies.EpsilonGreedyQ(), le, _zeros_q(4), ex
    ),
    ("learners", "ppo"): lambda le, ex: _collect(rf.games.Connect4(), rf.policies.Ppo(), le, _zeros_az(7), ex),
    ("learners", "alphazero"): lambda le, ex: _collect(
        rf.games.Connect4(), rf.policies.AlphaZero(num_simulations=2), le, _zeros_az(7), ex
    ),
    ("chance_modes", "always_resample"): lambda m, ex: _collect(
        rf.games.Backgammon(),
        rf.policies.AlphaZero(num_simulations=2, chance=m),
        rf.learners.AlphaZero(),
        _zeros_az(1352),
        ex,
    ),
    ("chance_modes", "committed"): lambda m, ex: _collect(
        rf.games.Backgammon(),
        rf.policies.AlphaZero(num_simulations=2, chance=m),
        rf.learners.AlphaZero(),
        _zeros_az(1352),
        ex,
    ),
    ("chance_modes", "expand_all"): lambda m, ex: _collect(
        rf.games.Backgammon(),
        rf.policies.AlphaZero(num_simulations=2, chance=m),
        rf.learners.AlphaZero(),
        _zeros_az(1352),
        ex,
    ),
    ("noise", "dirichlet"): lambda n, ex: _collect(
        rf.games.Connect4(),
        rf.policies.AlphaZero(num_simulations=2, noise=n),
        rf.learners.AlphaZero(),
        _zeros_az(7),
        ex,
    ),
}
USE_REQUIRED = {"policies", "learners", "chance_modes", "noise"}

# accepted-but-huge budgets make the composed collect take minutes, not expose new panic
# surface, so those cases compose the engine but skip the collect
SLOW_AT_SCALE = {"num_simulations", "expansion_budget", "depth", "max_depth", "samples", "top_k"}


def _use_for(mod: Any, name: str, is_game: bool) -> Any:
    if is_game:
        return lambda g, ex: _play(g)
    tag = mod.__name__.rsplit(".", 1)[-1]
    if tag in USE_REQUIRED:
        assert (tag, name) in USE, f"{tag}.{name} has no composition hook: enroll it in USE"
        return USE[(tag, name)]
    return None


def _ctors() -> list[tuple[Any, Any, bool]]:
    out = [
        (mod._REGISTRY[name], _use_for(mod, name, is_game), is_game)
        for mod, is_game in MODULES
        for name in mod.registered()
    ]
    out += [
        (rf.solvers.Cfr, lambda s, ex: s.iterate(1), False),
        (rf.solvers.DeepCfr, None, False),
        (rf.Env, lambda e, ex: _env_use(e), False),
        (rf.Engine, None, False),
        (rf.starts.RandomStartingMoves, None, False),
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


def _cases() -> list[tuple[Any, Any, str, Any]]:
    out = [
        (ctor, use, p.name, value)
        for ctor, use, _is_game in _ctors()
        for p in inspect.signature(ctor).parameters.values()
        if p.kind is not inspect.Parameter.VAR_KEYWORD
        for value in _bank(getattr(ctor, "__name__", str(ctor)), p)
    ]
    out += [(rf.Reward, None, "win", v) for v in FLOAT_EDGES]  # Reward is **weights; no signature to derive
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
    ("ctor", "use", "param", "value"),
    _cases(),
    ids=lambda v: getattr(v, "__name__", repr(v)[:40]),
)
def test_no_public_input_reaches_a_rust_panic(ctor: Any, use: Any, param: str, value: Any) -> None:
    kwargs = {k: make() for k, make in REQUIRED.get(getattr(ctor, "__name__", ""), {}).items()}
    kwargs[param] = value
    try:
        made = ctor(**kwargs)
    except BaseException as exc:  # PanicException derives from BaseException by design
        assert type(exc).__name__ != "PanicException", f"Rust panic escaped {ctor}({param}={value!r}): {exc}"
        return
    if use is None:
        return
    execute = not (param in SLOW_AT_SCALE and isinstance(value, int) and value > 4096)
    try:
        # construction passing is not enough: latent panics fire at compose/encode/step time
        use(made, execute)
    except BaseException as exc:
        assert type(exc).__name__ != "PanicException", f"Rust panic escaped using {ctor}({param}={value!r}): {exc}"


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


def test_deferred_n_heads_rejects_at_composition_not_collect() -> None:
    # handles store n_heads unchecked; the engine build must reject it before the
    # callback layer multiplies rows x heads x actions
    policy = rf.policies.EpsilonGreedyQ(n_heads=2**63)
    with pytest.raises(ValueError, match="n_heads must be <="):
        rf.Engine(rf.games.GridWorld(), rf.Reward(), policy, rf.learners.Dqn(), n_games=1)


def _gridworld_engine(**kwargs: Any) -> rf.Engine:
    return rf.Engine(
        rf.games.GridWorld(),
        rf.Reward(),
        rf.policies.EpsilonGreedyQ(),
        rf.learners.Dqn(),
        **kwargs,
    )


def test_flat_caps_accept_boundary_and_reject_boundary_plus_one() -> None:
    _gridworld_engine(n_games=2**16)
    with pytest.raises(ValueError, match="n_games must be <="):
        _gridworld_engine(n_games=2**16 + 1)
    _gridworld_engine(n_games=1, batch_size=2**20)
    with pytest.raises(ValueError, match="batch_size must be <="):
        _gridworld_engine(n_games=1, batch_size=2**20 + 1)


def test_buffer_ceiling_is_dimension_aware() -> None:
    def chess_engine(**kwargs: Any) -> rf.Engine:
        return rf.Engine(
            rf.games.Chess(encoder=rf.encoders.AlphaZeroChess(history_length=256)),
            rf.Reward(),
            rf.policies.AlphaZero(num_simulations=2),
            rf.learners.AlphaZero(),
            **kwargs,
        )

    env = rf.Env(rf.games.Chess(encoder=rf.encoders.AlphaZeroChess(history_length=256)), rf.Reward(), seed=0)
    env.reset()
    dim = env.observe(0).size
    # the byte budget: rows x dim f32 in, rows x heads x (actions+1) f64 out, 2^29 bytes each
    limit = min((2**29 // 4) // dim, (2**29 // 8) // (4672 + 1))
    assert limit < 2**16  # the dimension ceiling must bind below the flat cap here
    chess_engine(n_games=2, pad=True, batch_size=limit)
    with pytest.raises(ValueError, match="too large for this composition"):
        chess_engine(n_games=2, pad=True, batch_size=limit + 1)
    with pytest.raises(ValueError, match="too large for this composition"):
        chess_engine(n_games=limit + 1)


def test_buffer_ceiling_counts_simultaneous_agents() -> None:
    # simultaneous games stage one row per ACTIVE AGENT: 8-player snake multiplies the
    # per-call observation buffer x8 over the naive n_games x dim accounting
    def snake8_engine(**kwargs: Any) -> rf.Engine:
        return rf.Engine(
            rf.games.Snake(num_snakes=8),
            rf.Reward(),
            rf.policies.SelectiveExpectimax(),
            rf.learners.TreeStrap(),
            **kwargs,
        )

    env = rf.Env(rf.games.Snake(num_snakes=8), rf.Reward(), seed=0)
    env.reset()
    dim = env.observe(0).size
    naive_rows = (2**29 // 4) // dim
    assert 2**16 <= naive_rows  # the reviewer scenario: passes agent-blind accounting...
    with pytest.raises(ValueError, match="8 simultaneous agents"):
        snake8_engine(n_games=2**16)  # ...but stages ~4 GiB once agents are counted
    limit = naive_rows // 8
    snake8_engine(n_games=limit)
    with pytest.raises(ValueError, match="too large for this composition"):
        snake8_engine(n_games=limit + 1)


def test_search_budget_caps_boundary() -> None:
    def compose(policy: Any) -> None:
        rf.Engine(rf.games.Connect4(), rf.Reward(), policy, rf.learners.TreeStrap(), n_games=1)

    compose(rf.policies.Mcts(num_simulations=2**20))
    with pytest.raises(ValueError, match="num_simulations must be <="):
        compose(rf.policies.Mcts(num_simulations=2**20 + 1))
    with pytest.raises(ValueError, match="chance samples must be <="):
        compose(rf.policies.Mcts(chance=rf.chance_modes.Committed(samples=2**20 + 1)))
    rf.Engine(rf.games.GridWorld(), rf.Reward(), rf.policies.EpsilonGreedyQ(n_heads=4096), rf.learners.Dqn(), n_games=1)
    with pytest.raises(ValueError, match="n_heads must be <="):
        rf.Engine(
            rf.games.GridWorld(), rf.Reward(), rf.policies.EpsilonGreedyQ(n_heads=4097), rf.learners.Dqn(), n_games=1
        )


def test_choose_validates_like_engine_composition() -> None:
    # PolicyHandle.choose is its own composition point; unchecked handle params must be
    # rejected there too, not panic in the callback layer
    policy = rf.policies.EpsilonGreedyQ(n_heads=2**63)
    env = rf.Env(rf.games.GridWorld(), rf.Reward(), seed=0)
    env.reset()
    with pytest.raises(ValueError, match="n_heads must be <="):
        policy.choose([env], lambda obs: np.zeros((obs.shape[0], 1, 4)), gamma=0.99)


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
