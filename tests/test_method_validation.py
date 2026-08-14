"""No public method input reaches a Rust panic: the constructor sweep's contract extended to
the non-constructor surface — raw string/int parsers, untrusted snapshot bytes, env methods
with hostile agents/actions, and the python-layer adapters."""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import pytest
import reinfors as rf

VALID_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
HOSTILE_STRINGS = ["", "\x00", "garbage", "a" * 4096]
HOSTILE_INTS = [-1, 0, 4671, 4672, 2**31, 2**63]


def _no_panic(fn: Any) -> BaseException | None:
    try:
        fn()
    except BaseException as exc:
        assert type(exc).__name__ != "PanicException", exc
        return exc
    return None


@pytest.mark.parametrize("fen", [*HOSTILE_STRINGS, "8/8/8/8/8/8/8/8 w - - 0 1"])
def test_chess_helpers_reject_hostile_fens(fen: str) -> None:
    with pytest.raises(ValueError, match="invalid FEN"):
        rf.chess_uci_action("e2e4", fen)
    with pytest.raises(ValueError, match="invalid FEN"):
        rf.chess_action_uci(0, fen)


@pytest.mark.parametrize("uci", [*HOSTILE_STRINGS, "zz", "e9e9", "e2e4e6"])
def test_chess_uci_action_rejects_hostile_moves(uci: str) -> None:
    with pytest.raises(ValueError, match="not a legal move"):
        rf.chess_uci_action(uci, VALID_FEN)


@pytest.mark.parametrize("action", HOSTILE_INTS)
def test_chess_action_uci_rejects_hostile_actions(action: int) -> None:
    exc = _no_panic(lambda: rf.chess_action_uci(action, VALID_FEN))
    assert isinstance(exc, (ValueError, OverflowError))


@pytest.mark.parametrize("action", HOSTILE_INTS)
def test_encoder_index_maps_reject_hostile_ints(action: int) -> None:
    enc = rf.encoders.Connect4()
    for fn in (enc.head_index, enc.game_action):
        exc = _no_panic(lambda f=fn: f(action, 0))
        if action in (0, 4671):
            continue  # in range for some args; no error required
        assert isinstance(exc, (ValueError, OverflowError))


def test_env_methods_reject_hostile_agents_and_actions() -> None:
    env = rf.Env(rf.games.Connect4(), rf.Reward(), seed=0)
    env.reset()
    with pytest.raises(ValueError, match="out of range"):
        env.observe(99)
    with pytest.raises(OverflowError):
        env.legal_actions(-1)
    with pytest.raises(ValueError, match="illegal for agent"):
        env.step({0: 2**31})
    with pytest.raises(ValueError, match="not active"):
        env.step({99: 0})
    with pytest.raises(ValueError):
        env.information_state_key(99)
    with pytest.raises(OverflowError):
        env.fork(seed=-1)


@pytest.mark.parametrize(
    ("cls", "label"),
    [(rf.EnvSnapshot, "EnvSnapshot"), (rf.EngineSnapshot, "EngineSnapshot")],
)
def test_snapshot_from_bytes_rejects_garbage(cls: Any, label: str) -> None:
    with pytest.raises(ValueError, match=f"invalid {label}"):
        cls.from_bytes(b"")
    with pytest.raises(ValueError, match=f"invalid {label}"):
        cls.from_bytes(b"\x01\xff" * 200)


def _stepped_env(name: str) -> rf.Env:
    env = rf.Env(rf.games.make(name), rf.Reward(), seed=0)
    env.reset()
    rng = np.random.default_rng(1)
    for _ in range(6):
        if env.done():
            break
        env.step({a: int(rng.choice(env.legal_actions(a))) for a in env.active_agents()})
    return env


@pytest.mark.parametrize("name", rf.games.registered())
def test_env_snapshot_survives_bytewise_corruption(name: str) -> None:
    env = _stepped_env(name)
    raw = env.snapshot().to_bytes()
    positions = range(len(raw)) if len(raw) <= 400 else sorted(random.Random(2).sample(range(len(raw)), 400))
    for i in positions:
        corrupt = bytearray(raw)
        corrupt[i] ^= 0xFF

        def attempt(bs: bytes = bytes(corrupt)) -> None:
            snap = rf.EnvSnapshot.from_bytes(bs)
            fresh = rf.Env(rf.games.make(name), rf.Reward(), seed=0)
            fresh.restore(snap)
            # a corrupt-but-decodable state must also survive use, not just decode
            if not fresh.done():
                fresh.observe(fresh.active_agents()[0])

        _no_panic(attempt)


def test_engine_snapshot_survives_bytewise_corruption() -> None:
    def engine() -> rf.Engine:
        return rf.Engine(
            rf.games.Connect4(),
            rf.Reward(),
            rf.policies.Mcts(num_simulations=2),
            rf.learners.TreeStrap(),
            n_games=2,
            seed=0,
        )

    source = engine()
    source.collect(2, lambda obs: np.zeros((obs.shape[0], 1, 7)))
    raw = source.snapshot().to_bytes()
    for i in sorted(random.Random(0).sample(range(len(raw)), min(400, len(raw)))):
        corrupt = bytearray(raw)
        corrupt[i] ^= 0xFF

        def attempt(bs: bytes = bytes(corrupt)) -> None:
            engine().restore(rf.EngineSnapshot.from_bytes(bs))

        _no_panic(attempt)


def test_restore_rejects_a_different_composition() -> None:
    c4 = rf.Env(rf.games.Connect4(), rf.Reward(), seed=0)
    c4.reset()
    snake = rf.Env(rf.games.Snake(), rf.Reward(), seed=0)
    snake.reset()
    with pytest.raises(ValueError, match="different composition"):
        c4.restore(snake.snapshot())


def test_arena_validates_its_knobs() -> None:
    from reinfors.arena import External

    game, reward = rf.games.Connect4(), rf.Reward()
    two = [(rf.policies.Mcts(num_simulations=2), lambda o: np.zeros((o.shape[0], 1, 7)), 1.0)] * 2
    with pytest.raises(ValueError, match="exactly two contestants"):
        rf.Arena(game, reward, [])
    with pytest.raises(ValueError, match="n_slots"):
        rf.Arena(game, reward, two, n_slots=0)
    with pytest.raises(ValueError, match="batch_delay"):
        rf.Arena(game, reward, two, batch_delay=float("nan"))
    with pytest.raises(ValueError, match="batch_delay"):
        rf.Arena(game, reward, two, batch_delay=-1.0)
    with pytest.raises(ValueError):
        rf.Arena(game, reward, ["not-a-contestant", "also-not"])
    with pytest.raises(ValueError, match="workers"):
        External(lambda: None, workers=0)
    with pytest.raises(ValueError, match="timeout"):
        External(lambda: None, timeout=-1.0)


def test_gym_rejects_nonpositive_or_noninteger_episode_limits() -> None:
    pytest.importorskip("gymnasium")
    import reinfors.gym as rfg

    for bad in (0, -1, 1.5, float("nan"), float("inf"), True):
        with pytest.raises(ValueError, match="max_episode_steps"):
            rfg.make(rf.games.GridWorld(), max_episode_steps=bad)


_CONFIG_BASE: dict[str, Any] = {
    "game": {"name": "connect4"},
    "policy": {"name": "mcts", "num_simulations": 2},
    "learner": {"name": "treestrap"},
    "engine": {"n_games": 1},
}


def _config(**patch: Any) -> dict[str, Any]:
    import copy

    cfg = copy.deepcopy(_CONFIG_BASE)
    cfg.update(patch)
    return cfg


_HOSTILE_CONFIGS: list[Any] = [
    "garbage",
    {},
    {k: v for k, v in _CONFIG_BASE.items() if k != "game"},
    {k: v for k, v in _CONFIG_BASE.items() if k != "policy"},
    {k: v for k, v in _CONFIG_BASE.items() if k != "learner"},
    _config(game={}),
    _config(game="connect4"),
    _config(game={"name": "zork"}),
    _config(game={"name": "connect4", "zzz": 1}),
    _config(game={"name": "connect4", "encoder": {}}),
    _config(game={"name": "connect4", "encoder": {"name": "zork"}}),
    _config(policy={"name": "mcts", "zzz": 1}),
    _config(policy={"name": "mcts", "num_simulations": "x"}),
    _config(policy={"name": "mcts", "chance": {"name": "committed", "samples": "x"}}),
    _config(engine={"n_games": 1, "start_buffer": {"zzz": 1}}),
    _config(engine={"n_games": 1, "zzz": 1}),
    _config(reward=[1, 2]),
    _config(reward={"zzz": 1.0}),
    _config(game={"name": "connect4", "reward": False}),
    _config(game={"name": "connect4", "reward": []}),
    _config(game={"name": "connect4", "reward": 0}),
    _config(learner={"name": "treestrap", "gamma": float("nan")}),
    _config(schema_version=2),
    _config(schema_version=True),
    _config(schema_version=1.0),
]


@pytest.mark.parametrize("cfg", _HOSTILE_CONFIGS, ids=lambda c: repr(c)[:60])
def test_engine_from_config_rejects_hostile_dicts(cfg: Any) -> None:
    exc = _no_panic(lambda: rf.engine_from_config(cfg))
    assert isinstance(exc, (ValueError, KeyError, TypeError, AttributeError, OverflowError))


def test_engine_from_config_routes_through_composition_caps() -> None:
    # dicts must not smuggle values past the boundary validation the typed path enforces
    with pytest.raises(ValueError, match="n_games must be <="):
        rf.engine_from_config(_config(engine={"n_games": 2**31 - 1}))
    with pytest.raises(ValueError, match="num_simulations must be <="):
        rf.engine_from_config(_config(policy={"name": "mcts", "num_simulations": 2**31 - 1}))
    with pytest.raises(ValueError, match="n_heads must be <="):
        rf.engine_from_config(
            _config(
                game={"name": "gridworld"},
                policy={"name": "epsilon_greedy_q", "n_heads": 2**63},
                learner={"name": "dqn"},
            )
        )


def test_solver_methods_reject_hostile_inputs() -> None:
    with pytest.raises(ValueError, match="unknown CFR variant"):
        rf.solvers.Cfr(rf.games.KuhnPoker(), variant="bogus")
    with pytest.raises(ValueError, match="not compatible with CFR"):
        rf.solvers.Cfr(rf.games.Chess())
    with pytest.raises(OverflowError):
        rf.solvers.Cfr(rf.games.KuhnPoker()).iterate(-1)
    deep = rf.solvers.DeepCfr(rf.games.KuhnPoker())
    deep.next_iteration()
    exc = _no_panic(lambda: deep.collect(0, 1, lambda obs: "garbage"))
    assert exc is not None
