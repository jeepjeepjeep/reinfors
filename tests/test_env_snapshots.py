"""Opaque Env snapshots: exact capture/restore/fork with decode-time validation — the restoration
format is reinfors-produced bytes, never Python-built dictionaries."""

from __future__ import annotations

import numpy as np
import pytest
import reinfors as rf

GAMES = {
    "connect4": lambda: rf.games.Connect4(),
    "chess": lambda: rf.games.Chess(max_ticks=80),
    "backgammon": lambda: rf.games.Backgammon(max_ticks=60),
    "snake": lambda: rf.games.Snake(grid_size=6, initial_length=2, food=2, max_ticks=40),
    "gridworld": lambda: rf.games.GridWorld(size=5),
}


def _play(env: rf.Env, rng: np.random.Generator, plies: int) -> list[list[object]]:
    trace = []
    for _ in range(plies):
        if env.done():
            break
        acts = {a: int(rng.choice(env.legal_actions(a))) for a in env.active_agents()}
        env.step(acts)
        trace.append([env.done(), *(env.observe(a).tobytes() for a in range(env.num_agents()))])
    return trace


@pytest.mark.parametrize("name", sorted(GAMES))
def test_snapshot_restore_reproduces_the_exact_continuation(name: str) -> None:
    env = rf.Env(GAMES[name](), seed=11)
    env.reset()
    _play(env, np.random.default_rng(1), 6)  # advance into the midgame
    snap = env.snapshot()
    ahead = _play(env, np.random.default_rng(2), 8)  # same action stream after restore
    env.restore(snap)
    again = _play(env, np.random.default_rng(2), 8)
    assert ahead == again  # states, chance draws, terminality: all identical


@pytest.mark.parametrize("name", sorted(GAMES))
def test_bytes_round_trip_and_cross_env_restore(name: str) -> None:
    env = rf.Env(GAMES[name](), seed=3)
    env.reset()
    _play(env, np.random.default_rng(4), 5)
    blob = env.snapshot().to_bytes()
    snap = rf._reinfors.EnvSnapshot.from_bytes(blob)
    other = rf.Env(GAMES[name](), seed=999)  # fresh env, same composition
    other.reset()
    other.restore(snap)
    r = np.random.default_rng(5)
    assert _play(env.fork(), r, 6) == _play(other, np.random.default_rng(5), 6)


def test_fork_is_clone_exact_and_seed_diverges() -> None:
    env = rf.Env(rf.games.Backgammon(max_ticks=60), seed=7)  # dice: chance-heavy
    env.reset()
    _play(env, np.random.default_rng(0), 4)
    clone = env.fork()
    assert _play(clone, np.random.default_rng(9), 10) == _play(env.fork(), np.random.default_rng(9), 10)
    reseeded = env.fork(seed=123)
    a = _play(env.fork(), np.random.default_rng(9), 10)
    b = _play(reseeded, np.random.default_rng(9), 10)
    assert a != b  # divergent chance stream under the same action stream


def test_restore_rejects_wrong_composition_and_malformed_bytes() -> None:
    env = rf.Env(rf.games.Connect4(), seed=0)
    env.reset()
    snap = env.snapshot()
    other = rf.Env(rf.games.Connect4(), rf.Reward(win=2.0), seed=0)  # reward differs -> composition differs
    other.reset()
    with pytest.raises(ValueError, match="different composition"):
        other.restore(snap)
    chess = rf.Env(rf.games.Chess(), seed=0)
    chess.reset()
    with pytest.raises(ValueError, match="different composition"):
        chess.restore(snap)

    blob = bytearray(snap.to_bytes())
    with pytest.raises(ValueError):
        rf._reinfors.EnvSnapshot.from_bytes(bytes(blob[:-3]))  # truncated
    blob[0] ^= 0xFF
    with pytest.raises(ValueError, match="magic"):
        rf._reinfors.EnvSnapshot.from_bytes(bytes(blob))


def test_codec_validates_state_payload() -> None:
    env = rf.Env(rf.games.Backgammon(), seed=1)
    env.reset()
    blob = bytearray(env.snapshot().to_bytes())
    blob[-10] = 250  # corrupt inside the state payload: checker counts go inconsistent
    snap = rf._reinfors.EnvSnapshot.from_bytes(bytes(blob))  # envelope still parses
    with pytest.raises(ValueError, match="invalid snapshot state"):
        env.restore(snap)
    assert not env.done() and env.legal_actions(env.active_agents()[0])  # env unharmed


def test_restore_lands_at_a_step_boundary() -> None:
    env = rf.Env(rf.games.Connect4(), rf.Reward(), seed=0)
    env.reset()
    env.step({0: 3})
    assert env.rewards is not None
    snap = env.snapshot()
    env.restore(snap)
    assert env.rewards is None  # transient last-step output is not state
