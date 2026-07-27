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


def test_done_gates_active_agents_and_stepping() -> None:
    # snake, play_to_last=False: one survivor ends the game with no internal state flag — done
    # must gate active_agents so the step guard holds (incl. for restored terminal snapshots).
    env = rf.Env(rf.games.Snake(grid_size=6, initial_length=2, food=1, play_to_last=False), seed=3)
    env.reset()
    for _ in range(40):
        if env.done():
            break
        env.step(dict.fromkeys(env.active_agents(), 0))  # everyone forward: someone hits a wall
    assert env.done()
    assert env.active_agents() == [] and env.legal_actions(0) == []
    with pytest.raises(ValueError, match="episode is over"):
        env.step({0: 0})
    clone = env.fork()
    assert clone.done() and clone.active_agents() == []
    with pytest.raises(ValueError, match="episode is over"):
        clone.step({0: 0})


def test_done_mismatch_between_envelope_and_state_is_rejected() -> None:
    env = rf.Env(rf.games.GridWorld(size=5), seed=0)
    env.reset()
    blob = bytearray(env.snapshot().to_bytes())
    assert blob[-1] == 0  # gridworld state layout ends with its done byte
    blob[-1] = 1  # state says done, envelope still says live
    snap = rf.EnvSnapshot.from_bytes(bytes(blob))
    with pytest.raises(ValueError, match="disagrees"):
        env.restore(snap)


def _snap_with_state(env: rf.Env, state: bytes) -> rf.EnvSnapshot:
    """Rebuild a snapshot envelope around a forged state payload (fixed-width header: magic 4 +
    schema 1 + fp-len 4 + fp 64 + rng 8 + done 1)."""
    head = bytes(env.snapshot().to_bytes()[: 4 + 1 + 4 + 64 + 8 + 1])
    return rf.EnvSnapshot.from_bytes(head + len(state).to_bytes(4, "little") + state)


def test_deep_codec_invariants_reject_forged_states() -> None:
    import struct

    env = rf.Env(rf.games.Snake(grid_size=6, initial_length=2, food=1), seed=0)
    env.reset()

    def snake_state(bodies: list[list[tuple[int, int]]], food: list[tuple[int, int]]) -> bytes:
        out = bytearray([1])
        for body in bodies:
            out += struct.pack("<I", len(body))
            for r, c in body:
                out += struct.pack("<ii", r, c)
            out += bytes([0, 1])  # direction Up, alive
        out += struct.pack("<I", len(food))
        for r, c in food:
            out += struct.pack("<ii", r, c)
        return bytes(out)

    cases = {
        "revisits": snake_state([[(1, 1), (1, 2), (1, 1)], [(3, 3)]], []),
        "overlap": snake_state([[(1, 1)], [(1, 1)]], []),
        "duplicate food": snake_state([[(1, 1)], [(3, 3)]], [(2, 2), (2, 2)]),
        "overlaps a snake": snake_state([[(1, 1)], [(3, 3)]], [(1, 1)]),
    }
    for msg, forged in cases.items():
        with pytest.raises(ValueError, match=msg.split()[0]):
            env.restore(_snap_with_state(env, forged))

    c4 = rf.Env(rf.games.Connect4(), seed=0)
    c4.reset()
    board = bytearray([1] + [0] * 42 + [0, 0])
    board[1] = 1
    board[2] = 1  # two P0 pieces, none for P1, turn 0: alternation violated
    with pytest.raises(ValueError, match="inconsistent with turn"):
        c4.restore(_snap_with_state(c4, bytes(board)))
    bool_bad = bytearray([1] + [0] * 42 + [0, 2])  # done byte = 2
    with pytest.raises(ValueError, match="not a bool"):
        c4.restore(_snap_with_state(c4, bytes(bool_bad)))


def test_env_snapshot_is_public_api() -> None:
    assert rf.EnvSnapshot is rf._reinfors.EnvSnapshot
    env = rf.Env(rf.games.Connect4(), seed=1)
    env.reset()
    assert isinstance(rf.EnvSnapshot.from_bytes(env.snapshot().to_bytes()), rf.EnvSnapshot)
