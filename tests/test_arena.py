"""rf.Arena: paired parallel evaluation over Envs."""

import threading
import time

import numpy as np
import pytest
import reinfors as rf

_A = rf.games.Connect4().action_space().n
_R = rf.Reward(win=1.0, loss=-1.0)


def _az(sims=6):
    return rf.policies.AlphaZero(num_simulations=sims, temperature=0.0, noise=None)


def _flat_infer(obs, n=None):
    m = obs.shape[0]
    return np.zeros((m, _A), dtype=np.float32), np.zeros(m, dtype=np.float32)


def _center_infer(obs, n=None):
    # prefers the center column: a strictly stronger connect4 prior than uniform
    m = obs.shape[0]
    logits = np.zeros((m, _A), dtype=np.float32)
    logits[:, 3] = 2.0
    return logits, np.zeros(m, dtype=np.float32)


def _arena(contestants, n_slots=8, start=None, seed=0, batch_delay=0.0):
    return rf.Arena(
        rf.games.Connect4(),
        _R,
        contestants,
        n_slots=n_slots,
        start=start,
        seed=seed,
        batch_delay=batch_delay,
    )


class FirstLegalBot:
    """Deterministic scripted external agent; records its game's action stream."""

    instances = 0
    live = 0
    peak = 0
    lock = threading.Lock()

    def __init__(self):
        cls = FirstLegalBot
        with cls.lock:
            cls.instances += 1
            cls.live += 1
            cls.peak = max(cls.peak, cls.live)
        self.seen: list[int] = []
        self.closed = False

    def act(self, view) -> int:
        return view.legal_actions[0]

    def on_action(self, action) -> None:
        self.seen.append(action)

    def close(self) -> None:
        self.closed = True
        with FirstLegalBot.lock:
            FirstLegalBot.live -= 1

    @classmethod
    def reset_counters(cls):
        cls.instances = cls.live = cls.peak = 0


def test_searched_vs_searched_plays_and_pairs() -> None:
    result = _arena(
        [(_az(), _flat_infer, 1.0), (_az(), _center_infer, 1.0)],
        start=rf.starts.RandomStartingMoves(2),
    ).play(8)
    assert len(result.games) == 8
    for k in range(4):
        a, b = result.games[2 * k], result.games[2 * k + 1]
        assert a.opening_id == b.opening_id == k
        assert a.seats == (0, 1) and b.seats == (1, 0)
    for g in result.games:
        assert g.length > 2 and len(g.actions) == g.length - 2
        assert abs(sum(g.payoffs)) < 1e-9  # zero-sum reward


def test_searched_only_runs_are_reproducible() -> None:
    def run():
        return _arena(
            [(_az(), _flat_infer, 1.0), (_az(), _center_infer, 1.0)],
            start=rf.starts.RandomStartingMoves(2),
            seed=42,
        ).play(6)

    a, b = run(), run()
    assert [g.actions for g in a.games] == [g.actions for g in b.games]
    assert [g.payoffs for g in a.games] == [g.payoffs for g in b.games]


def test_stronger_prior_wins_the_match() -> None:
    result = _arena(
        [(_az(sims=8), _center_infer, 1.0), (_az(sims=8), _flat_infer, 1.0)],
        start=rf.starts.RandomStartingMoves(2),
        seed=7,
    ).play(20)
    mean, stderr = result.payoff(0)
    assert mean > 0.0, f"center prior should beat flat: {mean} +- {stderr}"
    seat_split = result.seat_payoffs(0)
    assert set(seat_split) == {0, 1}


def test_external_seat_plays_and_observes_every_action() -> None:
    FirstLegalBot.reset_counters()
    bots: list[FirstLegalBot] = []

    def factory():
        bot = FirstLegalBot()
        bots.append(bot)
        return bot

    result = _arena(
        [(_az(), _flat_infer, 1.0), rf.arena.External(factory, workers=2)],
    ).play(4)
    assert len(result.games) == 4
    assert FirstLegalBot.instances == 4
    assert all(b.closed for b in bots)
    total_actions = sum(len(g.actions) for g in result.games)
    assert sum(len(b.seen) for b in bots) == total_actions


def test_external_lease_bounds_live_agents() -> None:
    FirstLegalBot.reset_counters()
    _arena(
        [(_az(), _flat_infer, 1.0), rf.arena.External(FirstLegalBot, workers=2)],
        n_slots=8,
    ).play(8)
    assert FirstLegalBot.peak <= 2, f"peak live agents {FirstLegalBot.peak} > workers"


def test_external_exception_surfaces_with_context() -> None:
    class Broken:
        def act(self, view):
            raise RuntimeError("engine crashed")

    with pytest.raises(RuntimeError, match="external contestant 1 failed"):
        _arena([(_az(), _flat_infer, 1.0), rf.arena.External(Broken)]).play(2)


def test_external_illegal_action_rejected() -> None:
    class Cheater:
        def act(self, view):
            return 999

    with pytest.raises(RuntimeError, match="illegal action"):
        _arena([(_az(), _flat_infer, 1.0), rf.arena.External(Cheater)]).play(2)


def test_hung_external_times_out() -> None:
    class Hang:
        def act(self, view):
            time.sleep(60)

    start = time.monotonic()
    with pytest.raises(TimeoutError, match="move timeout"):
        _arena(
            [(_az(), _flat_infer, 1.0), rf.arena.External(Hang, timeout=0.3)],
        ).play(2)
    assert time.monotonic() - start < 10


def test_rejects_bad_configs() -> None:
    seat = (_az(), _flat_infer, 1.0)
    with pytest.raises(ValueError, match="two contestants"):
        _arena([seat]).play(2)
    with pytest.raises(ValueError, match="even"):
        _arena([seat, seat]).play(3)
    with pytest.raises(ValueError, match="gamma"):
        rf.Arena(rf.games.Connect4(), _R, [(_az(), _flat_infer), seat])
    with pytest.raises(ValueError, match="2-agent"):
        rf.Arena(rf.games.GridWorld(), rf.Reward(goal=1.0), [seat, seat])


def test_opening_generator_resamples_and_caps() -> None:
    snap = rf.starts.RandomStartingMoves(3).generate(rf.games.Connect4(), _R, seed=5)
    env = rf.Env(rf.games.Connect4(), _R)
    env.restore(snap)
    assert env.ticks == 3 and not env.done()
    with pytest.raises(ValueError, match="could not draw"):
        rf.starts.RandomStartingMoves(60).generate(rf.games.Connect4(), _R, seed=5)
