"""Quiescent engine snapshots: the record-exact contract — after restore, continued collection
yields byte-identical records (cold cache notwithstanding), across every composition family."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import reinfors as rf


def _mk(family: str) -> tuple[rf.Engine, Any]:
    if family == "az":
        e = rf.Engine(
            rf.games.Chess(max_ticks=50, encoder=rf.encoders.OpenSpielChess()),
            None,
            rf.policies.AlphaZero(num_simulations=6),
            rf.learners.AlphaZero(gamma=1.0),
            n_games=2,
            seed=5,
            infer_cache=4096,
        )
        return e, lambda obs: (np.zeros((obs.shape[0], 4672)), np.zeros(obs.shape[0]))
    if family == "dqn":
        e = rf.Engine(
            rf.games.Backgammon(max_ticks=40),
            rf.Reward(),
            rf.policies.EpsilonGreedyQ(n_heads=2, epsilon=0.2),
            rf.learners.Dqn(bootstrap_p=0.7),
            n_games=2,
            seed=9,
        )
        return e, lambda obs: np.full((obs.shape[0], 2, 1352), 0.1)
    e = rf.Engine(
        rf.games.Snake(grid_size=6, initial_length=2, food=2, max_ticks=30),
        rf.Reward(food=1.0, loss=-5.0),
        rf.policies.SelectiveExpectimax(expansion_budget=5, n_heads=2),
        rf.learners.TreeStrap(gamma=0.9, outcome_weight=0.5),
        n_games=2,
        seed=2,
        start_buffer=True,
        start_buffer_capacity=20,
        p_fresh=0.3,
    )
    return e, lambda obs: np.full((obs.shape[0], 2, 3), 0.25)


def _sig(e: rf.Engine, infer: Any, n: int) -> dict[str, bytes]:
    b = e.collect(n, infer)
    return {
        k: np.ascontiguousarray(getattr(b, k)).tobytes() for k in dir(b) if isinstance(getattr(b, k, None), np.ndarray)
    }


@pytest.mark.parametrize("family", ["az", "dqn", "treestrap"])
def test_restore_makes_continued_collection_record_exact(family: str) -> None:
    engine, infer = _mk(family)
    _sig(engine, infer, 60)  # advance well into collection (partial trajectories buffered)
    snap = engine.snapshot()
    ahead = _sig(engine, infer, 80)
    engine.restore(snap)
    again = _sig(engine, infer, 80)
    assert ahead == again


def test_bytes_round_trip_and_cross_engine_restore() -> None:
    engine, infer = _mk("dqn")
    _sig(engine, infer, 40)
    blob = engine.snapshot(policy_version="net-v7").to_bytes()
    snap = rf._reinfors.EngineSnapshot.from_bytes(blob)
    assert snap.policy_version == "net-v7" and len(snap.fingerprint) == 64
    other, _ = _mk("dqn")  # same composition, fresh engine
    other.restore(snap, expect_policy_version="net-v7")
    assert _sig(other, infer, 50) == _sig(engine, infer, 50)


def test_restore_rejections_leave_the_engine_intact() -> None:
    engine, infer = _mk("az")
    snap = engine.snapshot()
    other, _ = _mk("dqn")
    with pytest.raises(ValueError, match="different composition"):
        other.restore(snap)
    with pytest.raises(ValueError, match="policy_version mismatch"):
        engine.restore(snap, expect_policy_version="net-v1")
    blob = bytearray(snap.to_bytes())
    blob[-5] ^= 0xFF  # corrupt inside the payload
    bad = rf._reinfors.EngineSnapshot.from_bytes(bytes(blob))
    with pytest.raises(ValueError, match="invalid engine snapshot"):
        engine.restore(bad)
    _sig(engine, infer, 30)  # engine unharmed after every rejection


def test_weights_generation_travels_with_the_snapshot() -> None:
    engine, _infer = _mk("az")
    engine.weights_updated()
    engine.weights_updated()
    snap = engine.snapshot()
    assert snap.weights_generation == 2
    other, _ = _mk("az")
    other.restore(snap)
    assert other.snapshot().weights_generation == 2
