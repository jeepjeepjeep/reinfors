"""Quiescent engine snapshots: the record-exact contract — after restore, continued collection
yields byte-identical records (cold cache notwithstanding), across every composition family."""

from __future__ import annotations

from pathlib import Path
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


def test_documented_crash_resume_snippet_runs() -> None:
    """Execute the guide's restore block so documentation cannot invent a second fingerprint gate."""
    source, infer = _mk("dqn")
    _sig(source, infer, 40)
    model_version = "net-v7"
    saved_config = source.resolved_config()
    snapshot_payload = source.snapshot(policy_version=model_version).to_bytes()

    guide = Path(__file__).parents[1] / "docs/guides/configuration-and-checkpoints.md"
    resume_section = guide.read_text(encoding="utf-8").split("## Resume after a crash", 1)[1]
    snippet = resume_section.split("```python\n", 1)[1].split("\n```", 1)[0]
    namespace: dict[str, Any] = {
        "model_version": model_version,
        "saved_config": saved_config,
        "snapshot_payload": snapshot_payload,
    }
    exec(snippet, namespace)

    restored = namespace["engine"]
    snapshot = namespace["snapshot"]
    assert source.config_fingerprint() != snapshot.fingerprint  # Distinct hashes by design.
    assert _sig(restored, infer, 50) == _sig(source, infer, 50)


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


def test_restore_clears_a_warm_cache_from_other_weights() -> None:
    # Two engines restore the SAME snapshot but one has a cache warmed by a DIFFERENT net at an
    # equal generation number — its stale rows must not serve (the review reproduction).
    def build() -> rf.Engine:
        return rf.Engine(
            rf.games.Connect4(),
            None,
            rf.policies.EpsilonGreedyQ(n_heads=1, epsilon=0.0),
            rf.learners.Dqn(),
            n_games=1,
            seed=0,
            infer_cache=4096,
        )

    def net(best: int) -> Any:
        def infer(obs: np.ndarray) -> np.ndarray:
            q = np.zeros((obs.shape[0], 1, 7))
            q[:, :, best] = 1.0
            return q

        return infer

    warm, cold = build(), build()
    snap = warm.snapshot()
    warm.collect(4, net(3))  # warms the cache under the action-3 net
    warm.restore(snap)
    a = warm.collect(4, net(0))  # after restore, BOTH engines see the action-0 net
    cold.restore(snap)
    b = cold.collect(4, net(0))
    warm_actions = np.asarray(a.actions)
    cold_actions = np.asarray(b.actions)
    assert np.array_equal(warm_actions, cold_actions)  # stale rows would diverge these
    assert warm_actions[0] == 0  # opening move follows the NEW net, not the cached action-3 rows
    assert np.array_equal(np.asarray(a.obs), np.asarray(b.obs))


def test_forged_payload_semantics_are_rejected() -> None:
    engine, _ = _mk("az")
    snap = engine.snapshot()
    blob = bytearray(snap.to_bytes())
    # Envelope header: magic 4 + schema 1 + fp(4+64) + gen 8 + pv(1+4+0) + payload len 4.
    payload_off = 4 + 1 + 4 + 64 + 8 + 1 + 4 + 4
    # Payload (schema v4): version 1 + n_games 4 + agents 4 + buffer rng 8 + sweep
    # cursor 8, then the per-game section; forge the FIRST game's tick to u64::MAX.
    state_len_off = payload_off + 1 + 4 + 4 + 8 + 8
    state_len = int.from_bytes(blob[state_len_off : state_len_off + 4], "little")
    rng_off = state_len_off + 4 + state_len
    tick_off = rng_off + 8
    blob[tick_off : tick_off + 8] = (2**64 - 1).to_bytes(8, "little")
    forged = rf.EngineSnapshot.from_bytes(bytes(blob))
    with pytest.raises(ValueError, match="tick"):
        engine.restore(forged)
    engine.collect(8, _mk("az")[1])  # engine unharmed


def test_stream_pause_is_a_lossless_checkpoint_barrier() -> None:
    # Reference: synchronous engine, one batch, snapshot. Streamed engine: one delivered batch,
    # pause (drains queue + in-flight), snapshot. Payloads must MATCH — stop() cannot do this.
    def build() -> tuple[rf.Engine, Any]:
        return _mk("dqn")

    sync_engine, infer = build()
    sync_batches = [sync_engine.collect(32, infer)]
    while True:  # consume as many batches as the stream will deliver, decided below
        break

    stream_engine, s_infer = build()
    stream = stream_engine.collect_stream(32, s_infer, depth=1)
    first = stream.next()
    drained = stream.pause()
    # engine returned; total delivered = 1 + len(drained); mirror on the sync side
    for _ in range(len(drained)):
        sync_batches.append(sync_engine.collect(32, infer))
    a = sync_engine.snapshot().to_bytes()
    b = stream_engine.snapshot().to_bytes()
    assert bytes(a) == bytes(b)
    assert np.array_equal(sync_batches[0].obs, first.obs)
    for got, want in zip(drained, sync_batches[1:], strict=True):
        assert np.array_equal(got.obs, want.obs)
    # and the engines stay record-identical afterwards
    assert np.array_equal(sync_engine.collect(24, infer).obs, stream_engine.collect(24, s_infer).obs)


def test_engine_envelope_booleans_are_strict() -> None:
    engine, _ = _mk("az")
    blob = bytearray(engine.snapshot().to_bytes())
    blob[4 + 1 + 4 + 64 + 8] = 2  # policy_version presence byte
    with pytest.raises(ValueError, match="not a bool"):
        rf.EngineSnapshot.from_bytes(bytes(blob))


def test_zero_floor_collect_leaves_the_snapshot_unchanged() -> None:
    engine, infer = _mk("az")
    before = engine.snapshot().to_bytes()
    batch = engine.collect(0, infer)
    assert len(batch.obs) == 0
    assert engine.snapshot().to_bytes() == before


def test_snapshots_restore_across_scheduler_knobs() -> None:
    # batch_size/n_threads are configuration, not composition: a snapshot taken under one
    # setting restores under another and the collection continues validly.
    a, infer = _mk("az")
    a.collect(8, infer)
    snap = a.snapshot()
    b = rf.Engine(
        rf.games.Chess(max_ticks=50, encoder=rf.encoders.OpenSpielChess()),
        None,
        rf.policies.AlphaZero(num_simulations=6),
        rf.learners.AlphaZero(gamma=1.0),
        n_games=2,
        seed=5,
        infer_cache=4096,
        batch_size=7,
        n_threads=2,
    )
    b.restore(snap)
    assert b.collect(8, infer).obs.shape[0] >= 8
