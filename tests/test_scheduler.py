"""Threshold scheduler: determinism, knob policy, cancel, and streaming."""

import numpy as np
import pytest
import reinfors as rf
from conftest import az_zeros_infer, connect4_az_engine

_A = rf.games.Connect4().action_space().n

_uniform = az_zeros_infer(_A)


def test_collect_is_deterministic_per_seed() -> None:
    a = connect4_az_engine(seed=5).collect(40, _uniform)
    b = connect4_az_engine(seed=5).collect(40, _uniform)
    assert np.array_equal(a.obs, b.obs)
    assert np.array_equal(a.policy_targets, b.policy_targets)


def test_fanned_collect_matches_single_threaded_exactly() -> None:
    a = connect4_az_engine(seed=5, n_threads=1).collect(40, _uniform)
    b = connect4_az_engine(seed=5, n_threads=4).collect(40, _uniform)
    assert np.array_equal(a.obs, b.obs)
    assert np.array_equal(a.policy_targets, b.policy_targets)
    assert np.array_equal(a.value_targets, b.value_targets)


def test_batch_size_changes_call_grouping_not_totals() -> None:
    rows_a: list[int] = []
    rows_b: list[int] = []

    def recording(sink):
        def infer(obs, n=None):
            sink.append(obs.shape[0])
            return _uniform(obs)

        return infer

    connect4_az_engine(seed=5, batch_size=2).collect(40, recording(rows_a))
    connect4_az_engine(seed=5, batch_size=16).collect(40, recording(rows_b))
    assert sum(rows_a) == sum(rows_b), "grouping moved rows between calls, not created them"
    assert max(rows_b) > max(rows_a)


def test_scheduler_knobs_split_config_but_not_snapshots() -> None:
    base = connect4_az_engine(seed=5)
    for kwargs in ({"batch_size": 3}, {"n_threads": 4}, {"pad": True}):
        other = connect4_az_engine(seed=5, **kwargs)
        # configuration: differently tuned engines are distinguishable
        assert other.config_fingerprint() != base.config_fingerprint()
        # composition: checkpoints stay portable across the knobs
        other.restore(base.snapshot())


def test_rejects_absurd_scheduler_knobs() -> None:
    with pytest.raises(ValueError, match="n_threads must be <="):
        connect4_az_engine(n_threads=100_000)
    with pytest.raises(ValueError, match="batch_size must be <="):
        connect4_az_engine(batch_size=1 << 30)


def test_callback_error_cancels_without_hanging() -> None:
    engine = connect4_az_engine(seed=5)

    def flaky(obs, n=None):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        engine.collect(40, flaky)
    batch = engine.collect(40, _uniform)
    assert batch.obs.shape[0] >= 40, "the engine stays usable after a cancelled collect"


def test_collect_with_cache_is_sane() -> None:
    eng = connect4_az_engine(seed=9, infer_cache=4096)
    batch = eng.collect(60, _uniform)
    t = batch.telemetry
    assert t["cache_lookups"] > 0
    assert t["requested_rows"] >= t["infer_rows"]


def test_stream_first_batch_matches_direct_collect() -> None:
    with connect4_az_engine(seed=7).collect_stream(40, _uniform) as stream:
        a = stream.next()
    b = connect4_az_engine(seed=7).collect(40, _uniform)
    assert np.array_equal(a.obs, b.obs)
    assert np.array_equal(a.policy_targets, b.policy_targets)


def test_stream_callback_error_still_propagates() -> None:
    def flaky(obs, n=None):
        raise RuntimeError("stream boom")

    with connect4_az_engine(seed=7).collect_stream(40, flaky) as stream:
        with pytest.raises(RuntimeError, match="stream boom"):
            stream.next()


def test_zero_floor_is_a_no_op() -> None:
    batch = connect4_az_engine(seed=5).collect(0, _uniform)
    assert batch.obs.shape[0] == 0


def test_snapshot_restore_continues_exactly() -> None:
    a = connect4_az_engine(seed=11)
    a.collect(30, _uniform)
    snap = rf.EngineSnapshot.from_bytes(a.snapshot().to_bytes())
    follow_a = a.collect(30, _uniform)

    b = connect4_az_engine(seed=11)
    b.restore(snap)
    follow_b = b.collect(30, _uniform)
    assert np.array_equal(follow_a.obs, follow_b.obs)
    assert np.array_equal(follow_a.policy_targets, follow_b.policy_targets)
