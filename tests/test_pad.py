"""pad: every callback invocation at exactly batch_size rows, as a semantic no-op."""

import numpy as np
import pytest
import reinfors as rf
from conftest import az_zeros_infer, connect4_az_engine

_A = rf.games.Connect4().action_space().n


_uniform = az_zeros_infer(_A)


def test_padding_is_a_semantic_no_op() -> None:
    # batch_size above the pool's per-sweep row count forces drain batches, which pad fills
    a = connect4_az_engine(seed=5, batch_size=8).collect(60, _uniform)
    b = connect4_az_engine(seed=5, batch_size=8, pad=True).collect(60, _uniform)
    assert np.array_equal(a.obs, b.obs)
    assert np.array_equal(a.policy_targets, b.policy_targets)
    assert np.array_equal(a.value_targets, b.value_targets)
    assert b.telemetry["infer_rows"] == a.telemetry["infer_rows"]
    assert a.telemetry["padded_rows"] == 0
    assert b.telemetry["padded_rows"] > 0


def test_calls_arrive_at_exactly_batch_size_rows() -> None:
    seen = set()

    def recording(obs, n=None):
        seen.add(obs.shape[0])
        return _uniform(obs)

    batch = connect4_az_engine(seed=5, pad=True, batch_size=8).collect(60, recording)
    assert seen == {8}
    t = batch.telemetry
    assert t["infer_calls"] * 8 == t["infer_rows"] + t["padded_rows"]


def test_padded_stream_matches_padded_collect() -> None:
    with connect4_az_engine(seed=5, pad=True, batch_size=8).collect_stream(40, _uniform) as stream:
        a = stream.next()
    b = connect4_az_engine(seed=5, pad=True, batch_size=8).collect(40, _uniform)
    assert np.array_equal(a.obs, b.obs)
    assert np.array_equal(a.policy_targets, b.policy_targets)


def test_padding_rejects_per_player_callbacks() -> None:
    eng = connect4_az_engine(seed=5, pad=True)
    with pytest.raises(ValueError, match="single shared infer callback"):
        eng.collect(20, [_uniform, _uniform])
    with pytest.raises(ValueError, match="single shared infer callback"):
        eng.collect_stream(20, [_uniform, _uniform])
    # rejected before the engine moved into the worker: still usable
    assert eng.collect(20, _uniform).obs.shape[0] >= 20


def test_pad_is_not_fingerprinted() -> None:
    a = connect4_az_engine(seed=5)
    b = connect4_az_engine(seed=5, pad=True)
    assert a.config_fingerprint() == b.config_fingerprint()
    assert "pad_rows_to" not in a.resolved_config()["engine"]


def test_padding_composes_with_the_cache() -> None:
    eng = connect4_az_engine(seed=9, infer_cache=4096, pad=True)
    batch = eng.collect(60, _uniform)
    assert batch.telemetry["cache_lookups"] > 0
    assert batch.telemetry["padded_rows"] > 0
