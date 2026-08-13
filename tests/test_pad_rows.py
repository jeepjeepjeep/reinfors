"""pad_rows_to: fixed call shapes as a semantic no-op, telemetry, validation."""

import numpy as np
import pytest
import reinfors as rf

_A = rf.games.Connect4().action_space().n


def _uniform(obs, n=None):
    m = obs.shape[0]
    return np.zeros((m, _A), dtype=np.float32), np.zeros(m, dtype=np.float32)


def _engine(n_groups: int = 1, pad_rows_to: int = 0, seed: int = 5) -> rf.Engine:
    return rf.Engine(
        rf.games.Connect4(),
        rf.Reward(win=1.0, loss=-1.0),
        rf.policies.AlphaZero(num_simulations=12),
        rf.learners.AlphaZero(gamma=1.0),
        n_games=4,
        seed=seed,
        n_groups=n_groups,
        pad_rows_to=pad_rows_to,
    )


@pytest.mark.parametrize("n_groups", [1, 2])
def test_padding_is_a_semantic_no_op(n_groups: int) -> None:
    a = _engine(n_groups).collect(60, _uniform)
    b = _engine(n_groups, pad_rows_to=8).collect(60, _uniform)
    assert np.array_equal(a.obs, b.obs)
    assert np.array_equal(a.policy_targets, b.policy_targets)
    assert np.array_equal(a.value_targets, b.value_targets)
    assert b.telemetry["infer_rows"] == a.telemetry["infer_rows"]
    assert a.telemetry["padded_rows"] == 0
    assert b.telemetry["padded_rows"] > 0


def test_calls_arrive_at_the_fixed_shape() -> None:
    seen = set()

    def recording(obs, n=None):
        seen.add(obs.shape[0])
        return _uniform(obs)

    batch = _engine(pad_rows_to=8).collect(60, recording)
    assert seen == {8}
    t = batch.telemetry
    assert t["infer_calls"] * 8 == t["infer_rows"] + t["padded_rows"]


def test_oversize_calls_are_chunked_to_the_exact_shape() -> None:
    seen = set()

    def recording(obs, n=None):
        seen.add(obs.shape[0])
        return _uniform(obs)

    _engine(pad_rows_to=2).collect(60, recording)
    assert seen == {2}


def test_padded_stream_matches_padded_collect() -> None:
    with _engine(2, pad_rows_to=8).collect_stream(40, _uniform) as stream:
        a = stream.next()
    b = _engine(2, pad_rows_to=8).collect(40, _uniform)
    assert np.array_equal(a.obs, b.obs)
    assert np.array_equal(a.policy_targets, b.policy_targets)


def test_padding_rejects_per_player_callbacks() -> None:
    eng = _engine(pad_rows_to=8)
    with pytest.raises(ValueError, match="single shared infer callback"):
        eng.collect(20, [_uniform, _uniform])
    with pytest.raises(ValueError, match="single shared infer callback"):
        eng.collect_stream(20, [_uniform, _uniform])
    # rejected before the engine moved into the worker: still usable
    assert eng.collect(20, _uniform).obs.shape[0] >= 20


def test_pad_rows_to_is_fingerprinted() -> None:
    assert _engine().resolved_config()["engine"]["pad_rows_to"] == 0
    assert _engine(pad_rows_to=8).resolved_config()["engine"]["pad_rows_to"] == 8
    assert _engine().config_fingerprint() != _engine(pad_rows_to=8).config_fingerprint()


def test_padding_composes_with_the_cache() -> None:
    eng = rf.Engine(
        rf.games.Connect4(),
        rf.Reward(win=1.0, loss=-1.0),
        rf.policies.AlphaZero(num_simulations=12),
        rf.learners.AlphaZero(gamma=1.0),
        n_games=4,
        seed=9,
        infer_cache=4096,
        pad_rows_to=8,
    )
    batch = eng.collect(60, _uniform)
    assert batch.telemetry["cache_lookups"] > 0
    assert batch.telemetry["padded_rows"] > 0
