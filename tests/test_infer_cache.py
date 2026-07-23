"""The engine-level infer cache: behavior identity, hit telemetry, and weights_updated invalidation."""

import numpy as np
import reinfors as rf

_A = 7


def _infer(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # f32-quantized deterministic values: cache stores f32, so identity holds bit-exactly.
    s = arr.sum(axis=1, keepdims=True).astype(np.float32)
    logits = np.repeat(np.sin(s), _A, axis=1).astype(np.float32) + np.arange(_A, dtype=np.float32) * 0.01
    values = np.tanh(s[:, 0]).astype(np.float32)
    return logits.astype(np.float64), values.astype(np.float64)


def _engine(infer_cache: int = 0, seed: int = 0) -> rf.Engine:
    return rf.Engine(
        rf.games.Connect4(),
        rf.Reward(win=1.0, loss=-1.0),
        rf.policies.AlphaZero(num_simulations=16),
        rf.learners.AlphaZero(),
        n_games=2,
        seed=seed,
        infer_cache=infer_cache,
    )


def test_cache_is_behavior_identical() -> None:
    plain = _engine(0).collect(80, _infer)
    cached = _engine(1 << 16).collect(80, _infer)
    assert isinstance(plain, rf._reinfors.AlphaZeroBatch)
    assert isinstance(cached, rf._reinfors.AlphaZeroBatch)
    assert np.array_equal(plain.obs, cached.obs)
    assert np.array_equal(plain.policy_targets, cached.policy_targets)
    assert np.array_equal(plain.value_targets, cached.value_targets)


def test_cache_telemetry_and_row_savings() -> None:
    b0 = _engine(0).collect(80, _infer)
    b1 = _engine(1 << 16).collect(80, _infer)
    assert b0.telemetry["cache_lookups"] == 0 and b0.telemetry["cache_hits"] == 0
    assert b1.telemetry["cache_lookups"] > 0
    assert b1.telemetry["cache_hits"] > 0  # connect4 self-play repeats positions
    # hits reduce forwarded rows for the same search work
    assert b1.telemetry["infer_rows"] < b0.telemetry["infer_rows"]


def test_cache_persists_across_collects_until_weights_updated() -> None:
    eng = _engine(1 << 16)
    eng.collect(60, _infer)
    warm = eng.collect(60, _infer).telemetry  # warm cache from collect 1
    eng2 = _engine(1 << 16)
    eng2.collect(60, _infer)
    eng2.weights_updated()  # trainer installed new weights -> entries must not be reused
    cleared = eng2.collect(60, _infer).telemetry
    assert warm["cache_hits"] > cleared["cache_hits"], "weights_updated must clear cross-collect reuse"


def test_weights_updated_is_safe_without_cache_and_during_stream() -> None:
    eng = _engine(0)
    eng.weights_updated()  # no-op, no error
    eng = _engine(1 << 16)
    with eng.collect_stream(40, _infer, depth=1) as stream:
        stream.next()
        eng.weights_updated()  # callable while the worker holds the engine
        stream.next()
