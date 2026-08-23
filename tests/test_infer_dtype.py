"""float32 infer outputs: the exact-widening contract. f32 -> f64 conversion is exact, so an
infer returning the net's native float32 must produce BIT-IDENTICAL batches to one returning
`.astype(float64)` of the same arrays — pinned here per surface. float32 skips the
producer-side up-conversion and halves the boundary bytes (the GPU fast path)."""

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
import reinfors as rf

InferFn = Callable[[np.ndarray], Any]

_A = 7  # connect4 actions
_RNG = np.random.default_rng(11)
_W = _RNG.standard_normal((84, _A)).astype(np.float32)


def _f32_logits(obs: np.ndarray) -> np.ndarray:
    return (obs.reshape(obs.shape[0], -1) @ _W).astype(np.float32)


def _az_infer(dtype: type) -> "InferFn":
    def infer(obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        logits = _f32_logits(obs)  # computed in f32 ALWAYS; f64 variant widens (= .double())
        values = np.tanh(logits.sum(axis=1) / 8.0, dtype=np.float32)
        return logits.astype(dtype), values.astype(dtype)

    return infer


def _q_infer(dtype: type) -> "InferFn":
    def infer(obs: np.ndarray) -> np.ndarray:
        return _f32_logits(obs).reshape(-1, 1, _A).astype(dtype)

    return infer


def _az_engine(seed: int = 5) -> "rf.Engine":
    return rf.Engine(
        rf.games.Connect4(),
        rf.Reward(win=1.0, loss=-1.0),
        rf.policies.AlphaZero(num_simulations=8, c_puct=2.0),
        rf.learners.AlphaZero(gamma=1.0),
        n_games=2,
        seed=seed,
        n_threads=1,  # bit-identity is only contracted on the reproducible schedule
    )


def _q_engine(seed: int = 5) -> "rf.Engine":
    return rf.Engine(
        rf.games.Connect4(),
        rf.Reward(win=1.0, loss=-1.0),
        rf.policies.EpsilonGreedyQ(n_heads=1, epsilon=0.2),
        rf.learners.Dqn(),
        n_games=2,
        seed=seed,
        n_threads=1,  # bit-identity is only contracted on the reproducible schedule
    )


def _sig(batch: object) -> dict[str, bytes]:
    arrays = ((k, getattr(batch, k, None)) for k in dir(batch))
    return {k: np.ascontiguousarray(v).tobytes() for k, v in arrays if isinstance(v, np.ndarray)}


def test_alphazero_f32_infer_is_bit_identical() -> None:
    a = _az_engine().collect(60, _az_infer(np.float64))
    b = _az_engine().collect(60, _az_infer(np.float32))
    assert _sig(a) == _sig(b)


def test_value_family_f32_infer_is_bit_identical() -> None:
    a = _q_engine().collect(60, _q_infer(np.float64))
    b = _q_engine().collect(60, _q_infer(np.float32))
    assert _sig(a) == _sig(b)


def test_mixed_dtype_alphazero_tuple_is_accepted() -> None:
    def infer(obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        logits = _f32_logits(obs)
        return logits, np.zeros(obs.shape[0], dtype=np.float64)  # f32 logits, f64 values

    batch = _az_engine().collect(30, infer)
    assert batch.obs.shape[0] >= 30  # collect rounds up to episode boundaries


def test_collect_stream_accepts_f32() -> None:
    engine = _az_engine()
    with engine.collect_stream(16, _az_infer(np.float32)) as stream:
        batch = stream.next()
    assert batch.obs.shape[0] >= 16  # record floor, not an exact size


def test_wrong_dtype_names_the_accepted_ones() -> None:
    def int_infer(obs: np.ndarray) -> np.ndarray:
        return np.zeros((obs.shape[0], 1, _A), dtype=np.int64)

    with pytest.raises(TypeError, match="float64 or float32"):
        _q_engine().collect(20, int_infer)

    def bad_az(obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros((obs.shape[0], _A), dtype=np.int32), np.zeros(obs.shape[0])

    with pytest.raises(TypeError, match="float64 or float32"):
        _az_engine().collect(20, bad_az)


def test_deep_cfr_f32_infer_is_bit_identical() -> None:
    def run(dtype: type) -> dict[str, bytes]:
        rng_w = np.random.default_rng(3).standard_normal((6, 2)).astype(np.float32)

        def infer(obs: np.ndarray) -> np.ndarray:
            return (obs.reshape(obs.shape[0], -1) @ rng_w).astype(dtype)

        solver = rf.solvers.DeepCfr(rf.games.KuhnPoker(), seed=4)
        solver.next_iteration()
        b = solver.collect(player=0, traversals=32, infer=infer)
        return _sig(b)

    assert run(np.float64) == run(np.float32)


def test_exploitability_instrument_accepts_f32_exactly() -> None:
    solver = rf.solvers.DeepCfr(rf.games.KuhnPoker(), seed=0)

    def uniform(dtype: type) -> "InferFn":
        return lambda obs: np.full((obs.shape[0], 2), 0.5, dtype=dtype)

    assert solver.exploitability(uniform(np.float32)) == solver.exploitability(uniform(np.float64))


def test_padded_logits_are_accepted_and_identical() -> None:
    """A wider-than-A head (chess: 4674 over 4672) may be returned whole — the tail is
    ignored, so producers skip the pre-transfer slice (a device-side gather on GPU)."""

    def padded(obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        logits = _f32_logits(obs)
        junk = np.full((obs.shape[0], 2), 1e9, dtype=np.float32)  # tail must be ignored
        return np.concatenate([logits, junk], axis=1), np.tanh(logits.sum(axis=1) / 8.0, dtype=np.float32)

    a = _az_engine().collect(60, _az_infer(np.float32))
    b = _az_engine().collect(60, padded)
    assert _sig(a) == _sig(b)


def test_too_narrow_logits_still_error() -> None:
    def narrow(obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros((obs.shape[0], _A - 1), dtype=np.float32), np.zeros(obs.shape[0], dtype=np.float32)

    with pytest.raises(ValueError, match="policy_logits"):
        _az_engine().collect(20, narrow)
