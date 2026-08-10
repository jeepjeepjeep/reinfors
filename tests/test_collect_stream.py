"""Continuous background collection (`engine.collect_stream`): depth-1 parity with the synchronous
path, both infer families, engine hand-back on stop, error surfacing, and overlap smoke."""

import threading
import time
from collections.abc import Callable

import numpy as np
import pytest
import reinfors as rf

_A = 7


def _az_infer(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.zeros((arr.shape[0], _A)), np.zeros(arr.shape[0])


def _az_engine(seed: int = 0) -> rf.Engine:
    return rf.Engine(
        rf.games.Connect4(),
        rf.Reward(win=1.0, loss=-1.0),
        rf.policies.AlphaZero(num_simulations=12),
        rf.learners.AlphaZero(),
        n_games=2,
        seed=seed,
    )


def _snake_infer(arr: np.ndarray) -> np.ndarray:
    # deterministic obs-dependent values so search paths depend on the "net"
    s = arr.sum(axis=1, keepdims=True)
    base = np.stack([np.sin(s + h) for h in range(4)], axis=1)  # (N, 4, 1)
    return np.repeat(base, 3, axis=2).astype(np.float64)  # (N, K=4, A=3)


def _snake_engine(seed: int = 0) -> rf.Engine:
    return rf.Engine(
        rf.games.Snake(grid_size=10, max_ticks=60),
        rf.Reward(food=1.0, loss=-5.0),
        rf.policies.SelectiveExpectimax(expansion_budget=16, top_k=4, max_depth=4, n_heads=4),
        rf.learners.TreeStrap(gamma=0.99, outcome_weight=0.3),
        n_games=2,
        seed=seed,
    )


@pytest.mark.parametrize(
    ("make_engine", "infer"),
    [(_az_engine, _az_infer), (_snake_engine, _snake_infer)],
    ids=["alphazero", "treestrap"],
)
def test_depth1_stream_matches_sequential_collects(
    make_engine: Callable[[int], rf.Engine], infer: Callable[..., object]
) -> None:
    # Frozen "weights" (pure-function infer) + same seed: the streamed batches must be bit-identical
    # to sequential collect() calls — the stream is a scheduler, not a semantics change.
    sync_batches = []
    eng = make_engine(7)
    for _ in range(3):
        sync_batches.append(eng.collect(100, infer))

    eng2 = make_engine(7)
    stream = eng2.collect_stream(100, infer, depth=1)
    try:
        for expected in sync_batches:
            got = stream.next()
            assert np.array_equal(got.obs, expected.obs)
            assert np.array_equal(got[1], expected[1])  # targets / policy_targets
            assert got.telemetry["decisions"] == expected.telemetry["decisions"]
    finally:
        stream.stop()


def test_engine_is_held_and_returned() -> None:
    eng = _az_engine()
    stream = eng.collect_stream(50, _az_infer)
    with pytest.raises(RuntimeError, match="collect_stream"):
        eng.collect(10, _az_infer)
    with pytest.raises(RuntimeError, match="collect_stream"):
        eng.collect_stream(10, _az_infer)
    stream.next()
    stream.stop()
    # engine is back: both sync collect and a fresh stream work
    batch = eng.collect(10, _az_infer)
    assert batch.obs.shape[0] >= 10
    with eng.collect_stream(20, _az_infer) as s2:
        assert s2.next().obs.shape[0] >= 20


def test_stop_is_idempotent_and_next_after_stop_raises() -> None:
    eng = _az_engine()
    stream = eng.collect_stream(30, _az_infer)
    stream.next()
    stream.stop()
    stream.stop()
    with pytest.raises(RuntimeError, match="stopped"):
        stream.next()


def test_iteration_yields_batches_then_stops() -> None:
    eng = _az_engine()
    stream = eng.collect_stream(30, _az_infer, depth=2)
    seen = []
    for batch in stream:
        seen.append(batch.obs.shape[0])
        if len(seen) == 3:
            stream.stop()
    assert len(seen) == 3 and all(m >= 30 for m in seen)


def test_callback_error_surfaces_and_engine_recovers() -> None:
    calls = {"n": 0}

    def flaky(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        calls["n"] += 1
        if calls["n"] > 3:
            raise ValueError("boom in callback")
        return _az_infer(arr)

    eng = _az_engine()
    stream = eng.collect_stream(500, flaky, depth=1)
    with pytest.raises(ValueError, match="boom in callback"):
        # the error may land on the first or a later batch depending on call counts
        for _ in range(10):
            stream.next()
    stream.stop()  # still returns the engine after a worker error
    assert eng.collect(10, _az_infer).obs.shape[0] >= 10


def test_stream_reports_infer_dtype_and_rank() -> None:
    def bad_infer(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros((arr.shape[0], _A), dtype=np.int32), np.zeros(arr.shape[0])

    eng = _az_engine()
    stream = eng.collect_stream(20, bad_infer, depth=1)
    try:
        with pytest.raises(TypeError, match=r"float64 or float32 ndarray.*dtype int32"):
            stream.next()
    finally:
        stream.stop()


def test_unbounded_depth_runs_ahead() -> None:
    eng = _az_engine()
    stream = eng.collect_stream(30, _az_infer, depth=None)
    try:
        deadline = time.time() + 20.0
        while stream.pending() < 3 and time.time() < deadline:
            time.sleep(0.05)  # consumer idles; the worker must keep collecting past depth 1
        assert stream.pending() >= 3, "unbounded stream did not run ahead of the consumer"
    finally:
        stream.stop()


def test_rejects_bad_params() -> None:
    eng = _az_engine()
    with pytest.raises(ValueError, match="depth"):
        eng.collect_stream(10, _az_infer, depth=0)
    with pytest.raises(ValueError, match="collect_size"):
        eng.collect_stream(0, _az_infer)


def test_overlap_smoke_consumer_works_while_worker_collects() -> None:
    # Not a perf assertion — just that consumer-thread Python runs concurrently with the worker
    # (no deadlock, GIL is released in next()) and a mutating callback target is tolerated.
    box = {"bias": 0.0}

    def infer(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return np.full((arr.shape[0], _A), box["bias"]), np.zeros(arr.shape[0])

    eng = _az_engine()
    stream = eng.collect_stream(60, infer, depth=1)
    done = threading.Event()

    def busy() -> None:
        while not done.is_set():
            sum(i * i for i in range(1000))  # holds/releases the GIL in slices

    t = threading.Thread(target=busy)
    t.start()
    try:
        for i in range(4):
            batch = stream.next()
            assert batch.obs.shape[0] >= 60
            box["bias"] = float(i)  # "weight update" between batches
    finally:
        done.set()
        t.join()
        stream.stop()
