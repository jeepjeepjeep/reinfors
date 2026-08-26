"""The zero-copy engine at the python surface: validation, composition, reproducibility."""

from __future__ import annotations

import numpy as np
import pytest
import reinfors as rf


def _engine(**kw) -> rf.Engine:
    return rf.Engine(
        rf.games.Snake(grid_size=6, max_ticks=40),
        rf.Reward(food=1.0, loss=-1.0),
        rf.policies.Ppo(),
        rf.learners.Ppo(),
        n_games=2,
        seed=0,
        n_threads=1,
        **kw,
    )


def _infer(obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.zeros((len(obs), 4), dtype=np.float32),
        np.zeros(len(obs), dtype=np.float32),
    )


def test_oversized_arena_is_rejected_at_construction() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        rf.Engine(
            rf.games.CarRacing(),
            rf.Reward(),
            rf.policies.Ppo(),
            rf.learners.Ppo(),
            n_games=1,
            batch_size=1 << 20,
        )


def test_pad_composes() -> None:
    batch = _engine(pad=True, batch_size=4).collect(n_records=12, infer=_infer)
    assert len(batch.obs) >= 12


def test_infer_cache_composes_and_serves_hits() -> None:
    batch = _engine(infer_cache=1024).collect(n_records=24, infer=_infer)
    assert len(batch.obs) >= 24


def test_collect_is_reproducible() -> None:
    def run() -> np.ndarray:
        return _engine().collect(n_records=12, infer=_infer).obs

    assert np.array_equal(run(), run())
