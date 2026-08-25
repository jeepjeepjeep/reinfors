"""The zero_copy engine flag: validation, config round-trip, and a collect smoke."""

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


def test_resolved_config_round_trips() -> None:
    cfg = _engine(zero_copy=True).resolved_config()
    assert cfg["engine"]["zero_copy"] is True
    rebuilt = rf.engine_from_config(cfg)
    assert rebuilt.resolved_config() == cfg
    assert len(rebuilt.collect(n_records=8, infer=_infer).obs) >= 8
    assert _engine().resolved_config()["engine"]["zero_copy"] is None


def test_oversized_arena_is_rejected_at_construction() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        rf.Engine(
            rf.games.CarRacing(),
            rf.Reward(),
            rf.policies.Ppo(),
            rf.learners.Ppo(),
            n_games=1,
            zero_copy=True,
            batch_size=1 << 20,
        )


def test_unsupported_combinations_are_rejected() -> None:
    with pytest.raises(ValueError, match="zero_copy"):
        _engine(zero_copy=True, pad=True)
    with pytest.raises(ValueError, match="zero_copy"):
        _engine(zero_copy=True, infer_cache=64)


def test_collect_matches_the_classic_path() -> None:
    def run(zero_copy: bool) -> np.ndarray:
        batch = _engine(zero_copy=zero_copy).collect(n_records=12, infer=_infer)
        return batch.obs

    assert np.array_equal(run(False), run(True))
