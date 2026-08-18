"""PPO composition: batch contract, determinism, and config round-trip."""

from __future__ import annotations

import numpy as np
import pytest
import reinfors as rf


def zeros_pv(actions: int):
    def infer(obs: np.ndarray):
        rows = len(obs)
        return np.zeros((rows, actions), dtype=np.float32), np.zeros(rows, dtype=np.float32)

    return infer


def connect4_engine(seed: int = 0, **learner_kwargs: float) -> rf.Engine:
    return rf.Engine(
        rf.games.Connect4(),
        rf.Reward(win=1.0, loss=-1.0),
        rf.policies.Ppo(),
        rf.learners.Ppo(**learner_kwargs),
        n_games=4,
        seed=seed,
    )


def test_batch_contract() -> None:
    batch = connect4_engine().collect(n_records=32, infer=zeros_pv(7))
    assert isinstance(batch, rf.PpoBatch)
    m = len(batch.obs)
    assert m >= 32
    for field, dtype in [
        ("players", np.int64),
        ("actions", np.int64),
        ("behavior_log_probs", np.float64),
        ("advantages", np.float64),
        ("returns", np.float64),
        ("values", np.float64),
    ]:
        arr = getattr(batch, field)
        assert arr.shape == (m,) and arr.dtype == dtype, field
    assert batch.legal_offsets.shape == (m + 1,)
    # Uniform behavior under zero logits: log-prob is -log(n_legal) exactly.
    counts = np.diff(batch.legal_offsets)
    assert np.allclose(batch.behavior_log_probs, -np.log(counts))
    assert np.allclose(batch.returns, batch.advantages + batch.values)
    for i in range(m):
        legal = batch.legal_ids[batch.legal_offsets[i] : batch.legal_offsets[i + 1]]
        assert batch.actions[i] in legal


def test_deterministic_per_seed_and_stochastic_across_seeds() -> None:
    a = connect4_engine(seed=3).collect(n_records=16, infer=zeros_pv(7))
    b = connect4_engine(seed=3).collect(n_records=16, infer=zeros_pv(7))
    c = connect4_engine(seed=4).collect(n_records=16, infer=zeros_pv(7))
    assert np.array_equal(a.obs, b.obs) and np.array_equal(a.actions, b.actions)
    assert not np.array_equal(a.actions, c.actions), "sampling must vary across seeds"


def test_resolved_config_round_trip() -> None:
    engine = connect4_engine(gamma=0.9, lam=0.8)
    config = engine.resolved_config()
    assert config["policy"] == {"name": "ppo"}
    assert config["learner"] == {"name": "ppo", "gamma": 0.9, "lam": 0.8}
    rebuilt = rf.engine_from_config(config)
    assert rebuilt.config_fingerprint() == engine.config_fingerprint()


def test_learner_validation() -> None:
    for kwargs in ({"gamma": 1.5}, {"lam": -0.1}):
        with pytest.raises(ValueError):
            rf.learners.Ppo(**kwargs)


def test_rejects_non_ppo_pairings() -> None:
    with pytest.raises(ValueError, match="incompatible policy/learner composition"):
        rf.Engine(
            rf.games.Connect4(),
            rf.Reward(win=1.0, loss=-1.0),
            rf.policies.Ppo(),
            rf.learners.Dqn(),
            n_games=1,
        )


def test_imperfect_information_composes() -> None:
    batch = rf.Engine(
        rf.games.KuhnPoker(),
        None,
        rf.policies.Ppo(),
        rf.learners.Ppo(),
        n_games=2,
        seed=0,
    ).collect(n_records=8, infer=zeros_pv(2))
    assert len(batch.obs) >= 8


def test_windows_meet_the_record_floor() -> None:
    engine = connect4_engine()
    for n in (17, 64, 5):
        batch = engine.collect(n_records=n, infer=zeros_pv(7))
        assert n <= len(batch.obs) < n + 4, "floor met within one 4-game round"


def test_simultaneous_windows_overshoot_at_most_one_round() -> None:
    engine = rf.Engine(
        rf.games.Snake(grid_size=8, num_snakes=3),
        rf.Reward(food=1.0, loss=-1.0),
        rf.policies.Ppo(),
        rf.learners.Ppo(),
        n_games=2,
        seed=0,
    )
    batch = engine.collect(n_records=10, infer=zeros_pv(3))
    assert 10 <= len(batch.obs) < 10 + 6, "floor met within one 2-game round of 3 snakes"


def test_windows_are_single_version_across_collects() -> None:
    # Every record must carry its own window's critic constant: no fragment crosses the cut.
    engine = connect4_engine()

    def constant(v: float):
        def infer(obs: np.ndarray):
            rows = len(obs)
            return (
                np.zeros((rows, 7), dtype=np.float32),
                np.full(rows, v, dtype=np.float32),
            )

        return infer

    for n, v in ((16, 1.0), (24, 2.0), (8, 3.0)):
        batch = engine.collect(n_records=n, infer=constant(v))
        assert len(batch.obs) == n
        assert np.all(batch.values == v), f"stale step leaked into the v={v} window"


def test_grouped_windows_meet_the_floor_and_stay_single_version() -> None:
    engine = rf.Engine(
        rf.games.Connect4(),
        rf.Reward(win=1.0, loss=-1.0),
        rf.policies.Ppo(),
        rf.learners.Ppo(),
        n_games=4,
        seed=0,
        n_groups=2,
    )

    def constant(v: float):
        def infer(obs: np.ndarray):
            rows = len(obs)
            return (
                np.zeros((rows, 7), dtype=np.float32),
                np.full(rows, v, dtype=np.float32),
            )

        return infer

    for n, v in ((32, 1.0), (20, 2.0)):
        batch = engine.collect(n_records=n, infer=constant(v))
        assert n <= len(batch.obs) < n + 4, "both group floors met within one round"
        assert np.all(batch.values == v)


def test_snapshot_round_trips_across_a_fragment_cut() -> None:
    a = connect4_engine(seed=5)
    a.collect(n_records=16, infer=zeros_pv(7))
    snap = rf.EngineSnapshot.from_bytes(a.snapshot().to_bytes())
    follow_a = a.collect(n_records=16, infer=zeros_pv(7))

    b = connect4_engine(seed=5)
    b.restore(snap)
    follow_b = b.collect(n_records=16, infer=zeros_pv(7))
    assert np.array_equal(follow_a.obs, follow_b.obs)
    assert np.array_equal(follow_a.actions, follow_b.actions)
    assert np.array_equal(follow_a.advantages, follow_b.advantages)
