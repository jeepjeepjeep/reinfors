"""Grouped collect (n_groups=2): engine-owned scheduler — properties, config, validation.

Cacheless grouped collects with deterministic inference are exact (per-group floors +
persistent per-group rng streams); with the shared sharded cache live they are run-to-run
nondeterministic by declaration, so cache-on tests assert properties, not equality."""

import numpy as np
import pytest
import reinfors as rf

_A = rf.games.Connect4().action_space().n


def _uniform_infer(obs, n=None):
    m = obs.shape[0]
    return np.zeros((m, _A), dtype=np.float32), np.zeros(m, dtype=np.float32)


def _engine(n_groups: int, seed: int = 5) -> rf.Engine:
    return rf.Engine(
        rf.games.Connect4(),
        rf.Reward(win=1.0, loss=-1.0),
        rf.policies.AlphaZero(num_simulations=12),
        rf.learners.AlphaZero(gamma=1.0),
        n_games=4,
        seed=seed,
        n_groups=n_groups,
    )


def test_grouped_collect_is_deterministic_per_seed() -> None:
    a = _engine(2).collect(60, _uniform_infer)
    b = _engine(2).collect(60, _uniform_infer)
    assert np.array_equal(a.obs, b.obs)
    assert np.array_equal(a.policy_targets, b.policy_targets)
    assert np.array_equal(a.value_targets, b.value_targets)
    assert np.array_equal(a.legal_ids, b.legal_ids)


def test_grouped_collect_with_cache_is_sane() -> None:
    # The shared sharded cache makes grouped collects run-to-run nondeterministic by
    # declaration; assert the properties instead of equality.
    eng = rf.Engine(
        rf.games.Connect4(),
        rf.Reward(win=1.0, loss=-1.0),
        rf.policies.AlphaZero(num_simulations=12),
        rf.learners.AlphaZero(gamma=1.0),
        n_games=4,
        seed=9,
        n_groups=2,
        infer_cache=4096,
    )
    batch = eng.collect(60, _uniform_infer)
    assert batch.obs.shape[0] >= 60
    assert batch.telemetry["cache_lookups"] > 0
    assert 0 <= batch.telemetry["cache_hits"] <= batch.telemetry["cache_lookups"]
    sums = batch.policy_targets.sum(axis=1)
    assert np.allclose(sums[sums > 0], 1.0)


def test_n_groups_is_fingerprinted() -> None:
    g1, g2 = _engine(1), _engine(2)
    assert g1.resolved_config()["engine"]["n_groups"] == 1
    assert g2.resolved_config()["engine"]["n_groups"] == 2
    assert g1.config_fingerprint() != g2.config_fingerprint()


def test_grouped_collect_yields_comparable_volume() -> None:
    grouped = _engine(2).collect(60, _uniform_infer)
    ungrouped = _engine(1).collect(60, _uniform_infer)
    assert grouped.obs.shape[0] >= 60
    assert abs(int(grouped.obs.shape[0]) - int(ungrouped.obs.shape[0])) < 60


def test_rejects_bad_n_groups() -> None:
    with pytest.raises(ValueError, match="n_groups must be 1 or 2"):
        _make = rf.Engine(
            rf.games.Connect4(),
            rf.Reward(win=1.0, loss=-1.0),
            rf.policies.AlphaZero(num_simulations=8),
            rf.learners.AlphaZero(gamma=1.0),
            n_games=4,
            n_groups=3,
        )


def test_rejects_single_game_grouping() -> None:
    with pytest.raises(ValueError, match="n_games >= 2"):
        rf.Engine(
            rf.games.Connect4(),
            rf.Reward(win=1.0, loss=-1.0),
            rf.policies.AlphaZero(num_simulations=8),
            rf.learners.AlphaZero(gamma=1.0),
            n_games=1,
            n_groups=2,
        )


def test_expectimax_grouped_collect_works() -> None:
    # v2 is policy-agnostic: no grouping hooks, expectimax included.
    game = rf.games.Snake()
    a = game.action_space().n

    def infer(obs, n=None):
        return np.zeros((obs.shape[0], 1, a), dtype=np.float32)

    eng = rf.Engine(
        rf.games.Snake(),
        rf.Reward(food=1.0),
        rf.policies.SelectiveExpectimax(expansion_budget=16),
        rf.learners.TreeStrap(gamma=0.99),
        n_games=4,
        n_groups=2,
    )
    batch = eng.collect(60, infer)
    assert batch.obs.shape[0] >= 60
    assert batch.telemetry["decisions"] > 0


def test_truncation_tail_bootstrapping_works_grouped() -> None:
    # v1 excluded tail bootstrapping; the v2 service serves tail forwards like any other.
    chess_a = rf.games.Chess().action_space().n

    def az_infer(obs, n=None):
        m = obs.shape[0]
        return (
            np.zeros((m, chess_a), dtype=np.float32),
            np.zeros(m, dtype=np.float32),
        )

    eng = rf.Engine(
        rf.games.Chess(max_ticks=40),
        rf.Reward(win=1.0, loss=-1.0),
        rf.policies.AlphaZero(num_simulations=8),
        rf.learners.AlphaZero(gamma=1.0),
        n_games=4,
        n_groups=2,
    )
    batch = eng.collect(100, az_infer)
    assert batch.obs.shape[0] >= 100


def test_grouped_callback_error_propagates_without_hanging() -> None:
    eng = _engine(2)

    def bad(obs, n=None):
        raise RuntimeError("boom from infer")

    with pytest.raises(RuntimeError, match="boom from infer"):
        eng.collect(50, bad)


def test_grouped_snapshot_restore_continues_exactly() -> None:
    # cacheless + deterministic infer: grouped collects are exact, and per-group rng
    # streams live in the snapshot (schema v3) — a restored engine continues bit-for-bit.
    a1 = _engine(2, seed=7)
    b1 = _engine(2, seed=7)
    _ = a1.collect(60, _uniform_infer)
    _ = b1.collect(60, _uniform_infer)
    snap = b1.snapshot()
    b2 = _engine(2, seed=7)
    b2.restore(snap)
    batch_a = a1.collect(60, _uniform_infer)
    batch_b = b2.collect(60, _uniform_infer)
    assert np.array_equal(batch_a.obs, batch_b.obs)
    assert np.array_equal(batch_a.policy_targets, batch_b.policy_targets)


def _mcts_engine(n_groups: int, seed: int = 3) -> rf.Engine:
    return rf.Engine(
        rf.games.Connect4(),
        rf.Reward(win=1.0, loss=-1.0),
        rf.policies.Mcts(num_simulations=12),
        rf.learners.TreeStrap(gamma=1.0),
        n_games=4,
        seed=seed,
        n_groups=n_groups,
    )


def _q_infer(obs, n=None):
    return np.zeros((obs.shape[0], 1, _A), dtype=np.float32)


def test_mcts_grouped_collect_is_deterministic() -> None:
    a = _mcts_engine(2).collect(60, _q_infer)
    b = _mcts_engine(2).collect(60, _q_infer)
    assert np.array_equal(a.obs, b.obs)
    assert np.array_equal(a.targets, b.targets)
    assert np.array_equal(a.masks, b.masks)


def test_grouped_collect_stream_runs_and_reproduces() -> None:
    def first_batch(engine):
        with engine.collect_stream(40, _uniform_infer) as stream:
            return stream.next()

    a = first_batch(_engine(2))
    b = first_batch(_engine(2))
    assert a.obs.shape[0] >= 40
    assert np.array_equal(a.obs, b.obs)
    assert np.array_equal(a.policy_targets, b.policy_targets)


def test_grouped_zero_floor_is_a_no_op() -> None:
    eng = _engine(2)
    empty = eng.collect(0, _uniform_infer)
    assert empty.obs.shape[0] == 0
    assert empty.telemetry["decisions"] == 0
    # engine state untouched: the next collect matches a fresh engine's
    after = eng.collect(60, _uniform_infer)
    fresh = _engine(2).collect(60, _uniform_infer)
    assert np.array_equal(after.obs, fresh.obs)
    assert np.array_equal(after.policy_targets, fresh.policy_targets)


def test_grouped_rejects_per_player_callbacks_sync() -> None:
    eng = _engine(2)
    per_player = [_uniform_infer, _uniform_infer]
    with pytest.raises(ValueError, match="single shared infer callback"):
        eng.collect(20, per_player)


def test_grouped_rejects_per_player_callbacks_stream_without_forfeiting_engine() -> None:
    eng = _engine(2)
    per_player = [_uniform_infer, _uniform_infer]
    with pytest.raises(ValueError, match="single shared infer callback"):
        eng.collect_stream(20, per_player)
    # rejected BEFORE the engine moved into the worker: still usable
    batch = eng.collect(20, _uniform_infer)
    assert batch.obs.shape[0] >= 20
