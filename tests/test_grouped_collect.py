"""Double-buffered collect (n_groups=2): determinism, config surface, and validation."""

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


def test_grouped_collect_with_cache_is_deterministic() -> None:
    def eng():
        return rf.Engine(
            rf.games.Connect4(),
            rf.Reward(win=1.0, loss=-1.0),
            rf.policies.AlphaZero(num_simulations=12),
            rf.learners.AlphaZero(gamma=1.0),
            n_games=4,
            seed=9,
            n_groups=2,
            infer_cache=4096,
        )

    a = eng().collect(60, _uniform_infer)
    b = eng().collect(60, _uniform_infer)
    assert np.array_equal(a.obs, b.obs)
    assert np.array_equal(a.policy_targets, b.policy_targets)
    assert a.telemetry["cache_lookups"] > 0


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


def test_rejects_unpooled_policy() -> None:
    with pytest.raises(ValueError, match="pooled-search policy"):
        rf.Engine(
            rf.games.Snake(),
            rf.Reward(food=1.0),
            rf.policies.SelectiveExpectimax(expansion_budget=16),
            rf.learners.TreeStrap(),
            n_games=4,
            n_groups=2,
        )


def test_rejects_truncation_tail_bootstrapping() -> None:
    with pytest.raises(ValueError, match="truncation-tail"):
        rf.Engine(
            rf.games.Chess(max_ticks=64),
            rf.Reward(win=1.0, loss=-1.0),
            rf.policies.AlphaZero(num_simulations=8),
            rf.learners.AlphaZero(gamma=1.0),
            n_games=4,
            n_groups=2,
        )


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
