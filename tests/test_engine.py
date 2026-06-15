"""Engine rollout collector: shapes, determinism, bootstrap masks, and that its (unblended) root
targets are exactly the pooled search's outputs (tying the collected data to the search the parity
suite already validates).

No oracle needed — validates reinfors against itself, so it runs in CI.
"""

import numpy as np
import pytest
import reinfors

_G = 12
_K = 4
_REWARD = (0.0, 0.0, -10.0, -6.0, 20.0, 20.0, 0.0)  # step, food, loss, draw, kill, win, survival
# gamma, beta, budget, top_k, max_depth (shared by the engine's search and the direct check below)
_SEARCH = (0.99, 1.0, 24, 4, 6)


def _q_heads(obs: object, k: int) -> np.ndarray:
    s = float(np.asarray(obs).sum())
    return np.array(
        [[np.sin(s + 0.5 * h), np.cos(0.5 * s + 0.3 * h), np.sin(0.2 * s - 0.7 * h)] for h in range(k)],
        dtype=np.float64,
    )


def _infer(arr: np.ndarray) -> np.ndarray:
    return np.stack([_q_heads(row, _K) for row in arr])


def _engine(
    seed: int,
    *,
    n_games: int = 4,
    food: int = 3,
    max_ticks: int = 50,
    outcome_weight: float = 0.5,
    interior: bool = True,
    bootstrap_p: float = 0.8,
    survival: float = 0.0,
    n_heads: int = _K,
    epsilon: float = 0.1,
) -> object:
    reward = (*_REWARD[:6], survival)  # (step, food, loss, draw, kill, win, survival)
    return reinfors._reinfors.Engine(
        n_games,
        _G,
        3,
        False,
        None,
        food,  # n_games, grid, initial_length, play_to_last, win_food_lead, initial_food_count
        *_SEARCH,
        reward,
        "uniform",
        1.0,
        0.1,  # reward, opponent, opp_temperature, opp_floor
        epsilon,
        max_ticks,
        n_heads,  # epsilon, max_ticks, n_heads
        outcome_weight,
        interior,
        bootstrap_p,  # outcome_weight, interior_targets, bootstrap_p
        seed,
    )


def test_collect_shapes_and_dtypes() -> None:
    obs, tgt, mask, _stats = _engine(0).collect(50, _infer)
    m = obs.shape[0]
    assert m >= 50
    assert obs.shape == (m, 5 * _G * _G) and obs.dtype == np.float32
    assert tgt.shape == (m, _K, 3) and tgt.dtype == np.float64
    assert mask.shape == (m, _K) and mask.dtype == np.float32
    assert np.isin(mask, (0.0, 1.0)).all()


def test_collect_is_deterministic_for_a_seed() -> None:
    o1, t1, m1, _ = _engine(7).collect(60, _infer)
    o2, t2, m2, _ = _engine(7).collect(60, _infer)
    assert np.array_equal(o1, o2) and np.array_equal(t1, t2) and np.array_equal(m1, m2)


def test_distinct_seeds_diverge() -> None:
    o1, _, _, _ = _engine(1).collect(80, _infer)
    o2, _, _, _ = _engine(2).collect(80, _infer)
    assert not np.array_equal(o1, o2)


def test_bootstrap_p_extremes_set_all_or_no_heads() -> None:
    _, _, all_mask, _ = _engine(3, bootstrap_p=1.0).collect(40, _infer)
    assert (all_mask == 1.0).all()
    _, _, no_mask, _ = _engine(3, bootstrap_p=0.0).collect(40, _infer)
    assert (no_mask == 0.0).all()


def test_survival_bonus_propagates_through_z_mixing_on_truncation() -> None:
    # max_ticks=1 truncates every episode after one surviving, food-free decision; outcome_weight=1
    # makes the executed action's target equal the realized return, which on truncation includes the
    # survival bonus. Two engines that differ only in `survival` must produce targets differing by
    # exactly the bonus, and only in the executed action's entry: survival changes neither the search
    # values, the chosen action, nor the z-tail. Guards the previously-dead survival reward.
    bonus = 0.25
    kw = {"food": 0, "max_ticks": 1, "outcome_weight": 1.0, "interior": False}
    _, t0, _, _ = _engine(0, survival=0.0, **kw).collect(4, _infer)
    _, ts, _, _ = _engine(0, survival=bonus, **kw).collect(4, _infer)
    assert t0.shape == ts.shape and t0.shape[0] >= 4
    diff = ts - t0  # (M, K, A)
    for m in range(diff.shape[0]):
        for h in range(_K):
            changed = np.flatnonzero(np.abs(diff[m, h]) > 1e-9)
            assert changed.size == 1, "only the executed action's target should move"
            assert np.isclose(diff[m, h, changed[0]], bonus, atol=1e-9)


def test_first_targets_equal_a_direct_pooled_search() -> None:
    # One-tick, food-free episodes with outcome_weight=0 so the z-mix is a no-op: every game truncates
    # after its first decision and flushes its raw searched values in game-major order (A then B). So
    # the first two targets must equal a direct pooled search of both agents on the initial state,
    # pinning the collected targets to the (separately oracle-validated) search.
    _, tgt, _, _ = _engine(0, food=0, max_ticks=1, outcome_weight=0.0, interior=False).collect(2, _infer)
    env = reinfors._reinfors.SnakeEnv(_G, 3, False, None)
    results = reinfors._reinfors.selective_search_many(
        [env, env], [0, 1], *_SEARCH, _REWARD, "uniform", 1.0, 0.1, _infer
    )
    assert np.allclose(tgt[0], np.asarray(results[0][0]), atol=1e-9)  # game 0, agent A
    assert np.allclose(tgt[1], np.asarray(results[1][0]), atol=1e-9)  # game 0, agent B


def test_collect_returns_telemetry() -> None:
    # The 4th return is a logging telemetry dict: finished-episode summaries plus per-call search
    # aggregates. Interior off so episodes finish within the record budget (with it on, the floor is
    # hit via interior targets before any episode completes).
    eng = _engine(11, interior=False)
    episodes: list = []
    decisions = 0
    for _ in range(4):
        _obs, _tgt, _mask, stats = eng.collect(400, _infer)
        assert set(stats) >= {
            "episodes",
            "decisions",
            "max_depth",
            "mean_leaves",
            "mean_rounds",
            "mean_expansions",
            "mean_sigma",
            "mean_disagreement",
        }
        episodes += stats["episodes"]
        decisions += stats["decisions"]
        assert stats["max_depth"] > 0 and stats["mean_leaves"] > 0.0
        assert stats["mean_sigma"] >= 0.0 and stats["mean_disagreement"] >= 0.0
    assert decisions > 0 and len(episodes) > 0
    for reward_a, reward_b, length in episodes:
        assert 1 <= length <= 50  # bounded by max_ticks
        assert np.isfinite(reward_a) and np.isfinite(reward_b)


def test_telemetry_is_deterministic_for_a_seed() -> None:
    _, _, _, s1 = _engine(7).collect(120, _infer)
    _, _, _, s2 = _engine(7).collect(120, _infer)
    assert s1["episodes"] == s2["episodes"] and s1["decisions"] == s2["decisions"]


@pytest.mark.parametrize(
    "bad",
    [
        {"n_games": 0},
        {"max_ticks": 0},
        {"n_heads": 0},
        {"epsilon": 1.5},
        {"epsilon": -0.1},
        {"outcome_weight": 2.0},
        {"bootstrap_p": -0.1},
    ],
)
def test_engine_rejects_degenerate_params(bad: dict) -> None:
    with pytest.raises(ValueError):
        _engine(0, **bad)


def test_engine_rejects_head_count_mismatch() -> None:
    # The Engine is built for 3 heads but `_infer` returns _K=4 — the head count must match, else
    # Thompson sampling silently breaks. Surfaces as a clean ValueError, not a clamp (or a panic).
    engine = _engine(0, n_heads=3, interior=False)
    with pytest.raises(ValueError, match="n_heads"):
        engine.collect(10, _infer)


def test_infer_error_surfaces_during_collect() -> None:
    # A failing network forward must propagate its real error — the head-count check (success path
    # only) must not mask it. Regression guard: a raising infer with n_heads > 1 used to panic with a
    # head-count message because the error fallback (K=1) tripped a core-side assert.
    def boom(_arr: np.ndarray) -> np.ndarray:
        raise RuntimeError("MY_NETWORK_OOM_xyz")

    engine = _engine(0, n_heads=2, interior=False)
    with pytest.raises(RuntimeError, match="MY_NETWORK_OOM_xyz"):
        engine.collect(10, boom)
