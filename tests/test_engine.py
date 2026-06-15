"""Engine rollout collector: shapes, determinism, bootstrap masks, and that its (unblended) root
targets are exactly the pooled search's outputs (tying the collected data to the search the parity
suite already validates).

No oracle needed — validates reinfors against itself, so it runs in CI.
"""

import numpy as np
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
) -> object:
    return reinfors._reinfors.Engine(
        n_games,
        _G,
        3,
        False,
        None,
        food,  # n_games, grid, initial_length, play_to_last, win_food_lead, initial_food_count
        *_SEARCH,
        _REWARD,
        "uniform",
        1.0,
        0.1,  # reward, opponent, opp_temperature, opp_floor
        0.1,
        max_ticks,
        _K,  # epsilon, max_ticks, n_heads
        outcome_weight,
        interior,
        bootstrap_p,  # outcome_weight, interior_targets, bootstrap_p
        seed,
    )


def test_collect_shapes_and_dtypes() -> None:
    obs, tgt, mask = _engine(0).collect(50, _infer)
    m = obs.shape[0]
    assert m >= 50
    assert obs.shape == (m, 5 * _G * _G) and obs.dtype == np.float32
    assert tgt.shape == (m, _K, 3) and tgt.dtype == np.float64
    assert mask.shape == (m, _K) and mask.dtype == np.float32
    assert np.isin(mask, (0.0, 1.0)).all()


def test_collect_is_deterministic_for_a_seed() -> None:
    o1, t1, m1 = _engine(7).collect(60, _infer)
    o2, t2, m2 = _engine(7).collect(60, _infer)
    assert np.array_equal(o1, o2) and np.array_equal(t1, t2) and np.array_equal(m1, m2)


def test_distinct_seeds_diverge() -> None:
    o1, _, _ = _engine(1).collect(80, _infer)
    o2, _, _ = _engine(2).collect(80, _infer)
    assert not np.array_equal(o1, o2)


def test_bootstrap_p_extremes_set_all_or_no_heads() -> None:
    _, _, all_mask = _engine(3, bootstrap_p=1.0).collect(40, _infer)
    assert (all_mask == 1.0).all()
    _, _, no_mask = _engine(3, bootstrap_p=0.0).collect(40, _infer)
    assert (no_mask == 0.0).all()


def test_first_targets_equal_a_direct_pooled_search() -> None:
    # One-tick, food-free episodes with outcome_weight=0 so the z-mix is a no-op: every game truncates
    # after its first decision and flushes its raw searched values in game-major order (A then B). So
    # the first two targets must equal a direct pooled search of both agents on the initial state,
    # pinning the collected targets to the (separately oracle-validated) search.
    _, tgt, _ = _engine(0, food=0, max_ticks=1, outcome_weight=0.0, interior=False).collect(2, _infer)
    env = reinfors._reinfors.SnakeEnv(_G, 3, False, None)
    results = reinfors._reinfors.selective_search_many(
        [env, env], [0, 1], *_SEARCH, _REWARD, "uniform", 1.0, 0.1, _infer
    )
    assert np.allclose(tgt[0], np.asarray(results[0][0]), atol=1e-9)  # game 0, agent A
    assert np.allclose(tgt[1], np.asarray(results[1][0]), atol=1e-9)  # game 0, agent B
