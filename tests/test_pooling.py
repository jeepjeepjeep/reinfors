"""Pooled cross-game search is a pure throughput optimization: it must return exactly what running
each game's search alone returns, while issuing fewer (larger) inference calls.

No oracle needed — this validates reinfors against itself, so it runs in CI.
"""

import numpy as np
import reinfors

_REWARD = (0.0, 0.0, -10.0, -6.0, 20.0, 20.0, 0.0)  # step, food, loss, draw, kill, win, survival
_GAMMA, _BETA, _BUDGET, _TOP_K, _MAX_DEPTH = 0.99, 1.0, 40, 4, 8
_TEMP, _FLOOR = 1.0, 0.1
_K = 4


def _q_heads(obs: object, k: int) -> np.ndarray:
    s = float(np.asarray(obs).sum())
    return np.array(
        [[np.sin(s + 0.5 * h), np.cos(0.5 * s + 0.3 * h), np.sin(0.2 * s - 0.7 * h)] for h in range(k)],
        dtype=np.float64,
    )


def _env(a_body: list, a_dir: int, b_body: list, b_dir: int) -> object:
    e = reinfors._reinfors.SnakeEnv(12, 3, True, None)
    e.set_snakes(a_body, a_dir, True, b_body, b_dir, True)
    e.set_food([])
    return e


def test_pooled_matches_solo_and_issues_fewer_forwards() -> None:
    e0 = _env([(6, 5), (6, 4), (6, 3)], 3, [(2, 8), (2, 9), (1, 9)], 2)  # dirs: Right=3, Left=2
    e1 = _env([(3, 3), (3, 2), (3, 1)], 3, [(8, 8), (8, 9), (9, 9)], 2)

    calls = {"n": 0, "max_batch": 0}

    def infer(arr: np.ndarray) -> np.ndarray:
        calls["n"] += 1
        calls["max_batch"] = max(calls["max_batch"], arr.shape[0])
        return np.stack([_q_heads(row, _K) for row in arr])

    args = (_GAMMA, _BETA, _BUDGET, _TOP_K, _MAX_DEPTH, _REWARD, "distributional", _TEMP, _FLOOR, infer)

    calls["n"], calls["max_batch"] = 0, 0
    pooled = reinfors._reinfors.selective_search_many([e0, e1], [0, 1], *args)
    pooled_calls, pooled_batch = calls["n"], calls["max_batch"]

    calls["n"], calls["max_batch"] = 0, 0
    solo0 = e0.selective_search(0, *args)
    solo1 = e1.selective_search(1, *args)
    solo_calls, solo_batch = calls["n"], calls["max_batch"]

    # Pooling must not change any individual result.
    assert np.allclose(pooled[0][0], solo0[0]) and pooled[0][1] == solo0[1]
    assert np.allclose(pooled[1][0], solo1[0]) and pooled[1][1] == solo1[1]
    # ...but it batches: fewer, larger forwards.
    assert pooled_calls < solo_calls, f"pooled forwards {pooled_calls} vs solo {solo_calls}"
    assert pooled_batch > solo_batch, f"pooled batch {pooled_batch} vs solo {solo_batch}"
