"""Differential parity: reinfors' selective expectimax vs snake_RL's SelectiveExpectimaxPlanner.

Covers the matrix the oracle supports: opponent in {uniform, distributional (deferred)} x heads in
{1, 4} x both agents x several (budget, top_k, max_depth, beta) configs. Both sides share the exact
per-head value function `_q_heads` (reinfors calls it through the Rust -> Python inference callback),
so we assert the root action values match (1e-9) and the search shape (max_depth, expansions, leaves,
rounds) matches exactly — the latter proving the VOI/sigma priority orders expansions identically.

Temporary cross-repo harness: imports snake_RL from a sibling checkout; skipped where absent (CI).
"""

import os
import sys
from collections import deque

import numpy as np
import pytest

_SNAKE_RL_SRC = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "snake_RL", "src"))
if os.path.isdir(_SNAKE_RL_SRC) and _SNAKE_RL_SRC not in sys.path:
    sys.path.insert(0, _SNAKE_RL_SRC)

pytest.importorskip("snake_rl.agent.model_based.selective_expectimax", reason="snake_RL oracle not available")

import reinfors  # noqa: E402
from snake_rl.agent.model_based.expectimax import DistributionalSelfPlayOpponent, UniformOpponent  # noqa: E402
from snake_rl.agent.model_based.selective_expectimax import SelectiveExpectimaxPlanner  # noqa: E402
from snake_rl.agent.shared.observation import EgocentricGridObservation  # noqa: E402
from snake_rl.agent.shared.reward import MinimalReward  # noqa: E402
from snake_rl.environment.base import RELATIVE_ACTIONS, Action, BaseSnakeEnv, Snake, WorldState  # noqa: E402
from snake_rl.environment.clean import CleanSnakeEnv  # noqa: E402

_A = BaseSnakeEnv.PLAYER_A
_B = BaseSnakeEnv.PLAYER_B
_GRID = 12
_GAMMA = 0.99
_TEMP = 1.0
_FLOOR = 0.1
_ACT2I = {Action.UP: 0, Action.DOWN: 1, Action.LEFT: 2, Action.RIGHT: 3}
_REWARD = {"food": 0.0, "loss": -10.0, "draw": -6.0, "kill": 20.0, "win": 20.0, "step": 0.0, "survival": 0.0}
_REWARD_TUPLE = (
    _REWARD["step"],
    _REWARD["food"],
    _REWARD["loss"],
    _REWARD["draw"],
    _REWARD["kill"],
    _REWARD["win"],
    _REWARD["survival"],
)


def _q_heads(obs: object, k: int) -> np.ndarray:
    """Deterministic per-head value vector (K, 3); heads disagree so sigma > 0 for K > 1."""
    s = float(np.asarray(obs).sum())
    return np.array(
        [[np.sin(s + 0.5 * h), np.cos(0.5 * s + 0.3 * h), np.sin(0.2 * s - 0.7 * h)] for h in range(k)],
        dtype=np.float64,
    )


def _food_free_state() -> WorldState:
    return WorldState(
        snakes={
            _A: Snake(deque([(6, 5), (6, 4), (6, 3)]), Action.RIGHT),
            _B: Snake(deque([(2, 8), (2, 9), (1, 9)]), Action.LEFT),
        },
        food=set(),
        grid_size=_GRID,
    )


def _oracle(opponent: str, k: int, budget: int, top_k: int, max_depth: int, beta: float) -> SelectiveExpectimaxPlanner:
    obs_builder = EgocentricGridObservation(grid_size=_GRID)

    def qf(o: object) -> np.ndarray:
        return _q_heads(o, k)

    if opponent == "uniform":
        opp: object = UniformOpponent(RELATIVE_ACTIONS)
    else:
        opp = DistributionalSelfPlayOpponent(qf, obs_builder, RELATIVE_ACTIONS, temperature=_TEMP, floor=_FLOOR)
    return SelectiveExpectimaxPlanner(
        rules=CleanSnakeEnv(grid_size=_GRID, initial_food_count=0, play_to_last=True, seed=0),
        reward_fn=MinimalReward(**_REWARD),
        obs_builder=obs_builder,
        q_values=qf,
        opp_model=opp,
        actions=RELATIVE_ACTIONS,
        gamma=_GAMMA,
        q_value_batch=lambda obss: np.stack([_q_heads(o, k) for o in obss]),
        seed=0,
        expansion_budget=budget,
        top_k=top_k,
        max_depth=max_depth,
        beta=beta,
    )


def _reinfors_env(state: WorldState) -> object:
    env = reinfors._reinfors.SnakeEnv(_GRID, 3, True, None)  # play_to_last=True, win_food_lead=None
    sa, sb = state.snakes[_A], state.snakes[_B]
    env.set_snakes(
        [tuple(c) for c in sa.body],
        _ACT2I[sa.direction],
        sa.alive,
        [tuple(c) for c in sb.body],
        _ACT2I[sb.direction],
        sb.alive,
    )
    env.set_food([tuple(c) for c in state.food])
    return env


@pytest.mark.parametrize("agent_id", [_A, _B])
@pytest.mark.parametrize("opponent", ["uniform", "distributional"])
@pytest.mark.parametrize("k", [1, 4])
@pytest.mark.parametrize(("budget", "top_k", "max_depth", "beta"), [(40, 4, 8, 1.0), (24, 4, 6, 0.0), (64, 8, 10, 0.5)])
def test_selective_search_matches_oracle(
    agent_id: str, opponent: str, k: int, budget: int, top_k: int, max_depth: int, beta: float
) -> None:
    state = _food_free_state()
    planner = _oracle(opponent, k, budget, top_k, max_depth, beta)
    oracle_values = np.asarray(planner.action_values(state, agent_id))  # (A,) for K=1 else (K, A)
    stats = planner.last_stats[0]

    agent_idx = 0 if agent_id == _A else 1
    env = _reinfors_env(state)
    values, rein_stats = env.selective_search(
        agent_idx,
        _GAMMA,
        beta,
        budget,
        top_k,
        max_depth,
        _REWARD_TUPLE,
        opponent,
        _TEMP,
        _FLOOR,
        lambda arr: np.stack([_q_heads(row, k) for row in arr]),
    )
    rein = np.asarray(values)  # (K, A)
    rein = rein[0] if k == 1 else rein

    assert np.allclose(rein, oracle_values, atol=1e-9), f"values: {rein} vs {oracle_values}"
    assert tuple(rein_stats) == (stats.max_depth, stats.expansions, stats.leaves, stats.rounds), "search shape"
