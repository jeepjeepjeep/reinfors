"""Differential parity: reinfors' selective expectimax vs snake_RL's SelectiveExpectimaxPlanner.

Single-head, UniformOpponent, food-free root — the configuration the planner's own equivalence
tests use. Both sides share the exact value function `_q` (reinfors calls it through the Rust ->
Python inference callback); we then assert the root action values match (tight tolerance) and the
search shape (max_depth, expansions, leaves, rounds) matches exactly.

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
from snake_rl.agent.model_based.expectimax import UniformOpponent  # noqa: E402
from snake_rl.agent.model_based.selective_expectimax import SelectiveExpectimaxPlanner  # noqa: E402
from snake_rl.agent.shared.observation import EgocentricGridObservation  # noqa: E402
from snake_rl.agent.shared.reward import MinimalReward  # noqa: E402
from snake_rl.environment.base import RELATIVE_ACTIONS, Action, BaseSnakeEnv, Snake, WorldState  # noqa: E402
from snake_rl.environment.clean import CleanSnakeEnv  # noqa: E402

_A = BaseSnakeEnv.PLAYER_A
_B = BaseSnakeEnv.PLAYER_B
_GRID = 12
_GAMMA = 0.99
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


def _q(obs: np.ndarray) -> np.ndarray:
    """Deterministic state-dependent value vector — shared by both planners (matches the oracle test)."""
    s = float(np.asarray(obs).sum())
    return np.array([np.sin(s), np.cos(s), np.sin(s * 0.5)], dtype=np.float64)


def _q_batch(obss: object) -> np.ndarray:
    return np.stack([_q(o) for o in obss])


def _food_free_state() -> WorldState:
    return WorldState(
        snakes={
            _A: Snake(deque([(6, 5), (6, 4), (6, 3)]), Action.RIGHT),
            _B: Snake(deque([(2, 8), (2, 9), (1, 9)]), Action.LEFT),
        },
        food=set(),
        grid_size=_GRID,
    )


def _wall_hug_state() -> WorldState:
    # A runs along the top wall heading Right: its Left (= Up) move goes off-grid, so the root expands
    # into terminal edges at the very first ply (and again at each step along the wall). B is clear.
    return WorldState(
        snakes={
            _A: Snake(deque([(0, 5), (0, 4), (0, 3)]), Action.RIGHT),
            _B: Snake(deque([(6, 8), (6, 9), (6, 10)]), Action.LEFT),
        },
        food=set(),
        grid_size=_GRID,
    )


def _dead_opp_state() -> WorldState:
    # B is already dead, so the opponent branching takes the null path ([(None, 1.0)]) throughout —
    # the depth-0 dead-opponent branch that the food-free geometry never reaches.
    return WorldState(
        snakes={
            _A: Snake(deque([(6, 5), (6, 4), (6, 3)]), Action.RIGHT),
            _B: Snake(deque([(2, 8), (2, 9)]), Action.LEFT, alive=False),
        },
        food=set(),
        grid_size=_GRID,
    )


_STATES = {"food_free": _food_free_state, "wall_hug": _wall_hug_state, "dead_opp": _dead_opp_state}


def _oracle(budget: int, top_k: int, max_depth: int, beta: float) -> SelectiveExpectimaxPlanner:
    return SelectiveExpectimaxPlanner(
        rules=CleanSnakeEnv(grid_size=_GRID, initial_food_count=0, play_to_last=True, seed=0),
        reward_fn=MinimalReward(**_REWARD),
        obs_builder=EgocentricGridObservation(grid_size=_GRID),
        q_values=_q,
        opp_model=UniformOpponent(RELATIVE_ACTIONS),
        actions=RELATIVE_ACTIONS,
        gamma=_GAMMA,
        q_value_batch=_q_batch,
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


def _rein_infer(arr: np.ndarray) -> np.ndarray:
    return np.stack([_q(row) for row in arr])  # arr: (N, 5*g*g); _q sums it -> identical to the oracle


# Single beta: at single-head sigma == 0, so the VOI term is inert and beta does nothing. The
# beta sweep only becomes meaningful once heads disagree — that lives in the ensemble parity test.
_BETA = 1.0


@pytest.mark.parametrize(
    ("scenario", "agent_id"),
    [("food_free", _A), ("food_free", _B), ("wall_hug", _A), ("dead_opp", _A)],
)
@pytest.mark.parametrize(("budget", "top_k", "max_depth"), [(40, 4, 8), (24, 4, 6), (100, 16, 4)])
def test_selective_search_matches_oracle(scenario: str, agent_id: str, budget: int, top_k: int, max_depth: int) -> None:
    state = _STATES[scenario]()
    planner = _oracle(budget, top_k, max_depth, _BETA)
    oracle_values = planner.action_values(state, agent_id)
    stats = planner.last_stats[0]

    agent_idx = 0 if agent_id == _A else 1
    env = _reinfors_env(state)
    values, rein_stats = env.selective_search(
        agent_idx, _GAMMA, _BETA, budget, top_k, max_depth, _REWARD_TUPLE, _rein_infer
    )

    assert np.allclose(values, oracle_values, atol=1e-9), f"values: {values} vs {oracle_values}"
    assert rein_stats == (stats.max_depth, stats.expansions, stats.leaves, stats.rounds), "search shape"
