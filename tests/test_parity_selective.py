"""Differential parity: reinfors' selective expectimax vs snake_RL's SelectiveExpectimaxPlanner.

Covers the matrix the oracle supports: 3 root geometries (food-free, wall-hug, dead-opponent) x
opponent in {uniform, distributional (deferred)} x heads in {1, 4} x (budget, top_k, max_depth,
beta) configs. Both sides share the exact
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


def _wall_hug_state() -> WorldState:
    # A runs along the top wall heading Right: its Left (= Up) move goes off-grid, forcing terminal
    # edges at the very first ply (and at each step along the wall). B is clear.
    return WorldState(
        snakes={
            _A: Snake(deque([(0, 5), (0, 4), (0, 3)]), Action.RIGHT),
            _B: Snake(deque([(6, 8), (6, 9), (6, 10)]), Action.LEFT),
        },
        food=set(),
        grid_size=_GRID,
    )


def _dead_opp_state() -> WorldState:
    # B is already dead, so opponent branching takes the null path throughout (no opponent rows, so
    # the distributional softmax is exercised in the degenerate dead-opponent case too).
    return WorldState(
        snakes={
            _A: Snake(deque([(6, 5), (6, 4), (6, 3)]), Action.RIGHT),
            _B: Snake(deque([(2, 8), (2, 9)]), Action.LEFT, alive=False),
        },
        food=set(),
        grid_size=_GRID,
    )


_STATES = {"food_free": _food_free_state, "wall_hug": _wall_hug_state, "dead_opp": _dead_opp_state}


def _food_state() -> WorldState:
    # Apples in both agents' forward regions, so the search eats them within the horizon (and may eat
    # the spawned replacements deeper), exercising in-tree spawning along many lines.
    return WorldState(
        snakes={
            _A: Snake(deque([(6, 5), (6, 4), (6, 3)]), Action.RIGHT),
            _B: Snake(deque([(2, 8), (2, 9), (1, 9)]), Action.LEFT),
        },
        food={(6, 6), (6, 7), (6, 8), (5, 5), (7, 5), (2, 7), (2, 6), (3, 8)},
        grid_size=_GRID,
    )


def _oracle(
    opponent: str, k: int, budget: int, top_k: int, max_depth: int, beta: float, food_samples: int = 1
) -> SelectiveExpectimaxPlanner:
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
        food_samples=food_samples,
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


# Root geometries: food-free (both agents), wall-hug (terminal edges at the first ply), and a dead
# opponent (the null opponent-branch path). The beta sweep is meaningful here because K=4 heads
# disagree (sigma > 0), so it actually steers the VOI priority.
@pytest.mark.parametrize(
    ("scenario", "agent_id"), [("food_free", _A), ("food_free", _B), ("wall_hug", _A), ("dead_opp", _A)]
)
@pytest.mark.parametrize("opponent", ["uniform", "distributional"])
@pytest.mark.parametrize("k", [1, 4])
@pytest.mark.parametrize(("budget", "top_k", "max_depth", "beta"), [(40, 4, 8, 1.0), (24, 4, 6, 0.0), (64, 8, 10, 0.5)])
def test_selective_search_matches_oracle(
    scenario: str, agent_id: str, opponent: str, k: int, budget: int, top_k: int, max_depth: int, beta: float
) -> None:
    state = _STATES[scenario]()
    planner = _oracle(opponent, k, budget, top_k, max_depth, beta)
    oracle_values = np.asarray(planner.action_values(state, agent_id))  # (A,) for K=1 else (K, A)
    stats = planner.last_stats[0]

    agent_idx = 0 if agent_id == _A else 1
    env = _reinfors_env(state)
    values, _interior, rein_stats = env.selective_search(
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


def _search_with_food(
    env: object, agent_idx: int, opponent: str, k: int, *, food_samples: int, seed: int
) -> np.ndarray:
    budget, top_k, max_depth, beta = 40, 4, 8, 1.0
    values, _interior, _stats = env.selective_search(
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
        False,  # collect_interior
        food_samples,
        seed,
    )
    rein = np.asarray(values)
    return rein[0] if k == 1 else rein


@pytest.mark.parametrize("agent_id", [_A, _B])
@pytest.mark.parametrize("opponent", ["uniform", "distributional"])
def test_search_with_food_is_reproducible_and_stochastic(agent_id: str, opponent: str) -> None:
    # In-tree apple spawning is now genuinely stochastic (a uniform-random respawn drawn from the
    # search's seeded RNG, not a deterministic first-empty belief). So: a fixed seed is exactly
    # reproducible, and two different seeds generally give different values (the spawn is really
    # sampled). Both are exact (boolean) properties — no oracle, no tolerance.
    k = 4
    agent_idx = 0 if agent_id == _A else 1
    env = _reinfors_env(_food_state())

    a = _search_with_food(env, agent_idx, opponent, k, food_samples=1, seed=7)
    a_again = _search_with_food(env, agent_idx, opponent, k, food_samples=1, seed=7)
    assert np.array_equal(a, a_again), "same seed must reproduce the search exactly"

    b = _search_with_food(env, agent_idx, opponent, k, food_samples=1, seed=8)
    assert not np.array_equal(a, b), "different seeds must sample different spawns -> different values"


@pytest.mark.parametrize("agent_id", [_A, _B])
@pytest.mark.parametrize("opponent", ["uniform", "distributional"])
def test_search_with_food_matches_oracle_in_expectation(agent_id: str, opponent: str) -> None:
    # Distributional parity: reinfors and the oracle BOTH respawn apples uniformly at random over empty
    # cells (the shared deterministic rule is gone), so bit-parity is no longer the invariant — equality
    # of the Monte-Carlo *expectation* is. Averaging many independent spawn realizations on each side,
    # the mean root action values must converge to the same vector. This is the test that would catch a
    # reinfors spawn distribution that differed from the env's/oracle's uniform-over-empty draw.
    k = 4
    budget, top_k, max_depth, beta, food_samples, m = 40, 4, 8, 1.0, 4, 120
    agent_idx = 0 if agent_id == _A else 1
    state = _food_state()

    # Oracle: one planner whose persistent RNG yields a fresh random spawn set on each `action_values`.
    planner = _oracle(opponent, k, budget, top_k, max_depth, beta, food_samples=food_samples)
    oracle_mean = np.mean([np.asarray(planner.action_values(state, agent_id)) for _ in range(m)], axis=0)

    # reinfors: vary the search seed to draw independent spawn realizations of the same search.
    env = _reinfors_env(state)
    rein_mean = np.mean(
        [_search_with_food(env, agent_idx, opponent, k, food_samples=food_samples, seed=s) for s in range(m)],
        axis=0,
    )

    assert rein_mean.shape == oracle_mean.shape
    # Both means are Monte-Carlo estimates of the same expectation; the gap is sampling noise that
    # shrinks like 1/sqrt(m*food_samples). The tolerance is loose relative to bit-parity but far tighter
    # than the value scale (q in [-1, 1], rewards up to 20), so a distribution mismatch would still fail.
    assert np.allclose(rein_mean, oracle_mean, atol=0.05), f"E[values]: {rein_mean} vs {oracle_mean}"


@pytest.mark.parametrize("opponent", ["uniform", "distributional"])
@pytest.mark.parametrize("k", [1, 4])
def test_pooled_search_many_matches_oracle(opponent: str, k: int) -> None:
    # The production path: one pooled call searching BOTH snakes of a single state, compared against
    # the oracle's own search_many (its pooled path). This guards pooled-multi == oracle *directly*,
    # rather than only transitively via (pooled == reinfors-solo) and (reinfors-solo == oracle-solo).
    budget, top_k, max_depth, beta = 40, 4, 8, 1.0
    state = _food_free_state()
    planner = _oracle(opponent, k, budget, top_k, max_depth, beta)
    oracle = planner.search_many(state, [(_A, False), (_B, False)])  # oracle's pooled search

    env = _reinfors_env(state)
    results = reinfors._reinfors.selective_search_many(
        [env, env],  # same state, searched for both agents in one pooled call
        [0, 1],
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

    for idx in range(2):
        rein = np.asarray(results[idx][0])
        rein = rein[0] if k == 1 else rein
        oracle_values = np.asarray(oracle[idx][0])
        assert np.allclose(rein, oracle_values, atol=1e-9), f"agent {idx}: {rein} vs {oracle_values}"
        s = planner.last_stats[idx]
        assert tuple(results[idx][2]) == (s.max_depth, s.expansions, s.leaves, s.rounds), f"agent {idx} search shape"


@pytest.mark.parametrize("agent_id", [_A, _B])
@pytest.mark.parametrize("opponent", ["uniform", "distributional"])
@pytest.mark.parametrize("k", [1, 4])
def test_interior_targets_match_oracle(agent_id: str, opponent: str, k: int) -> None:
    # True TreeStrap: every expanded interior MAX node below the root is emitted as (obs, values). We
    # use a food-free state so the tree is fully deterministic (no stochastic spawning) and the interior
    # targets still match the oracle bit-for-bit: same count and, node-for-node in DFS order, identical
    # observations and backed-up values. (Stochastic-spawn parity is covered in expectation above.)
    budget, top_k, max_depth, beta = 40, 4, 8, 1.0
    state = _food_free_state()
    planner = _oracle(opponent, k, budget, top_k, max_depth, beta)
    _, oracle_interior = planner.search_many(state, [(agent_id, True)])[0]

    agent_idx = 0 if agent_id == _A else 1
    env = _reinfors_env(state)
    _, interior, _ = env.selective_search(
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
        True,  # collect_interior
    )

    assert len(interior) == len(oracle_interior) > 0, f"{len(interior)} vs {len(oracle_interior)}"
    for (r_obs, r_vals), (o_obs, o_vals) in zip(interior, oracle_interior, strict=True):
        r_v = np.asarray(r_vals)
        r_v = r_v[0] if k == 1 else r_v
        assert np.allclose(r_v, np.asarray(o_vals), atol=1e-9)
        assert np.array_equal(np.asarray(r_obs, dtype=np.float32), np.asarray(o_obs, dtype=np.float32).ravel())


@pytest.mark.parametrize("outcome_weight", [0.0, 0.3, 1.0])
@pytest.mark.parametrize("k", [1, 4])
def test_blend_outcome_targets_matches_oracle(outcome_weight: float, k: int) -> None:
    # z-mixing kernel: blend the realized return into the executed action's per-head entry. Compared
    # against the oracle's EnsembleTreeStrapRunner._blend_outcome_targets on an identical trajectory.
    # treestrap imports torch; skip (rather than error) where it is absent.
    treestrap = pytest.importorskip("snake_rl.agent.model_based.treestrap")
    EnsembleTrajectoryStep = treestrap.EnsembleTrajectoryStep
    EnsembleTreeStrapRunner = treestrap.EnsembleTreeStrapRunner

    rng = np.random.default_rng(0)
    t, a = 6, 3
    values = rng.standard_normal((t, k, a))
    actions = [int(i % a) for i in range(t)]
    rewards = [float(r) for r in rng.standard_normal(t)]
    tail = rng.standard_normal(k)

    oracle_traj = [EnsembleTrajectoryStep(np.zeros(1), values[i], actions[i], rewards[i]) for i in range(t)]
    oracle_targets, _gap = EnsembleTreeStrapRunner._blend_outcome_targets(oracle_traj, _GAMMA, outcome_weight, tail)
    oracle_arr = np.stack(oracle_targets)  # (T, K, A)

    rein = np.asarray(
        reinfors._reinfors.blend_outcome_targets(values, actions, rewards, _GAMMA, outcome_weight, list(tail))
    )
    assert np.allclose(rein, oracle_arr, atol=1e-12), f"{rein} vs {oracle_arr}"


def _infer_k2(arr: object) -> np.ndarray:
    return np.stack([_q_heads(row, 2) for row in arr])


@pytest.mark.parametrize("survival", [0.25, -0.1])
def test_survival_reward_matches_oracle_minimal_reward(survival: float) -> None:
    # The truncation survival bonus reaches z-mixing by exactly snake_RL's MinimalReward contribution.
    # A 1-tick (food-free, surviving) episode with outcome_weight=1 makes the executed action's target
    # equal the realized return; two engines differing only in `survival` differ by MinimalReward's
    # survived-vs-not delta, in the executed action's entry alone.
    from snake_rl.agent.shared.reward import MinimalReward
    from snake_rl.environment.base import StepEvent

    mr = MinimalReward(step=0.0, food=0.0, loss=-10.0, draw=-6.0, kill=20.0, win=20.0, survival=survival)
    delta = mr(StepEvent(survived_to_max_ticks=True)) - mr(StepEvent())

    def engine(surv: float) -> object:
        return reinfors.Engine(
            reinfors.games.Snake(
                grid_size=_GRID,
                initial_length=3,
                food=0,  # food-free
                play_to_last=False,
                win_food_lead=None,
                reward=reinfors.Reward(step=0.0, food=0.0, loss=-10.0, draw=-6.0, kill=20.0, win=20.0, survival=surv),
            ),
            reinfors.policies.SelectiveExpectimax(
                expansion_budget=24,
                top_k=4,
                max_depth=6,
                beta=1.0,
                food_samples=1,
                n_heads=2,
                epsilon=0.1,
                opponent="uniform",
                opp_temperature=_TEMP,
                opp_floor=_FLOOR,
            ),
            reinfors.learners.TreeStrap(gamma=_GAMMA, outcome_weight=1.0, bootstrap_p=1.0, interior_targets=False),
            n_games=2,
            max_ticks=1,
            seed=0,
        )

    _, t0, _, _ = engine(0.0).collect(2, _infer_k2)
    _, ts, _, _ = engine(survival).collect(2, _infer_k2)
    diff = np.asarray(ts) - np.asarray(t0)
    assert diff.shape[0] >= 2
    for m in range(diff.shape[0]):
        for h in range(2):
            changed = np.flatnonzero(np.abs(diff[m, h]) > 1e-9)
            assert changed.size == 1, "only the executed action's target should move"
            assert np.isclose(diff[m, h, changed[0]], delta, atol=1e-9)
