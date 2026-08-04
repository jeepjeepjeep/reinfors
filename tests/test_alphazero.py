"""The AlphaZero family through the Python Engine: the tuple infer contract, (obs, π, z) batches,
noise/temperature-driven self-play diversity under seeded determinism, and pairing/validation errors.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
import reinfors as rf

_A = 7  # connect4 columns


def _uniform_infer(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # The AlphaZero contract: (policy_logits (N, A), values (N,)) — no dummy heads, one forward.
    return np.zeros((arr.shape[0], _A)), np.zeros(arr.shape[0])


def _engine(
    seed: int = 0,
    *,
    n_games: int = 2,
    num_simulations: int = 24,
    noise_epsilon: float = 0.25,
    temperature: float = 1.0,
    temperature_drop: int | None = 8,
    gamma: float = 1.0,
) -> rf.Engine:
    return rf.Engine(
        rf.games.Connect4(),
        rf.Reward(win=1.0, loss=-1.0),
        rf.policies.AlphaZero(
            num_simulations=num_simulations,
            c_puct=1.5,
            noise=(rf.noise.Dirichlet(epsilon=noise_epsilon, alpha=0.3) if noise_epsilon > 0 else None),
            temperature=temperature,
            temperature_drop=temperature_drop,
        ),
        rf.learners.AlphaZero(gamma=gamma),
        n_games=n_games,
        seed=seed,
    )


def test_collect_returns_named_alphazero_batch() -> None:
    batch = _engine().collect(50, _uniform_infer)
    assert isinstance(batch, rf._reinfors.AlphaZeroBatch)
    obs, pi, z, w, telemetry = batch  # positional unpacking mirrors the named fields
    assert len(batch) == 5
    m = obs.shape[0]
    assert m >= 50
    assert obs.shape == (m, 2 * 6 * 7) and obs.dtype == np.float32
    assert pi.shape == (m, _A) and pi.dtype == np.float64
    assert z.shape == (m,) and z.dtype == np.float64
    assert w.shape == (m,) and w.dtype == np.float64
    assert (w == 1.0).all()  # 2p sequential: every row is a real decision
    assert np.array_equal(batch.obs, obs)
    assert np.array_equal(batch.policy_targets, pi)
    assert np.array_equal(batch.value_targets, z)
    assert np.array_equal(batch.policy_weights, w)
    assert batch.telemetry is telemetry
    assert "episodes" in batch.telemetry and batch.telemetry["decisions"] > 0


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("output", ["logits", "values"])
def test_collect_rejects_non_finite_inference_outputs(bad_value: float, output: str) -> None:
    def infer(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        logits = np.zeros((arr.shape[0], _A))
        values = np.zeros(arr.shape[0])
        (logits if output == "logits" else values).flat[0] = bad_value
        return logits, values

    with pytest.raises(ValueError, match="finite"):
        _engine(num_simulations=2).collect(1, infer)


def test_policy_targets_are_distributions() -> None:
    _, pi, _, _, _ = _engine().collect(80, _uniform_infer)
    assert (pi >= 0.0).all()
    np.testing.assert_allclose(pi.sum(axis=1), 1.0, atol=1e-12)


def test_value_targets_are_signed_outcomes() -> None:
    # gamma=1, win/loss=±1, no draws in short self-play: every z is ±1, and both signs appear
    # (each finished game contributes a winner's and a loser's trajectory).
    _, _, z, _, _ = _engine().collect(120, _uniform_infer)
    assert np.isin(z, (-1.0, 1.0)).all()
    assert (z == 1.0).any() and (z == -1.0).any()


def test_collect_is_deterministic_per_seed_with_noise_on() -> None:
    b1 = _engine(7).collect(60, _uniform_infer)
    b2 = _engine(7).collect(60, _uniform_infer)
    assert isinstance(b1, rf._reinfors.AlphaZeroBatch) and isinstance(b2, rf._reinfors.AlphaZeroBatch)
    assert np.array_equal(b1.obs, b2.obs)
    assert np.array_equal(b1.policy_targets, b2.policy_targets)
    assert np.array_equal(b1.value_targets, b2.value_targets)
    o3 = _engine(8).collect(60, _uniform_infer).obs
    assert not np.array_equal(b1.obs, o3)


def test_noise_and_temperature_diversify_self_play() -> None:
    # A fixed net + fixed start would replay one game forever; noise + the opening temperature must
    # produce distinct games within a collect.
    _, _, _, _, tel = _engine(0, n_games=1).collect(200, _uniform_infer)
    games = {(tuple(r), length) for r, length, _s in tel["episodes"]}
    assert len(games) > 1


def test_greedy_noise_free_replays_one_game() -> None:
    # The degenerate regime the knobs exist to escape — pin it so the mechanism stays honest.
    _, _, _, _, tel = _engine(0, n_games=1, noise_epsilon=0.0, temperature=0.0).collect(200, _uniform_infer)
    games = {(tuple(r), length) for r, length, _s in tel["episodes"]}
    assert len(games) == 1


def test_gridworld_composes() -> None:
    # Single-agent sequential game: same family, 4 actions.
    def infer(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros((arr.shape[0], 4)), np.zeros(arr.shape[0])

    engine = rf.Engine(
        rf.games.GridWorld(size=5, goal_row=0, goal_col=1, max_ticks=20),
        rf.Reward(goal=1.0),
        rf.policies.AlphaZero(num_simulations=8),
        rf.learners.AlphaZero(gamma=0.9),
        n_games=2,
        seed=0,
    )
    obs, pi, z, _, _ = engine.collect(30, infer)
    assert pi.shape[1] == 4 and obs.shape[0] == z.shape[0] >= 30


def test_alphazero_trains_on_simultaneous_stochastic_snake() -> None:
    # Snake composes the DUCT (simultaneous) and declared-chance tree capabilities: an engine-level
    # collect must run and produce per-agent AZ targets: pi over 3 actions, finite z (snake's food
    # rewards accumulate at gamma=1, so z is a return, not a bounded outcome).
    def infer(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros((arr.shape[0], 3)), np.zeros(arr.shape[0])

    engine = rf.Engine(
        rf.games.Snake(grid_size=8, max_ticks=40),
        rf.Reward(food=1.0, loss=-1.0),
        rf.policies.AlphaZero(num_simulations=12, chance=rf.chance_modes.Committed(samples=2)),
        rf.learners.AlphaZero(),
        n_games=2,
        seed=0,
    )
    batch = engine.collect(60, infer)
    assert isinstance(batch, rf._reinfors.AlphaZeroBatch)
    assert batch.obs.shape[0] >= 60
    assert batch.policy_targets.shape[1] == 3
    rows = batch.policy_targets.sum(axis=1)
    assert np.allclose(rows[rows > 0], 1.0)  # visit distributions normalize
    assert np.all(np.isfinite(batch.value_targets))
    assert batch.value_targets.min() >= -1.0 - 1e-9  # a death is the worst single outcome


def test_sequential_backup_kwarg() -> None:
    # The negamax-deletion measurement seam: "maxn" forces the vector backup at 2 agents and
    # emits value-only (weight-0) rows for the non-mover; "auto" keeps mover-only supervision.
    def collect(backup: str) -> object:
        return rf.Engine(
            rf.games.Connect4(),
            rf.Reward(win=1.0, loss=-1.0),
            rf.policies.AlphaZero(num_simulations=12, sequential_backup=backup),
            rf.learners.AlphaZero(),
            n_games=2,
            seed=2,
        ).collect(40, _uniform_infer)

    auto = collect("auto")
    assert (auto.policy_weights == 1.0).all()
    maxn = collect("maxn")
    w = maxn.policy_weights
    assert (w == 0.0).sum() == (w == 1.0).sum() > 0  # one value row per decision at N=2
    # zero-sum z on the value rows: the non-mover's return is the mover's negated at gamma 1
    cfg = rf.Engine(
        rf.games.Connect4(),
        rf.Reward(win=1.0, loss=-1.0),
        rf.policies.AlphaZero(num_simulations=12, sequential_backup="maxn"),
        rf.learners.AlphaZero(),
        n_games=1,
        seed=3,
    ).resolved_config()
    assert cfg["policy"]["sequential_backup"] == "maxn"
    rebuilt = rf.engine_from_config(cfg)
    assert rebuilt.resolved_config() == cfg
    with pytest.raises(ValueError, match="sequential_backup"):
        rf.policies.AlphaZero(sequential_backup="negamax")


def test_noise_scope_kwarg() -> None:
    rf.policies.AlphaZero(noise=rf.noise.Dirichlet(scope="requester"))
    rf.policies.AlphaZero(noise=rf.noise.Dirichlet(scope="all"))
    rf.policies.AlphaZero(noise=None)  # honestly off — no epsilon sentinel
    with pytest.raises(ValueError, match="scope"):
        rf.noise.Dirichlet(scope="both")  # the pre-rename name is gone, not aliased


@pytest.mark.parametrize(
    ("policy", "learner"),
    [
        (rf.policies.AlphaZero(), rf.learners.TreeStrap()),
        (rf.policies.AlphaZero(), rf.learners.Dqn()),
        (rf.policies.Mcts(), rf.learners.AlphaZero()),
        (rf.policies.EpsilonGreedyQ(), rf.learners.AlphaZero()),
    ],
)
def test_rejects_mismatched_pairings(policy: rf._reinfors.PolicyHandle, learner: rf._reinfors.LearnerHandle) -> None:
    with pytest.raises(ValueError, match="incompatible"):
        rf.Engine(rf.games.Connect4(), None, policy, learner, n_games=1)


@pytest.mark.parametrize(
    "bad",
    [
        {"num_simulations": 1},  # sim 1 evaluates the root; π needs at least one more
        {"noise_epsilon": 1.5},
        {"temperature": -1.0},
    ],
)
def test_rejects_degenerate_params(bad: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        _engine(0, **bad)


def test_rejects_bad_noise_alpha_and_c_puct() -> None:
    with pytest.raises(ValueError):
        rf.noise.Dirichlet(alpha=0.0)  # validated at the handle now
    with pytest.raises(ValueError):
        rf.Engine(
            rf.games.Connect4(),
            None,
            rf.policies.AlphaZero(c_puct=-1.0),
            rf.learners.AlphaZero(),
            n_games=1,
        )


@pytest.mark.parametrize(
    "bad_infer",
    [
        lambda arr: np.zeros((arr.shape[0], 1, _A)),  # old value contract, not a tuple
        lambda arr: (np.zeros((arr.shape[0], _A + 1)), np.zeros(arr.shape[0])),  # wrong A
        lambda arr: (np.zeros((arr.shape[0], _A)), np.zeros(arr.shape[0] + 1)),  # wrong N
    ],
)
def test_rejects_malformed_infer_output(bad_infer: Callable[[np.ndarray], object]) -> None:
    with pytest.raises((ValueError, TypeError)):
        _engine().collect(10, bad_infer)


def test_rejects_wrong_alphazero_dtype_with_observed_arrays() -> None:
    def bad_infer(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros((arr.shape[0], _A), dtype=np.float32), np.zeros(arr.shape[0])

    with pytest.raises(TypeError) as error:
        _engine().collect(10, bad_infer)
    message = str(error.value)
    assert "policy_logits as a float64 NumPy array with rank 2" in message
    assert "ndarray(dtype=float32" in message
    assert "ndarray(dtype=float64" in message


def test_make_by_name() -> None:
    # The config-driven path: both handles are name-addressable.
    engine = rf.Engine(
        rf.games.Connect4(),
        None,
        rf.make_policy("alphazero", num_simulations=8),
        rf.make_learner("alphazero"),
        n_games=1,
        seed=0,
    )
    obs, _, _, _, _ = engine.collect(10, _uniform_infer)
    assert obs.shape[0] >= 10
