"""Numerical tests for the hold'em example's C51 machinery (`examples/train_dqn_holdem.py`):
the categorical projection and the support/atom validation. torch-gated like the other
example tests.
"""

from __future__ import annotations

import importlib.util
import os
from typing import Any

import pytest

torch = pytest.importorskip("torch")


def _load_example() -> Any:
    path = os.path.join(os.path.dirname(__file__), "..", "examples", "train_dqn_holdem.py")
    spec = importlib.util.spec_from_file_location("train_dqn_holdem", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ex = _load_example()
SUPPORT = torch.linspace(-10.0, 30.0, 41)  # asymmetric, dz = 1


def _project(t_dist: torch.Tensor, r: list[float], disc: list[float]) -> torch.Tensor:
    return ex.project_distribution(t_dist, torch.tensor(r), torch.tensor(disc), SUPPORT)


def test_projection_conserves_unit_mass() -> None:
    g = torch.Generator().manual_seed(0)
    t_dist = torch.rand(8, 3, 41, generator=g).softmax(-1)
    r = torch.rand(8, generator=g).mul(40).sub(10)
    disc = (torch.rand(8, generator=g) > 0.5).float() * 0.9
    m = ex.project_distribution(t_dist, r, disc, SUPPORT)
    assert torch.allclose(m.sum(-1), torch.ones(8, 3), atol=1e-6)


def test_exact_atom_reward_projects_one_hot() -> None:
    t_dist = torch.full((1, 1, 41), 1 / 41.0)
    m = _project(t_dist, [3.0], [0.0])  # r sits exactly on an atom
    k = int((SUPPORT == 3.0).nonzero())
    assert m[0, 0, k] == pytest.approx(1.0)
    assert (m > 0).sum() == 1


def test_interpolation_splits_mass_between_neighbours() -> None:
    t_dist = torch.full((1, 1, 41), 1 / 41.0)
    m = _project(t_dist, [3.3], [0.0])
    assert (m[0, 0] > 1e-9).sum() == 2
    assert (m[0, 0] * SUPPORT).sum() == pytest.approx(3.3, abs=1e-6)


def test_terminal_projection_ignores_the_next_distribution() -> None:
    peaked = torch.zeros(1, 1, 41)
    peaked[0, 0, -1] = 1.0
    uniform = torch.full((1, 1, 41), 1 / 41.0)
    assert torch.allclose(_project(peaked, [5.5], [0.0]), _project(uniform, [5.5], [0.0]))


def test_bootstrap_shift_keeps_the_expected_value() -> None:
    t_dist = torch.zeros(1, 1, 41)
    t_dist[0, 0, 20] = 1.0  # delta at z = 10
    m = _project(t_dist, [1.0], [0.9])
    assert (m[0, 0] * SUPPORT).sum() == pytest.approx(1.0 + 0.9 * 10.0, abs=1e-6)


def test_returns_clip_to_both_support_edges() -> None:
    t_dist = torch.full((1, 1, 41), 1 / 41.0)
    below = _project(t_dist, [-100.0], [0.0])
    above = _project(t_dist, [100.0], [0.0])
    assert below[0, 0, 0] == pytest.approx(1.0)
    assert above[0, 0, -1] == pytest.approx(1.0)


def test_c51_parameters_are_validated() -> None:
    with pytest.raises(ValueError, match="atoms >= 2"):
        ex.QNet(4, 1, 3, atoms=1)
    for bounds in ((0.0, 0.0), (5.0, -5.0), (float("-inf"), 5.0), (0.0, float("nan"))):
        with pytest.raises(ValueError, match="finite with v_min < v_max"):
            ex.QNet(4, 1, 3, atoms=51, v_bounds=bounds)


def test_nstep_discounts_expose_shortened_windows() -> None:
    import numpy as np
    import reinfors as rf

    engine = rf.Engine(
        rf.games.GridWorld(),
        rf.Reward(goal=1.0, step=-0.01),
        rf.policies.EpsilonGreedyQ(epsilon=0.5),
        rf.learners.Dqn(n_step=3, gamma=0.5),
        n_games=2,
        seed=1,
    )
    batch = engine.collect(n_records=64, infer=lambda o: torch.zeros(len(o), 1, 4).numpy())
    d = np.asarray(batch.discounts)
    assert set(np.round(d[d > 0], 6)) <= {0.5, 0.25, 0.125}, "discounts are gamma^k, k <= n"
    assert ((d == 0) == ~np.asarray(batch.can_bootstrap)).all()
    assert "n_step" in engine.resolved_config()["learner"]
