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


def _project(t_dist: torch.Tensor, r: list[float], boot: list[bool], gamma: float = 0.9) -> torch.Tensor:
    return ex.project_distribution(t_dist, torch.tensor(r), torch.tensor(boot), SUPPORT, gamma)


def test_projection_conserves_unit_mass() -> None:
    g = torch.Generator().manual_seed(0)
    t_dist = torch.rand(8, 3, 41, generator=g).softmax(-1)
    r = torch.rand(8, generator=g).mul(40).sub(10)
    boot = torch.rand(8, generator=g) > 0.5
    m = ex.project_distribution(t_dist, r, boot, SUPPORT, 0.9)
    assert torch.allclose(m.sum(-1), torch.ones(8, 3), atol=1e-6)


def test_exact_atom_reward_projects_one_hot() -> None:
    t_dist = torch.full((1, 1, 41), 1 / 41.0)
    m = _project(t_dist, [3.0], [False])  # r sits exactly on an atom
    k = int((SUPPORT == 3.0).nonzero())
    assert m[0, 0, k] == pytest.approx(1.0)
    assert (m > 0).sum() == 1


def test_interpolation_splits_mass_between_neighbours() -> None:
    t_dist = torch.full((1, 1, 41), 1 / 41.0)
    m = _project(t_dist, [3.3], [False])
    assert (m[0, 0] > 1e-9).sum() == 2
    assert (m[0, 0] * SUPPORT).sum() == pytest.approx(3.3, abs=1e-6)


def test_terminal_projection_ignores_the_next_distribution() -> None:
    peaked = torch.zeros(1, 1, 41)
    peaked[0, 0, -1] = 1.0
    uniform = torch.full((1, 1, 41), 1 / 41.0)
    assert torch.allclose(_project(peaked, [5.5], [False]), _project(uniform, [5.5], [False]))


def test_bootstrap_shift_keeps_the_expected_value() -> None:
    t_dist = torch.zeros(1, 1, 41)
    t_dist[0, 0, 20] = 1.0  # delta at z = 10
    m = _project(t_dist, [1.0], [True], gamma=0.9)
    assert (m[0, 0] * SUPPORT).sum() == pytest.approx(1.0 + 0.9 * 10.0, abs=1e-6)


def test_returns_clip_to_both_support_edges() -> None:
    t_dist = torch.full((1, 1, 41), 1 / 41.0)
    below = _project(t_dist, [-100.0], [False])
    above = _project(t_dist, [100.0], [False])
    assert below[0, 0, 0] == pytest.approx(1.0)
    assert above[0, 0, -1] == pytest.approx(1.0)


def test_c51_parameters_are_validated() -> None:
    with pytest.raises(ValueError, match="atoms >= 2"):
        ex.QNet(4, 1, 3, atoms=1)
    for bounds in ((0.0, 0.0), (5.0, -5.0), (float("-inf"), 5.0), (0.0, float("nan"))):
        with pytest.raises(ValueError, match="finite with v_min < v_max"):
            ex.QNet(4, 1, 3, atoms=51, v_bounds=bounds)
