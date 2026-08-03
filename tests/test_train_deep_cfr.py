"""Smoke test for the Deep CFR reference trainer (`scripts/train_deep_cfr.py`) — the
`test_example.py` pattern: not a learning-quality test, but a guard that the reference keeps
running against the current API (the per-player infer list, the CSR batch fields, the
exploitability instrument). torch-gated.
"""

from __future__ import annotations

import importlib.util
import os
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")


def _load() -> Any:
    path = os.path.join(os.path.dirname(__file__), "..", "scripts", "train_deep_cfr.py")
    spec = importlib.util.spec_from_file_location("train_deep_cfr", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reservoir_densifies_csr_and_caps() -> None:
    ex = _load()
    reservoir = ex.Reservoir(capacity=4, dim=3, n_actions=2, seed=0)
    obs = np.arange(18, dtype=np.float32).reshape(6, 3)
    offsets = np.array([0, 1, 3, 4, 6, 7, 8])
    ids = np.array([1, 0, 1, 0, 0, 1, 1, 0])
    values = np.arange(8, dtype=np.float64)
    iterations = np.arange(1, 7)
    reservoir.add_csr(obs, offsets, ids, values, iterations)
    assert reservoir.size == 4 and reservoir.seen == 6, "reservoir caps at capacity"
    sampled = reservoir.sample(8, np.random.default_rng(0))
    assert sampled[0].shape == (8, 3) and sampled[1].shape == (8, 2)
    # The first inserted row survives densification: id 1 -> value 0.0, mask [False, True].
    assert reservoir.mask[0].tolist() == [False, True]
    assert reservoir.values[0][1] == 0.0


def test_infer_adapters_have_the_contract_shapes() -> None:
    ex = _load()
    net = ex.Mlp(6, 2, 16)
    out = ex.make_infer(net, "cpu")(np.zeros((5, 6), dtype=np.float32))
    assert out.shape == (5, 2) and out.dtype == np.float32  # native f32; solver widens exactly
    probs = ex.make_policy_infer(net, "cpu")(np.zeros((5, 6), dtype=np.float32))
    assert probs.shape == (5, 2)
    assert np.allclose(probs.sum(axis=1), 1.0), "policy adapter emits probabilities"


def test_three_iterations_end_to_end_on_kuhn(monkeypatch: pytest.MonkeyPatch) -> None:
    ex = _load()
    monkeypatch.setattr(
        "sys.argv",
        [
            "train_deep_cfr.py",
            "--game=kuhn_poker",
            "--iterations=3",
            "--traversals=32",
            "--train-steps=20",
            "--policy-train-steps=40",
            "--eval-every=0",
            "--width=32",
        ],
    )
    ex.main()  # runs the full loop incl. the final exploitability + strategy report
