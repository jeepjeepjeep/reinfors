"""A minimal smoke test for the example trainer (`examples/train_example.py`).

It does not test learning quality — only that the example still *runs against the current API*. The
example is the one reference for the Python<->Rust `infer` contract, and that contract is what's most
likely to drift, so this guards it from silent bit-rot. torch-gated (skips where torch is absent).
"""

from __future__ import annotations

import importlib.util
import os
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")


# `Any`: the example is loaded dynamically from a script path, so its members (ExampleNet, make_infer,
# …) can't be statically known — this is a smoke test that the example runs against the current API.
def _load_example() -> Any:
    path = os.path.join(os.path.dirname(__file__), "..", "examples", "train_example.py")
    spec = importlib.util.spec_from_file_location("train_example", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_G, _K, _A = 8, 2, 3


def test_net_and_infer_shapes() -> None:
    ex = _load_example()
    net = ex.ExampleNet((5, _G, _G), _A, _K)
    assert net(torch.zeros(4, 5, _G, _G)).shape == (4, _K, _A)
    out = ex.make_infer(net)(np.zeros((6, 5 * _G * _G), dtype=np.float32))
    assert out.shape == (6, _K, _A) and out.dtype == np.float64


def test_one_train_iteration_runs() -> None:
    ex = _load_example()
    net = ex.ExampleNet((5, _G, _G), _A, _K)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    engine = ex.build_engine(grid=_G, n_heads=_K, n_games=2, seed=0)
    obs, target, mask, _ = engine.collect(24, ex.make_infer(net))
    loss = ex.train_step(net, optimizer, obs, target, mask, "cpu")
    assert np.isfinite(loss)
