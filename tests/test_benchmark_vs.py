"""Smoke test for `scripts/benchmark_vs.py`: the reinfors backend yields positive throughput and the
optional (Pgx / OpenSpiel) backends degrade gracefully when their deps are absent. Guards the harness
and the reinfors path against API drift — the Pgx/OpenSpiel paths are validated by running the script on
a machine with those installed, not here."""

from __future__ import annotations

import importlib.util
import os
import sys
from typing import Any

import numpy as np
import pytest


def _load() -> Any:
    path = os.path.join(os.path.dirname(__file__), "..", "scripts", "benchmark_vs.py")
    spec = importlib.util.spec_from_file_location("benchmark_vs", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # so the module's class annotations resolve
    spec.loader.exec_module(module)
    return module


def test_reinfors_backend_throughput_positive() -> None:
    mod = _load()
    rf_backend = mod.ReinforsBackend()
    ok, _ = rf_backend.available()
    assert ok
    assert rf_backend.raw_step(batch=2, steps=20, repeats=1) > 0
    assert rf_backend.search(budget=16, decisions=8, repeats=1) > 0


def test_optional_backends_report_availability_without_crashing() -> None:
    mod = _load()
    for backend in (mod.PgxBackend(), mod.OpenSpielBackend()):
        ok, detail = backend.available()  # absent here -> (False, reason); must not raise
        assert isinstance(ok, bool) and isinstance(detail, str)


def test_shared_net_value_is_bounded() -> None:
    mod = _load()
    v = mod._SHARED_NET.value(np.zeros((3, 42), dtype=np.float32))
    assert v.shape == (3,) and np.all(np.abs(v) < 1.0)  # tanh output


def test_shared_net_encoding_matches_across_frameworks() -> None:
    # The whole point of a *shared* net: reinfors' and OpenSpiel's native connect4 encodings must yield
    # the SAME canonical board for the same position, so the net is one value function, not two.
    pyspiel = pytest.importorskip("pyspiel")
    import reinfors as rf

    mod = _load()
    state = pyspiel.load_game("connect_four").new_initial_state()
    env = rf.Env(rf.games.Connect4())
    env.reset()
    for col in [3, 3, 4, 2, 4, 5]:
        env.step({env.active_agents()[0]: col})
        state.apply_action(col)
    obs = env.observe(env.active_agents()[0]).reshape(1, -1).astype(np.float64)  # (2,6,7) -> (1,84)
    assert np.array_equal(mod.board_from_reinfors(obs), mod.board_from_openspiel(state))
