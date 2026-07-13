"""Smoke test for `scripts/benchmark_vs.py`: the reinfors backend yields positive throughput and the
optional (Pgx / OpenSpiel) backends degrade gracefully when their deps are absent. Guards the harness
and the reinfors path against API drift — the Pgx/OpenSpiel paths are validated by running the script on
a machine with those installed, not here."""

from __future__ import annotations

import importlib.util
import os
import sys
from typing import Any


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
