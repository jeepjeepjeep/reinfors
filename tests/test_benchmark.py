"""Smoke test for `scripts/benchmark.py`: the harness imports and yields positive throughput on a tiny
config. Guards the benchmark against API drift (it drives `Engine`/`Env` like a consumer) — not a
performance assertion. Numpy-only, so it runs in the normal test job."""

from __future__ import annotations

import importlib.util
import os
import sys
from typing import Any


def _load_benchmark() -> Any:
    path = os.path.join(os.path.dirname(__file__), "..", "scripts", "benchmark.py")
    spec = importlib.util.spec_from_file_location("benchmark", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # so the module's @dataclass string annotations resolve
    spec.loader.exec_module(module)
    return module


def test_collect_throughput_positive_across_modes_and_infer() -> None:
    bench = _load_benchmark()
    # reactive + zeros infer, and search + a synthetic net, on two games — exercises both engine paths.
    assert bench.bench_collect("gridworld", grid=5, mode="reactive", n_games=2, records=16, repeats=1, hidden=0) > 0
    assert bench.bench_collect("snake", grid=6, mode="search", n_games=2, records=16, repeats=1, hidden=8) > 0


def test_env_stepping_throughput_positive() -> None:
    bench = _load_benchmark()
    assert bench.bench_env_steps("connect4", grid=0, steps=100, repeats=1) > 0
