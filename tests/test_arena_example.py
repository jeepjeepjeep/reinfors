"""Keep the maintained Arena evaluation example aligned with the public API."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_arena_example_runs() -> None:
    script = Path(__file__).resolve().parents[1] / "examples" / "eval_arena.py"
    proc = subprocess.run(
        [sys.executable, str(script), "--games", "4", "--simulations", "2", "--slots", "2"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    assert "over 2 opening pairs" in proc.stdout
    assert "inf" not in proc.stdout
    assert "payoff by seat" in proc.stdout
