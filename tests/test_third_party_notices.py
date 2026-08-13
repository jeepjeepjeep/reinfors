"""The committed THIRD-PARTY-NOTICES must match regeneration exactly."""

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_notices_are_current() -> None:
    r = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "gen_third_party_notices.py"), "--check"],
        capture_output=True,
        text=True,
        cwd=REPO,
    )
    assert r.returncode == 0, r.stdout + r.stderr
