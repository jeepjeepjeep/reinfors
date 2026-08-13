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


def test_wheel_targets_cover_the_release_matrix() -> None:
    import importlib.util
    import re

    spec = importlib.util.spec_from_file_location("gen_notices", REPO / "scripts" / "gen_third_party_notices.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    triple = {
        ("ubuntu", "x86_64"): "x86_64-unknown-linux-gnu",
        ("ubuntu", "aarch64"): "aarch64-unknown-linux-gnu",
        ("macos", "x86_64"): "x86_64-apple-darwin",
        ("macos", "aarch64"): "aarch64-apple-darwin",
        ("windows", "x64"): "x86_64-pc-windows-msvc",
    }
    workflow = (REPO / ".github" / "workflows" / "release.yml").read_text()
    entries = re.findall(r"\{ os: ([a-z]+)[^,]*, target: (\w+) \}", workflow)
    assert entries, "release matrix not found"
    release_triples = {triple[e] for e in entries}  # KeyError = unmapped matrix entry
    assert release_triples == set(mod.WHEEL_TARGETS)
