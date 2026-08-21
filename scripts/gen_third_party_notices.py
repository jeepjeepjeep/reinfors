"""Regenerate THIRD-PARTY-NOTICES from the wheel's runtime dependency closure.

Closure = union of `cargo tree --edges normal` for every wheel target, so
proc-macro/build dependencies and non-shipped-target packages are excluded.
Crates whose packaged archive omits its license text must have a reviewed
override in scripts/license-overrides/, or generation fails.

    python scripts/gen_third_party_notices.py          # rewrite the file
    python scripts/gen_third_party_notices.py --check  # fail on drift (CI/test)
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "THIRD-PARTY-NOTICES"
# must cover every wheel in .github/workflows/release.yml (tied by test)
WHEEL_TARGETS = [
    "aarch64-apple-darwin",
    "aarch64-unknown-linux-gnu",
    "x86_64-apple-darwin",
    "x86_64-unknown-linux-gnu",
    "x86_64-pc-windows-msvc",
]
OVERRIDES = {
    # crates.io archives omit the workspace-level LICENSE for these
    "cozy-chess": REPO / "scripts" / "license-overrides" / "cozy-chess.LICENSE",
    "cozy-chess-types": REPO / "scripts" / "license-overrides" / "cozy-chess.LICENSE",
    "parry2d": REPO / "scripts" / "license-overrides" / "parry2d.LICENSE",
    "profiling": REPO / "scripts" / "license-overrides" / "profiling.LICENSE",
    "rapier2d": REPO / "scripts" / "license-overrides" / "rapier2d.LICENSE",
}


def runtime_closure() -> set[str]:
    names: set[str] = set()
    for target in WHEEL_TARGETS:
        out = subprocess.check_output(
            [
                "cargo",
                "tree",
                "-p",
                "reinfors-py",
                "--edges",
                "normal,no-proc-macro",
                "--target",
                target,
                "--prefix",
                "none",
                "--format",
                "{p}",
            ],
            cwd=REPO,
            text=True,
        )
        for line in out.splitlines():
            if line.strip():
                names.add(line.split()[0])
    return names


def build() -> str:
    meta = json.loads(subprocess.check_output(["cargo", "metadata", "--format-version", "1"], cwd=REPO))
    members = set(meta["workspace_members"])
    wanted = runtime_closure()
    packages = sorted(
        (p for p in meta["packages"] if p["id"] not in members and p["name"] in wanted),
        key=lambda p: (p["name"], p["version"]),
    )
    out = [
        "Third-party notices for reinfors",
        "",
        "The compiled extension statically contains the following Rust components.",
        "License texts follow; identical texts are printed once and referenced.",
        "",
    ]
    seen: dict[str, str] = {}
    missing: list[str] = []
    for p in packages:
        crate_dir = Path(p["manifest_path"]).parent
        files = sorted(
            f for pat in ("LICENSE*", "LICENCE*", "COPYING*", "NOTICE*") for f in crate_dir.glob(pat) if f.is_file()
        )
        if not files:
            override = OVERRIDES.get(p["name"])
            if override is None:
                missing.append(f"{p['name']} {p['version']}")
                continue
            files = [override]
        header = f"{p['name']} {p['version']} — {p.get('license') or 'see files'}"
        if p.get("repository"):
            header += f" — {p['repository']}"
        out.append("=" * 78)
        out.append(header)
        for f in files:
            text = f.read_text(errors="replace").strip()
            digest = hashlib.sha256(text.encode()).hexdigest()
            if digest in seen:
                out.append(f"[{f.name}: text identical to {seen[digest]}]")
            else:
                seen[digest] = f"{p['name']} {p['version']}/{f.name}"
                out.append(f"--- {f.name} ---")
                out.append(text)
        out.append("")
    if missing:
        sys.exit("no license text found and no reviewed override for: " + ", ".join(missing))
    while out and out[-1] == "":
        out.pop()
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    content = build()
    if "--check" in sys.argv[1:]:
        if not OUT.exists() or OUT.read_text() != content:
            sys.exit("THIRD-PARTY-NOTICES is stale — rerun the generator")
        print("THIRD-PARTY-NOTICES up to date")
    else:
        OUT.write_text(content)
        print(f"wrote {OUT.name}")
