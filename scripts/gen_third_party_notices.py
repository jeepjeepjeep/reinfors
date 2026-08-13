"""Regenerate THIRD-PARTY-NOTICES from the cargo dependency tree.

Release gate: rerun after any dependency change; the file ships in the wheel via
pyproject `license-files`.
"""

import hashlib
import json
import subprocess
from pathlib import Path

meta = json.loads(subprocess.check_output(["cargo", "metadata", "--format-version", "1"]))
members = set(meta["workspace_members"])
packages = [p for p in meta["packages"] if p["id"] not in members]
packages.sort(key=lambda p: (p["name"], p["version"]))

out = [
    "Third-party notices for reinfors\n",
    "This distribution statically contains the following Rust components.\n",
    "License texts follow; identical texts are printed once and referenced.\n",
]
seen: dict[str, str] = {}
for p in packages:
    crate_dir = Path(p["manifest_path"]).parent
    files = sorted(
        f for pat in ("LICENSE*", "LICENCE*", "COPYING*", "NOTICE*") for f in crate_dir.glob(pat) if f.is_file()
    )
    header = f"{p['name']} {p['version']} — {p.get('license') or 'see repository'}"
    if p.get("repository"):
        header += f" — {p['repository']}"
    out.append("=" * 78)
    out.append(header)
    if not files:
        out.append("(no license file shipped in the crate; see repository above)")
        continue
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

Path("THIRD-PARTY-NOTICES").write_text("\n".join(out) + "\n")
print(f"{len(packages)} components; {len(seen)} unique license texts")
