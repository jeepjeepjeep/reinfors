"""Check local Markdown links and fragments in README.md and docs/.

External URLs are intentionally left to their owners; MkDocs performs the strict site build.
"""

from __future__ import annotations

import re
import sys
import urllib.parse
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r"(?<!!)\[[^]]*]\(([^)]+)\)")
HEADING = re.compile(r"^#{1,6}\s+(.+?)\s*$")


def slug(text: str) -> str:
    """Approximate Python-Markdown's heading slug for the ASCII headings used here."""
    text = re.sub(r"<[^>]+>", "", text).strip().lower()
    text = re.sub(r"[^\w\- ]", "", text)
    return re.sub(r"[-\s]+", "-", text).strip("-")


def anchors(path: Path) -> set[str]:
    seen: Counter[str] = Counter()
    result: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        match = HEADING.match(line)
        if not match:
            continue
        base = slug(match.group(1))
        index = seen[base]
        seen[base] += 1
        result.add(base if index == 0 else f"{base}_{index}")
    return result


def markdown_files() -> list[Path]:
    return [ROOT / "README.md", *sorted((ROOT / "docs").rglob("*.md"))]


def main() -> int:
    errors: list[str] = []
    cache: dict[Path, set[str]] = {}
    for source in markdown_files():
        text = source.read_text(encoding="utf-8")
        for raw in LINK.findall(text):
            target = raw.strip().strip("<>").split(maxsplit=1)[0]
            parsed = urllib.parse.urlsplit(target)
            if parsed.scheme or parsed.netloc:
                continue
            target_path = source if not parsed.path else (source.parent / urllib.parse.unquote(parsed.path)).resolve()
            if target_path.is_dir():
                target_path /= "index.md"
            if not target_path.exists():
                errors.append(f"{source.relative_to(ROOT)}: missing link target {target!r}")
                continue
            if parsed.fragment and target_path.suffix.lower() == ".md":
                available = cache.setdefault(target_path, anchors(target_path))
                fragment = urllib.parse.unquote(parsed.fragment)
                if fragment not in available:
                    errors.append(
                        f"{source.relative_to(ROOT)}: missing fragment #{fragment} in {target_path.relative_to(ROOT)}"
                    )
    if errors:
        print("Documentation link errors:")
        print("\n".join(f"  {error}" for error in errors))
        return 1
    print(f"Checked local links in {len(markdown_files())} Markdown files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
