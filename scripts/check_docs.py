"""Check local links, fragments, and maintained code examples in README.md and docs/.

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
SYNCED_EXAMPLES = [(Path("docs/guides/training.md"), Path("examples/train_gridworld.py"))]


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


def first_python_block(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    marker = "```python\n"
    if marker not in text:
        raise ValueError("no Python code block")
    block = text.split(marker, 1)[1]
    if "\n```" not in block:
        raise ValueError("unclosed Python code block")
    return block.split("\n```", 1)[0] + "\n"


def main() -> int:
    errors: list[str] = []
    cache: dict[Path, set[str]] = {}
    sources = markdown_files()
    markdown_in_html = False
    for source in sources:
        text = source.read_text(encoding="utf-8")
        markdown_in_html |= bool(re.search(r"<[^>]+\smarkdown(?:\s|=|>)", text))
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
    for guide_rel, example_rel in SYNCED_EXAMPLES:
        guide = ROOT / guide_rel
        example = ROOT / example_rel
        try:
            documented = first_python_block(guide)
        except ValueError as error:
            errors.append(f"{guide_rel}: cannot read maintained example: {error}")
            continue
        if documented != example.read_text(encoding="utf-8"):
            errors.append(f"{guide_rel}: Python block differs from {example_rel}")
    mkdocs = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    if markdown_in_html and not re.search(r"^\s*-\s+md_in_html\s*$", mkdocs, re.MULTILINE):
        errors.append("mkdocs.yml: docs use Markdown inside HTML but md_in_html is not enabled")
    if errors:
        print("Documentation link errors:")
        print("\n".join(f"  {error}" for error in errors))
        return 1
    print(f"Checked local links in {len(sources)} Markdown files and maintained code examples")
    return 0


if __name__ == "__main__":
    sys.exit(main())
