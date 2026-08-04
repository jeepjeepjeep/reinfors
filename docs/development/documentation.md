# Maintaining documentation

Documentation is organized by user intent, with detail disclosed progressively:

- the README and landing page explain the value and shortest path;
- guides teach complete workflows;
- generated catalogues answer coverage questions;
- extension pages explain ownership and contribution contracts;
- reference pages hold exact shapes, fields, and boundaries;
- benchmark pages state reproducibility requirements separately from marketing.

Do not add a page per game or algorithm by default. Add one only when a component needs
substantial unique setup, theory, or interpretation that cannot fit a catalogue row and an
example.

## Update component coverage

Edit `python/reinfors/catalog.py`, not the generated Markdown:

```bash
python scripts/generate_docs.py
```

CI runs `--check` and fails when catalogue output is stale. Runtime registry assertions and
tests keep metadata names aligned with exported constructors.

## Write durable prose

- Link to generated catalogues instead of writing “N games” or enumerating the same list.
- State contracts and ownership, not internal implementation details likely to move.
- Put one canonical code sample in a tested script; keep prose snippets short.
- Use named batch fields and resolved configs in examples.
- Explain algorithm-specific limitations next to the relevant contract and summarize them in
  [current boundaries](../reference/limits.md).
- Do not publish benchmark claims without the artifacts required by the benchmark methodology.

## Preview

The catalogue generator and documentation checks do not import the native extension. Preview the
site with an isolated MkDocs tool environment, without installing the package or compiling Rust:

```bash
python scripts/generate_docs.py --check
python scripts/check_docs.py
uvx --with mkdocs-material==9.7.7 mkdocs==1.6.1 serve
```

Before committing:

```bash
python scripts/generate_docs.py --check
python scripts/check_docs.py
uvx --with mkdocs-material==9.7.7 mkdocs==1.6.1 build --strict
```
