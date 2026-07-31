# Python bindings

Python exposes opaque native handles rather than mirroring Rust state. Users compose handles,
then the engine or solver consumes them.

## Register a component

For a built-in component:

1. Add its PyO3 factory arm and convert Python arguments into a validated Rust config.
2. Add a typed static constructor to `python/reinfors/_reinfors.pyi`.
3. Export the constructor from the matching module (`games`, `policies`, `learners`,
   `encoders`, `chance_modes`, or `noise`).
4. Add its stable snake-case name to that module's `_REGISTRY`.
5. Add catalogue metadata, including compatible workflows, when it is a user-facing game or algorithm.
6. Test direct construction, `make(name, **kwargs)`, resolved-config reconstruction, and
   invalid input.

The registry modules assert against `reinfors.catalog` at import time, and CI regenerates the
catalogue pages in check mode. This prevents a registered component from silently disappearing
from documentation.

## Downstream composition

The intended stable extension boundary is published `reinfors-core` and `reinfors-games`
crates plus a documented PyO3 registration mechanism. A downstream extension can implement
native components and build its own Python extension that composes with the core contracts.

There is no v0 native build tool that accepts Python definitions and compiles Rust for the
user. Packaging an extension remains a normal Rust/PyO3 project responsibility.

## Binding design rules

- Keep Python callbacks at a small number of explicit seams; do not add per-node Python calls.
- Convert callback failures and constructor errors into contextual Python exceptions, never
  Rust panics.
- Validate array rank, shape, dtype, row count, and finite values before native indexing.
- Preserve named batch fields when adding data; positional tuple order is compatibility-only.
- Keep catalog metadata pure Python so the documentation can build without compiling Rust.
- Update type stubs in the same change as runtime bindings.

## Documentation update

Do not edit generated catalogue pages directly. Update `python/reinfors/catalog.py`, run:

```bash
python scripts/generate_docs.py
python scripts/generate_docs.py --check
```

Then add detailed prose only if the new component introduces a contract or workflow that the
tables cannot express.
