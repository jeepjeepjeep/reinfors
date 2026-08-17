# Contributing

Thanks for considering a contribution. Issues and pull requests are welcome; for anything
larger than a bug fix, open an issue first so the approach can be agreed before you invest
in an implementation.

## Setup

Follow the [development setup guide](docs/development/setup.md) for the editable build
(Cargo workspace + PyO3 extension via Maturin) and test invocations.

## Checks a pull request must pass

CI runs three jobs; every command is reproducible locally:

- **rust** — `cargo fmt --all -- --check`;
  `cargo clippy --workspace --all-targets -- -D warnings`;
  `cargo test -p reinfors-core -p reinfors-games`
- **python** — `uvx ruff@0.15.15 check .` and `uvx ruff@0.15.15 format --check .`;
  `uvx --with numpy mypy@2.1.0 python`; then the tests against a freshly built wheel in an
  isolated environment (not the editable build):

  ```bash
  uvx maturin@1.14.0 build --out dist --interpreter python3.12
  uv run --no-project --python 3.12 --with dist/*.whl --with pytest==9.1.0 pytest
  ```
- **docs** — `python scripts/generate_docs.py --check`; `python scripts/check_docs.py`;
  `uvx --with mkdocs-material==9.7.7 mkdocs==1.6.1 build --strict`

Installing the repo's [pre-commit](https://pre-commit.com) hooks with `uvx pre-commit install`
covers the formatting and lint gates at commit time.

## Code conventions

- Comments are reserved for constraints the code cannot express; explanations live in `docs/`.
  Before adding a comment, ask what breaks if it is deleted — if the answer is nothing, leave
  it out.
- Every public Python input must raise an ordinary Python exception, never a Rust panic.
  Constructors registered in the handle modules are enrolled in the adversarial sweep
  automatically; new parameters with non-obvious types fail test collection until they are
  given an edge-value bank (see `tests/test_constructor_validation.py`).

## Adding a game

Follow the end-to-end guides: [add a game](docs/extending/rust-components.md) for the
`Game` implementation (invariants, chance, snapshots, native tests), then
[register its Python binding](docs/extending/python-bindings.md) for the PyO3 dispatch,
stub constructor, and exports. The final registration stage is one entry in each registry —
`python/reinfors/games.py`, `python/reinfors/encoders.py`, and `python/reinfors/catalog.py` —
whose parity asserts and the adversarial sweep then enforce completeness automatically.
Games with an external reference implementation should ship a parity test against it,
following the existing `tests/test_*_parity.py` suites.

## Licensing

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion
in this work shall be dual-licensed under MIT OR Apache-2.0, without any additional terms or
conditions.
