# Development setup

Reinfors is a Cargo workspace with a PyO3 extension built by Maturin.

## Prerequisites

Extension work assumes basic Rust familiarity: you should be comfortable reading traits, enums,
`Result` values, match arms, and Cargo test output. Install:

- Python 3.10 or newer;
- Git;
- the stable Rust toolchain, including Cargo, rustfmt, and Clippy, through
  [rustup](https://rustup.rs/);
- a platform C/C++ build toolchain, such as Xcode Command Line Tools on macOS or the standard build
  tools for your Linux distribution.

Confirm the Rust tools before cloning:

```bash
rustup toolchain install stable --component rustfmt --component clippy
rustc --version
cargo --version
```

## Build the editable extension

```bash
git clone https://github.com/jeepjeepjeep/reinfors.git
cd reinfors
python -m venv .venv
. .venv/bin/activate
pip install maturin pytest numpy uv
maturin develop
```

`maturin develop` compiles the Rust crates and installs the resulting native extension into the
active virtual environment. A cold workspace build can take several minutes; later incremental
builds are normally much faster. Cargo's compile output indicates that the build is progressing.

Use the default debug build for development. Performance measurements require
`maturin develop --release`; confirm the active build at runtime with `rf.core_build_profile()`
before recording results.

Optional adapter and training tests need their extras:

```bash
pip install -e ".[test]"
```

## Rebuild after native changes

After changing anything under `crates/`, rebuild before running Python tests or trying a new public
constructor:

```bash
maturin develop
```

Pure Python and documentation changes do not require this step. An `AttributeError` for a newly
added constructor often means that Python is still loading the previous native build; rebuild in
the active virtual environment before debugging the registration code.

## Verify changes

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test -p reinfors-core -p reinfors-games
uvx ruff@0.15.15 check .
uvx ruff@0.15.15 format --check .
uvx --with numpy mypy@2.1.0 python
pytest
python scripts/generate_docs.py --check
python scripts/check_docs.py
uvx --with mkdocs-material==9.7.7 mkdocs==1.6.1 build --strict
```

## Repository map

| Path | Contents |
| --- | --- |
| `crates/reinfors-core` | Native traits, policies, learners, engine, and solvers |
| `crates/reinfors-games` | Built-in games, encoders, rewards, and codecs |
| `crates/reinfors-py` | PyO3 conversion and binding surface |
| `python/reinfors` | Python modules, registries, stubs, adapters, and catalogue metadata |
| `scripts` | Repository maintenance tools (docs generation and checks, git hooks) |
| `examples` | Runnable training and integration examples |
| `tests` | Python API and adapter tests |
| `docs` | Progressive user and contributor documentation |

## Next steps

- Choose an extension path from the [contributor overview](../extending/index.md).
- Use the [documentation workflow](documentation.md) for catalogue and site changes.
- Run performance work through the companion [reinfors-benchmarks](https://github.com/jeepjeepjeep/reinfors-benchmarks) repository (specs + runner; see its README).
