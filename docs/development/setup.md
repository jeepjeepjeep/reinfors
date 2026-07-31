# Development setup

Reinfors is a Cargo workspace with a PyO3 extension built by Maturin.

## Build locally

```bash
git clone https://github.com/jeepjeepjeep/reinfors.git
cd reinfors
python -m venv .venv
. .venv/bin/activate
pip install maturin pytest numpy
maturin develop
```

Optional adapter and training tests need their extras:

```bash
pip install -e ".[test]"
```

## Verify changes

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test -p reinfors-core -p reinfors-games
pytest
python scripts/generate_docs.py --check
python scripts/check_docs.py
mkdocs build --strict
```

Performance measurements require a release build. Confirm it at runtime with
`rf.core_build_profile()` before recording results.

## Repository map

| Path | Contents |
| --- | --- |
| `crates/reinfors-core` | Native traits, policies, learners, engine, and solvers |
| `crates/reinfors-games` | Built-in games, encoders, rewards, and codecs |
| `crates/reinfors-py` | PyO3 conversion and binding surface |
| `python/reinfors` | Python modules, registries, stubs, adapters, and catalogue metadata |
| `scripts` | End-to-end trainers and maintenance tools |
| `examples` | Minimal runnable integration examples |
| `tests` | Python API and adapter tests |
| `docs` | Progressive user and contributor documentation |
