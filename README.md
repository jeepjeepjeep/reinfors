# reinfors

A general-purpose, gym-style **simulation + batching engine** with a Rust core and a Python API —
"Polars for RL environments." The latency-sensitive, Python-slow parts of an RL pipeline (env
dynamics, observation construction, batched tree search) run in Rust; your network and training
loop stay in PyTorch/JAX. Rust owns *data generation*; Python owns *learning*.

## Layout (maturin "mixed" project — one repo, one wheel)

```
crates/reinfors-core   pure Rust engine (no Python); the value, unit-testable on its own
crates/reinfors-py     PyO3 bindings -> compiled module `reinfors._reinfors`
python/reinfors/       ergonomic Python API (the declarative game builder) — grows over time
```

## Status: Phase 1

Porting `snake_RL`'s `CleanSnakeEnv` dynamics + egocentric observation into `reinfors-core`,
differential-tested against the Python implementation (the oracle). Generic game abstractions and
the declarative builder come later, once the concrete slice is proven and measured.

## Build

```sh
uvx maturin build -o dist          # build the wheel
uvx maturin develop                # or: install into the active venv for iteration
cargo test -p reinfors-core        # pure-Rust unit tests (no Python)
```

## Git hooks

`main` is protected by a client-side guard that blocks direct pushes (changes go through a PR).
After cloning, enable the pre-commit and pre-push hooks once:

```sh
uvx pre-commit install --hook-type pre-commit --hook-type pre-push
```

The pre-push hook (`scripts/block-main-push.sh`) rejects `git push` to `main`; use a branch + PR
instead. (It can be bypassed with `git push --no-verify` for genuine emergencies.)
