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

## Status

The concrete snake slice is in place, differential-tested against `snake_RL` (the oracle):

- **env + egocentric observation** — bit-identical to `CleanSnakeEnv`.
- **selective expectimax search** — per-head ensemble (σ-VOI priority), uniform and distributional
  deferred opponents, in-tree apple spawning (deterministic first-empty belief), and pooled
  cross-game `search_many` (one batched `infer` per round across all games). Leaf values come from a
  Python inference callback.
- **rollout `Engine`** — drives N parallel games (apples spawned uniformly per game) through the
  pooled search, Thompson-samples a head per game, and `collect`s training records with the full
  `EnsembleTreeStrapRunner` semantics: episode-end **z-mixing** of the realized return into the
  executed action, optional **interior** MAX-node targets (true TreeStrap), and a per-head
  **bootstrap mask** on every record.
- **trainer** (`reinfors.training`) — the end-to-end actor-learner loop: an ensemble Q-network (a
  faithful port of the oracle's, so checkpoints are interchangeable) whose forward is the search's
  `infer` callback, regressed onto `collect`'s records with the per-head masked-Huber loss. Because
  `infer` reads the live network, each `collect` searches with the current weights — the weight sync
  is implicit. Optional `torch` dependency (`pip install reinfors[train]`).
- **parallel search + flat marshalling + benchmark** (this stage) — the per-search CPU work (expand,
  evaluate, back up) runs in parallel across the pooled requests via rayon, with only the pooled-obs
  gather and the one `infer` call per round serial; this is value-neutral (bit-identical regardless of
  thread count). The `infer` boundary passes obs in and values out as single contiguous row-major
  buffers (obs moved straight into numpy, no copy) instead of nested `Vec`s. On a **release** build
  (CPU value function, grid 20 / 16 games / 10 heads / budget 64) `scripts/benchmark.py` measures
  ~2.4 ms per searched decision — about **6x faster than the pure-Python oracle**, with the flat
  boundary worth ~1.6x over nested-`Vec` marshalling and rayon ~1.3x from 1→10 threads; the advantage
  compounds further with GPU inference. (Benchmark only a release build — a debug extension inflates
  reinfors' per-search cost several-fold and is meaningless.)

Generic game abstractions and the declarative builder come later, once the concrete slice is proven.

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
