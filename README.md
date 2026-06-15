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
  deferred opponents, in-tree apple spawning (deterministic first-empty belief, with `food_samples`
  Monte-Carlo fan-out of eating branches), and pooled cross-game `search_many` (one batched `infer`
  per round across all games). Leaf values come from a Python inference callback.
- **rollout `Engine`** — drives N parallel games (apples spawned uniformly per game) through the
  pooled search, Thompson-samples a head per game, and `collect`s training records with the full
  `EnsembleTreeStrapRunner` semantics: episode-end **z-mixing** of the realized return into the
  executed action, optional **interior** MAX-node targets (true TreeStrap), and a per-head
  **bootstrap mask** on every record.
- **trainer** (`reinfors.training`) — the end-to-end actor-learner loop: an ensemble Q-network (a
  faithful port of the oracle's, so checkpoints are interchangeable) whose forward is the search's
  `infer` callback. Each iteration pushes `collect`'s records into a ring `ReplayBuffer` (a port of
  the oracle's `EnsembleTreeStrapBuffer`) and takes several gradient steps on sampled minibatches with
  the per-head masked-Huber loss — off-policy replay that reuses each (expensive) searched record many
  times and decorrelates updates. Because `infer` reads the live network, each `collect` searches with
  the current weights — the weight sync is implicit. Optional `torch` dependency (`pip install reinfors[train]`).
- **parallel search + flat marshalling + benchmark** (this stage) — the per-search CPU work (expand,
  evaluate, back up) runs in parallel across the pooled requests via rayon, with only the pooled-obs
  gather and the one `infer` call per round serial; this is value-neutral (bit-identical regardless of
  thread count). The `infer` boundary passes obs in and values out as single contiguous row-major
  buffers (obs moved straight into numpy, no copy) instead of nested `Vec`s. On a **release** build
  (CPU value function, grid 20 / 16 games / 10 heads / budget 64) `scripts/benchmark.py` measures
  ~2.4 ms per searched decision — about **6x faster than the pure-Python oracle**, with the flat
  boundary worth ~1.6x over nested-`Vec` marshalling and rayon ~1.3x from 1→10 threads. (Benchmark
  only a release build — a debug extension inflates reinfors' per-search cost several-fold and is
  meaningless.)
- **GPU validation** (this stage) — the pipeline runs end to end on a real `BootstrappedQNetwork` on
  the GPU (MPS): `make_infer(net, "mps")` serves the search, gradient steps run on-device, and the
  forward matches CPU within float tolerance (all MPS-gated tests). `scripts/benchmark.py --net
  --device mps` confirms the founding premise — **pooling is what makes the GPU win**: with the real
  10-head conv net (grid 20, budget 64) a solo search ties CPU vs MPS (~30 ms/decision, MPS launch
  overhead cancels its compute edge), but the pooled per-round batch grows MPS to **~3.9x at 8 games
  and ~6.1x at 32** (≈4 ms/decision) over CPU inference, and rising with pool size. Against snake_RL's
  planner on the *same* MPS net and pooling (`--baseline --net`), reinfors stays **~3-4x faster**
  (0.34x at 8 games, 0.24x at 32) — the GPU forward is shared and amortized, so the Rust search vs the
  Python search is what separates them (a smaller margin than the ~6x on a cheap CPU value function,
  since the heavier shared forward takes a larger slice of both).

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
