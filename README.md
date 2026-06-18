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
- **training is the consumer's** — reinfors ships *no* model, loss, or training loop: learning is
  yours, in PyTorch/JAX. The only seam is the `infer` callback (an `(N, C*H*W) f32 -> (N, K, A) f64`
  forward), which the search calls once per pooled round, so each `collect` automatically searches with
  the live weights — the actor-learner sync is implicit. `scripts/train_example.py` is a tiny,
  self-contained reference for wiring a torch model to the engine (optional `torch`:
  `pip install reinfors[train]`); snake_RL's `scripts/train_reinfors.py` is a full production trainer
  on the same contract.
- **parallel search + flat marshalling + benchmark** (this stage) — the per-search CPU work (expand,
  evaluate, back up) runs in parallel across the pooled requests via rayon, with only the pooled-obs
  gather and the one `infer` call per round serial; this is value-neutral (bit-identical regardless of
  thread count). The `infer` boundary passes obs in and values out as single contiguous row-major
  buffers (obs moved straight into numpy, no copy) instead of nested `Vec`s. On a **release** build
  (CPU value function, grid 20 / 16 games / 10 heads / budget 64) benchmarking measured
  ~2.4 ms per searched decision — about **6x faster than the pure-Python oracle**, with the flat
  boundary worth ~1.6x over nested-`Vec` marshalling and rayon ~1.3x from 1→10 threads. (Benchmark
  only a release build — a debug extension inflates reinfors' per-search cost several-fold and is
  meaningless.)
- **GPU validation** (this stage) — the pipeline runs end to end on a real conv net on the GPU (MPS):
  a no-grad forward on MPS serves the search and gradient steps run on-device. Benchmarking with the
  real net on MPS confirmed the founding premise — **pooling is what makes the GPU win**: with the real
  10-head conv net (grid 20, budget 64) a solo search ties CPU vs MPS (~30 ms/decision, MPS launch
  overhead cancels its compute edge), but the pooled per-round batch grows MPS to **~3.9x at 8 games
  and ~6.1x at 32** (≈4 ms/decision) over CPU inference, and rising with pool size. Against snake_RL's
  planner on the *same* MPS net and pooling (`--baseline --net`), reinfors is ~3-4x faster pooled
  (0.34x at 8 games, 0.24x at 32). Decomposing that with the pool=1 control (single search, so no
  rayon parallelism): the raw Rust-vs-Python *search* gap is ~10x when the search dominates (pool=1, a
  cheap CPU value fn), but only ~1.3x with the real net at pool=1 — there the tiny per-round batch
  leaves both pinned on un-amortized GPU launches. So reinfors' pooled GPU lead over snake_RL is
  mostly pooling + parallelising the search across cores (which the GIL denies the Python planner),
  with the search-implementation gap re-emerging as the per-decision GPU cost shrinks.

Generic game abstractions and the declarative builder come later, once the concrete slice is proven.

## Multiple games through one generic core

The search and rollout engine are generic over a `Game` trait, so one unified `Engine` drives three
games — **snake** (2-player simultaneous), **Connect-4** (sequential 2-player, alternating MAX vs
modeled-opponent-chance nodes), and **GridWorld** (single-agent, pure MAX + lookahead). You compose an
`Engine` from three handles — a game, a policy, and a learner — and every game exposes the same
`observation_space()` / `action_space()` so a network can be sized from the game, not hard-coded:

```python
import reinfors as rf

game = rf.make_game("connect4")             # or rf.games.Connect4(...); pass size kwargs to variable-shape
obs_shape = game.observation_space().shape  # (2, 6, 7);  e.g. rf.make_game("gridworld", size=5)
n_actions = game.action_space().n           # 7
# size your own torch/JAX net from (obs_shape, n_actions) — reinfors ships no model
engine = rf.Engine(
    game,
    rf.make_policy("selective_expectimax", n_heads=4),
    rf.make_learner("treestrap"),
    n_games=16, max_ticks=200,
)
obs, targets, masks, telemetry = engine.collect(2048, infer)  # `infer`: (N, C*H*W) f32 -> (N, K, A) f64
```

`rf.registered_games()` / `registered_policies()` / `registered_learners()` list the names;
`rf.engine_from_config(...)` builds the same `Engine` from a (YAML-shaped) config dict.

## Build

```sh
uvx maturin build -o dist          # build the wheel
uvx maturin develop                # or: install into the active venv for iteration
cargo test -p reinfors-core        # pure-Rust unit tests (no Python)
```

## Training a model

reinfors generates data; you train. `scripts/train_example.py` is a minimal, self-contained reference
— a tiny conv net, the `infer` callback, and a short `collect` -> gradient-step loop — showing how a
torch model plugs into the engine:

```sh
maturin develop --release                                   # release build (see benchmark note)
uv run --with torch python scripts/train_example.py --iterations 20 --device mps
```

For a full production run — config-driven, replay buffer, TensorBoard logging, checkpoint/resume, and a
like-for-like speed/quality comparison against the pure-Python oracle — see snake_RL's
`scripts/train_reinfors.py`, which drives this same `Engine` + `infer` contract while keeping the
network and gradient step entirely on the snake_RL side.

## Git hooks

`main` is protected by a client-side guard that blocks direct pushes (changes go through a PR).
After cloning, enable the pre-commit and pre-push hooks once:

```sh
uvx pre-commit install --hook-type pre-commit --hook-type pre-push
```

The pre-push hook (`scripts/block-main-push.sh`) rejects `git push` to `main`; use a branch + PR
instead. (It can be bypassed with `git push --no-verify` for genuine emergencies.)
