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
  planner on the *same* MPS net and pooling (`--baseline --net`), reinfors is ~3-4x faster pooled
  (0.34x at 8 games, 0.24x at 32). Decomposing that with the pool=1 control (single search, so no
  rayon parallelism): the raw Rust-vs-Python *search* gap is ~10x when the search dominates (pool=1, a
  cheap CPU value fn), but only ~1.3x with the real net at pool=1 — there the tiny per-round batch
  leaves both pinned on un-amortized GPU launches. So reinfors' pooled GPU lead over snake_RL is
  mostly pooling + parallelising the search across cores (which the GIL denies the Python planner),
  with the search-implementation gap re-emerging as the per-decision GPU cost shrinks.

Generic game abstractions and the declarative builder come later, once the concrete slice is proven.

## Multiple games through one generic core

The search and rollout engine are generic over a `Game` trait, so the same core now drives three
games — **snake** (2-player simultaneous), **Connect-4** (sequential 2-player, alternating MAX vs
modeled-opponent-chance nodes), and **GridWorld** (single-agent, pure MAX + lookahead). Each is
exposed as its own PyO3 rollout engine — `reinfors._reinfors.Engine` (snake),
`Connect4Engine`, and `GridWorldEngine` — all with the same `collect(n_records, infer) -> (obs, targets,
masks, telemetry)` contract, differing only in their `#[new]` (the game's rules/rewards) and their
observation/action dimensions.

`reinfors.games` is a small registry that ties a game name to its engine class and shape metadata, so
a caller can size a `BootstrappedQNetwork` and pick an engine without hard-coding dimensions:

```python
import reinfors
from reinfors import games
from reinfors.training import BootstrappedQNetwork

obs_shape, n_actions = games.net_shape("connect4")  # ((2, 6, 7), 7); pass size kwargs for
                                                     # variable-shape games, e.g. games.net_shape("gridworld", size=5)
net = BootstrappedQNetwork(obs_shape, n_actions=n_actions, n_heads=4)
engine = reinfors._reinfors.Connect4Engine(1.0, -1.0, 0.0, ...)  # win/loss/draw rewards + search/rollout knobs
reinfors.training.train(engine, net, optimizer, iterations=..., collect_size=..., batch_size=...)
```

`games.get(name)` returns the full `GameSpec` (engine class, `action_count`, and an `obs_shape` that is
a fixed tuple for snake/Connect-4 and a callable for size-parameterized GridWorld); `games.net_shape`
is the convenience that returns just `(obs_shape, action_count)`.

## Build

```sh
uvx maturin build -o dist          # build the wheel
uvx maturin develop                # or: install into the active venv for iteration
cargo test -p reinfors-core        # pure-Rust unit tests (no Python)
```

## Training a model + comparing to snake_RL

`scripts/train.py` runs a config-driven training loop whose hyperparameters mirror snake_RL's
`configs/ensemble_treestrap.yaml`, logging the same TensorBoard scalars snake_RL's
`EnsembleTreeStrapRunner` does (`train/loss`, `train/mean_q`, `episode/reward_*`, `episode/length`,
`search/*`) plus `throughput/*`. Every scalar carries wall-clock, so TensorBoard's Relative/Wall
x-axis gives the time-based learning curve and the step axis gives the per-step one — both axes of the
comparison from one run.

```sh
maturin develop --release                                   # release build (see benchmark note)
python scripts/train.py --device mps --log-dir runs/reinfors_ensemble
# snake_RL side, same config (in the sibling checkout):
python scripts/train.py configs/ensemble_treestrap.yaml --device mps --log-dir runs/snake_rl
tensorboard --logdir runs                                   # both runs overlaid
```

Reading the comparison honestly:

- **Train speed** (the unconfounded axis): reinfors generates data in parallel Rust with one pooled
  GPU forward per round across `--n-games` games; snake_RL runs a single Python self-play env. Compare
  `throughput/*` and any curve on the Wall x-axis.
- **Quality per step** has known confounds in this phase, surfaced in the script's header: the
  `--n-games` **parallelism** (changes the replay mix) and the **train cadence** mapping (snake_RL's
  per-tick `train_every` vs reinfors' `collect_size / grad_steps`). The search's apple respawn is now a
  uniform-random draw shared by the env and the search (matching snake_RL), so spawning is no longer a
  confound.

## Git hooks

`main` is protected by a client-side guard that blocks direct pushes (changes go through a PR).
After cloning, enable the pre-commit and pre-push hooks once:

```sh
uvx pre-commit install --hook-type pre-commit --hook-type pre-push
```

The pre-push hook (`scripts/block-main-push.sh`) rejects `git push` to `main`; use a branch + PR
instead. (It can be bypassed with `git push --no-verify` for genuine emergencies.)
