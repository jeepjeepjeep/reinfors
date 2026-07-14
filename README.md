# reinfors

**Search-based RL data generation with a Rust core and a Python API.** A fast, parallel Rust engine
runs your games and a decision-time search, and calls out to *your* network — in any framework — for
evaluation. Rust owns *data generation*; Python (with torch/JAX/…) owns *learning*. The two meet at one
seam: an `infer` callback the search invokes once per pooled round, so every `collect` searches with the
live weights — the actor–learner sync is implicit.

reinfors ships **no** model, loss, or training loop. You bring the net.

## Why reinfors — where it fits

Two libraries already do parts of this well, and reinfors deliberately sits between them:

- **[Pgx](https://github.com/sotetsuk/pgx)** runs board-game environments *entirely on GPU/TPU* in JAX
  (vmapped across a batch, no host transfer) — unbeatable for board games at massive batch, but
  board-games-only (fixed-size state) and your network must be JAX.
- **[OpenSpiel](https://github.com/google-deepmind/open_spiel)** has a fast C++ MCTS and a huge game
  library, but its C++ search can only call *C++* evaluators; to drive it with a Python network you drop
  to its Python MCTS, which is roughly an order of magnitude slower.

reinfors' niche is the combination neither offers: a **compiled (Rust) search loop driven by your own
Python network**, parallelised across games on CPU, over game spaces too irregular for JAX to vectorise.

| | **reinfors** | **Pgx** | **OpenSpiel** |
|---|---|---|---|
| Search runs in | Rust (compiled) | JAX / XLA | C++ (fast) *or* Python (slow) |
| Your net can be | **any** framework, via a callback | JAX only | C++/libtorch (fast path) *or* Python (slow path) |
| Compiled search **+** a Python net | ✅ | ✗ (net is JAX) | ✗ (fast path needs a C++ net) |
| Hardware | CPU, parallel (rayon) | GPU / TPU, on-device | CPU |
| Game state | **any** — dynamic / irregular (e.g. variable-length) | fixed-size (board games) | broad (board, card, imperfect-info) |
| Cross-game batching | pooled across `n_games`, one net call/round | `vmap` over the batch | per-bot / C++ inference server |
| Games shipped today | few (3, growing) | many board games | very many |
| Sweet spot | flexible nets, irregular games, CPU / mid-scale | board games at large GPU batch | breadth of games + algorithms |

The honest summary: for **board games at massive GPU scale**, reach for Pgx; for the **broadest game and
algorithm library**, OpenSpiel; for **a compiled search you drive with your own Python network across
flexible game spaces on CPU** — that's reinfors. Concretely: for a Python-net workflow, reinfors' Rust
search is several times faster than OpenSpiel's Python MCTS (the only OpenSpiel path that accepts a Python
net); against OpenSpiel's C++ MCTS or Pgx+mctx on a GPU it trades raw board-game throughput for
generality. Reproducible head-to-heads — including where each *wins* — live in `scripts/benchmark_vs.py`.

## Install

```sh
pip install reinfors            # the engine (needs only numpy)
pip install reinfors[gym]       # + Gymnasium / PettingZoo adapters (reinfors.gym)
pip install reinfors[train]     # + torch, to run scripts/train_example.py
```

Prebuilt wheels are published for Linux, macOS, and Windows — one abi3 wheel per platform, Python 3.10+.

## Quick start

```python
import reinfors as rf

game    = rf.games.Connect4()
reward  = rf.Reward(win=1.0, loss=-1.0)          # reward is decoupled from the game's rules
policy  = rf.policies.Mcts(num_simulations=64)   # or SelectiveExpectimax, EpsilonGreedyQ
learner = rf.learners.TreeStrap(gamma=0.99)
engine  = rf.Engine(game, reward, policy, learner, n_games=16)

# `infer(obs) -> per-head action values` is YOUR network's forward pass (torch / JAX / numpy — anything).
# It's called once per pooled search round, batched across all n_games, and closes over the live weights.
obs, targets, masks, telemetry = engine.collect(2048, infer)   # training records — you run the grad step
```

Every game advertises its shapes, so you can size a network from the game instead of hard-coding it:

```python
game = rf.games.Snake(grid_size=20, max_ticks=750)
obs_shape = game.observation_space().shape        # (C, H, W)
n_actions = game.action_space().n
```

`scripts/train_example.py` is a minimal, self-contained PyTorch trainer — the reference for the
Python↔Rust `infer` contract (`(N, C*H*W) float32 -> (N, K, A) float64`).

## What's in the box

- **Games** (`rf.games`) — `Snake` (2-player simultaneous, variable-length bodies), `Connect4`
  (sequential), `GridWorld` (single-agent). Games implement a small `Game` trait, so adding one is local.
- **Policies** (`rf.policies`) — `SelectiveExpectimax` (best-first, ensemble/uncertainty-guided search),
  `Mcts` (UCT), `EpsilonGreedyQ` (reactive).
- **Learners** (`rf.learners`) — `TreeStrap` (regress the search's backed-up value targets), `Dqn`.
- **Standard-API adapters** (`rf.gym`) — expose any game as a `gymnasium.Env` (single-agent) or a
  PettingZoo `ParallelEnv` (simultaneous multi-agent), so it drops into SB3 / CleanRL / RLlib / Tianshou.
- **`rf.Env`** — a caller-driven single-game instance for play and evaluation.

Reward, observation encoding, and the start-state distribution are all **decoupled seams**: the game owns
only the rules, so the same game trains under any reward or encoding without touching its dynamics.
`rf.make_game` / `make_policy` / `make_learner` and `rf.engine_from_config(...)` are the name-addressable,
config-driven equivalents.

## How it works

`engine.collect` drives `n_games` in parallel (rayon). Each decision runs the search; the search pools
the leaves it needs to evaluate *across all active games* into a single `infer` call per round, so one
batched forward serves the whole pool — this is what amortises the Python↔net boundary and lets a GPU
network pay off. The search is value-neutral w.r.t. parallelism (bit-identical regardless of thread
count). The result is training records shaped for the learner; the model and the gradient step are yours.

## Status

Pre-1.0. The engine, three games, the search/learner families, and the standard-API adapters are in
place and tested; the game and policy libraries are actively growing. Current scope is perfect-information
games with up to two agents. For benchmarking, **build in release** — a debug extension runs the Rust
core roughly 10× slower (the `rf.core_build_profile()` guard warns when it's not release).

## Development

```sh
uvx maturin develop --release      # build + install into the active venv (release; see the note above)
cargo test -p reinfors-core        # pure-Rust unit tests (no Python)
uv pip install -e ".[test]"        # a dev/editor venv with the full test suite's imports
```

`main` is protected by a client-side hook that blocks direct pushes; changes go through a PR. After
cloning, enable the hooks once: `uvx pre-commit install --hook-type pre-commit --hook-type pre-push`.

## License

MIT — see [LICENSE](LICENSE).
