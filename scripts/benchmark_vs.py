"""Cross-framework benchmark on connect4 (Phase 2). MANUAL-RUN — not CI.

Compares reinfors against the packages it's genuinely comparable to, on the one game they all implement
(connect4): **Pgx** (JAX, GPU/TPU-resident, batched) and **OpenSpiel** (C++ core, MCTS). Because these
have different execution models and run on different hardware, there is no single fair number — so this
reports per *track*, swept over batch size / net size, with **every row tagged by device**. Never read a
GPU row against a CPU row as one ratio; read each within its track and hardware.

Three tracks:
  * Track A — raw env transitions/sec (apply one action, advance one state). reinfors and OpenSpiel are
    single-core CPU (a "batch" is just N independent envs stepped in a loop — they don't vectorize raw
    stepping), so their curves are ~flat in batch; Pgx vmaps on the accelerator, so its curve rises with
    batch. That contrast is the finding. reinfors' single-env number UNDERSELLS it — its product is
    Track B (parallel, search-driven), not raw stepping.
  * Track B — searched decisions/sec (UCT), swept over the shared net's size at a fixed budget. Three
    columns: **reinfors** (Rust MCTS + the shared net), **openspiel-py** (`open_spiel.python.algorithms.
    mcts` + the SAME shared net, fed the same canonical board), and **openspiel-c++** (the native C++
    `MCTSBot`). The C++ bot CAN'T call a Python net — `pyspiel.Evaluator` isn't Python-subclassable — so
    it uses a C++ rollout evaluator and is shown as a *constant reference* (ignores the net): OpenSpiel's
    real fast path. How to read it, by use case:
      - If your net is in PYTHON (the usual RL workflow), OpenSpiel's *only* option is openspiel-py — its
        C++ MCTS can't take a Python evaluator (only C++ evaluators) — so reinfors-vs-openspiel-py is the
        apples-to-apples comparison, and reinfors is ~4x faster at n_games=1 (more at scale). That is
        reinfors' real edge: a compiled (Rust) search loop that keeps your net in Python — a combination
        OpenSpiel doesn't offer.
      - openspiel-c++ (constant) is OpenSpiel's speed *only if you write the net in C++/libtorch* — the
        path that runs the net in-process. reinfors keeps the net in Python, which costs a per-eval
        Rust<->Python boundary: real at n_games=1 (batch-1 calls), but amortized by BATCHING the net
        across pooled games (Phase 1's n_games scaling). Whether reinfors' Python-net path matches an
        all-C++ net path at n_games=1 is UNTESTED here — it needs OpenSpiel built with libtorch + a C++
        net evaluator (the pip wheel omits it). This column uses a C++ *rollout* evaluator as a fast-loop
        proxy; it is NOT the C++-net comparison, so don't read it as "reinfors == OpenSpiel's fast net path".
      - a heavy net makes reinfors + openspiel-py net-bound (both fall below the C++ rollout reference).
  * Track C — batched MCTS: searched decisions/sec vs batch, with reinfors `n_games` == pgx batch. Both
    parallelize across B self-play games — reinfors (Rust trees + one batched net call/round) vs pgx+mctx
    (B JAX trees vmapped on-device, uniform-prior AlphaZero MCTS). THE board-game search comparison. pgx+
    mctx is a GPU-regime tool (UNVALIDATED / CPU here) — expect reinfors to lead at small batch (no
    compile/JAX overhead) and pgx+mctx at large batch on a GPU. Run it on your accelerator for the curve.

Install (on the machine you're benchmarking): `pip install jax[cuda12] pgx mctx open_spiel` (adjust the
jax wheel for your accelerator). reinfors must be a RELEASE build — this checks and warns.

    uv run --with numpy --with pgx --with mctx --with jax[cuda12] --with open_spiel python scripts/benchmark_vs.py

Backends whose deps aren't importable are skipped with a note; the reinfors backend always runs, and a
per-cell error in any backend degrades to `ERR` (never a crash). The Pgx backend's headline regime —
large batch on a GPU/TPU — is unrun here (CPU only); run it on your accelerator for those numbers.
"""

# The optional backends import jax/pgx/pyspiel/open_spiel, which aren't installed in the dev/CI env;
# suppress the resolver error file-wide (they're guarded at runtime by `available()`).
# pyright: reportMissingImports=false

from __future__ import annotations

import argparse
import os
import platform
import random
from statistics import median
from time import perf_counter
from typing import Any

import numpy as np
import reinfors as rf


def _throughput(work: Any, repeats: int) -> float:
    work()  # untimed warm-up (compiles JAX, primes caches)
    rates = []
    for _ in range(repeats):
        t0 = perf_counter()
        units = work()
        rates.append(units / (perf_counter() - t0))
    return median(rates)


# --------------------------------------------------------------------------------------------------
# Shared value net for Track B. A small fixed-weight MLP fed the SAME canonical connect4 board by both
# reinfors' and OpenSpiel-python's MCTS, so their leaf evaluation is a genuinely shared, equal-cost
# forward (the search implementation is what differs). Canonical board: 42-dim (+1 own, -1 opp, 0 empty)
# from the current player's perspective, index r*7+c with row 0 = bottom — the layout reinfors'
# Connect4Planes and OpenSpiel's `observation_tensor` both produce (a test asserts they agree). Its cost
# is swept via `hidden` to show the transition from search-bound to inference-bound.
# --------------------------------------------------------------------------------------------------


class SharedNet:
    def __init__(self, hidden: int) -> None:
        rng = np.random.default_rng(0)
        self.w1 = (rng.standard_normal((42, hidden)) / np.sqrt(42)).astype(np.float32)
        self.w2 = (rng.standard_normal((hidden, 1)) / np.sqrt(hidden)).astype(np.float32)

    def value(self, board: np.ndarray) -> np.ndarray:
        """(N, 42) canonical board -> (N,) value in (-1, 1), from the board's current-player perspective."""
        h = np.maximum(board.astype(np.float32) @ self.w1, 0.0)
        return np.tanh(h @ self.w2)[:, 0]


def board_from_reinfors(obs: np.ndarray) -> np.ndarray:
    """reinfors' Connect4Planes obs is `[own(42), opp(42)]` flat -> the canonical `(N, 42)` board."""
    return obs[:, :42] - obs[:, 42:]


def board_from_openspiel(state: Any) -> np.ndarray:
    """An OpenSpiel connect4 state -> the canonical `(1, 42)` board (planes are absolute per player; row
    0 = bottom, matching reinfors)."""
    cur = state.current_player()
    ot = np.asarray(state.observation_tensor(cur)).reshape(3, 6, 7)
    return (ot[cur] - ot[1 - cur]).reshape(1, 42)


# --------------------------------------------------------------------------------------------------
# Backends. Each: name, device(), available() -> (ok, detail), raw_step(batch, steps) -> transitions/s.
# Track B search methods are backend-specific (see each). A backend is self-contained and isolated, so a
# missing/broken one never taints the others.
# --------------------------------------------------------------------------------------------------


class ReinforsBackend:
    name = "reinfors"
    validated = True

    def available(self) -> tuple[bool, str]:
        return True, f"{rf.__version__} ({rf.core_build_profile()})"

    def device(self) -> str:
        return "cpu"

    def raw_step(self, batch: int, steps: int, repeats: int) -> float:
        rng = random.Random(0)
        envs = [rf.Env(rf.games.Connect4(), seed=i) for i in range(batch)]

        def work() -> int:
            for e in envs:
                if e.done():
                    e.reset()
                agent = e.active_agents()[0]
                e.step({agent: rng.choice(e.legal_actions(agent))})
            return batch

        for e in envs:
            e.reset()
        return _throughput(lambda: sum(work() for _ in range(steps)), repeats)

    def search(self, budget: int, decisions: int, repeats: int, net: SharedNet, n_games: int = 1) -> float:
        # Genuine UCT MCTS on connect4. `n_games` pools that many self-play games — the batch axis reinfors
        # parallelizes over (rayon + one batched net call per round), the analogue of pgx+mctx's batch and
        # OpenSpiel's per-bot. The whole loop runs in Rust and calls the net through the `infer` callback.
        action_count = rf.games.Connect4().action_space().n
        engine = rf.Engine(
            rf.games.Connect4(),
            rf.Reward(win=1.0, loss=-1.0, draw=0.0),
            rf.policies.Mcts(num_simulations=budget, uct_c=2.0, max_depth=64),
            rf.learners.TreeStrap(gamma=0.99, outcome_weight=0.3, bootstrap_p=1.0, interior_targets=False),
            n_games=n_games,
            seed=0,
        )

        def infer(arr: np.ndarray) -> np.ndarray:
            v = net.value(board_from_reinfors(arr))  # state value, broadcast across actions (MCTS max = V)
            out = np.empty((arr.shape[0], 1, action_count), dtype=np.float64)
            out[:, 0, :] = v[:, None]
            return out

        return _throughput(lambda: int(engine.collect(n_records=decisions, infer=infer).obs.shape[0]), repeats)


class PgxBackend:
    name = "pgx"
    validated = True  # runs on jax/pgx (CPU) here; its headline regime — large batch on GPU/TPU — is unrun

    def _import(self) -> Any:
        import jax
        import pgx

        return jax, pgx

    def available(self) -> tuple[bool, str]:
        try:
            jax, _pgx = self._import()
        except ImportError as e:
            return False, f"not importable ({e})"
        return True, f"pgx via jax {jax.__version__}"

    def device(self) -> str:
        import jax

        return str(jax.devices()[0].platform)  # 'gpu' / 'tpu' / 'cpu'

    def raw_step(self, batch: int, steps: int, repeats: int) -> float:
        import jax
        import jax.numpy as jnp
        import pgx
        from pgx.experimental import auto_reset

        env = pgx.make("connect_four")
        # pgx >= 2.0 auto_reset takes (state, action, key); vmap over the batch and run the whole
        # `steps`-long rollout on-device via fori_loop — the fair Pgx fast path (no per-step Python).
        step = jax.vmap(auto_reset(env.step, env.init))
        init = jax.jit(jax.vmap(env.init))
        state0 = init(jax.random.split(jax.random.PRNGKey(0), batch))

        @jax.jit
        def rollout(state: Any, key: Any) -> Any:
            def body(_i: Any, carry: Any) -> Any:
                state, key = carry
                action = jnp.argmax(state.legal_action_mask, axis=-1)  # first legal action
                key, sub = jax.random.split(key)
                state = step(state, action, jax.random.split(sub, batch))
                return state, key

            return jax.lax.fori_loop(0, steps, body, (state, key))[0]

        def work() -> int:
            jax.block_until_ready(rollout(state0, jax.random.PRNGKey(1)))
            return batch * steps

        return _throughput(work, repeats)

    def search_mcts(self, budget: int, decisions: int, repeats: int, batch: int) -> float:
        # Batched AlphaZero-style MCTS via DeepMind's mctx on pgx connect4: `batch` independent trees
        # advance in lockstep on-device (the analogue of reinfors' n_games). Uniform prior (no policy
        # head, to match reinfors' UCT) + a small JAX value net; 2-player zero-sum via discount=-1
        # (negamax). Env dynamics ARE mctx's recurrent_fn (perfect-information / AlphaZero). No root noise.
        import jax
        import jax.numpy as jnp
        import mctx
        import pgx
        from pgx.experimental import auto_reset

        env = pgx.make("connect_four")
        a_count = env.num_actions
        obs_dim = int(np.prod(env.observation_shape))
        k1, k2 = jax.random.split(jax.random.PRNGKey(0))
        w1 = jax.random.normal(k1, (obs_dim, 64)) / jnp.sqrt(float(obs_dim))
        w2 = jax.random.normal(k2, (64, 1)) / jnp.sqrt(64.0)
        uniform = jnp.zeros((batch, a_count))

        def value(state: Any) -> Any:
            x = state.observation.reshape(state.observation.shape[0], -1)
            return jnp.tanh(jnp.maximum(x @ w1, 0.0) @ w2)[:, 0]  # (batch,), current-player perspective

        def recurrent_fn(_params: Any, _rng: Any, action: Any, state: Any) -> Any:
            mover = state.current_player
            state = jax.vmap(env.step)(state, action)
            v = jnp.where(state.terminated, 0.0, value(state))
            out = mctx.RecurrentFnOutput(
                reward=state.rewards[jnp.arange(batch), mover],  # to the player who moved
                discount=jnp.where(state.terminated, 0.0, -1.0),  # negamax (2p zero-sum)
                prior_logits=uniform,
                value=v,
            )
            return out, state

        reset_step = jax.vmap(auto_reset(env.step, env.init))

        @jax.jit
        def one_round(state: Any, key: Any) -> Any:
            key, sk, rk = jax.random.split(key, 3)
            root = mctx.RootFnOutput(prior_logits=uniform, value=value(state), embedding=state)
            out = mctx.muzero_policy(
                None,
                sk,
                root,
                recurrent_fn,
                num_simulations=budget,
                invalid_actions=~state.legal_action_mask,
                dirichlet_fraction=0.0,
            )
            return reset_step(state, out.action, jax.random.split(rk, batch)), key

        state0 = jax.vmap(env.init)(jax.random.split(jax.random.PRNGKey(1), batch))
        rounds = max(1, decisions // batch)  # each round = `batch` decisions (one search per game)

        def work() -> int:
            state, key = state0, jax.random.PRNGKey(2)
            for _ in range(rounds):
                state, key = one_round(state, key)
            jax.block_until_ready(state)
            return rounds * batch

        return _throughput(work, repeats)


class OpenSpielBackend:
    name = "openspiel"
    validated = True  # runs on open_spiel (CPU) here

    def _game(self) -> Any:
        import pyspiel

        return pyspiel.load_game("connect_four")

    def available(self) -> tuple[bool, str]:
        try:
            import pyspiel  # noqa: F401
        except ImportError as e:
            return False, f"not importable ({e})"
        return True, "open_spiel"

    def device(self) -> str:
        return "cpu"

    def raw_step(self, batch: int, steps: int, repeats: int) -> float:
        game = self._game()
        rng = random.Random(0)
        states = [game.new_initial_state() for _ in range(batch)]

        def work() -> int:
            for i, s in enumerate(states):
                if s.is_terminal():
                    states[i] = s = game.new_initial_state()
                s.apply_action(rng.choice(s.legal_actions()))
            return batch

        return _throughput(lambda: sum(work() for _ in range(steps)), repeats)

    def _run_bot(self, game: Any, bot: Any, decisions: int, repeats: int) -> float:
        def work() -> int:
            state = game.new_initial_state()
            for _ in range(decisions):
                if state.is_terminal():
                    state = game.new_initial_state()
                state.apply_action(bot.step(state))  # one MCTS decision of `budget` simulations
            return decisions

        return _throughput(work, repeats)

    def search_python(self, budget: int, decisions: int, repeats: int, net: SharedNet) -> float:
        # OpenSpiel's *Python* reference MCTS with the shared net evaluator — the only way to feed it a
        # Python net. This is the Rust-loop-vs-Python-loop comparison against reinfors.
        from open_spiel.python.algorithms import mcts

        game = self._game()

        class NetEvaluator(mcts.Evaluator):
            def evaluate(self, state: Any) -> np.ndarray:
                v = float(net.value(board_from_openspiel(state))[0])
                return np.array([v, -v]) if state.current_player() == 0 else np.array([-v, v])

            def prior(self, state: Any) -> list[tuple[int, float]]:
                legal = state.legal_actions()
                p = 1.0 / len(legal)
                return [(a, p) for a in legal]

        bot = mcts.MCTSBot(game, uct_c=2.0, max_simulations=budget, evaluator=NetEvaluator())
        return self._run_bot(game, bot, decisions, repeats)

    def search_cpp(self, budget: int, decisions: int, repeats: int) -> float:
        # OpenSpiel's native C++ MCTSBot with a C++ rollout evaluator — its real fast path. It can't call
        # a Python net (`pyspiel.Evaluator` isn't subclassable), so this uses rollouts and is the honest
        # "how fast is OpenSpiel's search" reference (independent of the shared net's size).
        import pyspiel

        game = self._game()
        evaluator = pyspiel.RandomRolloutEvaluator(1, 42)  # (n_rollouts, seed)
        # (game, evaluator, uct_c, max_simulations, max_memory_mb, solve, seed, verbose)
        bot = pyspiel.MCTSBot(game, evaluator, 2.0, budget, 1000, False, 42, False)
        return self._run_bot(game, bot, decisions, repeats)


BACKENDS: list[Any] = [ReinforsBackend(), PgxBackend(), OpenSpielBackend()]


def _warn_if_not_release() -> None:
    if rf.core_build_profile() != "release":
        bar = "!" * 78
        print(
            f"\n{bar}\n!! reinfors is a '{rf.core_build_profile()}' build (~10x slow) "
            f"— its rows are meaningless.\n{bar}"
        )


def _table(title: str, header: tuple[str, ...], rows: list[tuple[str, ...]], note: str = "") -> None:
    widths = [max(len(header[i]), *(len(r[i]) for r in rows)) for i in range(len(header))]

    def fmt(cells: tuple[str, ...]) -> str:
        return "  ".join(c.rjust(widths[i]) if i else c.ljust(widths[i]) for i, c in enumerate(cells))

    print(f"\n{title}")
    if note:
        print(note)
    print(fmt(header))
    print("-" * (sum(widths) + 2 * (len(header) - 1)))
    for r in rows:
        print(fmt(r))


def _cell(backend: Any, label: str, fn: Any) -> str:
    """One measured table cell, isolated: a backend failure (e.g. an API drift) becomes an 'ERR' cell +
    a one-line note, never a crash that takes the whole comparison down."""
    try:
        return f"{fn():,.0f}"
    except Exception as e:  # a benchmark cell must never propagate a backend's error
        print(f"  ! {backend.name} {label} failed: {type(e).__name__}: {str(e).splitlines()[0][:100]}")
        return "ERR"


def _track_a(active: list[Any], args: argparse.Namespace) -> None:
    batches = [1, 8] if args.smoke else [1, 16, 256, 4096]
    steps = 50 if args.smoke else 2000
    rows = []
    for bs in batches:
        cells = tuple(
            _cell(b, f"raw_step(batch={bs})", lambda b=b, bs=bs: b.raw_step(bs, steps, args.repeats)) for b in active
        )
        rows.append((str(bs), *cells))
    _table(
        "Track A — raw env transitions/sec on connect4 (device in header; CPU rows ~flat, Pgx scales)",
        ("batch", *(f"{b.name} [{b.device()}]" for b in active)),
        rows,
        note="reinfors/OpenSpiel don't vectorize raw stepping (batch = N sequential envs); "
        "Track B is reinfors' real product.",
    )


def _track_b(active: list[Any], args: argparse.Namespace) -> None:
    rf_b = next((b for b in active if b.name == "reinfors"), None)
    os_b = next((b for b in active if b.name == "openspiel"), None)
    if rf_b is None:
        return
    budget = 64
    hiddens = [1, 64] if args.smoke else [1, 64, 512, 2048]
    decisions = 20 if args.smoke else 200

    header = ["net hidden", "reinfors [rust+net]"]
    cpp = None
    if os_b is not None:
        header += ["openspiel-py [+net]", "openspiel-c++ [rollout]"]
        cpp = _cell(os_b, "cpp-mcts", lambda: os_b.search_cpp(budget, decisions, args.repeats))

    rows = []
    for h in hiddens:
        net = SharedNet(h)
        cells = [str(h), _cell(rf_b, f"net={h}", lambda net=net: rf_b.search(budget, decisions, args.repeats, net))]
        if os_b is not None and cpp is not None:
            cells.append(
                _cell(os_b, f"py net={h}", lambda net=net: os_b.search_python(budget, decisions, args.repeats, net))
            )
            cells.append(cpp)  # constant reference: the C++ bot ignores the net
        rows.append(tuple(cells))
    _table(
        f"Track B — searched decisions/sec on connect4 (UCT, budget={budget}, per-core; rows = shared-net size)",
        tuple(header),
        rows,
        note="reinfors & openspiel-py run the SAME shared net (rows = hidden units); openspiel-c++ is the "
        "native C++ MCTSBot (can't call a Python net), constant across rows. For a PYTHON net — the usual "
        "workflow — openspiel-py is OpenSpiel's ONLY option, so reinfors is ~4x faster there: a compiled "
        "search loop + a Python net, which OpenSpiel doesn't offer. openspiel-c++ (a C++ *rollout* proxy, "
        "NOT a C++ net) is a fast-loop reference; reinfors-vs-an-all-C++-net path is untested here.",
    )


def _track_c(active: list[Any], args: argparse.Namespace) -> None:
    rf_b = next((b for b in active if b.name == "reinfors"), None)
    pgx_b = next((b for b in active if b.name == "pgx"), None)
    if rf_b is None:
        return
    pgx_ok = pgx_b is not None
    if pgx_ok:
        try:
            import mctx  # noqa: F401
        except ImportError:
            pgx_ok = False
    budget = 64
    batches = [1, 8] if args.smoke else [1, 8, 64, 512, 4096]
    decisions = 40 if args.smoke else 400
    net = SharedNet(64)

    header = ["batch (n_games)", "reinfors [cpu]"]
    if pgx_ok and pgx_b is not None:
        header.append(f"pgx+mctx [{pgx_b.device()}]")
    rows = []
    for bs in batches:
        cells = [
            str(bs),
            _cell(rf_b, f"n_games={bs}", lambda bs=bs: rf_b.search(budget, decisions, args.repeats, net, bs)),
        ]
        if pgx_ok and pgx_b is not None:
            cells.append(
                _cell(pgx_b, f"mcts batch={bs}", lambda bs=bs: pgx_b.search_mcts(budget, decisions, args.repeats, bs))
            )
        rows.append(tuple(cells))
    _table(
        f"Track C — batched MCTS decisions/sec vs batch (UCT, budget={budget}; reinfors n_games == pgx batch)",
        tuple(header),
        rows,
        note="THE board-game search comparison: both parallelize across B games (reinfors: Rust trees + one "
        "batched net call/round; pgx+mctx: B JAX trees vmapped on-device). Same budget + net size (weights "
        "differ). pgx+mctx is UNVALIDATED / CPU here — its regime is GPU; run on an accelerator for the real "
        "curve. Expect reinfors to lead at small batch (no compile/JAX overhead), pgx+mctx at large batch on GPU.",
    )


def run(args: argparse.Namespace) -> None:
    _warn_if_not_release()
    print(f"host: {platform.platform()} | {platform.processor() or 'cpu'} x{os.cpu_count()}")
    active = []
    for b in BACKENDS:
        ok, detail = b.available()
        print(f"  backend {b.name:10s} {'OK' if ok else 'skip'} — {detail}")
        if ok:
            active.append(b)
    _track_a(active, args)
    _track_b(active, args)
    _track_c(active, args)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repeats", type=int, default=3, help="timed runs per measurement (median reported)")
    p.add_argument("--smoke", action="store_true", help="tiny/fast run to shake out the optional backends")
    run(p.parse_args())


if __name__ == "__main__":
    main()
