"""Cross-framework benchmark on connect4 (Phase 2). MANUAL-RUN — not CI.

Compares reinfors against the packages it's genuinely comparable to, on the one game they all implement
(connect4): **Pgx** (JAX, GPU/TPU-resident, batched) and **OpenSpiel** (C++ core, MCTS). Because these
have different execution models and run on different hardware, there is no single fair number — so this
reports per *track*, swept over batch size / search budget, with **every row tagged by device**. Never
read a GPU row against a CPU row as one ratio; read each within its track and hardware.

Two tracks:
  * Track A — raw env transitions/sec (apply one action, advance one state). reinfors and OpenSpiel are
    single-core CPU (a "batch" is just N independent envs stepped in a loop — they don't vectorize raw
    stepping), so their curves are ~flat in batch; Pgx vmaps on the accelerator, so its curve rises with
    batch. That contrast is the finding. reinfors' single-env number UNDERSELLS it — its product is
    Track B (parallel, search-driven), not raw stepping.
  * Track B — searched decisions/sec at a fixed budget: reinfors `Mcts` (UCT) vs OpenSpiel `MCTSBot`,
    now a fully-controlled race — same algorithm (both UCT, matching `uct_c` and `num_simulations`) AND
    the same leaf evaluator (a shared small value net; see `SharedNet`), fed the same canonical board. So
    it isolates the search *implementation*: the difference is that reinfors runs the whole loop in Rust
    and batches the net across each round's pooled leaves, where OpenSpiel drives it per-leaf from Python.

Install (on the machine you're benchmarking): `pip install jax[cuda12] pgx open_spiel` (adjust the jax
wheel for your accelerator). reinfors must be a RELEASE build — this checks and warns.

    uv run --with numpy --with pgx --with jax[cuda12] --with open_spiel python scripts/benchmark_vs.py

Backends whose deps aren't importable are skipped with a note; the reinfors backend always runs, and a
per-cell error in any backend degrades to `ERR` (never a crash). The Pgx and OpenSpiel backends have
been run on CPU (validating their APIs), but Pgx's headline regime — large batch on a GPU/TPU — has not;
run on your accelerator for those numbers.
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
# frameworks' MCTS, so the leaf evaluation is a genuinely shared, equal-cost forward — not reinfors'
# zeros vs OpenSpiel's random rollouts. Canonical board: 42-dim (+1 own, -1 opp, 0 empty) from the
# current player's perspective, index r*7+c with row 0 = bottom — the layout reinfors' Connect4Planes
# and OpenSpiel's `observation_tensor` both produce (a test asserts they agree).
# --------------------------------------------------------------------------------------------------
_HIDDEN = 64


class SharedNet:
    def __init__(self) -> None:
        rng = np.random.default_rng(0)
        self.w1 = (rng.standard_normal((42, _HIDDEN)) / np.sqrt(42)).astype(np.float32)
        self.w2 = (rng.standard_normal((_HIDDEN, 1)) / np.sqrt(_HIDDEN)).astype(np.float32)

    def value(self, board: np.ndarray) -> np.ndarray:
        """(N, 42) canonical board -> (N,) value in (-1, 1), from the board's current-player perspective."""
        h = np.maximum(board.astype(np.float32) @ self.w1, 0.0)
        return np.tanh(h @ self.w2)[:, 0]


_SHARED_NET = SharedNet()


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
# Backends. Each: name, device(), available() -> (ok, detail), raw_step(batch, steps) -> transitions/s,
# search(budget, decisions) -> decisions/s or None. A backend is fully self-contained and isolated, so a
# missing/broken one never taints the others.
# --------------------------------------------------------------------------------------------------


class ReinforsBackend:
    name = "reinfors"
    validated = True
    supports_search = True

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

    def search(self, budget: int, decisions: int, repeats: int) -> float | None:
        # Genuine UCT MCTS on connect4, ONE game (per-core, for a fair head-to-head vs a single MCTS bot
        # — reinfors additionally scales across cores, see Phase 1). `budget` = num_simulations and uct_c
        # match the OpenSpiel backend, and both evaluate leaves with the SAME shared net — so this is a
        # controlled UCT-vs-UCT race. reinfors batches the net across the round's pooled leaves.
        action_count = rf.games.Connect4().action_space().n
        engine = rf.Engine(
            rf.games.Connect4(),
            rf.Reward(win=1.0, loss=-1.0, draw=0.0),
            rf.policies.Mcts(num_simulations=budget, uct_c=2.0, max_depth=64),
            rf.learners.TreeStrap(gamma=0.99, outcome_weight=0.3, bootstrap_p=1.0, interior_targets=False),
            n_games=1,
            seed=0,
        )

        def infer(arr: np.ndarray) -> np.ndarray:
            # State value from the shared net, broadcast across actions (MCTS uses the max = the value).
            v = _SHARED_NET.value(board_from_reinfors(arr))
            out = np.empty((arr.shape[0], 1, action_count), dtype=np.float64)
            out[:, 0, :] = v[:, None]
            return out

        return _throughput(lambda: int(engine.collect(decisions, infer).obs.shape[0]), repeats)


class PgxBackend:
    name = "pgx"
    validated = True  # runs on jax/pgx (CPU) here; its headline regime — large batch on GPU/TPU — is unrun
    supports_search = False  # batched MCTS via mctx is a follow-up

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

    def search(self, budget: int, decisions: int, repeats: int) -> float:
        raise NotImplementedError("batched MCTS via mctx is deferred to a follow-up")


class OpenSpielBackend:
    name = "openspiel"
    validated = True  # runs on open_spiel (CPU) here
    supports_search = True

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

    def search(self, budget: int, decisions: int, repeats: int) -> float | None:
        from open_spiel.python.algorithms import mcts

        game = self._game()

        class NetEvaluator(mcts.Evaluator):
            """Evaluate connect4 leaves with the shared net (value only; uniform prior — no policy head),
            so this is the SAME leaf eval reinfors' MCTS runs."""

            def evaluate(self, state: Any) -> np.ndarray:
                v = float(_SHARED_NET.value(board_from_openspiel(state))[0])
                return np.array([v, -v]) if state.current_player() == 0 else np.array([-v, v])

            def prior(self, state: Any) -> list[tuple[int, float]]:
                legal = state.legal_actions()
                p = 1.0 / len(legal)
                return [(a, p) for a in legal]

        bot = mcts.MCTSBot(game, uct_c=2.0, max_simulations=budget, evaluator=NetEvaluator())

        def work() -> int:
            state = game.new_initial_state()
            for _ in range(decisions):
                if state.is_terminal():
                    state = game.new_initial_state()
                state.apply_action(bot.step(state))  # one MCTS decision of `budget` simulations
            return decisions

        return _throughput(work, repeats)


BACKENDS = [ReinforsBackend(), PgxBackend(), OpenSpielBackend()]


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
    """One measured table cell, isolated: a backend failure (e.g. an UNVALIDATED backend's API drift)
    becomes an 'ERR' cell + a one-line note, never a crash that takes the whole comparison down."""
    try:
        return f"{fn():,.0f}"
    except Exception as e:  # a benchmark cell must never propagate a backend's error
        print(f"  ! {backend.name} {label} failed: {type(e).__name__}: {str(e).splitlines()[0][:100]}")
        return "ERR"


def run(args: argparse.Namespace) -> None:
    _warn_if_not_release()
    print(f"host: {platform.platform()} | {platform.processor() or 'cpu'} x{os.cpu_count()}")
    active = []
    for b in BACKENDS:
        ok, detail = b.available()
        tag = "" if b.validated else "  [UNVALIDATED]"
        print(f"  backend {b.name:10s} {'OK' if ok else 'skip'} — {detail}{tag if ok else ''}")
        if ok:
            active.append(b)

    batches = [1, 8] if args.smoke else [1, 16, 256, 4096]
    steps = 50 if args.smoke else 2000
    rows_a = []
    for bs in batches:
        cells = tuple(
            _cell(b, f"raw_step(batch={bs})", lambda b=b, bs=bs: b.raw_step(bs, steps, args.repeats)) for b in active
        )
        rows_a.append((str(bs), *cells))
    _table(
        "Track A — raw env transitions/sec on connect4 (device in header; CPU rows ~flat, Pgx scales)",
        ("batch", *(f"{b.name} [{b.device()}]" for b in active)),
        rows_a,
        note="reinfors/OpenSpiel don't vectorize raw stepping (batch = N sequential envs); "
        "Track B is reinfors' real product.",
    )

    searchers = [b for b in active if b.supports_search]
    budgets = [16, 64] if args.smoke else [16, 64, 256]
    decisions = 20 if args.smoke else 200
    if searchers:
        rows_b = []
        for bud in budgets:
            cells = tuple(
                _cell(b, f"search(budget={bud})", lambda b=b, bud=bud: b.search(bud, decisions, args.repeats))
                for b in searchers
            )
            rows_b.append((str(bud), *cells))
        _table(
            "Track B — searched decisions/sec on connect4 (budget = UCT simulations; per-core)",
            ("budget", *(f"{b.name} [{b.device()}]" for b in searchers)),
            rows_b,
            note="both UCT with the SAME shared net (matched uct_c + budget) — a controlled implementation "
            "race. reinfors batches the net across pooled leaves + scales across cores (Phase 1).",
        )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repeats", type=int, default=3, help="timed runs per measurement (median reported)")
    p.add_argument("--smoke", action="store_true", help="tiny/fast run to shake out the optional backends")
    run(p.parse_args())


if __name__ == "__main__":
    main()
