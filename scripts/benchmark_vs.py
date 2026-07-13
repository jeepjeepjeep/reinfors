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
  * Track B — searched decisions/sec at a fixed budget: reinfors selective-expectimax vs OpenSpiel
    `MCTSBot`, trivial evaluator (zeros / random rollout), budget matched (expansions ~ simulations).
    CAVEAT: these are DIFFERENT algorithms — this measures how fast each turns a compute budget into a
    decision, not which searches better. A follow-up adds a genuine MCTS planner to reinfors for a
    like-for-like race; another swaps the trivial evaluator for a shared small net.

Install (on the machine you're benchmarking): `pip install jax[cuda12] pgx open_spiel` (adjust the jax
wheel for your accelerator). reinfors must be a RELEASE build — this checks and warns.

    uv run --with numpy --with pgx --with jax[cuda12] --with open_spiel python scripts/benchmark_vs.py

Backends whose deps aren't importable are skipped with a note; the reinfors backend always runs. The
Pgx and OpenSpiel backends are written to those projects' public APIs but were NOT executable in the
dev environment they were written in — they are marked UNVALIDATED and want a shakeout (`--smoke`) on
your machine before you trust their numbers.
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

_BUDGET = 64  # matched search budget: reinfors expansion_budget ~ OpenSpiel max_simulations


def _throughput(work: Any, repeats: int) -> float:
    work()  # untimed warm-up (compiles JAX, primes caches)
    rates = []
    for _ in range(repeats):
        t0 = perf_counter()
        units = work()
        rates.append(units / (perf_counter() - t0))
    return median(rates)


# --------------------------------------------------------------------------------------------------
# Backends. Each: name, device(), available() -> (ok, detail), raw_step(batch, steps) -> transitions/s,
# search(budget, decisions) -> decisions/s or None. A backend is fully self-contained and isolated, so a
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

    def search(self, budget: int, decisions: int, repeats: int) -> float | None:
        # Selective expectimax on connect4, ONE game (per-core, for a fair head-to-head vs a single MCTS
        # bot — reinfors additionally scales across cores, see Phase 1), trivial zeros evaluator.
        action_count = rf.games.Connect4().action_space().n
        engine = rf.Engine(
            rf.games.Connect4(),
            rf.Reward(win=1.0, loss=-1.0, draw=0.0),
            rf.policies.SelectiveExpectimax(
                expansion_budget=budget,
                top_k=max(1, budget // 8),
                max_depth=8,
                beta=1.0,
                food_samples=1,
                n_heads=1,
                epsilon=0.0,
                opponent="uniform",
                opp_temperature=1.0,
                opp_floor=0.1,
            ),
            rf.learners.TreeStrap(gamma=0.99, outcome_weight=0.3, bootstrap_p=1.0, interior_targets=False),
            n_games=1,
            seed=0,
        )

        def infer(arr: np.ndarray) -> np.ndarray:
            return np.zeros((arr.shape[0], 1, action_count), dtype=np.float64)

        return _throughput(lambda: int(engine.collect(decisions, infer).obs.shape[0]), repeats)


class PgxBackend:
    name = "pgx"
    validated = False  # UNVALIDATED — written to Pgx's public API, not run in the authoring env

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
        init = jax.jit(jax.vmap(env.init))
        step = jax.jit(jax.vmap(auto_reset(env.step, env.init)))  # reset terminated envs in-place
        keys = jax.random.split(jax.random.PRNGKey(0), batch)
        state = init(keys)

        def one(s: Any) -> Any:
            action = jnp.argmax(s.legal_action_mask, axis=-1)  # first legal action (deterministic)
            return step(s, action)

        def work() -> int:
            nonlocal state
            for _ in range(steps):
                state = one(state)
            jax.block_until_ready(state)
            return batch * steps

        return _throughput(work, repeats)

    def search(self, budget: int, decisions: int, repeats: int) -> float | None:
        return None  # batched MCTS via mctx is deferred to a follow-up


class OpenSpielBackend:
    name = "openspiel"
    validated = False  # UNVALIDATED — written to OpenSpiel's public API, not run in the authoring env

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
        evaluator = mcts.RandomRolloutEvaluator(n_rollouts=1)
        bot = mcts.MCTSBot(game, uct_c=2.0, max_simulations=budget, evaluator=evaluator)

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


def _fmt(x: float | None) -> str:
    return "—" if x is None else f"{x:,.0f}"


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
    _table(
        "Track A — raw env transitions/sec on connect4 (device in header; CPU rows ~flat, Pgx scales)",
        ("batch", *(f"{b.name} [{b.device()}]" for b in active)),
        [(str(bs), *(_fmt(b.raw_step(bs, steps, args.repeats)) for b in active)) for bs in batches],
        note="reinfors/OpenSpiel don't vectorize raw stepping (batch = N sequential envs); "
        "Track B is reinfors' real product.",
    )

    searchers = [b for b in active if b.search(_BUDGET, 1, 1) is not None]
    budgets = [16, 64] if args.smoke else [16, 64, 256]
    decisions = 20 if args.smoke else 200
    if searchers:
        _table(
            "Track B — searched decisions/sec on connect4 (budget = expansions ~ simulations; per-core)",
            ("budget", *(f"{b.name} [{b.device()}]" for b in searchers)),
            [(str(bud), *(_fmt(b.search(bud, decisions, args.repeats)) for b in searchers)) for bud in budgets],
            note="DIFFERENT algorithms (selective-expectimax vs MCTS) at matched budget — throughput, not quality. "
            "reinfors shown at n_games=1; it also scales across cores (Phase 1).",
        )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repeats", type=int, default=3, help="timed runs per measurement (median reported)")
    p.add_argument("--smoke", action="store_true", help="tiny/fast run to shake out the optional backends")
    run(p.parse_args())


if __name__ == "__main__":
    main()
