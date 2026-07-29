"""Minimal example: solve Leduc hold'em with CFR+ and watch exploitability fall toward Nash.

CFR is reinfors' first SOLVER — the third execution shape next to policies (act) and learners
(train): it owns its own traversal of the game (no engine, no network) and outputs the
time-AVERAGED strategy, queryable by `env.information_state_key(agent)`. Exploitability
(exact best-response improvement, zero at Nash) is the convergence metric; on this game CFR+
reaches milli-blind exploitability in a few hundred iterations and a few seconds.

    python scripts/solve_leduc.py --iterations 1000
"""

from __future__ import annotations

import argparse
import time

import reinfors as rf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--game", choices=["leduc_poker", "kuhn_poker"], default="leduc_poker")
    parser.add_argument("--variant", choices=["vanilla", "plus", "external_mccfr"], default="plus")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--checkpoints", type=int, default=8, help="exploitability probes")
    parser.add_argument("--save", help="write the solved tables to this path")
    args = parser.parse_args()

    solver = rf.solvers.Cfr(rf.games.make(args.game), variant=args.variant, seed=0)
    t0 = time.perf_counter()
    done = 0
    for i in range(1, args.checkpoints + 1):
        target = args.iterations * i // args.checkpoints
        solver.iterate(target - done)
        done = target
        print(
            f"iter {done:6d}  wall {time.perf_counter() - t0:6.1f}s  "
            f"infosets {solver.num_infosets:5d}  exploitability {solver.exploitability():.3e}"
        )
    print(f"P0 value under the average profile: {solver.expected_value(0):+.5f}")
    if args.save:
        with open(args.save, "wb") as f:
            f.write(solver.save())
        print(f"tables saved to {args.save}")


if __name__ == "__main__":
    main()
