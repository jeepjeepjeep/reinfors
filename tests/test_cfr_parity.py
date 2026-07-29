"""ITERATION-EXACT parity with OpenSpiel's CFR: our vanilla/CFR+ update scheme mirrors
pyspiel's `cfr.py` (alternating passes, policy tables materialized between passes, RM+ clamp
and linear averaging for CFR+), so the exploitability trajectory must match to floating-point
noise at every checkpoint — on Kuhn AND Leduc, whose trees exercise root chance chains,
interior chance nodes, fold/call/raise grammars, and both bet sizes. MCCFR is excluded (its
sampling stream is ours, not pyspiel's). Dev-oracle only: skipped without pyspiel.
"""

from typing import Any

import pytest
import reinfors as rf

pyspiel = pytest.importorskip("pyspiel")

CHECKPOINTS = [1, 2, 5, 20, 60]


@pytest.mark.parametrize(
    ("game_name", "ours_game"),
    [("kuhn_poker", rf.games.KuhnPoker), ("leduc_poker", rf.games.LeducPoker)],
)
@pytest.mark.parametrize("variant", ["vanilla", "plus"])
def test_exploitability_trajectories_match_pyspiel(game_name: str, ours_game: Any, variant: str) -> None:
    from open_spiel.python.algorithms import cfr as ps_cfr  # pyright: ignore[reportMissingImports]

    ps_game = pyspiel.load_game(game_name)
    theirs = (ps_cfr.CFRSolver if variant == "vanilla" else ps_cfr.CFRPlusSolver)(ps_game)
    ours = rf.solvers.Cfr(ours_game(), variant=variant, seed=0)
    done = 0
    for k in CHECKPOINTS:
        for _ in range(k - done):
            theirs.evaluate_and_update_policy()
        done = k
        ours.iterate(k - ours.iterations)
        from open_spiel.python.algorithms import exploitability as ps_expl  # pyright: ignore[reportMissingImports]

        ps_e = ps_expl.exploitability(ps_game, theirs.average_policy())
        our_e = ours.exploitability()
        assert abs(ps_e - our_e) < 1e-10, f"iteration {k}: {our_e} vs pyspiel {ps_e}"
