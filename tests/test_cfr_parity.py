"""ITERATION-EXACT parity with OpenSpiel's CFR: our vanilla/CFR+ update scheme mirrors
pyspiel's `cfr.py` (alternating passes, policy tables materialized between passes, RM+ clamp
and linear averaging for CFR+), so the exploitability trajectory must match to floating-point
noise at every checkpoint — on Kuhn AND Leduc, whose trees exercise root chance chains,
interior chance nodes, fold/call/raise grammars, and both bet sizes. MCCFR is excluded (its
sampling stream is ours, not pyspiel's). Dev-oracle only: skipped without pyspiel.
"""

from typing import Any, cast

import pytest
import reinfors as rf

pyspiel = pytest.importorskip("pyspiel")

CHECKPOINTS = [1, 2, 5, 20, 60]


@pytest.mark.parametrize(
    ("game_name", "ours_game"),
    [
        ("kuhn_poker", rf.games.KuhnPoker),
        ("leduc_poker", rf.games.LeducPoker),
        ("kuhn_poker(players=3)", lambda: rf.games.KuhnPoker(players=3)),
    ],
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

        # pyspiel's exploitability helper is 2-player-only; nash_conv covers every N and the
        # two are related by exactly 1/num_players.
        ps_nc = cast(float, ps_expl.nash_conv(ps_game, theirs.average_policy()))
        assert abs(ps_nc - ours.nash_conv()) < 1e-10, f"iteration {k}: {ours.nash_conv()} vs pyspiel {ps_nc}"
        assert abs(ours.exploitability() - ours.nash_conv() / ps_game.num_players()) < 1e-12


def test_three_player_nash_conv_matches_pyspiel() -> None:
    """The N-player measurement surface, pinned against pyspiel's nash_conv/exploitability on
    the 3-player Kuhn average profile after a short vanilla solve."""
    from open_spiel.python.algorithms import cfr as ps_cfr  # pyright: ignore[reportMissingImports]
    from open_spiel.python.algorithms import exploitability as ps_expl  # pyright: ignore[reportMissingImports]

    ps_game = pyspiel.load_game("kuhn_poker(players=3)")
    theirs = ps_cfr.CFRSolver(ps_game)
    ours = rf.solvers.Cfr(rf.games.KuhnPoker(players=3), variant="vanilla", seed=0)
    for _ in range(20):
        theirs.evaluate_and_update_policy()
    ours.iterate(20)
    theirs_nc = cast(float, ps_expl.nash_conv(ps_game, theirs.average_policy()))
    assert abs(ours.nash_conv() - theirs_nc) < 1e-9, (ours.nash_conv(), theirs_nc)
    assert abs(ours.exploitability() - theirs_nc / 3) < 1e-9
    brs = ours.best_response_values()
    assert len(brs) == 3
    evs = [ours.expected_value(p) for p in range(3)]
    assert abs(sum(br - ev for br, ev in zip(brs, evs, strict=True)) - ours.nash_conv()) < 1e-12


def test_three_player_nash_conv_falls_to_a_plateau() -> None:
    """The honest N>2 claim: NashConv FALLS and STABILIZES — never asserted to reach zero
    (regret minimization carries no equilibrium guarantee past two players)."""
    ours = rf.solvers.Cfr(rf.games.KuhnPoker(players=3), variant="vanilla", seed=0)
    ours.iterate(1)
    start = ours.nash_conv()
    trail = []
    for _ in range(12):
        ours.iterate(25)
        trail.append(ours.nash_conv())
    assert trail[-1] < start / 3, f"NashConv must drop substantially: {start} -> {trail[-1]}"
    spread = max(trail[-4:]) - min(trail[-4:])
    assert spread < 0.05, f"and stabilize (plateau), got tail spread {spread}: {trail[-4:]}"
