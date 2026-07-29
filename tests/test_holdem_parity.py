"""Position-for-position parity with OpenSpiel's `universal_poker` (ACPC) on matched fixed-limit
configs: identical legal-action sets at every decision and identical terminal chip deltas, over
seeded random hands at 2-4 seats. Our env is the dealer — pyspiel's chance nodes are forced to
our cards (ACPC card id = rank * 4 + suit, our exact layout).

Scope note: ACPC LIMIT games carry no stacks (all-in does not exist there), so parity runs deep
enough that the betting caps bound every commitment — the all-in/side-pot logic is covered by
the Rust unit tests instead. Dev-oracle only: skipped wherever pyspiel is not installed.
"""

from typing import Any

import numpy as np
import pytest
import reinfors as rf

pyspiel = pytest.importorskip("pyspiel")

STACK = 1_000_000  # deep: the limit caps bound commitments at 240, so no all-in is reachable
SB, BB = 5, 10


def _pyspiel_game(n: int) -> Any:
    blind = " ".join(str(x) for x in [SB, BB] + [0] * (n - 2))
    first = "1 2 2 2" if n == 2 else "3 1 1 1"
    return pyspiel.load_game(
        "universal_poker",
        {
            "betting": "limit",
            "numPlayers": n,
            "numRounds": 4,
            "blind": blind,
            "firstPlayer": first,
            "raiseSize": f"{BB} {BB} {2 * BB} {2 * BB}",
            "maxRaises": "3 4 4 4",
            "numSuits": 4,
            "numRanks": 13,
            "numHoleCards": 2,
            "numBoardCards": "0 3 1 1",
        },
    )


@pytest.mark.parametrize("n", [2, 3, 4])
def test_legal_actions_and_payoffs_match_universal_poker(n: int) -> None:
    game = _pyspiel_game(n)
    env = rf.Env(
        rf.games.TexasHoldem(num_players=n, stack=STACK, small_blind=SB, big_blind=BB),
        rf.Reward(),
        seed=17 * n,
    )
    rng = np.random.default_rng(n)
    for _hand in range(500):
        env.reset()
        st = env.state()
        # Seat mapping: our small blind becomes pyspiel seat 0 (positions are fixed by the ACPC
        # config; our button rotates per episode).
        sb_seat = int(st["button"]) if n == 2 else (int(st["button"]) + 1) % n

        def to_ps(s: int, sb: int = sb_seat) -> int:
            return (s - sb) % n

        ps = game.new_initial_state()
        # Hole cards, player-major in pyspiel seat order, forced to our deal.
        for p in range(n):
            ours = st["hole"][(sb_seat + p) % n]
            for c in ours:
                assert ps.is_chance_node()
                ps.apply_action(int(c))
        board_seen = 0
        while not env.done():
            board = env.state()["board"]
            while board_seen < len(board):
                assert ps.is_chance_node(), "board reveal must be a pyspiel chance node"
                ps.apply_action(int(board[board_seen]))
                board_seen += 1
            (agent,) = env.active_agents()
            assert ps.current_player() == to_ps(agent), "actors must align"
            ours = sorted(env.legal_actions(agent))
            theirs = sorted(ps.legal_actions())
            assert ours == theirs, f"legal sets diverge: {ours} vs {theirs}"
            action = int(rng.choice(ours))
            env.step({agent: action})
            ps.apply_action(action)
        board = env.state()["board"]
        while board_seen < len(board):  # a river reveal can land on the terminal step
            ps.apply_action(int(board[board_seen]))
            board_seen += 1
        assert ps.is_terminal(), "both games must end together"
        rewards = env.rewards
        assert rewards is not None
        returns = ps.returns()
        for seat in range(n):
            ours, theirs = rewards[seat], returns[to_ps(seat)]
            if abs(theirs - round(theirs)) < 1e-9:
                assert abs(ours - theirs) < 1e-9, f"payoff diverges at seat {seat}: {ours} vs {theirs}"
            else:
                # Split pots: ACPC divides odd amounts FRACTIONALLY (half/third chips); real
                # poker — and we — award discrete odd chips (earliest seat after the button).
                # The difference is bounded below one chip per seat.
                assert abs(ours - theirs) < 1.0, (
                    f"split payoff diverges past the odd chip at seat {seat}: {ours} vs {theirs}"
                )
