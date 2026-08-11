"""Play Connect4 in a terminal through reinfors.Env."""

from __future__ import annotations

import argparse
import random

import reinfors as rf

SYMBOLS = {0: ".", 1: "X", 2: "O"}


def render_board(board: list[list[int]]) -> str:
    """Render Connect4's row-zero-at-the-bottom inspection state."""
    rows = ["| " + " ".join(SYMBOLS[cell] for cell in row) + " |" for row in reversed(board)]
    return "\n".join([*rows, "+---------------+", "  1 2 3 4 5 6 7"])


def read_action(player: int, legal: list[int]) -> int | None:
    choices = ", ".join(str(action + 1) for action in legal)
    while True:
        try:
            value = input(f"Player {SYMBOLS[player + 1]} — choose column [{choices}], or q: ").strip().lower()
        except EOFError:
            return None
        if value in {"q", "quit"}:
            return None
        try:
            action = int(value) - 1
        except ValueError:
            print("Enter a column number or q.")
            continue
        if action in legal:
            return action
        print(f"Column {value} is not available.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opponent", choices=("random", "human"), default="random")
    parser.add_argument("--seed", type=int, default=0, help="seed for the random opponent")
    args = parser.parse_args()

    env = rf.Env(rf.games.Connect4(), seed=args.seed)
    rng = random.Random(args.seed)
    env.reset()
    events = []

    print("Connect4: make four in a row. Player X moves first.\n")
    try:
        while not env.done():
            print(render_board(env.state()["board"]))
            (player,) = env.active_agents()
            legal = env.legal_actions(player)

            if player == 0 or args.opponent == "human":
                action = read_action(player, legal)
                if action is None:
                    print("Game ended.")
                    return
            else:
                action = rng.choice(legal)
                print(f"Computer O chooses column {action + 1}.\n")

            events = env.step({player: action})
    except KeyboardInterrupt:
        print("\nGame ended.")
        return

    print(render_board(env.state()["board"]))
    winner = next((player for player, event in events if event == "win"), None)
    print("Draw." if winner is None else f"Player {SYMBOLS[winner + 1]} wins!")


if __name__ == "__main__":
    main()
