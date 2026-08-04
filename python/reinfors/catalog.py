"""Machine-readable catalogue metadata used by the documentation.

This module deliberately has no extension-module imports, so documentation can be generated
without compiling Rust. Runtime registry tests ensure these names stay aligned with the public
constructors. Narrative documentation remains handwritten; only facts that routinely drift live
here.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GameInfo:
    label: str
    players: str
    dynamics: str
    chance: str
    information: str
    actions: str
    observation: str
    reward_keys: tuple[tuple[str, float], ...]
    adapters: str
    reached_state_starts: bool
    algorithms: tuple[str, ...]
    summary: str


@dataclass(frozen=True)
class AlgorithmInfo:
    label: str
    workflow: str
    policy_or_solver: str
    learner: str | None
    training_output: str
    example_label: str
    example_anchor: str
    players: str
    dynamics: str
    chance: str
    information: str
    network: str
    summary: str
    references: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class EncoderInfo:
    label: str
    game: str
    shape: str
    constructor: str
    summary: str


GAMES: dict[str, GameInfo] = {
    "backgammon": GameInfo(
        "Backgammon",
        "2",
        "Sequential",
        "Dice",
        "Perfect",
        "1,352 discrete ids",
        "Fixed CHW tensor",
        (("win", 1.0), ("gammon", 2.0), ("backgammon", 3.0)),
        "PettingZoo AEC",
        False,
        ("dqn", "treestrap_expectimax", "treestrap_mcts", "alphazero"),
        "Standard play without the doubling cube; win, gammon and backgammon outcomes.",
    ),
    "chess": GameInfo(
        "Chess",
        "2",
        "Sequential",
        "No",
        "Perfect",
        "AlphaZero 8x8x73",
        "Selectable fixed CHW encoder",
        (("win", 1.0), ("loss", -1.0), ("draw", 0.0)),
        "PettingZoo AEC",
        False,
        ("dqn", "treestrap_expectimax", "treestrap_mcts", "alphazero"),
        "Standard chess with minimal, relative, OpenSpiel and AlphaZero observation views.",
    ),
    "connect4": GameInfo(
        "Connect 4",
        "2",
        "Sequential",
        "No",
        "Perfect",
        "7 columns",
        "Fixed CHW tensor",
        (("win", 1.0), ("loss", -1.0), ("draw", 0.0)),
        "PettingZoo AEC",
        False,
        ("dqn", "treestrap_expectimax", "treestrap_mcts", "alphazero"),
        "Compact deterministic benchmark for value learning, MCTS and AlphaZero.",
    ),
    "gridworld": GameInfo(
        "GridWorld",
        "1",
        "Sequential",
        "No",
        "Perfect",
        "4 directions",
        "Fixed CHW tensor",
        (("step", 0.0), ("goal", 1.0)),
        "Gymnasium",
        False,
        ("dqn", "treestrap_expectimax", "treestrap_mcts", "alphazero"),
        "Small single-agent environment for checking basic training and integration loops.",
    ),
    "kuhn_poker": GameInfo(
        "Kuhn poker",
        "2-10",
        "Sequential",
        "Card deal",
        "Imperfect",
        "Pass / bet",
        "Fixed information-state tensor",
        (("scale", 1.0),),
        "PettingZoo AEC",
        False,
        ("dqn", "cfr", "external_mccfr", "deep_cfr"),
        "OpenSpiel-compatible N-player Kuhn poker for CFR and Deep CFR experiments.",
    ),
    "leduc_poker": GameInfo(
        "Leduc poker",
        "2",
        "Sequential",
        "Card deal",
        "Imperfect",
        "Fold / call / raise",
        "Fixed information-state tensor",
        (("scale", 1.0),),
        "PettingZoo AEC",
        False,
        ("dqn", "cfr", "external_mccfr", "deep_cfr"),
        "Two-round imperfect-information benchmark between Kuhn and full Hold'em.",
    ),
    "snake": GameInfo(
        "Snake",
        "2-8",
        "Simultaneous",
        "Placement and respawn",
        "Perfect",
        "3 relative moves",
        "Egocentric fixed CHW tensor",
        (
            ("step", 0.0),
            ("food", 0.0),
            ("loss", -1.0),
            ("draw", 0.0),
            ("kill", 0.0),
            ("win", 1.0),
            ("survival", 0.0),
        ),
        "PettingZoo Parallel",
        True,
        ("dqn", "treestrap_expectimax", "treestrap_mcts", "alphazero"),
        "Simultaneous multiplayer game with dynamic bodies and explicit respawn chance.",
    ),
    "texas_holdem": GameInfo(
        "Texas Hold'em",
        "2-9",
        "Sequential",
        "Deal and board",
        "Imperfect",
        "Fold / call / raise",
        "Egocentric fixed CHW tensor",
        (("scale", 1.0),),
        "PettingZoo AEC",
        False,
        ("dqn", "external_mccfr", "deep_cfr"),
        "Multiway no-limit-style poker surface with all-ins, side pots and chance runouts.",
    ),
}


ENCODER_INFO: dict[str, EncoderInfo] = {
    "snake": EncoderInfo(
        "Snake",
        "Snake",
        "(5, grid_size, grid_size); (5, 20, 20) by default",
        "rf.encoders.Snake()",
        "Mover-relative body, opponent, food, and wall planes.",
    ),
    "connect4": EncoderInfo(
        "Connect4",
        "Connect 4",
        "(2, 6, 7)",
        "rf.encoders.Connect4()",
        "Acting-player and opponent piece planes.",
    ),
    "minimal_chess": EncoderInfo(
        "MinimalChess",
        "Chess",
        "(19, 8, 8)",
        "rf.encoders.MinimalChess()",
        "Absolute piece, turn, castling, en-passant and clock planes; the Chess default.",
    ),
    "relative_chess": EncoderInfo(
        "RelativeChess",
        "Chess",
        "(19, 8, 8)",
        "rf.encoders.RelativeChess()",
        "Mover-relative board and matching action-head mapping.",
    ),
    "openspiel_chess": EncoderInfo(
        "OpenSpielChess",
        "Chess",
        "(20, 8, 8)",
        "rf.encoders.OpenSpielChess()",
        "OpenSpiel-compatible observation for parity and benchmarking.",
    ),
    "alphazero_chess": EncoderInfo(
        "AlphaZeroChess",
        "Chess",
        "(119, 8, 8) by default",
        "rf.encoders.AlphaZeroChess(history_length=8)",
        "Uses `14 * history_length + 7` channels for history, repetition, side, move-count, "
        "castling and clock features.",
    ),
    "backgammon": EncoderInfo(
        "Backgammon",
        "Backgammon",
        "(200, 1, 1)",
        "rf.encoders.Backgammon()",
        "Tesauro-style point, bar, borne-off, turn, and dice features.",
    ),
    "texas_holdem": EncoderInfo(
        "TexasHoldem",
        "Texas Hold'em",
        "(2 * num_players + 19, 4, 13); (31, 4, 13) by default",
        "rf.encoders.TexasHoldem()",
        "Player-relative cards, betting state, stacks, and positions.",
    ),
    "kuhn_poker": EncoderInfo(
        "KuhnPoker",
        "Kuhn poker",
        "(3 * players, 1, 1); (6, 1, 1) by default",
        "rf.encoders.KuhnPoker()",
        "Private card and public betting-history information state.",
    ),
    "leduc_poker": EncoderInfo(
        "LeducPoker",
        "Leduc poker",
        "(21, 1, 1)",
        "rf.encoders.LeducPoker()",
        "Private/public cards and two-round betting information state.",
    ),
    "gridworld": EncoderInfo(
        "GridWorld",
        "GridWorld",
        "(2, size, size); (2, 5, 5) by default",
        "rf.encoders.GridWorld()",
        "Agent-position and goal planes.",
    ),
}


ALGORITHMS: dict[str, AlgorithmInfo] = {
    "dqn": AlgorithmInfo(
        "DQN",
        "Engine",
        "rf.policies.EpsilonGreedyQ",
        "rf.learners.Dqn",
        "DqnBatch",
        "GridWorld DQN training example",
        "train-gridworld",
        "Any N",
        "Sequential or simultaneous",
        "Sampled by the engine",
        "Perfect or imperfect",
        "Q values: (rows, heads, actions)",
        "Observation-only off-policy value learning with sparse legal-action records.",
        (("Human-level control through deep reinforcement learning", "https://doi.org/10.1038/nature14236"),),
    ),
    "treestrap_expectimax": AlgorithmInfo(
        "TreeStrap + selective expectimax",
        "Engine",
        "rf.policies.SelectiveExpectimax",
        "rf.learners.TreeStrap",
        "TreeStrapBatch",
        "TreeStrap training example",
        "treestrap-snake",
        "Any N",
        "Sequential or simultaneous",
        "Committed or ExpandAll",
        "Perfect only",
        "Ensemble action values",
        "Best-first search whose backed-up values become supervised learning targets.",
        (
            (
                "Bootstrapping from Game Tree Search",
                "https://cgi.cse.unsw.edu.au/~blair/pubs/2009VenessSilverUtherBlairNIPS.pdf",
            ),
        ),
    ),
    "treestrap_mcts": AlgorithmInfo(
        "TreeStrap + UCT MCTS",
        "Engine",
        "rf.policies.Mcts",
        "rf.learners.TreeStrap",
        "TreeStrapBatch",
        "TreeStrap MCTS training example",
        "treestrap-snake",
        "Sequential <=2; simultaneous N",
        "Sequential or simultaneous",
        "AlwaysResample, Committed, or ExpandAll",
        "Perfect only",
        "One ensemble head of action values",
        "UCT for sequential games and decoupled UCT for simultaneous multiplayer games.",
        (("UCT", "https://doi.org/10.1007/11871842_29"),),
    ),
    "alphazero": AlgorithmInfo(
        "AlphaZero",
        "Engine",
        "rf.policies.AlphaZero",
        "rf.learners.AlphaZero",
        "AlphaZeroBatch",
        "AlphaZero training example",
        "alphazero-connect-4",
        "Any N",
        "Sequential or simultaneous",
        "AlwaysResample, Committed, or ExpandAll",
        "Perfect only",
        "Policy logits (rows, actions) plus values (rows,)",
        "PUCT self-play with policy targets and outcome values. For sequential games, "
        '`sequential_backup="auto"` uses negamax at one or two players and MaxN above two; '
        '`"maxn"` forces MaxN at two players.',
        (("AlphaZero", "https://doi.org/10.1126/science.aar6404"),),
    ),
    "cfr": AlgorithmInfo(
        "CFR / CFR+",
        "Standalone solver",
        "rf.solvers.Cfr",
        None,
        "Internal strategy tables",
        "CFR solving example",
        "solve-leduc",
        "2-10",
        "Sequential",
        "Exact enumeration",
        "Imperfect-information native",
        "No network",
        "Tabular alternating-update counterfactual regret minimization.",
        (
            ("Counterfactual regret minimization", "https://doi.org/10.5555/1795814.1795895"),
            ("CFR+", "https://arxiv.org/abs/1407.5042"),
        ),
    ),
    "external_mccfr": AlgorithmInfo(
        "External-sampling MCCFR",
        "Standalone solver",
        'rf.solvers.Cfr(variant="external_mccfr")',
        None,
        "Internal strategy tables",
        "MCCFR solving example",
        "solve-leduc",
        "2-10",
        "Sequential",
        "Sampled",
        "Imperfect-information native",
        "No network",
        "Sampled CFR variant for games whose chance tree is too large to enumerate.",
        (
            (
                "Monte Carlo CFR",
                "https://proceedings.neurips.cc/paper/2009/hash/00411460f7c92d2124a67ea0f4cb5f85-Abstract.html",
            ),
        ),
    ),
    "deep_cfr": AlgorithmInfo(
        "Deep CFR",
        "Standalone data-generating solver",
        "rf.solvers.DeepCfr",
        None,
        "DeepCfrBatch",
        "Deep CFR training example",
        "deep-cfr-training",
        "Kuhn 2-10; Leduc 2; Hold'em 2-9",
        "Sequential",
        "Sampled",
        "Imperfect-information native",
        "Per-player advantage networks; average-policy samples",
        "Batched external-sampling traversals with caller-owned buffers and training.",
        (("Deep CFR", "https://proceedings.mlr.press/v97/brown19b.html"),),
    ),
}


# Public constructor registries. Keeping this small list here lets documentation generation stay
# Rust-free; runtime tests compare it with every actual `registered()` result.
POLICIES = frozenset({"alphazero", "epsilon_greedy_q", "mcts", "selective_expectimax"})
LEARNERS = frozenset({"alphazero", "dqn", "treestrap"})
ENCODERS = frozenset(ENCODER_INFO)
CHANCE_MODES = frozenset({"always_resample", "committed", "expand_all"})
NOISE = frozenset({"dirichlet"})


__all__ = [
    "ALGORITHMS",
    "CHANCE_MODES",
    "ENCODERS",
    "ENCODER_INFO",
    "GAMES",
    "LEARNERS",
    "NOISE",
    "POLICIES",
    "AlgorithmInfo",
    "EncoderInfo",
    "GameInfo",
]
