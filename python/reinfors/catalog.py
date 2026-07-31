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
    adapters: str
    summary: str


@dataclass(frozen=True)
class AlgorithmInfo:
    label: str
    workflow: str
    players: str
    dynamics: str
    chance: str
    information: str
    network: str
    summary: str
    references: tuple[tuple[str, str], ...]


GAMES: dict[str, GameInfo] = {
    "backgammon": GameInfo(
        "Backgammon",
        "2",
        "Sequential",
        "Dice",
        "Perfect",
        "1,352 discrete ids",
        "Fixed CHW tensor",
        "PettingZoo AEC",
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
        "PettingZoo AEC",
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
        "PettingZoo AEC",
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
        "Gymnasium",
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
        "PettingZoo AEC",
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
        "PettingZoo AEC",
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
        "PettingZoo Parallel",
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
        "PettingZoo AEC",
        "Multiway no-limit-style poker surface with all-ins, side pots and chance runouts.",
    ),
}


ALGORITHMS: dict[str, AlgorithmInfo] = {
    "dqn": AlgorithmInfo(
        "DQN",
        "Engine: EpsilonGreedyQ + Dqn",
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
        "Any N",
        "Sequential or simultaneous",
        "Committed samples or expand-all",
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
        "Sequential <=2; simultaneous N",
        "Sequential or simultaneous",
        "Resample, committed or expand-all",
        "Perfect only",
        "Single-head action values",
        "UCT for sequential games and decoupled UCT for simultaneous multiplayer games.",
        (("UCT", "https://doi.org/10.1007/11871842_29"),),
    ),
    "alphazero": AlgorithmInfo(
        "AlphaZero",
        "Engine: AlphaZero policy + learner",
        "Any N",
        "Sequential or simultaneous",
        "Resample, committed or expand-all",
        "Perfect only",
        "Policy logits plus per-player values",
        "PUCT self-play with policy targets, outcome values and sequential MaxN backup.",
        (("AlphaZero", "https://doi.org/10.1126/science.aar6404"),),
    ),
    "cfr": AlgorithmInfo(
        "CFR / CFR+",
        "Standalone solver",
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
        "2-10",
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
ENCODERS = frozenset({"alphazero_chess", "minimal_chess", "openspiel_chess", "relative_chess"})
CHANCE_MODES = frozenset({"always_resample", "committed", "expand_all"})
NOISE = frozenset({"dirichlet"})


__all__ = [
    "ALGORITHMS",
    "CHANCE_MODES",
    "ENCODERS",
    "GAMES",
    "LEARNERS",
    "NOISE",
    "POLICIES",
    "AlgorithmInfo",
    "GameInfo",
]
