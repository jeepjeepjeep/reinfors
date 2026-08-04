from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
import reinfors as rf


def test_documented_runtime_types_are_top_level_exports() -> None:
    assert rf.CollectStream is rf._reinfors.CollectStream
    assert rf.DeepCfrBatch is rf._reinfors.DeepCfrBatch


def test_catalogue_names_match_runtime_registries() -> None:
    assert set(rf.games.registered()) == set(rf.catalog.GAMES)
    assert set(rf.policies.registered()) == set(rf.catalog.POLICIES)
    assert set(rf.learners.registered()) == set(rf.catalog.LEARNERS)
    assert set(rf.encoders.registered()) == set(rf.catalog.ENCODERS)
    assert set(rf.chance_modes.registered()) == set(rf.catalog.CHANCE_MODES)
    assert set(rf.noise.registered()) == set(rf.catalog.NOISE)


@pytest.mark.parametrize("game_name", rf.catalog.GAMES)
def test_catalogue_reward_defaults_match_runtime_schema(game_name: str) -> None:
    game = GAME_FACTORIES[game_name]()
    engine = rf.Engine(game, None, rf.policies.EpsilonGreedyQ(), rf.learners.Dqn(), n_games=1)
    assert engine.resolved_config()["reward"] == dict(rf.catalog.GAMES[game_name].reward_keys)


@pytest.mark.parametrize("encoder_name", rf.catalog.ENCODER_INFO)
def test_catalogue_encoder_default_shapes_match_runtime(encoder_name: str) -> None:
    info = rf.catalog.ENCODER_INFO[encoder_name]
    encoder = getattr(rf.encoders, info.label)()
    shape = rf.games.Chess(encoder=encoder).observation_space().shape
    assert str(shape) in info.shape


GAME_FACTORIES: dict[str, Callable[[], Any]] = {
    "backgammon": rf.games.Backgammon,
    "chess": rf.games.Chess,
    "connect4": rf.games.Connect4,
    "gridworld": rf.games.GridWorld,
    "kuhn_poker": rf.games.KuhnPoker,
    "leduc_poker": rf.games.LeducPoker,
    "snake": rf.games.Snake,
    "texas_holdem": rf.games.TexasHoldem,
}


def build_workflow(game: Any, algorithm: str) -> Any:
    if algorithm == "dqn":
        return rf.Engine(game, None, rf.policies.EpsilonGreedyQ(), rf.learners.Dqn(), n_games=1)
    if algorithm == "treestrap_expectimax":
        return rf.Engine(
            game,
            None,
            rf.policies.SelectiveExpectimax(n_heads=1),
            rf.learners.TreeStrap(),
            n_games=1,
        )
    if algorithm == "treestrap_mcts":
        return rf.Engine(game, None, rf.policies.Mcts(), rf.learners.TreeStrap(), n_games=1)
    if algorithm == "alphazero":
        return rf.Engine(game, None, rf.policies.AlphaZero(), rf.learners.AlphaZero(), n_games=1)
    if algorithm == "cfr":
        return rf.solvers.Cfr(game, variant="plus")
    if algorithm == "external_mccfr":
        return rf.solvers.Cfr(game, variant="external_mccfr")
    if algorithm == "deep_cfr":
        return rf.solvers.DeepCfr(game)
    raise AssertionError(f"catalogue has no compatibility test for {algorithm!r}")


@pytest.mark.parametrize("game_name", rf.catalog.GAMES)
@pytest.mark.parametrize("algorithm", rf.catalog.ALGORITHMS)
def test_compatibility_catalogue_matches_builtin_construction(game_name: str, algorithm: str) -> None:
    expected = algorithm in rf.catalog.GAMES[game_name].algorithms
    if expected:
        build_workflow(GAME_FACTORIES[game_name](), algorithm)
    else:
        with pytest.raises(ValueError, match=r"compatibility\.md"):
            build_workflow(GAME_FACTORIES[game_name](), algorithm)


@pytest.mark.parametrize("game_name", rf.catalog.GAMES)
def test_reached_state_start_catalogue_matches_builtin_construction(game_name: str) -> None:
    game = GAME_FACTORIES[game_name]()
    expected = rf.catalog.GAMES[game_name].reached_state_starts

    def build() -> Any:
        return rf.Engine(
            game,
            None,
            rf.policies.EpsilonGreedyQ(),
            rf.learners.Dqn(),
            n_games=1,
            start_buffer=True,
        )

    if expected:
        build()
    else:
        with pytest.raises(ValueError, match=r"compatibility\.md"):
            build()
