"""Chess through the Python surface: spaces, the AlphaZero and Mcts pairings at the 4672-action
width (legal masking end to end), Env play, and determinism."""

from typing import Any

import numpy as np
import pytest
import reinfors as rf
from conftest import az_zeros_infer

_A = 4672


_az_infer = az_zeros_infer(_A)


def _engine(seed: int = 0, **kwargs: Any) -> rf.Engine:
    return rf.Engine(
        rf.games.Chess(max_ticks=60),  # short horizon: zeros-net games otherwise shuffle for a while
        rf.Reward(win=1.0, loss=-1.0),
        rf.policies.AlphaZero(num_simulations=8, temperature_drop=8),
        rf.learners.AlphaZero(),
        n_games=2,
        seed=seed,
        **kwargs,
    )


def test_spaces() -> None:
    game = rf.games.Chess()
    assert tuple(game.observation_space().shape) == (19, 8, 8)
    assert game.action_space().n == _A
    assert game.truncation_horizon() == 512
    assert rf.games.Chess(max_ticks=None).truncation_horizon() is None
    az = rf.encoders.AlphaZeroChess()
    assert tuple(rf.games.Chess(encoder=az).observation_space().shape) == (119, 8, 8)
    short = rf.encoders.AlphaZeroChess(history_length=2)
    assert tuple(rf.games.Chess(encoder=short).observation_space().shape) == (35, 8, 8)  # 14*2+7
    assert tuple(rf.games.Chess(encoder=rf.encoders.MinimalChess()).observation_space().shape) == (19, 8, 8)
    with pytest.raises(ValueError, match="history_length"):
        rf.encoders.AlphaZeroChess(history_length=0)


def test_az119_engine_collects() -> None:
    def infer(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        assert arr.shape[1] == 119 * 8 * 8  # the history-bearing view reaches the net
        return np.zeros((arr.shape[0], _A)), np.zeros(arr.shape[0])

    engine = rf.Engine(
        rf.games.Chess(max_ticks=40, encoder=rf.encoders.AlphaZeroChess()),
        rf.Reward(win=1.0, loss=-1.0),
        rf.policies.AlphaZero(num_simulations=8),
        rf.learners.AlphaZero(),
        n_games=1,
        seed=0,
    )
    obs, pi, _, _, _ = engine.collect(20, infer)
    assert obs.shape == (obs.shape[0], 119 * 8 * 8)
    np.testing.assert_allclose(pi.sum(axis=1), 1.0, atol=1e-12)


def test_az119_env_observation() -> None:
    env = rf.Env(rf.games.Chess(encoder=rf.encoders.AlphaZeroChess()))
    obs = env.observe(0)
    assert obs.shape == (119, 8, 8)
    assert obs[112].sum() == 64.0  # white to move
    # history steps beyond t=0 are empty at the start position
    assert obs[14:112].sum() == 0.0


def test_alphazero_collect_masks_illegal_actions() -> None:
    obs, pi, z, w, telemetry = _engine().collect(60, _az_infer)
    m = obs.shape[0]
    assert m >= 60
    assert obs.shape == (m, 19 * 8 * 8) and pi.shape == (m, _A)
    np.testing.assert_allclose(pi.sum(axis=1), 1.0, atol=1e-12)
    # Legal masking end to end: chess never has more than ~218 legal moves, so every π row must be
    # supported on a tiny fraction of the 4672 ids.
    assert ((pi > 0).sum(axis=1) <= 218).all()
    assert telemetry["decisions"] > 0
    # Truncated episodes bootstrap z from the (zero) value head; outcomes stay in [-1, 1].
    assert (np.abs(z) <= 1.0).all()
    # 2p sequential: negamax consumes only the mover's perspective, so every row is a real
    # decision (no value-only rows) and the policy weights are uniformly 1.
    assert (w == 1.0).all()


def test_mcts_treestrap_pairs() -> None:
    def infer(arr: np.ndarray) -> np.ndarray:
        return np.zeros((arr.shape[0], 1, _A))

    engine = rf.Engine(
        rf.games.Chess(max_ticks=40),
        None,
        rf.policies.Mcts(num_simulations=8),
        rf.learners.TreeStrap(),
        n_games=1,
        seed=0,
    )
    obs, tgt, mask, _ = engine.collect(20, infer)
    assert tgt.shape == (obs.shape[0], 1, _A)
    assert mask.shape == (obs.shape[0], 1)


def test_collect_is_deterministic_per_seed() -> None:
    b1 = _engine(7).collect(40, _az_infer)
    b2 = _engine(7).collect(40, _az_infer)
    assert isinstance(b1, rf._reinfors.AlphaZeroBatch) and isinstance(b2, rf._reinfors.AlphaZeroBatch)
    assert np.array_equal(b1.obs, b2.obs)
    assert np.array_equal(b1.policy_targets, b2.policy_targets)


def test_make_encoder_by_name() -> None:
    assert {"alphazero_chess", "minimal_chess", "openspiel_chess", "relative_chess"} <= set(rf.encoders.registered())
    enc = rf.encoders.make("alphazero_chess", history_length=4)
    assert tuple(rf.games.Chess(encoder=enc).observation_space().shape) == (63, 8, 8)  # 14*4+7


def test_env_plays_uci_like_moves() -> None:
    env = rf.Env(rf.games.Chess())
    assert env.num_agents() == 2 and env.action_count() == _A
    st = env.state()
    assert st["fen"].startswith("rnbqkbnr/pppppppp") and st["turn"] == 0 and not st["done"]
    legal = env.legal_actions(0)
    assert len(legal) == 20  # the classic starting-position move count
    assert env.legal_actions(1) == []  # non-mover has no actions (sequential game)
    trace = env.step({0: legal[0]})
    assert trace == [], "an opening move settles nothing"
    assert env.state()["turn"] == 1 and env.active_agents() == [1]


def test_env_rejects_illegal_action_at_the_boundary() -> None:
    # Illegal ids never enter the core: the Env validates against the legal set and raises
    # (the game's internal illegal-move handling remains only as an unreachable backstop).
    env = rf.Env(rf.games.Chess(), reward=rf.Reward(win=1.0, loss=-1.0))
    legal = set(env.legal_actions(0))
    bogus = next(a for a in range(_A) if a not in legal)
    with pytest.raises(ValueError, match="illegal"):
        env.step({0: bogus})
    assert not env.done()


def test_start_buffer_rejected() -> None:
    with pytest.raises(ValueError, match=r"compatibility\.md"):
        rf.Engine(
            rf.games.Chess(),
            None,
            rf.policies.AlphaZero(),
            rf.learners.AlphaZero(),
            n_games=1,
            start_buffer=True,
        )


def test_relative_encoder_shape_and_symmetric_start() -> None:
    game = rf.games.Chess(encoder=rf.encoders.RelativeChess())
    assert tuple(game.observation_space().shape) == (19, 8, 8)
    env = rf.Env(game, seed=0)
    env.reset()
    w, b = env.observe(0), env.observe(1)
    # The start position is its own sigma image, so the two perspectives differ only in the
    # my-turn plane (12): all-ones for White (to move), all-zeros for Black.
    assert (w[12] == 1.0).all() and (b[12] == 0.0).all()
    mask = np.ones((19, 8, 8), dtype=bool)
    mask[12] = False
    assert (w[mask] == b[mask]).all()


def test_uci_interop_validates_before_rendering() -> None:
    start = rf.Env(rf.games.Chess(), seed=0).state()["fen"]
    assert rf.chess_action_uci(rf.chess_uci_action("e2e4", start), start) == "e2e4"
    with pytest.raises(ValueError, match="not legal"):
        rf.chess_action_uci(0, start)  # decodes to a1a2: geometrically fine, illegal here
    for bad in (4672, 4673, 10**6):  # OpenSpiel's castling ids and beyond: never a panic
        with pytest.raises(ValueError):
            rf.chess_action_uci(bad, start)
        with pytest.raises(ValueError):
            rf._reinfors.chess_action_uci(bad, start)
