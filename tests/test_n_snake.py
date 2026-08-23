"""N-player snake through the Python surface: construction/validation, Env play, engine collects
under every search family, snapshots, config round-trips, and the PettingZoo adapter at N=3."""

from typing import Any

import numpy as np
import pytest
import reinfors as rf


def _snake(n: int = 3, **kw: Any) -> Any:
    args: dict[str, Any] = {"grid_size": 8, "initial_length": 2, "food": 2, "max_ticks": 60, "num_snakes": n}
    args.update(kw)
    return rf.games.Snake(**args)


def test_construction_and_validation() -> None:
    assert rf.Env(_snake(3)).num_agents() == 3
    assert rf.Env(_snake(8, grid_size=14)).num_agents() == 8
    with pytest.raises(ValueError, match="num_snakes"):
        _snake(1)
    with pytest.raises(ValueError, match="num_snakes"):
        _snake(9, grid_size=20)
    with pytest.raises(ValueError, match="two-snake"):
        _snake(3, win_food_lead=2, play_to_last=False)
    # In-game multi-eats resolve as ONE combined ordered-tuple index, bounded at 2^53;
    # births chain one apple per draw, bounded by the grid and the chance-chain limit.
    with pytest.raises(ValueError, match="respawn index space"):
        _snake(3, grid_size=1000, food=3)  # ~1e18 ordered triples: past the 2^53 index guard
    with pytest.raises(ValueError, match="exceeds"):
        _snake(2, grid_size=4, food=17)  # more food than cells
    with pytest.raises(ValueError, match="chance-chain limit"):
        _snake(2, grid_size=101, food=10_001)  # a birth chain past the framework backstop


def test_default_three_snake_config_constructs_and_plays() -> None:
    # The motivating case for the compact chance declaration: default grid 20 / food 3 with
    # three snakes has a ~6.4e7 worst-case respawn index space — O(1) to declare, and the env
    # realizes triple-eats by drawing one index.
    env = rf.Env(rf.games.Snake(num_snakes=3), seed=1)
    env.reset()
    assert env.num_agents() == 3
    rng = np.random.default_rng(2)
    for _ in range(60):
        if env.done():
            break
        env.step({a: int(rng.choice(env.legal_actions(a))) for a in env.active_agents()})


def test_env_plays_three_snakes_to_the_end() -> None:
    env = rf.Env(_snake(3), seed=5)
    env.reset()
    assert env.num_agents() == 3
    assert sorted(env.active_agents()) == [0, 1, 2]
    obs = env.observe(2)
    assert obs.shape == (5, 8, 8)  # the merged-opponent encoding is N-independent
    rng = np.random.default_rng(0)
    for _ in range(200):
        if env.done():
            break
        env.step({a: int(rng.choice(env.legal_actions(a))) for a in env.active_agents()})
    # rf.Env never truncates (that is an Engine concern): the seeded random walk must end the
    # game by deaths — with these seeds all three snakes are gone within a handful of ticks.
    assert env.done()


def test_env_snapshot_round_trips_three_snakes() -> None:
    env = rf.Env(_snake(3), seed=11)
    env.reset()
    rng = np.random.default_rng(1)
    for _ in range(6):
        if env.done():
            break
        env.step({a: int(rng.choice(env.legal_actions(a))) for a in env.active_agents()})
    snap = env.snapshot()

    def play(e: rf.Env, seed: int) -> list[bytes]:
        r = np.random.default_rng(seed)
        out = []
        for _ in range(8):
            if e.done():
                break
            e.step({a: int(r.choice(e.legal_actions(a))) for a in e.active_agents()})
            out.extend(e.observe(a).tobytes() for a in range(3))
        return out

    ahead = play(env, 2)
    env.restore(snap)
    assert play(env, 2) == ahead
    other = rf.Env(_snake(2), seed=0)
    other.reset()
    with pytest.raises(ValueError, match="different composition"):
        other.restore(snap)  # num_snakes is part of the config fingerprint


def _mcts_engine(n: int) -> rf.Engine:
    return rf.Engine(
        _snake(n),
        rf.Reward(food=1.0, loss=-1.0),
        rf.policies.Mcts(num_simulations=12),
        rf.learners.TreeStrap(),
        n_games=2,
        seed=3,
    )


def test_mcts_treestrap_collects_three_snakes() -> None:
    obs, tgt, mask, _ = _mcts_engine(3).collect(20, lambda a: np.zeros((a.shape[0], 1, 3)))
    assert obs.shape[0] >= 20 and obs.shape[1] == 5 * 8 * 8
    assert tgt.shape[1:] == (1, 3)
    assert mask.shape[1] == 1


def test_alphazero_collects_three_snakes() -> None:
    def infer(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros((arr.shape[0], 3)), np.zeros(arr.shape[0])

    engine = rf.Engine(
        _snake(3),
        rf.Reward(food=1.0, loss=-1.0),
        rf.policies.AlphaZero(num_simulations=12, chance=rf.chance_modes.Committed(samples=2)),
        rf.learners.AlphaZero(),
        n_games=2,
        seed=4,
    )
    obs, pi, _z, w, _ = engine.collect(30, infer)
    assert obs.shape[0] >= 30
    # Simultaneous game: every active agent acts, so every row is policy-weighted.
    assert (w == 1.0).all()
    rows = pi.sum(axis=1)
    np.testing.assert_allclose(rows[rows > 0], 1.0, atol=1e-12)


def test_expectimax_treestrap_collects_three_snakes() -> None:
    engine = rf.Engine(
        _snake(3),
        rf.Reward(food=1.0, loss=-1.0),
        rf.policies.SelectiveExpectimax(
            expansion_budget=12, top_k=4, max_depth=4, n_heads=2, opponent="distributional"
        ),
        rf.learners.TreeStrap(),
        n_games=2,
        seed=5,
    )
    obs, tgt, _, telemetry = engine.collect(20, lambda a: np.zeros((a.shape[0], 2, 3)))
    assert obs.shape[0] >= 20
    assert tgt.shape[1:] == (2, 3)
    assert telemetry["decisions"] > 0


def test_start_buffer_runs_with_three_snakes() -> None:
    engine = rf.Engine(
        _snake(3),
        rf.Reward(food=1.0, loss=0.0, win=0.0),
        rf.policies.Mcts(num_simulations=8),
        rf.learners.TreeStrap(),
        n_games=2,
        seed=6,
        start_buffer=True,
        n_threads=1,
    )
    obs, _, _, telemetry = engine.collect(40, lambda a: np.zeros((a.shape[0], 1, 3)))
    assert obs.shape[0] >= 40
    # The episode tuples are (reward_vec, length, seeded): with the buffer on, some episodes must
    # actually have STARTED from a buffered state (deterministic with these seeds).
    assert any(ep[2] for ep in telemetry["episodes"])


def test_resolved_config_round_trips_num_snakes() -> None:
    engine = _mcts_engine(3)
    cfg = engine.resolved_config()
    assert cfg["game"]["num_snakes"] == 3
    rebuilt = rf.engine_from_config(cfg)
    assert rebuilt.resolved_config() == cfg
    assert rebuilt.config_fingerprint() == engine.config_fingerprint()


def test_pettingzoo_parallel_env_at_three_snakes() -> None:
    pytest.importorskip("pettingzoo")
    from pettingzoo.test import parallel_api_test

    env = rf.gym.parallel_env(_snake(3), rf.Reward(food=1.0, loss=-1.0))
    assert sorted(env.possible_agents) == ["player_0", "player_1", "player_2"]
    parallel_api_test(env, num_cycles=150)
