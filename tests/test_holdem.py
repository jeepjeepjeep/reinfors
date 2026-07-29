"""Fixed-limit hold'em through the Python surface: construction/validation, the information
gate (search families reject, the DQN family trains), Env play with chip conservation,
snapshots, config round-trips, and the PettingZoo AEC adapter."""

import numpy as np
import pytest
import reinfors as rf


def _game(n: int = 3, **kw) -> object:
    args = {"num_players": n, "stack": 200, "small_blind": 5, "big_blind": 10}
    args.update(kw)
    return rf.games.TexasHoldem(**args)


def test_construction_and_validation() -> None:
    assert rf.Env(_game(2)).num_agents() == 2
    assert rf.Env(_game(9)).num_agents() == 9
    with pytest.raises(ValueError, match="num_players"):
        _game(1)
    with pytest.raises(ValueError, match="num_players"):
        _game(10)
    with pytest.raises(ValueError, match="small_blind"):
        _game(3, small_blind=20)
    with pytest.raises(ValueError, match="stack"):
        _game(3, stack=5)


def test_search_families_reject_hidden_information() -> None:
    for policy, learner in [
        (rf.policies.AlphaZero(num_simulations=8), rf.learners.AlphaZero()),
        (rf.policies.Mcts(num_simulations=8), rf.learners.TreeStrap()),
        (rf.policies.SelectiveExpectimax(expansion_budget=8), rf.learners.TreeStrap()),
    ]:
        with pytest.raises(ValueError, match="clairvoyant"):
            rf.Engine(_game(3), rf.Reward(), policy, learner, n_games=1)


def test_dqn_family_trains_on_poker() -> None:
    engine = rf.Engine(
        _game(3),
        rf.Reward(scale=0.1),
        rf.policies.EpsilonGreedyQ(n_heads=2, epsilon=0.2),
        rf.learners.Dqn(),
        n_games=4,
        seed=2,
    )
    batch = engine.collect(80, lambda a: np.zeros((a.shape[0], 2, 3)))
    assert batch.obs.shape[0] >= 80
    assert batch.masks.shape[1] == 2
    assert batch.actions.max() <= 2
    for r, _length, _seeded in batch.telemetry["episodes"]:
        assert abs(sum(r)) < 1e-9, f"hands are zero-sum: {r}"


def test_env_plays_hands_and_conserves_chips() -> None:
    env = rf.Env(_game(4), rf.Reward(), seed=7)
    rng = np.random.default_rng(0)
    for _ in range(20):
        env.reset()
        guard = 0
        while not env.done():
            (agent,) = env.active_agents()
            legal = env.legal_actions(agent)
            assert legal, "live hands always offer an action"
            env.step({agent: int(rng.choice(legal))})
            guard += 1
            assert guard < 300
        st = env.state()
        assert st["done"] and st["street"] == "done"
        rewards = env.rewards
        assert rewards is not None
        assert abs(sum(rewards)) < 1e-9, "terminal deltas are zero-sum"
        # Board cards are unique and hidden holes exist for every seat.
        cards = [c for h in st["hole"] for c in h] + list(st["board"])
        assert len(set(cards)) == len(cards)


def test_observation_hides_other_holes() -> None:
    env = rf.Env(_game(3), seed=1)
    env.reset()
    st = env.state()
    obs = env.observe(0)
    assert obs.shape == (11 + 2 * 2, 4, 13)
    hole_plane = obs[0]
    assert hole_plane.sum() == 2.0
    for c in st["hole"][0]:
        assert hole_plane[c % 4, c // 4] == 1.0


def test_env_snapshot_round_trips() -> None:
    env = rf.Env(_game(3), seed=9)
    env.reset()
    rng = np.random.default_rng(3)
    for _ in range(3):
        if env.done():
            break
        (agent,) = env.active_agents()
        env.step({agent: int(rng.choice(env.legal_actions(agent)))})
    snap = env.snapshot()

    def play(e: rf.Env, seed: int) -> list:
        r = np.random.default_rng(seed)
        out = []
        for _ in range(40):
            if e.done():
                break
            (agent,) = e.active_agents()
            e.step({agent: int(r.choice(e.legal_actions(agent)))})
            out.append(e.state()["stacks"])
        return out

    ahead = play(env, 4)
    env.restore(snap)
    assert play(env, 4) == ahead


def test_resolved_config_round_trips() -> None:
    engine = rf.Engine(
        _game(3),
        rf.Reward(scale=0.5),
        rf.policies.EpsilonGreedyQ(n_heads=1),
        rf.learners.Dqn(),
        n_games=1,
        seed=0,
    )
    cfg = engine.resolved_config()
    assert cfg["game"]["name"] == "texas_holdem"
    assert cfg["game"]["num_players"] == 3
    assert cfg["reward"]["scale"] == 0.5
    rebuilt = rf.engine_from_config(cfg)
    assert rebuilt.resolved_config() == cfg
    assert rebuilt.config_fingerprint() == engine.config_fingerprint()


def test_reward_rejects_foreign_keys() -> None:
    with pytest.raises(ValueError):
        rf.Engine(
            _game(3),
            rf.Reward(win=1.0),  # not a poker key: the deltas ARE the reward
            rf.policies.EpsilonGreedyQ(),
            rf.learners.Dqn(),
            n_games=1,
        )


def test_pettingzoo_aec_conformance() -> None:
    pytest.importorskip("pettingzoo")
    from pettingzoo.test import api_test

    env = rf.gym.aec_env(_game(3), rf.Reward())
    assert sorted(env.possible_agents) == ["player_0", "player_1", "player_2"]
    api_test(env, num_cycles=200)
