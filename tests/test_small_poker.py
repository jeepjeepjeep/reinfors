"""Kuhn and Leduc poker — the small imperfect-information testbeds (CFR's validation games).
Covers the game surface: construction/registry, hidden-information gating, DQN collection,
env play with zero-sum settlement, information-state keys (the solver seam), and snapshots.
"""

import numpy as np
import pytest
import reinfors as rf

GAMES = [
    ("kuhn_poker", rf.games.KuhnPoker, 2, (6, 1, 1)),
    ("leduc_poker", rf.games.LeducPoker, 3, (21, 1, 1)),
]


@pytest.mark.parametrize(("name", "ctor", "n_actions", "obs_shape"), GAMES)
def test_construction_registry_and_spaces(
    name: str, ctor: object, n_actions: int, obs_shape: tuple[int, int, int]
) -> None:
    g = ctor()  # type: ignore[operator]
    assert rf.games.make(name) is not None
    env = rf.Env(g, rf.Reward(), seed=0)
    assert env.observation_space().shape == obs_shape
    cfg = rf.Engine(
        g,
        rf.Reward(),
        rf.policies.EpsilonGreedyQ(n_heads=1, epsilon=0.1),
        rf.learners.Dqn(),
        n_games=1,
        seed=0,
    ).resolved_config()
    assert cfg["game"]["name"] == name
    assert cfg["reward"]["scale"] == 1.0


@pytest.mark.parametrize(("name", "ctor", "n_actions", "obs_shape"), GAMES)
def test_search_families_reject_hidden_information(
    name: str, ctor: object, n_actions: int, obs_shape: tuple[int, int, int]
) -> None:
    pairs = [
        (rf.policies.SelectiveExpectimax(), rf.learners.TreeStrap()),
        (rf.policies.Mcts(num_simulations=8), rf.learners.TreeStrap()),
        (rf.policies.AlphaZero(num_simulations=8), rf.learners.AlphaZero()),
    ]
    for policy, learner in pairs:
        with pytest.raises(ValueError, match="clairvoyant"):
            rf.Engine(ctor(), rf.Reward(), policy, learner, n_games=1, seed=0)  # type: ignore[operator]


@pytest.mark.parametrize(("name", "ctor", "n_actions", "obs_shape"), GAMES)
def test_dqn_collects_zero_sum_hands(name: str, ctor: object, n_actions: int, obs_shape: tuple[int, int, int]) -> None:
    engine = rf.Engine(
        ctor(),  # type: ignore[operator]
        rf.Reward(),
        rf.policies.EpsilonGreedyQ(n_heads=2, epsilon=0.3),
        rf.learners.Dqn(),
        n_games=4,
        seed=1,
    )
    batch = engine.collect(200, lambda a: np.zeros((a.shape[0], 2, n_actions)))
    assert batch.obs.shape[0] >= 200
    assert batch.obs.shape[1:] == (int(np.prod(obs_shape)),)
    episodes = batch.telemetry["episodes"]
    assert episodes, "hands finish inside one collect"
    for rewards, _length, _seeded in episodes:
        assert abs(sum(rewards)) < 1e-12, "episode rewards are zero-sum"


@pytest.mark.parametrize(("name", "ctor", "n_actions", "obs_shape"), GAMES)
def test_env_hands_settle_zero_sum(name: str, ctor: object, n_actions: int, obs_shape: tuple[int, int, int]) -> None:
    rng = np.random.default_rng(2)
    for seed in range(30):
        env = rf.Env(ctor(), rf.Reward(), seed=seed)  # type: ignore[operator]
        env.reset()
        assert len(env.state()["cards"]) == 2, "the deal realizes at reset"
        while not env.done():
            (agent,) = env.active_agents()
            legal = env.legal_actions(agent)
            assert legal, "live states always offer an action"
            env.step({agent: int(rng.choice(legal))})
        rewards = env.rewards
        assert rewards is not None
        assert abs(sum(rewards)) < 1e-12


@pytest.mark.parametrize(("name", "ctor", "n_actions", "obs_shape"), GAMES)
def test_information_keys_partition_by_agent_knowledge(
    name: str, ctor: object, n_actions: int, obs_shape: tuple[int, int, int]
) -> None:
    env = rf.Env(ctor(), rf.Reward(), seed=5)  # type: ignore[operator]
    env.reset()
    k0, k1 = env.information_state_key(0), env.information_state_key(1)
    assert isinstance(k0, bytes) and k0 != k1, "perspectives differ"
    (agent,) = env.active_agents()
    before = env.information_state_key(agent)
    env.step({agent: int(env.legal_actions(agent)[0])})
    if not env.done():
        assert env.information_state_key(agent) != before, "history extends the key"
    with pytest.raises(ValueError, match="out of range"):
        env.information_state_key(9)


def test_games_without_information_states_reject_the_key_query() -> None:
    env = rf.Env(rf.games.Connect4(), rf.Reward(), seed=0)
    env.reset()
    with pytest.raises(ValueError, match="does not declare information states"):
        env.information_state_key(0)


@pytest.mark.parametrize(("name", "ctor", "n_actions", "obs_shape"), GAMES)
def test_env_snapshots_round_trip(name: str, ctor: object, n_actions: int, obs_shape: tuple[int, int, int]) -> None:
    env = rf.Env(ctor(), rf.Reward(), seed=11)  # type: ignore[operator]
    env.reset()
    (agent,) = env.active_agents()
    env.step({agent: int(env.legal_actions(agent)[0])})
    snap = env.snapshot()
    restored = rf.Env(ctor(), rf.Reward(), seed=99)  # type: ignore[operator]
    restored.restore(snap)
    assert restored.state() == env.state()
    assert restored.information_state_key(0) == env.information_state_key(0)
