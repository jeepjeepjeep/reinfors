"""Per-player infer on the engine: the polymorphic `infer` argument (bare callable = shared
network, sequence = one per player), the `players` column on DQN batches (the primary routing
mechanism for heterogeneous training), `learn_players` (the frozen-opponent source filter),
per-player `weights_updated`, and the family gate (search families reject the sequence form
until their pooled paths land).
"""

import numpy as np
import pytest
import reinfors as rf


def q_net(prefer: int, n_actions: int):
    def f(obs: np.ndarray) -> np.ndarray:
        out = np.zeros((obs.shape[0], 1, n_actions))
        out[:, :, prefer] = 1.0
        return out

    return f


def kuhn_engine(**kwargs: object) -> object:
    return rf.Engine(
        rf.games.KuhnPoker(),
        rf.Reward(),
        rf.policies.EpsilonGreedyQ(n_heads=1, epsilon=0.0),
        rf.learners.Dqn(),
        n_games=4,
        seed=5,
        **kwargs,  # type: ignore[arg-type]
    )


def test_per_player_list_routes_each_players_network() -> None:
    engine = kuhn_engine()
    batch = engine.collect(60, [q_net(0, 2), q_net(1, 2)])
    players = batch.players
    actions = batch.actions
    assert set(players.tolist()) == {0, 1}, "records carry both players"
    assert (actions == players).all(), "greedy actions read each player's own network"


def test_a_shared_callable_equals_an_identical_list() -> None:
    shared = q_net(1, 2)
    a = kuhn_engine().collect(40, shared)
    b = kuhn_engine().collect(40, [shared, shared])
    assert np.array_equal(a.obs, b.obs)
    assert np.array_equal(a.actions, b.actions)
    assert np.array_equal(a.players, b.players)


def test_infer_argument_validation() -> None:
    engine = kuhn_engine()
    with pytest.raises(ValueError, match="expected 2 per-player"):
        engine.collect(8, [q_net(0, 2)])
    with pytest.raises(TypeError, match="callable or a sequence"):
        engine.collect(8, 42)


def test_search_families_reject_the_sequence_form() -> None:
    engine = rf.Engine(
        rf.games.Connect4(),
        rf.Reward(),
        rf.policies.AlphaZero(num_simulations=4),
        rf.learners.AlphaZero(),
        n_games=1,
        seed=0,
    )
    with pytest.raises(ValueError, match="follow-up"):
        engine.collect(4, [q_net(0, 7), q_net(1, 7)])


def test_learn_players_filters_records_at_source() -> None:
    engine = kuhn_engine(learn_players=[1])
    batch = engine.collect(30, [q_net(0, 2), q_net(1, 2)])
    assert (batch.players == 1).all(), "the frozen player leaves no records"
    assert engine.resolved_config()["engine"]["learn_players"] == [1]
    with pytest.raises(ValueError, match="out of range"):
        kuhn_engine(learn_players=[2])
    with pytest.raises(ValueError, match="at least one player"):
        kuhn_engine(learn_players=[])


def test_weights_updated_accepts_a_player() -> None:
    engine = kuhn_engine()
    engine.weights_updated()
    engine.weights_updated(0)
    engine.weights_updated(player=1)
    with pytest.raises(ValueError, match="out of range"):
        engine.weights_updated(2)


def test_engine_infer_shapes_are_validated_exactly() -> None:
    # A transposed (n, A, K) return has the correct element count when K != A — the old
    # count-only check let it through as garbage evaluations.
    engine = rf.Engine(
        rf.games.TexasHoldem(num_players=2),
        rf.Reward(),
        rf.policies.EpsilonGreedyQ(n_heads=2, epsilon=0.1),
        rf.learners.Dqn(),
        n_games=2,
        seed=1,
    )
    with pytest.raises(ValueError, match="returned shape"):
        engine.collect(8, lambda obs: np.zeros((obs.shape[0], 3, 2)))


def test_collect_stream_accepts_the_sequence_form() -> None:
    engine = kuhn_engine()
    stream = engine.collect_stream(20, [q_net(0, 2), q_net(1, 2)], depth=1)
    batch = next(stream)
    assert (batch.actions == batch.players).all()
    stream.stop()


def test_exploiter_calibration_on_leduc() -> None:
    """The instrument this feature exists for, checked against ground truth: a learning
    player vs a FROZEN UNIFORM player on Leduc. The exploiter's mean winnings per hand are a
    lower bound on the frozen policy's exploitability and must approach the EXACT best
    response value (pinned by the Rust solver: BR0 = 2.0875, BR1 = 2.6597 chips/hand vs
    uniform). Numpy Q-table learner — no torch needed at Leduc scale.

    The frozen player is expressed through its callback: RANDOM one-hot Q rows make greedy
    action selection uniform (the callback owns the mixing — the idiom for frozen mixed
    policies generally).
    """
    frozen_rng = np.random.default_rng(7)

    def frozen_uniform(obs: np.ndarray) -> np.ndarray:
        out = np.zeros((obs.shape[0], 1, 3))
        out[np.arange(obs.shape[0]), 0, frozen_rng.integers(3, size=obs.shape[0])] = 1.0
        return out

    table: dict[bytes, np.ndarray] = {}

    def learner_net(obs: np.ndarray) -> np.ndarray:
        out = np.zeros((obs.shape[0], 1, 3))
        for i, row in enumerate(obs):
            hit = table.get(row.tobytes())
            if hit is not None:
                out[i, 0] = hit
        return out

    exploiter_player = 1  # BR1 = 2.6597: player 1 exploiting a uniform player 0
    engine = rf.Engine(
        rf.games.LeducPoker(),
        rf.Reward(),
        rf.policies.EpsilonGreedyQ(n_heads=1, epsilon=0.15),
        rf.learners.Dqn(),
        n_games=8,
        seed=11,
        learn_players=[exploiter_player],
    )
    infer = [frozen_uniform, learner_net]
    gamma = 1.0
    for _ in range(60):
        batch = engine.collect(600, infer)
        offsets = batch.next_legal_offsets
        for i in range(batch.obs.shape[0]):
            ids = batch.next_legal_ids[offsets[i] : offsets[i + 1]]
            if len(ids) and not batch.dones[i]:
                nxt = table.get(batch.next_obs[i].tobytes())
                bootstrap = max(nxt[ids]) if nxt is not None else 0.0
            else:
                bootstrap = 0.0
            target = batch.rewards[i] + gamma * bootstrap
            q = table.setdefault(batch.obs[i].tobytes(), np.zeros(3))
            a = batch.actions[i]
            q[a] += 0.2 * (target - q[a])

    # Evaluate greedily: mean chips/hand for the exploiter over fresh episodes.
    total, hands = 0.0, 0
    eval_rng = np.random.default_rng(3)
    for seed in range(400):
        env = rf.Env(rf.games.LeducPoker(), rf.Reward(), seed=seed)
        env.reset()
        while not env.done():
            (agent,) = env.active_agents()
            legal = env.legal_actions(agent)
            if agent == exploiter_player:
                q = table.get(env.observe(agent).reshape(-1).tobytes())
                action = int(max(legal, key=lambda x: q[x])) if q is not None else legal[0]
            else:
                action = int(eval_rng.choice(legal))
            env.step({agent: action})
        rewards = env.rewards
        assert rewards is not None
        total += rewards[exploiter_player]
        hands += 1
    winnings = total / hands
    exact_br = 2.6597222222222223
    assert winnings > 0.55 * exact_br, (
        f"exploiter reaches a meaningful fraction of the exact best response: {winnings:.3f} vs BR {exact_br:.3f}"
    )
    assert winnings < exact_br + 0.35, "and cannot exceed it (sampling tolerance)"
