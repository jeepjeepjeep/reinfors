"""PolicyHandle.choose: stateless batched decisions over Envs.

Reproducible for the same ordered batch, seed and deterministic inference; external
schedulers that reorder batches get statistically seeded, non-replayable results.
"""

import numpy as np
import pytest
import reinfors as rf

_A = rf.games.Connect4().action_space().n
_R = rf.Reward(win=1.0, loss=-1.0)


def _az(temperature=0.0, noise=0.0, sims=8, temperature_drop=8):
    handle = rf.noise.Dirichlet(epsilon=noise, alpha=0.5) if noise > 0 else None
    return rf.policies.AlphaZero(
        num_simulations=sims,
        temperature=temperature,
        temperature_drop=temperature_drop,
        noise=handle,
    )


def _az_infer(obs, n=None):
    m = obs.shape[0]
    return np.zeros((m, _A), dtype=np.float32), np.zeros(m, dtype=np.float32)


def _envs(n):
    # distinct openings so requests do not dedupe to one observation
    envs = []
    for i in range(n):
        env = rf.Env(rf.games.Connect4(), _R, seed=i)
        env.step({0: i % _A})
        envs.append(env)
    return envs


def test_choose_returns_legal_actions() -> None:
    envs = _envs(5)
    actions = _az().choose(envs, _az_infer, gamma=1.0)
    assert len(actions) == 5
    for env, action in zip(envs, actions, strict=True):
        agent = env.active_agents()[0]
        assert action in env.legal_actions(agent)


def test_choose_is_deterministic_per_batch_and_seed() -> None:
    envs = _envs(4)
    policy = _az(temperature=1.0, noise=0.25)
    a = policy.choose(envs, _az_infer, seed=7, gamma=1.0)
    b = policy.choose(envs, _az_infer, seed=7, gamma=1.0)
    assert a == b


def test_seed_varies_sampling() -> None:
    envs = _envs(1)
    policy = _az(temperature=1.0, sims=16)
    picks = {policy.choose(envs, _az_infer, seed=s, gamma=1.0)[0] for s in range(20)}
    assert len(picks) > 1


def test_choose_does_not_mutate_envs() -> None:
    envs = _envs(3)
    before = [e.snapshot().to_bytes() for e in envs]
    _az(temperature=1.0, noise=0.25).choose(envs, _az_infer, seed=3, gamma=1.0)
    after = [e.snapshot().to_bytes() for e in envs]
    assert before == after
    assert [e.ticks for e in envs] == [1, 1, 1]


def test_choose_batches_across_envs() -> None:
    rows_per_call = []

    def counting_infer(obs, n=None):
        rows_per_call.append(obs.shape[0])
        return _az_infer(obs)

    _az().choose(_envs(6), counting_infer, gamma=1.0)
    assert max(rows_per_call) >= 6, f"no pooled batch observed: {rows_per_call}"


def test_temperature_drop_boundary() -> None:
    policy = _az(temperature=1.0, sims=16, temperature_drop=4)
    env = _envs(1)
    # one ply past the drop: greedy, identical across seeds
    at_drop = {policy.choose(env, _az_infer, seed=s, plies=[4], gamma=1.0)[0] for s in range(20)}
    assert len(at_drop) == 1
    # final ply before the drop: visit sampling still live
    before_drop = {policy.choose(env, _az_infer, seed=s, plies=[3], gamma=1.0)[0] for s in range(20)}
    assert len(before_drop) > 1


def test_default_plies_derive_from_env_ticks() -> None:
    # an env stepped past the drop must behave greedily WITHOUT explicit plies
    policy = _az(temperature=1.0, sims=16, temperature_drop=2)
    env = rf.Env(rf.games.Connect4(), _R, seed=9)
    for column in (0, 1, 2):
        agent = env.active_agents()[0]
        env.step({agent: column})
    assert env.ticks == 3
    derived = {policy.choose([env], _az_infer, seed=s, gamma=1.0)[0] for s in range(20)}
    explicit = {policy.choose([env], _az_infer, seed=s, plies=[3], gamma=1.0)[0] for s in range(20)}
    assert derived == explicit
    assert len(derived) == 1, "past the drop the derived ply must select greedily"


def test_restored_snapshot_keeps_tick_count() -> None:
    env = rf.Env(rf.games.Connect4(), _R, seed=2)
    for column in (0, 1, 2, 3):
        env.step({env.active_agents()[0]: column})
    snap = env.snapshot()
    restored = rf.Env(rf.games.Connect4(), _R, seed=99)
    restored.restore(snap)
    assert restored.ticks == 4
    policy = _az(temperature=1.0, sims=16, temperature_drop=4)
    a = [policy.choose([env], _az_infer, seed=s, gamma=1.0)[0] for s in range(8)]
    b = [policy.choose([restored], _az_infer, seed=s, gamma=1.0)[0] for s in range(8)]
    assert a == b


def test_old_env_snapshot_schema_rejected() -> None:
    snap = rf.Env(rf.games.Connect4(), _R).snapshot()
    raw = bytearray(snap.to_bytes())
    raw[4] = 1  # schema byte back to v1
    with pytest.raises(ValueError, match="schema"):
        rf.EnvSnapshot.from_bytes(bytes(raw))


def test_fork_carries_ticks() -> None:
    env = rf.Env(rf.games.Connect4(), _R, seed=5)
    env.step({0: 2})
    env.step({1: 3})
    assert env.fork().ticks == 2


def test_gamma_is_required() -> None:
    with pytest.raises(TypeError):
        _az().choose(_envs(1), _az_infer)


def test_expectimax_and_qgreedy_choose() -> None:
    game = rf.games.GridWorld()
    a = game.action_space().n

    def q_infer(obs, n=None):
        return np.zeros((obs.shape[0], 1, a), dtype=np.float32)

    envs = [rf.Env(rf.games.GridWorld(), rf.Reward(goal=1.0), seed=i) for i in range(3)]
    ex = rf.policies.SelectiveExpectimax(expansion_budget=16)
    assert len(ex.choose(envs, q_infer, gamma=0.99)) == 3

    def q4_infer(obs, n=None):
        return np.zeros((obs.shape[0], 1, _A), dtype=np.float32)

    c4 = _envs(2)
    actions = rf.policies.EpsilonGreedyQ().choose(c4, q4_infer, gamma=1.0)
    for env, action in zip(c4, actions, strict=True):
        assert action in env.legal_actions(env.active_agents()[0])


def test_thompson_heads_vary_across_seeds() -> None:
    # stateless contract: heads draw per call from the seed
    game = rf.games.GridWorld()
    a = game.action_space().n

    def heads_infer(obs, n=None):
        m = obs.shape[0]
        rows = np.zeros((m, 2, a), dtype=np.float32)
        rows[:, 0, 0] = 1.0
        rows[:, 1, 1] = 1.0
        return rows

    env = [rf.Env(rf.games.GridWorld(), rf.Reward(goal=1.0), seed=3)]
    q = rf.policies.EpsilonGreedyQ(n_heads=2, epsilon=0.0)
    picks = {q.choose(env, heads_infer, seed=s, gamma=1.0)[0] for s in range(12)}
    assert len(picks) > 1


def test_rejects_bad_inputs() -> None:
    envs = _envs(2)
    with pytest.raises(ValueError, match="at least one env"):
        _az().choose([], _az_infer, gamma=1.0)
    with pytest.raises(ValueError, match="plies for"):
        _az().choose(envs, _az_infer, plies=[0], gamma=1.0)
    with pytest.raises(ValueError, match="one composition"):
        mixed = [envs[0], rf.Env(rf.games.Connect4(), rf.Reward(win=0.5, loss=-0.5))]
        _az().choose(mixed, _az_infer, gamma=1.0)
    with pytest.raises(ValueError, match="with a reward"):
        _az().choose([rf.Env(rf.games.Connect4())], _az_infer, gamma=1.0)
    with pytest.raises(ValueError, match="finished env"):
        done = rf.Env(rf.games.Connect4(), _R)
        while not done.done():
            agent = done.active_agents()[0]
            done.step({agent: done.legal_actions(agent)[0]})
        _az().choose([done], _az_infer, gamma=1.0)
    with pytest.raises(ValueError, match="one active agent"):
        snake = rf.Env(rf.games.Snake(), rf.Reward(food=1.0))
        _az().choose([snake], _az_infer, gamma=1.0)


def test_rejects_hidden_information_games() -> None:
    env = rf.Env(rf.games.KuhnPoker(), rf.Reward(scale=1.0), seed=0)
    with pytest.raises(ValueError, match=r"hidden-information|clairvoyant|imperfect"):
        _az().choose([env], _az_infer, gamma=1.0)


def test_callback_junk_raises_not_panics() -> None:
    envs = _envs(2)

    def wrong_shape(obs, n=None):
        return np.zeros((1, 2), dtype=np.float32), np.zeros(1, dtype=np.float32)

    with pytest.raises(ValueError):
        _az().choose(envs, wrong_shape, gamma=1.0)

    def raises(obs, n=None):
        raise RuntimeError("boom from infer")

    with pytest.raises(RuntimeError, match="boom from infer"):
        _az().choose(envs, raises, gamma=1.0)
