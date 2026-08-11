"""PolicyHandle.choose: batched on-demand decisions over Envs."""

import numpy as np
import pytest
import reinfors as rf

_A = rf.games.Connect4().action_space().n
_R = rf.Reward(win=1.0, loss=-1.0)


def _az(temperature=0.0, noise=0.0, sims=8):
    handle = rf.noise.Dirichlet(epsilon=noise, alpha=0.5) if noise > 0 else None
    return rf.policies.AlphaZero(num_simulations=sims, temperature=temperature, noise=handle)


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


def _seeds(n, base=100):
    return [base + i for i in range(n)]


def test_choose_returns_legal_actions() -> None:
    envs = _envs(5)
    actions = _az().choose(envs, _az_infer, _seeds(5))
    assert len(actions) == 5
    for env, action in zip(envs, actions, strict=True):
        agent = env.active_agents()[0]
        assert action in env.legal_actions(agent)


def test_choose_is_deterministic() -> None:
    envs = _envs(4)
    policy = _az(temperature=1.0, noise=0.25)
    a = policy.choose(envs, _az_infer, _seeds(4), plies=[0, 1, 2, 3])
    b = policy.choose(envs, _az_infer, _seeds(4), plies=[0, 1, 2, 3])
    assert a == b


def test_choose_does_not_mutate_envs() -> None:
    envs = _envs(3)
    before = [e.snapshot().to_bytes() for e in envs]
    _az(temperature=1.0, noise=0.25).choose(envs, _az_infer, _seeds(3))
    after = [e.snapshot().to_bytes() for e in envs]
    assert before == after


def test_choose_is_independent_of_batch_composition() -> None:
    # randomness live (temperature + noise): index-derived streams would break this
    policy = _az(temperature=1.0, noise=0.25)
    envs = _envs(6)
    seeds = _seeds(6)
    plies = [0, 1, 0, 2, 1, 0]
    full = policy.choose(envs, _az_infer, seeds, plies=plies)

    singles = [policy.choose([envs[i]], _az_infer, [seeds[i]], plies=[plies[i]])[0] for i in range(6)]
    assert full == singles

    perm = [3, 0, 5, 1, 4, 2]
    permuted = policy.choose(
        [envs[i] for i in perm],
        _az_infer,
        [seeds[i] for i in perm],
        plies=[plies[i] for i in perm],
    )
    assert permuted == [full[i] for i in perm]


def test_choose_batches_across_envs() -> None:
    rows_per_call = []

    def counting_infer(obs, n=None):
        rows_per_call.append(obs.shape[0])
        return _az_infer(obs)

    envs = _envs(6)
    _az().choose(envs, counting_infer, _seeds(6))
    assert max(rows_per_call) >= 6, f"no pooled batch observed: {rows_per_call}"


def test_temperature_drop_progression() -> None:
    policy = _az(temperature=1.0, sims=16)
    env = _envs(1)
    # past the drop (default drop=10 -> use ply beyond it): argmax, seed-independent
    past = {policy.choose(env, _az_infer, [s], plies=[100])[0] for s in range(20)}
    assert len(past) == 1
    # before the drop: visit-proportional sampling varies across seeds
    early = {policy.choose(env, _az_infer, [s], plies=[0])[0] for s in range(20)}
    assert len(early) > 1


def test_expectimax_and_qgreedy_choose() -> None:
    game = rf.games.GridWorld()
    a = game.action_space().n

    def q_infer(obs, n=None):
        return np.zeros((obs.shape[0], 1, a), dtype=np.float32)

    envs = [rf.Env(rf.games.GridWorld(), rf.Reward(goal=1.0), seed=i) for i in range(3)]
    ex = rf.policies.SelectiveExpectimax(expansion_budget=16)
    actions = ex.choose(envs, q_infer, _seeds(3), gamma=0.99)
    assert len(actions) == 3

    def q4_infer(obs, n=None):
        return np.zeros((obs.shape[0], 1, _A), dtype=np.float32)

    c4 = _envs(2)
    actions = rf.policies.EpsilonGreedyQ().choose(c4, q4_infer, _seeds(2))
    for env, action in zip(c4, actions, strict=True):
        assert action in env.legal_actions(env.active_agents()[0])


def test_thompson_head_is_stable_per_seed_across_plies() -> None:
    # multi-head Thompson state: the episode seed fixes the head; the ply must not redraw it
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
    per_seed = []
    for s in range(12):
        picks = {q.choose(env, heads_infer, [s], plies=[p])[0] for p in range(4)}
        per_seed.append(picks)
    assert all(len(p) == 1 for p in per_seed), per_seed
    assert len({tuple(sorted(p)) for p in per_seed}) > 1, "heads never varied across seeds"


def test_rejects_bad_inputs() -> None:
    envs = _envs(2)
    with pytest.raises(ValueError, match="at least one env"):
        _az().choose([], _az_infer, [])
    with pytest.raises(ValueError, match="seeds for"):
        _az().choose(envs, _az_infer, [1])
    with pytest.raises(ValueError, match="plies for"):
        _az().choose(envs, _az_infer, [1, 2], plies=[0])
    with pytest.raises(ValueError, match="one composition"):
        mixed = [envs[0], rf.Env(rf.games.Connect4(), rf.Reward(win=0.5, loss=-0.5))]
        _az().choose(mixed, _az_infer, [1, 2])
    with pytest.raises(ValueError, match="with a reward"):
        _az().choose([rf.Env(rf.games.Connect4())], _az_infer, [1])
    with pytest.raises(ValueError, match="finished env"):
        done = rf.Env(rf.games.Connect4(), _R)
        while not done.done():
            agent = done.active_agents()[0]
            done.step({agent: done.legal_actions(agent)[0]})
        _az().choose([done], _az_infer, [1])


def test_rejects_hidden_information_games() -> None:
    env = rf.Env(rf.games.KuhnPoker(), rf.Reward(scale=1.0), seed=0)
    with pytest.raises(ValueError, match=r"hidden-information|clairvoyant|imperfect"):
        _az().choose([env], _az_infer, [1])


def test_callback_junk_raises_not_panics() -> None:
    envs = _envs(2)

    def wrong_shape(obs, n=None):
        return np.zeros((1, 2), dtype=np.float32), np.zeros(1, dtype=np.float32)

    with pytest.raises(ValueError):
        _az().choose(envs, wrong_shape, _seeds(2))

    def raises(obs, n=None):
        raise RuntimeError("boom from infer")

    with pytest.raises(RuntimeError, match="boom from infer"):
        _az().choose(envs, raises, _seeds(2))
