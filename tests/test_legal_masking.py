"""Sparse-legality completion: DQN acting stays inside legal sets, DQN batches carry legality
masks for masked TD targets, and the Env boundary rejects illegal actions."""

import numpy as np
import pytest
import reinfors as rf
from reinfors._reinfors import DqnBatch


def _chess_dqn(seed: int = 0) -> rf.Engine:
    return rf.Engine(
        rf.games.Chess(max_ticks=30),
        rf.Reward(win=1.0, loss=-1.0),
        rf.policies.EpsilonGreedyQ(epsilon=0.3),
        rf.learners.Dqn(),
        n_games=2,
        seed=seed,
    )


def _zeros_infer(arr: np.ndarray) -> np.ndarray:
    return np.zeros((arr.shape[0], 1, 4672))


def test_dqn_on_chess_acts_and_masks_legally() -> None:
    # Chess: ~35 legal of 4672 ids. Every recorded action must be legal in its own state, and the
    # batch's masks must be sparse and consistent (previously: dense argmax over phantom Qs).
    batch = _chess_dqn().collect(60, _zeros_infer)
    assert isinstance(batch, DqnBatch)
    m, a = batch.legal_masks.shape
    assert a == 4672
    assert m >= 60
    # each action was legal in its state
    assert np.all(batch.legal_masks[np.arange(m), batch.actions] == 1.0)
    # sparse boards: far fewer legal actions than the id space
    assert np.all(batch.legal_masks.sum(axis=1) < 300)
    assert np.all(batch.legal_masks.sum(axis=1) >= 1)
    # next-state masks: all-zero at terminals; interior steps bootstrap from the agent's own
    # NEXT TURN (>= 1 legal). The only permitted non-terminal zeros are truncation-final steps
    # (alternating game: the post-move view is opponent-to-move -> documented no-bootstrap).
    next_counts = batch.next_legal_masks.sum(axis=1)
    assert np.all(next_counts[batch.dones] == 0.0)
    live = next_counts[~batch.dones]
    assert (live >= 1).mean() > 0.8, "most interior steps must have an own-turn successor"
    assert np.all(live[live >= 1] < 300)


def test_dqn_masks_are_dense_on_all_legal_games() -> None:
    def infer(arr: np.ndarray) -> np.ndarray:
        return np.zeros((arr.shape[0], 1, 4))

    engine = rf.Engine(
        rf.games.GridWorld(),
        rf.Reward(step=0.0, goal=1.0),
        rf.policies.EpsilonGreedyQ(epsilon=0.5),
        rf.learners.Dqn(),
        n_games=2,
        seed=0,
    )
    batch = engine.collect(40, infer)
    assert isinstance(batch, DqnBatch)
    assert np.all(batch.legal_masks == 1.0)
    nl = batch.next_legal_masks
    assert np.all(nl[~batch.dones] == 1.0)
    assert np.all(nl[batch.dones] == 0.0)


def test_documented_td_target_is_finite_for_every_record() -> None:
    # The COMPLETE documented recipe — not just the masked max — must produce a finite target for
    # every record class: bootstrappable interior steps, terminals, and truncation tails (done =
    # False with an all-zero next mask, which chess with a tight max_ticks reliably produces).
    batch = _chess_dqn(seed=1).collect(60, _zeros_infer)
    assert isinstance(batch, DqnBatch)
    gamma = 0.99
    q_next = np.random.default_rng(0).normal(size=batch.next_legal_masks.shape)
    # the documented incantation, verbatim:
    q = np.where(batch.next_legal_masks > 0, q_next, -np.inf).max(axis=1)
    td = batch.rewards + gamma * np.where(np.isfinite(q), q, 0.0)
    assert np.all(np.isfinite(td)), "the complete target must never be inf/NaN"
    # no-bootstrap rows (terminals AND truncation tails) collapse to target = r exactly
    no_bootstrap = batch.next_legal_masks.sum(axis=1) == 0
    assert np.any(no_bootstrap & ~batch.dones), "a tight max_ticks must produce truncation tails"
    assert np.allclose(td[no_bootstrap], batch.rewards[no_bootstrap])
    # and the naive (1 - done) * max pattern really does blow up on these records — the reason
    # the docs forbid it (regression guard for the documentation itself)
    with np.errstate(invalid="ignore"):
        naive = batch.rewards + gamma * (1.0 - batch.dones) * q
    assert not np.all(np.isfinite(naive))


def test_env_rejects_illegal_actions() -> None:
    env = rf.Env(rf.games.Chess(), seed=0)
    mover = env.active_agents()[0]
    legal = set(env.legal_actions(mover))
    illegal = next(a for a in range(4672) if a not in legal)
    with pytest.raises(ValueError, match="illegal"):
        env.step({mover: illegal})
    env.step({mover: next(iter(legal))})  # legal actions still work
