"""The chance_mode surface: kwargs parse, invalid modes reject, and deterministic games are inert
(no game today declares chance_outcomes — the tree semantics are pinned by the Rust chance suite)."""

import numpy as np
import pytest
import reinfors as rf
from reinfors._reinfors import AlphaZeroBatch, PolicyHandle


def _infer(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.zeros((arr.shape[0], 7)), np.zeros(arr.shape[0])


def _collect(policy: PolicyHandle) -> AlphaZeroBatch:
    engine = rf.Engine(
        rf.games.Connect4(),
        rf.Reward(win=1.0, loss=-1.0),
        policy,
        rf.learners.AlphaZero(),
        n_games=2,
        seed=0,
    )
    batch = engine.collect(40, _infer)
    assert isinstance(batch, AlphaZeroBatch)
    return batch


def test_chance_mode_kwargs_accepted() -> None:
    for mode in ("always_resample", "committed", "expand_all"):
        rf.policies.Mcts(chance_mode=mode)
        rf.policies.AlphaZero(chance_mode=mode, chance_samples=2)


def test_invalid_chance_mode_rejected() -> None:
    with pytest.raises(ValueError, match="chance_mode"):
        rf.policies.Mcts(chance_mode="sample")
    with pytest.raises(ValueError, match="chance_samples"):
        rf.policies.AlphaZero(chance_mode="committed", chance_samples=0)


def test_chance_mode_inert_for_deterministic_games() -> None:
    # connect4 declares no chance: every mode must produce bit-identical batches, zero fan rows.
    batches = [
        _collect(rf.policies.AlphaZero(num_simulations=8, chance_mode=mode, chance_samples=2))
        for mode in ("always_resample", "committed", "expand_all")
    ]
    for b in batches[1:]:
        assert np.array_equal(batches[0].obs, b.obs)
        assert np.array_equal(batches[0].policy_targets, b.policy_targets)
        assert np.array_equal(batches[0].value_targets, b.value_targets)
    assert all(b.telemetry["fan_extra_rows"] == 0 for b in batches)


def test_identity_includes_fan_term() -> None:
    t = _collect(rf.policies.AlphaZero(num_simulations=8)).telemetry
    lhs = t["decisions"] * 8
    rhs = (
        t["fresh_rows"]
        + t["hit_rows"]
        + t["shared_rows"]
        + t["terminal_sims"]
        + t["depthcap_sims"]
        - t["fan_extra_rows"]
    )
    assert lhs == rhs
