"""The chance_mode surface: kwargs parse, invalid modes reject, and deterministic games are inert
(the tree semantics over declared chance states are pinned by the Rust chance suite)."""

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


def test_chance_mode_handles_accepted() -> None:
    for mode in (
        rf.chance_modes.AlwaysResample(),
        rf.chance_modes.Committed(samples=2),
        rf.chance_modes.ExpandAll(),
    ):
        rf.policies.Mcts(chance=mode)
        rf.policies.AlphaZero(chance=mode)


def test_invalid_chance_config_rejected() -> None:
    with pytest.raises(ValueError, match="samples"):
        rf.chance_modes.Committed(samples=0)
    with pytest.raises(KeyError, match="unknown chance mode"):
        rf.chance_modes.make("bogus")


def test_chance_mode_inert_for_deterministic_games() -> None:
    # connect4 declares no chance: every mode must produce bit-identical batches, zero fan rows.
    batches = [
        _collect(rf.policies.AlphaZero(num_simulations=8, chance=mode))
        for mode in (
            rf.chance_modes.AlwaysResample(),
            rf.chance_modes.Committed(samples=2),
            rf.chance_modes.ExpandAll(),
        )
    ]
    for b in batches[1:]:
        assert np.array_equal(batches[0].obs, b.obs)
        assert np.array_equal(batches[0].policy_targets, b.policy_targets)
        assert np.array_equal(batches[0].value_targets, b.value_targets)
    assert all(b.telemetry["extra_eval_rows"] == 0 for b in batches)


def test_identity_includes_fan_term() -> None:
    t = _collect(rf.policies.AlphaZero(num_simulations=8)).telemetry
    lhs = t["decisions"] * 8
    rhs = t["requested_rows"] - t["tail_rows"] + t["terminal_sims"] + t["depthcap_sims"] - t["extra_eval_rows"]
    assert lhs == rhs
