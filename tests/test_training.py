"""End-to-end trainer: the actor-learner loop runs, learns, and the masked-Huber loss + ensemble
network match snake_RL's. torch-gated (skips where torch is absent, e.g. CI).
"""

import os
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import reinfors  # noqa: E402
from reinfors.training import BootstrappedQNetwork, make_infer, train, treestrap_loss  # noqa: E402

# Make the sibling snake_RL checkout importable for the oracle forward-parity test (skipped if absent).
_SNAKE_RL_SRC = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "snake_RL", "src"))
if os.path.isdir(_SNAKE_RL_SRC) and _SNAKE_RL_SRC not in sys.path:
    sys.path.insert(0, _SNAKE_RL_SRC)

_G = 8
_K = 2
_A = 3
_REWARD = (0.0, 0.0, -10.0, -6.0, 20.0, 20.0, 0.0)


def _engine(seed: int) -> object:
    return reinfors._reinfors.Engine(
        4,
        _G,
        3,
        False,
        None,
        2,  # n_games, grid, initial_length, play_to_last, win_food_lead, initial_food_count
        0.99,
        1.0,
        16,
        4,
        5,  # gamma, beta, expansion_budget, top_k, max_depth
        _REWARD,
        "uniform",
        1.0,
        0.1,  # reward, opponent, opp_temperature, opp_floor
        0.1,
        30,
        _K,  # epsilon, max_ticks, n_heads
        0.5,
        True,
        0.8,  # outcome_weight, interior_targets, bootstrap_p
        seed,
    )


def _net(seed: int) -> BootstrappedQNetwork:
    torch.manual_seed(seed)
    return BootstrappedQNetwork((5, _G, _G), _A, _K)


def test_train_loop_runs_and_is_deterministic() -> None:
    def run() -> list[float]:
        net = _net(0)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        return train(_engine(7), net, opt, iterations=4, batch_size=24)

    losses1, losses2 = run(), run()
    assert len(losses1) == 4
    assert all(np.isfinite(losses1))
    assert losses1 == losses2  # same net init + engine seed -> identical loop


def test_infer_callback_shape_and_dtype() -> None:
    infer = make_infer(_net(0))
    out = infer(np.zeros((6, 5 * _G * _G), dtype=np.float32))
    assert out.shape == (6, _K, _A) and out.dtype == np.float64


def test_overfits_a_fixed_batch() -> None:
    # Train repeatedly on one collected batch: the net + masked-Huber loss + optimizer must drive the
    # loss down. (The full loop re-collects each step; here we fix the batch to isolate "it learns".)
    net = _net(0)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    obs, target, mask = _engine(1).collect(64, make_infer(net))
    obs_t = torch.from_numpy(obs).reshape(-1, 5, _G, _G)
    target_t = torch.from_numpy(target).float()
    mask_t = torch.from_numpy(mask)
    first = treestrap_loss(net(obs_t), target_t, mask_t).item()
    for _ in range(200):
        loss = treestrap_loss(net(obs_t), target_t, mask_t)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.5 * first, f"loss did not drop: {first} -> {loss.item()}"


def test_treestrap_loss_masking() -> None:
    rng = np.random.default_rng(0)
    q = torch.from_numpy(rng.standard_normal((8, _K, _A)))
    target = torch.from_numpy(rng.standard_normal((8, _K, _A)))
    # All-zero mask contributes no loss; all-ones mask is the plain mean Huber.
    assert treestrap_loss(q, target, torch.zeros(8, _K)).item() == 0.0
    full = treestrap_loss(q, target, torch.ones(8, _K))
    expected = torch.nn.functional.smooth_l1_loss(q, target)
    assert torch.allclose(full, expected)


def test_forward_matches_oracle_network() -> None:
    # The port is bit-compatible with snake_RL's BootstrappedQNetwork: same architecture and state_dict
    # keys, so the oracle's weights load and produce the same (B, K, A) forward.
    pytest.importorskip("snake_rl.agent.shared.network")
    from snake_rl.agent.shared.network import BootstrappedQNetwork as OracleNet

    torch.manual_seed(3)
    oracle = OracleNet((5, _G, _G), _A, _K)
    rein = BootstrappedQNetwork((5, _G, _G), _A, _K)
    rein.load_state_dict(oracle.state_dict())
    rein.eval()
    oracle.eval()
    x = torch.from_numpy(np.random.default_rng(0).standard_normal((5, 5, _G, _G)).astype(np.float32))
    assert torch.allclose(rein(x), oracle(x), atol=1e-6)
