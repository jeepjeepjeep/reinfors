"""End-to-end trainer: the actor-learner loop runs, learns, and the masked-Huber loss + ensemble
network match snake_RL's. torch-gated (skips where torch is absent, e.g. CI).
"""

import os
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import reinfors  # noqa: E402
from reinfors.training import (  # noqa: E402
    BootstrappedQNetwork,
    CollectReport,
    ReplayBuffer,
    StepMetrics,
    make_infer,
    train,
    treestrap_loss,
)

# Make the sibling snake_RL checkout importable for the oracle forward-parity test (skipped if absent).
_SNAKE_RL_SRC = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "snake_RL", "src"))
if os.path.isdir(_SNAKE_RL_SRC) and _SNAKE_RL_SRC not in sys.path:
    sys.path.insert(0, _SNAKE_RL_SRC)

_G = 8
_K = 2
_A = 3
_REWARD = (0.0, 0.0, -10.0, -6.0, 20.0, 20.0, 0.0)


def _engine(seed: int, *, interior: bool = True, max_ticks: int = 30) -> object:
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
        max_ticks,
        _K,  # epsilon, max_ticks, n_heads
        0.5,
        interior,
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
        return train(
            _engine(7),
            net,
            opt,
            iterations=4,
            collect_size=24,
            batch_size=16,
            grad_steps_per_collect=2,
            min_buffer_size=16,
            seed=0,
        )

    losses1, losses2 = run(), run()
    assert len(losses1) == 8  # 4 collects x 2 grad steps (buffer fills on the first collect)
    assert all(np.isfinite(losses1))
    assert losses1 == losses2  # same net init + engine + buffer seeds -> identical loop


def test_train_stops_at_max_episodes() -> None:
    # max_episodes is the budget that matches snake_RL's num_episodes: the loop runs until that many
    # self-play episodes finish, regardless of how many collects/gradient steps that takes. Short
    # episodes (max_ticks=5) so the budget is reached in a few collects; the generous iteration cap
    # must NOT be what stops it.
    net = _net(0)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    eps: list = []
    its: list[int] = []

    def on_collect(it: int, r: CollectReport) -> None:
        eps.extend(r.telemetry["episodes"])  # type: ignore[arg-type]
        its.append(it)

    train(
        _engine(7, interior=False, max_ticks=5),
        net,
        opt,
        max_episodes=5,
        iterations=500,  # safety cap; episode budget should stop the loop well before this
        collect_size=16,
        batch_size=16,
        min_buffer_size=16,
        on_collect=on_collect,
    )
    assert len(eps) >= 5
    assert max(its) < 100, "should stop on the episode budget, not the iteration cap"


def test_train_requires_a_budget() -> None:
    net = _net(0)
    with pytest.raises(ValueError, match="iterations"):
        train(
            _engine(7),
            net,
            torch.optim.Adam(net.parameters()),
            collect_size=16,
            batch_size=16,
        )


def test_train_telemetry_callbacks() -> None:
    # on_step fires once per gradient step with finite metrics; on_collect once per iteration (incl.
    # the warm-up collect before the buffer fills) carrying the Engine telemetry + throughput, so a
    # caller can log the loss curve, self-play learning curve, and data-gen speed.
    net = _net(0)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    steps: list[StepMetrics] = []
    reports: list[CollectReport] = []
    losses = train(
        _engine(7),
        net,
        opt,
        iterations=4,
        collect_size=24,
        batch_size=16,
        grad_steps_per_collect=2,
        min_buffer_size=16,
        on_step=lambda _i, m: steps.append(m),
        on_collect=lambda _i, r: reports.append(r),
    )
    assert len(steps) == len(losses)
    for m in steps:
        assert np.isfinite(m.loss) and np.isfinite(m.mean_q) and np.isfinite(m.mean_target_q)
    assert [m.loss for m in steps] == losses
    assert len(reports) == 4  # one per iteration, including warm-up collects
    for it, r in enumerate(reports):
        assert r.iteration == it and r.records > 0 and r.seconds >= 0.0
        assert "episodes" in r.telemetry and "mean_sigma" in r.telemetry
        for reward_a, reward_b, length in r.telemetry["episodes"]:
            assert length >= 1 and np.isfinite(reward_a) and np.isfinite(reward_b)


def test_train_reuses_records() -> None:
    # The whole point of the buffer: amortise the (expensive) search by reusing records. With more
    # grad steps per collect, the same number of collects yields proportionally more gradient steps.
    def n_steps(grad_steps: int) -> int:
        net = _net(0)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        losses = train(
            _engine(7),
            net,
            opt,
            iterations=3,
            collect_size=24,
            batch_size=16,
            grad_steps_per_collect=grad_steps,
            min_buffer_size=16,
        )
        return len(losses)

    assert n_steps(4) == 4 * n_steps(1)


def test_replay_buffer_ring_and_sample_shape() -> None:
    rng = np.random.default_rng(0)
    cap, dim = 10, 5 * _G * _G
    buf = ReplayBuffer(cap, dim, _K, _A, seed=0)
    # Overfill so the ring wraps; size saturates at capacity.
    for _ in range(4):
        buf.push_batch(
            rng.standard_normal((4, dim)).astype(np.float32),
            rng.standard_normal((4, _K, _A)).astype(np.float32),
            (rng.random((4, _K)) < 0.8).astype(np.float32),
        )
    assert buf.size == cap
    batch = buf.sample(7)
    assert tuple(batch.shape) == (7, dim + _K * _A + _K) and batch.dtype == torch.float32


def test_replay_buffer_matches_oracle() -> None:
    # Faithful port of EnsembleTreeStrapBuffer: identical row layout + ring semantics, so after the
    # same pushes a same-seed sample returns identical rows.
    treestrap = pytest.importorskip("snake_rl.agent.model_based.treestrap")
    cap, dim, seed = 64, 5 * _G * _G, 1
    rein = ReplayBuffer(cap, dim, _K, _A, seed=seed)
    oracle = treestrap.EnsembleTreeStrapBuffer(cap, (5, _G, _G), _A, _K, seed)
    rng = np.random.default_rng(0)
    for _ in range(50):
        obs = rng.standard_normal((1, dim)).astype(np.float32)
        target = rng.standard_normal((1, _K, _A)).astype(np.float32)
        mask = (rng.random((1, _K)) < 0.8).astype(np.float32)
        rein.push_batch(obs, target, mask)
        oracle.push(treestrap.EnsembleSearchTarget(obs[0], target[0], mask[0]))
    assert rein.size == oracle.size
    assert np.array_equal(rein.sample(16).numpy(), oracle.sample(16).numpy())


def test_infer_callback_shape_and_dtype() -> None:
    infer = make_infer(_net(0))
    out = infer(np.zeros((6, 5 * _G * _G), dtype=np.float32))
    assert out.shape == (6, _K, _A) and out.dtype == np.float64


def test_mps_infer_matches_cpu_forward() -> None:
    # The real-net forward on the GPU (MPS) matches CPU within float tolerance, so the search gets
    # the same values whichever device serves inference. Skips where MPS is unavailable (CI, non-Mac).
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available")
    net = _net(0)
    obs = np.random.default_rng(0).standard_normal((8, 5 * _G * _G)).astype(np.float32)
    cpu = make_infer(net, "cpu")(obs)
    mps = make_infer(net, "mps")(obs)
    assert cpu.shape == mps.shape == (8, _K, _A)
    assert np.allclose(cpu, mps, atol=1e-4), f"max |cpu-mps| = {np.abs(cpu - mps).max()}"


def test_pipeline_runs_on_mps() -> None:
    # The whole loop — pooled search calling the GPU `infer`, replay, masked-Huber gradient steps, all
    # on MPS — runs end to end with finite losses. Validates the design target: real GPU inference
    # driving the Rust data generator (it's a moving-target loop, so we don't assert loss descent here
    # — that's the fixed-batch test below). Skips without MPS.
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available")
    net = _net(0)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    losses = train(
        _engine(7),
        net,
        opt,
        iterations=4,
        collect_size=24,
        batch_size=16,
        grad_steps_per_collect=2,
        min_buffer_size=16,
        device="mps",
    )
    assert len(losses) == 8 and all(np.isfinite(losses))


def test_mps_training_reduces_loss_on_fixed_batch() -> None:
    # GPU training learns: collect one batch (via the MPS net) and overfit it on MPS — the masked-Huber
    # loss falls past half. The MPS analogue of test_overfits_a_fixed_batch. Skips without MPS.
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available")
    net = _net(0)
    obs, target, mask, _ = _engine(1).collect(64, make_infer(net, "mps"))  # make_infer moved net to MPS
    obs_t = torch.from_numpy(obs).reshape(-1, 5, _G, _G).to("mps")
    target_t = torch.from_numpy(target).float().to("mps")
    mask_t = torch.from_numpy(mask).to("mps")
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    first = treestrap_loss(net(obs_t), target_t, mask_t).item()
    for _ in range(200):
        loss = treestrap_loss(net(obs_t), target_t, mask_t)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.5 * first, f"loss did not drop on MPS: {first} -> {loss.item()}"


def test_overfits_a_fixed_batch() -> None:
    # Train repeatedly on one collected batch: the net + masked-Huber loss + optimizer must drive the
    # loss down. (The full loop re-collects each step; here we fix the batch to isolate "it learns".)
    net = _net(0)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    obs, target, mask, _ = _engine(1).collect(64, make_infer(net))
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


def test_infer_is_mode_neutral_and_train_ends_in_train_mode() -> None:
    # `infer` runs its forward in eval mode (correct inference if BatchNorm/Dropout are ever added) but
    # restores the net's prior mode — so it's side-effect-free. And `train()` leaves the net in train
    # mode regardless of whether the buffer filled, so its post-condition doesn't depend on whether a
    # gradient step happened (the never-fills edge the split eval/train handling had).
    obs = np.zeros((4, 5 * _G * _G), dtype=np.float32)
    net = _net(0)
    net.eval()
    make_infer(net)(obs)
    assert not net.training, "infer must restore eval mode"
    net.train()
    make_infer(net)(obs)
    assert net.training, "infer must restore train mode"
    # A run whose buffer never fills takes no gradient step, yet must still end in train mode.
    net.eval()
    train(
        _engine(7),
        net,
        torch.optim.Adam(net.parameters()),
        iterations=2,
        collect_size=8,
        batch_size=16,
        grad_steps_per_collect=1,
        min_buffer_size=10**9,
    )
    assert net.training, "train() must end in train mode even when no gradient step ran"


def test_priors_stay_frozen_during_training() -> None:
    # The randomized priors are fixed per-head offsets that keep the heads disagreeing — the
    # epistemic-uncertainty signal (sigma) the whole selective search expands on. They must never
    # receive gradient or change; a regression leaking gradient in would still drop the loss and pass
    # every other test while silently destroying that signal. Snapshot, train, assert frozen.
    net = _net(0)
    priors = [h.prior.weight.detach().clone() for h in net.heads]
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    train(
        _engine(7),
        net,
        opt,
        iterations=5,
        collect_size=24,
        batch_size=16,
        grad_steps_per_collect=3,
        min_buffer_size=16,
    )
    for h, before in zip(net.heads, priors, strict=True):
        assert h.prior.weight.grad is None, "a prior received a gradient"
        assert torch.equal(h.prior.weight, before), "a prior weight changed during training"


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
