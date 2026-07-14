"""Rust-native nets (`rf.nn`, candle — pure Rust). Skipped whole when reinfors is built with
`--no-default-features` — `rf.nn.Conv` raises `ImportError`, which these tests treat as "not built"."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import reinfors as rf

K, A, SHAPE = 8, 7, (2, 6, 7)  # connect4-shaped: (C, H, W), 7 columns, 8 ensemble heads


def _conv_or_skip():
    try:
        return rf.nn.Conv(SHAPE, A, K)
    except ImportError as e:
        pytest.skip(str(e))


def test_forward_has_pooled_khead_shape() -> None:
    net = _conv_or_skip()
    out = net.forward(np.zeros((5, 2 * 6 * 7), dtype=np.float32))
    assert out.shape == (5, K, A) and out.dtype == np.float64
    assert net.n_heads == K and net.n_actions == A


def test_forward_matches_torch() -> None:
    # Export the candle net's weights into an equivalent torch module: the same architecture (Conv2d ·
    # ReLU · Linear · K heads) must reproduce candle's forward to numerical tolerance. Not bit-exact —
    # candle and torch are independent implementations — so this guards the layout/semantics, not kernels.
    torch = pytest.importorskip("torch")
    import torch.nn as tnn

    net = _conv_or_skip()
    obs = np.random.default_rng(0).standard_normal((5, 2 * 6 * 7)).astype(np.float32)
    w = net.get_weights()

    class Ref(tnn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = tnn.Conv2d(2, 16, 3, padding=1)
            self.head = tnn.Linear(16 * 6 * 7, K * A)

        def forward(self, x):  # type: ignore[no-untyped-def]
            return self.head(torch.flatten(self.conv(x).relu(), 1)).view(-1, K, A)

    ref = Ref()
    ref.load_state_dict(
        {
            "conv.weight": torch.tensor(w[0]),
            "conv.bias": torch.tensor(w[1]),
            "head.weight": torch.tensor(w[2]),
            "head.bias": torch.tensor(w[3]),
        }
    )
    with torch.no_grad():
        ref_out = ref(torch.tensor(obs).view(5, 2, 6, 7)).numpy().astype(np.float64)
    assert np.abs(ref_out - net.forward(obs)).max() < 1e-4


def test_set_weights_reaches_live_params() -> None:
    net = _conv_or_skip()
    obs = np.random.default_rng(1).standard_normal((3, 2 * 6 * 7)).astype(np.float32)
    net.set_weights([np.zeros_like(x) for x in net.get_weights()])
    assert np.abs(net.forward(obs)).max() == 0.0  # zeroed weights -> zero output


def test_engine_collect_accepts_a_native_net() -> None:
    # The net is just another `infer` source: `collect` runs the search's forward in Rust, no callback.
    net = _conv_or_skip()
    obs, targets, _masks, telemetry = _engine(net).collect(256, net)
    assert obs.shape[0] > 0 and targets.shape[1:] == (K, A) and telemetry["decisions"] > 0


def _engine(net: object) -> Any:
    return rf.Engine(
        rf.games.Connect4(),
        rf.Reward(win=1.0, loss=-1.0),
        rf.policies.SelectiveExpectimax(n_heads=K, expansion_budget=16, top_k=4, max_depth=4),
        rf.learners.TreeStrap(gamma=0.99, outcome_weight=0.3),
        n_games=8,
        seed=0,
    )


def test_fused_engine_train_runs_entirely_in_rust() -> None:
    # collect (net's Rust forward) + grad step, all in Rust — records never touch Python. Deterministic
    # (seeded), so the loss trajectory is reproducible; it should improve over the initial step.
    net = _conv_or_skip()
    trainer = rf.nn.TreeStrapTrainer(net, lr=1e-3)
    losses = _engine(net).train(net, trainer, steps=10, collect_size=256)
    assert len(losses) == 10 and all(np.isfinite(losses))
    assert min(losses) < losses[0]  # learning happened


def test_stepwise_trainer_update_moves_weights() -> None:
    # Model C: Python owns the loop, Rust does the collect + gradient step on a collected batch.
    net = _conv_or_skip()
    trainer = rf.nn.TreeStrapTrainer(net, lr=1e-3)
    engine = _engine(net)
    before = [w.copy() for w in net.get_weights()]
    obs, targets, masks, _ = engine.collect(256, net)
    loss = trainer.update(net, obs, targets, masks)
    assert np.isfinite(loss)
    assert any(not np.array_equal(a, b) for a, b in zip(net.get_weights(), before, strict=True))


def test_train_head_mismatch_errors_clearly() -> None:
    # A net whose head count differs from the policy's must fail with a clear message, not a candle panic.
    _conv_or_skip()
    net = rf.nn.Conv(SHAPE, A, 3)  # 3 heads vs the policy's K=8
    with pytest.raises(ValueError, match="n_heads"):
        _engine(net).train(net, rf.nn.TreeStrapTrainer(net), steps=1, collect_size=64)
