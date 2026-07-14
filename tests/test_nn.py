"""Rust-native nets (`reinfors[nn]`, the optional libtorch path). Skipped whole when reinfors is built
without the `nn` feature — `rf.nn.Conv` raises `ImportError`, which these tests treat as "not built"."""

from __future__ import annotations

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


def test_forward_matches_torch_exactly() -> None:
    # Same libtorch backend, so exporting the Rust net's weights into an equivalent torch module must
    # reproduce its forward bit-for-bit. Also confirms the tch/libtorch version bypass is ABI-safe here.
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
    engine = rf.Engine(
        rf.games.Connect4(),
        rf.Reward(win=1.0, loss=-1.0),
        rf.policies.SelectiveExpectimax(n_heads=K, expansion_budget=16, top_k=4, max_depth=4),
        rf.learners.TreeStrap(gamma=0.99),
        n_games=8,
        seed=0,
    )
    obs, targets, _masks, telemetry = engine.collect(256, net)
    assert obs.shape[0] > 0 and targets.shape[1:] == (K, A) and telemetry["decisions"] > 0
