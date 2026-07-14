"""Optional Rust-native value nets (libtorch via tch), the `reinfors[nn]` extra.

These satisfy the search's `infer` contract entirely in Rust, so `engine.collect(n, net)` runs the
forward pass without the per-round Python callback. Weights round-trip as numpy in the net's fixed
parameter order (the torch `state_dict` layout: conv/linear weight then bias, trunk before head), so a
torch-trained checkpoint syncs in with `set_weights` and `get_weights` exports for a parity check.

Available only when reinfors is built with the `nn` feature; the factories raise a clear error otherwise.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from . import _reinfors

# `Net` exists only when built with the `nn` feature (libtorch); absent otherwise. `getattr` keeps this
# import-clean either way, and the factories raise a clear error when it's missing.
_Net = getattr(_reinfors, "Net", None)


def _require() -> Any:
    if _Net is None:
        raise ImportError(
            "Rust-native nets need reinfors built with the 'nn' extra (libtorch). "
            "Install `reinfors[nn]`, or build with `maturin develop --features nn`."
        )
    return _Net


def Conv(obs_shape: Sequence[int], n_actions: int, n_heads: int) -> Any:
    """Conv trunk + K linear heads, sized from a planar observation shape `(C, H, W)`."""
    c, h, w = obs_shape
    return _require().conv((c, h, w), n_actions, n_heads)


def Mlp(in_dim: int, hidden: int, n_actions: int, n_heads: int) -> Any:
    """Two-layer MLP + K linear heads, for a flattened observation vector of `in_dim`."""
    return _require().mlp(in_dim, hidden, n_actions, n_heads)
