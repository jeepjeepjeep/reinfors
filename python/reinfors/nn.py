"""Rust-native value nets (candle — pure Rust, no external library), included in the wheel by default.

These satisfy the search's `infer` contract entirely in Rust, so `engine.collect(n, net)` runs the
forward pass without the per-round Python callback. Weights round-trip as numpy in the net's fixed
parameter order (the torch `state_dict` layout: conv/linear weight then bias, trunk before head), so a
torch-trained checkpoint syncs in with `set_weights` and `get_weights` exports for a parity check.

On by default; absent only if reinfors was built with `--no-default-features` (the factories then raise
a clear error). GPU: build reinfors with `--features nn-metal` (Apple) or `--features nn-cuda` (NVIDIA).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from . import _reinfors

# `Net`/`TreeStrapTrainer` are absent only in a `--no-default-features` build. `getattr` keeps this
# import-clean either way, and the factories raise a clear error when missing.
_Net = getattr(_reinfors, "Net", None)
_Trainer = getattr(_reinfors, "TreeStrapTrainer", None)


def _require() -> Any:
    if _Net is None:
        raise ImportError(
            "Rust-native nets are missing — reinfors was built with --no-default-features. "
            "Rebuild with the default features (`maturin develop`), which include `nn`."
        )
    return _Net


def Conv(obs_shape: Sequence[int], n_actions: int, n_heads: int) -> Any:
    """Conv trunk + K linear heads, sized from a planar observation shape `(C, H, W)`."""
    c, h, w = obs_shape
    return _require().conv((c, h, w), n_actions, n_heads)


def Mlp(in_dim: int, hidden: int, n_actions: int, n_heads: int) -> Any:
    """Two-layer MLP + K linear heads, for a flattened observation vector of `in_dim`."""
    return _require().mlp(in_dim, hidden, n_actions, n_heads)


def TreeStrapTrainer(net: Any, lr: float = 2.5e-4) -> Any:
    """Adam + masked-Huber trainer over `net`'s parameters — the in-Rust learning half. Drive it fused
    (`engine.train(net, trainer, steps, collect_size)`, whole loop in Rust) or step-by-step from a Python
    loop (`trainer.update(net, obs, targets, masks)` on a collected batch)."""
    if _Trainer is None:
        raise ImportError(
            "Rust-native training is missing — reinfors was built with --no-default-features. "
            "Rebuild with the default features (`maturin develop`)."
        )
    return _Trainer(net, lr)
