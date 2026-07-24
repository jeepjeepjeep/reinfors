"""``rf.chance_modes`` — how a search consumes a stochastic transition's declared distribution,
passed to a search policy's ``chance=`` kwarg.

The game *declares* chance (``chance_outcomes``/``apply_chance``); these handles pick the search
policy over it. Parameterized variants carry their parameters here (the ``rf.encoders`` pattern):
``AlwaysResample()`` (fresh draw per descent — unbiased, the MCTS/AZ default), ``Committed(samples=k)``
(freeze k draws per edge and plan deeply inside them — expectimax's default at k=1, the wide-fan
trade), ``ExpandAll()`` (evaluate every outcome at expansion — exact, narrow fans). Expand-once
searches (SelectiveExpectimax) reject per-traversal modes at construction.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from . import _reinfors

AlwaysResample = _reinfors.ChanceModeHandle.AlwaysResample
Committed = _reinfors.ChanceModeHandle.Committed
ExpandAll = _reinfors.ChanceModeHandle.ExpandAll

_REGISTRY: dict[str, Callable[..., Any]] = {
    "always_resample": AlwaysResample,
    "committed": Committed,
    "expand_all": ExpandAll,
}


def registered() -> list[str]:
    """The registered chance-mode names, for `make`."""
    return sorted(_REGISTRY)


def make(name: str, **kwargs: Any) -> Any:
    """Construct a chance-mode handle by name (the config-driven path)."""
    try:
        ctor = _REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown chance mode {name!r}; registered: {registered()}") from None
    return ctor(**kwargs)
