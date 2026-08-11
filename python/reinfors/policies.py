"""``rf.policies`` — acting-policy handles for composing an ``Engine``.

Each constructor returns an opaque policy handle. ``n_heads`` (ensemble size) lives here — the single
source the learner reads from. ``make`` / ``registered`` are the name-addressable form.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from . import _reinfors
from .catalog import POLICIES

SelectiveExpectimax = _reinfors.PolicyHandle.SelectiveExpectimax
EpsilonGreedyQ = _reinfors.PolicyHandle.EpsilonGreedyQ
Mcts = _reinfors.PolicyHandle.Mcts
Minimax = _reinfors.PolicyHandle.Minimax
AlphaZero = _reinfors.PolicyHandle.AlphaZero

_REGISTRY: dict[str, Callable[..., Any]] = {
    "selective_expectimax": SelectiveExpectimax,
    "epsilon_greedy_q": EpsilonGreedyQ,
    "mcts": Mcts,
    "minimax": Minimax,
    "alphazero": AlphaZero,
}
assert _REGISTRY.keys() == POLICIES, "policy registry and documentation catalogue diverged"


def registered() -> list[str]:
    """The registered policy names, for `make`."""
    return sorted(_REGISTRY)


def make(name: str, **kwargs: Any) -> Any:
    """Construct a policy handle by name (the config-driven path); kwargs match the typed constructor."""
    try:
        ctor = _REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown policy {name!r}; registered: {registered()}") from None
    return ctor(**kwargs)
