"""``rf.games`` — game handles for composing an ``Engine``.

Each constructor returns an opaque game handle (carrying the game's config); pass it to ``rf.Engine``
alongside a policy and a learner. ``make`` / ``registered`` are the name-addressable form for
config-driven construction. Adding a game = one entry in ``_REGISTRY``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from . import _reinfors

Snake = _reinfors.GameHandle.Snake
Chess = _reinfors.GameHandle.Chess
Connect4 = _reinfors.GameHandle.Connect4
GridWorld = _reinfors.GameHandle.GridWorld

_REGISTRY: dict[str, Callable[..., Any]] = {
    "snake": Snake,
    "chess": Chess,
    "connect4": Connect4,
    "gridworld": GridWorld,
}


def registered() -> list[str]:
    """The registered game names, for `make`."""
    return sorted(_REGISTRY)


def make(name: str, **kwargs: Any) -> Any:
    """Construct a game handle by name (the config-driven path); kwargs match the typed constructor."""
    try:
        ctor = _REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown game {name!r}; registered: {registered()}") from None
    return ctor(**kwargs)
