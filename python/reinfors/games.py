"""``rf.games`` — game handles for composing an ``Engine``.

Each constructor returns an opaque game handle (carrying the game's config); pass it to ``rf.Engine``
alongside a policy and a learner. ``make`` / ``registered`` are the name-addressable form for
config-driven construction. Adding a game = one entry in ``_REGISTRY``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from . import _reinfors
from .catalog import GAMES

Snake = _reinfors.GameHandle.Snake
TexasHoldem = _reinfors.GameHandle.TexasHoldem
KuhnPoker = _reinfors.GameHandle.KuhnPoker
LeducPoker = _reinfors.GameHandle.LeducPoker
Chess = _reinfors.GameHandle.Chess
Backgammon = _reinfors.GameHandle.Backgammon
Connect4 = _reinfors.GameHandle.Connect4
GridWorld = _reinfors.GameHandle.GridWorld
CarRacing = _reinfors.GameHandle.CarRacing

_REGISTRY: dict[str, Callable[..., Any]] = {
    "snake": Snake,
    "texas_holdem": TexasHoldem,
    "kuhn_poker": KuhnPoker,
    "leduc_poker": LeducPoker,
    "chess": Chess,
    "backgammon": Backgammon,
    "connect4": Connect4,
    "gridworld": GridWorld,
    "car_racing": CarRacing,
}
assert _REGISTRY.keys() == GAMES.keys(), "game registry and documentation catalogue diverged"


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
