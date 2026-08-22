"""``rf.encoders`` — observation-encoder handles, passed to a game handle's ``encoder=`` kwarg.

An encoder is a configurable *view* of a game's state — the game owns dynamics, the encoder owns
representation (the decoupling the ``StateEncoder`` seam exists for). Every game constructor accepts
one and defaults to its registered standard view. Encoders are game-specific; any state bookkeeping
a view needs (e.g. the AlphaZero chess history ring) is enabled in the game automatically when that
encoder is selected. ``make`` / ``registered`` are the name-addressable form.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from . import _reinfors
from .catalog import ENCODERS

Snake = _reinfors.EncoderHandle.Snake
Connect4 = _reinfors.EncoderHandle.Connect4
MinimalChess = _reinfors.EncoderHandle.MinimalChess
RelativeChess = _reinfors.EncoderHandle.RelativeChess
OpenSpielChess = _reinfors.EncoderHandle.OpenSpielChess
AlphaZeroChess = _reinfors.EncoderHandle.AlphaZeroChess
Backgammon = _reinfors.EncoderHandle.Backgammon
TexasHoldem = _reinfors.EncoderHandle.TexasHoldem
KuhnPoker = _reinfors.EncoderHandle.KuhnPoker
LeducPoker = _reinfors.EncoderHandle.LeducPoker
GridWorld = _reinfors.EncoderHandle.GridWorld
CarRacingPixels = _reinfors.EncoderHandle.CarRacingPixels
CarRacingVec = _reinfors.EncoderHandle.CarRacingVec

_REGISTRY: dict[str, Callable[..., Any]] = {
    "snake": Snake,
    "connect4": Connect4,
    "minimal_chess": MinimalChess,
    "relative_chess": RelativeChess,
    "openspiel_chess": OpenSpielChess,
    "alphazero_chess": AlphaZeroChess,
    "backgammon": Backgammon,
    "texas_holdem": TexasHoldem,
    "kuhn_poker": KuhnPoker,
    "leduc_poker": LeducPoker,
    "gridworld": GridWorld,
    "car_racing_pixels": CarRacingPixels,
    "car_racing_vec": CarRacingVec,
}
assert _REGISTRY.keys() == ENCODERS, "encoder registry and documentation catalogue diverged"


def registered() -> list[str]:
    """The registered encoder names, for `make`."""
    return sorted(_REGISTRY)


def make(name: str, **kwargs: Any) -> Any:
    """Construct an encoder handle by name (the config-driven path); kwargs match the typed constructor."""
    try:
        ctor = _REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown encoder {name!r}; registered: {registered()}") from None
    return ctor(**kwargs)
