"""``rf.encoders`` — observation-encoder handles, passed to a game handle's ``encoder=`` kwarg.

An encoder is a configurable *view* of a game's state — the game owns dynamics, the encoder owns
representation (the decoupling the ``StateEncoder`` seam exists for). Encoders are game-specific;
any state bookkeeping a view needs (e.g. the AlphaZero chess history ring) is enabled in the game
automatically when that encoder is selected. ``make`` / ``registered`` are the name-addressable form.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from . import _reinfors

MinimalChess = _reinfors.EncoderHandle.MinimalChess
RelativeChess = _reinfors.EncoderHandle.RelativeChess
OpenSpielChess = _reinfors.EncoderHandle.OpenSpielChess
AlphaZeroChess = _reinfors.EncoderHandle.AlphaZeroChess

_REGISTRY: dict[str, Callable[..., Any]] = {
    "minimal_chess": MinimalChess,
    "relative_chess": RelativeChess,
    "openspiel_chess": OpenSpielChess,
    "alphazero_chess": AlphaZeroChess,
}


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
