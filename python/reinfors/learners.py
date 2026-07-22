"""``rf.learners`` — learning-algorithm handles for composing an ``Engine``.

Each constructor returns an opaque learner handle. A learner pairs with the policy family that
produces the evaluation it consumes (``TreeStrap`` ↔ ``SelectiveExpectimax``/``Mcts``, ``Dqn`` ↔
``EpsilonGreedyQ``, ``AlphaZero`` ↔ ``AlphaZero``); ``Engine`` rejects an incompatible pairing.
``make`` / ``registered`` are the name-addressable form.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from . import _reinfors

TreeStrap = _reinfors.LearnerHandle.TreeStrap
Dqn = _reinfors.LearnerHandle.Dqn
AlphaZero = _reinfors.LearnerHandle.AlphaZero

_REGISTRY: dict[str, Callable[..., Any]] = {
    "treestrap": TreeStrap,
    "dqn": Dqn,
    "alphazero": AlphaZero,
}


def registered() -> list[str]:
    """The registered learner names, for `make`."""
    return sorted(_REGISTRY)


def make(name: str, **kwargs: Any) -> Any:
    """Construct a learner handle by name (the config-driven path); kwargs match the typed constructor."""
    try:
        ctor = _REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown learner {name!r}; registered: {registered()}") from None
    return ctor(**kwargs)
