"""``rf.noise`` — root-exploration-noise handles for ``rf.policies.AlphaZero(noise=...)``.

``Dirichlet(epsilon, alpha, scope)`` is the classic AlphaZero root mix
``(1-epsilon)*P + epsilon*Dir(alpha)``, drawn from the seeded stream (collects stay reproducible).
``scope`` ("requester" | "all") picks which root priors are perturbed in a *simultaneous* search
tree. Pass ``noise=None`` to disable noise honestly (no ``epsilon=0`` sentinel); omit the kwarg
for the self-play default ``Dirichlet(0.25, 0.3, "requester")``.
"""

from __future__ import annotations

from typing import Any

from . import _reinfors
from .catalog import NOISE

Dirichlet = _reinfors.NoiseHandle.Dirichlet

_REGISTRY = {
    "dirichlet": Dirichlet,
}
assert _REGISTRY.keys() == NOISE, "noise registry and documentation catalogue diverged"


def registered() -> list[str]:
    """The registered noise names, for `make`."""
    return sorted(_REGISTRY)


def make(name: str, **kwargs: Any) -> Any:
    """Construct a noise handle by name (the config-driven path)."""
    try:
        ctor = _REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown noise {name!r}; registered: {registered()}") from None
    return ctor(**kwargs)
