"""``rf.noise`` — root-exploration-noise handles for ``rf.policies.AlphaZero(noise=...)``.

``Dirichlet(epsilon, alpha, scope)`` is the classic AlphaZero root mix
``(1-epsilon)*P + epsilon*Dir(alpha)``, drawn from the seeded stream (collects stay reproducible).
``scope`` ("requester" | "both") picks which root priors are perturbed in a *simultaneous* search
tree. Pass ``noise=None`` to disable noise honestly (no ``epsilon=0`` sentinel); omit the kwarg
for the self-play default ``Dirichlet(0.25, 0.3, "requester")``.
"""

from __future__ import annotations

from . import _reinfors

Dirichlet = _reinfors.NoiseHandle.Dirichlet
