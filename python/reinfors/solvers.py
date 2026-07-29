"""``rf.solvers`` — offline equilibrium solvers.

A solver owns its own traversal of the game (no engine, no network) and produces a strategy
artifact. ``Cfr`` covers counterfactual regret minimization (vanilla / CFR+ / external-sampling
MCCFR) over 2-player games with fully declared chance and information-state keys — the poker
family. Query the result with ``average_strategy(env.information_state_key(agent))``; measure
convergence with ``exploitability()`` (exact, Kuhn/Leduc-sized games).
"""

from __future__ import annotations

from . import _reinfors

Cfr = _reinfors.Cfr
DeepCfr = _reinfors.DeepCfr

__all__ = ["Cfr", "DeepCfr"]
