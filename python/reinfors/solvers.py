"""``rf.solvers`` — offline equilibrium solvers.

A solver owns its own traversal of the game (no engine, no network) and produces a strategy
artifact. ``Cfr`` covers counterfactual regret minimization (vanilla / CFR+ / external-sampling
MCCFR) over compatible sequential games with declared chance and information-state keys. Query
the result with ``average_strategy(env.information_state_key(agent))``; exact metrics are available
when the tree fits the enumeration cap. The generated compatibility catalogue is the canonical list
of built-in compositions.
"""

from __future__ import annotations

from . import _reinfors

Cfr = _reinfors.Cfr
DeepCfr = _reinfors.DeepCfr

__all__ = ["Cfr", "DeepCfr"]
