"""``rf.spaces`` — observation/action space descriptors a game advertises.

Query a game handle for them (``game.observation_space()`` / ``game.action_space()``) to size a value
network without hard-coding any game's dimensions::

    game = rf.games.Snake(grid_size=20)
    obs_shape, n_actions = game.observation_space().shape, game.action_space().n

``Box`` mirrors Gymnasium's: ``shape`` is the contract; ``low`` / ``high`` are numpy arrays broadcast
to ``shape``. ``Discrete`` is a choice from ``0..n``.
"""

from __future__ import annotations

from ._reinfors import Box, Discrete

__all__ = ["Box", "Discrete"]
