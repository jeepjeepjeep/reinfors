"""Game registry: maps a game name to its compiled rollout `Engine` class and shape metadata.

Lets callers pick a game by name, size a `BootstrappedQNetwork` from its observation shape and action
count, and construct the matching engine — without hard-coding any game's dimensions. Snake and
Connect-4 have fixed observation shapes; GridWorld's depends on its `size`, so its shape is a callable.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from . import _reinfors

ObsShape = tuple[int, int, int]  # (C, H, W)


@dataclass(frozen=True)
class GameSpec:
    """One registered game: its rollout `Engine` class, action count, and observation shape.

    `obs_shape` is either a fixed `(C, H, W)` tuple or, for size-parameterized games, a callable
    `(**kwargs) -> (C, H, W)` reading the same keyword args its engine takes (e.g. GridWorld's `size`).
    """

    engine: type
    action_count: int
    obs_shape: ObsShape | Callable[..., ObsShape]

    def resolve_obs_shape(self, **kwargs: object) -> ObsShape:
        """The concrete `(C, H, W)` for this game; pass the engine's size kwargs for variable shapes."""
        if callable(self.obs_shape):
            return self.obs_shape(**kwargs)
        return self.obs_shape


def _gridworld_obs_shape(*, size: int, **_: object) -> ObsShape:
    return (2, size, size)


REGISTRY: dict[str, GameSpec] = {
    "snake": GameSpec(
        engine=_reinfors.Engine, action_count=3, obs_shape=lambda *, grid_size, **_: (5, grid_size, grid_size)
    ),
    "connect4": GameSpec(engine=_reinfors.Connect4Engine, action_count=7, obs_shape=(2, 6, 7)),
    "gridworld": GameSpec(engine=_reinfors.GridWorldEngine, action_count=4, obs_shape=_gridworld_obs_shape),
}


def get(name: str) -> GameSpec:
    """The `GameSpec` for `name`, or a `KeyError` listing the registered games."""
    try:
        return REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown game {name!r}; registered: {sorted(REGISTRY)}") from None


def net_shape(name: str, **kwargs: object) -> tuple[ObsShape, int]:
    """The `(obs_shape, action_count)` for `name`, to size a `BootstrappedQNetwork`. Pass any size
    kwargs the game's observation shape depends on (e.g. `grid_size=12` for snake, `size=5` for
    gridworld); fixed-shape games (connect4) ignore them."""
    spec = get(name)
    return spec.resolve_obs_shape(**kwargs), spec.action_count
