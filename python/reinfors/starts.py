"""Start-state generators for Arena openings."""

from __future__ import annotations

from typing import Any

from . import _reinfors
from .arena import _mix64


class RandomStartingMoves:
    """An opening of ``n_moves`` seeded-uniform legal moves.

    ``generate`` returns an ``EnvSnapshot`` at the post-opening position; Arena restores
    the same snapshot into both games of a pair (seats swapped), so both play the exact
    position and — in chance games — the same post-opening chance stream (duplicate
    format). Openings that end the game are resampled up to ``max_retries`` times.
    """

    def __init__(self, n_moves: int, max_retries: int = 32) -> None:
        if n_moves < 1:
            raise ValueError("n_moves must be >= 1")
        self.n_moves = n_moves
        self.max_retries = max_retries

    def generate(self, game: Any, reward: Any, seed: int) -> Any:
        for attempt in range(self.max_retries):
            env = _reinfors.Env(game, reward, seed=_mix64(seed, attempt))
            rng_state = _mix64(seed, attempt, 1)
            aborted = False
            for ply in range(self.n_moves):
                if env.done():
                    aborted = True
                    break
                agent = env.active_agents()[0]
                legal = env.legal_actions(agent)
                rng_state = _mix64(rng_state, ply)
                env.step({agent: legal[rng_state % len(legal)]})
            if not aborted and not env.done():
                return env.snapshot()
        raise ValueError(
            f"could not draw a live {self.n_moves}-move opening in "
            f"{self.max_retries} attempts — the game may be too short"
        )
