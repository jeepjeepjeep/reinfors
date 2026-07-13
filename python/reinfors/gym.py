"""``reinfors.gym`` — expose a reinfors game through the standard RL environment APIs.

Downstream algorithm libraries (Stable-Baselines3, CleanRL, RLlib, Tianshou, …) consume the
`Gymnasium <https://gymnasium.farama.org>`_ single-agent API and the `PettingZoo
<https://pettingzoo.farama.org>`_ multi-agent API. This module adapts reinfors' native ``rf.Env`` to
both, so a reinfors game drops into an existing pipeline unchanged::

    import reinfors as rf
    from reinfors import gym

    env = gym.make(rf.games.GridWorld(size=8), rf.Reward(goal=1.0))   # -> gymnasium.Env
    env = gym.make(rf.games.Snake(grid_size=10), rf.Reward(food=1.0)) # -> pettingzoo ParallelEnv

Single-agent games become a ``gymnasium.Env``; simultaneous multi-agent games (e.g. snake) become a
PettingZoo ``ParallelEnv``. Turn-based (sequential) multi-agent games are not yet exposed (a PettingZoo
AEC adapter is the planned follow-up).

The reward is supplied here rather than by ``rf.Env`` itself (which stays reward-free for play/eval):
the standard APIs must return a scalar reward from ``step``, and the game's event→reward mapping stays
in Rust — the adapter only reads back ``env.rewards``. reinfors' ``rf.Env`` has no truncation horizon of
its own, so the time limit is applied here (defaulting to the game's ``truncation_horizon()``), exactly
as Gymnasium's ``TimeLimit`` wrapper does.

``gymnasium`` / ``pettingzoo`` are optional dependencies (``pip install reinfors[gym]``); they are
imported lazily, so importing this module without them installed is fine until you call a constructor.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import _reinfors
from ._reinfors import Box, Discrete, Reward

__all__ = ["gymnasium_env", "make", "parallel_env"]


def _to_gym_box(space: Box) -> Any:
    import gymnasium  # pyright: ignore[reportMissingImports]  # optional dep (reinfors[gym])

    return gymnasium.spaces.Box(
        low=np.asarray(space.low, dtype=np.float32),
        high=np.asarray(space.high, dtype=np.float32),
        shape=space.shape,
        dtype=np.float32,
    )


def _to_gym_discrete(space: Discrete) -> Any:
    import gymnasium  # pyright: ignore[reportMissingImports]  # optional dep (reinfors[gym])

    return gymnasium.spaces.Discrete(space.n)


def _default_reward(reward: Reward | None) -> Reward:
    # An empty `Reward()` resolves to the game's default weights (see `resolve_reward` in Rust); the
    # standard APIs always need *some* scalar, unlike reward-free play/eval.
    return reward if reward is not None else Reward()


def _probe(game: Any) -> tuple[int, bool]:
    """(num_agents, is_simultaneous) — read off a throwaway reset so `make` can pick the right API."""
    env = _reinfors.Env(game)
    env.reset()
    n = env.num_agents()
    return n, len(env.active_agents()) == n


# The adapter classes subclass the optional bases, so they are defined lazily on first use and memoized.
_GYM_CLS: Any = None
_PARALLEL_CLS: Any = None


def _missing(pkg: str) -> ImportError:
    return ImportError(f"reinfors.gym needs `{pkg}`; install the adapter backends with `pip install reinfors[gym]`")


def _gym_cls() -> Any:
    global _GYM_CLS
    if _GYM_CLS is not None:
        return _GYM_CLS
    try:
        import gymnasium  # pyright: ignore[reportMissingImports]  # optional dep (reinfors[gym])
    except ModuleNotFoundError as e:
        raise _missing("gymnasium") from e

    class ReinforsGymEnv(gymnasium.Env):
        """A single-agent reinfors game as a ``gymnasium.Env``."""

        # `metadata` (with the same `{"render_modes": []}` default) is inherited from `gymnasium.Env`.

        def __init__(
            self,
            game: Any,
            reward: Reward | None = None,
            *,
            max_episode_steps: int | None = None,
            seed: int | None = None,
            render_mode: str | None = None,
        ) -> None:
            self._game = game
            self._reward = _default_reward(reward)
            self._env = _reinfors.Env(game, self._reward, seed=seed or 0)
            if self._env.num_agents() != 1:
                raise ValueError("gymnasium_env is single-agent only; use parallel_env for multi-agent")
            self.observation_space = _to_gym_box(game.observation_space())
            self.action_space = _to_gym_discrete(game.action_space())
            self._max_episode_steps = max_episode_steps if max_episode_steps is not None else game.truncation_horizon()
            self._elapsed = 0
            self.render_mode = render_mode

        def reset(
            self, *, seed: int | None = None, options: dict[str, Any] | None = None
        ) -> tuple[Any, dict[str, Any]]:
            super().reset(seed=seed)
            if seed is not None:  # rebuild so the episode is reproducible from `seed` (Gymnasium contract)
                self._env = _reinfors.Env(self._game, self._reward, seed=seed)
            else:
                self._env.reset()
            self._elapsed = 0
            return self._env.observe(0), {}

        def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
            self._env.step({0: int(action)})
            self._elapsed += 1
            rewards = self._env.rewards
            assert rewards is not None  # the adapter always builds the Env with a reward
            reward = float(rewards[0])
            terminated = self._env.done()
            truncated = (
                not terminated and self._max_episode_steps is not None and self._elapsed >= self._max_episode_steps
            )
            return self._env.observe(0), reward, terminated, truncated, {}

    _GYM_CLS = ReinforsGymEnv
    return _GYM_CLS


def _parallel_cls() -> Any:
    global _PARALLEL_CLS
    if _PARALLEL_CLS is not None:
        return _PARALLEL_CLS
    try:
        from pettingzoo import ParallelEnv  # pyright: ignore[reportMissingImports]  # optional dep (reinfors[gym])
    except ModuleNotFoundError as e:
        raise _missing("pettingzoo") from e

    class ReinforsParallelEnv(ParallelEnv):
        """A simultaneous multi-agent reinfors game as a PettingZoo ``ParallelEnv``."""

        metadata: dict[str, Any] = {"name": "reinfors", "render_modes": []}  # noqa: RUF012

        def __init__(
            self,
            game: Any,
            reward: Reward | None = None,
            *,
            max_episode_steps: int | None = None,
            seed: int | None = None,
            render_mode: str | None = None,
        ) -> None:
            self._game = game
            self._reward = _default_reward(reward)
            self._env = _reinfors.Env(game, self._reward, seed=seed or 0)
            n = self._env.num_agents()
            self.possible_agents = [f"player_{i}" for i in range(n)]
            self.agents: list[str] = []
            self._index = {name: i for i, name in enumerate(self.possible_agents)}
            self._obs_space = _to_gym_box(game.observation_space())
            self._act_space = _to_gym_discrete(game.action_space())
            self._max_episode_steps = max_episode_steps if max_episode_steps is not None else game.truncation_horizon()
            self._elapsed = 0
            self.render_mode = render_mode

        def observation_space(self, agent: str) -> Any:
            return self._obs_space

        def action_space(self, agent: str) -> Any:
            return self._act_space

        def reset(
            self, seed: int | None = None, options: dict[str, Any] | None = None
        ) -> tuple[dict[str, Any], dict[str, Any]]:
            if seed is not None:
                self._env = _reinfors.Env(self._game, self._reward, seed=seed)
            else:
                self._env.reset()
            self.agents = list(self.possible_agents)
            self._elapsed = 0
            obs = {a: self._env.observe(self._index[a]) for a in self.agents}
            return obs, {a: {} for a in self.agents}

        def step(
            self, actions: dict[str, Any]
        ) -> tuple[dict[str, Any], dict[str, float], dict[str, bool], dict[str, bool], dict[str, Any]]:
            joint = {self._index[a]: int(act) for a, act in actions.items()}
            self._env.step(joint)
            self._elapsed += 1
            rewards_vec = self._env.rewards
            assert rewards_vec is not None  # the adapter always builds the Env with a reward
            active_after = set(self._env.active_agents())
            episode_over = self._env.done()
            hit_limit = self._max_episode_steps is not None and self._elapsed >= self._max_episode_steps
            acted = list(self.agents)
            obs: dict[str, Any] = {}
            rewards: dict[str, float] = {}
            terminations: dict[str, bool] = {}
            truncations: dict[str, bool] = {}
            infos: dict[str, Any] = {}
            for a in acted:
                i = self._index[a]
                # An agent is done when the episode ends or it is no longer among the movers (it died).
                term = episode_over or i not in active_after
                obs[a] = self._env.observe(i)
                rewards[a] = float(rewards_vec[i])
                terminations[a] = term
                truncations[a] = hit_limit and not term
                infos[a] = {}
            self.agents = [a for a in acted if not (terminations[a] or truncations[a])]
            return obs, rewards, terminations, truncations, infos

    _PARALLEL_CLS = ReinforsParallelEnv
    return _PARALLEL_CLS


def gymnasium_env(game: Any, reward: Reward | None = None, **kwargs: Any) -> Any:
    """A single-agent reinfors game as a ``gymnasium.Env`` (raises for multi-agent games)."""
    return _gym_cls()(game, reward, **kwargs)


def parallel_env(game: Any, reward: Reward | None = None, **kwargs: Any) -> Any:
    """A simultaneous multi-agent reinfors game as a PettingZoo ``ParallelEnv``."""
    return _parallel_cls()(game, reward, **kwargs)


def make(game: Any, reward: Reward | None = None, **kwargs: Any) -> Any:
    """Adapt `game` to the standard API that fits it: ``gymnasium.Env`` for a single-agent game, a
    PettingZoo ``ParallelEnv`` for a simultaneous multi-agent one. Turn-based multi-agent games raise
    (a PettingZoo AEC adapter is the planned follow-up)."""
    n, simultaneous = _probe(game)
    if n == 1:
        return gymnasium_env(game, reward, **kwargs)
    if simultaneous:
        return parallel_env(game, reward, **kwargs)
    raise NotImplementedError(
        "turn-based (sequential) multi-agent games are not yet exposed via the standard API "
        "(a PettingZoo AEC adapter is planned); only single-agent (Gymnasium) and simultaneous "
        "multi-agent (PettingZoo Parallel) games are supported"
    )
