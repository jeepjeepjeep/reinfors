"""``reinfors.gym`` — expose a reinfors game through the standard RL environment APIs.

Downstream algorithm libraries (Stable-Baselines3, CleanRL, RLlib, Tianshou, …) consume the
`Gymnasium <https://gymnasium.farama.org>`_ single-agent API and the `PettingZoo
<https://pettingzoo.farama.org>`_ multi-agent API. This module adapts reinfors' native ``rf.Env`` to
both, so a reinfors game drops into an existing pipeline unchanged::

    import reinfors as rf
    from reinfors import gym

    env = gym.make(game=rf.games.GridWorld(size=8), reward=rf.Reward(goal=1.0))   # -> gymnasium.Env
    env = gym.make(game=rf.games.Snake(grid_size=10), reward=rf.Reward(food=1.0)) # -> ParallelEnv

Single-agent games become a ``gymnasium.Env``; simultaneous multi-agent games (e.g. snake) become a
PettingZoo ``ParallelEnv``; turn-based (sequential) multi-agent games (connect4, chess, backgammon)
become a PettingZoo ``AECEnv``. Legality follows the ecosystem's conventions: AEC observations are
``{"observation": ..., "action_mask": ...}`` dicts (as in PettingZoo's own classic games, and how
``pettingzoo.test.api_test`` samples legally), and the Parallel adapter carries the mask in
``infos[agent]["action_mask"]`` so its observations stay plain arrays.

The reward is supplied to the adapter because the standard APIs must return a scalar reward from
``step``. The game's event→reward mapping stays in Rust: the adapter builds its native ``rf.Env`` with
that mapping and only reads back ``env.rewards``. reinfors' ``rf.Env`` has no truncation horizon of its
own, so the time limit is applied here (defaulting to the game's ``truncation_horizon()``), exactly as
Gymnasium's ``TimeLimit`` wrapper does.

``gymnasium`` / ``pettingzoo`` are optional dependencies (``pip install reinfors[gym]``); they are
imported lazily, so importing this module without them installed is fine until you call a constructor.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import _reinfors
from ._reinfors import Box, Discrete, Reward

__all__ = ["aec_env", "gymnasium_env", "make", "parallel_env"]


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


def _mask(legal: list[int], n: int) -> np.ndarray[Any, np.dtype[np.int8]]:
    # int8 is the dtype `gymnasium.spaces.Discrete.sample(mask)` requires.
    m = np.zeros(n, dtype=np.int8)
    m[legal] = 1
    return m


# The adapter classes subclass the optional bases, so they are defined lazily on first use and memoized.
_GYM_CLS: Any = None
_PARALLEL_CLS: Any = None
_AEC_CLS: Any = None


def _episode_limit(max_episode_steps: int | None, game: Any) -> int | None:
    if max_episode_steps is None:
        horizon: int | None = game.truncation_horizon()
        return horizon
    if not isinstance(max_episode_steps, int) or isinstance(max_episode_steps, bool):
        raise ValueError(f"max_episode_steps must be an int, got {max_episode_steps!r}")
    if max_episode_steps < 1:
        raise ValueError("max_episode_steps must be >= 1")
    return max_episode_steps


def _missing(pkg: str) -> ImportError:
    return ImportError(f"reinfors.gym needs `{pkg}`; install the adapter backends with `pip install reinfors[gym]`")


# car_racing is withheld until the HWC-u8 frame presentation lands: Gymnasium's CarRacing
# contract is uint8 HWC frames, and silently handing CHW floats to gym-ecosystem pipelines
# would break them without an error. Track: FrameProvider follow-up.
_WITHHELD_FROM_ADAPTERS = frozenset({"car_racing"})


def _check_adapter_supported(game: Any) -> None:
    name = game.name
    if name in _WITHHELD_FROM_ADAPTERS:
        raise ValueError(
            f"{name} is not available through the standard-API adapters yet; "
            "use rf.Env directly (see the catalogue entry for why)"
        )


def _gym_cls() -> Any:
    global _GYM_CLS
    if _GYM_CLS is not None:
        return _GYM_CLS
    try:
        import gymnasium  # pyright: ignore[reportMissingImports]  # optional dep (reinfors[gym])
    except ModuleNotFoundError as e:
        raise _missing("gymnasium") from e

    class ReinforsGymEnv(gymnasium.Env["np.ndarray[Any, np.dtype[np.float32]]", int]):
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
            _check_adapter_supported(game)
            self._game = game
            self._reward = _default_reward(reward)
            self._env = _reinfors.Env(game, self._reward, seed=seed or 0)
            if self._env.num_agents() != 1:
                raise ValueError("gymnasium_env is single-agent only; use parallel_env for multi-agent")
            self.observation_space = _to_gym_box(game.observation_space())
            self.action_space = _to_gym_discrete(game.action_space())
            self._max_episode_steps = _episode_limit(max_episode_steps, game)
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
            _check_adapter_supported(game)
            self._game = game
            self._reward = _default_reward(reward)
            self._env = _reinfors.Env(game, self._reward, seed=seed or 0)
            n = self._env.num_agents()
            self.possible_agents = [f"player_{i}" for i in range(n)]
            self.agents: list[str] = []
            self._index = {name: i for i, name in enumerate(self.possible_agents)}
            self._obs_space = _to_gym_box(game.observation_space())
            self._act_space = _to_gym_discrete(game.action_space())
            self._max_episode_steps = _episode_limit(max_episode_steps, game)
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
            n = self._act_space.n
            obs = {a: self._env.observe(self._index[a]) for a in self.agents}
            infos = {a: {"action_mask": _mask(self._env.legal_actions(self._index[a]), n)} for a in self.agents}
            return obs, infos

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
            n = self._act_space.n
            for a in acted:
                i = self._index[a]
                # An agent is done when the episode ends or it is no longer among the movers (it died).
                term = episode_over or i not in active_after
                obs[a] = self._env.observe(i)
                rewards[a] = float(rewards_vec[i])
                terminations[a] = term
                truncations[a] = hit_limit and not term
                done = terminations[a] or truncations[a]
                infos[a] = {"action_mask": _mask([] if done else self._env.legal_actions(i), n)}
            self.agents = [a for a in acted if not (terminations[a] or truncations[a])]
            return obs, rewards, terminations, truncations, infos

    _PARALLEL_CLS = ReinforsParallelEnv
    return _PARALLEL_CLS


def _aec_cls() -> Any:
    global _AEC_CLS
    if _AEC_CLS is not None:
        return _AEC_CLS
    try:
        import gymnasium  # pyright: ignore[reportMissingImports]  # optional dep (reinfors[gym])
        from pettingzoo import AECEnv  # pyright: ignore[reportMissingImports]  # optional dep (reinfors[gym])
    except ModuleNotFoundError as e:
        raise _missing("pettingzoo") from e

    class ReinforsAecEnv(AECEnv):
        """A turn-based (sequential) multi-agent reinfors game as a PettingZoo ``AECEnv``.

        Observations follow the classic-games convention: ``{"observation": float32 array,
        "action_mask": int8 array}``, the mask naming the mover's legal actions (all-zero for a
        non-mover), so downstream code samples legally without knowing the game — ``rf.Env``
        rejects illegal actions at the boundary.
        """

        metadata: dict[str, Any] = {"name": "reinfors", "render_modes": [], "is_parallelizable": False}  # noqa: RUF012

        def __init__(
            self,
            game: Any,
            reward: Reward | None = None,
            *,
            max_episode_steps: int | None = None,
            seed: int | None = None,
            render_mode: str | None = None,
        ) -> None:
            super().__init__()
            _check_adapter_supported(game)
            self._game = game
            self._reward = _default_reward(reward)
            self._env = _reinfors.Env(game, self._reward, seed=seed or 0)
            n = self._env.num_agents()
            self.possible_agents = [f"player_{i}" for i in range(n)]
            self.agents: list[str] = []
            self._index = {name: i for i, name in enumerate(self.possible_agents)}
            self._act_space = _to_gym_discrete(game.action_space())
            self._obs_space = gymnasium.spaces.Dict(
                {
                    "observation": _to_gym_box(game.observation_space()),
                    "action_mask": gymnasium.spaces.Box(low=0, high=1, shape=(self._act_space.n,), dtype=np.int8),
                }
            )
            self._max_episode_steps = _episode_limit(max_episode_steps, game)
            self._elapsed = 0
            self.render_mode = render_mode

        def observation_space(self, agent: str) -> Any:
            return self._obs_space

        def action_space(self, agent: str) -> Any:
            return self._act_space

        def observe(self, agent: str) -> dict[str, Any]:
            i = self._index[agent]
            mover = (
                agent in self.agents
                and agent == self.agent_selection
                and not (self.terminations[agent] or self.truncations[agent])
            )
            legal = self._env.legal_actions(i) if mover else []
            return {"observation": self._env.observe(i), "action_mask": _mask(legal, self._act_space.n)}

        def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> None:
            if seed is not None:
                self._env = _reinfors.Env(self._game, self._reward, seed=seed)
            else:
                self._env.reset()
            self.agents = list(self.possible_agents)
            self._elapsed = 0
            self.rewards = dict.fromkeys(self.agents, 0.0)
            self._cumulative_rewards = dict.fromkeys(self.agents, 0.0)
            self.terminations = dict.fromkeys(self.agents, False)
            self.truncations = dict.fromkeys(self.agents, False)
            self.infos: dict[str, Any] = {a: {} for a in self.agents}
            self.agent_selection = self.possible_agents[self._env.active_agents()[0]]

        def step(self, action: Any) -> None:
            agent = self.agent_selection
            if self.terminations[agent] or self.truncations[agent]:
                self._was_dead_step(action)
                return
            # `last()` just reported this agent's accumulated reward; restart its accumulation.
            self._cumulative_rewards[agent] = 0.0
            self._env.step({self._index[agent]: int(action)})
            self._elapsed += 1
            rewards_vec = self._env.rewards
            assert rewards_vec is not None  # the adapter always builds the Env with a reward
            self.rewards = {a: float(rewards_vec[self._index[a]]) for a in self.agents}
            terminated = self._env.done()
            truncated = (
                not terminated and self._max_episode_steps is not None and self._elapsed >= self._max_episode_steps
            )
            if terminated or truncated:
                self.terminations = dict.fromkeys(self.agents, terminated)
                self.truncations = dict.fromkeys(self.agents, truncated)
                # Every agent still gets a `last()` look at its final reward before its dead step;
                # start that round on the next agent so the non-mover sees its outcome first.
                self.agent_selection = self.agents[(self.agents.index(agent) + 1) % len(self.agents)]
            else:
                self.agent_selection = self.possible_agents[self._env.active_agents()[0]]
            self._accumulate_rewards()

        def render(self) -> None:
            return None

        def close(self) -> None:
            return None

    _AEC_CLS = ReinforsAecEnv
    return _AEC_CLS


def gymnasium_env(game: Any, reward: Reward | None = None, **kwargs: Any) -> Any:
    """A single-agent reinfors game as a ``gymnasium.Env`` (raises for multi-agent games)."""
    return _gym_cls()(game, reward, **kwargs)


def parallel_env(game: Any, reward: Reward | None = None, **kwargs: Any) -> Any:
    """A simultaneous multi-agent reinfors game as a PettingZoo ``ParallelEnv``."""
    return _parallel_cls()(game, reward, **kwargs)


def aec_env(game: Any, reward: Reward | None = None, **kwargs: Any) -> Any:
    """A turn-based (sequential) multi-agent reinfors game as a PettingZoo ``AECEnv``."""
    return _aec_cls()(game, reward, **kwargs)


def make(game: Any, reward: Reward | None = None, **kwargs: Any) -> Any:
    """Adapt `game` to the standard API that fits it: ``gymnasium.Env`` for a single-agent game, a
    PettingZoo ``ParallelEnv`` for a simultaneous multi-agent one, a PettingZoo ``AECEnv`` for a
    turn-based (sequential) multi-agent one."""
    n, simultaneous = _probe(game)
    if n == 1:
        return gymnasium_env(game, reward, **kwargs)
    if simultaneous:
        return parallel_env(game, reward, **kwargs)
    return aec_env(game, reward, **kwargs)
