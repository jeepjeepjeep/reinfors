"""reinfors — general-purpose, search-first RL engine (Rust core, Python API).

Compose a rollout `Engine` from a game, a reward, a policy, and a learner. The reward is decoupled
from the game (the game owns only the rules): the same game trains under any reward, and `rf.Env` —
which only plays/evaluates — needs no reward at all.

    import reinfors as rf
    game    = rf.games.Snake(grid_size=20)
    reward  = rf.Reward(food=1.0, loss=-10.0)
    policy  = rf.policies.SelectiveExpectimax(n_heads=10, expansion_budget=64)
    learner = rf.learners.TreeStrap(gamma=0.99, outcome_weight=0.3)
    engine  = rf.Engine(game, reward, policy, learner, n_games=16, max_ticks=750)
    batch   = engine.collect(2048, infer)

The `make_*` / `registered_*` functions and `engine_from_config` are the name-addressable,
config-driven equivalents (e.g. for YAML-driven training).
"""

from __future__ import annotations

from typing import Any

from . import _reinfors, games, learners, policies, spaces
from ._reinfors import Engine, Env, Reward


def make_game(name: str, **kwargs: Any) -> Any:
    """Construct a game handle by name (see `registered_games`)."""
    return games.make(name, **kwargs)


def make_policy(name: str, **kwargs: Any) -> Any:
    """Construct a policy handle by name (see `registered_policies`)."""
    return policies.make(name, **kwargs)


def make_learner(name: str, **kwargs: Any) -> Any:
    """Construct a learner handle by name (see `registered_learners`)."""
    return learners.make(name, **kwargs)


def registered_games() -> list[str]:
    return games.registered()


def registered_policies() -> list[str]:
    return policies.registered()


def registered_learners() -> list[str]:
    return learners.registered()


def engine_from_config(config: dict[str, Any]) -> Engine:
    """Build an `Engine` from a nested config block, e.g.::

        {"game": {"name": "snake", "grid_size": 20, "reward": {"food": 1.0, "loss": -10.0}},
         "policy": {"name": "selective_expectimax", "n_heads": 10, ...},
         "learner": {"name": "treestrap", "gamma": 0.99, ...},
         "engine": {"n_games": 16, "max_ticks": 750, "seed": 0}}

    Each block's `name` selects the handle; the remaining keys are its constructor kwargs. The reward
    is decoupled from the game in the API, but a `reward` mapping nested under `game` (or given as its
    own top-level block) is still accepted and wrapped into an `rf.Reward` — so a config parsed straight
    from YAML (`yaml.safe_load(...)`) works as-is — and routed to the `Engine`.
    """

    def _split(block: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        return block["name"], {k: v for k, v in block.items() if k != "name"}

    g_name, g_kw = _split(config["game"])
    # The reward rides with the game block (YAML-friendly) or its own block; it goes to the Engine now.
    reward_cfg = g_kw.pop("reward", None) or config.get("reward")
    reward = Reward(**reward_cfg) if isinstance(reward_cfg, dict) else reward_cfg
    p_name, p_kw = _split(config["policy"])
    l_name, l_kw = _split(config["learner"])
    return Engine(
        make_game(g_name, **g_kw),
        reward,
        make_policy(p_name, **p_kw),
        make_learner(l_name, **l_kw),
        **config.get("engine", {}),
    )


__all__ = [
    "Engine",
    "Env",
    "Reward",
    "_reinfors",
    "engine_from_config",
    "games",
    "learners",
    "make_game",
    "make_learner",
    "make_policy",
    "policies",
    "registered_games",
    "registered_learners",
    "registered_policies",
    "spaces",
]
__version__ = "0.0.0"
