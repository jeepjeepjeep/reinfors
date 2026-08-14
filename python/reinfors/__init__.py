"""reinfors — general-purpose, search-first RL engine (Rust core, Python API).

Compose a rollout `Engine` from a game, a reward, a policy, and a learner. The reward is decoupled
from the game (the game owns only the rules): the same game trains under any reward, and `rf.Env` —
which only plays/evaluates — needs no reward at all.

    import reinfors as rf
    game    = rf.games.Snake(grid_size=20, max_ticks=750)
    reward  = rf.Reward(food=1.0, loss=-10.0)
    policy  = rf.policies.SelectiveExpectimax(n_heads=10, expansion_budget=64)
    learner = rf.learners.TreeStrap(gamma=0.99, outcome_weight=0.3)
    engine  = rf.Engine(game, reward, policy, learner, n_games=16)
    batch   = engine.collect(2048, infer)

The `make_*` / `registered_*` functions and `engine_from_config` are the name-addressable,
config-driven equivalents (e.g. for YAML-driven training).
"""

from __future__ import annotations

from typing import Any

from . import (
    _reinfors,
    arena,
    catalog,
    chance_modes,
    encoders,
    games,
    gym,
    learners,
    noise,
    policies,
    solvers,
    spaces,
    starts,
)
from ._reinfors import (
    AlphaZeroBatch,
    CollectStream,
    DeepCfrBatch,
    DqnBatch,
    Engine,
    EngineSnapshot,
    Env,
    EnvSnapshot,
    Reward,
    TreeStrapBatch,
    build_info,
    chess_action_uci,
    chess_uci_action,
    core_version,
)
from .arena import Arena


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

        {"game": {"name": "snake", "grid_size": 20, "max_ticks": 750, "reward": {"food": 1.0, "loss": -10.0}},
         "policy": {"name": "selective_expectimax", "n_heads": 10, ...},
         "learner": {"name": "treestrap", "gamma": 0.99, ...},
         "engine": {"n_games": 16, "seed": 0}}

    Each block's `name` selects the handle; the remaining keys are its constructor kwargs. The reward
    and `max_ticks` (the truncation horizon) are both game-side now, but for YAML friendliness a
    `reward` mapping and a `max_ticks` given in the `engine` block are still accepted and routed to the
    right place — so a config parsed straight from YAML (`yaml.safe_load(...)`) works as-is.
    """

    def _split(block: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        return block["name"], {k: v for k, v in block.items() if k != "name"}

    def _coerce_handles(kw: dict[str, Any]) -> dict[str, Any]:
        # Nested handle blocks (as `resolved_config` renders them) become typed handles, so the
        # round-trip `engine_from_config(engine.resolved_config())` holds for every composition.
        if isinstance(kw.get("encoder"), dict):
            e = dict(kw["encoder"])
            kw["encoder"] = encoders.make(e.pop("name"), **e)
        if isinstance(kw.get("chance"), dict):
            c = dict(kw["chance"])
            kw["chance"] = chance_modes.make(c.pop("name"), **c)
        if isinstance(kw.get("noise"), dict):
            n = dict(kw["noise"])
            noise_name = n.pop("name", None)
            if noise_name is None:
                raise ValueError("noise block requires a name")
            kw["noise"] = noise.make(noise_name, **n)
        return kw

    schema = config.get("schema_version")
    if schema is not None and (not isinstance(schema, int) or isinstance(schema, bool) or schema != 1):
        msg = f"unsupported config schema_version {schema!r}; this reinfors supports 1"
        raise ValueError(msg)
    g_name, g_kw = _split(config["game"])
    g_kw = _coerce_handles(g_kw)
    engine_kw = dict(config.get("engine", {}))
    # start_buffer renders as null (off) or a {capacity, p_fresh} block (on) — unpack it back
    # into the flat constructor kwargs (a plain bool is also accepted, for hand-written configs).
    if "start_buffer" in engine_kw and not isinstance(engine_kw["start_buffer"], bool):
        sb = engine_kw.pop("start_buffer")
        engine_kw["start_buffer"] = sb is not None
        if sb is not None:
            engine_kw["start_buffer_capacity"] = sb["capacity"]
            engine_kw["p_fresh"] = sb["p_fresh"]
    # The reward rides with the game block (YAML-friendly) or its own block; it goes to the Engine now.
    # Fall back on None only: a falsy malformed value must reach type validation, not vanish.
    reward_cfg = g_kw.pop("reward", None)
    if reward_cfg is None:
        reward_cfg = config.get("reward")
    reward = Reward(**reward_cfg) if isinstance(reward_cfg, dict) else reward_cfg
    # `max_ticks` is the game's truncation horizon; accept it in the engine block (legacy) and route it.
    if "max_ticks" in engine_kw:
        g_kw.setdefault("max_ticks", engine_kw.pop("max_ticks"))
    p_name, p_kw = _split(config["policy"])
    p_kw = _coerce_handles(p_kw)
    l_name, l_kw = _split(config["learner"])
    return Engine(
        make_game(g_name, **g_kw),
        reward,
        make_policy(p_name, **p_kw),
        make_learner(l_name, **l_kw),
        **engine_kw,
    )


__all__ = [
    "AlphaZeroBatch",
    "Arena",
    "CollectStream",
    "DeepCfrBatch",
    "DqnBatch",
    "Engine",
    "EngineSnapshot",
    "Env",
    "EnvSnapshot",
    "Reward",
    "TreeStrapBatch",
    "_reinfors",
    "arena",
    "build_info",
    "catalog",
    "chance_modes",
    "chess_action_uci",
    "chess_uci_action",
    "core_version",
    "encoders",
    "engine_from_config",
    "games",
    "gym",
    "learners",
    "make_game",
    "make_learner",
    "make_policy",
    "noise",
    "policies",
    "registered_games",
    "registered_learners",
    "registered_policies",
    "solvers",
    "spaces",
    "starts",
]
__version__ = core_version()
