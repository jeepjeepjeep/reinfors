"""`engine.resolved_config()` — the fully resolved immutable composition. The load-bearing
property is the ROUND-TRIP: `rf.engine_from_config(engine.resolved_config())` reconstructs an
engine whose collected records are byte-identical, for every composition family."""

from __future__ import annotations

import json

import numpy as np
import pytest
import reinfors as rf


def _az_chess() -> rf.Engine:
    return rf.Engine(
        rf.games.Chess(max_ticks=60, encoder=rf.encoders.OpenSpielChess()),
        None,
        rf.policies.AlphaZero(num_simulations=8, chance=rf.chance_modes.Committed(samples=2)),
        rf.learners.AlphaZero(gamma=0.97),
        n_games=2,
        seed=3,
        infer_cache=1024,
    )


def test_defaults_are_resolved_and_json_compatible() -> None:
    cfg = _az_chess().resolved_config()
    json.dumps(cfg)  # JSON-compatible throughout
    assert cfg["schema_version"] == 1 and cfg["reinfors_version"]
    assert cfg["game"] == {
        "name": "chess",
        "max_ticks": 60,
        "encoder": {"name": "openspiel_chess"},
    }
    assert cfg["reward"] == {"win": 1.0, "loss": -1.0, "draw": 0.0}  # game defaults, resolved
    pol = cfg["policy"]
    assert pol["name"] == "alphazero" and pol["c_puct"] == 1.5  # ctor default surfaced
    assert pol["chance"] == {"name": "committed", "samples": 2}
    assert pol["noise"] == {"name": "dirichlet", "epsilon": 0.25, "alpha": 0.3, "scope": "requester"}
    assert cfg["learner"] == {"name": "alphazero", "gamma": 0.97}
    assert cfg["engine"]["n_games"] == 2 and cfg["engine"]["infer_cache"] == 1024


def test_schema_version_is_checked() -> None:
    cfg = _az_chess().resolved_config()
    cfg["schema_version"] = 2
    with pytest.raises(ValueError, match="schema_version"):
        rf.engine_from_config(cfg)


def test_disabled_start_buffer_renders_null_and_cannot_split_fingerprints() -> None:
    def build(cap: int, p: float) -> rf.Engine:
        return rf.Engine(
            rf.games.Connect4(),
            None,
            rf.policies.EpsilonGreedyQ(),
            rf.learners.Dqn(),
            n_games=1,
            start_buffer=False,
            start_buffer_capacity=cap,
            p_fresh=p,
        )

    a, b = build(1, 0.0), build(999, 0.9)  # ignored args must not reach the fingerprint
    assert a.resolved_config()["engine"]["start_buffer"] is None
    assert a.config_fingerprint() == b.config_fingerprint()


def test_noise_off_renders_null() -> None:
    e = rf.Engine(
        rf.games.Connect4(),
        None,
        rf.policies.AlphaZero(num_simulations=4, noise=None),
        rf.learners.AlphaZero(gamma=1.0),
        n_games=1,
    )
    assert e.resolved_config()["policy"]["noise"] is None


def test_fingerprint_separates_configs_and_is_stable() -> None:
    a, b = _az_chess(), _az_chess()
    assert a.config_fingerprint() == b.config_fingerprint()
    assert len(a.config_fingerprint()) == 64  # SHA-256 hex
    c = rf.Engine(
        rf.games.Chess(max_ticks=60, encoder=rf.encoders.MinimalChess()),  # encoder differs
        None,
        rf.policies.AlphaZero(num_simulations=8, chance=rf.chance_modes.Committed(samples=2)),
        rf.learners.AlphaZero(gamma=0.97),
        n_games=2,
        seed=3,
        infer_cache=1024,
    )
    assert c.config_fingerprint() != a.config_fingerprint()


@pytest.mark.parametrize("family", ["az", "dqn", "treestrap"])
def test_round_trip_reconstructs_a_record_identical_engine(family: str) -> None:
    def build() -> rf.Engine:
        if family == "az":
            return _az_chess()
        if family == "dqn":
            return rf.Engine(
                rf.games.Backgammon(max_ticks=40),
                rf.Reward(win=2.0),
                rf.policies.EpsilonGreedyQ(n_heads=2, epsilon=0.1),
                rf.learners.Dqn(bootstrap_p=0.5),
                n_games=2,
                seed=7,
            )
        return rf.Engine(
            rf.games.Snake(grid_size=6, initial_length=2, food=2, max_ticks=30),
            rf.Reward(food=1.0, loss=-5.0),
            rf.policies.SelectiveExpectimax(expansion_budget=6, n_heads=2, opponent="distributional"),
            rf.learners.TreeStrap(gamma=0.9, outcome_weight=0.3),
            n_games=2,
            seed=1,
            start_buffer=True,
            start_buffer_capacity=50,
            p_fresh=0.2,
        )

    original = build()
    rebuilt = rf.engine_from_config(original.resolved_config())
    assert rebuilt.resolved_config() == original.resolved_config()
    assert rebuilt.config_fingerprint() == original.config_fingerprint()

    def _norm(v: object) -> object:
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, (list, tuple)):
            return [_norm(x) for x in v]
        return v

    def collect(e: rf.Engine) -> object:
        a = e.resolved_config()["game"]["name"]
        ka = {"backgammon": (2, 1352), "snake": (2, 3)}.get(a)

        def infer(obs: np.ndarray, *rest: object) -> object:
            rows = obs.shape[0]
            if a == "chess":
                return np.zeros((rows, 4672)), np.zeros(rows)
            assert ka is not None
            return np.full((rows, *ka), 0.25)

        b = e.collect(40, infer)
        # EVERY learner-specific array plus deterministic telemetry — not just obs. Timing
        # fields (seconds/rates) are the only exclusion.
        arrays = {
            k: np.ascontiguousarray(getattr(b, k)).tobytes()
            for k in dir(b)
            if isinstance(getattr(b, k, None), np.ndarray)
        }
        assert len(arrays) >= 3, sorted(arrays)  # the batch really exposes its arrays
        telemetry = {
            k: _norm(v) for k, v in b.telemetry.items() if "seconds" not in k and "time" not in k and "per_s" not in k
        }
        return (arrays, telemetry)

    assert collect(original) == collect(rebuilt)
