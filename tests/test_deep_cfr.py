"""The `rf.solvers.DeepCfr` surface: batch shapes and CSR consistency, the polymorphic infer
argument, determinism, callback error propagation, the exact exploitability instrument, and a
numpy table-emulated convergence run (the no-torch stand-in for real networks; the torch
reference lives in examples/train_deep_cfr.py).
"""

from collections.abc import Callable

import numpy as np
import pytest
import reinfors as rf


def zeros2(obs: np.ndarray) -> np.ndarray:
    return np.zeros((obs.shape[0], 2))


def test_batches_are_engine_shaped_and_csr_consistent() -> None:
    solver = rf.solvers.DeepCfr(rf.games.KuhnPoker(), seed=0)
    solver.next_iteration()
    batch = solver.collect(player=0, traversals=32, infer=zeros2)
    m = batch.advantage_obs.shape[0]
    assert batch.advantage_obs.shape == (m, 6)
    assert batch.advantage_iterations.shape == (m,)
    assert batch.advantage_legal_offsets.shape == (m + 1,)
    assert batch.advantage_legal_offsets[-1] == len(batch.advantage_legal_ids)
    assert len(batch.advantage_targets) == len(batch.advantage_legal_ids)
    assert (batch.advantage_iterations == 1).all()
    n = batch.strategy_obs.shape[0]
    assert batch.strategy_legal_offsets.shape == (n + 1,)
    assert (batch.strategy_players == 1).all(), "player 0's pass samples opponent 1"
    # Probabilities per strategy row sum to one.
    counts = np.diff(batch.strategy_legal_offsets)
    sums = np.add.reduceat(batch.strategy_probs, batch.strategy_legal_offsets[:-1])
    assert np.allclose(sums[counts > 0], 1.0)
    for key in ("traversals", "infer_calls", "cache_hits", "advantage_samples"):
        assert key in batch.telemetry


def test_infer_accepts_a_callable_or_a_per_player_sequence() -> None:
    solver = rf.solvers.DeepCfr(rf.games.KuhnPoker(), seed=3)
    solver.next_iteration()
    a = solver.collect(player=0, traversals=16, infer=zeros2)
    solver2 = rf.solvers.DeepCfr(rf.games.KuhnPoker(), seed=3)
    solver2.next_iteration()
    b = solver2.collect(player=0, traversals=16, infer=[zeros2, zeros2])
    assert np.array_equal(a.advantage_obs, b.advantage_obs), "shared net == identical list"
    assert np.array_equal(a.advantage_targets, b.advantage_targets)
    with pytest.raises(ValueError, match="2 per-player"):
        solver.collect(player=0, traversals=1, infer=[zeros2, zeros2, zeros2])
    with pytest.raises(TypeError, match="callable or a sequence"):
        solver.collect(player=0, traversals=1, infer=42)


def test_collects_are_deterministic_per_seed() -> None:
    def run(seed: int) -> np.ndarray:
        solver = rf.solvers.DeepCfr(rf.games.KuhnPoker(), seed=seed)
        solver.next_iteration()
        return solver.collect(player=0, traversals=32, infer=zeros2).advantage_obs

    assert np.array_equal(run(5), run(5))
    assert not np.array_equal(run(5), run(6))


def test_callback_errors_and_bad_shapes_propagate() -> None:
    solver = rf.solvers.DeepCfr(rf.games.KuhnPoker(), seed=0)
    solver.next_iteration()

    def boom(obs: np.ndarray) -> np.ndarray:
        raise RuntimeError("net exploded")

    with pytest.raises(RuntimeError, match="net exploded"):
        solver.collect(player=0, traversals=4, infer=boom)
    with pytest.raises(ValueError, match="one row of 2 advantages"):
        solver.collect(player=0, traversals=4, infer=lambda obs: np.zeros((obs.shape[0], 5)))
    with pytest.raises(ValueError, match="finite"):
        solver.collect(player=0, traversals=4, infer=lambda obs: np.full((obs.shape[0], 2), np.nan))
    with pytest.raises(TypeError) as error:
        solver.collect(
            player=0,
            traversals=4,
            infer=lambda obs: np.zeros((obs.shape[0], 2), dtype=np.int32),
        )
    assert "float64 or float32 ndarray" in str(error.value)
    assert "dtype int32" in str(error.value)
    with pytest.raises(ValueError, match="player must be below 2"):
        solver.collect(player=2, traversals=1, infer=zeros2)
    fresh = rf.solvers.DeepCfr(rf.games.KuhnPoker(), seed=0)
    with pytest.raises(ValueError, match="next_iteration"):
        fresh.collect(player=0, traversals=1, infer=zeros2)


def test_transposed_infer_shapes_are_rejected() -> None:
    # Kuhn's action count is 2, so a transposed (2, rows) return has the RIGHT element count
    # and would silently be flattened into garbage without the exact-shape check.
    solver = rf.solvers.DeepCfr(rf.games.KuhnPoker(), seed=0)
    solver.next_iteration()
    with pytest.raises(ValueError, match=r"expected .* one row"):
        solver.collect(player=0, traversals=8, infer=lambda obs: np.zeros((2, obs.shape[0])))
    with pytest.raises(ValueError, match="policy_infer returned shape"):
        solver.exploitability(lambda obs: np.full((2, obs.shape[0]), 0.5))
    with pytest.raises(TypeError, match=r"float64 or float32 ndarray.*dtype int32"):
        solver.exploitability(lambda obs: np.ones((obs.shape[0], 2), dtype=np.int32))


def test_failed_collects_are_transactional() -> None:
    # A raised callback must not consume the sampling sequence: retrying after the error
    # yields exactly what a fresh solver produces from the same seed and iteration.
    def run(fail_first: bool) -> np.ndarray:
        solver = rf.solvers.DeepCfr(rf.games.KuhnPoker(), seed=21)
        solver.next_iteration()
        if fail_first:
            calls = {"n": 0}

            def boom(obs: np.ndarray) -> np.ndarray:
                calls["n"] += 1
                if calls["n"] >= 2:
                    raise RuntimeError("mid-call failure")
                return np.zeros((obs.shape[0], 2))

            with pytest.raises(RuntimeError):
                solver.collect(player=0, traversals=16, infer=boom)
        return solver.collect(player=0, traversals=16, infer=zeros2).advantage_obs

    assert np.array_equal(run(False), run(True))


def test_construction_gates() -> None:
    with pytest.raises(ValueError, match=r"catalogue/compatibility"):
        rf.solvers.DeepCfr(rf.games.Connect4())
    # N-player hold'em now constructs (no Nash guarantee past 2 — documented at the gate).
    rf.solvers.DeepCfr(rf.games.TexasHoldem(num_players=3))


def test_exploitability_pins_the_uniform_values() -> None:
    kuhn = rf.solvers.DeepCfr(rf.games.KuhnPoker(), seed=0)
    e = kuhn.exploitability(lambda obs: np.full((obs.shape[0], 2), 0.5))
    assert abs(e - 11 / 24) < 1e-12
    leduc = rf.solvers.DeepCfr(rf.games.LeducPoker(), seed=0)
    e = leduc.exploitability(lambda obs: np.full((obs.shape[0], 3), 1 / 3))
    assert abs(e - 2.373611111111111) < 1e-9


def test_resolved_config_names_the_composition() -> None:
    solver = rf.solvers.DeepCfr(rf.games.LeducPoker(), seed=9)
    cfg = solver.resolved_config()
    assert cfg["solver"] == {"name": "deep_cfr", "seed": 9}
    assert cfg["game"]["name"] == "leduc_poker"
    assert cfg["reward"] == {"scale": 1.0}


def test_table_emulated_deep_cfr_converges_on_kuhn() -> None:
    # Numpy stand-in for the networks: per-player t-weighted mean tables over observation
    # bytes (the regression optimum a real net approximates). Mirrors the Rust oracle test at
    # smaller scale; the point here is the PYTHON loop shape users will write with torch.
    tables: list[dict[bytes, tuple[np.ndarray, float]]] = [{}, {}]
    policy: dict[bytes, np.ndarray] = {}

    def infer_for(player: int) -> Callable[[np.ndarray], np.ndarray]:
        def f(obs: np.ndarray) -> np.ndarray:
            out = np.zeros((obs.shape[0], 2))
            for i, row in enumerate(obs):
                hit = tables[player].get(row.tobytes())
                if hit is not None:
                    out[i] = hit[0] / hit[1]
            return out

        return f

    solver = rf.solvers.DeepCfr(rf.games.KuhnPoker(), seed=13)
    for _ in range(150):
        solver.next_iteration()
        t = float(solver.iteration)
        for player in (0, 1):
            batch = solver.collect(player=player, traversals=16, infer=[infer_for(0), infer_for(1)])
            offsets = batch.advantage_legal_offsets
            for i, row in enumerate(batch.advantage_obs):
                ids = batch.advantage_legal_ids[offsets[i] : offsets[i + 1]]
                targets = batch.advantage_targets[offsets[i] : offsets[i + 1]]
                sums, weight = tables[player].get(row.tobytes(), (np.zeros(2), 0.0))
                sums = sums.copy()
                sums[ids] += t * targets
                tables[player][row.tobytes()] = (sums, weight + t)
            s_off = batch.strategy_legal_offsets
            for i, row in enumerate(batch.strategy_obs):
                ids = batch.strategy_legal_ids[s_off[i] : s_off[i + 1]]
                probs = batch.strategy_probs[s_off[i] : s_off[i + 1]]
                acc = policy.setdefault(row.tobytes(), np.zeros(2))
                acc[ids] += t * probs

    def policy_infer(obs: np.ndarray) -> np.ndarray:
        out = np.zeros((obs.shape[0], 2))
        for i, row in enumerate(obs):
            acc = policy.get(row.tobytes())
            if acc is not None and acc.sum() > 0:
                out[i] = acc / acc.sum()
        return out

    exploitability = solver.exploitability(policy_infer)
    assert exploitability < 0.15, f"table-emulated Deep CFR approaches Nash: {exploitability}"


def test_three_player_kuhn_collects_and_measures() -> None:
    """The N-player lift end to end: per-player infer list of 3, strategy samples from
    player (traverser + 1) % 3 only (the simple average estimator), and the exact instrument
    (NashConv / num_players) on the uniform policy."""
    solver = rf.solvers.DeepCfr(rf.games.KuhnPoker(players=3), seed=0)
    solver.next_iteration()

    def net(obs: np.ndarray) -> np.ndarray:
        return np.zeros((obs.shape[0], 2))

    with pytest.raises(ValueError, match="expected 3 per-player"):
        solver.collect(player=0, traversals=4, infer=[net, net])
    with pytest.raises(ValueError, match="player must be below 3"):
        solver.collect(player=3, traversals=4, infer=net)
    seen: set[int] = set()
    for player in range(3):
        batch = solver.collect(player=player, traversals=32, infer=[net, net, net])
        assert batch.advantage_obs.shape[1] == 9
        others = set(batch.strategy_players.tolist())
        assert others == {(player + 1) % 3}, "simple estimator: exactly the next player"
        seen |= others
    assert seen == {0, 1, 2}
    # Uniform policy instrument: NashConv/3 for uniform 3p Kuhn, pinned from the tabular solver.
    e = solver.exploitability(lambda obs: np.full((obs.shape[0], 2), 0.5))
    assert abs(e - 2.0625 / 3) < 1e-9, e


def test_multiplayer_holdem_collection_smoke() -> None:
    """3- and 6-player hold'em: samples flow through the N-player traversal machinery."""
    for players in (3, 6):
        solver = rf.solvers.DeepCfr(rf.games.TexasHoldem(num_players=players), seed=1)
        solver.next_iteration()

        def net(obs: np.ndarray) -> np.ndarray:
            return np.zeros((obs.shape[0], 3))

        batch = solver.collect(player=players - 1, traversals=8, infer=net)
        assert batch.advantage_obs.shape[0] > 0
        assert set(batch.strategy_players.tolist()) <= set(range(players))


def test_oversized_exact_metrics_raise_value_error_not_panic() -> None:
    """7-player Kuhn is a valid game whose tree exceeds the exact best-response arena cap.
    The boundary contract: that surfaces as ValueError, never a PanicException."""
    solver = rf.solvers.DeepCfr(rf.games.KuhnPoker(players=7), seed=0)
    with pytest.raises(ValueError, match="cap"):
        solver.exploitability(lambda obs: np.full((obs.shape[0], 2), 0.5))
