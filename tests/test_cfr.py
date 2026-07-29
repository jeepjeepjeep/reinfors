"""The `rf.solvers.Cfr` surface: convergence on the analytic testbeds, strategy queries by
information-state key, persistence, and construction gates. (Iteration-exact pyspiel parity
lives in test_cfr_parity.py, dev-oracle only.)
"""

import pytest
import reinfors as rf


def test_kuhn_converges_to_the_analytic_equilibrium() -> None:
    solver = rf.solvers.Cfr(rf.games.KuhnPoker(), variant="plus", seed=0)
    solver.iterate(500)
    assert solver.exploitability() < 1e-3
    assert abs(solver.expected_value(0) - (-1 / 18)) < 1e-3, "P0 value is -1/18 at Nash"
    assert solver.num_infosets == 12
    assert solver.iterations == 500


def test_variants_rank_as_theory_predicts() -> None:
    exploitabilities = {}
    for variant in ("vanilla", "plus"):
        solver = rf.solvers.Cfr(rf.games.KuhnPoker(), variant=variant, seed=0)
        solver.iterate(200)
        exploitabilities[variant] = solver.exploitability()
    assert exploitabilities["plus"] < exploitabilities["vanilla"]
    mccfr = rf.solvers.Cfr(rf.games.KuhnPoker(), variant="external_mccfr", seed=0)
    mccfr.iterate(20_000)
    assert mccfr.exploitability() < 0.03, "sampled convergence"


def test_leduc_converges_under_cfr_plus() -> None:
    solver = rf.solvers.Cfr(rf.games.LeducPoker(), variant="plus", seed=0)
    solver.iterate(200)
    assert solver.exploitability() < 0.05


def test_average_strategy_is_keyed_by_env_information_state() -> None:
    solver = rf.solvers.Cfr(rf.games.KuhnPoker(), variant="plus", seed=0)
    solver.iterate(200)
    env = rf.Env(rf.games.KuhnPoker(), rf.Reward(), seed=4)
    env.reset()
    (agent,) = env.active_agents()
    strat = solver.average_strategy(env.information_state_key(agent))
    assert strat is not None
    actions, probs = strat
    assert actions == env.legal_actions(agent)
    assert abs(sum(probs) - 1.0) < 1e-12
    assert solver.average_strategy(b"never-visited") is None


def test_solves_round_trip_through_save_load() -> None:
    solver = rf.solvers.Cfr(rf.games.KuhnPoker(), variant="plus", seed=0)
    solver.iterate(50)
    blob = solver.save()
    restored = rf.solvers.Cfr(rf.games.KuhnPoker(), variant="plus", seed=9)
    restored.load(blob)
    assert restored.iterations == solver.iterations
    assert restored.save() == blob, "canonical serialization"
    solver.iterate(10)
    restored.iterate(10)
    assert restored.save() == solver.save(), "the solve continues identically"
    with pytest.raises(ValueError):
        restored.load(b"junk")


def test_mccfr_checkpoints_continue_bit_identically() -> None:
    solver = rf.solvers.Cfr(rf.games.KuhnPoker(), variant="external_mccfr", seed=5)
    solver.iterate(200)
    restored = rf.solvers.Cfr(rf.games.KuhnPoker(), variant="external_mccfr", seed=99)
    restored.load(solver.save())
    solver.iterate(100)
    restored.iterate(100)
    assert restored.save() == solver.save(), "the sampling rng rides in the checkpoint"


def test_snapshots_refuse_a_different_composition() -> None:
    solver = rf.solvers.Cfr(rf.games.KuhnPoker(), variant="plus", seed=0)
    solver.iterate(10)
    payload = solver.save()
    other_variant = rf.solvers.Cfr(rf.games.KuhnPoker(), variant="vanilla", seed=0)
    with pytest.raises(ValueError, match="different composition"):
        other_variant.load(payload)
    other_game = rf.solvers.Cfr(rf.games.LeducPoker(), variant="plus", seed=0)
    with pytest.raises(ValueError, match="different composition"):
        other_game.load(payload)


def test_expected_value_validates_the_player() -> None:
    solver = rf.solvers.Cfr(rf.games.KuhnPoker(), variant="plus", seed=0)
    solver.iterate(10)
    assert abs(solver.expected_value(0) + solver.expected_value(1)) < 1e-12, "zero-sum"
    with pytest.raises(ValueError, match="player must be 0 or 1"):
        solver.expected_value(2)


def test_mccfr_runs_on_heads_up_holdem() -> None:
    solver = rf.solvers.Cfr(rf.games.TexasHoldem(num_players=2, stack=20), variant="external_mccfr", seed=1)
    solver.iterate(100)
    assert solver.num_infosets > 100, "tables fill under sampling"


def test_construction_gates() -> None:
    with pytest.raises(ValueError, match="information-state"):
        rf.solvers.Cfr(rf.games.Connect4())
    with pytest.raises(ValueError, match="2-player"):
        rf.solvers.Cfr(rf.games.TexasHoldem(num_players=3), variant="external_mccfr")
    with pytest.raises(ValueError, match="external_mccfr"):
        rf.solvers.Cfr(rf.games.TexasHoldem(num_players=2), variant="plus")
    with pytest.raises(ValueError, match="unknown CFR variant"):
        rf.solvers.Cfr(rf.games.KuhnPoker(), variant="zap")
