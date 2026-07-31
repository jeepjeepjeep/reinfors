//! CFR on the small poker testbeds: exploitability convergence, the analytic Kuhn value,
//! CFR+ acceleration, MCCFR statistical convergence, table persistence, and the construction
//! gates. (Iteration-exact pyspiel parity lives in the Python harness.)

use reinfors_core::{best_response_value, exploitability, CfrSolver, CfrVariant};
use reinfors_games::{HoldemReward, KuhnPoker, LeducPoker};

fn kuhn_solver(variant: CfrVariant) -> CfrSolver<KuhnPoker> {
    CfrSolver::new(
        KuhnPoker::default(),
        Box::new(HoldemReward { scale: 1.0 }),
        variant,
        7,
    )
}

#[test]
fn vanilla_cfr_approaches_the_kuhn_equilibrium() {
    let mut solver = kuhn_solver(CfrVariant::Vanilla);
    solver.iterate(20);
    let coarse = solver.exploitability().unwrap();
    solver.iterate(980);
    let fine = solver.exploitability().unwrap();
    assert!(fine < coarse, "exploitability falls: {coarse} -> {fine}");
    assert!(fine < 2e-3, "near-Nash after 1000 iterations: {fine}");
    // The analytic game value: -1/18 for the first player at equilibrium.
    let v0 = solver.expected_value(0);
    assert!((v0 - (-1.0 / 18.0)).abs() < 2e-3, "P0 value {v0} vs -1/18");
    assert_eq!(solver.num_infosets(), 12, "Kuhn has exactly 12 infosets");
}

#[test]
fn kuhn_equilibrium_has_the_known_structure() {
    let mut solver = kuhn_solver(CfrVariant::Plus);
    solver.iterate(2000);
    assert!(solver.exploitability().unwrap() < 1e-4);
    // Known equilibrium facts (any alpha in [0, 1/3]): with the JACK facing a bet, player 1
    // always folds; with the KING facing a bet, player 1 always calls; player 1 having the
    // KING after player 0 checks always bets.
    let g = KuhnPoker::default();
    let key = |cards: [u8; 2], hist: &[u8], agent: usize| {
        use reinfors_core::Game;
        let state = reinfors_games::KuhnState {
            cards: cards.to_vec(),
            history: hist.to_vec(),
        };
        g.information_state_key(&state, agent)
    };
    let probs = |k: Vec<u8>| solver.average_strategy(&k).expect("visited").1;
    let j_facing_bet = probs(key([2, 0], &[1], 1)); // p1 holds J, p0 bet
    assert!(j_facing_bet[0] > 0.99, "fold J to a bet: {j_facing_bet:?}");
    let k_facing_bet = probs(key([0, 2], &[1], 1)); // p1 holds K, p0 bet
    assert!(
        k_facing_bet[1] > 0.99,
        "call a bet with K: {k_facing_bet:?}"
    );
    let k_after_check = probs(key([0, 2], &[0], 1)); // p1 holds K, p0 checked
    assert!(
        k_after_check[1] > 0.99,
        "bet K behind a check: {k_after_check:?}"
    );
}

#[test]
fn cfr_plus_converges_faster_than_vanilla() {
    let mut vanilla = kuhn_solver(CfrVariant::Vanilla);
    let mut plus = kuhn_solver(CfrVariant::Plus);
    vanilla.iterate(200);
    plus.iterate(200);
    assert!(
        plus.exploitability().unwrap() < vanilla.exploitability().unwrap(),
        "CFR+ {} vs vanilla {}",
        plus.exploitability().unwrap(),
        vanilla.exploitability().unwrap()
    );
}

#[test]
fn leduc_exploitability_falls_toward_nash() {
    let mut solver = CfrSolver::new(
        LeducPoker,
        Box::new(HoldemReward { scale: 1.0 }),
        CfrVariant::Plus,
        3,
    );
    solver.iterate(20);
    let coarse = solver.exploitability().unwrap();
    solver.iterate(180);
    let fine = solver.exploitability().unwrap();
    assert!(fine < coarse / 3.0, "Leduc converges: {coarse} -> {fine}");
    assert!(fine < 0.05, "near-Nash after 200 CFR+ iterations: {fine}");
}

#[test]
fn external_mccfr_converges_statistically() {
    let mut solver = kuhn_solver(CfrVariant::ExternalMccfr);
    solver.iterate(20_000);
    let e = solver.exploitability().unwrap();
    assert!(e < 0.03, "sampled convergence: {e}");
}

#[test]
fn tables_round_trip_through_save_load() {
    let mut solver = kuhn_solver(CfrVariant::Plus);
    solver.iterate(50);
    let bytes = solver.save();
    let mut restored = kuhn_solver(CfrVariant::Plus);
    restored.load(&bytes).unwrap();
    assert_eq!(restored.iterations(), solver.iterations());
    assert_eq!(restored.num_infosets(), solver.num_infosets());
    assert_eq!(restored.save(), bytes, "canonical serialization");
    assert!((restored.exploitability().unwrap() - solver.exploitability().unwrap()).abs() < 1e-15);
    // Continuing the solve from the restored tables matches continuing the original.
    solver.iterate(10);
    restored.iterate(10);
    assert_eq!(restored.save(), solver.save());
    assert!(restored.load(b"junk").is_err());
}

#[test]
fn mccfr_checkpoints_are_exact() {
    // The payload carries the sampling rng: a restored MCCFR solve continues bit-identically
    // with the original — the checkpoint property the deterministic variants get for free.
    let mut solver = kuhn_solver(CfrVariant::ExternalMccfr);
    solver.iterate(50);
    let mut restored = kuhn_solver(CfrVariant::ExternalMccfr);
    restored.load(&solver.save()).unwrap();
    solver.iterate(25);
    restored.iterate(25);
    assert_eq!(restored.save(), solver.save());
}

#[test]
fn payloads_refuse_incompatible_solvers() {
    let mut plus = kuhn_solver(CfrVariant::Plus);
    plus.iterate(10);
    let payload = plus.save();
    // A different variant must refuse the payload...
    let mut vanilla = kuhn_solver(CfrVariant::Vanilla);
    assert!(vanilla
        .load(&payload)
        .unwrap_err()
        .contains("different CFR variant"));
    // ...and so must a game with a different action space (Kuhn: 2, Leduc: 3).
    let mut leduc = CfrSolver::new(
        LeducPoker,
        Box::new(HoldemReward { scale: 1.0 }),
        CfrVariant::Plus,
        3,
    );
    assert!(leduc
        .load(&payload)
        .unwrap_err()
        .contains("different action space"));
}

#[test]
fn best_response_exploits_a_uniform_profile() {
    // Uniform play is far from equilibrium; the exact best response must find real value, and
    // exploitability must be symmetric-positive.
    let uniform = |_key: &[u8], legal: usize| vec![1.0 / legal as f64; legal];
    let g = KuhnPoker::default();
    let r = HoldemReward { scale: 1.0 };
    let br0 = best_response_value(&g, &r, &uniform, 0).unwrap();
    let br1 = best_response_value(&g, &r, &uniform, 1).unwrap();
    assert!(
        br0 > 0.0 && br1 > 0.0,
        "uniform is exploitable: {br0}, {br1}"
    );
    let e = exploitability(&g, &r, &uniform).unwrap();
    assert!((e - (br0 + br1) / 2.0).abs() < 1e-15);
}

#[test]
fn the_solver_accepts_more_than_two_players() {
    // N-player CFR: no Nash guarantee past 2 players (documented at the gate), but the solver
    // runs — 3-player hold'em constructs and MCCFR iterates.
    let mut solver = CfrSolver::new(
        reinfors_games::TexasHoldem {
            num_players: 3,
            stack: 200,
            small_blind: 5,
            big_blind: 10,
        },
        Box::new(HoldemReward { scale: 1.0 }),
        CfrVariant::ExternalMccfr,
        0,
    );
    solver.iterate(2);
    assert!(solver.num_infosets() > 0);
}

#[test]
#[should_panic(expected = "information-state keys")]
fn the_solver_rejects_games_without_information_states() {
    let _ = CfrSolver::new(
        reinfors_games::Connect4,
        Box::new(reinfors_games::Connect4Reward {
            win: 1.0,
            loss: -1.0,
            draw: 0.0,
        }),
        CfrVariant::Vanilla,
        0,
    );
}

#[test]
#[should_panic(expected = "sequential turn-taking only")]
fn chance_root_simultaneous_games_fail_at_construction() {
    // The stub lives in deep_cfr.rs; a minimal inline twin here keeps the two solver gates
    // independently pinned. Chance root -> simultaneous decisions: the raw-root probe used
    // to pass this and panic mid-solve.
    struct Sim;
    #[derive(Clone, PartialEq, serde::Serialize, serde::Deserialize)]
    struct St {
        tick: u8,
    }
    impl reinfors_core::Game for Sim {
        type State = St;
        type Event = f64;
        fn num_agents(&self) -> usize {
            2
        }
        fn action_count(&self) -> usize {
            2
        }
        fn information_states(&self) -> bool {
            true
        }
        fn information_state_key(&self, s: &St, agent: usize) -> Vec<u8> {
            vec![agent as u8, s.tick]
        }
        fn actor(&self, s: &St) -> reinfors_core::Actor {
            if s.tick == 0 {
                reinfors_core::Actor::Chance
            } else {
                reinfors_core::Actor::Simultaneous
            }
        }
        fn chance_node(&self, _s: &St) -> reinfors_core::ChanceDist {
            reinfors_core::ChanceDist::Uniform(2)
        }
        fn apply_chance_node(&self, s: &St, _outcome: usize) -> reinfors_core::Transition<St, f64> {
            reinfors_core::Transition {
                next_state: St { tick: s.tick + 1 },
                events: vec![None; 2],
                terminal: false,
            }
        }
        fn legal_actions(&self, s: &St, _agent: usize) -> Vec<usize> {
            if s.tick == 1 {
                vec![0, 1]
            } else {
                Vec::new()
            }
        }
        fn step(&self, _s: &St, _a: &[usize]) -> reinfors_core::Transition<St, f64> {
            reinfors_core::Transition {
                next_state: St { tick: 2 },
                events: vec![None; 2],
                terminal: true,
            }
        }
        fn initial_state(&self) -> St {
            St { tick: 0 }
        }
    }
    let _ = CfrSolver::new(
        Sim,
        Box::new(HoldemReward { scale: 1.0 }),
        CfrVariant::Vanilla,
        0,
    );
}

#[test]
fn mccfr_runs_on_heads_up_holdem() {
    // Full hold'em is far beyond exact solving, but the sampled traversal must run: tables
    // grow with visited infosets, chance is drawn rather than enumerated.
    let mut solver = CfrSolver::new(
        reinfors_games::TexasHoldem {
            num_players: 2,
            stack: 20,
            small_blind: 5,
            big_blind: 10,
        },
        Box::new(HoldemReward { scale: 1.0 }),
        CfrVariant::ExternalMccfr,
        1,
    );
    solver.iterate(200);
    assert!(solver.num_infosets() > 100, "tables fill under sampling");
}

#[test]
#[should_panic(expected = "player 2 out of range")]
fn expected_value_rejects_an_out_of_range_player() {
    // A fixed-width Vals row exists for every slot up to MAX_CFR_PLAYERS; without the guard,
    // expected_value(9) on a 2-player game silently returns 0.0.
    let mut solver = kuhn_solver(CfrVariant::Vanilla);
    solver.iterate(1);
    solver.expected_value(2);
}
