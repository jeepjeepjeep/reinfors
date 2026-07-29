//! Deep CFR's data generator, validated without any neural network: sample semantics under a
//! zeros net (uniform play), end-to-end convergence with TABLE-emulated "networks" (a
//! t-weighted-mean table is the regression optimum a real net approximates, so the emulation
//! must converge like linear MCCFR), the exploitability instrument's exact pins, determinism,
//! and the construction gates.

use std::collections::HashMap;

use reinfors_core::DeepCfrSolver;
use reinfors_games::{HoldemReward, KuhnEncoder, KuhnPoker, LeducEncoder, LeducPoker};

fn kuhn_solver(seed: u64) -> DeepCfrSolver<KuhnPoker> {
    DeepCfrSolver::new(
        KuhnPoker,
        Box::new(KuhnEncoder),
        Box::new(HoldemReward { scale: 1.0 }),
        seed,
    )
}

fn zeros(_player: usize, _obs: Vec<f32>, rows: usize) -> Vec<f64> {
    vec![0.0; rows * 2]
}

#[test]
fn zeros_net_samples_follow_the_argmax_fallback() {
    let mut solver = kuhn_solver(7);
    solver.next_iteration();
    let (advantage, strategy, stats) = solver.collect(0, 64, zeros);
    assert_eq!(stats.traversals, 64);
    assert!(!advantage.is_empty() && !strategy.is_empty());
    for s in &advantage {
        assert_eq!(s.iteration, 1);
        assert_eq!(s.legal.len(), s.targets.len());
        // A zeros net has no positive advantage, so Brown's fallback plays PURE ARGMAX —
        // the first action on ties. The baseline is then v(first): its own target is zero.
        assert!(
            s.targets[0].abs() < 1e-12,
            "argmax-fallback baseline: {:?}",
            s.targets
        );
    }
    for s in &strategy {
        assert_eq!(s.player, 1, "player 0's traversals sample opponent 1");
        assert_eq!(
            s.probs[0], 1.0,
            "zeros net: argmax fallback is one-hot on the first action"
        );
        assert!(s.probs[1..].iter().all(|&p| p == 0.0));
    }
    // Under a deterministic zeros net the machines stay depth-synchronized, so the sharing
    // mechanism is WITHIN-ROUND DEDUP (64 traversals collapse to a handful of unique rows);
    // the cache itself pays once real nets desynchronize the machines. Assert the dedup.
    assert!(
        stats.infer_rows * 4 < stats.cache_lookups,
        "queries dedupe hard on Kuhn: {} rows for {} queries",
        stats.infer_rows,
        stats.cache_lookups
    );
}

#[test]
fn traversals_are_deterministic_per_seed() {
    let run = |seed: u64| {
        let mut solver = kuhn_solver(seed);
        solver.next_iteration();
        let (advantage, strategy, _) = solver.collect(0, 32, zeros);
        let a: Vec<(Vec<u32>, Vec<i64>)> = advantage
            .iter()
            .map(|s| {
                (
                    s.obs.iter().map(|f| f.to_bits()).collect(),
                    s.targets.iter().map(|t| (t * 1e12) as i64).collect(),
                )
            })
            .collect();
        let b: Vec<Vec<u32>> = strategy
            .iter()
            .map(|s| s.obs.iter().map(|f| f.to_bits()).collect())
            .collect();
        (a, b)
    };
    assert_eq!(run(3), run(3), "same seed, same samples");
    assert_ne!(run(3), run(4), "different seed, different deals");
}

#[test]
fn table_emulated_deep_cfr_converges_on_kuhn() {
    // Emulate the advantage nets with t-weighted running-mean tables (the regression optimum
    // a real net trains toward; regret matching is scale-invariant, so mean vs cumulative
    // regrets yield the same strategies). If the emitted samples carry correct external-
    // sampling semantics, this loop IS linear MCCFR and must converge toward the Kuhn
    // equilibrium.
    struct Table {
        rows: HashMap<Vec<u8>, (Vec<f64>, f64)>, // weighted sums + total weight
    }
    impl Table {
        fn predict(&self, obs: &[f32]) -> Vec<f64> {
            let key: Vec<u8> = obs.iter().flat_map(|f| f.to_le_bytes()).collect();
            match self.rows.get(&key) {
                Some((sums, w)) => sums.iter().map(|s| s / w).collect(),
                None => vec![0.0; 2],
            }
        }
        fn learn(&mut self, obs: &[f32], legal: &[usize], targets: &[f64], t: f64) {
            let key: Vec<u8> = obs.iter().flat_map(|f| f.to_le_bytes()).collect();
            let (sums, w) = self.rows.entry(key).or_insert_with(|| (vec![0.0; 2], 0.0));
            for (&a, &target) in legal.iter().zip(targets) {
                sums[a] += t * target;
            }
            *w += t;
        }
    }
    let mut tables = [
        Table {
            rows: HashMap::new(),
        },
        Table {
            rows: HashMap::new(),
        },
    ];
    // The average policy: t-weighted σ accumulation per infoset, keyed by obs bytes.
    let mut policy: HashMap<Vec<u8>, Vec<f64>> = HashMap::new();
    let mut solver = kuhn_solver(11);
    for _ in 0..250 {
        solver.next_iteration();
        let t = solver.iteration() as f64;
        for player in 0..2 {
            let (advantage, strategy, _) = solver.collect(player, 32, |who, obs, rows| {
                let dim = obs.len() / rows.max(1);
                (0..rows)
                    .flat_map(|i| tables[who].predict(&obs[i * dim..(i + 1) * dim]))
                    .collect()
            });
            for s in &advantage {
                tables[player].learn(&s.obs, &s.legal, &s.targets, t);
            }
            for s in &strategy {
                let key: Vec<u8> = s.obs.iter().flat_map(|f| f.to_le_bytes()).collect();
                let acc = policy
                    .entry(key)
                    .or_insert_with(|| vec![0.0; s.legal.len()]);
                for (i, &p) in s.probs.iter().enumerate() {
                    acc[i] += t * p;
                }
            }
        }
    }
    // Normalize the average policy and re-key it by information-set key for the instrument.
    let features = solver.infoset_features();
    assert_eq!(features.len(), 12, "Kuhn has exactly 12 infosets");
    let mut probs: HashMap<Vec<u8>, Vec<f64>> = HashMap::new();
    for (key, obs, legal) in &features {
        let obs_key: Vec<u8> = obs.iter().flat_map(|f| f.to_le_bytes()).collect();
        if let Some(acc) = policy.get(&obs_key) {
            let total: f64 = acc.iter().sum();
            if total > 0.0 {
                probs.insert(key.clone(), acc.iter().map(|a| a / total).collect());
            }
        }
        assert!(!legal.is_empty());
    }
    let exploitability = solver.exploitability_of(&probs);
    assert!(
        exploitability < 0.1,
        "table-emulated Deep CFR approaches Nash: {exploitability}"
    );
}

#[test]
fn uniform_policy_exploitability_matches_the_known_values() {
    // No probs at all = uniform everywhere; the exact values are pinned by the tabular CFR
    // parity harness (iteration-1 CFR averages are uniform).
    let kuhn = kuhn_solver(0);
    let e = kuhn.exploitability_of(&HashMap::new());
    assert!((e - 11.0 / 24.0).abs() < 1e-12, "Kuhn uniform: {e}");
    let leduc = DeepCfrSolver::new(
        LeducPoker,
        Box::new(LeducEncoder),
        Box::new(HoldemReward { scale: 1.0 }),
        0,
    );
    let e = leduc.exploitability_of(&HashMap::new());
    assert!((e - 2.373611111111111).abs() < 1e-9, "Leduc uniform: {e}");
}

#[test]
fn strategy_samples_come_from_both_seats_across_passes() {
    let mut solver = kuhn_solver(5);
    solver.next_iteration();
    let (_, strat0, _) = solver.collect(0, 16, zeros);
    let (_, strat1, _) = solver.collect(1, 16, zeros);
    assert!(strat0.iter().all(|s| s.player == 1));
    assert!(strat1.iter().all(|s| s.player == 0));
}

#[test]
#[should_panic(expected = "next_iteration")]
fn collecting_before_the_first_iteration_is_a_misuse() {
    let mut solver = kuhn_solver(0);
    let _ = solver.collect(0, 1, zeros);
}

#[test]
#[should_panic(expected = "2-player zero-sum")]
fn the_solver_rejects_more_than_two_players() {
    let game = reinfors_games::TexasHoldem {
        num_players: 3,
        stack: 200,
        small_blind: 5,
        big_blind: 10,
    };
    let _ = DeepCfrSolver::new(
        game,
        Box::new(reinfors_games::HoldemEgocentric {
            num_players: 3,
            stack: 200,
        }),
        Box::new(HoldemReward { scale: 1.0 }),
        0,
    );
}

#[test]
#[should_panic(expected = "information-state keys")]
fn the_solver_rejects_games_without_information_states() {
    let _ = DeepCfrSolver::new(
        reinfors_games::Connect4,
        Box::new(reinfors_games::Connect4Planes),
        Box::new(reinfors_games::Connect4Reward {
            win: 1.0,
            loss: -1.0,
            draw: 0.0,
        }),
        0,
    );
}

/// A fully declared game whose chance ROOT leads to SIMULTANEOUS decisions — the raw-root
/// probe saw only `Actor::Chance` and let it through to a mid-collect panic (review repro).
struct ChanceRootSim;

#[derive(Clone, PartialEq, serde::Serialize, serde::Deserialize)]
struct SimState {
    tick: u8,
}

struct SimEnc;
impl reinfors_core::ActionView for SimEnc {}
impl reinfors_core::StateEncoder for SimEnc {
    type State = SimState;
    fn encode(&self, s: &SimState, _agent: usize) -> Vec<f32> {
        vec![f32::from(s.tick)]
    }
    fn obs_shape(&self) -> (usize, usize, usize) {
        (1, 1, 1)
    }
    fn observation_space(&self) -> reinfors_core::Space {
        reinfors_core::Space::unit_box(vec![1, 1, 1])
    }
}

impl reinfors_core::Game for ChanceRootSim {
    type State = SimState;
    type Event = f64;
    fn num_agents(&self) -> usize {
        2
    }
    fn action_count(&self) -> usize {
        2
    }
    fn perfect_information(&self) -> bool {
        false
    }
    fn all_chance_declared(&self) -> bool {
        true
    }
    fn information_states(&self) -> bool {
        true
    }
    fn information_state_key(&self, s: &SimState, agent: usize) -> Vec<u8> {
        vec![agent as u8, s.tick]
    }
    fn actor(&self, s: &SimState) -> reinfors_core::Actor {
        if s.tick == 0 {
            reinfors_core::Actor::Chance
        } else {
            reinfors_core::Actor::Simultaneous
        }
    }
    fn chance_node(&self, _s: &SimState) -> reinfors_core::ChanceDist {
        reinfors_core::ChanceDist::Uniform(2)
    }
    fn apply_chance_node(
        &self,
        s: &SimState,
        _outcome: usize,
    ) -> reinfors_core::Transition<SimState, f64> {
        reinfors_core::Transition {
            next_state: SimState { tick: s.tick + 1 },
            events: vec![None; 2],
            terminal: false,
        }
    }
    fn legal_actions(&self, s: &SimState, _agent: usize) -> Vec<usize> {
        if s.tick == 1 {
            vec![0, 1]
        } else {
            Vec::new()
        }
    }
    fn step(&self, _s: &SimState, _actions: &[usize]) -> reinfors_core::Transition<SimState, f64> {
        reinfors_core::Transition {
            next_state: SimState { tick: 2 },
            events: vec![None; 2],
            terminal: true,
        }
    }
    fn initial_state(&self, _rng: &mut dyn reinfors_core::Rng) -> SimState {
        SimState { tick: 0 }
    }
}

#[test]
#[should_panic(expected = "sequential 2-player only")]
fn chance_root_simultaneous_games_fail_at_construction() {
    let _ = DeepCfrSolver::new(
        ChanceRootSim,
        Box::new(SimEnc),
        Box::new(HoldemReward { scale: 1.0 }),
        0,
    );
}

#[test]
fn holdem_traversals_run_at_scale() {
    // Full hold'em is the target scale: sampled traversals must run (no enumeration
    // anywhere on this path), with the deal drawn per traversal by the machines' rng.
    let game = reinfors_games::TexasHoldem {
        num_players: 2,
        stack: 200,
        small_blind: 5,
        big_blind: 10,
    };
    let mut solver = DeepCfrSolver::new(
        game,
        Box::new(reinfors_games::HoldemEgocentric {
            num_players: 2,
            stack: 200,
        }),
        Box::new(HoldemReward { scale: 1.0 }),
        9,
    );
    solver.next_iteration();
    let (advantage, strategy, stats) = solver.collect(0, 100, |_p, _obs, rows| vec![0.0; rows * 3]);
    assert!(
        advantage.len() >= 100,
        "every traversal reaches the traverser"
    );
    assert!(!strategy.is_empty());
    // Hold'em observations embed hole cards + history: at small K, distinct deals mean
    // near-zero sharing — batching is the throughput feature here; dedup/caching pay at
    // Brown-scale K and on small games.
    assert!(stats.infer_calls > 0);
    assert!(
        stats.infer_calls * 5 < stats.infer_rows,
        "rows batch per call"
    );
}
