//! evaluate_seeded: scalar-path parity and independence from batch composition.

use reinfors_core::policies::tree::alphazero::{AlphaZero, AlphaZeroConfig};
use reinfors_core::policies::tree::mcts::{NoiseScope, SequentialBackup};
use reinfors_core::{
    Actor, ChanceMode, Evaluator, Game, InferMode, Policy, SearchConfig, SelectiveExpectimax,
    Space, StateEncoder, Transition,
};

#[derive(Clone)]
struct St {
    tick: usize,
}
struct Count;
impl Game for Count {
    type State = St;
    type Event = ();
    fn num_agents(&self) -> usize {
        1
    }
    fn action_count(&self) -> usize {
        3
    }
    fn actor(&self, _s: &St) -> Actor {
        Actor::Agent(0)
    }
    fn legal_actions(&self, _s: &St, _agent: usize) -> Vec<usize> {
        vec![0, 1, 2]
    }
    fn step(&self, s: &St, a: &[usize]) -> Transition<St, ()> {
        Transition {
            next_state: St {
                tick: s.tick + 1 + a[0],
            },
            events: vec![None],
            terminal: s.tick + 1 + a[0] >= 6,
        }
    }
    fn initial_state(&self) -> St {
        St { tick: 0 }
    }
}
struct Enc;
impl reinfors_core::ActionView for Enc {}
impl StateEncoder for Enc {
    type State = St;
    fn encode(&self, s: &St, _agent: usize) -> Vec<f32> {
        vec![s.tick as f32, 1.0]
    }
    fn obs_shape(&self) -> (usize, usize, usize) {
        (1, 1, 2)
    }
    fn observation_space(&self) -> Space {
        Space::unit_box(vec![1, 1, 2])
    }
}
struct Zero;
impl reinfors_core::Reward for Zero {
    type Event = ();
    fn step_reward(&self, _e: &(), _agent: usize) -> f64 {
        0.0
    }
}

fn az() -> AlphaZero {
    AlphaZero::new(AlphaZeroConfig {
        num_simulations: 12,
        c_puct: 1.5,
        gamma: 1.0,
        max_depth: 64,
        noise_epsilon: 0.4,
        noise_alpha: 0.5,
        temperature: 1.0,
        temperature_drop: 4,
        chance: ChanceMode::AlwaysResample,
        noise_scope: NoiseScope::Requester,
        sequential_backup: SequentialBackup::Auto,
    })
}

fn az_infer(_p: usize, obs: Vec<f32>, n: usize) -> Vec<f64> {
    assert_eq!(obs.len(), n * 2);
    (0..n).flat_map(|_| [0.3, 0.1, 0.2, 0.5]).collect()
}

fn requests(ticks: &[usize]) -> Vec<(St, usize)> {
    ticks.iter().map(|&t| (St { tick: t }, 0)).collect()
}

#[test]
fn az_seeded_with_legacy_derivation_matches_scalar_path() {
    let seed = 42u64;
    let mut f1 = az_infer;
    let mut e1 = Evaluator::new(&mut f1, InferMode::Shared, None);
    let scalar = az().evaluate(
        &Count,
        &Enc,
        &Zero,
        requests(&[0, 1, 2]),
        seed,
        false,
        &mut e1,
    );

    let seeds: Vec<u64> = (0..3)
        .map(|ti| seed ^ (ti as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
        .collect();
    let mut f2 = az_infer;
    let mut e2 = Evaluator::new(&mut f2, InferMode::Shared, None);
    let seeded = az()
        .evaluate_seeded(
            &Count,
            &Enc,
            &Zero,
            requests(&[0, 1, 2]),
            &seeds,
            false,
            &mut e2,
        )
        .unwrap();
    for (a, b) in scalar.iter().zip(&seeded) {
        assert_eq!(a.visits, b.visits);
        assert_eq!(a.values, b.values);
    }
}

#[test]
fn az_seeded_result_is_independent_of_batch_composition() {
    // Noise is live (eps 0.4), so any index-derived stream would break this.
    let seeds = [7u64, 8, 9];
    let mut f1 = az_infer;
    let mut e1 = Evaluator::new(&mut f1, InferMode::Shared, None);
    let full = az()
        .evaluate_seeded(
            &Count,
            &Enc,
            &Zero,
            requests(&[0, 1, 2]),
            &seeds,
            false,
            &mut e1,
        )
        .unwrap();

    let mut f2 = az_infer;
    let mut e2 = Evaluator::new(&mut f2, InferMode::Shared, None);
    let alone = az()
        .evaluate_seeded(
            &Count,
            &Enc,
            &Zero,
            requests(&[1]),
            &seeds[1..2],
            false,
            &mut e2,
        )
        .unwrap();
    assert_eq!(full[1].visits, alone[0].visits);
    assert_eq!(full[1].values, alone[0].values);

    let mut f3 = az_infer;
    let mut e3 = Evaluator::new(&mut f3, InferMode::Shared, None);
    let permuted = az()
        .evaluate_seeded(
            &Count,
            &Enc,
            &Zero,
            requests(&[2, 0, 1]),
            &[seeds[2], seeds[0], seeds[1]],
            false,
            &mut e3,
        )
        .unwrap();
    for (i, j) in [(0usize, 1usize), (1, 2), (2, 0)] {
        assert_eq!(full[i].visits, permuted[j].visits);
    }
}

#[test]
fn expectimax_seeded_result_is_independent_of_batch_composition() {
    let policy = SelectiveExpectimax::new(
        SearchConfig {
            gamma: 1.0,
            expansion_budget: 16,
            top_k: 4,
            max_depth: 8,
            beta: 1.0,
            chance: ChanceMode::Committed { samples: 1 },
            opponent: reinfors_core::Opponent::Uniform,
        },
        1,
        0.0,
    );
    let q_infer = |_p: usize, obs: Vec<f32>, n: usize| -> Vec<f64> {
        assert_eq!(obs.len(), n * 2);
        vec![0.25; n * 3]
    };
    let seeds = [11u64, 12, 13];
    let mut f1 = q_infer;
    let mut e1 = Evaluator::new(&mut f1, InferMode::Shared, None);
    let full = policy
        .evaluate_seeded(
            &Count,
            &Enc,
            &Zero,
            requests(&[0, 1, 2]),
            &seeds,
            false,
            &mut e1,
        )
        .unwrap();
    let mut f2 = q_infer;
    let mut e2 = Evaluator::new(&mut f2, InferMode::Shared, None);
    let alone = policy
        .evaluate_seeded(
            &Count,
            &Enc,
            &Zero,
            requests(&[2]),
            &seeds[2..3],
            false,
            &mut e2,
        )
        .unwrap();
    assert_eq!(full[2].values, alone[0].values);
    assert_eq!(full[2].legal, alone[0].legal);
}
