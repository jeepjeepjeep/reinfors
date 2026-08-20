//! Stage A equivalence: the stepped machine driven in lockstep must reproduce
//! `Policy::evaluate` exactly for the migrated policies.

use reinfors_core::encoder::PermTable;
use reinfors_core::rollout::driver::drive_to_completion;
use reinfors_core::rollout::evaluator::Evaluator;
use reinfors_core::{
    Actor, EpsilonGreedyQ, Game, InferMode, Policy, PpoActor, Reward, Space, StateEncoder,
    Transition,
};

#[derive(Clone)]
struct St {
    tick: usize,
}

struct RR;
impl Game for RR {
    type State = St;
    type Event = ();
    fn num_agents(&self) -> usize {
        2
    }
    fn action_count(&self) -> usize {
        3
    }
    fn actor(&self, s: &St) -> Actor {
        Actor::Agent(s.tick % 2)
    }
    fn legal_actions(&self, s: &St, agent: usize) -> Vec<usize> {
        if agent == s.tick % 2 {
            vec![0, 1, 2]
        } else {
            Vec::new()
        }
    }
    fn step(&self, s: &St, _actions: &[usize]) -> Transition<St, ()> {
        Transition {
            next_state: St { tick: s.tick + 1 },
            events: vec![None; 2],
            terminal: s.tick >= 6,
        }
    }
    fn initial_state(&self) -> St {
        St { tick: 0 }
    }
}

struct Enc;
impl reinfors_core::ActionView for Enc {
    fn head_index(&self, action: usize, agent: usize) -> usize {
        // A real permuting encoder so the PermTable path is exercised.
        if agent == 1 {
            2 - action
        } else {
            action
        }
    }
    fn game_action(&self, head: usize, agent: usize) -> usize {
        if agent == 1 {
            2 - head
        } else {
            head
        }
    }
}
impl StateEncoder for Enc {
    type State = St;
    fn encode(&self, s: &St, agent: usize) -> Vec<f32> {
        vec![s.tick as f32, agent as f32]
    }
    fn obs_shape(&self) -> (usize, usize, usize) {
        (1, 1, 2)
    }
    fn observation_space(&self) -> Space {
        Space::unit_box(vec![1, 1, 2])
    }
}

struct Zero;
impl Reward for Zero {
    type Event = ();
    fn step_reward(&self, _e: &(), _agent: usize) -> f64 {
        0.0
    }
}

fn requests() -> Vec<(St, usize)> {
    (0..5).map(|t| (St { tick: t }, t % 2)).collect()
}

fn infer(width: usize) -> impl FnMut(usize, Vec<f32>, usize) -> Vec<f64> {
    move |player, obs: Vec<f32>, n| {
        let mut out = Vec::with_capacity(n * width);
        for i in 0..n {
            for j in 0..width {
                out.push(obs[i * 2] as f64 * 10.0 + player as f64 + j as f64 * 0.1);
            }
        }
        out
    }
}

#[test]
fn epsilon_greedy_stepped_matches_evaluate() {
    let policy = EpsilonGreedyQ::new(2, 0.1);
    let perms = PermTable::build(&Enc, 3, 2);
    let mut f1 = infer(6);
    let mut e1 = Evaluator::new(&mut f1, InferMode::Shared, None);
    let old = policy.evaluate(&RR, &Enc, &Zero, requests(), 7, false, &mut e1);
    let decisions: Vec<(St, Vec<usize>)> =
        requests().into_iter().map(|(s, a)| (s, vec![a])).collect();
    let mut f2 = infer(6);
    let mut e2 = Evaluator::new(&mut f2, InferMode::Shared, None);
    let mut rng = reinfors_core::SplitMix64::new(0);
    let new = drive_to_completion(
        &policy, &RR, &Enc, &Zero, &perms, false, &decisions, &mut rng, &mut e2,
    );
    assert_eq!(old.len(), new.len());
    for (o, n) in old.iter().zip(&new) {
        let (eval, targets) = &n[0];
        assert!(targets.is_empty());
        assert_eq!(o.values, eval.values);
        assert_eq!(o.legal, eval.legal);
    }
}

#[test]
fn ppo_stepped_matches_evaluate() {
    let policy = PpoActor::new();
    let perms = PermTable::build(&Enc, 3, 2);
    let mut f1 = infer(4);
    let mut e1 = Evaluator::new(&mut f1, InferMode::Shared, None);
    let old = policy.evaluate(&RR, &Enc, &Zero, requests(), 7, false, &mut e1);
    let decisions: Vec<(St, Vec<usize>)> =
        requests().into_iter().map(|(s, a)| (s, vec![a])).collect();
    let mut f2 = infer(4);
    let mut e2 = Evaluator::new(&mut f2, InferMode::Shared, None);
    let mut rng = reinfors_core::SplitMix64::new(0);
    let new = drive_to_completion(
        &policy, &RR, &Enc, &Zero, &perms, false, &decisions, &mut rng, &mut e2,
    );
    for (o, n) in old.iter().zip(&new) {
        let (eval, _) = &n[0];
        assert_eq!(o.log_probs, eval.log_probs);
        assert_eq!(o.value, eval.value);
        assert_eq!(o.legal, eval.legal);
    }
}

#[test]
fn minimax_stepped_matches_evaluate_on_a_chance_free_game() {
    use reinfors_core::{ChanceMode, Minimax};
    // Chance-free game + Committed{1}: the search consumes no randomness, so the stepped
    // machine must reproduce `evaluate` byte-for-byte through the shared engine.
    let policy = Minimax::new(2, None, ChanceMode::Committed { samples: 1 }, 1.0);
    let perms = PermTable::build(&Enc, 3, 2);
    let mut f1 = infer(3);
    let mut e1 = Evaluator::new(&mut f1, InferMode::Shared, None);
    let old = policy.evaluate(&RR, &Enc, &Zero, requests(), 7, true, &mut e1);
    let decisions: Vec<(St, Vec<usize>)> =
        requests().into_iter().map(|(s, a)| (s, vec![a])).collect();
    let mut f2 = infer(3);
    let mut e2 = Evaluator::new(&mut f2, InferMode::Shared, None);
    let mut rng = reinfors_core::SplitMix64::new(0);
    let new = drive_to_completion(
        &policy, &RR, &Enc, &Zero, &perms, true, &decisions, &mut rng, &mut e2,
    );
    for (o, n) in old.iter().zip(&new) {
        let (eval, targets) = &n[0];
        assert_eq!(o.values, eval.values);
        assert_eq!(o.legal, eval.legal);
        assert_eq!(
            o.interior, *targets,
            "interior targets move to the paired return"
        );
    }
}

#[test]
fn mcts_stepped_matches_evaluate_on_a_deterministic_config() {
    use reinfors_core::policies::tree::mcts::{ActBy, Mcts, MctsConfig};
    use reinfors_core::ChanceMode;
    // Chance-free game + UCT: search results must match `evaluate` exactly; telemetry
    // counters (rounds, hit/shared rows) legitimately differ under per-eval suspension.
    let policy = Mcts::new(
        MctsConfig {
            num_simulations: 8,
            uct_c: 1.4,
            gamma: 1.0,
            max_depth: 16,
            temperature: 0.0,
            temperature_drop: 0,
            chance: ChanceMode::AlwaysResample,
        },
        ActBy::Visits,
    );
    let perms = PermTable::build(&Enc, 3, 2);
    let mut f1 = infer(3);
    let mut e1 = Evaluator::new(&mut f1, InferMode::Shared, None);
    let old = policy.evaluate(&RR, &Enc, &Zero, requests(), 7, false, &mut e1);
    let decisions: Vec<(St, Vec<usize>)> =
        requests().into_iter().map(|(s, a)| (s, vec![a])).collect();
    let mut f2 = infer(3);
    let mut e2 = Evaluator::new(&mut f2, InferMode::Shared, None);
    let mut rng = reinfors_core::SplitMix64::new(0);
    let new = drive_to_completion(
        &policy, &RR, &Enc, &Zero, &perms, false, &decisions, &mut rng, &mut e2,
    );
    for (o, n) in old.iter().zip(&new) {
        let (eval, targets) = &n[0];
        assert!(targets.is_empty());
        assert_eq!(o.values, eval.values);
        assert_eq!(o.visits, eval.visits);
        assert_eq!(o.legal, eval.legal);
    }
}
