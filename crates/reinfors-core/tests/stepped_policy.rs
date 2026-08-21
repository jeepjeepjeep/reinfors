//! Stepped-machine verification. Equivalence against `Policy::evaluate` was proven before
//! its deletion (commits bfa00b90/b426beba/15a791ad); these tests keep the properties alive:
//! hand-computed outputs through a permuting encoder, and drive-twice determinism.

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

fn decisions() -> Vec<(St, Vec<usize>)> {
    (0..5).map(|t| (St { tick: t }, vec![t % 2])).collect()
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
fn epsilon_greedy_stepped_produces_hand_computed_permuted_values() {
    let policy = EpsilonGreedyQ::new(2, 0.1);
    let perms = PermTable::build(&Enc, 3, 2);
    let mut f = infer(6);
    let mut e = Evaluator::new(&mut f, InferMode::Shared, None);
    let mut rng = reinfors_core::SplitMix64::new(0);
    let out = drive_to_completion(
        &policy,
        &RR,
        &Enc,
        &Zero,
        &perms,
        false,
        &decisions(),
        &mut rng,
        &mut e,
    );
    // Decision 0 (tick 0, agent 0, identity frame): head row is direct.
    let (eval0, targets0) = &out[0][0];
    assert!(targets0.is_empty());
    assert_eq!(eval0.values[0], vec![0.0, 0.1, 0.2]);
    assert_eq!(eval0.values[1], vec![3.0 * 0.1, 4.0 * 0.1, 5.0 * 0.1]);
    // Decision 1 (tick 1, agent 1, mirrored frame): game action g reads head column 2-g.
    // InferMode::Shared routes every row through the shared callback as player 0.
    let (eval1, _) = &out[1][0];
    let head0: Vec<f64> = vec![10.0, 10.1, 10.2];
    let expect: Vec<f64> = (0..3).map(|g| head0[2 - g]).collect();
    assert_eq!(eval1.values[0], expect);
    assert_eq!(eval1.legal, vec![0, 1, 2]);
}

#[test]
fn ppo_stepped_log_probs_are_the_masked_softmax_of_permuted_logits() {
    use reinfors_core::policies::modelfree::ppo::masked_log_probs;
    let policy = PpoActor::new();
    let perms = PermTable::build(&Enc, 3, 2);
    let mut f = infer(4);
    let mut e = Evaluator::new(&mut f, InferMode::Shared, None);
    let mut rng = reinfors_core::SplitMix64::new(0);
    let out = drive_to_completion(
        &policy,
        &RR,
        &Enc,
        &Zero,
        &perms,
        false,
        &decisions(),
        &mut rng,
        &mut e,
    );
    let (eval0, _) = &out[0][0];
    let row: Vec<f64> = vec![0.0, 0.1, 0.2];
    assert_eq!(eval0.log_probs, masked_log_probs(&row, &[0, 1, 2]));
    assert_eq!(eval0.value, 3.0 * 0.1);
    // Mirrored agent: legality maps into the head frame before the softmax; Shared mode
    // routes rows as player 0.
    let (eval1, _) = &out[1][0];
    let head_row: Vec<f64> = vec![10.0, 10.1, 10.2];
    assert_eq!(eval1.log_probs, masked_log_probs(&head_row, &[2, 1, 0]));
    assert_eq!(eval1.value, 10.3);
}

fn drive_twice_identical<P>(
    policy: &P,
    width: usize,
    collect_interior: bool,
    eq: impl Fn(&P::Evaluation, &P::Evaluation) -> bool,
) where
    P: Policy,
{
    let perms = PermTable::build(&Enc, 3, 2);
    let run = |seed: u64| {
        let mut f = infer(width);
        let mut e = Evaluator::new(&mut f, InferMode::Shared, None);
        let mut rng = reinfors_core::SplitMix64::new(seed);
        drive_to_completion(
            policy,
            &RR,
            &Enc,
            &Zero,
            &perms,
            collect_interior,
            &decisions(),
            &mut rng,
            &mut e,
        )
    };
    let a = run(3);
    let b = run(3);
    for (x, y) in a.iter().zip(&b) {
        for ((ex, tx), (ey, ty)) in x.iter().zip(y) {
            assert!(eq(ex, ey));
            assert_eq!(tx, ty);
        }
    }
}

#[test]
fn minimax_stepped_is_deterministic_per_seed_with_interior_targets() {
    use reinfors_core::{ChanceMode, Minimax};
    let policy = Minimax::new(2, None, ChanceMode::Committed { samples: 1 }, 1.0);
    drive_twice_identical(&policy, 3, true, |a, b| {
        a.values == b.values && a.visits == b.visits && a.legal == b.legal
    });
}

#[test]
fn mcts_stepped_is_deterministic_per_seed() {
    use reinfors_core::policies::tree::mcts::{ActBy, Mcts, MctsConfig};
    use reinfors_core::ChanceMode;
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
    drive_twice_identical(&policy, 3, false, |a, b| {
        a.values == b.values && a.visits == b.visits && a.legal == b.legal
    });
}

#[test]
fn single_threaded_scheduling_is_reproducible_and_fan_meets_the_floor() {
    use reinfors_core::policies::tree::mcts::{ActBy, Mcts, MctsConfig};
    use reinfors_core::rollout::engine::{Engine, EngineParams};
    use reinfors_core::{ChanceMode, Reward as _};
    let _ = &Zero.step_reward(&(), 0);
    let run = |n_threads: Option<usize>| {
        let mut engine = Engine::new(
            RR,
            Box::new(Enc),
            Box::new(Zero),
            Mcts::new(
                MctsConfig {
                    num_simulations: 6,
                    uct_c: 1.4,
                    gamma: 1.0,
                    max_depth: 16,
                    temperature: 0.0,
                    temperature_drop: 0,
                    chance: ChanceMode::AlwaysResample,
                },
                ActBy::Visits,
            ),
            reinfors_core::TreeStrap::new(0.99, 0.3, 1.0, false),
            EngineParams {
                n_games: 4,
                seed: 9,
                batch_size: Some(3),
                n_threads,
                ..Default::default()
            },
        );
        let mut infer = |_obs: Vec<f32>, n: usize| vec![0.25; n * 3];
        engine.collect(24, &mut infer)
    };
    // The determinism contract: n_threads=1 is exactly reproducible; a fanned run is a
    // valid collection (completion order may change window composition) that still
    // meets the floor.
    let (a, _) = run(Some(1));
    let (b, _) = run(Some(1));
    assert_eq!(a.len(), b.len(), "n_threads=1 reruns must match");
    for (x, y) in a.iter().zip(&b) {
        assert_eq!(x.0, y.0, "obs differ between n_threads=1 reruns");
        assert_eq!(x.1, y.1, "targets differ between n_threads=1 reruns");
    }
    let (fanned, _) = run(Some(4));
    assert!(fanned.len() >= 24, "fanned run must meet the record floor");
}

/// A deliberately malformed policy: one Pending round, then `finish` returns TWO
/// evaluations for however many perspectives it was given.
struct DoubleFinish;
impl Policy for DoubleFinish {
    type Evaluation = ();
    type PolicyState = ();
    type Search<S: Send> = bool;
    fn begin_search<G: Game + Sync>(
        &self,
        _ctx: reinfors_core::policy::SearchCtx<'_, G>,
        _state: &G::State,
        _perspectives: &[usize],
    ) -> Self::Search<G::State>
    where
        G::State: Send,
    {
        false
    }
    fn round<G: Game + Sync>(
        &self,
        _ctx: reinfors_core::policy::SearchCtx<'_, G>,
        emitted: &mut Self::Search<G::State>,
        out: &mut reinfors_core::policy::RequestSink,
    ) -> reinfors_core::policy::RoundStatus
    where
        G::State: Send,
    {
        if *emitted {
            return reinfors_core::policy::RoundStatus::Done;
        }
        *emitted = true;
        out.push(0, &[0.0, 0.0]);
        reinfors_core::policy::RoundStatus::Pending
    }
    fn absorb<G: Game + Sync>(
        &self,
        _ctx: reinfors_core::policy::SearchCtx<'_, G>,
        _search: &mut Self::Search<G::State>,
        _rows: reinfors_core::policy::RowsView<'_>,
    ) where
        G::State: Send,
    {
    }
    fn finish<G: Game + Sync>(
        &self,
        _ctx: reinfors_core::policy::SearchCtx<'_, G>,
        _search: Self::Search<G::State>,
    ) -> Vec<((), Vec<reinfors_core::learner::InteriorTarget>)>
    where
        G::State: Send,
    {
        vec![((), Vec::new()), ((), Vec::new())]
    }
    fn max_agents(&self, _sequential: bool) -> Option<usize> {
        None
    }
    fn supports_imperfect_information(&self) -> bool {
        true
    }
    fn begin_episode(&self, _rng: &mut dyn reinfors_core::Rng) {}
    fn encode_eval(&self, _eval: &(), _out: &mut Vec<u8>) {}
    fn decode_eval(
        &self,
        _r: &mut reinfors_core::codec::bytes::Reader,
        _action_count: usize,
    ) -> Result<(), String> {
        Ok(())
    }
    fn policy_state_to_u64(&self, _s: &()) -> u64 {
        0
    }
    fn policy_state_from_u64(&self, _v: u64) -> Result<(), String> {
        Ok(())
    }
    fn select(&self, _eval: &(), _state: &mut (), _rng: &mut dyn reinfors_core::Rng) -> usize {
        0
    }
    fn fold_telemetry(&self, _eval: &(), _stats: &mut reinfors_core::CollectStats) {}
}

#[test]
#[should_panic(expected = "one evaluation per perspective")]
fn finish_cardinality_is_checked_per_search() {
    // A policy over- or under-producing evaluations must fail loudly at ITS search, not
    // silently shift every later game's evaluations (the global-total trap).
    let perms = PermTable::build(&Enc, 3, 2);
    let mut f = infer(3);
    let mut e = Evaluator::new(&mut f, InferMode::Shared, None);
    let mut rng = reinfors_core::SplitMix64::new(0);
    let _ = drive_to_completion(
        &DoubleFinish,
        &RR,
        &Enc,
        &Zero,
        &perms,
        false,
        &decisions(),
        &mut rng,
        &mut e,
    );
}

#[test]
#[should_panic(expected = "batch_size must be >= 1")]
fn zero_batch_size_is_rejected_at_construction() {
    use reinfors_core::rollout::engine::{Engine, EngineParams};
    let _ = Engine::new(
        RR,
        Box::new(Enc),
        Box::new(Zero),
        PpoActor::new(),
        reinfors_core::Ppo::new(1.0, 0.95),
        EngineParams {
            n_games: 2,
            seed: 0,
            batch_size: Some(0),
            ..Default::default()
        },
    );
}
