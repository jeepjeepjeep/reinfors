//! The agent-count capability seam (N-player phase 0): policies declare `max_agents`, engine
//! construction enforces it (a real assert, not a debug one — an unsupported count computes
//! silently wrong values, e.g. negamax past two players), the search entry points carry their own
//! backstops, and a capability-free policy drives an N>2 game through the engine end to end.

use reinfors_core::{
    mcts_many, search_many, ActBy, Actor, AlphaZero, AlphaZeroConfig, ChanceMode, Dqn, Engine,
    EngineParams, EpsilonGreedyQ, Evaluator, Game, Mcts, MctsConfig, Opponent, Policy, Reward, Rng,
    SearchConfig, SelectiveExpectimax, Space, StateEncoder, Transition, TreeStrap,
};

#[derive(Clone)]
struct St {
    tick: usize,
}

/// A minimal 3-agent simultaneous game: everyone picks one of two actions each tick, nothing
/// matters, terminal after three ticks.
struct ThreeWay;

impl Game for ThreeWay {
    type State = St;
    type Event = ();
    fn num_agents(&self) -> usize {
        3
    }
    fn action_count(&self) -> usize {
        2
    }
    fn actor(&self, _s: &St) -> Actor {
        Actor::Simultaneous
    }
    fn legal_actions(&self, _s: &St, _agent: usize) -> Vec<usize> {
        vec![0, 1]
    }
    fn step(&self, s: &St, _actions: &[usize]) -> Transition<St, ()> {
        Transition {
            next_state: St { tick: s.tick + 1 },
            events: vec![(); 3],
            terminal: s.tick + 1 >= 3,
        }
    }
    fn initial_state(&self, _rng: &mut dyn Rng) -> St {
        St { tick: 0 }
    }
}

struct Enc;
impl reinfors_core::ActionView for Enc {}
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

fn search_cfg() -> SearchConfig {
    SearchConfig {
        gamma: 0.99,
        beta: 1.0,
        expansion_budget: 4,
        top_k: 2,
        max_depth: 2,
        chance: ChanceMode::Committed { samples: 1 },
        opponent: Opponent::Uniform,
    }
}

#[test]
fn policies_declare_their_agent_capability() {
    // Expectimax stays 2-capped on this branch under either dynamics (its N support lands in
    // the stacked phase-2 PR).
    assert_eq!(
        SelectiveExpectimax::new(search_cfg(), 2, 0.0).max_agents(true),
        Some(2)
    );
    assert_eq!(
        SelectiveExpectimax::new(search_cfg(), 2, 0.0).max_agents(false),
        Some(2)
    );
    let mcts_cfg = MctsConfig {
        num_simulations: 4,
        uct_c: 1.0,
        gamma: 0.99,
        max_depth: 4,
        temperature: 0.0,
        temperature_drop: 0,
        chance: ChanceMode::Committed { samples: 1 },
    };
    let mcts = Mcts::new(mcts_cfg, ActBy::Value);
    // Dynamics-aware: UCT sequential caps at 2 (Q leaf values need own-turn decision points);
    // simultaneous DUCT runs at any N.
    assert_eq!(mcts.max_agents(true), Some(2));
    assert_eq!(mcts.max_agents(false), None);
    let az_cfg = AlphaZeroConfig {
        num_simulations: 4,
        c_puct: 1.5,
        gamma: 0.99,
        max_depth: 4,
        noise_epsilon: 0.0,
        noise_alpha: 0.3,
        temperature: 0.0,
        temperature_drop: 0,
        chance: ChanceMode::Committed { samples: 1 },
        noise_scope: reinfors_core::NoiseScope::Requester,
    };
    assert_eq!(AlphaZero::new(az_cfg).max_agents(true), None);
    assert_eq!(EpsilonGreedyQ::new(1, 0.0).max_agents(true), None);
}

/// 3 agents taking turns (agent = tick mod 3); nothing matters, terminal after three plies.
struct RoundRobin;

impl Game for RoundRobin {
    type State = St;
    type Event = ();
    fn num_agents(&self) -> usize {
        3
    }
    fn action_count(&self) -> usize {
        2
    }
    fn actor(&self, s: &St) -> Actor {
        Actor::Agent(s.tick % 3)
    }
    fn legal_actions(&self, s: &St, agent: usize) -> Vec<usize> {
        if agent == s.tick % 3 && s.tick < 3 {
            vec![0, 1]
        } else {
            Vec::new()
        }
    }
    fn step(&self, s: &St, _actions: &[usize]) -> Transition<St, ()> {
        Transition {
            next_state: St { tick: s.tick + 1 },
            events: vec![(); 3],
            terminal: s.tick + 1 >= 3,
        }
    }
    fn initial_state(&self, _rng: &mut dyn Rng) -> St {
        St { tick: 0 }
    }
}

#[test]
#[should_panic(expected = "at most 2 agents")]
fn engine_rejects_a_capped_policy_on_a_three_agent_game() {
    let policy = SelectiveExpectimax::new(search_cfg(), 2, 0.0);
    let learner = TreeStrap::new(0.99, 0.3, 1.0, false);
    let _ = Engine::new(
        ThreeWay,
        Box::new(Enc),
        Box::new(Zero),
        policy,
        learner,
        EngineParams {
            n_games: 1,
            seed: 0,
        },
    );
}

#[test]
#[should_panic(expected = "at most 2 agents")]
fn expectimax_search_backstop_rejects_three_agents() {
    let mut infer = |_obs: Vec<f32>, n: usize| vec![0.0; n * 2];
    let _ = search_many(
        &ThreeWay,
        &Enc,
        &Zero,
        &search_cfg(),
        vec![(St { tick: 0 }, 0)],
        false,
        0,
        &mut infer,
    );
}

fn mcts_cfg() -> MctsConfig {
    MctsConfig {
        num_simulations: 16,
        uct_c: 1.0,
        gamma: 0.99,
        max_depth: 6,
        temperature: 0.0,
        temperature_drop: 0,
        chance: ChanceMode::Committed { samples: 1 },
    }
}

#[test]
#[should_panic(expected = "value head")]
fn uct_rejects_sequential_three_agent_games() {
    // Q-derived (UCT) leaf values exist only at the evaluated agent's own decision points, which
    // a sequential game gives non-movers none of — N>2 sequential search needs PUCT.
    let mut infer = |_obs: Vec<f32>, n: usize| vec![0.0; n * 2];
    let mut eval = Evaluator::new(&mut infer, None);
    let _ = mcts_many(
        &RoundRobin,
        &Enc,
        &Zero,
        &mcts_cfg(),
        vec![(St { tick: 0 }, 0)],
        0,
        &mut eval,
    );
}

#[test]
fn uct_searches_a_simultaneous_three_agent_game() {
    // DUCT-N: every agent owns a decoupled table, so simultaneous games search at any N.
    let mut infer = |_obs: Vec<f32>, n: usize| vec![0.0; n * 2];
    let mut eval = Evaluator::new(&mut infer, None);
    let evals = mcts_many(
        &ThreeWay,
        &Enc,
        &Zero,
        &mcts_cfg(),
        vec![(St { tick: 0 }, 0), (St { tick: 0 }, 2)],
        0,
        &mut eval,
    );
    for e in &evals {
        assert_eq!(e.legal, vec![0, 1]);
        assert_eq!(e.visits.iter().sum::<f64>() as usize, 16); // requester's table, all sims
    }
}

#[test]
fn mcts_engine_collects_on_a_three_agent_simultaneous_game() {
    let policy = Mcts::new(mcts_cfg(), ActBy::Value);
    let learner = TreeStrap::new(0.99, 0.3, 1.0, false);
    let mut engine = Engine::new(
        ThreeWay,
        Box::new(Enc),
        Box::new(Zero),
        policy,
        learner,
        EngineParams {
            n_games: 2,
            seed: 3,
        },
    );
    let (records, stats) = engine.collect(9, |_obs: Vec<f32>, n: usize| vec![0.0; n * 2]);
    assert!(records.len() >= 9);
    assert!(stats.decisions > 0 && !stats.episodes.is_empty());
}

#[test]
fn alphazero_engine_collects_on_a_three_agent_sequential_game() {
    // Sequential N>2 runs Max^N under PUCT: every leaf is evaluated from all three perspectives
    // through the same pooled forward.
    let az_cfg = AlphaZeroConfig {
        num_simulations: 8,
        c_puct: 1.5,
        gamma: 0.99,
        max_depth: 6,
        noise_epsilon: 0.0,
        noise_alpha: 0.3,
        temperature: 0.0,
        temperature_drop: 0,
        chance: ChanceMode::Committed { samples: 1 },
        noise_scope: reinfors_core::NoiseScope::Requester,
    };
    let policy = AlphaZero::new(az_cfg);
    let learner = reinfors_core::AlphaZeroLearner::new(0.99);
    let mut engine = Engine::new(
        RoundRobin,
        Box::new(Enc),
        Box::new(Zero),
        policy,
        learner,
        EngineParams {
            n_games: 2,
            seed: 5,
        },
    );
    let (records, stats) = engine.collect(9, |_obs: Vec<f32>, n: usize| vec![0.0; n * 3]);
    assert!(records.len() >= 9);
    // Every self-play state yields one row per perspective: the mover's (weight 1, real pi) and
    // both non-movers' (weight 0, inert zero pi) — Max^N's per-perspective values supervised.
    let movers = records.iter().filter(|r| r.3 == 1.0).count();
    let value_only = records.iter().filter(|r| r.3 == 0.0).count();
    assert_eq!(movers + value_only, records.len());
    assert_eq!(value_only, 2 * movers);
    for (obs, pi, _z, w) in &records {
        assert_eq!(obs.len(), 2);
        let pi_sum: f64 = pi.iter().sum();
        if *w == 1.0 {
            assert!((pi_sum - 1.0).abs() < 1e-9);
        } else {
            assert_eq!(pi_sum, 0.0);
        }
    }
    assert!(stats.decisions > 0 && !stats.episodes.is_empty());
}

/// RoundRobin with a terminal payoff vector [1, 2, 3]: with gamma 1, EVERY record's z for agent
/// i is exactly i+1 — per-perspective returns from each agent's own reward stream, value-only
/// rows included (their rewards land on the correct tick, not the previous decision).
struct PayoutSeq;

impl Game for PayoutSeq {
    type State = St;
    type Event = f64;
    fn num_agents(&self) -> usize {
        3
    }
    fn action_count(&self) -> usize {
        2
    }
    fn actor(&self, s: &St) -> Actor {
        Actor::Agent(s.tick % 3)
    }
    fn legal_actions(&self, s: &St, agent: usize) -> Vec<usize> {
        if agent == s.tick % 3 && s.tick < 3 {
            vec![0, 1]
        } else {
            Vec::new()
        }
    }
    fn step(&self, s: &St, _actions: &[usize]) -> Transition<St, f64> {
        let terminal = s.tick + 1 >= 3;
        Transition {
            next_state: St { tick: s.tick + 1 },
            events: if terminal {
                vec![1.0, 2.0, 3.0]
            } else {
                vec![0.0; 3]
            },
            terminal,
        }
    }
    fn initial_state(&self, _rng: &mut dyn Rng) -> St {
        St { tick: 0 }
    }
}

struct PayoutReward;
impl Reward for PayoutReward {
    type Event = f64;
    fn step_reward(&self, e: &f64, _agent: usize) -> f64 {
        *e
    }
}

#[test]
fn value_only_rows_carry_each_agents_own_return() {
    let az_cfg = AlphaZeroConfig {
        num_simulations: 8,
        c_puct: 1.5,
        gamma: 1.0,
        max_depth: 6,
        noise_epsilon: 0.0,
        noise_alpha: 0.3,
        temperature: 0.0,
        temperature_drop: 0,
        chance: ChanceMode::Committed { samples: 1 },
        noise_scope: reinfors_core::NoiseScope::Requester,
    };
    let mut engine = Engine::new(
        PayoutSeq,
        Box::new(Enc),
        Box::new(PayoutReward),
        AlphaZero::new(az_cfg),
        reinfors_core::AlphaZeroLearner::new(1.0),
        EngineParams {
            n_games: 1,
            seed: 2,
        },
    );
    let (records, _) = engine.collect(9, |_obs: Vec<f32>, n: usize| vec![0.0; n * 3]);
    assert!(records.len() >= 9);
    for (obs, _pi, z, _w) in &records {
        let agent = obs[1] as f64; // Enc encodes [tick, agent]
        assert!(
            (z - (agent + 1.0)).abs() < 1e-12,
            "agent {agent}'s z must be its own payoff, got {z}"
        );
    }
}

#[test]
fn a_capability_free_policy_collects_on_a_three_agent_game() {
    // The engine rollout is agent-count-generic; with the gate expressed per policy, a cap-free
    // policy (per-agent greedy Q) drives a 3-agent game end to end.
    let policy = EpsilonGreedyQ::new(1, 0.0);
    let learner = Dqn::new(1, 1.0);
    let mut engine = Engine::new(
        ThreeWay,
        Box::new(Enc),
        Box::new(Zero),
        policy,
        learner,
        EngineParams {
            n_games: 2,
            seed: 7,
        },
    );
    let (records, stats) = engine.collect(12, |_obs: Vec<f32>, n: usize| vec![0.0; n * 2]);
    assert!(records.len() >= 12);
    assert!(stats.decisions > 0 && !stats.episodes.is_empty());
}

/// RoundRobin that never terminates on its own — the engine's truncation horizon ends episodes.
struct EndlessRR;

impl Game for EndlessRR {
    type State = St;
    type Event = ();
    fn num_agents(&self) -> usize {
        3
    }
    fn action_count(&self) -> usize {
        2
    }
    fn actor(&self, s: &St) -> Actor {
        Actor::Agent(s.tick % 3)
    }
    fn legal_actions(&self, s: &St, agent: usize) -> Vec<usize> {
        if agent == s.tick % 3 {
            vec![0, 1]
        } else {
            Vec::new()
        }
    }
    fn step(&self, s: &St, _actions: &[usize]) -> Transition<St, ()> {
        Transition {
            next_state: St { tick: s.tick + 1 },
            events: vec![(); 3],
            terminal: false,
        }
    }
    fn initial_state(&self, _rng: &mut dyn Rng) -> St {
        St { tick: 0 }
    }
    fn truncation_horizon(&self) -> Option<usize> {
        Some(2)
    }
}

#[test]
fn truncation_bootstraps_every_perspectives_own_tail() {
    // The net returns a DISTINCT value per encoded agent ((agent+1)/10). With zero rewards and
    // gamma 1, every record's z must equal its own agent's value of the truncated final state —
    // non-mover value-only trajectories included (they were previously seeded with zero because
    // only the final mover counted as active).
    let az_cfg = AlphaZeroConfig {
        num_simulations: 8,
        c_puct: 1.5,
        gamma: 1.0,
        max_depth: 6,
        noise_epsilon: 0.0,
        noise_alpha: 0.3,
        temperature: 0.0,
        temperature_drop: 0,
        chance: ChanceMode::Committed { samples: 1 },
        noise_scope: reinfors_core::NoiseScope::Requester,
    };
    let mut engine = Engine::new(
        EndlessRR,
        Box::new(Enc),
        Box::new(Zero),
        AlphaZero::new(az_cfg),
        reinfors_core::AlphaZeroLearner::new(1.0),
        EngineParams {
            n_games: 1,
            seed: 6,
        },
    );
    let infer = |obs: Vec<f32>, n: usize| {
        let mut out = Vec::with_capacity(n * 3);
        for r in 0..n {
            let agent = f64::from(obs[r * 2 + 1]);
            out.extend([0.0, 0.0, (agent + 1.0) / 10.0]); // 2 logits + the per-agent value
        }
        out
    };
    let (records, stats) = engine.collect(6, infer);
    assert!(records.len() >= 6);
    assert_eq!(
        stats.episodes[0].length, 2,
        "the horizon truncates at tick 2"
    );
    for (obs, _pi, z, _w) in &records {
        let expect = (f64::from(obs[1]) + 1.0) / 10.0;
        assert!(
            (z - expect).abs() < 1e-12,
            "agent {}'s z must bootstrap from ITS tail: got {z}, want {expect}",
            obs[1]
        );
    }
}
