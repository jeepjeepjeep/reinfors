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
            events: vec![None; 3],
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
    // Expectimax: single-perspective search, agent-count-free under either dynamics.
    assert_eq!(
        SelectiveExpectimax::new(search_cfg(), 2, 0.0).max_agents(true),
        None
    );
    assert_eq!(
        SelectiveExpectimax::new(search_cfg(), 2, 0.0).max_agents(false),
        None
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
        sequential_backup: Default::default(),
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
            events: vec![None; 3],
            terminal: s.tick + 1 >= 3,
        }
    }
    fn initial_state(&self, _rng: &mut dyn Rng) -> St {
        St { tick: 0 }
    }
}

/// Every built-in policy is agent-count-free now, so the construction gate is pinned with a stub
/// that still declares a cap (future capped policies keep this seam honest).
struct CappedStub;
impl Policy for CappedStub {
    type Evaluation = ();
    type PolicyState = ();
    fn max_agents(&self, _sequential: bool) -> Option<usize> {
        Some(2)
    }
    fn supports_chance_nodes(&self) -> bool {
        true
    }

    fn supports_imperfect_information(&self) -> bool {
        false
    }
    fn begin_episode(&self, _rng: &mut dyn Rng) {}
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
    fn evaluate<G, F>(
        &self,
        _game: &G,
        _enc: &dyn StateEncoder<State = G::State>,
        _reward: &dyn reinfors_core::Reward<Event = G::Event>,
        _requests: Vec<(G::State, usize)>,
        _seed: u64,
        _collect_interior: bool,
        _eval: &mut Evaluator<'_, F>,
    ) -> Vec<()>
    where
        G: Game + Sync,
        G::State: Send,
        F: FnMut(usize, Vec<f32>, usize) -> Vec<f64>,
    {
        unimplemented!("construction panics before any evaluation")
    }
    fn select(&self, _eval: &(), _state: &mut (), _rng: &mut dyn Rng) -> usize {
        unimplemented!()
    }
    fn fold_telemetry(&self, _eval: &(), _stats: &mut reinfors_core::CollectStats) {}
}

struct StubLearner;
impl reinfors_core::Learner<()> for StubLearner {
    type Record = ();
    fn eval_records(
        &self,
        _evaluation: &mut (),
        _view: &dyn reinfors_core::ActionView,
        _agent: usize,
        _rng: &mut dyn Rng,
    ) -> Vec<()> {
        unimplemented!()
    }
    fn episode_records(
        &self,
        _trajectory: &[reinfors_core::Step<()>],
        _tail: &[f64],
        _view: &dyn reinfors_core::ActionView,
        _agent: usize,
        _rng: &mut dyn Rng,
    ) -> Vec<()> {
        unimplemented!()
    }
}

#[test]
#[should_panic(expected = "at most 2 agents")]
fn engine_rejects_a_capped_policy_on_a_three_agent_game() {
    let _ = Engine::new(
        ThreeWay,
        Box::new(Enc),
        Box::new(Zero),
        CappedStub,
        StubLearner,
        EngineParams {
            n_games: 1,
            seed: 0,
        },
    );
}

#[test]
fn expectimax_engine_collects_on_a_three_agent_simultaneous_game() {
    // Factored co-mover expectimax end to end: the searcher's MAX edges fan over both co-movers'
    // joint (uniform here), TreeStrap consumes the evaluations.
    let policy = SelectiveExpectimax::new(search_cfg(), 1, 0.0);
    let learner = TreeStrap::new(0.99, 0.3, 1.0, false);
    let mut engine = Engine::new(
        ThreeWay,
        Box::new(Enc),
        Box::new(Zero),
        policy,
        learner,
        EngineParams {
            n_games: 2,
            seed: 4,
        },
    );
    let (records, stats) = engine.collect(9, |_obs: Vec<f32>, n: usize| vec![0.0; n * 2]);
    assert!(records.len() >= 9);
    assert!(stats.decisions > 0 && !stats.episodes.is_empty());
}

#[test]
fn expectimax_searches_a_three_agent_game() {
    let mut infer = |_players: &[usize], _obs: Vec<f32>, n: usize| vec![0.0; n * 4]; // K=2 heads x A=2
    let results = search_many(
        &ThreeWay,
        &Enc,
        &Zero,
        &search_cfg(),
        vec![(St { tick: 0 }, 0)],
        false,
        0,
        &mut infer,
    );
    let (values, _, stats) = &results[0];
    assert_eq!(values.len(), 2);
    assert!(values.iter().all(|row| row.len() == 2));
    assert!(stats.expansions > 0);
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
    let mut infer = |_p: usize, _obs: Vec<f32>, n: usize| vec![0.0; n * 2];
    let mut eval = Evaluator::new(&mut infer, reinfors_core::InferMode::Shared, None);
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
    let mut infer = |_p: usize, _obs: Vec<f32>, n: usize| vec![0.0; n * 2];
    let mut eval = Evaluator::new(&mut infer, reinfors_core::InferMode::Shared, None);
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

fn az_collect_cfg() -> AlphaZeroConfig {
    AlphaZeroConfig {
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
        sequential_backup: Default::default(),
    }
}

#[test]
fn alphazero_engine_collects_on_a_three_agent_sequential_game() {
    // Sequential N>2 runs Max^N under PUCT: every leaf is evaluated from all three perspectives
    // through the same pooled forward.
    let policy = AlphaZero::new(az_collect_cfg());
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
    for (obs, pi, _z, w, _player) in &records {
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

#[test]
fn learn_players_filters_value_only_perspectives() {
    // Frozen players must not leak through the sequential-Max^N value-only path either: with
    // only player 0 learning, each RoundRobin episode leaves its mover record (ply 0) plus its
    // two value-only perspectives (plies 1 and 2) — nothing from players 1 and 2.
    let mut engine = Engine::new(
        RoundRobin,
        Box::new(Enc),
        Box::new(Zero),
        AlphaZero::new(az_collect_cfg()),
        reinfors_core::AlphaZeroLearner::new(0.99),
        EngineParams {
            n_games: 2,
            seed: 5,
        },
    )
    .with_learn_players(&[0]);
    let (records, _) = engine.collect(6, |_obs: Vec<f32>, n: usize| vec![0.0; n * 3]);
    let movers = records.iter().filter(|r| r.3 == 1.0).count();
    let value_only = records.iter().filter(|r| r.3 == 0.0).count();
    assert!(movers >= 2);
    assert_eq!(movers + value_only, records.len());
    assert_eq!(value_only, movers * 2);
    // Enc writes the encoded-for agent into obs[1]: the unfiltered output would also have a 2:1
    // value-only ratio, but its records would carry all three perspectives, not only player 0's.
    for r in &records {
        assert_eq!(r.0[1], 0.0, "every record is player 0's perspective");
    }
}

/// Two players alternating for four plies; free actions, zero reward.
struct TwoTurn;

impl Game for TwoTurn {
    type State = St;
    type Event = ();
    fn num_agents(&self) -> usize {
        2
    }
    fn action_count(&self) -> usize {
        2
    }
    fn actor(&self, s: &St) -> Actor {
        Actor::Agent(s.tick % 2)
    }
    fn legal_actions(&self, s: &St, agent: usize) -> Vec<usize> {
        if agent == s.tick % 2 && s.tick < 4 {
            vec![0, 1]
        } else {
            Vec::new()
        }
    }
    fn step(&self, s: &St, _actions: &[usize]) -> Transition<St, ()> {
        Transition {
            next_state: St { tick: s.tick + 1 },
            events: vec![None; 2],
            terminal: s.tick + 1 >= 4,
        }
    }
    fn initial_state(&self, _rng: &mut dyn Rng) -> St {
        St { tick: 0 }
    }
}

/// `Enc` writes the encoded-for agent into `obs[1]`, so "net `p` only ever receives observations
/// encoded for perspective `p`" pins every pooled row's routing — leaf, value-only,
/// opponent-model, and bootstrap rows alike.
fn assert_rows_encoded_for(p: usize, obs: &[f32]) {
    for row in obs.chunks(2) {
        assert_eq!(
            row[1] as usize, p,
            "net {p} received an obs encoded for agent {}",
            row[1]
        );
    }
}

#[test]
fn alphazero_routes_each_perspective_to_its_own_network() {
    let mut engine = Engine::new(
        RoundRobin,
        Box::new(Enc),
        Box::new(Zero),
        AlphaZero::new(az_collect_cfg()),
        reinfors_core::AlphaZeroLearner::new(0.99),
        EngineParams {
            n_games: 2,
            seed: 5,
        },
    );
    let mut seen = std::collections::HashSet::new();
    let (records, _) =
        engine.collect_routed(9, reinfors_core::InferMode::PerPlayer, |p, obs, n| {
            assert_rows_encoded_for(p, &obs);
            seen.insert(p);
            vec![0.0; n * 3]
        });
    assert!(records.len() >= 9);
    assert_eq!(
        seen,
        (0..3).collect(),
        "sequential Max^N requests all three perspectives"
    );
}

#[test]
fn uct_routes_sequential_leaf_rows_to_the_leaf_mover() {
    let mut engine = Engine::new(
        TwoTurn,
        Box::new(Enc),
        Box::new(Zero),
        Mcts::new(mcts_cfg(), ActBy::Value),
        TreeStrap::new(0.99, 0.3, 1.0, false),
        EngineParams {
            n_games: 2,
            seed: 3,
        },
    );
    let mut seen = std::collections::HashSet::new();
    let (records, _) =
        engine.collect_routed(6, reinfors_core::InferMode::PerPlayer, |p, obs, n| {
            assert_rows_encoded_for(p, &obs);
            seen.insert(p);
            vec![0.0; n * 2]
        });
    assert!(records.len() >= 6);
    assert_eq!(
        seen,
        (0..2).collect(),
        "negamax leaves alternate between both movers' networks"
    );
}

#[test]
fn duct_routes_every_perspective_to_its_own_network() {
    let mut engine = Engine::new(
        ThreeWay,
        Box::new(Enc),
        Box::new(Zero),
        Mcts::new(mcts_cfg(), ActBy::Value),
        TreeStrap::new(0.99, 0.3, 1.0, false),
        EngineParams {
            n_games: 2,
            seed: 3,
        },
    );
    let mut seen = std::collections::HashSet::new();
    let (records, _) =
        engine.collect_routed(9, reinfors_core::InferMode::PerPlayer, |p, obs, n| {
            assert_rows_encoded_for(p, &obs);
            seen.insert(p);
            vec![0.0; n * 2]
        });
    assert!(records.len() >= 9);
    assert_eq!(
        seen,
        (0..3).collect(),
        "DUCT requests every co-mover's perspective"
    );
}

#[test]
fn expectimax_routes_opponent_model_rows_to_that_mover() {
    // Distributional opponent: opponent nodes evaluate the MOVER's own observation through the
    // mover's network; requester leaves stay on the requester's.
    let cfg = SearchConfig {
        opponent: Opponent::Distributional {
            temperature: 1.0,
            floor: 0.1,
        },
        ..search_cfg()
    };
    let mut engine = Engine::new(
        TwoTurn,
        Box::new(Enc),
        Box::new(Zero),
        SelectiveExpectimax::new(cfg, 1, 0.0),
        TreeStrap::new(0.99, 0.3, 1.0, false),
        EngineParams {
            n_games: 2,
            seed: 4,
        },
    );
    let mut seen = std::collections::HashSet::new();
    let (records, _) =
        engine.collect_routed(6, reinfors_core::InferMode::PerPlayer, |p, obs, n| {
            assert_rows_encoded_for(p, &obs);
            seen.insert(p);
            vec![0.0; n * 2]
        });
    assert!(records.len() >= 6);
    assert_eq!(
        seen,
        (0..2).collect(),
        "opponent-model rows reach the opponent's network"
    );
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
                vec![Some(1.0), Some(2.0), Some(3.0)]
            } else {
                vec![Some(0.0); 3]
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
        sequential_backup: Default::default(),
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
    for (obs, _pi, z, _w, _player) in &records {
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
            events: vec![None; 3],
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
        sequential_backup: Default::default(),
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
    for (obs, _pi, z, _w, _player) in &records {
        let expect = (f64::from(obs[1]) + 1.0) / 10.0;
        assert!(
            (z - expect).abs() < 1e-12,
            "agent {}'s z must bootstrap from ITS tail: got {z}, want {expect}",
            obs[1]
        );
    }
}

/// A simultaneous 2-agent game whose every transition declares a combinatorial Uniform chance —
/// the shape the compact declaration exists for.
struct WideChance;
#[derive(Clone)]
struct WSt(usize);
impl Game for WideChance {
    type State = WSt;
    type Event = ();
    fn num_agents(&self) -> usize {
        2
    }
    fn action_count(&self) -> usize {
        2
    }
    fn actor(&self, _s: &WSt) -> Actor {
        Actor::Simultaneous
    }
    fn legal_actions(&self, s: &WSt, _agent: usize) -> Vec<usize> {
        if s.0 >= 3 {
            Vec::new()
        } else {
            vec![0, 1]
        }
    }
    fn step(&self, s: &WSt, _actions: &[usize]) -> Transition<WSt, ()> {
        Transition {
            next_state: WSt(s.0 + 1),
            events: vec![None; 2],
            terminal: s.0 + 1 >= 3,
        }
    }
    fn chance_outcomes(
        &self,
        _s: &WSt,
        t: &Transition<WSt, ()>,
    ) -> Option<reinfors_core::ChanceDist> {
        (!t.terminal).then_some(reinfors_core::ChanceDist::Uniform(50_000_000))
    }
    fn apply_chance(&self, _s: &WSt, t: &Transition<WSt, ()>, _outcome: usize) -> WSt {
        t.next_state.clone()
    }
    fn initial_state(&self, _rng: &mut dyn Rng) -> WSt {
        WSt(0)
    }
}

struct WEnc;
impl reinfors_core::ActionView for WEnc {}
impl StateEncoder for WEnc {
    type State = WSt;
    fn encode(&self, s: &WSt, agent: usize) -> Vec<f32> {
        vec![s.0 as f32, agent as f32]
    }
    fn obs_shape(&self) -> (usize, usize, usize) {
        (1, 1, 2)
    }
}

#[test]
fn sampling_modes_traverse_combinatorial_uniform_chance() {
    // Committed and AlwaysResample draw single indices from a 5e7-outcome space — no vector, no
    // dense child array (AlwaysResample materializes children sparsely per distinct outcome).
    for chance in [
        ChanceMode::Committed { samples: 2 },
        ChanceMode::AlwaysResample,
    ] {
        let cfg = MctsConfig {
            num_simulations: 32,
            uct_c: 1.0,
            gamma: 0.99,
            max_depth: 6,
            temperature: 0.0,
            temperature_drop: 0,
            chance,
        };
        let mut infer = |_p: usize, _obs: Vec<f32>, n: usize| vec![0.0; n * 2];
        let mut eval = Evaluator::new(&mut infer, reinfors_core::InferMode::Shared, None);
        let evals = mcts_many(
            &WideChance,
            &WEnc,
            &Zero2,
            &cfg,
            vec![(WSt(0), 0)],
            0,
            &mut eval,
        );
        assert_eq!(evals[0].visits.iter().sum::<f64>() as usize, 32);
    }
}

struct Zero2;
impl Reward for Zero2 {
    type Event = ();
    fn step_reward(&self, _e: &(), _agent: usize) -> f64 {
        0.0
    }
}

#[test]
#[should_panic(expected = "ExpandAll cannot enumerate")]
fn expand_all_rejects_combinatorial_outcome_spaces() {
    let cfg = MctsConfig {
        num_simulations: 8,
        uct_c: 1.0,
        gamma: 0.99,
        max_depth: 6,
        temperature: 0.0,
        temperature_drop: 0,
        chance: ChanceMode::ExpandAll,
    };
    let mut infer = |_p: usize, _obs: Vec<f32>, n: usize| vec![0.0; n * 2];
    let mut eval = Evaluator::new(&mut infer, reinfors_core::InferMode::Shared, None);
    let _ = mcts_many(
        &WideChance,
        &WEnc,
        &Zero2,
        &cfg,
        vec![(WSt(0), 0)],
        0,
        &mut eval,
    );
}

#[test]
fn uniform_draws_cover_the_index_space() {
    use reinfors_core::ChanceDist;
    struct R(u64);
    impl Rng for R {
        fn below(&mut self, n: usize) -> usize {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
            (self.0 >> 33) as usize % n.max(1)
        }
        fn unit(&mut self) -> f64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }
    }
    let dist = ChanceDist::Uniform(37);
    let mut rng = R(9);
    let mut hits = [0usize; 37];
    for _ in 0..37_00 {
        let i = dist.draw(&mut rng);
        assert!(i < 37);
        hits[i] += 1;
    }
    assert!(
        hits.iter().all(|&h| h > 0),
        "every index reachable: {hits:?}"
    );
    // Loose uniformity: no bucket more than 3x the mean.
    assert!(hits.iter().all(|&h| h < 300), "roughly uniform: {hits:?}");
}

/// `TwoRobin` with hidden information declared — the search entries must reject it directly,
/// not only via engine construction (a direct caller would otherwise get clairvoyant values).
struct HiddenTwo;

impl Game for HiddenTwo {
    type State = St;
    type Event = ();
    fn num_agents(&self) -> usize {
        2
    }
    fn action_count(&self) -> usize {
        2
    }
    fn perfect_information(&self) -> bool {
        false
    }
    fn actor(&self, s: &St) -> Actor {
        Actor::Agent(s.tick % 2)
    }
    fn legal_actions(&self, s: &St, agent: usize) -> Vec<usize> {
        if agent == s.tick % 2 && s.tick < 4 {
            vec![0, 1]
        } else {
            Vec::new()
        }
    }
    fn step(&self, s: &St, _actions: &[usize]) -> Transition<St, ()> {
        Transition {
            next_state: St { tick: s.tick + 1 },
            events: vec![None; 2],
            terminal: s.tick + 1 >= 4,
        }
    }
    fn initial_state(&self, _rng: &mut dyn Rng) -> St {
        St { tick: 0 }
    }
}

/// `TwoRobin` declaring chance NODES — outcome-dependent payouts the searches cannot score.
struct NodeyTwo;

impl Game for NodeyTwo {
    type State = St;
    type Event = ();
    fn num_agents(&self) -> usize {
        2
    }
    fn action_count(&self) -> usize {
        2
    }
    fn chance_nodes(&self) -> bool {
        true
    }
    fn actor(&self, s: &St) -> Actor {
        Actor::Agent(s.tick % 2)
    }
    fn legal_actions(&self, s: &St, agent: usize) -> Vec<usize> {
        if agent == s.tick % 2 && s.tick < 4 {
            vec![0, 1]
        } else {
            Vec::new()
        }
    }
    fn step(&self, s: &St, _actions: &[usize]) -> Transition<St, ()> {
        Transition {
            next_state: St { tick: s.tick + 1 },
            events: vec![None; 2],
            terminal: s.tick + 1 >= 4,
        }
    }
    fn initial_state(&self, _rng: &mut dyn Rng) -> St {
        St { tick: 0 }
    }
}

#[test]
#[should_panic(expected = "clairvoyant")]
fn direct_mcts_rejects_hidden_information() {
    let mut infer = |_p: usize, _obs: Vec<f32>, n: usize| vec![0.0; n * 2];
    let mut eval = Evaluator::new(&mut infer, reinfors_core::InferMode::Shared, None);
    let _ = mcts_many(
        &HiddenTwo,
        &Enc,
        &Zero,
        &mcts_cfg(),
        vec![(St { tick: 0 }, 0)],
        0,
        &mut eval,
    );
}

#[test]
#[should_panic(expected = "clairvoyant")]
fn direct_expectimax_rejects_hidden_information() {
    let _ = search_many(
        &HiddenTwo,
        &Enc,
        &Zero,
        &search_cfg(),
        vec![(St { tick: 0 }, 0)],
        false,
        0,
        |_players: &[usize], _obs: Vec<f32>, n: usize| vec![0.0; n * 2],
    );
}

#[test]
#[should_panic(expected = "chance-node")]
fn direct_mcts_rejects_chance_node_games() {
    let mut infer = |_p: usize, _obs: Vec<f32>, n: usize| vec![0.0; n * 2];
    let mut eval = Evaluator::new(&mut infer, reinfors_core::InferMode::Shared, None);
    let _ = mcts_many(
        &NodeyTwo,
        &Enc,
        &Zero,
        &mcts_cfg(),
        vec![(St { tick: 0 }, 0)],
        0,
        &mut eval,
    );
}

#[test]
#[should_panic(expected = "chance-node")]
fn direct_expectimax_rejects_chance_node_games() {
    let _ = search_many(
        &NodeyTwo,
        &Enc,
        &Zero,
        &search_cfg(),
        vec![(St { tick: 0 }, 0)],
        false,
        0,
        |_players: &[usize], _obs: Vec<f32>, n: usize| vec![0.0; n * 2],
    );
}

#[test]
#[should_panic(expected = "chance-node")]
fn engine_rejects_search_policies_on_chance_node_games() {
    let _ = Engine::new(
        NodeyTwo,
        Box::new(Enc),
        Box::new(Zero),
        Mcts::new(mcts_cfg(), ActBy::Value),
        TreeStrap::new(0.99, 0.3, 1.0, false),
        EngineParams {
            n_games: 1,
            seed: 0,
        },
    );
}

/// `NodeyTwo` whose root IS the chance node — a declared deal, realized at episode birth by
/// the chain machinery (root chance is first-class; see `Game::all_chance_declared`).
struct RootNodey;

impl Game for RootNodey {
    type State = St;
    type Event = ();
    fn num_agents(&self) -> usize {
        2
    }
    fn action_count(&self) -> usize {
        2
    }
    fn all_chance_declared(&self) -> bool {
        true // the root draw below is this game's only randomness
    }
    fn actor(&self, s: &St) -> Actor {
        if s.tick == 0 {
            Actor::Chance
        } else {
            Actor::Agent(s.tick % 2)
        }
    }
    fn chance_node(&self, _s: &St) -> reinfors_core::ChanceDist {
        reinfors_core::ChanceDist::Uniform(2)
    }
    fn apply_chance_node(&self, s: &St, _outcome: usize) -> Transition<St, ()> {
        Transition {
            next_state: St { tick: s.tick + 1 },
            events: vec![None; 2],
            terminal: false,
        }
    }
    fn legal_actions(&self, s: &St, agent: usize) -> Vec<usize> {
        if s.tick > 0 && agent == s.tick % 2 && s.tick < 4 {
            vec![0, 1]
        } else {
            Vec::new()
        }
    }
    fn step(&self, s: &St, _actions: &[usize]) -> Transition<St, ()> {
        Transition {
            next_state: St { tick: s.tick + 1 },
            events: vec![None; 2],
            terminal: s.tick + 1 >= 4,
        }
    }
    fn initial_state(&self, _rng: &mut dyn Rng) -> St {
        St { tick: 0 }
    }
}

#[test]
fn realized_roots_expose_the_true_decision_dynamics() {
    // The raw root of a declared-deal game is Actor::Chance — probing it directly would
    // misclassify the game as simultaneous. The canonical realization helper answers with the
    // first DECISION state, where turn-taking is visible (RootNodey is sequential).
    struct P(u64);
    impl Rng for P {
        fn below(&mut self, n: usize) -> usize {
            self.0 = self.0.wrapping_mul(48271) % 0x7FFF_FFFF;
            self.0 as usize % n.max(1)
        }
        fn unit(&mut self) -> f64 {
            self.below(1 << 20) as f64 / (1 << 20) as f64
        }
    }
    assert!(matches!(
        Game::actor(&RootNodey, &RootNodey.initial_state(&mut P(3))),
        Actor::Chance
    ));
    let realized = reinfors_core::realize_initial_state(&RootNodey, &mut P(3));
    assert!(matches!(
        Game::actor(&RootNodey, &realized),
        Actor::Agent(_)
    ));
}

#[test]
fn episode_birth_realizes_root_chance_chains() {
    // The root chance node (a declared deal) is realized at birth: construction succeeds,
    // every episode starts at the post-deal decision state, and collect proceeds normally.
    // Note RootNodey::chance_nodes() stays FALSE — the capability describes POST-birth
    // states, so a root-only-chance game keeps its tree-search compatibility.
    assert!(!Game::chance_nodes(&RootNodey));
    let mut engine = Engine::new(
        RootNodey,
        Box::new(Enc),
        Box::new(Zero),
        EpsilonGreedyQ::new(2, 0.1),
        Dqn::new(2, 1.0),
        EngineParams {
            n_games: 2,
            seed: 0,
        },
    );
    let mut infer = |_obs: Vec<f32>, n: usize| vec![0.0; n * 2 * 2];
    let (records, _stats) = engine.collect(32, &mut infer);
    assert!(!records.is_empty(), "root-chance games collect normally");
}

#[test]
#[should_panic(expected = "cannot start at a chance node")]
fn start_distribution_restores_must_be_decision_states() {
    // A game with an INTERIOR chance node (tick 5, unreachable in play): construction passes on
    // the tick-0 decision state, then a hostile start distribution restores the chance node —
    // the restore path must hold the same decision-state start contract as `initial_state`.
    struct MidNodey;
    impl Game for MidNodey {
        type State = St;
        type Event = ();
        fn num_agents(&self) -> usize {
            2
        }
        fn action_count(&self) -> usize {
            2
        }
        fn chance_nodes(&self) -> bool {
            true
        }
        fn actor(&self, s: &St) -> Actor {
            if s.tick >= 5 {
                Actor::Chance
            } else {
                Actor::Agent(s.tick % 2)
            }
        }
        fn legal_actions(&self, s: &St, agent: usize) -> Vec<usize> {
            if s.tick < 4 && agent == s.tick % 2 {
                vec![0, 1]
            } else {
                Vec::new()
            }
        }
        fn step(&self, s: &St, _actions: &[usize]) -> Transition<St, ()> {
            Transition {
                next_state: St { tick: s.tick + 1 },
                events: vec![None; 2],
                terminal: s.tick + 1 >= 4,
            }
        }
        fn initial_state(&self, _rng: &mut dyn Rng) -> St {
            St { tick: 0 }
        }
    }
    struct ChanceRestore;
    impl reinfors_core::StartDistribution<St> for ChanceRestore {
        fn choose(&mut self, _rng: &mut dyn Rng) -> reinfors_core::Start<St> {
            reinfors_core::Start::Restore(St { tick: 5 })
        }
    }
    let mut engine = Engine::new(
        MidNodey,
        Box::new(Enc),
        Box::new(Zero),
        EpsilonGreedyQ::new(2, 0.1),
        Dqn::new(2, 1.0),
        EngineParams {
            n_games: 1,
            seed: 0,
        },
    )
    .with_start_distribution(Box::new(ChanceRestore));
    let mut infer = |_obs: Vec<f32>, n: usize| vec![0.0; n * 2 * 2];
    for _ in 0..8 {
        let _ = engine.collect(64, &mut infer);
    }
}

/// Two agents taking turns; terminal after four plies with payoffs [1, 2].
struct TwoRobin;

impl Game for TwoRobin {
    type State = St;
    type Event = f64;
    fn num_agents(&self) -> usize {
        2
    }
    fn action_count(&self) -> usize {
        2
    }
    fn actor(&self, s: &St) -> Actor {
        Actor::Agent(s.tick % 2)
    }
    fn legal_actions(&self, s: &St, agent: usize) -> Vec<usize> {
        if agent == s.tick % 2 && s.tick < 4 {
            vec![0, 1]
        } else {
            Vec::new()
        }
    }
    fn step(&self, s: &St, _actions: &[usize]) -> Transition<St, f64> {
        let terminal = s.tick + 1 >= 4;
        Transition {
            next_state: St { tick: s.tick + 1 },
            events: if terminal {
                vec![Some(1.0), Some(2.0)]
            } else {
                vec![Some(0.0); 2]
            },
            terminal,
        }
    }
    fn initial_state(&self, _rng: &mut dyn Rng) -> St {
        St { tick: 0 }
    }
}

#[test]
fn forced_maxn_supervises_both_perspectives_at_two_agents() {
    // The negamax-deletion measurement seam: SequentialBackup::MaxN at 2 agents runs the vector
    // backup AND emits value-only rows for the non-mover — supervised ≡ consumed. Auto keeps the
    // mover-only negamax pipeline byte-for-byte.
    let cfg = |backup| AlphaZeroConfig {
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
        sequential_backup: backup,
    };
    let run = |backup| {
        let mut engine = Engine::new(
            TwoRobin,
            Box::new(Enc),
            Box::new(PayoutReward),
            AlphaZero::new(cfg(backup)),
            reinfors_core::AlphaZeroLearner::new(1.0),
            EngineParams {
                n_games: 1,
                seed: 9,
            },
        );
        engine
            .collect(8, |_obs: Vec<f32>, n: usize| vec![0.0; n * 3])
            .0
    };
    let auto = run(reinfors_core::SequentialBackup::Auto);
    assert!(auto.iter().all(|r| r.3 == 1.0), "Auto: mover rows only");
    let maxn = run(reinfors_core::SequentialBackup::MaxN);
    let value_only = maxn.iter().filter(|r| r.3 == 0.0).count();
    let movers = maxn.iter().filter(|r| r.3 == 1.0).count();
    assert_eq!(value_only, movers, "one non-mover row per decision at N=2");
    // gamma 1, rewards only at the end: every row's z is its own agent's payoff.
    for (obs, _pi, z, _w, _player) in &maxn {
        let expect = f64::from(obs[1]) + 1.0;
        assert!(
            (z - expect).abs() < 1e-12,
            "per-perspective z: got {z}, want {expect}"
        );
    }
}

/// Two agents taking turns forever; the engine's horizon truncates at tick 2.
struct EndlessTwo;

impl Game for EndlessTwo {
    type State = St;
    type Event = ();
    fn num_agents(&self) -> usize {
        2
    }
    fn action_count(&self) -> usize {
        2
    }
    fn actor(&self, s: &St) -> Actor {
        Actor::Agent(s.tick % 2)
    }
    fn legal_actions(&self, s: &St, agent: usize) -> Vec<usize> {
        if agent == s.tick % 2 {
            vec![0, 1]
        } else {
            Vec::new()
        }
    }
    fn step(&self, s: &St, _actions: &[usize]) -> Transition<St, ()> {
        Transition {
            next_state: St { tick: s.tick + 1 },
            events: vec![None; 2],
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
fn forced_maxn_truncation_bootstraps_both_perspectives() {
    // sequential_backup="maxn" at TWO agents: the tree consumes both perspectives' leaf values
    // and both hold per-tick trajectories — so a truncation must tail-bootstrap BOTH, not just
    // the currently active agent (the regression: the tail gate hardcoded n > 2, silently
    // seeding the non-mover's z from zero).
    let cfg = AlphaZeroConfig {
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
        sequential_backup: reinfors_core::SequentialBackup::MaxN,
    };
    let mut engine = Engine::new(
        EndlessTwo,
        Box::new(Enc),
        Box::new(Zero),
        AlphaZero::new(cfg),
        reinfors_core::AlphaZeroLearner::new(1.0),
        EngineParams {
            n_games: 1,
            seed: 8,
        },
    );
    // The net returns a distinct value per encoded agent ((agent+1)/10); zero rewards, gamma 1:
    // every record's z must equal ITS OWN agent's tail value.
    let infer = |obs: Vec<f32>, n: usize| {
        let mut out = Vec::with_capacity(n * 3);
        for r in 0..n {
            let agent = f64::from(obs[r * 2 + 1]);
            out.extend([0.0, 0.0, (agent + 1.0) / 10.0]);
        }
        out
    };
    let (records, stats) = engine.collect(4, infer);
    assert_eq!(
        stats.episodes[0].length, 2,
        "the horizon truncates at tick 2"
    );
    assert!(
        records.iter().any(|r| r.3 == 0.0),
        "value-only rows present"
    );
    for (obs, _pi, z, _w, _player) in &records {
        let expect = (f64::from(obs[1]) + 1.0) / 10.0;
        assert!(
            (z - expect).abs() < 1e-12,
            "agent {}'s z must bootstrap from ITS tail: got {z}, want {expect}",
            obs[1]
        );
    }
}
