use reinfors_core::{
    mcts_many, search_many, ActBy, Actor, AlphaZero, AlphaZeroConfig, ChanceMode, Dqn, Engine,
    EngineParams, EpsilonGreedyQ, Evaluator, Game, Mcts, MctsConfig, Opponent, Policy, Reward, Rng,
    SearchConfig, SelectiveExpectimax, Space, StateEncoder, Transition, TreeStrap,
};

#[derive(Clone)]
struct St {
    tick: usize,
}

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
    fn initial_state(&self) -> St {
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
    fn initial_state(&self) -> St {
        St { tick: 0 }
    }
}

struct CappedStub;
impl Policy for CappedStub {
    type Evaluation = ();
    type PolicyState = ();
    fn max_agents(&self, _sequential: bool) -> Option<usize> {
        Some(2)
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
            n_groups: 1,
            ..Default::default()
        },
    );
}

#[test]
fn expectimax_engine_collects_on_a_three_agent_simultaneous_game() {
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
            n_groups: 1,
            ..Default::default()
        },
    );
    let (records, stats) = engine.collect(9, |_obs: Vec<f32>, n: usize| vec![0.0; n * 2]);
    assert!(records.len() >= 9);
    assert!(stats.decisions > 0 && !stats.episodes.is_empty());
}

#[test]
fn expectimax_searches_a_three_agent_game() {
    let mut infer = |_players: &[usize], _obs: Vec<f32>, n: usize| vec![0.0; n * 4];
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
#[should_panic(expected = "catalogue/compatibility")]
fn uct_rejects_sequential_three_agent_games() {
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
        assert_eq!(e.visits.iter().sum::<f64>() as usize, 16);
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
            n_groups: 1,
            ..Default::default()
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
    // Sequential N>2 needs PUCT values for every perspective; Q-only UCT has no non-mover leaves.
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
            n_groups: 1,
            ..Default::default()
        },
    );
    let (records, stats) = engine.collect(9, |_obs: Vec<f32>, n: usize| vec![0.0; n * 3]);
    assert!(records.len() >= 9);
    let movers = records.iter().filter(|r| r.3 == 1.0).count();
    let value_only = records.iter().filter(|r| r.3 == 0.0).count();
    assert_eq!(movers + value_only, records.len());
    assert_eq!(value_only, 2 * movers);
    for (obs, pi, _z, w, _player, _legal) in &records {
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
    // Filtering to player 0 should retain its mover row plus two value-only perspectives per tick.
    let mut engine = Engine::new(
        RoundRobin,
        Box::new(Enc),
        Box::new(Zero),
        AlphaZero::new(az_collect_cfg()),
        reinfors_core::AlphaZeroLearner::new(0.99),
        EngineParams {
            n_games: 2,
            seed: 5,
            n_groups: 1,
            ..Default::default()
        },
    )
    .with_learn_players(&[0]);
    let (records, _) = engine.collect(6, |_obs: Vec<f32>, n: usize| vec![0.0; n * 3]);
    let movers = records.iter().filter(|r| r.3 == 1.0).count();
    let value_only = records.iter().filter(|r| r.3 == 0.0).count();
    assert!(movers >= 2);
    assert_eq!(movers + value_only, records.len());
    assert_eq!(value_only, movers * 2);
    // The same 2:1 ratio exists without filtering; only the encoded perspectives below prove that
    // frozen players did not leak into either record path.
    for r in &records {
        assert_eq!(r.0[1], 0.0, "every record is player 0's perspective");
    }
}

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
    fn initial_state(&self) -> St {
        St { tick: 0 }
    }
}

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
            n_groups: 1,
            ..Default::default()
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
            n_groups: 1,
            ..Default::default()
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
            n_groups: 1,
            ..Default::default()
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
            n_groups: 1,
            ..Default::default()
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

/// Each terminal event pays agent i exactly i+1. With gamma 1, this pins event/tick attribution for
/// mover and value-only trajectories to each perspective's own reward stream.
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
    fn initial_state(&self) -> St {
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
            n_groups: 1,
            ..Default::default()
        },
    );
    let (records, _) = engine.collect(9, |_obs: Vec<f32>, n: usize| vec![0.0; n * 3]);
    assert!(records.len() >= 9);
    for (obs, _pi, z, _w, _player, _legal) in &records {
        let agent = obs[1] as f64;
        assert!(
            (z - (agent + 1.0)).abs() < 1e-12,
            "agent {agent}'s z must be its own payoff, got {z}"
        );
    }
}

#[test]
fn a_capability_free_policy_collects_on_a_three_agent_game() {
    let policy = EpsilonGreedyQ::new(1, 0.0);
    let learner = Dqn::new(1, 1.0, 1, 0.99);
    let mut engine = Engine::new(
        ThreeWay,
        Box::new(Enc),
        Box::new(Zero),
        policy,
        learner,
        EngineParams {
            n_games: 2,
            seed: 7,
            n_groups: 1,
            ..Default::default()
        },
    );
    let (records, stats) = engine.collect(12, |_obs: Vec<f32>, n: usize| vec![0.0; n * 2]);
    assert!(records.len() >= 12);
    assert!(stats.decisions > 0 && !stats.episodes.is_empty());
}

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
    fn initial_state(&self) -> St {
        St { tick: 0 }
    }
    fn truncation_horizon(&self) -> Option<usize> {
        Some(2)
    }
}

#[test]
fn truncation_bootstraps_every_perspectives_own_tail() {
    // Rewards are zero and gamma is one, so each record must equal its encoded agent's distinct tail.
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
            n_groups: 1,
            ..Default::default()
        },
    );
    let infer = |obs: Vec<f32>, n: usize| {
        let mut out = Vec::with_capacity(n * 3);
        for r in 0..n {
            let agent = f64::from(obs[r * 2 + 1]);
            out.extend([0.0, 0.0, (agent + 1.0) / 10.0]);
        }
        out
    };
    let (records, stats) = engine.collect(6, infer);
    assert!(records.len() >= 6);
    assert_eq!(
        stats.episodes[0].length, 2,
        "the horizon truncates at tick 2"
    );
    for (obs, _pi, z, _w, _player, _legal) in &records {
        let expect = (f64::from(obs[1]) + 1.0) / 10.0;
        assert!(
            (z - expect).abs() < 1e-12,
            "agent {}'s z must bootstrap from ITS tail: got {z}, want {expect}",
            obs[1]
        );
    }
}

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
    fn actor(&self, s: &WSt) -> Actor {
        if !s.0.is_multiple_of(2) {
            Actor::Chance
        } else {
            Actor::Simultaneous
        }
    }
    fn legal_actions(&self, s: &WSt, _agent: usize) -> Vec<usize> {
        if s.0.is_multiple_of(2) && s.0 < 6 {
            vec![0, 1]
        } else {
            Vec::new()
        }
    }
    fn step(&self, s: &WSt, _actions: &[usize]) -> Transition<WSt, ()> {
        Transition {
            next_state: WSt(s.0 + 1),
            events: vec![None; 2],
            terminal: false,
        }
    }
    fn chance_node(&self, _s: &WSt) -> reinfors_core::ChanceDist {
        reinfors_core::ChanceDist::Uniform(50_000_000)
    }
    fn apply_chance_node(&self, s: &WSt, _outcome: usize) -> Transition<WSt, ()> {
        Transition {
            next_state: WSt(s.0 + 1),
            events: vec![None; 2],
            terminal: s.0 + 1 >= 6,
        }
    }
    fn initial_state(&self) -> WSt {
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
    assert!(hits.iter().all(|&h| h < 300), "roughly uniform: {hits:?}");
}

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
    fn initial_state(&self) -> St {
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
        |_players: &[usize], _obs: Vec<f32>, n: usize| vec![0.0; n * 4],
    );
}

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
    fn initial_state(&self) -> St {
        St { tick: 0 }
    }
}

#[test]
fn realized_roots_expose_the_true_decision_dynamics() {
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
        Game::actor(&RootNodey, &RootNodey.initial_state()),
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
    let mut engine = Engine::new(
        RootNodey,
        Box::new(Enc),
        Box::new(Zero),
        EpsilonGreedyQ::new(2, 0.1),
        Dqn::new(2, 1.0, 1, 0.99),
        EngineParams {
            n_games: 2,
            seed: 0,
            n_groups: 1,
            ..Default::default()
        },
    );
    let mut infer = |_obs: Vec<f32>, n: usize| vec![0.0; n * 2 * 2];
    let (records, _stats) = engine.collect(32, &mut infer);
    assert!(!records.is_empty(), "root-chance games collect normally");
}

#[test]
#[should_panic(expected = "cannot start at a chance node")]
fn start_distribution_restores_must_be_decision_states() {
    // Ordinary play terminates at tick 4; tick 5 exists only so a hostile restore can inject chance.
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
        fn initial_state(&self) -> St {
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
        Dqn::new(2, 1.0, 1, 0.99),
        EngineParams {
            n_games: 1,
            seed: 0,
            n_groups: 1,
            ..Default::default()
        },
    )
    .with_start_distribution(Box::new(ChanceRestore));
    let mut infer = |_obs: Vec<f32>, n: usize| vec![0.0; n * 2 * 2];
    for _ in 0..8 {
        let _ = engine.collect(64, &mut infer);
    }
}

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
    fn initial_state(&self) -> St {
        St { tick: 0 }
    }
}

#[test]
fn forced_maxn_supervises_both_perspectives_at_two_agents() {
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
                n_groups: 1,
                ..Default::default()
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
    for (obs, _pi, z, _w, _player, _legal) in &maxn {
        let expect = f64::from(obs[1]) + 1.0;
        assert!(
            (z - expect).abs() < 1e-12,
            "per-perspective z: got {z}, want {expect}"
        );
    }
}

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
    fn initial_state(&self) -> St {
        St { tick: 0 }
    }
    fn truncation_horizon(&self) -> Option<usize> {
        Some(2)
    }
}

#[test]
fn forced_maxn_truncation_bootstraps_both_perspectives() {
    // Regression: the tail gate once used `n > 2`, omitting the non-mover under forced 2p MaxN.
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
            n_groups: 1,
            ..Default::default()
        },
    );
    // Rewards are zero and gamma is one, so z must be each encoded agent's distinct tail value.
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
    for (obs, _pi, z, _w, _player, _legal) in &records {
        let expect = (f64::from(obs[1]) + 1.0) / 10.0;
        assert!(
            (z - expect).abs() < 1e-12,
            "agent {}'s z must bootstrap from ITS tail: got {z}, want {expect}",
            obs[1]
        );
    }
}

#[test]
fn ppo_truncation_bootstraps_every_perspectives_own_tail() {
    // Zero rewards, gamma 1, per-agent constant critic: ret must equal the agent's OWN
    // value; a missing non-mover tail would leave ret = 0.
    let mut engine = Engine::new(
        EndlessRR,
        Box::new(Enc),
        Box::new(Zero),
        reinfors_core::PpoActor::new(),
        reinfors_core::Ppo::new(1.0, 0.95),
        EngineParams {
            n_games: 1,
            seed: 9,
            n_groups: 1,
            ..Default::default()
        },
    );
    let infer = |obs: Vec<f32>, n: usize| {
        let mut out = Vec::with_capacity(n * 3);
        for r in 0..n {
            let agent = f64::from(obs[r * 2 + 1]);
            out.extend([0.0, 0.0, (agent + 1.0) / 10.0]);
        }
        out
    };
    let (records, _stats) = engine.collect(4, infer);
    assert!(!records.is_empty());
    for r in &records {
        let expect = (r.player as f64 + 1.0) / 10.0;
        assert!(
            (r.ret - expect).abs() < 1e-12,
            "agent {} must bootstrap from its own tail: ret {} want {expect}",
            r.player,
            r.ret
        );
        assert!(r.advantage.abs() < 1e-12);
    }
}

#[test]
fn ppo_windows_meet_the_floor_and_stay_single_version() {
    // Complete-round floor; every record carries THIS window's critic constant.
    let mut engine = Engine::new(
        EndlessRR,
        Box::new(Enc),
        Box::new(Zero),
        reinfors_core::PpoActor::new(),
        reinfors_core::Ppo::new(1.0, 0.95),
        EngineParams {
            n_games: 3,
            seed: 11,
            n_groups: 1,
            ..Default::default()
        },
    );
    let constant = |v: f64| {
        move |_obs: Vec<f32>, n: usize| (0..n).flat_map(|_| [0.0, 0.0, v]).collect::<Vec<f64>>()
    };
    for (window, v) in [(5usize, 1.0), (7, 2.0), (4, 3.0)] {
        let (records, _stats) = engine.collect(window, constant(v));
        assert!(
            records.len() >= window && records.len() < window + 3,
            "floor within one 3-game round at v={v}, got {}",
            records.len()
        );
        for r in &records {
            assert_eq!(
                r.value, v,
                "a stale-version step leaked into the v={v} window"
            );
        }
    }
}

#[test]
fn grouped_tail_failure_leaves_the_pool_respawnable() {
    // A callback that dies exactly on truncation-tail inference (obs tick 2 exists only
    // there) must not strand finished episodes: the retry may see no over-horizon state.
    let mut engine = Engine::new(
        EndlessRR,
        Box::new(Enc),
        Box::new(Zero),
        reinfors_core::PpoActor::new(),
        reinfors_core::Ppo::new(1.0, 0.95),
        EngineParams {
            n_games: 2,
            seed: 3,
            n_groups: 2,
            ..Default::default()
        },
    );
    let flaky = reinfors_core::ServiceHost::spawn(|_p, obs: Vec<f32>, n| {
        if obs.chunks(2).any(|row| row[0] >= 2.0) {
            panic!("tail inference failed");
        }
        (0..n).flat_map(|_| [0.0, 0.0, 1.0]).collect()
    });
    // The error surfaces as a panic (the Python layer converts it to an exception); the
    // engine must stay usable afterwards, exactly as it does across the binding.
    let aborted = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        engine.collect_grouped_hosted(8, reinfors_core::InferMode::Shared, &flaky)
    }));
    assert!(
        aborted.is_err(),
        "the flaky callback must surface its failure"
    );
    let ok = reinfors_core::ServiceHost::spawn(|_p, _obs, n: usize| {
        (0..n).flat_map(|_| [0.0, 0.0, 2.0]).collect::<Vec<f64>>()
    });
    let (records, _stats) = engine.collect_grouped_hosted(8, reinfors_core::InferMode::Shared, &ok);
    assert!(records.len() >= 8, "retry met the floor: {}", records.len());
    for r in &records {
        assert!(
            r.obs[0] < 2.0,
            "a stranded over-horizon episode was advanced: obs {:?}",
            r.obs
        );
        assert_eq!(r.value, 2.0, "an aborted-window step leaked into the retry");
    }
}

#[test]
fn nstep_alternating_truncation_tails_cannot_bootstrap() {
    // EndlessRR truncates at tick 2, when the state belongs to agent 2: every buffered
    // trajectory's tail sees an empty own-perspective legal set, so even wide n-step windows
    // must emit discount 0 (target = the window's reward sum, no bootstrap).
    let mut engine = Engine::new(
        EndlessRR,
        Box::new(Enc),
        Box::new(Zero),
        EpsilonGreedyQ::new(1, 0.1),
        Dqn::new(1, 1.0, 3, 0.9),
        EngineParams {
            n_games: 2,
            seed: 4,
            n_groups: 1,
            ..Default::default()
        },
    );
    let mut infer = |_obs: Vec<f32>, n: usize| vec![0.0; n * 2];
    let (records, _stats) = engine.collect(12, &mut infer);
    assert!(!records.is_empty());
    for r in &records {
        assert!(
            r.next_legal.is_empty(),
            "the truncated state is the opponent's"
        );
        assert_eq!(r.discount, 0.0, "alternating tails must not bootstrap");
    }
}
