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
    assert_eq!(
        SelectiveExpectimax::new(search_cfg(), 2, 0.0).max_agents(),
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
    assert_eq!(Mcts::new(mcts_cfg, ActBy::Value).max_agents(), Some(2));
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
    assert_eq!(AlphaZero::new(az_cfg).max_agents(), Some(2));
    assert_eq!(EpsilonGreedyQ::new(1, 0.0).max_agents(), None);
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

#[test]
#[should_panic(expected = "at most 2 agents")]
fn mcts_search_backstop_rejects_three_agents() {
    let cfg = MctsConfig {
        num_simulations: 4,
        uct_c: 1.0,
        gamma: 0.99,
        max_depth: 4,
        temperature: 0.0,
        temperature_drop: 0,
        chance: ChanceMode::Committed { samples: 1 },
    };
    let mut infer = |_obs: Vec<f32>, n: usize| vec![0.0; n * 2];
    let mut eval = Evaluator::new(&mut infer, None);
    let _ = mcts_many(
        &ThreeWay,
        &Enc,
        &Zero,
        &cfg,
        vec![(St { tick: 0 }, 0)],
        0,
        &mut eval,
    );
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
