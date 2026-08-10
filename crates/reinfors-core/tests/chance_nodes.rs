use reinfors_core::{
    mcts_many, search_many, ActBy, Actor, ChanceDist, ChanceMode, Engine, EngineParams, Evaluator,
    Game, InferMode, Mcts, MctsConfig, Opponent, Reward, SearchConfig, Space, StateEncoder,
    Transition, TreeStrap,
};

#[derive(Clone)]
struct St {
    tick: usize,
}

struct GuardEnc(fn(&St) -> bool);
impl reinfors_core::ActionView for GuardEnc {}
impl StateEncoder for GuardEnc {
    type State = St;
    fn encode(&self, s: &St, agent: usize) -> Vec<f32> {
        assert!(!(self.0)(s), "the net must never see a chance state");
        vec![s.tick as f32, agent as f32]
    }
    fn obs_shape(&self) -> (usize, usize, usize) {
        (1, 1, 2)
    }
    fn observation_space(&self) -> Space {
        Space::unit_box(vec![1, 1, 2])
    }
}

struct Pass;
impl Reward for Pass {
    type Event = f64;
    fn step_reward(&self, e: &f64, _agent: usize) -> f64 {
        *e
    }
}

fn mcts_cfg(sims: usize, max_depth: i32, gamma: f64, chance: ChanceMode) -> MctsConfig {
    MctsConfig {
        num_simulations: sims,
        uct_c: 1.0,
        gamma,
        max_depth,
        temperature: 0.0,
        temperature_drop: 0,
        chance,
    }
}

fn xmax_cfg(gamma: f64, chance: ChanceMode) -> SearchConfig {
    SearchConfig {
        gamma,
        beta: 1.0,
        expansion_budget: 16,
        top_k: 2,
        max_depth: 8,
        chance,
        opponent: Opponent::Uniform,
    }
}

struct PayoutFan;
impl Game for PayoutFan {
    type State = St;
    type Event = f64;
    fn num_agents(&self) -> usize {
        1
    }
    fn action_count(&self) -> usize {
        1
    }
    fn actor(&self, s: &St) -> Actor {
        if s.tick == 1 {
            Actor::Chance
        } else {
            Actor::Agent(0)
        }
    }
    fn legal_actions(&self, s: &St, _agent: usize) -> Vec<usize> {
        if s.tick == 0 {
            vec![0]
        } else {
            Vec::new()
        }
    }
    fn step(&self, s: &St, _actions: &[usize]) -> Transition<St, f64> {
        assert_eq!(s.tick, 0);
        Transition {
            next_state: St { tick: 1 },
            events: vec![None],
            terminal: false,
        }
    }
    fn chance_node(&self, _s: &St) -> ChanceDist {
        ChanceDist::Weighted(vec![0.25, 0.75])
    }
    fn apply_chance_node(&self, _s: &St, outcome: usize) -> Transition<St, f64> {
        Transition {
            next_state: St { tick: 2 },
            events: vec![Some(10.0 + 10.0 * outcome as f64)],
            terminal: true,
        }
    }
    fn initial_state(&self) -> St {
        St { tick: 0 }
    }
}
fn fan_chance(s: &St) -> bool {
    s.tick == 1
}

fn run_mcts<G>(
    g: &G,
    guard: fn(&St) -> bool,
    cfg: &MctsConfig,
    reqs: Vec<(St, usize)>,
) -> Vec<reinfors_core::SearchEvaluation>
where
    G: Game<State = St, Event = f64> + Sync,
{
    let a = g.action_count();
    let mut infer = move |_p: usize, _obs: Vec<f32>, n: usize| vec![0.0; n * a];
    let mut eval = Evaluator::new(&mut infer, InferMode::Shared, None);
    mcts_many(g, &GuardEnc(guard), &Pass, cfg, reqs, 7, &mut eval)
}

#[test]
fn expand_all_backs_up_the_exact_expectation() {
    // 0.25*10 + 0.75*20 = 17.5 on this tick; discounting the chance edge would wrongly give 8.75.
    let cfg = mcts_cfg(1, 8, 0.5, ChanceMode::ExpandAll);
    let e = run_mcts(&PayoutFan, fan_chance, &cfg, vec![(St { tick: 0 }, 0)]);
    assert!(
        (e[0].values[0][0] - 17.5).abs() < 1e-12,
        "{}",
        e[0].values[0][0]
    );

    let results = search_many(
        &PayoutFan,
        &GuardEnc(fan_chance),
        &Pass,
        &xmax_cfg(0.5, ChanceMode::ExpandAll),
        vec![(St { tick: 0 }, 0)],
        false,
        0,
        |_p: &[usize], _obs: Vec<f32>, n: usize| vec![0.0; n],
    );
    let (values, _, _) = &results[0];
    assert!((values[0][0] - 17.5).abs() < 1e-9, "{}", values[0][0]);
}

#[test]
fn resampling_converges_to_the_expectation() {
    let cfg = mcts_cfg(4000, 8, 0.5, ChanceMode::AlwaysResample);
    let e = run_mcts(&PayoutFan, fan_chance, &cfg, vec![(St { tick: 0 }, 0)]);
    assert!(
        (e[0].values[0][0] - 17.5).abs() < 0.5,
        "{}",
        e[0].values[0][0]
    );
}

#[test]
fn committed_draws_are_seed_deterministic() {
    let cfg = mcts_cfg(64, 8, 0.5, ChanceMode::Committed { samples: 3 });
    let a = run_mcts(&PayoutFan, fan_chance, &cfg, vec![(St { tick: 0 }, 0)]);
    let b = run_mcts(&PayoutFan, fan_chance, &cfg, vec![(St { tick: 0 }, 0)]);
    assert_eq!(a[0].values, b[0].values);
    assert_eq!(a[0].visits, b[0].visits);
}

struct ChainTick;
impl Game for ChainTick {
    type State = St;
    type Event = f64;
    fn num_agents(&self) -> usize {
        1
    }
    fn action_count(&self) -> usize {
        1
    }
    fn actor(&self, s: &St) -> Actor {
        if (1..=2).contains(&s.tick) {
            Actor::Chance
        } else {
            Actor::Agent(0)
        }
    }
    fn legal_actions(&self, s: &St, _agent: usize) -> Vec<usize> {
        if s.tick == 0 || s.tick == 3 {
            vec![0]
        } else {
            Vec::new()
        }
    }
    fn step(&self, s: &St, _actions: &[usize]) -> Transition<St, f64> {
        match s.tick {
            0 => Transition {
                next_state: St { tick: 1 },
                events: vec![Some(1.0)],
                terminal: false,
            },
            3 => Transition {
                next_state: St { tick: 4 },
                events: vec![Some(8.0)],
                terminal: true,
            },
            _ => unreachable!("no decisions at chance ticks"),
        }
    }
    fn chance_node(&self, _s: &St) -> ChanceDist {
        ChanceDist::Weighted(vec![1.0])
    }
    fn apply_chance_node(&self, s: &St, _outcome: usize) -> Transition<St, f64> {
        Transition {
            next_state: St { tick: s.tick + 1 },
            events: vec![Some(1.0 + s.tick as f64)],
            terminal: false,
        }
    }
    fn initial_state(&self) -> St {
        St { tick: 0 }
    }
}
fn chain_chance(s: &St) -> bool {
    (1..=2).contains(&s.tick)
}

#[test]
fn chain_rewards_join_their_tick_undiscounted_and_add_no_depth() {
    // The depth cap is exactly tight: chance-transparent depth reaches the second decision.
    // Its value is (1+2+3) + 0.5*8 = 10.
    let cfg = mcts_cfg(256, 2, 0.5, ChanceMode::Committed { samples: 1 });
    let e = run_mcts(&ChainTick, chain_chance, &cfg, vec![(St { tick: 0 }, 0)]);
    let q = e[0].values[0][0];
    assert!(q > 9.9 && q <= 10.0 + 1e-9, "{q}");

    let results = search_many(
        &ChainTick,
        &GuardEnc(chain_chance),
        &Pass,
        &xmax_cfg(0.5, ChanceMode::Committed { samples: 1 }),
        vec![(St { tick: 0 }, 0)],
        false,
        0,
        |_p: &[usize], _obs: Vec<f32>, n: usize| vec![0.0; n],
    );
    let (values, _, _) = &results[0];
    assert!((values[0][0] - 10.0).abs() < 1e-9, "{}", values[0][0]);
}

struct MixedFan {
    both_terminal: bool,
}
impl Game for MixedFan {
    type State = St;
    type Event = f64;
    fn num_agents(&self) -> usize {
        1
    }
    fn action_count(&self) -> usize {
        1
    }
    fn actor(&self, s: &St) -> Actor {
        if s.tick == 1 {
            Actor::Chance
        } else {
            Actor::Agent(0)
        }
    }
    fn legal_actions(&self, s: &St, _agent: usize) -> Vec<usize> {
        if s.tick == 0 || s.tick == 2 {
            vec![0]
        } else {
            Vec::new()
        }
    }
    fn step(&self, s: &St, _actions: &[usize]) -> Transition<St, f64> {
        match s.tick {
            0 => Transition {
                next_state: St { tick: 1 },
                events: vec![None],
                terminal: false,
            },
            2 => Transition {
                next_state: St { tick: 3 },
                events: vec![Some(0.0)],
                terminal: true,
            },
            _ => unreachable!(),
        }
    }
    fn chance_node(&self, _s: &St) -> ChanceDist {
        ChanceDist::Weighted(vec![0.5, 0.5])
    }
    fn apply_chance_node(&self, _s: &St, outcome: usize) -> Transition<St, f64> {
        if outcome == 0 {
            Transition {
                next_state: St { tick: 9 },
                events: vec![Some(4.0)],
                terminal: true,
            }
        } else if self.both_terminal {
            Transition {
                next_state: St { tick: 9 },
                events: vec![Some(0.0)],
                terminal: true,
            }
        } else {
            Transition {
                next_state: St { tick: 2 },
                events: vec![None],
                terminal: false,
            }
        }
    }
    fn initial_state(&self) -> St {
        St { tick: 0 }
    }
}

#[test]
fn expand_all_handles_terminal_outcomes() {
    let cfg = mcts_cfg(1, 8, 0.5, ChanceMode::ExpandAll);
    let e = run_mcts(
        &MixedFan {
            both_terminal: false,
        },
        fan_chance,
        &cfg,
        vec![(St { tick: 0 }, 0)],
    );
    assert!(
        (e[0].values[0][0] - 2.0).abs() < 1e-12,
        "{}",
        e[0].values[0][0]
    );

    let e = run_mcts(
        &MixedFan {
            both_terminal: true,
        },
        fan_chance,
        &cfg,
        vec![(St { tick: 0 }, 0)],
    );
    assert!(
        (e[0].values[0][0] - 2.0).abs() < 1e-12,
        "{}",
        e[0].values[0][0]
    );
}

struct SimChance;
impl Game for SimChance {
    type State = St;
    type Event = f64;
    fn num_agents(&self) -> usize {
        2
    }
    fn action_count(&self) -> usize {
        2
    }
    fn actor(&self, s: &St) -> Actor {
        if s.tick == 1 {
            Actor::Chance
        } else {
            Actor::Simultaneous
        }
    }
    fn legal_actions(&self, s: &St, _agent: usize) -> Vec<usize> {
        if s.tick == 0 {
            vec![0, 1]
        } else {
            Vec::new()
        }
    }
    fn step(&self, _s: &St, _actions: &[usize]) -> Transition<St, f64> {
        Transition {
            next_state: St { tick: 1 },
            events: vec![None, None],
            terminal: false,
        }
    }
    fn chance_node(&self, _s: &St) -> ChanceDist {
        ChanceDist::Weighted(vec![1.0])
    }
    fn apply_chance_node(&self, _s: &St, _outcome: usize) -> Transition<St, f64> {
        Transition {
            next_state: St { tick: 2 },
            events: vec![Some(3.0), Some(-3.0)],
            terminal: true,
        }
    }
    fn initial_state(&self) -> St {
        St { tick: 0 }
    }
}

#[test]
fn simultaneous_trees_take_per_agent_chance_rewards() {
    let cfg = mcts_cfg(32, 8, 0.5, ChanceMode::Committed { samples: 1 });
    let e = run_mcts(
        &SimChance,
        fan_chance,
        &cfg,
        vec![(St { tick: 0 }, 0), (St { tick: 0 }, 1)],
    );
    for (ei, want) in [(0usize, 3.0f64), (1, -3.0)] {
        for (a, &v) in e[ei].values[0].iter().enumerate() {
            if e[ei].visits[a] > 0.0 {
                assert!((v - want).abs() < 1e-9, "agent {ei} action {a}: {v}");
            }
        }
    }
}

struct TurnFlip;
impl Game for TurnFlip {
    type State = St;
    type Event = f64;
    fn num_agents(&self) -> usize {
        2
    }
    fn action_count(&self) -> usize {
        1
    }
    fn actor(&self, s: &St) -> Actor {
        match s.tick {
            1 => Actor::Chance,
            2 => Actor::Agent(1),
            _ => Actor::Agent(0),
        }
    }
    fn legal_actions(&self, s: &St, agent: usize) -> Vec<usize> {
        match (s.tick, agent) {
            (0, 0) | (2, 1) => vec![0],
            _ => Vec::new(),
        }
    }
    fn step(&self, s: &St, _actions: &[usize]) -> Transition<St, f64> {
        match s.tick {
            0 => Transition {
                next_state: St { tick: 1 },
                events: vec![None, None],
                terminal: false,
            },
            2 => Transition {
                next_state: St { tick: 3 },
                events: vec![Some(-1.0), Some(1.0)],
                terminal: true,
            },
            _ => unreachable!(),
        }
    }
    fn chance_node(&self, _s: &St) -> ChanceDist {
        ChanceDist::Weighted(vec![1.0])
    }
    fn apply_chance_node(&self, _s: &St, _outcome: usize) -> Transition<St, f64> {
        Transition {
            next_state: St { tick: 2 },
            events: vec![None, None],
            terminal: false,
        }
    }
    fn initial_state(&self) -> St {
        St { tick: 0 }
    }
}

#[test]
fn perspective_flips_at_the_handoff_not_the_chain() {
    let cfg = mcts_cfg(256, 8, 1.0, ChanceMode::Committed { samples: 1 });
    let e = run_mcts(&TurnFlip, fan_chance, &cfg, vec![(St { tick: 0 }, 0)]);
    let q = e[0].values[0][0];
    assert!(q < -0.98, "P0's root value must approach −1, got {q}");
}

#[test]
fn engine_collects_with_search_policies_through_chance() {
    let mut engine = Engine::new(
        TurnFlip,
        Box::new(GuardEnc(fan_chance)),
        Box::new(Pass),
        Mcts::new(
            mcts_cfg(8, 8, 1.0, ChanceMode::Committed { samples: 1 }),
            ActBy::Value,
        ),
        TreeStrap::new(1.0, 0.3, 1.0, false),
        EngineParams {
            n_games: 2,
            seed: 3,
        },
    );
    let (records, stats) = engine.collect(4, |_obs: Vec<f32>, n: usize| vec![0.0; n]);
    assert!(records.len() >= 4);
    assert!(stats.decisions > 0);
}

struct BranchChain;
impl Game for BranchChain {
    type State = St;
    type Event = f64;
    fn num_agents(&self) -> usize {
        1
    }
    fn action_count(&self) -> usize {
        1
    }
    fn actor(&self, s: &St) -> Actor {
        if (1..=2).contains(&s.tick) {
            Actor::Chance
        } else {
            Actor::Agent(0)
        }
    }
    fn legal_actions(&self, s: &St, _agent: usize) -> Vec<usize> {
        if s.tick == 0 {
            vec![0]
        } else {
            Vec::new()
        }
    }
    fn step(&self, _s: &St, _actions: &[usize]) -> Transition<St, f64> {
        Transition {
            next_state: St { tick: 1 },
            events: vec![None],
            terminal: false,
        }
    }
    fn chance_node(&self, s: &St) -> ChanceDist {
        if s.tick == 1 {
            ChanceDist::Weighted(vec![0.5, 0.5])
        } else {
            ChanceDist::Weighted(vec![1.0])
        }
    }
    fn apply_chance_node(&self, s: &St, outcome: usize) -> Transition<St, f64> {
        match (s.tick, outcome) {
            (1, 0) => Transition {
                next_state: St { tick: 9 },
                events: vec![Some(2.0)],
                terminal: true,
            },
            (1, 1) => Transition {
                next_state: St { tick: 2 },
                events: vec![None],
                terminal: false,
            },
            (2, _) => Transition {
                next_state: St { tick: 9 },
                events: vec![Some(6.0)],
                terminal: true,
            },
            _ => unreachable!(),
        }
    }
    fn initial_state(&self) -> St {
        St { tick: 0 }
    }
}
fn branch_chance(s: &St) -> bool {
    (1..=2).contains(&s.tick)
}

#[test]
fn expand_all_flattens_chance_chains() {
    let cfg = mcts_cfg(1, 8, 0.5, ChanceMode::ExpandAll);
    let e = run_mcts(&BranchChain, branch_chance, &cfg, vec![(St { tick: 0 }, 0)]);
    assert!(
        (e[0].values[0][0] - 4.0).abs() < 1e-12,
        "{}",
        e[0].values[0][0]
    );

    let results = search_many(
        &BranchChain,
        &GuardEnc(branch_chance),
        &Pass,
        &xmax_cfg(0.5, ChanceMode::ExpandAll),
        vec![(St { tick: 0 }, 0)],
        false,
        0,
        |_p: &[usize], _obs: Vec<f32>, n: usize| vec![0.0; n],
    );
    let (values, _, _) = &results[0];
    assert!((values[0][0] - 4.0).abs() < 1e-9, "{}", values[0][0]);
}

#[test]
fn expand_all_flattens_chains_with_decision_continuations() {
    let cfg = mcts_cfg(256, 2, 0.5, ChanceMode::ExpandAll);
    let e = run_mcts(&ChainTick, chain_chance, &cfg, vec![(St { tick: 0 }, 0)]);
    let q = e[0].values[0][0];
    assert!(q > 9.9 && q <= 10.0 + 1e-9, "{q}");
}

struct WideChain;
impl Game for WideChain {
    type State = St;
    type Event = f64;
    fn num_agents(&self) -> usize {
        1
    }
    fn action_count(&self) -> usize {
        1
    }
    fn actor(&self, s: &St) -> Actor {
        if (1..=2).contains(&s.tick) {
            Actor::Chance
        } else {
            Actor::Agent(0)
        }
    }
    fn legal_actions(&self, s: &St, _agent: usize) -> Vec<usize> {
        if s.tick == 0 {
            vec![0]
        } else {
            Vec::new()
        }
    }
    fn step(&self, _s: &St, _actions: &[usize]) -> Transition<St, f64> {
        Transition {
            next_state: St { tick: 1 },
            events: vec![None],
            terminal: false,
        }
    }
    fn chance_node(&self, s: &St) -> ChanceDist {
        if s.tick == 1 {
            ChanceDist::Weighted(vec![0.5, 0.5])
        } else {
            ChanceDist::Uniform(reinfors_core::MAX_ENUMERATED_OUTCOMES)
        }
    }
    fn apply_chance_node(&self, s: &St, outcome: usize) -> Transition<St, f64> {
        if s.tick == 1 && outcome == 0 {
            Transition {
                next_state: St { tick: 2 },
                events: vec![None],
                terminal: false,
            }
        } else {
            Transition {
                next_state: St { tick: 9 },
                events: vec![None],
                terminal: true,
            }
        }
    }
    fn initial_state(&self) -> St {
        St { tick: 0 }
    }
}

#[test]
#[should_panic(expected = "flattened fan exceeds the enumeration bound")]
fn the_flattened_fan_cap_counts_unprocessed_outcomes() {
    // One seed expands to a cap-sized fan while its sibling remains pending; counting only the
    // expanded seed would admit almost twice the bound.
    let cfg = mcts_cfg(1, 8, 0.5, ChanceMode::ExpandAll);
    let _ = run_mcts(&WideChain, branch_chance, &cfg, vec![(St { tick: 0 }, 0)]);
}

struct Cycler;
impl Game for Cycler {
    type State = St;
    type Event = f64;
    fn num_agents(&self) -> usize {
        1
    }
    fn action_count(&self) -> usize {
        1
    }
    fn actor(&self, s: &St) -> Actor {
        if s.tick >= 1 {
            Actor::Chance
        } else {
            Actor::Agent(0)
        }
    }
    fn legal_actions(&self, s: &St, _agent: usize) -> Vec<usize> {
        if s.tick == 0 {
            vec![0]
        } else {
            Vec::new()
        }
    }
    fn step(&self, _s: &St, _actions: &[usize]) -> Transition<St, f64> {
        Transition {
            next_state: St { tick: 1 },
            events: vec![None],
            terminal: false,
        }
    }
    fn chance_node(&self, _s: &St) -> ChanceDist {
        ChanceDist::Weighted(vec![1.0])
    }
    fn apply_chance_node(&self, s: &St, _outcome: usize) -> Transition<St, f64> {
        Transition {
            next_state: St { tick: s.tick + 1 },
            events: vec![None],
            terminal: false,
        }
    }
    fn initial_state(&self) -> St {
        St { tick: 0 }
    }
}
fn cycler_chance(s: &St) -> bool {
    s.tick >= 1
}

#[test]
#[should_panic(expected = "chance-node chain exceeded")]
fn a_cycling_chain_panics_in_mcts() {
    let cfg = mcts_cfg(1, 8, 0.5, ChanceMode::Committed { samples: 1 });
    let _ = run_mcts(&Cycler, cycler_chance, &cfg, vec![(St { tick: 0 }, 0)]);
}

#[test]
#[should_panic(expected = "chance-node chain exceeded")]
fn a_cycling_chain_panics_in_expectimax() {
    let _ = search_many(
        &Cycler,
        &GuardEnc(cycler_chance),
        &Pass,
        &xmax_cfg(0.5, ChanceMode::Committed { samples: 1 }),
        vec![(St { tick: 0 }, 0)],
        false,
        0,
        |_p: &[usize], _obs: Vec<f32>, n: usize| vec![0.0; n],
    );
}
