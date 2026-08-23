use reinfors_core::{
    mcts_many, realize_initial_state, search_many, ActBy, Actor, ChanceDist, ChanceMode, Engine,
    EngineParams, Evaluator, Game, InferMode, Mcts, MctsConfig, Opponent, Reward, SearchConfig,
    Space, StateEncoder, Transition, TreeStrap,
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

/// Half the fan terminates with +4 and half continues at zero, so its exact value is 2. Setting
/// `both_terminal` makes the second half terminal too and exercises a fan with no staged rows.
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

/// P0 moves, chance hands the turn to P1, then P1 ends at +1: P0's value must flip only at the
/// decision handoff, never merely while traversing chance.
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
            ..Default::default()
        },
    );
    let (records, stats) = engine.collect(4, |_obs: Vec<f32>, n: usize| vec![0.0; n]);
    assert!(records.len() >= 4);
    assert!(stats.decisions > 0);
}

/// Half terminates at +2 and half chains to a second chance node paying +6: exact value 4.
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
    // Reuses ChainTick's derivation above: (1+2+3) + 0.5*8 = 10.
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

/// A root chance over a seed space too large to enumerate; outcome parity pays 10 or 20.
struct SampleOnlySeed;
impl Game for SampleOnlySeed {
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
    fn chance_enumerable(&self) -> bool {
        false
    }
    fn chance_node(&self, _s: &St) -> ChanceDist {
        ChanceDist::SampleOnlyUniform(std::num::NonZeroU32::new(1 << 20).unwrap())
    }
    fn apply_chance_node(&self, _s: &St, outcome: usize) -> Transition<St, f64> {
        Transition {
            next_state: St { tick: 2 },
            events: vec![Some(10.0 + 10.0 * (outcome % 2) as f64)],
            terminal: true,
        }
    }
    fn initial_state(&self) -> St {
        St { tick: 0 }
    }
}

#[test]
fn sample_only_resampling_converges() {
    let cfg = mcts_cfg(4000, 8, 0.5, ChanceMode::AlwaysResample);
    let e = run_mcts(&SampleOnlySeed, fan_chance, &cfg, vec![(St { tick: 0 }, 0)]);
    assert!(
        (e[0].values[0][0] - 15.0).abs() < 0.5,
        "{}",
        e[0].values[0][0]
    );
}

#[test]
fn sample_only_committed_is_seed_deterministic() {
    let cfg = mcts_cfg(64, 8, 0.5, ChanceMode::Committed { samples: 3 });
    let a = run_mcts(&SampleOnlySeed, fan_chance, &cfg, vec![(St { tick: 0 }, 0)]);
    let b = run_mcts(&SampleOnlySeed, fan_chance, &cfg, vec![(St { tick: 0 }, 0)]);
    assert_eq!(a[0].values, b[0].values);
}

#[test]
#[should_panic(expected = "sample-only")]
fn sample_only_expand_all_panics_in_mcts() {
    let cfg = mcts_cfg(1, 8, 0.5, ChanceMode::ExpandAll);
    let _ = run_mcts(&SampleOnlySeed, fan_chance, &cfg, vec![(St { tick: 0 }, 0)]);
}

#[test]
#[should_panic(expected = "sample-only")]
fn sample_only_expand_all_panics_in_expectimax() {
    let _ = search_many(
        &SampleOnlySeed,
        &GuardEnc(fan_chance),
        &Pass,
        &xmax_cfg(0.5, ChanceMode::ExpandAll),
        vec![(St { tick: 0 }, 0)],
        false,
        0,
        |_p: &[usize], _obs: Vec<f32>, n: usize| vec![0.0; n],
    );
}

#[test]
fn sample_only_engine_collection_draws_outcomes() {
    let mut engine = Engine::new(
        SampleOnlySeed,
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
            n_threads: Some(1),
            ..Default::default()
        },
    );
    let (records, stats) = engine.collect(4, |_obs: Vec<f32>, n: usize| vec![0.0; n]);
    assert!(records.len() >= 4);
    assert!(stats.decisions > 0);
}

#[derive(Clone)]
struct TwoSt(u8);
struct SampleOnlyDuel;
impl Game for SampleOnlyDuel {
    type State = TwoSt;
    type Event = f64;
    fn num_agents(&self) -> usize {
        2
    }
    fn action_count(&self) -> usize {
        2
    }
    fn actor(&self, s: &TwoSt) -> Actor {
        if s.0 == 1 {
            Actor::Chance
        } else {
            Actor::Agent((s.0 % 2) as usize)
        }
    }
    fn legal_actions(&self, s: &TwoSt, _agent: usize) -> Vec<usize> {
        if s.0 == 1 {
            Vec::new()
        } else {
            vec![0, 1]
        }
    }
    fn step(&self, s: &TwoSt, _actions: &[usize]) -> Transition<TwoSt, f64> {
        Transition {
            next_state: TwoSt(s.0 + 1),
            events: vec![None, None],
            terminal: s.0 >= 3,
        }
    }
    fn chance_node(&self, s: &TwoSt) -> ChanceDist {
        assert_eq!(s.0, 1);
        ChanceDist::SampleOnlyUniform(std::num::NonZeroU32::new(1 << 16).unwrap())
    }
    fn apply_chance_node(&self, s: &TwoSt, _outcome: usize) -> Transition<TwoSt, f64> {
        assert_eq!(s.0, 1);
        Transition::silent(TwoSt(2), 2)
    }
    fn information_states(&self) -> bool {
        true
    }
    fn information_state_key(&self, s: &TwoSt, agent: usize) -> Vec<u8> {
        vec![s.0, agent as u8]
    }
    fn chance_enumerable(&self) -> bool {
        false
    }
    fn initial_state(&self) -> TwoSt {
        TwoSt(0)
    }
}

#[test]
#[should_panic(expected = "require enumerable chance")]
fn cfr_rejects_sample_only_chance_at_construction() {
    let _ = reinfors_core::CfrSolver::new(
        SampleOnlyDuel,
        Box::new(Pass),
        reinfors_core::CfrVariant::Vanilla,
        7,
    );
}

#[test]
fn mccfr_trains_on_sample_only_chance_but_exact_surfaces_refuse() {
    let mut solver = reinfors_core::CfrSolver::new(
        SampleOnlyDuel,
        Box::new(Pass),
        reinfors_core::CfrVariant::ExternalMccfr,
        7,
    );
    solver.iterate(16);
    assert!(reinfors_core::CfrSolver::exploitability(&solver).is_err());
    assert!(reinfors_core::CfrSolver::expected_value(&solver, 0).is_err());
}

/// Root-only sample chance: the initial state is the chance node, play is chance-free —
/// the CarRacing shape and claim pair (whole-game false, post-realization true).
struct RootOnlyRace;
impl Game for RootOnlyRace {
    type State = St;
    type Event = f64;
    fn num_agents(&self) -> usize {
        1
    }
    fn action_count(&self) -> usize {
        1
    }
    fn actor(&self, s: &St) -> Actor {
        if s.tick == 0 {
            Actor::Chance
        } else {
            Actor::Agent(0)
        }
    }
    fn legal_actions(&self, s: &St, _agent: usize) -> Vec<usize> {
        if s.tick == 0 || s.tick >= 100 {
            Vec::new()
        } else {
            vec![0]
        }
    }
    fn step(&self, s: &St, _actions: &[usize]) -> Transition<St, f64> {
        Transition {
            next_state: St { tick: 100 },
            events: vec![Some(s.tick as f64)],
            terminal: true,
        }
    }
    fn chance_enumerable(&self) -> bool {
        false
    }
    fn searchable_chance_enumerable(&self) -> bool {
        true
    }
    fn chance_node(&self, s: &St) -> ChanceDist {
        assert_eq!(s.tick, 0, "the only chance node is the root");
        ChanceDist::SampleOnlyUniform(std::num::NonZeroU32::new(1 << 20).unwrap())
    }
    fn apply_chance_node(&self, s: &St, outcome: usize) -> Transition<St, f64> {
        assert_eq!(s.tick, 0);
        Transition::silent(
            St {
                tick: 1 + outcome % 64,
            },
            1,
        )
    }
    fn initial_state(&self) -> St {
        St { tick: 0 }
    }
}

#[test]
fn root_only_sample_chance_realizes_and_varies() {
    let g = RootOnlyRace;
    assert!(!g.chance_enumerable(), "the raw root is sample-only");
    assert!(
        g.searchable_chance_enumerable(),
        "post-realization play is chance-free"
    );
    let mut rng = reinfors_core::SplitMix64::new(3);
    let ticks: Vec<usize> = (0..8)
        .map(|_| {
            let s = realize_initial_state(&g, &mut rng);
            assert!(matches!(g.actor(&s), Actor::Agent(0)));
            s.tick
        })
        .collect();
    assert!(
        ticks.iter().collect::<std::collections::HashSet<_>>().len() > 1,
        "repeated realization should draw different outcomes: {ticks:?}"
    );
}

#[test]
fn root_only_sample_chance_supports_expand_all_and_episodes() {
    let never_chance = |s: &St| s.tick == 0;
    let cfg = mcts_cfg(4, 4, 1.0, ChanceMode::ExpandAll);
    let e = run_mcts(&RootOnlyRace, never_chance, &cfg, vec![(St { tick: 5 }, 0)]);
    assert_eq!(e.len(), 1);

    let mut engine = Engine::new(
        RootOnlyRace,
        Box::new(GuardEnc(never_chance)),
        Box::new(Pass),
        Mcts::new(cfg, ActBy::Value),
        TreeStrap::new(1.0, 0.3, 1.0, false),
        EngineParams {
            n_games: 2,
            seed: 5,
            n_threads: Some(1),
            ..Default::default()
        },
    );
    let (records, stats) = engine.collect(12, |_obs: Vec<f32>, n: usize| vec![0.0; n]);
    assert!(
        records.len() >= 12,
        "multi-episode collection realizes each root"
    );
    assert!(stats.decisions > 0);
}

/// Root-only sample chance, two players with information states: exact CFR must reject
/// (it enumerates the raw root) while sampling solvers and engine search stay supported.
#[derive(Clone)]
struct RootDuelSt(u8);
struct RootOnlyDuel;
impl Game for RootOnlyDuel {
    type State = RootDuelSt;
    type Event = f64;
    fn num_agents(&self) -> usize {
        2
    }
    fn action_count(&self) -> usize {
        2
    }
    fn actor(&self, s: &RootDuelSt) -> Actor {
        if s.0 == 0 {
            Actor::Chance
        } else {
            Actor::Agent((s.0 % 2) as usize)
        }
    }
    fn legal_actions(&self, s: &RootDuelSt, _agent: usize) -> Vec<usize> {
        if s.0 == 0 {
            Vec::new()
        } else {
            vec![0, 1]
        }
    }
    fn step(&self, s: &RootDuelSt, _actions: &[usize]) -> Transition<RootDuelSt, f64> {
        Transition {
            next_state: RootDuelSt(s.0 + 1),
            events: vec![None, None],
            terminal: s.0 >= 3,
        }
    }
    fn information_states(&self) -> bool {
        true
    }
    fn information_state_key(&self, s: &RootDuelSt, agent: usize) -> Vec<u8> {
        vec![s.0, agent as u8]
    }
    fn chance_enumerable(&self) -> bool {
        false
    }
    fn searchable_chance_enumerable(&self) -> bool {
        true
    }
    fn chance_node(&self, s: &RootDuelSt) -> ChanceDist {
        assert_eq!(s.0, 0, "the only chance node is the root");
        ChanceDist::SampleOnlyUniform(std::num::NonZeroU32::new(1 << 20).unwrap())
    }
    fn apply_chance_node(&self, s: &RootDuelSt, outcome: usize) -> Transition<RootDuelSt, f64> {
        assert_eq!(s.0, 0);
        Transition::silent(RootDuelSt(1 + (outcome % 2) as u8), 2)
    }
    fn initial_state(&self) -> RootDuelSt {
        RootDuelSt(0)
    }
}

#[test]
#[should_panic(expected = "require enumerable chance")]
fn exact_cfr_rejects_root_only_sample_chance() {
    let _ = reinfors_core::CfrSolver::new(
        RootOnlyDuel,
        Box::new(Pass),
        reinfors_core::CfrVariant::Vanilla,
        7,
    );
}

#[test]
fn sampling_solvers_support_root_only_sample_chance() {
    let mut mccfr = reinfors_core::CfrSolver::new(
        RootOnlyDuel,
        Box::new(Pass),
        reinfors_core::CfrVariant::ExternalMccfr,
        7,
    );
    mccfr.iterate(16);
    assert!(reinfors_core::CfrSolver::expected_value(&mccfr, 0).is_err());

    struct DuelEnc;
    impl reinfors_core::ActionView for DuelEnc {}
    impl StateEncoder for DuelEnc {
        type State = RootDuelSt;
        fn encode(&self, s: &RootDuelSt, agent: usize) -> Vec<f32> {
            vec![f32::from(s.0), agent as f32]
        }
        fn obs_shape(&self) -> (usize, usize, usize) {
            (1, 1, 2)
        }
        fn observation_space(&self) -> Space {
            Space::unit_box(vec![1, 1, 2])
        }
    }
    let mut deep =
        reinfors_core::DeepCfrSolver::new(RootOnlyDuel, Box::new(DuelEnc), Box::new(Pass), 7);
    deep.next_iteration();
    let (adv, strat, _) =
        deep.collect(0, 4, |_p: usize, _obs: Vec<f32>, n: usize| vec![0.0; n * 2]);
    assert!(!adv.is_empty() || !strat.is_empty());
}
