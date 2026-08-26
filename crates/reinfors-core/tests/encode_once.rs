//! The encode-once seam: a policy that marks its request row as the canonical
//! observation (`RequestSink::push_root`) spares the engine's training record a second
//! encode, byte-identically, across rounds and routing modes — and marker misuse
//! fails loudly.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use reinfors_core::codec::bytes::Reader;
use reinfors_core::policies::modelfree::ppo::masked_log_probs;
use reinfors_core::policy::{RequestSink, RoundStatus, RowsView, SearchCtx};
use reinfors_core::rollout::engine::{Engine, EngineParams};
use reinfors_core::rollout::evaluator::InferMode;
use reinfors_core::{
    Actor, Game, Policy, PpoActor, PpoEvaluation, Reward, Rng, Space, StateEncoder, Transition,
};

#[derive(Clone)]
struct St {
    tick: usize,
}

/// Single-agent line: terminal after 3 decisions.
struct Line;
impl Game for Line {
    type State = St;
    type Event = ();
    fn num_agents(&self) -> usize {
        1
    }
    fn action_count(&self) -> usize {
        2
    }
    fn actor(&self, _s: &St) -> Actor {
        Actor::Agent(0)
    }
    fn legal_actions(&self, _s: &St, agent: usize) -> Vec<usize> {
        if agent == 0 {
            vec![0, 1]
        } else {
            Vec::new()
        }
    }
    fn step(&self, s: &St, _a: &[usize]) -> Transition<St, ()> {
        Transition {
            next_state: St { tick: s.tick + 1 },
            events: vec![None],
            terminal: s.tick + 1 >= 3,
        }
    }
    fn initial_state(&self) -> St {
        St { tick: 0 }
    }
}

/// Two-agent round-robin variant for per-player routing.
struct RR;
impl Game for RR {
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
    fn step(&self, s: &St, _a: &[usize]) -> Transition<St, ()> {
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

struct CountingEnc {
    calls: Arc<AtomicUsize>,
}
impl reinfors_core::ActionView for CountingEnc {}
impl StateEncoder for CountingEnc {
    type State = St;
    fn encode(&self, s: &St, agent: usize) -> Vec<f32> {
        self.calls.fetch_add(1, Ordering::Relaxed);
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

/// PPO-shaped one-shot test policy whose sink behavior is configurable, so marked and
/// unmarked variants are otherwise identical (same rng usage, same evaluations).
#[derive(Clone, Copy, PartialEq)]
enum SinkMode {
    Plain,
    Marked,
    /// Marks in round 1, emits an extra plain request in round 2.
    MarkedTwoRound,
    DuplicateMark,
    ForeignPerspectiveMark,
}

struct TestActor {
    mode: SinkMode,
}

struct TestSearch {
    agents: Vec<usize>,
    legal: Vec<Vec<usize>>,
    obs: Vec<Vec<f32>>,
    round: usize,
    results: Vec<PpoEvaluation>,
}

impl Policy for TestActor {
    type Evaluation = PpoEvaluation;
    type PolicyState = ();
    type Search<S: Send> = TestSearch;

    fn max_agents(&self, _sequential: bool) -> Option<usize> {
        Some(usize::MAX)
    }
    fn supports_imperfect_information(&self) -> bool {
        true
    }
    fn begin_episode(&self, _rng: &mut dyn Rng) {}
    fn encode_eval(&self, _eval: &PpoEvaluation, _out: &mut Vec<u8>) {
        unimplemented!("no buffering in this test")
    }
    fn decode_eval(&self, _r: &mut Reader, _n: usize) -> Result<PpoEvaluation, String> {
        unimplemented!("no buffering in this test")
    }
    fn policy_state_to_u64(&self, _s: &()) -> u64 {
        0
    }
    fn policy_state_from_u64(&self, _v: u64) -> Result<(), String> {
        Ok(())
    }

    fn begin_search<G: Game + Sync>(
        &self,
        ctx: SearchCtx<'_, G>,
        state: &G::State,
        perspectives: &[usize],
    ) -> TestSearch
    where
        G::State: Send,
    {
        TestSearch {
            agents: perspectives.to_vec(),
            legal: perspectives
                .iter()
                .map(|&a| ctx.game.legal_actions(state, a))
                .collect(),
            obs: perspectives
                .iter()
                .map(|&a| ctx.enc.encode(state, a))
                .collect(),
            round: 0,
            results: Vec::new(),
        }
    }

    fn round<G: Game + Sync>(
        &self,
        _ctx: SearchCtx<'_, G>,
        search: &mut TestSearch,
        out: &mut RequestSink<'_, G::State>,
    ) -> RoundStatus
    where
        G::State: Send,
    {
        match (search.round, self.mode) {
            (0, SinkMode::Plain) => {
                for (agent, obs) in search.agents.iter().zip(search.obs.iter()) {
                    out.push(*agent, obs);
                }
            }
            (0, SinkMode::Marked) | (0, SinkMode::MarkedTwoRound) => {
                for (agent, obs) in search.agents.iter().zip(search.obs.drain(..)) {
                    out.push_root(*agent, obs, *agent);
                }
            }
            (0, SinkMode::DuplicateMark) => {
                let agent = search.agents[0];
                out.push_root(agent, search.obs[0].clone(), agent);
                out.push_root(agent, search.obs[0].clone(), agent);
            }
            (0, SinkMode::ForeignPerspectiveMark) => {
                out.push_root(search.agents[0], search.obs[0].clone(), 99);
            }
            (1, SinkMode::MarkedTwoRound) => {
                out.push(search.agents[0], &[0.0, 0.0]);
            }
            _ => return RoundStatus::Done,
        }
        search.round += 1;
        RoundStatus::Pending
    }

    fn absorb<G: Game + Sync>(
        &self,
        _ctx: SearchCtx<'_, G>,
        search: &mut TestSearch,
        rows: RowsView<'_>,
    ) where
        G::State: Send,
    {
        if search.round != 1 {
            return;
        }
        search.results = search
            .legal
            .iter()
            .enumerate()
            .map(|(i, legal)| {
                let row = rows.row(i);
                PpoEvaluation {
                    log_probs: masked_log_probs(&row[..legal.len()], legal),
                    value: 0.0,
                    legal: legal.clone(),
                }
            })
            .collect();
    }

    fn finish<G: Game + Sync>(
        &self,
        _ctx: SearchCtx<'_, G>,
        search: TestSearch,
    ) -> Vec<(PpoEvaluation, Vec<reinfors_core::InteriorTarget>)>
    where
        G::State: Send,
    {
        search
            .results
            .into_iter()
            .map(|e| (e, Vec::new()))
            .collect()
    }

    fn select(&self, eval: &PpoEvaluation, _state: &mut (), rng: &mut dyn Rng) -> usize {
        let mut r = rng.unit();
        for (i, lp) in eval.log_probs.iter().enumerate() {
            r -= lp.exp();
            if r <= 0.0 {
                return eval.legal[i];
            }
        }
        eval.legal[0]
    }

    fn fold_telemetry(
        &self,
        _eval: &PpoEvaluation,
        _stats: &mut reinfors_core::rollout::engine::CollectStats,
    ) {
    }
}

fn engine_with<G: Game<State = St, Event = ()> + Sync>(
    game: G,
    mode: SinkMode,
    calls: Arc<AtomicUsize>,
) -> Engine<G, TestActor, reinfors_core::Ppo>
where
    G::State: Send,
{
    Engine::new(
        game,
        Box::new(CountingEnc { calls }),
        Box::new(Zero),
        TestActor { mode },
        reinfors_core::Ppo::new(1.0, 0.95),
        EngineParams {
            n_games: 2,
            seed: 11,
            n_threads: Some(1), // deterministic record order for A/B comparisons
            ..Default::default()
        },
    )
}

fn ppo_infer(obs: Vec<f32>, n: usize) -> Vec<f64> {
    // Two logits + the value the tail bootstrap reads, mildly obs-dependent.
    let dim = obs.len() / n.max(1);
    (0..n)
        .flat_map(|i| [f64::from(obs[i * dim]) * 0.1, 0.3, 0.05])
        .collect()
}

fn record_key(r: &reinfors_core::PpoRecord) -> (usize, Vec<u32>, usize, u64, u64) {
    (
        r.player,
        r.obs.iter().map(|f| f.to_bits()).collect(),
        r.action,
        r.behavior_log_prob.to_bits(),
        r.ret.to_bits(),
    )
}

#[test]
fn marked_and_plain_policies_produce_identical_records() {
    let run = |mode: SinkMode| {
        let calls = Arc::new(AtomicUsize::new(0));
        let mut e = engine_with(Line, mode, calls.clone());
        let (records, stats) = e.collect(12, |obs: Vec<f32>, n: usize| ppo_infer(obs, n));
        let keys: Vec<_> = records.iter().map(record_key).collect();
        (keys, stats.decisions, calls.load(Ordering::Relaxed))
    };
    let (plain, plain_decisions, plain_encodes) = run(SinkMode::Plain);
    let (marked, marked_decisions, marked_encodes) = run(SinkMode::Marked);
    assert_eq!(
        plain, marked,
        "marked rows must be byte-identical to fallback encodes"
    );
    assert_eq!(plain_decisions, marked_decisions);
    // Reuse drops the count by exactly one record encode per decision.
    assert_eq!(plain_encodes, marked_encodes + marked_decisions);
}

#[test]
fn a_mark_emitted_in_an_early_round_survives_later_rounds() {
    let run = |mode: SinkMode| {
        let calls = Arc::new(AtomicUsize::new(0));
        let mut e = engine_with(Line, mode, calls.clone());
        let (records, stats) = e.collect(12, |obs: Vec<f32>, n: usize| ppo_infer(obs, n));
        let keys: Vec<_> = records.iter().map(record_key).collect();
        (keys, stats.decisions, calls.load(Ordering::Relaxed))
    };
    let (one_round, _, one_round_encodes) = run(SinkMode::Marked);
    let (two_round, _, two_round_encodes) = run(SinkMode::MarkedTwoRound);
    assert_eq!(
        one_round, two_round,
        "the extra round must not disturb records"
    );
    assert_eq!(one_round_encodes, two_round_encodes);
}

#[test]
fn per_player_routing_preserves_marks() {
    let calls = Arc::new(AtomicUsize::new(0));
    let mut e = engine_with(RR, SinkMode::Marked, calls.clone());
    let (records, stats) =
        e.collect_routed(12, InferMode::PerPlayer, |_player, obs: Vec<f32>, n| {
            ppo_infer(obs, n)
        });
    assert!(!records.is_empty());
    // Slack above one-per-decision is fragment-cut tail bootstraps.
    let encodes = calls.load(Ordering::Relaxed);
    let decisions = stats.decisions;
    assert!(
        encodes < 2 * decisions,
        "per-player routing lost the mark: {encodes} encodes for {decisions} decisions"
    );
}

#[test]
fn duplicate_marks_panic() {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut e = engine_with(Line, SinkMode::DuplicateMark, Arc::new(AtomicUsize::new(0)));
        e.collect(4, |obs: Vec<f32>, n: usize| ppo_infer(obs, n))
    }));
    let err = match result {
        Ok(_) => panic!("duplicate push_root must fail loudly"),
        Err(e) => e,
    };
    let msg = err
        .downcast_ref::<String>()
        .cloned()
        .unwrap_or_else(|| (*err.downcast_ref::<&str>().unwrap_or(&"")).to_string());
    assert!(msg.contains("twice"), "unexpected panic: {msg}");
}

#[test]
fn foreign_perspective_marks_panic() {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut e = engine_with(
            Line,
            SinkMode::ForeignPerspectiveMark,
            Arc::new(AtomicUsize::new(0)),
        );
        e.collect(4, |obs: Vec<f32>, n: usize| ppo_infer(obs, n))
    }));
    let err = match result {
        Ok(_) => panic!("foreign-perspective push_root must fail loudly"),
        Err(e) => e,
    };
    let msg = err
        .downcast_ref::<String>()
        .cloned()
        .unwrap_or_else(|| (*err.downcast_ref::<&str>().unwrap_or(&"")).to_string());
    assert!(msg.contains("not"), "unexpected panic: {msg}");
}

#[test]
fn builtin_ppo_encodes_once_per_decision() {
    let calls = Arc::new(AtomicUsize::new(0));
    let mut e = Engine::new(
        Line,
        Box::new(CountingEnc {
            calls: calls.clone(),
        }),
        Box::new(Zero),
        PpoActor::new(),
        reinfors_core::Ppo::new(1.0, 0.95),
        EngineParams {
            n_games: 2,
            seed: 3,
            ..Default::default()
        },
    );
    let (records, stats) = e.collect(12, |_obs: Vec<f32>, n: usize| vec![0.0; n * 3]);
    assert!(!records.is_empty());
    let encodes = calls.load(Ordering::Relaxed);
    let decisions = stats.decisions;
    assert!(
        encodes >= decisions && encodes <= decisions + 2,
        "expected ~one encode per decision, got {encodes} for {decisions} decisions"
    );
}

#[test]
fn builtin_dqn_chains_successor_obs_single_agent() {
    let calls = Arc::new(AtomicUsize::new(0));
    let mut e = Engine::new(
        Line,
        Box::new(CountingEnc {
            calls: calls.clone(),
        }),
        Box::new(Zero),
        reinfors_core::EpsilonGreedyQ::new(1, 0.5),
        reinfors_core::Dqn::new(1, 1.0, 1, 0.99),
        EngineParams {
            n_games: 2,
            seed: 3,
            ..Default::default()
        },
    );
    let (records, stats) = e.collect(12, |_obs: Vec<f32>, n: usize| vec![0.0; n * 3]);
    assert!(!records.is_empty());
    let encodes = calls.load(Ordering::Relaxed);
    let decisions = stats.decisions;
    assert!(
        encodes < 2 * decisions,
        "chaining regressed: {encodes} encodes for {decisions} decisions"
    );
    for r in &records {
        assert!(
            r.terminal || !r.next_obs.is_empty(),
            "missing successor obs"
        );
    }
}

/// Action-dependent episode length: action 0 after the first tick terminates, so
/// two slots finish at different times and a collect boundary can cut one mid-episode.
struct VarLine;
impl Game for VarLine {
    type State = St;
    type Event = ();
    fn num_agents(&self) -> usize {
        1
    }
    fn action_count(&self) -> usize {
        2
    }
    fn actor(&self, _s: &St) -> Actor {
        Actor::Agent(0)
    }
    fn legal_actions(&self, _s: &St, agent: usize) -> Vec<usize> {
        if agent == 0 {
            vec![0, 1]
        } else {
            Vec::new()
        }
    }
    fn step(&self, s: &St, a: &[usize]) -> Transition<St, ()> {
        Transition {
            next_state: St { tick: s.tick + 1 },
            events: vec![None],
            terminal: s.tick >= 1 && a[0] == 0,
        }
    }
    fn initial_state(&self) -> St {
        St { tick: 0 }
    }
    fn truncation_horizon(&self) -> Option<usize> {
        Some(7)
    }
}

/// Line variant with a truncation horizon and no natural terminal.
struct TruncLine;
impl Game for TruncLine {
    type State = St;
    type Event = ();
    fn num_agents(&self) -> usize {
        1
    }
    fn action_count(&self) -> usize {
        2
    }
    fn actor(&self, _s: &St) -> Actor {
        Actor::Agent(0)
    }
    fn legal_actions(&self, _s: &St, agent: usize) -> Vec<usize> {
        if agent == 0 {
            vec![0, 1]
        } else {
            Vec::new()
        }
    }
    fn step(&self, s: &St, _a: &[usize]) -> Transition<St, ()> {
        Transition {
            next_state: St { tick: s.tick + 1 },
            events: vec![None],
            terminal: false,
        }
    }
    fn initial_state(&self) -> St {
        St { tick: 0 }
    }
    fn truncation_horizon(&self) -> Option<usize> {
        Some(4)
    }
}

/// `Dqn` with chaining disabled: the legacy every-step-encodes path.
struct NoChain(reinfors_core::Dqn);
impl reinfors_core::Learner<reinfors_core::QEvaluation> for NoChain {
    type Record = reinfors_core::DqnRecord;
    fn needs_next_obs(&self) -> bool {
        true
    }
    fn eval_records(
        &self,
        e: &reinfors_core::QEvaluation,
        t: Vec<reinfors_core::InteriorTarget>,
        view: &dyn reinfors_core::ActionView,
        agent: usize,
        rng: &mut dyn Rng,
    ) -> Vec<reinfors_core::DqnRecord> {
        self.0.eval_records(e, t, view, agent, rng)
    }
    fn episode_records(
        &self,
        trajectory: &[reinfors_core::Step<reinfors_core::QEvaluation>],
        tail: &[f64],
        view: &dyn reinfors_core::ActionView,
        agent: usize,
        rng: &mut dyn Rng,
    ) -> Vec<reinfors_core::DqnRecord> {
        self.0.episode_records(trajectory, tail, view, agent, rng)
    }
}

fn dqn_key(r: &reinfors_core::DqnRecord) -> Vec<u64> {
    let mut k = vec![
        r.player as u64,
        r.action as u64,
        r.reward.to_bits(),
        u64::from(r.terminal),
        r.discount.to_bits(),
    ];
    k.extend(r.obs.iter().map(|f| u64::from(f.to_bits())));
    k.push(u64::MAX);
    k.extend(r.next_obs.iter().map(|f| u64::from(f.to_bits())));
    k.push(u64::MAX);
    k.extend(r.mask.iter().map(|f| u64::from(f.to_bits())));
    k.extend(r.legal.iter().map(|&v| v as u64));
    k.push(u64::MAX);
    k.extend(r.next_legal.iter().map(|&v| v as u64));
    k
}

// Two slots so a collect boundary can leave one trajectory live mid-episode.
#[allow(clippy::type_complexity)]
fn dqn_collect<G, L>(game: G, learner: L, floors: &[usize]) -> (Vec<Vec<u64>>, Vec<(usize, usize)>)
where
    G: Game<State = St, Event = ()> + Sync,
    L: reinfors_core::Learner<reinfors_core::QEvaluation, Record = reinfors_core::DqnRecord> + Sync,
{
    let mut e = Engine::new(
        game,
        Box::new(CountingEnc {
            calls: Arc::new(AtomicUsize::new(0)),
        }),
        Box::new(Zero),
        reinfors_core::EpsilonGreedyQ::new(1, 0.5),
        learner,
        EngineParams {
            n_games: 2,
            seed: 9,
            n_threads: Some(1),
            ..Default::default()
        },
    );
    let mut keys = Vec::new();
    let mut per_collect = Vec::new();
    for &floor in floors {
        let (records, stats) = e.collect(floor, |_obs: Vec<f32>, n: usize| vec![0.0; n * 2]);
        per_collect.push((records.len(), stats.decisions));
        keys.extend(records.iter().map(dqn_key));
    }
    (keys, per_collect)
}

#[test]
fn chained_dqn_records_match_the_legacy_path_bytewise() {
    let dqn = || reinfors_core::Dqn::new(1, 1.0, 1, 0.99);
    // Terminals; a truncation boundary; a mid-episode cut spanning two collects.
    assert_eq!(
        dqn_collect(Line, dqn(), &[12]).0,
        dqn_collect(Line, NoChain(dqn()), &[12]).0
    );
    assert_eq!(
        dqn_collect(TruncLine, dqn(), &[8]).0,
        dqn_collect(TruncLine, NoChain(dqn()), &[8]).0
    );
    let (chained, shape) = dqn_collect(VarLine, dqn(), &[3, 3]);
    let (legacy, _) = dqn_collect(VarLine, NoChain(dqn()), &[3, 3]);
    assert_eq!(chained, legacy);
    // Collect #1 must cut mid-episode: more decisions than emitted one-step records
    // means a trajectory crossed the boundary and finished in collect #2.
    let (emitted, decisions) = shape[0];
    assert!(
        decisions > emitted,
        "no live trajectory at the boundary: {decisions} decisions, {emitted} records"
    );
}

#[test]
fn needs_next_obs_without_chaining_still_receives_every_row() {
    let mut e = Engine::new(
        Line,
        Box::new(CountingEnc {
            calls: Arc::new(AtomicUsize::new(0)),
        }),
        Box::new(Zero),
        reinfors_core::EpsilonGreedyQ::new(1, 0.5),
        AssertNextObs,
        EngineParams {
            n_games: 2,
            seed: 9,
            n_threads: Some(1),
            ..Default::default()
        },
    );
    let (records, _) = e.collect(4, |_obs: Vec<f32>, n: usize| vec![0.0; n * 2]);
    assert!(records.iter().sum::<usize>() >= 4);
}

/// Asserts the public contract directly: every buffered step arrives with its
/// post-transition observation, no chaining assumed.
struct AssertNextObs;
impl reinfors_core::Learner<reinfors_core::QEvaluation> for AssertNextObs {
    type Record = usize;
    fn needs_next_obs(&self) -> bool {
        true
    }
    fn eval_records(
        &self,
        _e: &reinfors_core::QEvaluation,
        _t: Vec<reinfors_core::InteriorTarget>,
        _view: &dyn reinfors_core::ActionView,
        _agent: usize,
        _rng: &mut dyn Rng,
    ) -> Vec<usize> {
        Vec::new()
    }
    fn episode_records(
        &self,
        trajectory: &[reinfors_core::Step<reinfors_core::QEvaluation>],
        _tail: &[f64],
        _view: &dyn reinfors_core::ActionView,
        _agent: usize,
        _rng: &mut dyn Rng,
    ) -> Vec<usize> {
        for (i, step) in trajectory.iter().enumerate() {
            assert!(
                !step.next_obs.is_empty(),
                "step {i} of {} lost its post-transition observation",
                trajectory.len()
            );
        }
        vec![trajectory.len()]
    }
}
