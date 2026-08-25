//! Stage-1 zero-copy scheduler: records match the classic path byte-for-byte,
//! collection is deterministic, rounds split across arenas, per-player routing
//! holds, truncation tails ride worker tasks, and callback panics unwind cleanly.

use std::sync::{Arc, Mutex};

use reinfors_core::codec::bytes::Reader;
use reinfors_core::policies::modelfree::ppo::masked_log_probs;
use reinfors_core::policy::{RequestSink, RoundStatus, RowsView, SearchCtx};
use reinfors_core::rollout::engine::{Engine, EngineParams};
use reinfors_core::rollout::evaluator::InferMode;
use reinfors_core::{
    Actor, Game, Policy, PpoEvaluation, Reward, Rng, Space, StateEncoder, Transition,
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

/// Never-terminal single-agent line truncated by the horizon: exercises tails.
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

/// Two-agent round-robin for per-player routing.
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
            events: vec![None, None],
            terminal: s.tick + 1 >= 6,
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
        Space::Box {
            shape: vec![1, 1, 2],
            low: f32::NEG_INFINITY,
            high: f32::INFINITY,
        }
    }
}

struct Zero;
impl Reward for Zero {
    type Event = ();
    fn step_reward(&self, _e: &(), _agent: usize) -> f64 {
        0.0
    }
}

/// One-round PPO-shaped policy emitting `fan` rows per perspective; extra rows
/// beyond the first are inference ballast that exercises arena splitting.
struct FanActor {
    fan: usize,
}

struct FanSearch {
    agents: Vec<usize>,
    legal: Vec<Vec<usize>>,
    obs: Vec<Vec<f32>>,
    round: usize,
    results: Vec<PpoEvaluation>,
}

impl Policy for FanActor {
    type Evaluation = PpoEvaluation;
    type PolicyState = ();
    type Search<S: Send> = FanSearch;

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
    ) -> FanSearch
    where
        G::State: Send,
    {
        FanSearch {
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
        search: &mut FanSearch,
        out: &mut RequestSink,
    ) -> RoundStatus
    where
        G::State: Send,
    {
        if search.round > 0 {
            return RoundStatus::Done;
        }
        for (agent, obs) in search.agents.iter().zip(search.obs.iter()) {
            out.push_root(*agent, obs.clone(), *agent);
            for extra in 1..self.fan {
                let mut ballast = obs.clone();
                ballast[0] += 1000.0 * extra as f32;
                out.push(*agent, &ballast);
            }
        }
        search.round += 1;
        RoundStatus::Pending
    }

    fn absorb<G: Game + Sync>(
        &self,
        _ctx: SearchCtx<'_, G>,
        search: &mut FanSearch,
        rows: RowsView<'_>,
    ) where
        G::State: Send,
    {
        search.results = search
            .legal
            .iter()
            .enumerate()
            .map(|(i, legal)| {
                let row = rows.row(i * self.fan);
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
        search: FanSearch,
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
}

fn engine_with<G: Game<State = St, Event = ()> + Sync>(
    game: G,
    fan: usize,
    batch_size: usize,
    zero_copy: bool,
) -> Engine<G, FanActor, reinfors_core::Ppo>
where
    G::State: Send,
{
    Engine::new(
        game,
        Box::new(Enc),
        Box::new(Zero),
        FanActor { fan },
        reinfors_core::Ppo::new(1.0, 0.95),
        EngineParams {
            n_games: 2,
            seed: 11,
            n_threads: Some(1),
            batch_size: Some(batch_size),
            zero_copy,
            ..Default::default()
        },
    )
}

fn ppo_infer(obs: &[f32], n: usize) -> Vec<f64> {
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

type Keys = Vec<(usize, Vec<u32>, usize, u64, u64)>;
type RoutedBatches = Arc<Mutex<Vec<(usize, Vec<f32>, usize)>>>;

fn run_shared<G: Game<State = St, Event = ()> + Sync>(
    game: impl Fn() -> G,
    fan: usize,
    batch_size: usize,
    n_records: usize,
    zero_copy: bool,
) -> (
    Keys,
    reinfors_core::rollout::engine::CollectStats,
    Vec<usize>,
)
where
    G::State: Send,
{
    let batches = Arc::new(Mutex::new(Vec::new()));
    let seen = batches.clone();
    let mut e = engine_with(game(), fan, batch_size, zero_copy);
    let (records, stats) = e.collect(n_records, move |obs: Vec<f32>, n: usize| {
        seen.lock().unwrap().push(n);
        ppo_infer(&obs, n)
    });
    let keys = records.iter().map(record_key).collect();
    let batches = batches.lock().unwrap().clone();
    (keys, stats, batches)
}

#[test]
fn zero_copy_matches_the_classic_path() {
    let (classic, classic_stats, _) = run_shared(|| Line, 1, 2, 12, false);
    let (zero, zero_stats, _) = run_shared(|| Line, 1, 2, 12, true);
    assert!(!classic.is_empty());
    assert_eq!(classic, zero, "records must be byte-identical across paths");
    assert_eq!(classic_stats.decisions, zero_stats.decisions);
    assert_eq!(
        classic_stats.sum_requested_rows,
        zero_stats.sum_requested_rows
    );
    assert_eq!(classic_stats.infer_rows, zero_stats.infer_rows);
}

#[test]
fn zero_copy_is_deterministic() {
    let (a, _, batches_a) = run_shared(|| Line, 3, 4, 12, true);
    let (b, _, batches_b) = run_shared(|| Line, 3, 4, 12, true);
    assert_eq!(a, b, "fixed-seed zero-copy runs must be byte-identical");
    assert_eq!(batches_a, batches_b, "fire cadence must be deterministic");
}

#[test]
fn rounds_split_across_arenas() {
    // fan 5 > batch_size 4: every round crosses an arena boundary.
    let (classic, _, _) = run_shared(|| Line, 5, 4, 12, false);
    let (zero, zero_stats, batches) = run_shared(|| Line, 5, 4, 12, true);
    assert_eq!(classic, zero, "split rounds must not disturb records");
    assert!(
        batches.iter().filter(|&&n| n == 4).count() >= 2,
        "capacity fires must dominate: {batches:?}"
    );
    assert!(
        batches.iter().all(|&n| n <= 4),
        "no fire may exceed the arena capacity: {batches:?}"
    );
    assert_eq!(batches.iter().sum::<usize>(), zero_stats.infer_rows);
}

#[test]
fn truncation_tails_ride_worker_tasks() {
    let (classic, classic_stats, _) = run_shared(|| TruncLine, 1, 2, 8, false);
    let (zero, zero_stats, _) = run_shared(|| TruncLine, 1, 2, 8, true);
    assert_eq!(classic, zero, "tail-bootstrapped records must match");
    assert!(zero_stats.sum_tail_rows > 0, "horizon must produce tails");
    assert_eq!(classic_stats.sum_tail_rows, zero_stats.sum_tail_rows);
}

#[test]
fn per_player_routing_stays_partitioned() {
    let run = |zero_copy: bool| {
        let batches: RoutedBatches = Arc::new(Mutex::new(Vec::new()));
        let seen = batches.clone();
        let mut e = engine_with(RR, 2, 3, zero_copy);
        let (records, _) =
            e.collect_routed(10, InferMode::PerPlayer, move |player, obs: Vec<f32>, n| {
                seen.lock().unwrap().push((player, obs.clone(), n));
                ppo_infer(&obs, n)
            });
        let keys: Keys = records.iter().map(record_key).collect();
        let seen = batches.lock().unwrap().clone();
        (keys, seen)
    };
    let (classic, _) = run(false);
    let (zero, zero_batches) = run(true);
    assert_eq!(classic, zero, "per-player records must match across paths");
    for (player, obs, n) in zero_batches {
        for i in 0..n {
            assert_eq!(
                obs[i * 2 + 1],
                player as f32,
                "row routed to the wrong player's callback"
            );
        }
    }
}

#[test]
fn callback_panics_unwind_cleanly() {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut e = engine_with(Line, 1, 2, true);
        let mut calls = 0;
        e.collect(12, move |obs: Vec<f32>, n: usize| {
            calls += 1;
            assert!(calls < 2, "die on the second fire");
            ppo_infer(&obs, n)
        })
    }));
    assert!(result.is_err(), "the callback panic must surface");
}
