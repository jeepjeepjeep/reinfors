//! Stage-1 zero-copy scheduler vs the classic path.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use reinfors_core::codec::bytes::Reader;
use reinfors_core::policies::modelfree::ppo::masked_log_probs;
use reinfors_core::policy::{RequestSink, RoundStatus, RowsView, SearchCtx};
use reinfors_core::rollout::engine::{Engine, EngineParams};
use reinfors_core::rollout::evaluator::InferMode;
use reinfors_core::rollout::infer_cache::InferCache;
use reinfors_core::{
    Actor, CacheHasher, Game, Policy, PpoEvaluation, Reward, Rng, Space, StateEncoder, Transition,
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

/// PPO-shaped policy; rows beyond the first per perspective are ballast that
/// exercises arena splitting, extra rounds exercise the Blocked/Resume cycle.
struct FanActor {
    fan: usize,
    rounds: usize,
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
        out: &mut RequestSink<'_, G::State>,
    ) -> RoundStatus
    where
        G::State: Send,
    {
        match search.round {
            0 => {
                for (agent, obs) in search.agents.iter().zip(search.obs.iter()) {
                    out.push_root(*agent, obs.clone(), *agent);
                    for extra in 1..self.fan {
                        let mut ballast = obs.clone();
                        ballast[0] += 1000.0 * extra as f32;
                        out.push(*agent, &ballast);
                    }
                }
            }
            r if r < self.rounds => {
                for agent in &search.agents {
                    out.push(*agent, &[9000.0 + r as f32, *agent as f32]);
                }
            }
            _ => return RoundStatus::Done,
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
        if search.round != 1 {
            return;
        }
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
    rounds: usize,
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
        FanActor { fan, rounds },
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
    rounds: usize,
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
    let mut e = engine_with(game(), fan, rounds, batch_size, zero_copy);
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
    let (classic, classic_stats, _) = run_shared(|| Line, 1, 1, 2, 12, false);
    let (zero, zero_stats, _) = run_shared(|| Line, 1, 1, 2, 12, true);
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
    let (a, _, batches_a) = run_shared(|| Line, 3, 1, 4, 12, true);
    let (b, _, batches_b) = run_shared(|| Line, 3, 1, 4, 12, true);
    assert_eq!(a, b, "fixed-seed zero-copy runs must be byte-identical");
    assert_eq!(batches_a, batches_b, "fire cadence must be deterministic");
}

#[test]
fn rounds_split_across_arenas() {
    let (classic, _, _) = run_shared(|| Line, 5, 1, 4, 12, false);
    let (zero, zero_stats, batches) = run_shared(|| Line, 5, 1, 4, 12, true);
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
    let (classic, classic_stats, _) = run_shared(|| TruncLine, 1, 1, 2, 8, false);
    let (zero, zero_stats, _) = run_shared(|| TruncLine, 1, 1, 2, 8, true);
    assert_eq!(classic, zero, "tail-bootstrapped records must match");
    assert!(zero_stats.sum_tail_rows > 0, "horizon must produce tails");
    assert_eq!(classic_stats.sum_tail_rows, zero_stats.sum_tail_rows);
}

#[test]
fn per_player_routing_stays_partitioned() {
    let run = |zero_copy: bool| {
        let batches: RoutedBatches = Arc::new(Mutex::new(Vec::new()));
        let seen = batches.clone();
        let mut e = engine_with(RR, 2, 1, 3, zero_copy);
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
        let mut e = engine_with(Line, 1, 1, 2, true);
        let mut calls = 0;
        e.collect(12, move |obs: Vec<f32>, n: usize| {
            calls += 1;
            assert!(calls < 2, "die on the second fire");
            ppo_infer(&obs, n)
        })
    }));
    assert!(result.is_err(), "the callback panic must surface");
}

#[test]
fn multi_round_searches_match_the_classic_path() {
    let (classic, _, _) = run_shared(|| Line, 3, 2, 4, 12, false);
    let (zero, _, _) = run_shared(|| Line, 3, 2, 4, 12, true);
    assert!(!classic.is_empty());
    assert_eq!(classic, zero, "multi-round records must match across paths");
}

#[test]
fn dqn_chained_records_match() {
    fn key(r: &reinfors_core::DqnRecord) -> (usize, Vec<u32>, usize, u64, Vec<u32>, bool, u64) {
        (
            r.player,
            r.obs.iter().map(|f| f.to_bits()).collect(),
            r.action,
            r.reward.to_bits(),
            r.next_obs.iter().map(|f| f.to_bits()).collect(),
            r.terminal,
            r.discount.to_bits(),
        )
    }
    let run = |zero_copy: bool| {
        let mut e = Engine::new(
            Line,
            Box::new(Enc),
            Box::new(Zero),
            reinfors_core::EpsilonGreedyQ::new(1, 0.25),
            reinfors_core::Dqn::new(1, 1.0, 1, 0.99),
            EngineParams {
                n_games: 2,
                seed: 7,
                n_threads: Some(1),
                batch_size: Some(2),
                zero_copy,
                ..Default::default()
            },
        );
        let (records, _) = e.collect(10, |obs: Vec<f32>, n: usize| {
            let dim = obs.len() / n.max(1);
            (0..n)
                .flat_map(|i| [f64::from(obs[i * dim]) * 0.1, 0.2])
                .collect()
        });
        records.iter().map(key).collect::<Vec<_>>()
    };
    let classic = run(false);
    assert!(!classic.is_empty());
    assert_eq!(
        classic,
        run(true),
        "chained DQN records must match across paths"
    );
}

#[test]
fn multi_threaded_stress_races_arena_replacement() {
    let batches = Arc::new(Mutex::new(Vec::new()));
    let seen = batches.clone();
    let mut e = Engine::new(
        TruncLine,
        Box::new(Enc),
        Box::new(Zero),
        FanActor { fan: 7, rounds: 2 },
        reinfors_core::Ppo::new(1.0, 0.95),
        EngineParams {
            n_games: 8,
            seed: 3,
            n_threads: Some(4),
            batch_size: Some(4),
            zero_copy: true,
            ..Default::default()
        },
    );
    let (records, stats) = e.collect(300, move |obs: Vec<f32>, n: usize| {
        seen.lock().unwrap().push(n);
        ppo_infer(&obs, n)
    });
    assert!(records.len() >= 300);
    let batches = batches.lock().unwrap().clone();
    assert!(batches.iter().all(|&n| n <= 4), "fires exceed capacity");
    assert_eq!(batches.iter().sum::<usize>(), stats.infer_rows);
}

#[test]
fn callback_panic_with_tails_in_flight_leaves_the_engine_reusable() {
    for panic_at in 1..=10 {
        let mut e = engine_with(TruncLine, 1, 1, 1, true);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut calls = 0;
            e.collect(24, move |obs: Vec<f32>, n: usize| {
                calls += 1;
                assert!(calls != panic_at, "die at fire {panic_at}");
                ppo_infer(&obs, n)
            })
        }));
        if result.is_ok() {
            continue;
        }
        let (_, stats) = e.collect(12, |obs: Vec<f32>, n: usize| ppo_infer(&obs, n));
        for ep in &stats.episodes {
            assert!(
                ep.length <= 4,
                "slot stranded over the horizon after abort at fire {panic_at}: length {}",
                ep.length
            );
        }
    }
}

#[test]
fn noop_collect_allocates_nothing_and_the_engine_reuses() {
    let mut e = engine_with(Line, 1, 1, 2, true);
    let (records, stats) = e.collect(0, |obs: Vec<f32>, n: usize| ppo_infer(&obs, n));
    assert!(records.is_empty());
    assert_eq!(stats.infer_calls, 0);
    let (records, _) = e.collect(6, |obs: Vec<f32>, n: usize| ppo_infer(&obs, n));
    assert!(!records.is_empty());
}

/// Constant observation per agent: every decision after the first is a cache hit.
struct ConstEnc;
impl reinfors_core::ActionView for ConstEnc {}
impl StateEncoder for ConstEnc {
    type State = St;
    fn encode(&self, _s: &St, agent: usize) -> Vec<f32> {
        vec![1.0, agent as f32]
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

/// f32-exact outputs so cached (f32-stored) rows are bit-identical to fresh ones.
fn exact_infer(obs: &[f32], n: usize) -> Vec<f64> {
    let dim = obs.len() / n.max(1);
    (0..n)
        .flat_map(|i| [f64::from(obs[i * dim]) * 0.25, 0.5, 0.25])
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn engine_full<G: Game<State = St, Event = ()> + Sync>(
    game: G,
    enc: Box<dyn StateEncoder<State = St>>,
    fan: usize,
    rounds: usize,
    n_games: usize,
    batch_size: usize,
    zero_copy: bool,
    cache: Option<usize>,
) -> (
    Engine<G, FanActor, reinfors_core::Ppo>,
    Option<Arc<AtomicU64>>,
)
where
    G::State: Send,
{
    let n_agents = game.num_agents();
    let mut e = Engine::new(
        game,
        enc,
        Box::new(Zero),
        FanActor { fan, rounds },
        reinfors_core::Ppo::new(1.0, 0.95),
        EngineParams {
            n_games,
            seed: 11,
            n_threads: Some(1),
            batch_size: Some(batch_size),
            zero_copy,
            ..Default::default()
        },
    );
    let mut generation = None;
    if let Some(cap) = cache {
        let shared = Arc::new(AtomicU64::new(0));
        e = e.with_infer_caches(
            (0..=n_agents)
                .map(|_| InferCache::new(cap, shared.clone()))
                .collect(),
        );
        generation = Some(shared);
    }
    (e, generation)
}

#[test]
fn cached_records_match_uncached_and_skip_inference() {
    let run = |cache: Option<usize>| {
        let (mut e, _) = engine_full(Line, Box::new(ConstEnc), 1, 1, 1, 1, true, cache);
        let (records, stats) = e.collect(9, |obs: Vec<f32>, n: usize| exact_infer(&obs, n));
        (records.iter().map(record_key).collect::<Keys>(), stats)
    };
    let (plain, pstats) = run(None);
    let (cached, cstats) = run(Some(64));
    assert!(!plain.is_empty());
    assert_eq!(plain, cached, "cache hits must not change record content");
    assert!(cstats.cache_hits > 0);
    assert!(
        cstats.infer_rows < pstats.infer_rows,
        "hits must skip inference"
    );
}

#[test]
fn cached_zero_copy_matches_the_classic_cached_path() {
    let run = |zero_copy: bool| {
        let (mut e, _) = engine_full(Line, Box::new(ConstEnc), 1, 1, 1, 1, zero_copy, Some(64));
        let (records, stats) = e.collect(9, |obs: Vec<f32>, n: usize| exact_infer(&obs, n));
        (records.iter().map(record_key).collect::<Keys>(), stats)
    };
    let (classic, classic_stats) = run(false);
    let (zero, zero_stats) = run(true);
    assert_eq!(classic, zero, "cached records must match across paths");
    assert_eq!(classic_stats.cache_hits, zero_stats.cache_hits);
    assert_eq!(classic_stats.infer_rows, zero_stats.infer_rows);
}

#[test]
fn cached_collection_is_deterministic() {
    let run = || {
        let (mut e, _) = engine_full(TruncLine, Box::new(ConstEnc), 2, 2, 2, 3, true, Some(64));
        let (records, stats) = e.collect(20, |obs: Vec<f32>, n: usize| exact_infer(&obs, n));
        (
            records.iter().map(record_key).collect::<Keys>(),
            stats.infer_rows,
            stats.cache_hits,
        )
    };
    assert_eq!(run(), run());
}

#[test]
fn generation_bump_mid_collect_demotes_and_stays_correct() {
    let (plain, _) = {
        let (mut e, _) = engine_full(Line, Box::new(ConstEnc), 1, 1, 2, 1, true, None);
        let (records, stats) = e.collect(9, |obs: Vec<f32>, n: usize| exact_infer(&obs, n));
        (records.iter().map(record_key).collect::<Keys>(), stats)
    };
    for bump_at in 1..=4 {
        let (mut e, generation) = engine_full(Line, Box::new(ConstEnc), 1, 1, 2, 1, true, Some(64));
        let generation = generation.expect("cache installed");
        let mut calls = 0;
        let (records, _) = e.collect(9, move |obs: Vec<f32>, n: usize| {
            calls += 1;
            if calls == bump_at {
                generation.fetch_add(1, Ordering::Relaxed);
            }
            exact_infer(&obs, n)
        });
        let keys: Keys = records.iter().map(record_key).collect();
        assert_eq!(plain, keys, "bump at call {bump_at} corrupted records");
    }
}

#[test]
fn per_player_cached_routing_matches_uncached() {
    let run = |cache: Option<usize>| {
        let (mut e, _) = engine_full(RR, Box::new(Enc), 1, 1, 2, 3, true, cache);
        let (records, stats) =
            e.collect_routed(16, InferMode::PerPlayer, |_player, obs: Vec<f32>, n| {
                exact_infer(&obs, n)
            });
        (records.iter().map(record_key).collect::<Keys>(), stats)
    };
    let (mut plain, _) = run(None);
    let (mut cached, cstats) = run(Some(64));
    // Gating legitimately reorders cross-slot flushes at the cut: compare multisets.
    plain.sort();
    cached.sort();
    assert_eq!(plain, cached);
    assert!(cstats.cache_hits > 0, "repeated states must hit per route");
}

#[test]
fn the_cache_persists_across_collects() {
    let (mut e, _) = engine_full(Line, Box::new(ConstEnc), 1, 1, 1, 1, true, Some(64));
    let (_, first) = e.collect(3, |obs: Vec<f32>, n: usize| exact_infer(&obs, n));
    assert!(first.infer_calls > 0);
    let (records, second) = e.collect(3, |obs: Vec<f32>, n: usize| exact_infer(&obs, n));
    assert!(!records.is_empty());
    assert_eq!(
        second.infer_calls, 0,
        "every row must hit the persisted cache"
    );
    assert!(second.cache_hits > 0);
}

#[test]
fn reinstalling_caches_discards_the_old_zero_copy_cache() {
    let (mut e, _) = engine_full(Line, Box::new(ConstEnc), 1, 1, 1, 1, true, Some(64));
    let (r1, s1) = e.collect(3, |obs: Vec<f32>, n: usize| exact_infer(&obs, n));
    assert!(s1.infer_calls > 0);
    let shared = Arc::new(AtomicU64::new(0));
    let mut e = e.with_infer_caches(
        (0..=1)
            .map(|_| InferCache::new(64, shared.clone()))
            .collect(),
    );
    let (r2, s2) = e.collect(3, |obs: Vec<f32>, n: usize| {
        exact_infer(&obs, n).iter().map(|v| v * 2.0).collect()
    });
    assert!(
        s2.infer_calls > 0,
        "the replaced cache served the old collection's values"
    );
    let k1: Keys = r1.iter().map(record_key).collect();
    let k2: Keys = r2.iter().map(record_key).collect();
    assert_ne!(k1, k2, "records must reflect the new callback's outputs");
}

/// Counts encode calls; `cache_key` streams the exact observation identity.
struct KeyedEnc {
    encodes: Arc<AtomicU64>,
    keyed: bool,
}
impl reinfors_core::ActionView for KeyedEnc {}
impl StateEncoder for KeyedEnc {
    type State = St;
    fn encode(&self, s: &St, agent: usize) -> Vec<f32> {
        self.encodes.fetch_add(1, Ordering::Relaxed);
        vec![(s.tick % 2) as f32, agent as f32]
    }
    fn encode_into(&self, s: &St, agent: usize, dst: &mut [f32]) {
        self.encodes.fetch_add(1, Ordering::Relaxed);
        dst[0] = (s.tick % 2) as f32;
        dst[1] = agent as f32;
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
    fn cache_key(&self, s: &St, _perspective: usize, hasher: &mut CacheHasher) -> bool {
        if !self.keyed {
            return false;
        }
        hasher.write_u64((s.tick % 2) as u64);
        true
    }
}

/// One-round policy emitting the root row plus `fan - 1` state-backed requests.
struct StateActor {
    fan: usize,
}

struct StateSearch<S> {
    state: S,
    agents: Vec<usize>,
    legal: Vec<Vec<usize>>,
    round: usize,
    results: Vec<PpoEvaluation>,
}

impl Policy for StateActor {
    type Evaluation = PpoEvaluation;
    type PolicyState = ();
    type Search<S: Send> = StateSearch<S>;

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
    ) -> StateSearch<G::State>
    where
        G::State: Send,
    {
        StateSearch {
            state: state.clone(),
            agents: perspectives.to_vec(),
            legal: perspectives
                .iter()
                .map(|&a| ctx.game.legal_actions(state, a))
                .collect(),
            round: 0,
            results: Vec::new(),
        }
    }

    fn round<G: Game + Sync>(
        &self,
        ctx: SearchCtx<'_, G>,
        search: &mut StateSearch<G::State>,
        out: &mut RequestSink<'_, G::State>,
    ) -> RoundStatus
    where
        G::State: Send,
    {
        if search.round > 0 {
            return RoundStatus::Done;
        }
        for agent in search.agents.clone() {
            out.push_root(agent, ctx.enc.encode(&search.state, agent), agent);
            for _ in 1..self.fan {
                out.push_state(ctx.enc, agent, &search.state);
            }
        }
        search.round += 1;
        RoundStatus::Pending
    }

    fn absorb<G: Game + Sync>(
        &self,
        _ctx: SearchCtx<'_, G>,
        search: &mut StateSearch<G::State>,
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
        search: StateSearch<G::State>,
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

type StateEngineParts<G> = (
    Engine<G, StateActor, reinfors_core::Ppo>,
    Arc<AtomicU64>,
    Option<Arc<AtomicU64>>,
);

#[allow(clippy::too_many_arguments)]
fn state_engine<G: Game<State = St, Event = ()> + Sync>(
    game: G,
    keyed: bool,
    fan: usize,
    n_games: usize,
    batch_size: usize,
    zero_copy: bool,
    cache: Option<usize>,
) -> StateEngineParts<G>
where
    G::State: Send,
{
    let encodes = Arc::new(AtomicU64::new(0));
    let n_agents = game.num_agents();
    let mut e = Engine::new(
        game,
        Box::new(KeyedEnc {
            encodes: encodes.clone(),
            keyed,
        }),
        Box::new(Zero),
        StateActor { fan },
        reinfors_core::Ppo::new(1.0, 0.95),
        EngineParams {
            n_games,
            seed: 11,
            n_threads: Some(1),
            batch_size: Some(batch_size),
            zero_copy,
            ..Default::default()
        },
    );
    let mut generation = None;
    if let Some(cap) = cache {
        let shared = Arc::new(AtomicU64::new(0));
        e = e.with_infer_caches(
            (0..=n_agents)
                .map(|_| InferCache::new(cap, shared.clone()))
                .collect(),
        );
        generation = Some(shared);
    }
    (e, encodes, generation)
}

#[test]
fn push_state_matches_the_classic_path() {
    let run = |zero_copy: bool| {
        let (mut e, _, _) = state_engine(Line, false, 3, 2, 4, zero_copy, None);
        let (records, stats) = e.collect(12, |obs: Vec<f32>, n: usize| exact_infer(&obs, n));
        (records.iter().map(record_key).collect::<Keys>(), stats)
    };
    let (classic, cs) = run(false);
    let (zero, zs) = run(true);
    assert!(!classic.is_empty());
    assert_eq!(classic, zero, "state-backed rows must match across paths");
    assert_eq!(cs.infer_rows, zs.infer_rows);
}

#[test]
fn encoder_key_hits_skip_the_encoder() {
    let run = |cache: Option<usize>| {
        let (mut e, encodes, _) = state_engine(Line, true, 4, 1, 1, true, cache);
        let (records, stats) = e.collect(9, |obs: Vec<f32>, n: usize| exact_infer(&obs, n));
        (
            records.iter().map(record_key).collect::<Keys>(),
            stats,
            encodes.load(Ordering::Relaxed),
        )
    };
    let (plain, _, plain_encodes) = run(None);
    let (cached, cstats, cached_encodes) = run(Some(64));
    assert_eq!(plain, cached, "encoder-key hits must not change records");
    assert!(cstats.cache_hits > 0);
    assert!(
        cached_encodes < plain_encodes,
        "validated non-root hits must skip encoding: {cached_encodes} vs {plain_encodes}"
    );
}

#[test]
fn scratch_fallback_hits_match_uncached() {
    let run = |cache: Option<usize>| {
        let (mut e, _, _) = state_engine(Line, false, 4, 1, 1, true, cache);
        let (records, stats) = e.collect(9, |obs: Vec<f32>, n: usize| exact_infer(&obs, n));
        (records.iter().map(record_key).collect::<Keys>(), stats)
    };
    let (plain, pstats) = run(None);
    let (cached, cstats) = run(Some(64));
    assert_eq!(
        plain, cached,
        "obs-hash fallback hits must not change records"
    );
    assert!(cstats.cache_hits > 0);
    assert!(cstats.infer_rows < pstats.infer_rows);
}

#[test]
fn stale_encoder_key_hits_demote_by_reencoding() {
    let (plain, _) = {
        let (mut e, _, _) = state_engine(Line, true, 4, 2, 1, true, None);
        let (records, stats) = e.collect(9, |obs: Vec<f32>, n: usize| exact_infer(&obs, n));
        (records.iter().map(record_key).collect::<Keys>(), stats)
    };
    for bump_at in 1..=4 {
        let (mut e, _, generation) = state_engine(Line, true, 4, 2, 1, true, Some(64));
        let generation = generation.expect("cache installed");
        let mut calls = 0;
        let (records, _) = e.collect(9, move |obs: Vec<f32>, n: usize| {
            calls += 1;
            if calls == bump_at {
                generation.fetch_add(1, Ordering::Relaxed);
            }
            exact_infer(&obs, n)
        });
        let keys: Keys = records.iter().map(record_key).collect();
        assert_eq!(plain, keys, "bump at call {bump_at} corrupted records");
    }
}

#[test]
fn stage2_collection_is_deterministic() {
    let run = || {
        let (mut e, _, _) = state_engine(TruncLine, true, 3, 2, 3, true, Some(64));
        let (records, stats) = e.collect(20, |obs: Vec<f32>, n: usize| exact_infer(&obs, n));
        (
            records.iter().map(record_key).collect::<Keys>(),
            stats.infer_rows,
            stats.cache_hits,
        )
    };
    assert_eq!(run(), run());
}

#[test]
fn equal_cache_key_streams_encode_identically() {
    let enc = KeyedEnc {
        encodes: Arc::new(AtomicU64::new(0)),
        keyed: true,
    };
    let mut by_key: std::collections::HashMap<u64, Vec<f32>> = std::collections::HashMap::new();
    for tick in 0..32 {
        let state = St { tick };
        let mut hasher = CacheHasher::seeded(0);
        assert!(enc.cache_key(&state, 0, &mut hasher));
        let key = (state.tick % 2) as u64;
        let row = enc.encode(&state, 0);
        match by_key.entry(key) {
            std::collections::hash_map::Entry::Vacant(v) => {
                v.insert(row);
            }
            std::collections::hash_map::Entry::Occupied(o) => {
                assert_eq!(
                    o.get(),
                    &row,
                    "states with equal cache_key streams must encode identically"
                );
            }
        }
    }
}
