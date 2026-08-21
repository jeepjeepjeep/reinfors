//! Threshold-scheduler semantics: firing shape, round/batch decoupling, cursor state,
//! per-player routing, start distributions under fan, and CPU/inference overlap.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use reinfors_core::rollout::engine::{Engine, EngineParams};
use reinfors_core::rollout::start::{Start, StartDistribution};
use reinfors_core::{
    Actor, Game, Policy, PpoActor, Reward, Rng, Space, StateCodec, StateEncoder, Transition,
};

#[derive(Clone)]
struct St {
    tick: usize,
}

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

struct Enc;
impl reinfors_core::ActionView for Enc {}
impl StateEncoder for Enc {
    type State = St;
    fn encode(&self, s: &St, _agent: usize) -> Vec<f32> {
        vec![s.tick as f32, 1.0]
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

struct Codec;
impl StateCodec for Codec {
    type State = St;
    fn encode(&self, s: &St) -> Vec<u8> {
        (s.tick as u64).to_le_bytes().to_vec()
    }
    fn decode(&self, b: &[u8]) -> Result<St, String> {
        let arr: [u8; 8] = b.try_into().map_err(|_| "bad state".to_string())?;
        Ok(St {
            tick: u64::from_le_bytes(arr) as usize,
        })
    }
    fn validate_decoded_state(&self, s: &St, _done: bool) -> Result<(), String> {
        if s.tick <= 3 {
            Ok(())
        } else {
            Err("tick out of range".into())
        }
    }
}

fn ppo_engine(
    n_games: usize,
    batch_size: usize,
    n_threads: usize,
) -> Engine<Line, PpoActor, reinfors_core::Ppo> {
    Engine::new(
        Line,
        Box::new(Enc),
        Box::new(Zero),
        PpoActor::new(),
        reinfors_core::Ppo::new(1.0, 0.95),
        EngineParams {
            n_games,
            seed: 7,
            batch_size: Some(batch_size),
            n_threads: Some(n_threads),
            ..Default::default()
        },
    )
}

#[test]
fn callbacks_fire_at_the_threshold_with_short_drains() {
    let sizes = Arc::new(std::sync::Mutex::new(Vec::new()));
    let s = sizes.clone();
    let mut engine = ppo_engine(4, 2, 1);
    let (records, _) = engine.collect(8, move |_obs: Vec<f32>, n: usize| {
        s.lock().unwrap().push(n);
        vec![0.0; n * 3]
    });
    assert!(records.len() >= 8);
    let sizes = sizes.lock().unwrap();
    // The final call is the fragment cut's gathered tail flush (batched by design,
    // pool-sized); every search-round call obeys the threshold.
    let (tail_flush, rounds) = sizes.split_last().unwrap();
    assert!(
        rounds.iter().all(|&n| n <= 2),
        "no round call may exceed batch_size: {sizes:?}"
    );
    assert!(
        rounds.iter().filter(|&&n| n == 2).count() >= 2,
        "threshold batches must dominate: {sizes:?}"
    );
    assert_eq!(*tail_flush, 4, "the fragment cut gathers every game's tail");
}

#[test]
fn oversized_pools_drain_short_when_starved() {
    // batch_size above everything the pool can queue: every call is a drain.
    let sizes = Arc::new(std::sync::Mutex::new(Vec::new()));
    let s = sizes.clone();
    let mut engine = ppo_engine(3, 64, 1);
    let (records, _) = engine.collect(6, move |_obs: Vec<f32>, n: usize| {
        s.lock().unwrap().push(n);
        vec![0.0; n * 3]
    });
    assert!(records.len() >= 6);
    let sizes = sizes.lock().unwrap();
    assert!(
        sizes.iter().all(|&n| n <= 3),
        "drains carry at most the pool's queued rows: {sizes:?}"
    );
}

#[test]
fn the_cursor_advances_per_collect_and_survives_restores() {
    // Snapshot layout: version(1) + counts(8) + rngs(16), then the sweep cursor u64.
    fn cursor(bytes: &[u8]) -> u64 {
        u64::from_le_bytes(bytes[25..33].try_into().unwrap())
    }
    let mut engine = ppo_engine(3, 2, 1);
    let infer = |_obs: Vec<f32>, n: usize| vec![0.0; n * 3];
    assert_eq!(cursor(&engine.snapshot_bytes(&Codec).unwrap()), 0);
    engine.collect(2, infer);
    engine.collect(2, infer);
    let snap = engine.snapshot_bytes(&Codec).unwrap();
    assert_eq!(cursor(&snap), 2, "one rotation step per admitting collect");
    engine.collect(0, infer);
    assert_eq!(
        cursor(&engine.snapshot_bytes(&Codec).unwrap()),
        2,
        "a no-op collect must not rotate"
    );
    let mut fresh = ppo_engine(3, 2, 1);
    fresh.restore_bytes(&Codec, &snap).unwrap();
    assert_eq!(cursor(&fresh.snapshot_bytes(&Codec).unwrap()), 2);
}

#[test]
fn start_distribution_serves_respawns_under_fan() {
    struct Counting(Arc<AtomicUsize>);
    impl StartDistribution<St> for Counting {
        fn observe(&mut self, _state: &St, _rng: &mut dyn Rng) {}
        fn choose(&mut self, _rng: &mut dyn Rng) -> Start<St> {
            self.0.fetch_add(1, Ordering::Relaxed);
            Start::Fresh
        }
    }
    let chosen = Arc::new(AtomicUsize::new(0));
    let mut engine =
        ppo_engine(4, 2, 4).with_start_distribution(Box::new(Counting(chosen.clone())));
    let (records, _) = engine.collect(24, |_obs: Vec<f32>, n: usize| vec![0.0; n * 3]);
    assert!(records.len() >= 24);
    assert!(
        chosen.load(Ordering::Relaxed) > 0,
        "episode ends must consult the start distribution"
    );
}

#[test]
fn rounds_execute_while_the_callback_runs() {
    static ROUNDS: AtomicUsize = AtomicUsize::new(0);

    struct SlowRound;
    impl Policy for SlowRound {
        type Evaluation = ();
        type PolicyState = ();
        type Search<S: Send> = bool;
        fn begin_search<G: Game + Sync>(
            &self,
            _ctx: reinfors_core::policy::SearchCtx<'_, G>,
            _state: &G::State,
            _perspectives: &[usize],
        ) -> bool
        where
            G::State: Send,
        {
            false
        }
        fn round<G: Game + Sync>(
            &self,
            _ctx: reinfors_core::policy::SearchCtx<'_, G>,
            emitted: &mut bool,
            out: &mut reinfors_core::policy::RequestSink,
        ) -> reinfors_core::policy::RoundStatus
        where
            G::State: Send,
        {
            if *emitted {
                return reinfors_core::policy::RoundStatus::Done;
            }
            std::thread::sleep(std::time::Duration::from_millis(2));
            ROUNDS.fetch_add(1, Ordering::SeqCst);
            *emitted = true;
            out.push(0, &[0.0, 0.0]);
            reinfors_core::policy::RoundStatus::Pending
        }
        fn absorb<G: Game + Sync>(
            &self,
            _ctx: reinfors_core::policy::SearchCtx<'_, G>,
            _search: &mut bool,
            _rows: reinfors_core::policy::RowsView<'_>,
        ) where
            G::State: Send,
        {
        }
        fn finish<G: Game + Sync>(
            &self,
            _ctx: reinfors_core::policy::SearchCtx<'_, G>,
            _search: bool,
        ) -> Vec<((), Vec<reinfors_core::learner::InteriorTarget>)>
        where
            G::State: Send,
        {
            vec![((), Vec::new())]
        }
        fn max_agents(&self, _sequential: bool) -> Option<usize> {
            None
        }
        fn supports_imperfect_information(&self) -> bool {
            true
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
        fn select(&self, _eval: &(), _state: &mut (), _rng: &mut dyn Rng) -> usize {
            0
        }
        fn fold_telemetry(&self, _eval: &(), _stats: &mut reinfors_core::CollectStats) {}
    }

    struct OneRecord;
    impl reinfors_core::Learner<()> for OneRecord {
        type Record = ();
        fn eval_records(
            &self,
            _evaluation: &(),
            _targets: Vec<reinfors_core::learner::InteriorTarget>,
            _view: &dyn reinfors_core::ActionView,
            _agent: usize,
            _rng: &mut dyn Rng,
        ) -> Vec<()> {
            vec![()]
        }
        fn episode_records(
            &self,
            _trajectory: &[reinfors_core::Step<()>],
            _tail: &[f64],
            _view: &dyn reinfors_core::ActionView,
            _agent: usize,
            _rng: &mut dyn Rng,
        ) -> Vec<()> {
            Vec::new()
        }
    }

    let mut engine = Engine::new(
        Line,
        Box::new(Enc),
        Box::new(Zero),
        SlowRound,
        OneRecord,
        EngineParams {
            n_games: 4,
            seed: 3,
            batch_size: Some(1),
            n_threads: Some(2),
            ..Default::default()
        },
    );
    let mut overlapped = false;
    let (records, _) = engine.collect(16, |_obs: Vec<f32>, n: usize| {
        let before = ROUNDS.load(Ordering::SeqCst);
        std::thread::sleep(std::time::Duration::from_millis(10));
        let after = ROUNDS.load(Ordering::SeqCst);
        if after > before {
            overlapped = true;
        }
        vec![0.0; n * 2]
    });
    assert!(records.len() >= 16);
    assert!(
        overlapped,
        "search rounds must keep executing while the callback runs"
    );
}
