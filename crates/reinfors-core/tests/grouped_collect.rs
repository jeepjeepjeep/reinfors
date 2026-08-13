//! Grouped (double-buffered) collect: determinism and lifecycle at the engine level.

use reinfors_core::policies::tree::alphazero::{AlphaZero, AlphaZeroConfig};
use reinfors_core::{
    Actor, AlphaZeroLearner, Engine, EngineParams, Game, InferMode, Space, StateEncoder, Transition,
};

#[derive(Clone)]
struct St {
    tick: usize,
}
struct Count;
impl Game for Count {
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
    fn legal_actions(&self, _s: &St, _agent: usize) -> Vec<usize> {
        vec![0, 1]
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
impl reinfors_core::Reward for Zero {
    type Event = ();
    fn step_reward(&self, _e: &(), _agent: usize) -> f64 {
        0.0
    }
}

fn engine(
    n_games: usize,
    n_groups: usize,
    seed: u64,
) -> Engine<Count, AlphaZero, AlphaZeroLearner> {
    Engine::new(
        Count,
        Box::new(Enc),
        Box::new(Zero),
        AlphaZero::new(AlphaZeroConfig {
            num_simulations: 6,
            c_puct: 1.5,
            gamma: 1.0,
            max_depth: 64,
            noise_epsilon: 0.25,
            noise_alpha: 0.5,
            temperature: 1.0,
            temperature_drop: 4,
            chance: reinfors_core::ChanceMode::AlwaysResample,
            noise_scope: reinfors_core::policies::tree::mcts::NoiseScope::Requester,
            sequential_backup: reinfors_core::policies::tree::mcts::SequentialBackup::Auto,
        }),
        AlphaZeroLearner::new(1.0),
        EngineParams {
            n_games,
            seed,
            n_groups,
        },
    )
}

fn infer(_p: usize, obs: Vec<f32>, n: usize) -> Vec<f64> {
    assert_eq!(obs.len(), n * 2);
    vec![0.1; n * 3] // 2 policy logits + 1 value per row
}

#[test]
fn grouped_collect_is_deterministic_per_seed() {
    let runs: Vec<_> = (0..2)
        .map(|_| {
            let host = reinfors_core::ServiceHost::spawn(infer);
            engine(4, 2, 11).collect_grouped_hosted(24, InferMode::Shared, &host)
        })
        .collect();
    let (a, sa) = &runs[0];
    let (b, sb) = &runs[1];
    assert!(!a.is_empty());
    assert_eq!(a.len(), b.len());
    for (ra, rb) in a.iter().zip(b) {
        assert_eq!(ra.0, rb.0, "obs");
        assert_eq!(ra.1, rb.1, "pi");
        assert_eq!(ra.2, rb.2, "z");
        assert_eq!(ra.5, rb.5, "legal");
    }
    assert_eq!(sa.infer_rows, sb.infer_rows);
    assert_eq!(sa.decisions, sb.decisions);
}

#[test]
fn hosted_results_are_host_independent() {
    let one = reinfors_core::ServiceHost::spawn(infer);
    let (a, sa) = engine(4, 2, 11).collect_grouped_hosted(24, InferMode::Shared, &one);
    let host = reinfors_core::ServiceHost::spawn(infer);
    let (b, sb) = engine(4, 2, 11).collect_grouped_hosted(24, InferMode::Shared, &host);
    assert!(!a.is_empty());
    assert_eq!(a.len(), b.len());
    for (ra, rb) in a.iter().zip(&b) {
        assert_eq!(ra.0, rb.0, "obs");
        assert_eq!(ra.1, rb.1, "pi");
        assert_eq!(ra.2, rb.2, "z");
        assert_eq!(ra.5, rb.5, "legal");
    }
    assert_eq!(sa.infer_rows, sb.infer_rows);
    assert_eq!(sa.decisions, sb.decisions);
}

#[test]
fn hosted_collects_share_one_service_thread() {
    use std::sync::{Arc, Mutex};
    let threads: Arc<Mutex<std::collections::HashSet<std::thread::ThreadId>>> =
        Arc::new(Mutex::new(std::collections::HashSet::new()));
    let seen = threads.clone();
    let host = reinfors_core::ServiceHost::spawn(move |p, obs, n| {
        seen.lock().unwrap().insert(std::thread::current().id());
        infer(p, obs, n)
    });
    let mut eng = engine(4, 2, 7);
    let (r1, _) = eng.collect_grouped_hosted(16, InferMode::Shared, &host);
    let (r2, _) = eng.collect_grouped_hosted(16, InferMode::Shared, &host);
    assert!(!r1.is_empty() && !r2.is_empty());
    let seen = threads.lock().unwrap();
    assert_eq!(seen.len(), 1, "one thread across collects: {seen:?}");
    assert!(!seen.contains(&std::thread::current().id()));
}

#[test]
fn hosted_collect_callback_panic_reports_and_host_survives() {
    let host = reinfors_core::ServiceHost::spawn(|p, obs, n| {
        if p == usize::MAX {
            unreachable!()
        }
        let _ = (&obs, n);
        panic!("boom");
    });
    let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        engine(4, 2, 5).collect_grouped_hosted(8, InferMode::Shared, &host)
    }));
    assert!(panicked.is_err());
    drop(host); // still joinable after the failed collect
}

#[test]
fn padded_grouped_collect_matches_unpadded() {
    let h1 = reinfors_core::ServiceHost::spawn(infer);
    let (a, sa) = engine(4, 2, 11).collect_grouped_hosted(24, InferMode::Shared, &h1);
    let h2 = reinfors_core::ServiceHost::spawn(infer);
    let (b, sb) = engine(4, 2, 11)
        .with_pad_rows_to(Some(8))
        .collect_grouped_hosted(24, InferMode::Shared, &h2);
    assert!(!a.is_empty());
    assert_eq!(a.len(), b.len());
    for (ra, rb) in a.iter().zip(&b) {
        assert_eq!(ra.0, rb.0, "obs");
        assert_eq!(ra.1, rb.1, "pi");
        assert_eq!(ra.2, rb.2, "z");
    }
    assert_eq!(
        sa.infer_rows, sb.infer_rows,
        "telemetry counts real rows only"
    );
    assert_eq!(sa.padded_rows, 0);
    assert!(sb.padded_rows > 0);
}

#[test]
fn grouped_collect_produces_sane_telemetry() {
    let host = reinfors_core::ServiceHost::spawn(infer);
    let (records, stats) = engine(4, 2, 3).collect_grouped_hosted(16, InferMode::Shared, &host);
    assert!(records.len() >= 16);
    assert!(stats.infer_rows > 0);
    assert!(stats.infer_calls > 0);
    assert!(!stats.episodes.is_empty());
}

#[test]
#[should_panic(expected = "collect via collect_grouped_hosted")]
fn plain_collect_rejects_grouped_engine() {
    let _ = engine(4, 2, 0).collect(8, |obs, n| infer(0, obs, n));
}

#[test]
#[should_panic(expected = "single shared infer callback")]
fn grouped_collect_rejects_per_player_mode() {
    let host = reinfors_core::ServiceHost::spawn(infer);
    let _ = engine(4, 2, 0).collect_grouped_hosted(8, InferMode::PerPlayer, &host);
}

#[test]
#[should_panic(expected = "requires n_groups=2")]
fn grouped_collect_rejects_ungrouped_engine() {
    let host = reinfors_core::ServiceHost::spawn(infer);
    let _ = engine(4, 1, 0).collect_grouped_hosted(8, InferMode::Shared, &host);
}

#[test]
#[should_panic(expected = "n_groups=2 needs at least 2 games")]
fn constructor_rejects_single_game_grouping() {
    let _ = engine(1, 2, 0);
}

struct PanicEnc {
    owner: std::sync::Mutex<Option<std::thread::ThreadId>>,
    calls: std::sync::Arc<std::sync::atomic::AtomicUsize>,
}
impl reinfors_core::ActionView for PanicEnc {}
impl StateEncoder for PanicEnc {
    type State = St;
    fn encode(&self, s: &St, _agent: usize) -> Vec<f32> {
        self.calls
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let me = std::thread::current().id();
        let is_peer = {
            let mut owner = self.owner.lock().unwrap();
            match *owner {
                None => {
                    *owner = Some(me);
                    false
                }
                Some(t) => t != me,
            }
        };
        // panic outside the lock so the owning thread is not poisoned into a second panic
        if is_peer {
            panic!("peer group panicked");
        }
        vec![s.tick as f32, 1.0]
    }
    fn obs_shape(&self) -> (usize, usize, usize) {
        (1, 1, 2)
    }
    fn observation_space(&self) -> Space {
        Space::unit_box(vec![1, 1, 2])
    }
}

#[test]
fn worker_panic_cancels_the_peer_group_promptly() {
    let calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let enc = PanicEnc {
        owner: std::sync::Mutex::new(None),
        calls: calls.clone(),
    };
    let mut eng = Engine::new(
        Count,
        Box::new(enc),
        Box::new(Zero),
        AlphaZero::new(AlphaZeroConfig {
            num_simulations: 6,
            c_puct: 1.5,
            gamma: 1.0,
            max_depth: 64,
            noise_epsilon: 0.25,
            noise_alpha: 0.5,
            temperature: 1.0,
            temperature_drop: 4,
            chance: reinfors_core::ChanceMode::AlwaysResample,
            noise_scope: reinfors_core::policies::tree::mcts::NoiseScope::Requester,
            sequential_backup: reinfors_core::policies::tree::mcts::SequentialBackup::Auto,
        }),
        AlphaZeroLearner::new(1.0),
        EngineParams {
            n_games: 4,
            seed: 7,
            n_groups: 2,
        },
    );
    let host = reinfors_core::ServiceHost::spawn(infer);
    let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
        eng.collect_grouped_hosted(50_000, InferMode::Shared, &host)
    }));
    let payload = match res {
        Err(payload) => payload,
        Ok(_) => panic!("collect should have panicked"),
    };
    let msg = payload.downcast_ref::<&str>().copied().unwrap_or("");
    assert_eq!(msg, "peer group panicked");
    let seen = calls.load(std::sync::atomic::Ordering::Relaxed);
    assert!(
        seen < 5_000,
        "peer group not cancelled promptly: {seen} encodes"
    );
}

#[test]
#[should_panic(expected = "grouped collect infer callback failed")]
fn callback_panic_surfaces_without_hanging() {
    let host =
        reinfors_core::ServiceHost::spawn(|_p: usize, _o: Vec<f32>, _n: usize| -> Vec<f64> {
            panic!("boom")
        });
    let _ = engine(4, 2, 1).collect_grouped_hosted(16, InferMode::Shared, &host);
}
