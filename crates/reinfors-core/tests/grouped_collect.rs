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
        .map(|_| engine(4, 2, 11).collect_grouped(24, InferMode::Shared, infer))
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
fn grouped_collect_produces_sane_telemetry() {
    let (records, stats) = engine(4, 2, 3).collect_grouped(16, InferMode::Shared, infer);
    assert!(records.len() >= 16);
    assert!(stats.infer_rows > 0);
    assert!(stats.infer_calls > 0);
    assert!(!stats.episodes.is_empty());
}

#[test]
#[should_panic(expected = "requires n_groups=2")]
fn grouped_collect_rejects_ungrouped_engine() {
    let _ = engine(4, 1, 0).collect_grouped(8, InferMode::Shared, infer);
}

#[test]
#[should_panic(expected = "n_groups=2 needs at least 2 games")]
fn constructor_rejects_single_game_grouping() {
    let _ = engine(1, 2, 0);
}
