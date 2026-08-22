//! Scheduler-overhead probe (ignored by default): a no-op callback makes every
//! record/second delta pure scheduling cost. Run with:
//!   cargo test --release --test overhead_bench -- --ignored --nocapture

use reinfors_core::policies::tree::alphazero::{AlphaZero, AlphaZeroConfig};
use reinfors_core::rollout::engine::{Engine, EngineParams};
use reinfors_core::{
    Actor, AlphaZeroLearner, ChanceMode, Game, NoiseScope, SequentialBackup, Space, StateEncoder,
    Transition,
};

#[derive(Clone)]
struct St {
    cells: [i8; 42],
    to_move: u8,
    filled: u8,
}

struct C4ish;
impl Game for C4ish {
    type State = St;
    type Event = ();
    fn num_agents(&self) -> usize {
        2
    }
    fn action_count(&self) -> usize {
        7
    }
    fn actor(&self, s: &St) -> Actor {
        Actor::Agent(s.to_move as usize)
    }
    fn legal_actions(&self, s: &St, agent: usize) -> Vec<usize> {
        if agent == s.to_move as usize && s.filled < 42 {
            (0..7).collect()
        } else {
            Vec::new()
        }
    }
    fn step(&self, s: &St, a: &[usize]) -> Transition<St, ()> {
        let mut next = s.clone();
        let col = a[s.to_move as usize] % 7;
        next.cells[(s.filled as usize % 6) * 7 + col] = s.to_move as i8 + 1;
        next.filled += 1;
        next.to_move ^= 1;
        Transition {
            next_state: next,
            events: vec![None; 2],
            terminal: s.filled + 1 >= 24,
        }
    }
    fn initial_state(&self) -> St {
        St {
            cells: [0; 42],
            to_move: 0,
            filled: 0,
        }
    }
}

struct Enc;
impl reinfors_core::ActionView for Enc {}
impl StateEncoder for Enc {
    type State = St;
    fn encode(&self, s: &St, _agent: usize) -> Vec<f32> {
        s.cells.iter().map(|&c| c as f32).collect()
    }
    fn obs_shape(&self) -> (usize, usize, usize) {
        (1, 6, 7)
    }
    fn observation_space(&self) -> Space {
        Space::unit_box(vec![1, 6, 7])
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
    extra: impl FnOnce(&mut EngineParams),
) -> Engine<C4ish, AlphaZero, AlphaZeroLearner> {
    let mut params = EngineParams {
        n_games,
        seed: 7,
        ..Default::default()
    };
    extra(&mut params);
    Engine::new(
        C4ish,
        Box::new(Enc),
        Box::new(Zero),
        AlphaZero::new(AlphaZeroConfig {
            num_simulations: 48,
            c_puct: 1.5,
            gamma: 1.0,
            max_depth: 64,
            noise_epsilon: 0.25,
            noise_alpha: 0.3,
            temperature: 1.0,
            temperature_drop: 8,
            chance: ChanceMode::AlwaysResample,
            noise_scope: NoiseScope::Requester,
            sequential_backup: SequentialBackup::Auto,
        }),
        AlphaZeroLearner::new(1.0),
        params,
    )
}

#[test]
#[ignore = "manual overhead probe"]
fn scheduler_overhead_probe() {
    let infer = |_obs: Vec<f32>, n: usize| vec![0.1; n * 8];
    for (label, n_games) in [("n32", 32usize), ("n64", 64usize)] {
        for reps in 0..3 {
            let mut eng = engine(n_games, |_| {});
            eng.collect(512, infer); // warmup
            let t0 = std::time::Instant::now();
            let (a, _) = eng.collect(512, infer);
            let (b, _) = eng.collect(512, infer);
            let dt = t0.elapsed().as_secs_f64();
            println!(
                "{label} rep{reps} scheduler-default {:8.0} rec/s",
                (a.len() + b.len()) as f64 / dt
            );
        }
    }
}
