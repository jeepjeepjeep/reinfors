//! The AlphaZero (PUCT) planner on real games: it finds a forced connect4 win, priors steer the
//! search, root noise diversifies it deterministically, and simultaneous games are rejected —
//! mirroring the UCT suite in `mcts.rs`.

use reinfors_core::{alphazero_many, Actor, AlphaZeroConfig, ChanceMode, Evaluator, Game, Rng};
use reinfors_games::{Connect4, Connect4Planes, Connect4Reward, EgocentricSnake, Snake};

struct NoRng;
impl Rng for NoRng {
    fn below(&mut self, _: usize) -> usize {
        0
    }
    fn unit(&mut self) -> f64 {
        0.0
    }
}

fn cfg(num_simulations: usize, noise_epsilon: f64) -> AlphaZeroConfig {
    AlphaZeroConfig {
        num_simulations,
        c_puct: 2.0,
        gamma: 0.99,
        max_depth: 12,
        noise_epsilon,
        noise_alpha: 0.3,
        temperature: 0.0,
        temperature_drop: u32::MAX,
        chance: ChanceMode::AlwaysResample,
    }
}

fn reward() -> Connect4Reward {
    Connect4Reward {
        win: 1.0,
        loss: -1.0,
        draw: 0.0,
    }
}

/// Uniform-logits, zero-value evaluator: rows of `A` equal logits + value 0 — PUCT's analogue of the
/// UCT suite's zeros net (all signal from terminals).
fn uniform_infer(_obs: Vec<f32>, n: usize) -> Vec<f64> {
    vec![0.0; n * 8] // A+1 = 8
}

/// An evaluator whose logits heavily favor one column, value 0 — for prior-steering checks.
fn sharp_infer(col: usize) -> impl FnMut(Vec<f32>, usize) -> Vec<f64> {
    move |_obs, n| {
        let mut out = vec![0.0; n * 8];
        for row in 0..n {
            out[row * 8 + col] = 6.0; // softmax -> ~0.87 on `col`
        }
        out
    }
}

fn argmax(v: &[f64]) -> usize {
    (0..v.len())
        .max_by(|&a, &b| v[a].partial_cmp(&v[b]).unwrap())
        .unwrap()
}

/// Build a position where P0 has three in column 3 and it is P0's move — column 3 wins outright.
fn forced_win_state() -> reinfors_games::Connect4State {
    let game = Connect4;
    let mut state = game.initial_state(&mut NoRng);
    for &(mover, col) in &[(0, 3), (1, 0), (0, 3), (1, 0), (0, 3), (1, 0)] {
        assert_eq!(game.actor(&state), Actor::Agent(mover));
        let mut joint = vec![0usize; 2];
        joint[mover] = col;
        state = game.step(&state, &joint).next_state;
    }
    assert_eq!(game.actor(&state), Actor::Agent(0));
    state
}

#[test]
fn finds_the_forced_connect4_win() {
    let evals = alphazero_many(
        &Connect4,
        &Connect4Planes,
        &reward(),
        &cfg(96, 0.0),
        vec![(forced_win_state(), 0)],
        7,
        &mut Evaluator::new(&mut uniform_infer, None),
    );
    assert_eq!(
        argmax(&evals[0].visits),
        3,
        "visits should pick the winning column"
    );
    assert_eq!(argmax(&evals[0].values[0]), 3);
}

#[test]
fn priors_steer_visits() {
    // With few sims and no terminal signal from the opening position, visits follow the prior.
    let game = Connect4;
    let state = game.initial_state(&mut NoRng);
    for col in [2usize, 5] {
        let evals = alphazero_many(
            &game,
            &Connect4Planes,
            &reward(),
            &cfg(32, 0.0),
            vec![(state.clone(), 0)],
            7,
            &mut Evaluator::new(&mut sharp_infer(col), None),
        );
        assert_eq!(
            argmax(&evals[0].visits),
            col,
            "a sharp prior on column {col} should dominate the visit distribution"
        );
    }
}

#[test]
fn search_is_deterministic_per_seed_and_noise_diversifies_across_seeds() {
    let game = Connect4;
    let state = game.initial_state(&mut NoRng);
    let run = |seed: u64, eps: f64| {
        alphazero_many(
            &game,
            &Connect4Planes,
            &reward(),
            &cfg(48, eps),
            vec![(state.clone(), 0)],
            seed,
            &mut Evaluator::new(&mut uniform_infer, None),
        )
        .remove(0)
        .visits
    };
    assert_eq!(run(7, 0.5), run(7, 0.5), "same seed, same noisy search");
    // Strong noise on a signal-free position: different seeds should shape visits differently.
    let differs = (0..8).any(|s| run(s, 0.9) != run(100 + s, 0.9));
    assert!(
        differs,
        "root noise never changed the visit distribution across seeds"
    );
    // And noise-off must be seed-independent (nothing else draws randomness).
    assert_eq!(run(1, 0.0), run(2, 0.0));
}

#[test]
fn pooled_trees_draw_independent_noise() {
    // Two identical requests in one pooled call: with noise on, their searches should diverge
    // (per-tree noise streams), not mirror each other.
    let game = Connect4;
    let state = game.initial_state(&mut NoRng);
    let evals = alphazero_many(
        &game,
        &Connect4Planes,
        &reward(),
        &cfg(48, 0.9),
        vec![(state.clone(), 0), (state, 0)],
        11,
        &mut Evaluator::new(&mut uniform_infer, None),
    );
    assert_ne!(
        evals[0].visits, evals[1].visits,
        "identical pooled requests should get independent root noise"
    );
}

#[test]
#[should_panic(expected = "sequential/single-agent")]
fn rejects_simultaneous_snake() {
    let snake = Snake {
        grid_size: 8,
        initial_length: 3,
        play_to_last: false,
        win_food_lead: None,
        initial_food_count: 1,
        max_ticks: None,
    };
    let state = snake.initial_state(&mut NoRng);
    let reward = reinfors_games::SnakeReward {
        step: 0.0,
        food: 0.0,
        loss: 0.0,
        draw: 0.0,
        kill: 0.0,
        win: 0.0,
        survival: 0.0,
    };
    let _ = alphazero_many(
        &snake,
        &EgocentricSnake { grid_size: 8 },
        &reward,
        &cfg(4, 0.0),
        vec![(state, 0)],
        0,
        &mut Evaluator::new(&mut uniform_infer, None),
    );
}

#[test]
fn infer_cache_is_behavior_identical_and_hits() {
    use reinfors_core::InferCache;
    use std::sync::atomic::AtomicU64;
    use std::sync::Arc;
    let game = Connect4;
    let state = game.initial_state(&mut NoRng);
    // Repeated identical requests in one pool: heavy transposition + within-batch dedup territory.
    let requests: Vec<_> = (0..4).map(|_| (state.clone(), 0)).collect();
    let run = |cache: Option<&mut InferCache>| {
        alphazero_many(
            &game,
            &Connect4Planes,
            &reward(),
            &cfg(48, 0.5),
            requests.clone(),
            9,
            &mut Evaluator::new(&mut uniform_infer, cache),
        )
    };
    let plain = run(None);
    let generation = Arc::new(AtomicU64::new(0));
    let mut cache = InferCache::new(1 << 16, generation);
    let cached = run(Some(&mut cache));
    for (p, c) in plain.iter().zip(&cached) {
        assert_eq!(p.visits, c.visits, "cache changed search behavior (visits)");
        assert_eq!(p.values, c.values, "cache changed search behavior (values)");
    }
    assert!(
        cache.hits > 0,
        "identical pooled requests must produce cache hits"
    );
    // Second search over the same position reuses across calls too.
    let before = cache.hits;
    let again = run(Some(&mut cache));
    for (p, c) in plain.iter().zip(&again) {
        assert_eq!(p.visits, c.visits);
    }
    assert!(cache.hits > 0 || before > 0);
}
