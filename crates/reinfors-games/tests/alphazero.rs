use reinfors_core::{
    alphazero_many, Actor, AlphaZeroConfig, ChanceMode, Evaluator, Game, NoiseScope, Rng,
};
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
        noise_scope: NoiseScope::Requester,
        sequential_backup: Default::default(),
    }
}

fn reward() -> Connect4Reward {
    Connect4Reward {
        win: 1.0,
        loss: -1.0,
        draw: 0.0,
    }
}

fn uniform_infer(_p: usize, _obs: Vec<f32>, n: usize) -> Vec<f64> {
    vec![0.0; n * 8]
}

fn sharp_infer(col: usize) -> impl FnMut(usize, Vec<f32>, usize) -> Vec<f64> {
    move |_p, _obs, n| {
        let mut out = vec![0.0; n * 8];
        for row in 0..n {
            // Against six zero logits, 6 assigns roughly 98.5% prior mass to this column.
            out[row * 8 + col] = 6.0;
        }
        out
    }
}

fn argmax(v: &[f64]) -> usize {
    (0..v.len())
        .max_by(|&a, &b| v[a].partial_cmp(&v[b]).unwrap())
        .unwrap()
}

fn forced_win_state() -> reinfors_games::Connect4State {
    let game = Connect4;
    let mut state = game.initial_state();
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
        &mut Evaluator::new(&mut uniform_infer, reinfors_core::InferMode::Shared, None),
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
    // With few simulations and no terminal signal from the opening, visits follow the prior.
    let game = Connect4;
    let state = game.initial_state();
    for col in [2usize, 5] {
        let evals = alphazero_many(
            &game,
            &Connect4Planes,
            &reward(),
            &cfg(32, 0.0),
            vec![(state.clone(), 0)],
            7,
            &mut Evaluator::new(
                &mut sharp_infer(col),
                reinfors_core::InferMode::Shared,
                None,
            ),
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
    let state = game.initial_state();
    let run = |seed: u64, eps: f64| {
        alphazero_many(
            &game,
            &Connect4Planes,
            &reward(),
            &cfg(48, eps),
            vec![(state.clone(), 0)],
            seed,
            &mut Evaluator::new(&mut uniform_infer, reinfors_core::InferMode::Shared, None),
        )
        .remove(0)
        .visits
    };
    assert_eq!(run(7, 0.5), run(7, 0.5), "same seed, same noisy search");
    let differs = (0..8).any(|s| run(s, 0.9) != run(100 + s, 0.9));
    assert!(
        differs,
        "root noise never changed the visit distribution across seeds"
    );
    assert_eq!(
        run(1, 0.0),
        run(2, 0.0),
        "with noise disabled, no search randomness remains"
    );
}

#[test]
fn pooled_trees_draw_independent_noise() {
    let game = Connect4;
    let state = game.initial_state();
    let evals = alphazero_many(
        &game,
        &Connect4Planes,
        &reward(),
        &cfg(48, 0.9),
        vec![(state.clone(), 0), (state, 0)],
        11,
        &mut Evaluator::new(&mut uniform_infer, reinfors_core::InferMode::Shared, None),
    );
    assert_ne!(
        evals[0].visits, evals[1].visits,
        "identical pooled requests should get independent root noise"
    );
}

#[test]
fn searches_simultaneous_stochastic_snake() {
    let snake = Snake {
        num_snakes: 2,
        grid_size: 8,
        initial_length: 3,
        play_to_last: false,
        win_food_lead: None,
        initial_food_count: 1,
        max_ticks: None,
    };
    let state = reinfors_core::realize_initial_state(&snake, &mut NoRng);
    let reward = reinfors_games::SnakeReward {
        step: 0.0,
        food: 1.0,
        loss: -1.0,
        draw: 0.0,
        kill: 0.0,
        win: 1.0,
        survival: 0.0,
    };
    let run = |seed: u64| {
        let mut infer = |_p: usize, _obs: Vec<f32>, n: usize| vec![0.0; n * 4];
        alphazero_many(
            &snake,
            &EgocentricSnake { grid_size: 8 },
            &reward,
            &AlphaZeroConfig {
                num_simulations: 32,
                c_puct: 2.0,
                gamma: 0.99,
                max_depth: 10,
                noise_epsilon: 0.25,
                noise_alpha: 0.3,
                temperature: 0.0,
                temperature_drop: u32::MAX,
                chance: ChanceMode::Committed { samples: 2 },
                noise_scope: NoiseScope::Requester,
                sequential_backup: Default::default(),
            },
            vec![(state.clone(), 0), (state.clone(), 1)],
            seed,
            &mut Evaluator::new(&mut infer, reinfors_core::InferMode::Shared, None),
        )
    };
    let evals = run(7);
    assert_eq!(evals.len(), 2);
    for e in &evals {
        assert_eq!(e.visits.len(), 3);
        assert!(e.visits.iter().sum::<f64>() > 0.0);
        assert!(e.values[0].iter().all(|v| v.is_finite()));
    }
    let again = run(7);
    for (a, b) in evals.iter().zip(&again) {
        assert_eq!(
            a.visits, b.visits,
            "seeded snake search must be deterministic"
        );
    }
}

#[test]
fn infer_cache_is_behavior_identical_and_hits() {
    use reinfors_core::InferCache;
    use std::sync::atomic::AtomicU64;
    use std::sync::Arc;
    let game = Connect4;
    let state = game.initial_state();
    let requests: Vec<_> = (0..4).map(|_| (state.clone(), 0)).collect();
    let run = |cache: Option<&mut [InferCache]>| {
        alphazero_many(
            &game,
            &Connect4Planes,
            &reward(),
            &cfg(48, 0.5),
            requests.clone(),
            9,
            &mut Evaluator::new(&mut uniform_infer, reinfors_core::InferMode::Shared, cache),
        )
    };
    let plain = run(None);
    let generation = Arc::new(AtomicU64::new(0));
    let mut cache = InferCache::new(1 << 16, generation);
    let cached = run(Some(std::slice::from_mut(&mut cache)));
    for (p, c) in plain.iter().zip(&cached) {
        assert_eq!(p.visits, c.visits, "cache changed search behavior (visits)");
        assert_eq!(p.values, c.values, "cache changed search behavior (values)");
    }
    assert!(
        cache.hits > 0,
        "identical pooled requests must produce cache hits"
    );
    let before = cache.hits;
    let again = run(Some(std::slice::from_mut(&mut cache)));
    for (p, c) in plain.iter().zip(&again) {
        assert_eq!(p.visits, c.visits);
    }
    assert!(cache.hits > 0 || before > 0);
}

#[test]
fn searches_backgammon_dice_chance() {
    use reinfors_games::{Backgammon, BackgammonReward, BackgammonTesauro};
    let g = Backgammon::default();
    let state = reinfors_core::realize_initial_state(&g, &mut NoRng);
    let reward = BackgammonReward::default();
    let run = |seed: u64| {
        let mut infer = |_p: usize, _obs: Vec<f32>, n: usize| vec![0.0; n * 1353];
        alphazero_many(
            &g,
            &BackgammonTesauro,
            &reward,
            &AlphaZeroConfig {
                num_simulations: 24,
                c_puct: 2.0,
                gamma: 1.0,
                max_depth: 12,
                noise_epsilon: 0.25,
                noise_alpha: 0.3,
                temperature: 0.0,
                temperature_drop: u32::MAX,
                chance: ChanceMode::AlwaysResample,
                noise_scope: NoiseScope::Requester,
                sequential_backup: Default::default(),
            },
            vec![(state.clone(), 0)],
            seed,
            &mut Evaluator::new(&mut infer, reinfors_core::InferMode::Shared, None),
        )
        .remove(0)
    };
    let e = run(13);
    assert_eq!(e.visits.len(), 1352);
    assert!(e.visits.iter().sum::<f64>() > 0.0);
    let legal: Vec<usize> = g.legal_actions(&state, 0);
    for (a, &v) in e.visits.iter().enumerate() {
        if v > 0.0 {
            assert!(legal.contains(&a), "visited illegal action {a}");
        }
    }
    let again = run(13);
    assert_eq!(
        e.visits, again.visits,
        "seeded dice-chance search is deterministic"
    );
}
