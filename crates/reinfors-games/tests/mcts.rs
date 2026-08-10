use reinfors_core::{mcts_many, Actor, ChanceMode, Evaluator, Game, MctsConfig, Rng};
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

fn cfg(num_simulations: usize) -> MctsConfig {
    MctsConfig {
        num_simulations,
        uct_c: 2.0,
        gamma: 0.99,
        max_depth: 12,
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

fn zeros_infer(_p: usize, _obs: Vec<f32>, n: usize) -> Vec<f64> {
    vec![0.0; n * 7]
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
fn mcts_finds_a_forced_connect4_win() {
    let evals = mcts_many(
        &Connect4,
        &Connect4Planes,
        &reward(),
        &cfg(128),
        vec![(forced_win_state(), 0)],
        0,
        &mut Evaluator::new(&mut zeros_infer, reinfors_core::InferMode::Shared, None),
    );
    let values = &evals[0].values[0];
    assert_eq!(
        argmax(values),
        3,
        "MCTS should play the immediately-winning column"
    );
    assert!(
        values[3] > 0.9,
        "the winning move's value should approach the win reward"
    );
    assert_eq!(evals[0].visits.len(), 7);
}

fn opponent_threat_state() -> reinfors_games::Connect4State {
    let game = Connect4;
    let mut state = game.initial_state();
    for &(mover, col) in &[(0, 1), (1, 0), (0, 1), (1, 0), (0, 2), (1, 0)] {
        assert_eq!(game.actor(&state), Actor::Agent(mover));
        let mut joint = vec![0usize; 2];
        joint[mover] = col;
        state = game.step(&state, &joint).next_state;
    }
    assert_eq!(game.actor(&state), Actor::Agent(0));
    state
}

#[test]
fn mcts_blocks_opponent_win() {
    let evals = mcts_many(
        &Connect4,
        &Connect4Planes,
        &reward(),
        &cfg(400),
        vec![(opponent_threat_state(), 0)],
        0,
        &mut Evaluator::new(&mut zeros_infer, reinfors_core::InferMode::Shared, None),
    );
    let values = &evals[0].values[0];
    assert_eq!(
        argmax(values),
        0,
        "MCTS should block the opponent's winning column"
    );
    assert!(
        values[0] > values[1],
        "blocking must beat a move that hands P1 the win"
    );
}

#[test]
fn mcts_is_deterministic() {
    let run = || {
        mcts_many(
            &Connect4,
            &Connect4Planes,
            &reward(),
            &cfg(64),
            vec![(forced_win_state(), 0)],
            0,
            &mut Evaluator::new(&mut zeros_infer, reinfors_core::InferMode::Shared, None),
        )[0]
        .values[0]
            .clone()
    };
    assert_eq!(run(), run());
}

#[test]
fn mcts_pools_multiple_requests() {
    let evals = mcts_many(
        &Connect4,
        &Connect4Planes,
        &reward(),
        &cfg(128),
        vec![(forced_win_state(), 0), (forced_win_state(), 0)],
        0,
        &mut Evaluator::new(&mut zeros_infer, reinfors_core::InferMode::Shared, None),
    );
    assert_eq!(evals.len(), 2);
    assert_eq!(argmax(&evals[0].values[0]), 3);
    assert_eq!(argmax(&evals[1].values[0]), 3);
}

#[test]
fn mcts_searches_simultaneous_snake() {
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
        food: 0.0,
        loss: 0.0,
        draw: 0.0,
        kill: 0.0,
        win: 0.0,
        survival: 0.0,
    };
    let mut infer = |_p: usize, _obs: Vec<f32>, n: usize| vec![0.0; n * 3];
    let evals = mcts_many(
        &snake,
        &EgocentricSnake { grid_size: 8 },
        &reward,
        &cfg(16),
        vec![(state, 0)],
        0,
        &mut Evaluator::new(&mut infer, reinfors_core::InferMode::Shared, None),
    );
    assert_eq!(evals[0].visits.len(), 3);
    assert!(evals[0].visits.iter().sum::<f64>() > 0.0);
}
