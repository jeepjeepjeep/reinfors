//! The MCTS (UCT) planner on real games: it finds a forced connect4 win, is deterministic, and rejects
//! simultaneous games. Mirrors how the binding pairs `Mcts` with a sequential game.

use reinfors_core::{mcts_many, Actor, Game, MctsConfig, Rng};
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
    }
}

fn reward() -> Connect4Reward {
    Connect4Reward {
        win: 1.0,
        loss: -1.0,
        draw: 0.0,
    }
}

// A trivial zeros evaluator (K=1): leaf values are 0, so all signal comes from terminal win/loss —
// enough for MCTS to solve tactics, exactly like a random-rollout MCTS.
fn zeros_infer(_obs: Vec<f32>, n: usize) -> Vec<f64> {
    vec![0.0; n * 7] // K=1, A=7
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
fn mcts_finds_a_forced_connect4_win() {
    let evals = mcts_many(
        &Connect4,
        &Connect4Planes,
        &reward(),
        &cfg(128),
        vec![(forced_win_state(), 0)],
        None,
        &mut zeros_infer,
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
    assert_eq!(evals[0].visits.len(), 7); // per-action visit counts, full action space
}

/// A position where P1 has three in column 0 and it is P0's move — if P0 doesn't play column 0, P1 wins
/// there next turn. P0's only safe move is to block by playing column 0.
fn opponent_threat_state() -> reinfors_games::Connect4State {
    let game = Connect4;
    let mut state = game.initial_state(&mut NoRng);
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
    // Exercises the negamax turn-change: the opponent's win one ply deeper must propagate to the root
    // as a loss, so P0 prefers the (neutral) block over any move that lets P1 win. A sign error in the
    // backup would make P0 choose a losing move instead.
    let evals = mcts_many(
        &Connect4,
        &Connect4Planes,
        &reward(),
        &cfg(400),
        vec![(opponent_threat_state(), 0)],
        None,
        &mut zeros_infer,
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
            None,
            &mut zeros_infer,
        )[0]
        .values[0]
            .clone()
    };
    assert_eq!(run(), run());
}

#[test]
fn mcts_pools_multiple_requests() {
    // Two requests searched together (pooled, batched infer) each resolve their own tree correctly.
    let evals = mcts_many(
        &Connect4,
        &Connect4Planes,
        &reward(),
        &cfg(128),
        vec![(forced_win_state(), 0), (forced_win_state(), 0)],
        None,
        &mut zeros_infer,
    );
    assert_eq!(evals.len(), 2);
    assert_eq!(argmax(&evals[0].values[0]), 3);
    assert_eq!(argmax(&evals[1].values[0]), 3);
}

#[test]
#[should_panic(expected = "sequential/single-agent")]
fn mcts_rejects_simultaneous_games() {
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
    let mut infer = |_obs: Vec<f32>, n: usize| vec![0.0; n * 3];
    let _ = mcts_many(
        &snake,
        &EgocentricSnake { grid_size: 8 },
        &reward,
        &cfg(8),
        vec![(state, 0)],
        None,
        &mut infer,
    );
}
