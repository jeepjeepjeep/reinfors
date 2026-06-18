//! `Env` over the concrete games: a sequential game (Connect-4) played to a known win, and a
//! simultaneous game (snake) stepped a tick. Validates the caller-driven single-game loop end to end.

use reinfors_core::Env;
use reinfors_games::{Connect4, Connect4Planes, EgocentricSnake, Snake, SnakeReward};

#[test]
fn connect4_played_to_a_vertical_win() {
    let mut env = Env::new(Connect4::default(), Box::new(Connect4Planes), 0);
    // P0 stacks column 0, P1 column 1; P0 completes four-in-a-column first. Moves alternate, matching
    // the game's turn order, which `active_agents` reports.
    let moves = [(0, 0), (1, 1), (0, 0), (1, 1), (0, 0), (1, 1), (0, 0)];
    let mut last = Vec::new();
    for (agent, col) in moves {
        assert_eq!(env.active_agents(), vec![agent]); // sequential: one mover per tick
        let mut joint = vec![0usize; env.num_agents()];
        joint[agent] = col;
        last = env.step(&joint);
    }
    assert!(env.done());
    assert!(last[0] > 0.0 && last[1] < 0.0, "P0 should win: {last:?}"); // zero-sum win/loss
    assert!(env.active_agents().is_empty());
}

#[test]
fn snake_steps_both_agents_simultaneously() {
    let game = Snake {
        grid_size: 8,
        initial_length: 3,
        play_to_last: false,
        win_food_lead: None,
        initial_food_count: 1,
        reward: SnakeReward {
            step: 0.0,
            food: 1.0,
            loss: -10.0,
            draw: -5.0,
            kill: 5.0,
            win: 10.0,
            survival: 0.0,
        },
    };
    let mut env = Env::new(game, Box::new(EgocentricSnake { grid_size: 8 }), 0);
    assert_eq!(env.active_agents(), vec![0, 1]); // simultaneous: both live agents act
    assert_eq!(env.observe(0).len(), 5 * 8 * 8);
    let rewards = env.step(&[1, 1]); // both move forward
    assert_eq!(rewards.len(), 2);
}
