//! Each concrete game advertises the observation/action `Space` the framework + bindings size networks
//! from. The defaults derive from `obs_shape`/`action_count`; these pin the values per game.

use reinfors_core::{Game, Space};
use reinfors_games::{Connect4, GridWorld, Reward, SnakeGame};

fn unbounded(shape: Vec<usize>) -> Space {
    Space::Box {
        shape,
        low: f32::NEG_INFINITY,
        high: f32::INFINITY,
    }
}

#[test]
fn snake_advertises_egocentric_box_and_three_actions() {
    let game = SnakeGame {
        grid_size: 12,
        initial_length: 3,
        play_to_last: false,
        win_food_lead: None,
        initial_food_count: 3,
        reward: Reward {
            step: 0.0,
            food: 0.0,
            loss: 0.0,
            draw: 0.0,
            kill: 0.0,
            win: 0.0,
            survival: 0.0,
        },
    };
    assert_eq!(game.observation_space(), unbounded(vec![5, 12, 12]));
    assert_eq!(game.action_space(), Space::Discrete { n: 3 });
}

#[test]
fn connect4_advertises_two_plane_box_and_seven_columns() {
    let game = Connect4::default();
    assert_eq!(game.observation_space(), unbounded(vec![2, 6, 7]));
    assert_eq!(game.action_space(), Space::Discrete { n: 7 });
}

#[test]
fn gridworld_advertises_size_scaled_box_and_four_moves() {
    let game = GridWorld {
        size: 5,
        goal: (4, 4),
        step_reward: 0.0,
        goal_reward: 1.0,
    };
    assert_eq!(game.observation_space(), unbounded(vec![2, 5, 5]));
    assert_eq!(game.action_space(), Space::Discrete { n: 4 });
}
