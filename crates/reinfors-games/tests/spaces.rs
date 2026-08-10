use reinfors_core::{Env, Game, Space, StateEncoder};
use reinfors_games::{
    Connect4, Connect4Planes, EgocentricSnake, GridWorld, GridWorldPlanes, Snake,
};

fn unit(shape: Vec<usize>) -> Space {
    Space::unit_box(shape)
}

#[test]
fn snake_advertises_egocentric_box_and_three_actions() {
    let game = Snake {
        num_snakes: 2,
        grid_size: 12,
        initial_length: 3,
        play_to_last: false,
        win_food_lead: None,
        initial_food_count: 3,
        max_ticks: None,
    };
    assert_eq!(
        EgocentricSnake { grid_size: 12 }.observation_space(),
        unit(vec![5, 12, 12])
    );
    assert_eq!(game.action_space(), Space::Discrete { n: 3 });
}

#[test]
fn connect4_advertises_two_plane_box_and_seven_columns() {
    let game = Connect4;
    assert_eq!(Connect4Planes.observation_space(), unit(vec![2, 6, 7]));
    assert_eq!(game.action_space(), Space::Discrete { n: 7 });
}

#[test]
fn gridworld_advertises_size_scaled_box_and_four_moves() {
    let game = GridWorld {
        size: 5,
        goal: (4, 4),
        max_ticks: None,
    };
    assert_eq!(
        GridWorldPlanes {
            size: 5,
            goal: (4, 4)
        }
        .observation_space(),
        unit(vec![2, 5, 5])
    );
    assert_eq!(game.action_space(), Space::Discrete { n: 4 });
}

fn assert_obs_within_advertised_bounds<G: Game>(mut env: Env<G>, steps: usize) {
    let (low, high) = match env.observation_space() {
        Space::Box { low, high, .. } => (low, high),
        s => panic!("expected a Box observation space, got {s:?}"),
    };
    for _ in 0..=steps {
        for agent in 0..env.num_agents() {
            assert!(
                env.observe(agent).iter().all(|&v| low <= v && v <= high),
                "encoded observation fell outside the advertised [{low}, {high}] bounds"
            );
        }
        if env.done() {
            break;
        }
        // Every game exercised here admits action 0 on its opening ticks.
        env.step(&vec![0usize; env.num_agents()]);
    }
}

#[test]
fn encoders_emit_observations_within_the_advertised_bounds() {
    assert_obs_within_advertised_bounds(
        Env::new(
            Snake {
                num_snakes: 2,
                grid_size: 8,
                initial_length: 3,
                play_to_last: false,
                win_food_lead: None,
                initial_food_count: 2,
                max_ticks: None,
            },
            Box::new(EgocentricSnake { grid_size: 8 }),
            0,
        ),
        4,
    );
    assert_obs_within_advertised_bounds(Env::new(Connect4, Box::new(Connect4Planes), 0), 4);
    assert_obs_within_advertised_bounds(
        Env::new(
            GridWorld {
                size: 5,
                goal: (4, 4),
                max_ticks: None,
            },
            Box::new(GridWorldPlanes {
                size: 5,
                goal: (4, 4),
            }),
            0,
        ),
        4,
    );
}
