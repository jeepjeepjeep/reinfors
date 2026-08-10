use reinfors_core::Env;
use reinfors_games::{Connect4, Connect4Event, Connect4Planes, EgocentricSnake, Snake};

#[test]
fn connect4_played_to_a_vertical_win() {
    let mut env = Env::new(Connect4, Box::new(Connect4Planes), 0);
    let moves = [(0, 0), (1, 1), (0, 0), (1, 1), (0, 0), (1, 1), (0, 0)];
    let mut last = Vec::new();
    for (agent, col) in moves {
        assert_eq!(env.active_agents(), vec![agent]);
        let mut joint = vec![0usize; env.num_agents()];
        joint[agent] = col;
        last = env.step(&joint);
    }
    assert!(env.done());
    assert_eq!(
        last,
        vec![(0, Connect4Event::Win), (1, Connect4Event::Loss)],
        "{last:?}"
    );
    assert!(env.active_agents().is_empty());
}

#[test]
fn snake_steps_both_agents_simultaneously() {
    let game = Snake {
        num_snakes: 2,
        grid_size: 8,
        initial_length: 3,
        play_to_last: false,
        win_food_lead: None,
        initial_food_count: 1,
        max_ticks: None,
    };
    let mut env = Env::new(game, Box::new(EgocentricSnake { grid_size: 8 }), 0);
    assert_eq!(env.active_agents(), vec![0, 1]);
    assert_eq!(env.observe(0).len(), 5 * 8 * 8);
    let events = env.step(&[1, 1]);
    assert_eq!(events.len(), 2);
}
