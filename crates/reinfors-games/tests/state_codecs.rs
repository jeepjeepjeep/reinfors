use reinfors_core::{Game, Rng, StateCodec};
use reinfors_games::{
    Backgammon, Chess, Connect4, GridState, GridWorld, KuhnPoker, LeducPoker, Snake, TexasHoldem,
};

struct Lcg(u64);
impl Rng for Lcg {
    fn below(&mut self, n: usize) -> usize {
        self.0 = self
            .0
            .wrapping_mul(2862933555777941757)
            .wrapping_add(3037000493);
        (self.0 >> 33) as usize % n.max(1)
    }
    fn unit(&mut self) -> f64 {
        self.below(1 << 20) as f64 / (1 << 20) as f64
    }
}

fn birth<G: Game>(game: &G, rng: &mut dyn reinfors_core::Rng) -> <G as Game>::State {
    let mut s = game.initial_state();
    while matches!(game.actor(&s), reinfors_core::Actor::Chance) {
        let o = game.chance_node(&s).draw(rng);
        s = game.apply_chance_node(&s, o).next_state;
    }
    s
}

fn reachable_states_round_trip<G>(game: G, plies: usize)
where
    G: Game + StateCodec<State = <G as Game>::State>,
    <G as Game>::State: Clone,
{
    let mut rng = Lcg(7);
    let mut state = birth(&game, &mut rng);
    for _ in 0..plies {
        let bytes = game.encode(&state);
        let back = game
            .decode(&bytes)
            .unwrap_or_else(|e| panic!("reachable state failed to decode: {e}"));
        assert_eq!(
            game.encode(&back),
            bytes,
            "re-encode must be byte-identical"
        );
        game.validate_decoded_state(&back, false)
            .unwrap_or_else(|e| panic!("reachable live state failed validation: {e}"));

        let joint: Vec<usize> = (0..game.num_agents())
            .map(|a| {
                let legal = game.legal_actions(&state, a);
                if legal.is_empty() {
                    0
                } else {
                    legal[rng.below(legal.len())]
                }
            })
            .collect();
        let t = reinfors_core::game::step_env(&game, &state, &joint, &mut rng);
        if t.terminal {
            let bytes = game.encode(&t.next_state);
            let back = game
                .decode(&bytes)
                .unwrap_or_else(|e| panic!("terminal state failed to decode: {e}"));
            assert_eq!(game.encode(&back), bytes);
            game.validate_decoded_state(&back, true)
                .unwrap_or_else(|e| panic!("reachable terminal state failed validation: {e}"));
            state = birth(&game, &mut rng);
        } else {
            state = t.next_state;
        }
    }
}

#[test]
fn every_game_round_trips_reachable_states() {
    reachable_states_round_trip(
        GridWorld {
            size: 5,
            goal: (4, 4),
            max_ticks: None,
        },
        200,
    );
    reachable_states_round_trip(Connect4, 200);
    reachable_states_round_trip(
        Snake {
            num_snakes: 2,
            grid_size: 6,
            initial_length: 2,
            play_to_last: true,
            win_food_lead: None,
            initial_food_count: 2,
            max_ticks: None,
        },
        200,
    );
    reachable_states_round_trip(
        Snake {
            num_snakes: 2,
            grid_size: 6,
            initial_length: 2,
            play_to_last: true,
            win_food_lead: Some(1),
            initial_food_count: 2,
            max_ticks: None,
        },
        200,
    );
    reachable_states_round_trip(Backgammon { max_ticks: None }, 300);
    reachable_states_round_trip(KuhnPoker::default(), 200);
    reachable_states_round_trip(LeducPoker, 300);
    reachable_states_round_trip(
        TexasHoldem {
            num_players: 4,
            stack: 100,
            small_blind: 5,
            big_blind: 10,
        },
        300,
    );
    reachable_states_round_trip(
        Chess {
            max_ticks: None,
            history_len: 8,
        },
        150,
    );
}

#[test]
fn validators_reject_unsafe_states() {
    use reinfors_games::snake::{Action, SnakeBody, SnakeState};
    use std::collections::{HashSet, VecDeque};

    let snake_game = Snake {
        num_snakes: 2,
        grid_size: 6,
        initial_length: 2,
        play_to_last: false,
        win_food_lead: None,
        initial_food_count: 1,
        max_ticks: None,
    };
    let body = |cells: &[(i32, i32)], alive: bool| SnakeBody {
        body: VecDeque::from(cells.to_vec()),
        direction: Action::Up,
        alive,
    };
    let mk = |a: SnakeBody, b: SnakeBody, food: &[(i32, i32)]| SnakeState {
        snakes: vec![a, b],
        food: HashSet::from_iter(food.iter().copied()),
        pending_food: 0,
        birth: false,
    };
    let three = Snake {
        num_snakes: 3,
        grid_size: 6,
        initial_length: 2,
        play_to_last: false,
        win_food_lead: None,
        initial_food_count: 1,
        max_ticks: None,
    };
    let two_state = SnakeState {
        snakes: vec![body(&[(1, 1)], true), body(&[(3, 3)], true)],
        food: HashSet::new(),
        pending_food: 0,
        birth: false,
    };
    assert!(three
        .validate_decoded_state(&two_state, false)
        .unwrap_err()
        .contains("has 2 snakes"));

    let both_alive = mk(body(&[(1, 1)], true), body(&[(3, 3)], true), &[]);
    assert!(snake_game
        .validate_decoded_state(&both_alive, true)
        .unwrap_err()
        .contains("terminal=false"));
    let survivor = mk(body(&[(1, 1)], true), body(&[(3, 3)], false), &[]);
    snake_game.validate_decoded_state(&survivor, true).unwrap();
    assert!(snake_game
        .validate_decoded_state(&survivor, false)
        .unwrap_err()
        .contains("terminal=true"));
    let lead_game = Snake {
        win_food_lead: Some(1),
        ..snake_game
    };
    let led = mk(body(&[(1, 1), (1, 2)], true), body(&[(3, 3)], true), &[]);
    lead_game.validate_decoded_state(&led, true).unwrap();
    assert!(lead_game
        .validate_decoded_state(&led, false)
        .unwrap_err()
        .contains("terminal=true"));
    let off_grid = mk(body(&[(6, 0)], true), body(&[(3, 3)], true), &[]);
    assert!(snake_game
        .validate_decoded_state(&off_grid, false)
        .unwrap_err()
        .contains("outside the grid"));
    let off_grid_food = mk(body(&[(1, 1)], true), body(&[(3, 3)], true), &[(0, -1)]);
    assert!(snake_game
        .validate_decoded_state(&off_grid_food, false)
        .unwrap_err()
        .contains("outside the grid"));
    let headless = mk(body(&[], true), body(&[(3, 3)], true), &[]);
    assert!(snake_game
        .validate_decoded_state(&headless, false)
        .unwrap_err()
        .contains("empty body"));

    let c4 = Connect4;
    let mut bad_cell = vec![2u8, 42, 3];
    bad_cell.extend([0u8; 41]);
    bad_cell.push(0);
    let forged = c4.decode(&bad_cell).unwrap();
    assert!(c4
        .validate_decoded_state(&forged, false)
        .unwrap_err()
        .contains("cell value"));
    let mut bad_turn = vec![2u8, 42];
    bad_turn.extend([0u8; 42]);
    bad_turn.push(2);
    let forged = c4.decode(&bad_turn).unwrap();
    assert!(c4
        .validate_decoded_state(&forged, false)
        .unwrap_err()
        .contains("turn 2"));

    let gw = GridWorld {
        size: 5,
        goal: (4, 4),
        max_ticks: None,
    };
    let live = GridState {
        pos: (0, 0),
        done: false,
    };
    assert!(gw
        .validate_decoded_state(&live, true)
        .unwrap_err()
        .contains("disagrees"));
}

#[test]
fn derived_flags_are_recomputed_at_decode() {
    let c4 = Connect4;
    let mut rng = Lcg(3);
    let mut s = c4.initial_state();
    loop {
        let mover = match c4.actor(&s) {
            reinfors_core::Actor::Agent(a) => a,
            _ => unreachable!(),
        };
        let legal = c4.legal_actions(&s, mover);
        let mut joint = vec![0usize; 2];
        joint[mover] = legal[rng.below(legal.len())];
        let t = c4.step(&s, &joint);
        s = t.next_state;
        if t.terminal {
            break;
        }
    }
    let back = c4.decode(&c4.encode(&s)).unwrap();
    assert!(back.is_done(), "terminal flag must be recomputed at decode");
    c4.validate_decoded_state(&back, true).unwrap();

    let gw = GridWorld {
        size: 5,
        goal: (0, 1),
        max_ticks: None,
    };
    let s = reinfors_games::GridState {
        pos: (0, 1),
        done: true,
    };
    let back = gw.decode(&gw.encode(&s)).unwrap();
    assert!(back.done, "goal position must decode as done");
    gw.validate_decoded_state(&back, true).unwrap();
}

#[test]
fn unreachable_but_safe_states_are_accepted() {
    use reinfors_games::snake::{Action, SnakeBody, SnakeState};
    use std::collections::{HashSet, VecDeque};

    let game = Snake {
        num_snakes: 2,
        grid_size: 6,
        initial_length: 2,
        play_to_last: false,
        win_food_lead: Some(1),
        initial_food_count: 1,
        max_ticks: None,
    };
    let body = |cells: &[(i32, i32)]| SnakeBody {
        body: VecDeque::from(cells.to_vec()),
        direction: Action::Up,
        alive: true,
    };
    let overlap = SnakeState {
        snakes: vec![body(&[(1, 1)]), body(&[(1, 1)])],
        food: HashSet::from_iter([(1, 1)]),
        pending_food: 0,
        birth: false,
    };
    game.validate_decoded_state(&overlap, false).unwrap();

    let c4 = Connect4;
    let mut bytes = vec![2u8, 42, 1, 1];
    bytes.extend([0u8; 40]);
    bytes.push(0);
    let s = c4.decode(&bytes).unwrap();
    c4.validate_decoded_state(&s, false).unwrap();
    assert!(!c4.legal_actions(&s, 0).is_empty());
}
