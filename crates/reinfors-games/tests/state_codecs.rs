//! The reachability property every state codec must satisfy: any state produced by real play
//! round-trips canonically (encode∘decode = identity on bytes) and validates as a live state.
//! This is the false-REJECTION guard for `validate_state` — targeted false-ACCEPT probes live
//! next to each game's codec.

use reinfors_core::{Game, Rng, StateCodec};
use reinfors_games::{Backgammon, Chess, Connect4, GridWorld, Snake};

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

fn reachable_states_round_trip<G>(game: G, plies: usize)
where
    G: Game + StateCodec<State = <G as Game>::State>,
    <G as Game>::State: Clone,
{
    let mut rng = Lcg(7);
    let mut state = game.initial_state(&mut rng);
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
        game.validate_state(&back, false)
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
            state = game.initial_state(&mut rng);
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
            grid_size: 6,
            initial_length: 2,
            play_to_last: true,
            win_food_lead: None,
            initial_food_count: 2,
            max_ticks: None,
        },
        200,
    );
    reachable_states_round_trip(Backgammon { max_ticks: None }, 300);
    reachable_states_round_trip(
        Chess {
            max_ticks: None,
            history_len: 8,
        },
        150,
    );
}

/// False-ACCEPT probes: hand-built invalid states each validator must reject.
#[test]
fn validators_reject_invalid_states() {
    use reinfors_games::snake::{Action, SnakeBody, SnakeState};
    use std::collections::{HashSet, VecDeque};

    let snake_game = Snake {
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
        snakes: [a, b],
        food: HashSet::from_iter(food.iter().copied()),
    };
    // living overlap rejected; the same overlap with one snake DEAD is legitimate
    let overlap = mk(body(&[(1, 1)], true), body(&[(1, 1)], true), &[]);
    assert!(snake_game
        .validate_state(&overlap, false)
        .unwrap_err()
        .contains("overlap"));
    let corpse = mk(body(&[(1, 1)], true), body(&[(1, 1)], false), &[]);
    snake_game.validate_state(&corpse, true).unwrap(); // lone survivor: terminal (no play_to_last)
    assert!(snake_game
        .validate_state(&corpse, false)
        .unwrap_err()
        .contains("terminal"));
    let food_on_body = mk(body(&[(1, 1)], true), body(&[(3, 3)], true), &[(1, 1)]);
    assert!(snake_game
        .validate_state(&food_on_body, false)
        .unwrap_err()
        .contains("living snake"));

    // connect4: terminal consistency BOTH directions (the byte-era codec missed the done=true
    // side). Fields are private, so the undecided-but-done state is built via decode: postcard
    // layout = version, cells len varint, 42 cells, turn, done.
    let c4 = Connect4;
    let mut undecided_done = vec![2u8, 42];
    undecided_done.extend([0u8; 42]);
    undecided_done.extend([0u8, 1]); // turn 0, done TRUE on an empty board
    let forged = c4.decode(&undecided_done).unwrap();
    assert!(c4
        .validate_state(&forged, true)
        .unwrap_err()
        .contains("neither won nor full"));

    let gw = GridWorld {
        size: 5,
        goal: (4, 4),
        max_ticks: None,
    };
    let mut at_goal = gw.initial_state(&mut Lcg(1));
    let bytes = gw.encode(&at_goal);
    at_goal = gw.decode(&bytes).unwrap();
    assert!(gw
        .validate_state(&at_goal, true)
        .unwrap_err()
        .contains("disagrees"));
}
