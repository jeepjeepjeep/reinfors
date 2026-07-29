//! The reachability property every state codec must satisfy: any state produced by real play
//! round-trips canonically (encode∘decode = identity on bytes) and validates — live states with
//! done=false, terminal states with done=true. This is the false-REJECTION guard for
//! `validate_decoded_state`. The contract is safety, not reachability: probes below check that
//! unsafe states reject, that derived flags are recomputed at decode rather than transported,
//! and that unreachable-but-safe states are ACCEPTED (only reinfors-produced snapshots have
//! meaningful gameplay semantics).

use reinfors_core::{Game, Rng, StateCodec};
use reinfors_games::{Backgammon, Chess, Connect4, GridWorld, Snake, TexasHoldem};

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
            // TERMINAL states are snapshot-restorable too — the round-trip + validation
            // property must hold for them with done=true (the recomputed state-side flags must
            // agree with the envelope).
            let bytes = game.encode(&t.next_state);
            let back = game
                .decode(&bytes)
                .unwrap_or_else(|e| panic!("terminal state failed to decode: {e}"));
            assert_eq!(game.encode(&back), bytes);
            game.validate_decoded_state(&back, true)
                .unwrap_or_else(|e| panic!("reachable terminal state failed validation: {e}"));
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

/// False-ACCEPT probes for the SAFETY contract: states game methods cannot operate on safely
/// (out-of-bounds indexing, panicking representation) must reject.
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
    };
    // a state with the wrong snake count would index out of the game's agent range
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
    };
    assert!(three
        .validate_decoded_state(&two_state, false)
        .unwrap_err()
        .contains("has 2 snakes"));

    // lifecycle coherence, both directions: `done` gates whether the Env may continue, so it
    // must match the shared terminal rule (`advance`'s own) — a live envelope over a decided
    // position would let play continue past the end of the game.
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
    // out-of-grid cells index past the observation planes
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
    // an alive snake with no body panics `head()`
    let headless = mk(body(&[], true), body(&[(3, 3)], true), &[]);
    assert!(snake_game
        .validate_decoded_state(&headless, false)
        .unwrap_err()
        .contains("empty body"));

    // connect4: cell codes outside {empty, p0, p1} and out-of-range movers reject. Fields are
    // private, so states are built via decode: postcard layout = version, cells len varint,
    // 42 cells, turn (done is derived, never on the wire).
    let c4 = Connect4;
    let mut bad_cell = vec![2u8, 42, 3];
    bad_cell.extend([0u8; 41]);
    bad_cell.push(0); // turn 0
    let forged = c4.decode(&bad_cell).unwrap();
    assert!(c4
        .validate_decoded_state(&forged, false)
        .unwrap_err()
        .contains("cell value"));
    let mut bad_turn = vec![2u8, 42];
    bad_turn.extend([0u8; 42]);
    bad_turn.push(2); // turn 2
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
    // lifecycle coherence: envelope done must agree with the recomputed state flag
    let live = gw.initial_state(&mut Lcg(1));
    assert!(gw
        .validate_decoded_state(&live, true)
        .unwrap_err()
        .contains("disagrees"));
}

/// Derived flags never travel: decode recomputes them from the same rule functions `step` uses,
/// so a stored flag cannot disagree with the position (the old duplicated-fact forgeries are now
/// unrepresentable, not just rejected).
#[test]
fn derived_flags_are_recomputed_at_decode() {
    // connect4: play to a win, round-trip, and the decoded state knows it is done.
    let c4 = Connect4;
    let mut rng = Lcg(3);
    let mut s = c4.initial_state(&mut rng);
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

    // gridworld: a state at the goal decodes as done.
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

/// The narrowed contract's flip side: unreachable-but-SAFE states are accepted. No occupancy or
/// alternation rules are re-proved at the boundary (lifecycle coherence — envelope done vs the
/// shared terminal rule — is checked, but history is not).
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
    // two living snakes on the same cell, food under both: impossible through play (occupancy is
    // not re-proved), safe to step, and live under the terminal rule (equal lengths, both alive)
    let overlap = SnakeState {
        snakes: vec![body(&[(1, 1)]), body(&[(1, 1)])],
        food: HashSet::from_iter([(1, 1)]),
    };
    game.validate_decoded_state(&overlap, false).unwrap();

    // connect4: a parity-violating board (two p0 pieces, none for p1) is safe to play on.
    let c4 = Connect4;
    let mut bytes = vec![2u8, 42, 1, 1];
    bytes.extend([0u8; 40]);
    bytes.push(0); // turn 0 again — alternation violated
    let s = c4.decode(&bytes).unwrap();
    c4.validate_decoded_state(&s, false).unwrap();
    assert!(!c4.legal_actions(&s, 0).is_empty());
}
