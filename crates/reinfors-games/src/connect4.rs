//! Connect-4 — a sequential 2-player zero-sum `Game`, the first adversarial non-snake game. It
//! validates the search's sequential path: alternating `Actor::Agent(0)` / `Actor::Agent(1)` nodes
//! (the searching agent's MAX turns vs the opponent's modeled-chance turns), driven through the same
//! generic search + rollout engine as snake. Deterministic, so declared chance is the default
//! (`None`). Standard rules: only non-full columns are legal (the searches mask to the legal set;
//! the retired "full column = immediate loss" rule predates sparse legality).

use reinfors_core::{ActionView, Actor, Game, Reward, Rng, Space, StateEncoder, Transition};

const COLS: usize = 7;
const ROWS: usize = 6;
const CONNECT: usize = 4;

/// serde stops at 32-element arrays; the 42-cell board round-trips through a length-checked Vec.
mod cells_serde {
    use serde::de::Error as _;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S: Serializer>(
        cells: &[u8; super::COLS * super::ROWS],
        ser: S,
    ) -> Result<S::Ok, S::Error> {
        cells.as_slice().serialize(ser)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(
        de: D,
    ) -> Result<[u8; super::COLS * super::ROWS], D::Error> {
        let v = Vec::<u8>::deserialize(de)?;
        v.try_into().map_err(|v: Vec<u8>| {
            D::Error::custom(format!("board has {} cells, expected 42", v.len()))
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Connect4State {
    #[serde(with = "cells_serde")]
    cells: [u8; COLS * ROWS], // 0 empty, 1 player-0, 2 player-1; index = row*COLS + col, row 0 = bottom
    turn: usize, // whose move it is (0 or 1)
    /// Derived (some line completed, or the board full), so the codec recomputes it at decode
    /// rather than transporting a second copy of the fact.
    #[serde(skip)]
    done: bool,
}

impl Connect4State {
    /// The board as `[row][col]` cell codes (0 empty, 1 player-0, 2 player-1), row 0 = bottom — for
    /// rendering / inspection (the encoder owns the network view).
    pub fn board(&self) -> Vec<Vec<u8>> {
        (0..ROWS)
            .map(|r| (0..COLS).map(|c| self.cells[r * COLS + c]).collect())
            .collect()
    }

    /// Whose move it is (0 or 1).
    pub fn turn(&self) -> usize {
        self.turn
    }

    pub fn is_done(&self) -> bool {
        self.done
    }
}

/// A player's outcome on one tick: nothing yet (game ongoing), or the terminal result from its view.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Connect4Event {
    #[default]
    Ongoing,
    Win,
    Loss,
    Draw,
}

/// Connect-4's terminal reward weights (zero-sum: the loser gets `loss`, the winner `win`).
#[derive(Clone, Copy, Debug)]
pub struct Connect4Reward {
    pub win: f64,
    pub loss: f64,
    pub draw: f64,
}

impl Default for Connect4Reward {
    fn default() -> Self {
        Connect4Reward {
            win: 1.0,
            loss: -1.0,
            draw: 0.0,
        }
    }
}

impl Reward for Connect4Reward {
    type Event = Connect4Event;

    fn step_reward(&self, event: &Connect4Event, _agent: usize) -> f64 {
        match event {
            Connect4Event::Ongoing => 0.0,
            Connect4Event::Win => self.win,
            Connect4Event::Loss => self.loss,
            Connect4Event::Draw => self.draw,
        }
    }
}

/// Standard 7x6 Connect-4. Rules only; the reward (zero-sum win/loss/draw) is the decoupled
/// [`Connect4Reward`].
#[derive(Default)]
pub struct Connect4;

impl Connect4 {
    fn at(cells: &[u8], r: usize, c: usize) -> u8 {
        cells[r * COLS + c]
    }

    /// Lowest empty row in `col`, or `None` if the column is full.
    fn drop_row(cells: &[u8], col: usize) -> Option<usize> {
        (0..ROWS).find(|&r| Self::at(cells, r, col) == 0)
    }

    /// Whether the piece just placed at `(r, c)` by `player` completes a line of `CONNECT`.
    fn wins(cells: &[u8], r: usize, c: usize, player: u8) -> bool {
        const DIRS: [(i32, i32); 4] = [(0, 1), (1, 0), (1, 1), (1, -1)]; // horiz, vert, two diagonals
        for (dr, dc) in DIRS {
            let mut count = 1;
            for sign in [1i32, -1] {
                let (mut rr, mut cc) = (r as i32 + sign * dr, c as i32 + sign * dc);
                while rr >= 0
                    && rr < ROWS as i32
                    && cc >= 0
                    && cc < COLS as i32
                    && Self::at(cells, rr as usize, cc as usize) == player
                {
                    count += 1;
                    rr += sign * dr;
                    cc += sign * dc;
                }
            }
            if count >= CONNECT {
                return true;
            }
        }
        false
    }

    /// Whether the position is terminal by the rules alone: some completed line, or a full board.
    /// Quantifies [`Self::wins`] — the one definition of a line — over the grid; `step` applies
    /// the same primitive incrementally at the last drop. The codec recomputes `done` from this
    /// at decode, so the flag is never transported.
    fn board_terminal(cells: &[u8; COLS * ROWS]) -> bool {
        let has_win = (0..ROWS).any(|r| {
            (0..COLS).any(|c| {
                let v = cells[r * COLS + c];
                v != 0 && Self::wins(cells, r, c, v)
            })
        });
        has_win || cells.iter().all(|&c| c != 0)
    }

    /// Per-agent terminal event vector for a win by `winner`, or a draw.
    fn outcome_events(winner: Option<usize>) -> Vec<Connect4Event> {
        match winner {
            Some(w) => (0..2)
                .map(|a| {
                    if a == w {
                        Connect4Event::Win
                    } else {
                        Connect4Event::Loss
                    }
                })
                .collect(),
            None => vec![Connect4Event::Draw; 2],
        }
    }
}

impl Game for Connect4 {
    type State = Connect4State;
    type Event = Connect4Event;

    fn num_agents(&self) -> usize {
        2
    }

    fn action_count(&self) -> usize {
        COLS
    }

    fn actor(&self, state: &Connect4State) -> Actor {
        Actor::Agent(state.turn)
    }

    fn legal_actions(&self, state: &Connect4State, agent: usize) -> Vec<usize> {
        // Standard rules: only non-full columns are playable (matching every reference
        // implementation, incl. OpenSpiel). The pre-chess "full column = immediate loss" rule was
        // scaffolding for the retired all-actions-always-legal contract.
        if agent == state.turn && !state.done {
            (0..COLS)
                .filter(|&c| Self::drop_row(&state.cells, c).is_some())
                .collect()
        } else {
            Vec::new()
        }
    }

    fn step(
        &self,
        state: &Connect4State,
        actions: &[usize],
    ) -> Transition<Connect4State, Connect4Event> {
        let mover = state.turn;
        let col = actions[mover];
        let mut cells = state.cells;
        let (next_turn, winner, terminal) = match Self::drop_row(&cells, col) {
            // Full column: unreachable via legal play (searches mask to the legal set, the Env
            // boundary validates) — kept as a losing backstop rather than a panic for direct core
            // callers.
            None => (mover, Some(1 - mover), true),
            Some(r) => {
                cells[r * COLS + col] = (mover + 1) as u8;
                if Self::wins(&cells, r, col, (mover + 1) as u8) {
                    (mover, Some(mover), true)
                } else if cells.iter().all(|&v| v != 0) {
                    (mover, None, true) // board full: draw
                } else {
                    (1 - mover, None, false)
                }
            }
        };
        let events = if terminal {
            Self::outcome_events(winner).into_iter().map(Some).collect()
        } else {
            vec![None; 2]
        };
        Transition {
            next_state: Connect4State {
                cells,
                turn: next_turn,
                done: terminal,
            },
            events,
            terminal,
        }
    }

    fn initial_state(&self, _rng: &mut dyn Rng) -> Connect4State {
        Connect4State {
            cells: [0; COLS * ROWS],
            turn: 0,
            done: false,
        }
    }

    // Deterministic transitions: no chance states.
}

/// The default Connect-4 observation: two own/opponent piece planes from the mover's perspective.
pub struct Connect4Planes;

impl ActionView for Connect4Planes {} // absolute: identity action view

impl StateEncoder for Connect4Planes {
    type State = Connect4State;

    fn encode(&self, state: &Connect4State, agent: usize) -> Vec<f32> {
        let mine = (agent + 1) as u8;
        let plane = ROWS * COLS;
        let mut obs = vec![0.0f32; 2 * plane];
        for (i, &v) in state.cells.iter().enumerate() {
            if v == mine {
                obs[i] = 1.0;
            } else if v != 0 {
                obs[plane + i] = 1.0;
            }
        }
        obs
    }

    fn obs_shape(&self) -> (usize, usize, usize) {
        (2, ROWS, COLS) // channel 0 = own pieces, 1 = opponent's
    }

    fn observation_space(&self) -> Space {
        let (c, h, w) = self.obs_shape();
        Space::unit_box(vec![c, h, w]) // both planes are one-hot occupancy: values in [0, 1]
    }
}

impl reinfors_core::StateCodec for Connect4 {
    type State = Connect4State;

    fn encode(&self, s: &Connect4State) -> Vec<u8> {
        crate::codec_util::serde_encode(2, s)
    }

    fn decode(&self, bytes: &[u8]) -> Result<Connect4State, String> {
        let mut s: Connect4State = crate::codec_util::serde_decode(2, bytes)?;
        s.done = Self::board_terminal(&s.cells);
        Ok(s)
    }

    fn validate_decoded_state(&self, state: &Connect4State, done: bool) -> Result<(), String> {
        if state.cells.iter().any(|&c| c > 2) {
            return Err("cell value out of range".into());
        }
        if state.turn > 1 {
            return Err(format!("turn {} out of range", state.turn));
        }
        if state.done != done {
            return Err(format!(
                "derived done flag {} disagrees with envelope done {done}",
                state.done
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use reinfors_core::{
        search_many, ChanceMode, Engine, EngineParams, Opponent, SearchConfig, SelectiveExpectimax,
        TreeStrap,
    };

    fn cfg() -> SearchConfig {
        SearchConfig {
            gamma: 0.99,
            beta: 1.0,
            expansion_budget: 48,
            top_k: 4,
            max_depth: 8,
            chance: ChanceMode::Committed { samples: 1 },
            opponent: Opponent::Uniform,
        }
    }

    // K=2 heads, A=7, zero leaf values — so the only signal is the terminal win/loss reward.
    fn zero_infer(_players: &[usize], _obs: Vec<f32>, n: usize) -> Vec<f64> {
        vec![0.0; n * 2 * 7]
    }

    fn at(cells: &mut [u8], r: usize, c: usize, v: u8) {
        cells[r * COLS + c] = v;
    }

    #[test]
    fn search_is_invariant_to_illegal_action_values() {
        // The review oracle: two nets IDENTICAL on legal columns and wildly different only on the
        // full column's phantom slot must produce identical search output — root values, interior
        // TreeStrap targets, and their count. Guards every legality seam at once: leaf bootstraps
        // (per-head max over the mover's legal set), the distributional opponent (softmax gathered
        // to the legal set, weights summing to 1), and densified targets.
        let mut cells = [0u8; 42];
        for r in 0..ROWS {
            at(&mut cells, r, 0, if r % 2 == 0 { 1 } else { 2 });
        }
        let state = Connect4State {
            cells,
            turn: 0,
            done: false,
        };
        for opponent in [
            Opponent::Uniform,
            Opponent::Distributional {
                temperature: 1.0,
                floor: 0.05,
            },
        ] {
            let cfg = SearchConfig {
                gamma: 0.99,
                beta: 1.0,
                expansion_budget: 16,
                top_k: 4,
                max_depth: 4,
                chance: ChanceMode::Committed { samples: 1 },
                opponent,
            };
            let run = |phantom: f64| {
                let mut infer = move |_players: &[usize], _obs: Vec<f32>, n: usize| -> Vec<f64> {
                    let mut out = Vec::with_capacity(n * 7);
                    for _ in 0..n {
                        out.push(phantom); // column 0 is full (illegal) throughout this subtree
                        for c in 1..7 {
                            out.push(-0.5 + 0.1 * c as f64); // all-negative legal values
                        }
                    }
                    out
                };
                search_many(
                    &Connect4,
                    &Connect4Planes,
                    &Connect4Reward::default(),
                    &cfg,
                    vec![(state.clone(), 0)],
                    true,
                    3,
                    &mut infer,
                )
                .remove(0)
            };
            let (lo_v, lo_i, _) = run(-100.0);
            let (hi_v, hi_i, _) = run(100.0);
            assert_eq!(lo_v, hi_v, "root values must ignore the phantom slot");
            assert_eq!(lo_i.len(), hi_i.len(), "record counts must match");
            assert_eq!(lo_i, hi_i, "interior targets must ignore the phantom slot");
            // and the phantom slot itself is a densified zero, never the max of anything
            assert_eq!(lo_v[0][0], 0.0);
        }
    }

    #[test]
    fn drop_win_full_column_and_turn_flip() {
        let g = Connect4;
        let empty = Connect4State {
            cells: [0; 42],
            turn: 0,
            done: false,
        };
        // A normal move stacks at the bottom and passes the turn.
        let t = g.step(&empty, &[0]);
        assert_eq!(t.next_state.cells[0], 1); // player 0 at (0,0)
        assert_eq!(t.next_state.turn, 1);
        assert!(!t.terminal && t.events == vec![None, None]);
        // Completing four-in-a-row wins (player 0 has 3 across the bottom; col 3 finishes it).
        let mut cells = [0u8; 42];
        for c in 0..3 {
            at(&mut cells, 0, c, 1);
        }
        let t = g.step(
            &Connect4State {
                cells,
                turn: 0,
                done: false,
            },
            &[3],
        );
        assert!(
            t.terminal && t.events == vec![Some(Connect4Event::Win), Some(Connect4Event::Loss)]
        );
        // Standard rules: a full column is ILLEGAL — legal_actions excludes it, and stepping it
        // anyway (unreachable via legal play) hits the losing backstop rather than corrupting.
        let mut full = [0u8; 42];
        for r in 0..ROWS {
            at(&mut full, r, 0, if r % 2 == 0 { 1 } else { 2 });
        }
        let full_state = Connect4State {
            cells: full,
            turn: 0,
            done: false,
        };
        assert_eq!(
            g.legal_actions(&full_state, 0),
            (1..COLS).collect::<Vec<_>>(),
            "the full column must be masked out"
        );
        let t = g.step(
            &Connect4State {
                cells: full,
                turn: 0,
                done: false,
            },
            &[0],
        );
        assert!(
            t.terminal && t.events == vec![Some(Connect4Event::Loss), Some(Connect4Event::Win)]
        );
    }

    #[test]
    fn sequential_metadata_and_legality() {
        let g = Connect4;
        let s = Connect4State {
            cells: [0; 42],
            turn: 1,
            done: false,
        };
        assert_eq!(g.num_agents(), 2);
        assert_eq!(g.action_count(), 7);
        assert_eq!(g.actor(&s), Actor::Agent(1)); // whose turn it is
        assert_eq!(g.legal_actions(&s, 1).len(), 7); // the mover has all columns
        assert!(g.legal_actions(&s, 0).is_empty()); // the other player has no move
    }

    #[test]
    fn search_finds_the_winning_drop() {
        // Player 0 to move with three across the bottom (cols 0-2) and player 1 elsewhere: dropping in
        // column 3 wins immediately, so it must be the best root action with value = the win reward.
        // This drives the sequential MAX-vs-opponent-chance search.
        let g = Connect4;
        let mut cells = [0u8; 42];
        for c in 0..3 {
            at(&mut cells, 0, c, 1); // player 0
        }
        for c in 4..7 {
            at(&mut cells, 0, c, 2); // player 1 (so move counts are consistent with turn 0)
        }
        let state = Connect4State {
            cells,
            turn: 0,
            done: false,
        };
        let results = search_many(
            &g,
            &Connect4Planes,
            &Connect4Reward::default(),
            &cfg(),
            vec![(state, 0)],
            false,
            0,
            zero_infer,
        );
        let values = &results[0].0; // [K][7]
        for head in values {
            let best = (0..7)
                .max_by(|&a, &b| head[a].partial_cmp(&head[b]).unwrap())
                .unwrap();
            assert_eq!(best, 3, "the winning drop (col 3) should be best: {head:?}");
            assert!(
                (head[3] - 1.0).abs() < 1e-9,
                "its value is the undiscounted win reward"
            );
        }
    }

    #[test]
    fn engine_rolls_out_a_sequential_two_player_game() {
        // The rollout engine plays full Connect-4 games — turns alternate (only the mover is active
        // each tick), both players' trajectories are z-mixed at game end, and records are [K][A=7].
        // Exercises the Actor::Agent(other) opponent nodes end to end.
        let policy = SelectiveExpectimax::new(cfg(), 2, 0.0); // n_heads, epsilon
        let learner = TreeStrap::new(0.99, 0.3, 1.0, false); // gamma, outcome_weight, bootstrap_p, interior
        let params = EngineParams {
            n_games: 3,
            seed: 0,
        };
        let mut engine = Engine::new(
            Connect4,
            Box::new(Connect4Planes),
            Box::new(Connect4Reward::default()),
            policy,
            learner,
            params,
        );
        let (records, stats) = engine.collect(60, |o, n| zero_infer(&[], o, n));
        assert!(records.len() >= 60);
        for (obs, tgt, mask, _player) in &records {
            assert_eq!(obs.len(), 2 * ROWS * COLS);
            assert_eq!(tgt.len(), 2); // K heads
            assert!(tgt.iter().all(|row| row.len() == 7)); // A columns
            assert_eq!(mask.len(), 2);
        }
        assert!(
            stats.decisions > 0 && !stats.episodes.is_empty(),
            "games should finish"
        );
    }
}
