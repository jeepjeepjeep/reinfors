//! Connect-4 — a sequential 2-player zero-sum `Game`, the first adversarial non-snake game. It
//! validates the search's sequential path: alternating `Actor::Agent(0)` / `Actor::Agent(1)` nodes
//! (the searching agent's MAX turns vs the opponent's modeled-chance turns), driven through the same
//! generic search + rollout engine as snake. Deterministic, so `sample_chance` is the default
//! (empty). Action legality is fixed — all 7 columns are always selectable; a move into a full column
//! is an immediate loss for the mover — so the framework needs no action masking here.

use reinfors_core::{Actor, Game, Rng, StateEncoder, Transition};

const COLS: usize = 7;
const ROWS: usize = 6;
const CONNECT: usize = 4;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Connect4State {
    cells: [u8; COLS * ROWS], // 0 empty, 1 player-0, 2 player-1; index = row*COLS + col, row 0 = bottom
    turn: usize,              // whose move it is (0 or 1)
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

/// Standard 7x6 Connect-4 with zero-sum terminal rewards.
#[derive(Default)]
pub struct Connect4 {
    pub reward: Connect4Reward,
}

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

    /// Per-agent terminal reward vector for a win by `winner`, or a draw.
    fn outcome_rewards(&self, winner: Option<usize>) -> Vec<f64> {
        match winner {
            Some(w) => (0..2)
                .map(|a| {
                    if a == w {
                        self.reward.win
                    } else {
                        self.reward.loss
                    }
                })
                .collect(),
            None => vec![self.reward.draw; 2],
        }
    }
}

impl Game for Connect4 {
    type State = Connect4State;

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
        // Only the player to move acts, and all columns are always selectable (a full column loses).
        if agent == state.turn && !state.done {
            (0..COLS).collect()
        } else {
            Vec::new()
        }
    }

    fn step(&self, state: &Connect4State, actions: &[usize]) -> Transition<Connect4State> {
        let mover = state.turn;
        let col = actions[mover];
        let mut cells = state.cells;
        let (next_turn, winner, terminal) = match Self::drop_row(&cells, col) {
            None => (mover, Some(1 - mover), true), // full column: the mover loses
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
        let rewards = if terminal {
            self.outcome_rewards(winner)
        } else {
            vec![0.0; 2]
        };
        Transition {
            next_state: Connect4State {
                cells,
                turn: next_turn,
                done: terminal,
            },
            rewards,
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

    // Deterministic: no `sample_chance` / `step_env` override needed (the trait defaults suffice).
}

/// The default Connect-4 observation: two own/opponent piece planes from the mover's perspective.
pub struct Connect4Planes;

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
}

#[cfg(test)]
mod tests {
    use super::*;
    use reinfors_core::{
        search_many, Engine, EngineParams, Opponent, SearchConfig, SelectiveExpectimax, TreeStrap,
    };

    fn cfg() -> SearchConfig {
        SearchConfig {
            gamma: 0.99,
            beta: 1.0,
            expansion_budget: 48,
            top_k: 4,
            max_depth: 8,
            food_samples: 1,
            opponent: Opponent::Uniform,
        }
    }

    // K=2 heads, A=7, zero leaf values — so the only signal is the terminal win/loss reward.
    fn zero_infer(_obs: Vec<f32>, n: usize) -> Vec<f64> {
        vec![0.0; n * 2 * 7]
    }

    fn at(cells: &mut [u8], r: usize, c: usize, v: u8) {
        cells[r * COLS + c] = v;
    }

    #[test]
    fn drop_win_full_column_and_turn_flip() {
        let g = Connect4::default();
        let empty = Connect4State {
            cells: [0; 42],
            turn: 0,
            done: false,
        };
        // A normal move stacks at the bottom and passes the turn.
        let t = g.step(&empty, &[0]);
        assert_eq!(t.next_state.cells[0], 1); // player 0 at (0,0)
        assert_eq!(t.next_state.turn, 1);
        assert!(!t.terminal && t.rewards == vec![0.0, 0.0]);
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
        assert!(t.terminal && t.rewards == vec![1.0, -1.0]);
        // A move into a full column loses immediately.
        let mut full = [0u8; 42];
        for r in 0..ROWS {
            at(&mut full, r, 0, if r % 2 == 0 { 1 } else { 2 });
        }
        let t = g.step(
            &Connect4State {
                cells: full,
                turn: 0,
                done: false,
            },
            &[0],
        );
        assert!(t.terminal && t.rewards == vec![-1.0, 1.0]);
    }

    #[test]
    fn sequential_metadata_and_legality() {
        let g = Connect4::default();
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
        let g = Connect4::default();
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
            max_ticks: 50,
            seed: 0,
        };
        let mut engine = Engine::new(
            Connect4::default(),
            Box::new(Connect4Planes),
            policy,
            learner,
            params,
        );
        let (records, stats) = engine.collect(60, zero_infer);
        assert!(records.len() >= 60);
        for (obs, tgt, mask) in &records {
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
