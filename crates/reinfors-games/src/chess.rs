//! Chess via `cozy-chess` (MIT; engine-grade bitboard movegen, perft-validated upstream). The crate
//! is an implementation detail behind the `Game` trait: reinfors owns the action-id mapping, the
//! termination policy, events/reward, and the observation encoder; cozy-chess owns move generation,
//! make-move, and check/checkmate/stalemate detection — the parts where silent bugs poison training
//! data.
//!
//! **Action space: the AlphaZero 8×8×73 = 4672 encoding**, absolute orientation (no per-side board
//! flip — one convention for both players, matching the absolute observation planes):
//! `index = from_square·73 + move_type`, with move types
//!   0..56   ray moves: direction (N, NE, E, SE, S, SW, W, NW) × distance (1..7)
//!   56..64  knight moves (8 patterns)
//!   64..73  underpromotions: pawn direction (forward, capture toward file−1, toward file+1) ×
//!           piece (Knight, Bishop, Rook)
//! Queen promotions ride the ray encoding (a pawn ray-move onto the last rank decodes with
//! `promotion = Queen`). Castling needs no special case: cozy-chess encodes it king-takes-own-rook,
//! which is an ordinary E/W ray move of distance 3–4.
//!
//! Only a tiny fraction of the 4672 ids is legal in any position (~35 typical) — the tree searches
//! consume `legal_actions` and keep nodes sparse, so the width costs the net's policy head, not the
//! search.
//!
//! **Termination**: checkmate (mover wins), stalemate, the fifty-move rule (both via
//! `Board::status`), threefold repetition (a hash history kept in the state, reset on irreversible
//! moves), and an insufficient-material draw covering every no-pawn/rook/queen position with at
//! most one minor piece PER SIDE — bare kings, K+minor vs K, and K+minor vs K+minor. The last is a
//! deliberate liberty vs strict FIDE (KNvKN / opposite-bishop KBvKB admit helpmates, so FIDE would
//! play on): checkmate is detected before this branch, so no mate is ever mis-scored — it only
//! adjudicates practically-dead endgames early, where the value target is ~0 anyway. A common
//! self-play adjudication.
//! An illegal action id (impossible through the masked searches; reachable through a raw `Env`)
//! is an immediate loss for the mover — the same posture as connect4's full-column rule.

use cozy_chess::{Board, Color, File, GameStatus, Move, Piece, Rank, Square};
use reinfors_core::{Actor, Game, Reward, Rng, StateEncoder, Transition};

pub const CHESS_ACTIONS: usize = 64 * 73; // 4672

/// (file delta, rank delta) per ray direction, in move-type order.
const RAY_DIRS: [(i8, i8); 8] = [
    (0, 1),   // N
    (1, 1),   // NE
    (1, 0),   // E
    (1, -1),  // SE
    (0, -1),  // S
    (-1, -1), // SW
    (-1, 0),  // W
    (-1, 1),  // NW
];
const KNIGHT_DIRS: [(i8, i8); 8] = [
    (1, 2),
    (2, 1),
    (2, -1),
    (1, -2),
    (-1, -2),
    (-2, -1),
    (-2, 1),
    (-1, 2),
];
const UNDERPROMO_PIECES: [Piece; 3] = [Piece::Knight, Piece::Bishop, Piece::Rook];

fn sq_index(sq: Square) -> usize {
    sq as usize // LERF: a1 = 0, rank-major
}

fn sq_from(file: i8, rank: i8) -> Option<Square> {
    if !(0..8).contains(&file) || !(0..8).contains(&rank) {
        return None;
    }
    Some(Square::new(
        File::index(file as usize),
        Rank::index(rank as usize),
    ))
}

/// Encode a cozy-chess move into its 8×8×73 action id. `side` disambiguates underpromotion
/// direction (a pawn's "forward" depends on color; the encoding itself stays absolute).
pub fn encode_move(mv: Move, side: Color) -> usize {
    let from = sq_index(mv.from);
    let (ff, fr) = (mv.from.file() as i8, mv.from.rank() as i8);
    let (tf, tr) = (mv.to.file() as i8, mv.to.rank() as i8);
    let (df, dr) = (tf - ff, tr - fr);

    let move_type = match mv.promotion {
        Some(p) if p != Piece::Queen => {
            // Underpromotion: direction relative to the mover's forward.
            let fwd: i8 = if side == Color::White { 1 } else { -1 };
            debug_assert_eq!(dr, fwd);
            let dir = match df {
                0 => 0,  // push
                -1 => 1, // capture toward file-1
                1 => 2,  // capture toward file+1
                _ => unreachable!("pawn promotion with |file delta| > 1"),
            };
            let piece = UNDERPROMO_PIECES.iter().position(|&q| q == p).unwrap();
            64 + dir * 3 + piece
        }
        _ => {
            // Ray or knight move (queen promotions ride the ray encoding).
            if let Some(k) = KNIGHT_DIRS.iter().position(|&(f, r)| f == df && r == dr) {
                56 + k
            } else {
                let dist = df.abs().max(dr.abs());
                debug_assert!(dist >= 1);
                let unit = (df / dist.max(1), dr / dist.max(1));
                let dir = RAY_DIRS
                    .iter()
                    .position(|&(f, r)| (f, r) == unit)
                    .expect("non-ray, non-knight move");
                dir * 7 + (dist as usize - 1)
            }
        }
    };
    from * 73 + move_type
}

/// Decode an action id back into a cozy-chess move for `board`'s position. Returns `None` for
/// geometrically impossible ids (off-board targets). Queen promotion is inferred when a pawn
/// ray-moves onto the last rank.
pub fn decode_move(action: usize, board: &Board) -> Option<Move> {
    let from_idx = action / 73;
    let move_type = action % 73;
    let from = Square::new(File::index(from_idx % 8), Rank::index(from_idx / 8));
    let (ff, fr) = (from.file() as i8, from.rank() as i8);

    let (to, promotion) = if move_type < 56 {
        let (df, dr) = RAY_DIRS[move_type / 7];
        let dist = (move_type % 7 + 1) as i8;
        let to = sq_from(ff + df * dist, fr + dr * dist)?;
        // A pawn arriving on the last rank via a ray move is a queen promotion.
        let is_pawn = board.pieces(Piece::Pawn).has(from);
        let last_rank = matches!(to.rank(), Rank::First | Rank::Eighth);
        let promo = if is_pawn && last_rank {
            Some(Piece::Queen)
        } else {
            None
        };
        (to, promo)
    } else if move_type < 64 {
        let (df, dr) = KNIGHT_DIRS[move_type - 56];
        (sq_from(ff + df, fr + dr)?, None)
    } else {
        let t = move_type - 64;
        let (dir, piece) = (t / 3, t % 3);
        let side = board.side_to_move();
        let fwd: i8 = if side == Color::White { 1 } else { -1 };
        let df = [0i8, -1, 1][dir];
        (sq_from(ff + df, fr + fwd)?, Some(UNDERPROMO_PIECES[piece]))
    };
    Some(Move {
        from,
        to,
        promotion,
    })
}

#[derive(Clone)]
pub struct ChessState {
    pub(crate) board: Board,
    /// Position hashes since the last irreversible move (pawn move / capture / castling-rights
    /// change resets the halfmove clock, which we mirror) — the threefold-repetition window.
    hashes: Vec<u64>,
    finished: Option<ChessOutcome>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum ChessOutcome {
    /// The stored agent index won (its opponent is mated or played an illegal action).
    WonBy(usize),
    Draw,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ChessEvent {
    Ongoing,
    Win,
    Loss,
    Draw,
}

/// Zero-sum chess reward (defaults win 1 / loss −1 / draw 0, the AlphaZero z).
pub struct ChessReward {
    pub win: f64,
    pub loss: f64,
    pub draw: f64,
}

impl Reward for ChessReward {
    type Event = ChessEvent;
    fn step_reward(&self, event: &ChessEvent, _agent: usize) -> f64 {
        match event {
            ChessEvent::Ongoing => 0.0,
            ChessEvent::Win => self.win,
            ChessEvent::Loss => self.loss,
            ChessEvent::Draw => self.draw,
        }
    }
}

/// Standard chess. Rules and outcomes only; the reward is the decoupled [`ChessReward`]. The
/// truncation horizon (`max_ticks`) exists because self-play with weak nets can shuffle
/// indefinitely inside the fifty-move window.
pub struct Chess {
    pub max_ticks: Option<usize>,
}

impl Default for Chess {
    fn default() -> Self {
        Chess {
            max_ticks: Some(512),
        }
    }
}

fn agent_of(color: Color) -> usize {
    match color {
        Color::White => 0,
        Color::Black => 1,
    }
}

/// King + at most one minor piece per side and no other material. Broader than FIDE's dead-position
/// rule (K+minor vs K+minor can still be helpmated) — see the module doc for why that liberty is
/// safe and intended.
fn insufficient_material(board: &Board) -> bool {
    if !(board.pieces(Piece::Pawn) | board.pieces(Piece::Rook) | board.pieces(Piece::Queen))
        .is_empty()
    {
        return false;
    }
    let minors = board.pieces(Piece::Knight) | board.pieces(Piece::Bishop);
    (minors & board.colors(Color::White)).len() <= 1
        && (minors & board.colors(Color::Black)).len() <= 1
}

impl ChessState {
    fn repetition_count(&self) -> usize {
        let current = *self.hashes.last().expect("hash history is never empty");
        self.hashes.iter().filter(|&&h| h == current).count()
    }

    pub fn is_done(&self) -> bool {
        self.finished.is_some()
    }

    /// The board as a FEN string (for play/debug surfaces).
    pub fn fen(&self) -> String {
        format!("{}", self.board)
    }

    pub fn turn(&self) -> usize {
        agent_of(self.board.side_to_move())
    }
}

impl Game for Chess {
    type State = ChessState;
    type Event = ChessEvent;

    fn num_agents(&self) -> usize {
        2
    }

    fn action_count(&self) -> usize {
        CHESS_ACTIONS
    }

    fn actor(&self, state: &ChessState) -> Actor {
        Actor::Agent(state.turn())
    }

    fn legal_actions(&self, state: &ChessState, agent: usize) -> Vec<usize> {
        if state.is_done() || agent != state.turn() {
            return Vec::new();
        }
        let side = state.board.side_to_move();
        let mut out = Vec::with_capacity(48);
        state.board.generate_moves(|moves| {
            for mv in moves {
                out.push(encode_move(mv, side));
            }
            false
        });
        out
    }

    fn step(&self, state: &ChessState, actions: &[usize]) -> Transition<ChessState, ChessEvent> {
        let mover = state.turn();
        let other = 1 - mover;
        let mut next = state.clone();
        let mut events = vec![ChessEvent::Ongoing; 2];

        let decoded = decode_move(actions[mover], &state.board);
        let legal = decoded.is_some_and(|mv| next.board.try_play(mv).is_ok());
        if !legal {
            // Unreachable through the masked searches; a raw Env can still submit anything. Same
            // posture as connect4's full column: an illegal action is an immediate loss.
            next.finished = Some(ChessOutcome::WonBy(other));
            events[mover] = ChessEvent::Loss;
            events[other] = ChessEvent::Win;
            return Transition {
                next_state: next,
                events,
                terminal: true,
            };
        }

        // Maintain the repetition window: an irreversible move (halfmove clock reset) clears it.
        if next.board.halfmove_clock() == 0 {
            next.hashes.clear();
        }
        next.hashes.push(next.board.hash());

        let outcome = match next.board.status() {
            GameStatus::Won => Some(ChessOutcome::WonBy(mover)), // side to move is mated
            GameStatus::Drawn => Some(ChessOutcome::Draw),       // stalemate or fifty-move
            GameStatus::Ongoing => {
                if next.repetition_count() >= 3 || insufficient_material(&next.board) {
                    Some(ChessOutcome::Draw)
                } else {
                    None
                }
            }
        };
        if let Some(outcome) = outcome {
            next.finished = Some(outcome);
            match outcome {
                ChessOutcome::WonBy(w) => {
                    events[w] = ChessEvent::Win;
                    events[1 - w] = ChessEvent::Loss;
                }
                ChessOutcome::Draw => {
                    events = vec![ChessEvent::Draw; 2];
                }
            }
        }
        let terminal = next.finished.is_some();
        Transition {
            next_state: next,
            events,
            terminal,
        }
    }

    fn initial_state(&self, _rng: &mut dyn Rng) -> ChessState {
        let board = Board::default();
        let hashes = vec![board.hash()];
        ChessState {
            board,
            hashes,
            finished: None,
        }
    }

    fn truncation_horizon(&self) -> Option<usize> {
        self.max_ticks
    }
}

/// Minimal observation planes `(19, 8, 8)`, absolute orientation (H = rank, W = file, rank 1 at
/// row 0 — one convention for both players, matching the absolute action encoding):
///   0..6   White P N B R Q K   6..12  Black P N B R Q K
///   12     side to move (all-ones when White)
///   13..17 castling rights: W short, W long, B short, B long (all-ones planes)
///   17     en-passant file (one-hot column)
///   18     halfmove clock / 100
/// A history-carrying AlphaZero-style encoder can land later as a second `StateEncoder` — the
/// encoder seam is exactly where such views stay configurable.
pub struct ChessPlanesMinimal;

impl StateEncoder for ChessPlanesMinimal {
    type State = ChessState;

    fn encode(&self, state: &ChessState, _agent: usize) -> Vec<f32> {
        const PLANE: usize = 64;
        let mut obs = vec![0.0f32; 19 * PLANE];
        let board = &state.board;
        for (ci, color) in [Color::White, Color::Black].into_iter().enumerate() {
            for (pi, piece) in Piece::ALL.into_iter().enumerate() {
                let bb = board.pieces(piece) & board.colors(color);
                for sq in bb {
                    obs[(ci * 6 + pi) * PLANE + sq_index(sq)] = 1.0;
                }
            }
        }
        if board.side_to_move() == Color::White {
            obs[12 * PLANE..13 * PLANE].fill(1.0);
        }
        for (i, color) in [Color::White, Color::Black].into_iter().enumerate() {
            let rights = board.castle_rights(color);
            if rights.short.is_some() {
                obs[(13 + i * 2) * PLANE..(14 + i * 2) * PLANE].fill(1.0);
            }
            if rights.long.is_some() {
                obs[(14 + i * 2) * PLANE..(15 + i * 2) * PLANE].fill(1.0);
            }
        }
        if let Some(file) = board.en_passant() {
            let f = file as usize;
            for rank in 0..8 {
                obs[17 * PLANE + rank * 8 + f] = 1.0;
            }
        }
        let hm = f32::from(board.halfmove_clock()) / 100.0;
        obs[18 * PLANE..19 * PLANE].fill(hm);
        obs
    }

    fn obs_shape(&self) -> (usize, usize, usize) {
        (19, 8, 8)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct NoRng;
    impl Rng for NoRng {
        fn below(&mut self, _: usize) -> usize {
            0
        }
        fn unit(&mut self) -> f64 {
            0.0
        }
    }

    fn start() -> (Chess, ChessState) {
        let game = Chess::default();
        let state = game.initial_state(&mut NoRng);
        (game, state)
    }

    /// Perft through OUR wrapper (legal_actions -> decode -> step), so it validates the action-id
    /// bijection and the Game plumbing on top of cozy-chess's movegen.
    fn perft(game: &Chess, state: &ChessState, depth: usize) -> u64 {
        if depth == 0 {
            return 1;
        }
        let mover = state.turn();
        let mut nodes = 0;
        for action in game.legal_actions(state, mover) {
            let mut joint = vec![0usize; 2];
            joint[mover] = action;
            let t = game.step(state, &joint);
            // Terminal positions still count as one node at depth 1 (standard perft counts moves).
            nodes += if depth == 1 || t.terminal {
                if depth == 1 {
                    1
                } else {
                    0
                }
            } else {
                perft(game, &t.next_state, depth - 1)
            };
        }
        nodes
    }

    #[test]
    fn perft_matches_known_values() {
        let (game, state) = start();
        assert_eq!(perft(&game, &state, 1), 20);
        assert_eq!(perft(&game, &state, 2), 400);
        assert_eq!(perft(&game, &state, 3), 8_902);
        assert_eq!(perft(&game, &state, 4), 197_281);
    }

    fn play_uci(game: &Chess, state: ChessState, moves: &[&str]) -> (ChessState, bool) {
        let mut s = state;
        let mut terminal = false;
        for uci in moves {
            let mv: Move = uci.parse().unwrap();
            let action = encode_move(mv, s.board.side_to_move());
            let mover = s.turn();
            let mut joint = vec![0usize; 2];
            joint[mover] = action;
            let t = game.step(&s, &joint);
            s = t.next_state;
            terminal = t.terminal;
        }
        (s, terminal)
    }

    #[test]
    fn scholars_mate_is_a_white_win() {
        let (game, state) = start();
        let mover_events = |s: &ChessState| s.finished;
        let (s, terminal) = play_uci(
            &game,
            state,
            &["e2e4", "e7e5", "f1c4", "b8c6", "d1h5", "g8f6", "h5f7"],
        );
        assert!(terminal);
        assert_eq!(mover_events(&s), Some(ChessOutcome::WonBy(0)));
    }

    #[test]
    fn threefold_repetition_is_a_draw() {
        let (game, state) = start();
        // Shuffle knights: the start position recurs after every 4 plies; third recurrence draws.
        let shuffle = ["g1f3", "g8f6", "f3g1", "f6g8"];
        let seq: Vec<&str> = shuffle.iter().cycle().take(8).copied().collect();
        let (s, terminal) = play_uci(&game, state, &seq);
        assert!(
            terminal,
            "third occurrence of the start position should draw"
        );
        assert_eq!(s.finished, Some(ChessOutcome::Draw));
    }

    #[test]
    fn illegal_action_is_immediate_loss() {
        let (game, state) = start();
        let bogus = encode_move("e2e4".parse().unwrap(), Color::White) + 1; // e2 ray, wrong slot
        let t = game.step(&state, &[bogus, 0]);
        assert!(t.terminal);
        assert_eq!(t.events[0], ChessEvent::Loss);
        assert_eq!(t.events[1], ChessEvent::Win);
    }

    #[test]
    fn encode_decode_round_trips_all_legal_moves_along_a_game() {
        // Follow a game with promotions/castling flavor; at every position, every legal move must
        // round-trip through the 4672-id encoding.
        let (game, mut state) = start();
        let line = [
            "e2e4", "d7d5", "e4d5", "c7c6", "d5c6", "g8f6", "c6b7", "e7e6", "b7a8q",
        ];
        for uci in line {
            let side = state.board.side_to_move();
            let mut moves: Vec<Move> = Vec::new();
            state.board.generate_moves(|ms| {
                moves.extend(ms);
                false
            });
            for mv in moves {
                let action = encode_move(mv, side);
                assert!(action < CHESS_ACTIONS);
                let back = decode_move(action, &state.board).expect("decodable");
                assert_eq!(back, mv, "round-trip failed for {mv} (action {action})");
            }
            let (s, _) = play_uci(&game, state, &[uci]);
            state = s;
        }
        // The pawn promoted by ray-encoding: confirm a queen appeared for White on a8.
        assert!(state.board.pieces(Piece::Queen).has(Square::A8));
    }

    #[test]
    fn encoder_shape_and_start_position_sanity() {
        let (_, state) = start();
        let enc = ChessPlanesMinimal;
        let obs = enc.encode(&state, 0);
        assert_eq!(obs.len(), 19 * 64);
        let plane = |p: usize| &obs[p * 64..(p + 1) * 64];
        assert_eq!(plane(0).iter().sum::<f32>(), 8.0); // 8 white pawns
        assert_eq!(plane(6).iter().sum::<f32>(), 8.0); // 8 black pawns
        assert_eq!(plane(5).iter().sum::<f32>(), 1.0); // 1 white king
        assert_eq!(plane(12).iter().sum::<f32>(), 64.0); // white to move
        for p in 13..17 {
            assert_eq!(plane(p).iter().sum::<f32>(), 64.0); // all castling rights
        }
        assert_eq!(plane(17).iter().sum::<f32>(), 0.0); // no en passant
        assert_eq!(plane(18).iter().sum::<f32>(), 0.0); // halfmove clock 0
    }

    #[test]
    fn insufficient_material_draws() {
        let game = Chess::default();
        // K+B vs K: capture into bare-minor endgame must draw immediately.
        let board: Board = "8/8/8/3k4/8/2K1B3/8/8 w - - 0 1".parse().unwrap();
        let hashes = vec![board.hash()];
        let state = ChessState {
            board,
            hashes,
            finished: None,
        };
        assert!(insufficient_material(&state.board));
        // Any legal move keeps insufficient material -> the game should end in a draw on next step.
        let mover = state.turn();
        let action = game.legal_actions(&state, mover)[0];
        let mut joint = vec![0usize; 2];
        joint[mover] = action;
        let t = game.step(&state, &joint);
        assert!(t.terminal);
        assert_eq!(t.events[0], ChessEvent::Draw);
    }
}
