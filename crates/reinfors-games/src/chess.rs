//! Chess backed by `cozy-chess`, using the AlphaZero 8×8×73 action vocabulary.

use cozy_chess::{Board, Color, File, GameStatus, Move, Piece, Rank, Square};
use reinfors_core::{ActionView, Actor, Game, Reward, StateEncoder, Transition};

/// AlphaZero's 8×8×73 action map: `from_square * 73 + move_type`.
/// Move types are 0..56 ray direction × distance, 56..64 knight directions,
/// and 64..73 pawn direction × {knight, bishop, rook} underpromotions. Ray
/// directions are N, NE, E, SE, S, SW, W, NW; queen promotions use ray slots.
pub const CHESS_ACTIONS: usize = 64 * 73;

const RAY_DIRS: [(i8, i8); 8] = [
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
    (-1, 0),
    (-1, 1),
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
    sq as usize
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

pub fn encode_move(mv: Move, side: Color) -> usize {
    let from = sq_index(mv.from);
    let (ff, fr) = (mv.from.file() as i8, mv.from.rank() as i8);
    let (tf, tr) = (mv.to.file() as i8, mv.to.rank() as i8);
    let (df, dr) = (tf - ff, tr - fr);

    let move_type = match mv.promotion {
        Some(p) if p != Piece::Queen => {
            let fwd: i8 = if side == Color::White { 1 } else { -1 };
            debug_assert_eq!(dr, fwd);
            let dir = match df {
                0 => 0,
                -1 => 1,
                1 => 2,
                _ => unreachable!("pawn promotion with |file delta| > 1"),
            };
            let piece = UNDERPROMO_PIECES.iter().position(|&q| q == p).unwrap();
            64 + dir * 3 + piece
        }
        _ => {
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

pub fn move_to_uci(mv: Move, board: &Board) -> String {
    let stm = board.side_to_move();
    // cozy-chess represents castling as the king taking its own rook.
    let is_castle = (board.pieces(Piece::King) & board.colors(stm)).has(mv.from)
        && (board.pieces(Piece::Rook) & board.colors(stm)).has(mv.to);
    if is_castle {
        let to_file = if (mv.to.file() as i8) > (mv.from.file() as i8) {
            File::G
        } else {
            File::C
        };
        format!("{}{}", mv.from, Square::new(to_file, mv.from.rank()))
    } else {
        format!("{mv}")
    }
}

pub fn uci_to_action(uci: &str, board: &Board) -> Option<usize> {
    let mut found: Option<Move> = None;
    board.generate_moves(|ms| {
        for mv in ms {
            if move_to_uci(mv, board) == uci {
                found = Some(mv);
            }
        }
        false
    });
    found.map(|mv| encode_move(mv, board.side_to_move()))
}

pub fn decode_move(action: usize, board: &Board) -> Option<Move> {
    if action >= CHESS_ACTIONS {
        return None;
    }
    let from_idx = action / 73;
    let move_type = action % 73;
    let from = Square::new(File::index(from_idx % 8), Rank::index(from_idx / 8));
    let (ff, fr) = (from.file() as i8, from.rank() as i8);

    let (to, promotion) = if move_type < 56 {
        let (df, dr) = RAY_DIRS[move_type / 7];
        let dist = (move_type % 7 + 1) as i8;
        let to = sq_from(ff + df * dist, fr + dr * dist)?;
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
    // Geometric decoding does not imply legality in this position.
    Some(Move {
        from,
        to,
        promotion,
    })
}

mod board_serde {
    use cozy_chess::Board;
    use serde::de::Error as _;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S: Serializer>(board: &Board, ser: S) -> Result<S::Ok, S::Error> {
        format!("{board}").serialize(ser)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(de: D) -> Result<Board, D::Error> {
        String::deserialize(de)?
            .parse()
            .map_err(|_| D::Error::custom("invalid FEN"))
    }

    pub mod pairs {
        use super::*;

        pub fn serialize<S: Serializer>(v: &[(Board, u8)], ser: S) -> Result<S::Ok, S::Error> {
            let strs: Vec<(String, u8)> = v.iter().map(|(b, c)| (format!("{b}"), *c)).collect();
            strs.serialize(ser)
        }

        pub fn deserialize<'de, D: Deserializer<'de>>(de: D) -> Result<Vec<(Board, u8)>, D::Error> {
            Vec::<(String, u8)>::deserialize(de)?
                .into_iter()
                .map(|(s, c)| {
                    s.parse()
                        .map(|b| (b, c))
                        .map_err(|_| D::Error::custom("invalid FEN in history"))
                })
                .collect()
        }
    }
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ChessState {
    #[serde(with = "board_serde")]
    pub(crate) board: Board,
    // Repetition window since the last irreversible move.
    hashes: Vec<u64>,
    #[serde(with = "board_serde::pairs")]
    // Counts are stored at each historical moment, not recomputed from today's window.
    recent: Vec<(Board, u8)>,
    #[serde(skip)]
    finished: Option<ChessOutcome>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum ChessOutcome {
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

#[derive(Clone)]
pub struct Chess {
    pub max_ticks: Option<usize>,
    pub history_len: usize,
}

impl Default for Chess {
    fn default() -> Self {
        Chess {
            max_ticks: Some(512),
            history_len: 0,
        }
    }
}

fn agent_of(color: Color) -> usize {
    match color {
        Color::White => 0,
        Color::Black => 1,
    }
}

/// The material-only sufficient subset of FIDE §5.2.2: KNvKN and
/// opposite-coloured KBvKB admit helpmates, so deliberately play on.
fn insufficient_material(board: &Board) -> bool {
    if !(board.pieces(Piece::Pawn) | board.pieces(Piece::Rook) | board.pieces(Piece::Queen))
        .is_empty()
    {
        return false;
    }
    let knights = board.pieces(Piece::Knight);
    let bishops = board.pieces(Piece::Bishop);
    if bishops.is_empty() {
        return knights.len() <= 1;
    }
    if !knights.is_empty() {
        return false;
    }
    let mut square_colors = bishops
        .into_iter()
        .map(|sq| (sq.rank() as usize + sq.file() as usize) % 2);
    let first = square_colors.next().expect("non-empty bishops");
    square_colors.all(|c| c == first)
}

/// FIDE/OpenSpiel repetition identity: an en-passant square only distinguishes
/// positions when a capturing pawn threatens it.
fn repetition_hash(board: &Board) -> u64 {
    let Some(ep_file) = board.en_passant() else {
        return board.hash();
    };
    let stm = board.side_to_move();
    let ep_square = Square::new(ep_file, Rank::Sixth.relative_to(stm));
    let threats = cozy_chess::get_pawn_attacks(ep_square, !stm)
        & board.colors(stm)
        & board.pieces(Piece::Pawn);
    if !threats.is_empty() {
        return board.hash();
    }
    let mut b = cozy_chess::BoardBuilder::from_board(board);
    b.en_passant = None;
    match b.build() {
        Ok(cleared) => cleared.hash(),
        Err(_) => board.hash(),
    }
}

fn position_outcome(state: &ChessState) -> Option<ChessOutcome> {
    match state.board.status() {
        GameStatus::Won => Some(ChessOutcome::WonBy(1 - state.turn())),
        GameStatus::Drawn => Some(ChessOutcome::Draw),
        GameStatus::Ongoing => {
            if state.repetition_count() >= 3
                || insufficient_material(&state.board)
                || state.board.halfmove_clock() >= 100
            {
                Some(ChessOutcome::Draw)
            } else {
                None
            }
        }
    }
}

impl ChessState {
    fn repetition_count(&self) -> usize {
        let current = *self.hashes.last().expect("hash history is never empty");
        self.hashes.iter().filter(|&&h| h == current).count()
    }

    pub fn is_done(&self) -> bool {
        self.finished.is_some()
    }

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
        let mut events = vec![None, None];

        let decoded = decode_move(actions[mover], &state.board);
        let legal = decoded.is_some_and(|mv| next.board.try_play(mv).is_ok());
        if !legal {
            next.finished = Some(ChessOutcome::WonBy(other));
            events[mover] = Some(ChessEvent::Loss);
            events[other] = Some(ChessEvent::Win);
            return Transition {
                next_state: next,
                events,
                terminal: true,
            };
        }

        if next.board.halfmove_clock() == 0 {
            next.hashes.clear();
        }
        next.hashes.push(repetition_hash(&next.board));
        if self.history_len > 0 {
            let count = next.repetition_count().min(u8::MAX as usize) as u8;
            next.recent.push((next.board.clone(), count));
            if next.recent.len() > self.history_len {
                next.recent.remove(0);
            }
        }

        if let Some(outcome) = position_outcome(&next) {
            next.finished = Some(outcome);
            match outcome {
                ChessOutcome::WonBy(w) => {
                    events[w] = Some(ChessEvent::Win);
                    events[1 - w] = Some(ChessEvent::Loss);
                }
                ChessOutcome::Draw => {
                    events = vec![Some(ChessEvent::Draw); 2];
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

    fn initial_state(&self) -> ChessState {
        let board = Board::default();
        let hashes = vec![repetition_hash(&board)];
        let recent = if self.history_len > 0 {
            vec![(board.clone(), 1)]
        } else {
            Vec::new()
        };
        ChessState {
            board,
            hashes,
            recent,
            finished: None,
        }
    }

    fn truncation_horizon(&self) -> Option<usize> {
        self.max_ticks
    }
}

pub struct ChessPlanesMinimal;

impl ActionView for ChessPlanesMinimal {}

impl StateEncoder for ChessPlanesMinimal {
    type State = ChessState;

    fn encode(&self, state: &ChessState, _agent: usize) -> Vec<f32> {
        const PLANE: usize = 64;
        let mut obs = vec![0.0f32; 19 * PLANE];
        let board = &state.board;
        fill_piece_planes(&mut obs, 0, board, [Color::White, Color::Black], sq_index);
        if board.side_to_move() == Color::White {
            obs[12 * PLANE..13 * PLANE].fill(1.0);
        }
        fill_castling_planes(&mut obs, 13, board, [Color::White, Color::Black]);
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

// ChessPlanesOpenSpiel must NOT use these helpers: its layout is pinned to their
// chess.cc (piece-major interleave, long-before-short castling).
fn fill_piece_planes(
    obs: &mut [f32],
    base: usize,
    board: &Board,
    colors: [Color; 2],
    at: impl Fn(Square) -> usize,
) {
    const PLANE: usize = 64;
    for (ci, color) in colors.into_iter().enumerate() {
        for (pi, piece) in Piece::ALL.into_iter().enumerate() {
            let bb = board.pieces(piece) & board.colors(color);
            for sq in bb {
                obs[base + (ci * 6 + pi) * PLANE + at(sq)] = 1.0;
            }
        }
    }
}

fn fill_castling_planes(obs: &mut [f32], base_plane: usize, board: &Board, colors: [Color; 2]) {
    const PLANE: usize = 64;
    for (i, color) in colors.into_iter().enumerate() {
        let rights = board.castle_rights(color);
        if rights.short.is_some() {
            obs[(base_plane + i * 2) * PLANE..(base_plane + 1 + i * 2) * PLANE].fill(1.0);
        }
        if rights.long.is_some() {
            obs[(base_plane + 1 + i * 2) * PLANE..(base_plane + 2 + i * 2) * PLANE].fill(1.0);
        }
    }
}

// Rank reflection maps N↔S, NE↔SE and NW↔SW; E/W remain fixed.
const RAY_REFLECT: [usize; 8] = [4, 3, 2, 1, 0, 7, 6, 5];
const KNIGHT_REFLECT: [usize; 8] = [3, 2, 1, 0, 7, 6, 5, 4];

pub(crate) fn reflect_action(action: usize) -> usize {
    // This sigma must remain coherent with ChessPlanesRelative's observation
    // reflection. Underpromotion slots are fixed because forward is implicit.
    let from = action / 73;
    let mt = action % 73;
    let from_r = (7 - from / 8) * 8 + from % 8;
    let mt_r = if mt < 56 {
        RAY_REFLECT[mt / 7] * 7 + mt % 7
    } else if mt < 64 {
        56 + KNIGHT_REFLECT[mt - 56]
    } else {
        mt
    };
    from_r * 73 + mt_r
}

pub struct ChessPlanesRelative;

impl ActionView for ChessPlanesRelative {
    fn head_index(&self, action: usize, agent: usize) -> usize {
        if agent == 1 {
            reflect_action(action)
        } else {
            action
        }
    }

    fn game_action(&self, head: usize, agent: usize) -> usize {
        if agent == 1 {
            reflect_action(head)
        } else {
            head
        }
    }
}

impl StateEncoder for ChessPlanesRelative {
    type State = ChessState;

    fn encode(&self, state: &ChessState, agent: usize) -> Vec<f32> {
        const PLANE: usize = 64;
        let mut obs = vec![0.0f32; 19 * PLANE];
        let board = &state.board;
        let persp = if agent == 0 {
            Color::White
        } else {
            Color::Black
        };
        let at = |sq: Square| -> usize {
            let i = sq_index(sq);
            if persp == Color::Black {
                (7 - i / 8) * 8 + i % 8
            } else {
                i
            }
        };
        fill_piece_planes(&mut obs, 0, board, [persp, !persp], at);
        if board.side_to_move() == persp {
            obs[12 * PLANE..13 * PLANE].fill(1.0);
        }
        fill_castling_planes(&mut obs, 13, board, [persp, !persp]);
        if let Some(file) = board.en_passant() {
            // Sigma reflects ranks only, so the en-passant file is unchanged.
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

/// OpenSpiel's `(20, 8, 8)` layout, pinned to
/// `chess.cc::ObservationTensor` at commit `d15d49f8`:
///
/// - 0..12: K/Q/R/B/N/P occupancy, White then Black per piece type
/// - 12: empty squares; 13: `(repetition - 1) / 2`
/// - 14: side to move (`ColorToPlayer`, ones for White); 15: halfmove / 101
/// - 16..20: White long/short, then Black long/short castling rights
pub struct ChessPlanesOpenSpiel;

impl ActionView for ChessPlanesOpenSpiel {}

impl StateEncoder for ChessPlanesOpenSpiel {
    type State = ChessState;

    fn encode(&self, state: &ChessState, _agent: usize) -> Vec<f32> {
        const PLANE: usize = 64;
        const TYPES: [Piece; 6] = [
            Piece::King,
            Piece::Queen,
            Piece::Rook,
            Piece::Bishop,
            Piece::Knight,
            Piece::Pawn,
        ];
        let mut obs = vec![0.0f32; 20 * PLANE];
        let board = &state.board;
        for (ti, piece) in TYPES.into_iter().enumerate() {
            for (ci, color) in [Color::White, Color::Black].into_iter().enumerate() {
                let bb = board.pieces(piece) & board.colors(color);
                for sq in bb {
                    obs[(ti * 2 + ci) * PLANE + sq_index(sq)] = 1.0;
                }
            }
        }
        let occupied = board.occupied();
        for i in 0..PLANE {
            if !occupied.has(Square::index(i)) {
                obs[12 * PLANE + i] = 1.0;
            }
        }
        let rep = (state.repetition_count() as f32 - 1.0) / 2.0;
        obs[13 * PLANE..14 * PLANE].fill(rep);
        if board.side_to_move() == Color::White {
            obs[14 * PLANE..15 * PLANE].fill(1.0);
        }
        obs[15 * PLANE..16 * PLANE].fill(f32::from(board.halfmove_clock()) / 101.0);
        for (i, color) in [Color::White, Color::Black].into_iter().enumerate() {
            let rights = board.castle_rights(color);
            if rights.long.is_some() {
                obs[(16 + i * 2) * PLANE..(17 + i * 2) * PLANE].fill(1.0);
            }
            if rights.short.is_some() {
                obs[(17 + i * 2) * PLANE..(18 + i * 2) * PLANE].fill(1.0);
            }
        }
        obs
    }

    fn obs_shape(&self) -> (usize, usize, usize) {
        (20, 8, 8)
    }
}

/// AlphaZero-style history with a deliberately absolute frame: unlike the
/// paper, boards are not flipped; a side-to-move plane is supplied instead.
/// History is newest-first. This layout and the absolute action map form a
/// checkpoint contract and must not be changed independently.
pub struct ChessPlanesAz119 {
    pub history: usize,
}

impl Default for ChessPlanesAz119 {
    fn default() -> Self {
        ChessPlanesAz119 { history: 8 }
    }
}

impl ChessPlanesAz119 {
    fn step_planes(obs: &mut [f32], base: usize, board: &Board, count: usize) {
        const PLANE: usize = 64;
        fill_piece_planes(obs, base, board, [Color::White, Color::Black], sq_index);
        if count >= 2 {
            obs[base + 12 * PLANE..base + 13 * PLANE].fill(1.0);
        }
        if count >= 3 {
            obs[base + 13 * PLANE..base + 14 * PLANE].fill(1.0);
        }
    }
}

impl ActionView for ChessPlanesAz119 {}

impl StateEncoder for ChessPlanesAz119 {
    type State = ChessState;

    fn encode(&self, state: &ChessState, _agent: usize) -> Vec<f32> {
        const PLANE: usize = 64;
        let aux = 14 * self.history;
        let mut obs = vec![0.0f32; (aux + 7) * PLANE];
        let board = &state.board;
        if state.recent.is_empty() {
            Self::step_planes(&mut obs, 0, board, state.repetition_count());
        } else {
            for (t, (past, count)) in state.recent.iter().rev().take(self.history).enumerate() {
                Self::step_planes(&mut obs, t * 14 * PLANE, past, *count as usize);
            }
        }
        if board.side_to_move() == Color::White {
            obs[aux * PLANE..(aux + 1) * PLANE].fill(1.0);
        }
        let fullmove = f32::from(board.fullmove_number()) / 100.0;
        obs[(aux + 1) * PLANE..(aux + 2) * PLANE].fill(fullmove);
        fill_castling_planes(&mut obs, aux + 2, board, [Color::White, Color::Black]);
        let hm = f32::from(board.halfmove_clock()) / 100.0;
        obs[(aux + 6) * PLANE..(aux + 7) * PLANE].fill(hm);
        obs
    }

    fn obs_shape(&self) -> (usize, usize, usize) {
        (14 * self.history + 7, 8, 8)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn start() -> (Chess, ChessState) {
        let game = Chess::default();
        let state = game.initial_state();
        (game, state)
    }

    fn start_fen(fen: &str) -> (Chess, ChessState) {
        let board: Board = fen.parse().unwrap();
        let hashes = vec![repetition_hash(&board)];
        let state = ChessState {
            board,
            hashes,
            recent: Vec::new(),
            finished: None,
        };
        (Chess::default(), state)
    }

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
    fn fifty_move_rule_draws_at_100_halfmoves() {
        let game = Chess::default();
        // KR vs K (sufficient material), one reversible move short of 100 half-moves
        let board: Board = "8/8/8/3k4/8/3K4/8/6R1 w - - 99 60".parse().unwrap();
        let hashes = vec![repetition_hash(&board)];
        let state = ChessState {
            board,
            hashes,
            recent: Vec::new(),
            finished: None,
        };
        assert_eq!(
            position_outcome(&state),
            None,
            "99 half-moves is not yet a draw"
        );
        let (s, terminal) = play_uci(&game, state, &["g1g2"]);
        assert!(terminal, "the 100th reversible half-move should draw");
        assert_eq!(s.finished, Some(ChessOutcome::Draw));
    }

    #[test]
    fn noncapturable_ep_position_counts_toward_repetition() {
        // game-43 shape: a double push nobody can capture, then a 4-ply king shuffle;
        // the post-push position recurs at plies +4 and +8 = threefold under FIDE
        let (game, state) = start_fen("7k/4p3/8/8/8/8/8/7K b - - 0 1");
        let shuffle = ["h1g1", "h8g8", "g1h1", "g8h8"];
        let mut seq = vec!["e7e5"];
        seq.extend(shuffle.iter().cycle().take(7).copied());
        let (_, terminal) = play_uci(&game, state.clone(), &seq);
        assert!(!terminal, "two occurrences is not yet a draw");
        let mut seq = vec!["e7e5"];
        seq.extend(shuffle.iter().cycle().take(8).copied());
        let (s, terminal) = play_uci(&game, state, &seq);
        assert!(
            terminal,
            "third occurrence of the post-push position should draw"
        );
        assert_eq!(s.finished, Some(ChessOutcome::Draw));
    }

    #[test]
    fn capturable_ep_position_stays_distinct() {
        // with a white pawn on f5 the push is ep-capturable, so the post-push position
        // is NOT identical to the later shuffles; no draw at the +8 ply
        let (game, state) = start_fen("7k/4p3/8/5P2/8/8/8/7K b - - 0 1");
        let shuffle = ["h1g1", "h8g8", "g1h1", "g8h8"];
        let mut seq = vec!["e7e5"];
        seq.extend(shuffle.iter().cycle().take(8).copied());
        let (_, terminal) = play_uci(&game, state, &seq);
        assert!(
            !terminal,
            "the ep-capturable first occurrence must not count"
        );
    }

    #[test]
    fn illegal_action_is_immediate_loss() {
        let (game, state) = start();
        let bogus = encode_move("e2e4".parse().unwrap(), Color::White) + 1;
        let t = game.step(&state, &[bogus, 0]);
        assert!(t.terminal);
        assert_eq!(t.events[0], Some(ChessEvent::Loss));
        assert_eq!(t.events[1], Some(ChessEvent::Win));
    }

    #[test]
    fn encode_decode_round_trips_all_legal_moves_along_a_game() {
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
        assert!(state.board.pieces(Piece::Queen).has(Square::A8));
    }

    #[test]
    fn encoder_shape_and_start_position_sanity() {
        let (_, state) = start();
        let enc = ChessPlanesMinimal;
        let obs = enc.encode(&state, 0);
        assert_eq!(obs.len(), 19 * 64);
        let plane = |p: usize| &obs[p * 64..(p + 1) * 64];
        assert_eq!(plane(0).iter().sum::<f32>(), 8.0);
        assert_eq!(plane(6).iter().sum::<f32>(), 8.0);
        assert_eq!(plane(5).iter().sum::<f32>(), 1.0);
        assert_eq!(plane(12).iter().sum::<f32>(), 64.0);
        for p in 13..17 {
            assert_eq!(plane(p).iter().sum::<f32>(), 64.0);
        }
        assert_eq!(plane(17).iter().sum::<f32>(), 0.0);
        assert_eq!(plane(18).iter().sum::<f32>(), 0.0);
    }

    #[test]
    fn insufficient_material_draws() {
        let game = Chess::default();
        let board: Board = "8/8/8/3k4/8/2K1B3/8/8 w - - 0 1".parse().unwrap();
        let hashes = vec![repetition_hash(&board)];
        let state = ChessState {
            board,
            hashes,
            recent: Vec::new(),
            finished: None,
        };
        assert!(insufficient_material(&state.board));
        let mover = state.turn();
        let action = game.legal_actions(&state, mover)[0];
        let mut joint = vec![0usize; 2];
        joint[mover] = action;
        let t = game.step(&state, &joint);
        assert!(t.terminal);
        assert_eq!(t.events[0], Some(ChessEvent::Draw));
    }
}

#[cfg(test)]
mod az119_tests {
    use super::*;

    fn hist_game() -> (Chess, ChessState) {
        let game = Chess {
            max_ticks: Some(512),
            history_len: 8,
        };
        let state = game.initial_state();
        (game, state)
    }

    fn play(game: &Chess, state: ChessState, moves: &[&str]) -> ChessState {
        let mut s = state;
        for uci in moves {
            let mv: Move = uci.parse().unwrap();
            let action = encode_move(mv, s.board.side_to_move());
            let mover = s.turn();
            let mut joint = vec![0usize; 2];
            joint[mover] = action;
            s = game.step(&s, &joint).next_state;
        }
        s
    }

    const PLANE: usize = 64;

    #[test]
    fn shape_and_current_step_matches_minimal_pieces() {
        let (_, state) = hist_game();
        let az = ChessPlanesAz119::default().encode(&state, 0);
        assert_eq!(az.len(), 119 * PLANE);
        let minimal = ChessPlanesMinimal.encode(&state, 0);
        assert_eq!(&az[..12 * PLANE], &minimal[..12 * PLANE]);
    }

    #[test]
    fn history_step_one_is_the_previous_position() {
        let (game, state) = hist_game();
        let before = ChessPlanesAz119::default().encode(&state, 0);
        let after = ChessPlanesAz119::default().encode(&play(&game, state, &["e2e4"]), 0);
        assert_eq!(&after[14 * PLANE..26 * PLANE], &before[..12 * PLANE]);
        assert_ne!(&after[..12 * PLANE], &before[..12 * PLANE]);
    }

    #[test]
    fn repetition_planes_activate_exactly_when_positions_recur() {
        let (game, state) = hist_game();
        let shuffle = ["g1f3", "g8f6", "f3g1", "f6g8"];
        let s4 = play(&game, state, &shuffle);
        let obs4 = ChessPlanesAz119::default().encode(&s4, 0);
        assert!(obs4[12 * PLANE..13 * PLANE].iter().all(|&v| v == 1.0));
        assert!(obs4[13 * PLANE..14 * PLANE].iter().all(|&v| v == 0.0));
        assert!(obs4[(14 + 12) * PLANE..(14 + 13) * PLANE]
            .iter()
            .all(|&v| v == 0.0));
        let t4 = 4 * 14 * PLANE;
        assert!(obs4[t4 + 12 * PLANE..t4 + 13 * PLANE]
            .iter()
            .all(|&v| v == 0.0));
    }

    #[test]
    fn aux_planes() {
        let (game, state) = hist_game();
        let obs = ChessPlanesAz119::default().encode(&state, 0);
        assert!(obs[112 * PLANE..113 * PLANE].iter().all(|&v| v == 1.0));
        assert!((obs[113 * PLANE] - 0.01).abs() < 1e-6);
        for p in 114..118 {
            assert!(obs[p * PLANE..(p + 1) * PLANE].iter().all(|&v| v == 1.0));
        }
        let s = play(&game, state, &["e2e4"]);
        let obs = ChessPlanesAz119::default().encode(&s, 0);
        assert!(obs[112 * PLANE..113 * PLANE].iter().all(|&v| v == 0.0));
    }

    #[test]
    fn short_history_shapes_and_truncates() {
        let game = Chess {
            max_ticks: Some(512),
            history_len: 2,
        };
        let enc = ChessPlanesAz119 { history: 2 };
        assert_eq!(enc.obs_shape(), (35, 8, 8));
        let state = game.initial_state();
        let s = play(&game, state, &["e2e4", "e7e5", "g1f3"]);
        let obs = enc.encode(&s, 0);
        assert_eq!(obs.len(), 35 * PLANE);
        assert!(obs[..12 * PLANE].contains(&1.0));
        assert!(obs[14 * PLANE..26 * PLANE].contains(&1.0));
        assert!(obs[28 * PLANE..29 * PLANE].iter().all(|&v| v == 0.0));
    }

    #[test]
    fn history_off_populates_only_the_current_step() {
        let game = Chess::default();
        let state = game.initial_state();
        let s = play(&game, state, &["e2e4", "e7e5"]);
        let obs = ChessPlanesAz119::default().encode(&s, 0);
        assert!(obs[..12 * PLANE].contains(&1.0));
        assert!(obs[14 * PLANE..112 * PLANE].iter().all(|&v| v == 0.0));
    }
}

#[cfg(test)]
mod relative_frame_tests {
    use super::*;
    use reinfors_core::check_action_view;

    fn sigma_fen(fen: &str) -> String {
        let parts: Vec<&str> = fen.split(' ').collect();
        let placement: Vec<String> = parts[0]
            .split('/')
            .rev()
            .map(|row| {
                row.chars()
                    .map(|c| {
                        if c.is_ascii_uppercase() {
                            c.to_ascii_lowercase()
                        } else if c.is_ascii_lowercase() {
                            c.to_ascii_uppercase()
                        } else {
                            c
                        }
                    })
                    .collect()
            })
            .collect();
        let side = if parts[1] == "w" { "b" } else { "w" };
        let castling = if parts[2] == "-" {
            "-".to_string()
        } else {
            let mut cs: Vec<char> = parts[2]
                .chars()
                .map(|c| {
                    if c.is_ascii_uppercase() {
                        c.to_ascii_lowercase()
                    } else {
                        c.to_ascii_uppercase()
                    }
                })
                .collect();
            cs.sort_by_key(|c| "KQkq".find(*c).unwrap());
            cs.into_iter().collect()
        };
        let ep = if parts[3] == "-" {
            "-".to_string()
        } else {
            parts[3]
                .chars()
                .map(|c| match c {
                    '3' => '6',
                    '6' => '3',
                    r => r,
                })
                .collect()
        };
        format!(
            "{} {} {} {} {} {}",
            placement.join("/"),
            side,
            castling,
            ep,
            parts[4],
            parts[5]
        )
    }

    fn state_of(board: Board) -> ChessState {
        ChessState {
            board,
            hashes: Vec::new(),
            recent: Vec::new(),
            finished: None,
        }
    }

    fn reflect_sq(sq: Square) -> Square {
        Square::new(sq.file(), Rank::index(7 - sq.rank() as usize))
    }

    #[test]
    fn action_view_contract_holds_over_the_full_space() {
        check_action_view(&ChessPlanesRelative, CHESS_ACTIONS, 2);
    }

    #[test]
    fn obs_and_actions_transform_together_under_sigma() {
        let enc = ChessPlanesRelative;
        let mut board = Board::default();
        for ply in 0..60 {
            let s = state_of(board.clone());
            let mover = usize::from(s.board.side_to_move() == Color::Black);
            let t = state_of(
                Board::from_fen(&sigma_fen(&format!("{}", s.board)), false)
                    .expect("sigma of a legal position is legal"),
            );
            assert_eq!(
                enc.encode(&s, mover),
                enc.encode(&t, 1 - mover),
                "obs equivariance broke at ply {ply}"
            );
            let mut moves = Vec::new();
            board.generate_moves(|ms| {
                moves.extend(ms);
                false
            });
            if moves.is_empty() {
                break;
            }
            for &mv in &moves {
                let id = encode_move(mv, s.board.side_to_move());
                let smv = Move {
                    from: reflect_sq(mv.from),
                    to: reflect_sq(mv.to),
                    promotion: mv.promotion,
                };
                let sid = encode_move(smv, t.board.side_to_move());
                assert_eq!(
                    enc.head_index(id, mover),
                    enc.head_index(sid, 1 - mover),
                    "action coherence broke at ply {ply} for {mv}"
                );
            }
            let mv = moves[(ply * 7) % moves.len()];
            board.play(mv);
        }
    }

    #[test]
    fn start_position_is_sigma_symmetric_except_the_turn_plane() {
        let enc = ChessPlanesRelative;
        let s = state_of(Board::default());
        let (w, b) = (enc.encode(&s, 0), enc.encode(&s, 1));
        for (i, (x, y)) in w.iter().zip(&b).enumerate() {
            if (12 * 64..13 * 64).contains(&i) {
                continue;
            }
            assert_eq!(x, y, "plane mismatch at flat index {i}");
        }
        assert_eq!(w[12 * 64], 1.0, "White to move: agent 0 sees my-turn");
        assert_eq!(b[12 * 64], 0.0, "White to move: agent 1 sees not-my-turn");
    }
}

#[cfg(test)]
mod openspiel_obs_tests {
    use super::*;

    #[test]
    fn start_position_matches_the_documented_layout() {
        let game = Chess {
            max_ticks: None,
            history_len: 0,
        };
        let s = { game.initial_state() };
        let obs = ChessPlanesOpenSpiel.encode(&s, 0);
        let plane = |p: usize| obs[p * 64..(p + 1) * 64].iter().sum::<f32>();
        for (p, n) in [(0, 1.0), (1, 1.0), (2, 1.0), (3, 1.0), (4, 2.0), (5, 2.0)] {
            assert_eq!(plane(p), n, "kings/queens/rooks plane {p}");
        }
        assert_eq!(plane(10), 8.0);
        assert_eq!(plane(11), 8.0);
        assert_eq!(plane(12), 32.0);
        assert_eq!(plane(13), 0.0);
        assert_eq!(plane(14), 64.0);
        assert_eq!(plane(15), 0.0);
        for p in 16..20 {
            assert_eq!(plane(p), 64.0, "castling plane {p}");
        }
        assert_eq!(obs[4 * 64], 1.0);
    }
}

#[cfg(test)]
mod fide_dead_position_tests {
    use super::*;

    fn dead(fen: &str) -> bool {
        insufficient_material(&fen.parse::<Board>().unwrap())
    }

    #[test]
    fn material_boundary_matches_fide() {
        assert!(dead("8/8/4k3/8/8/3K4/8/8 w - - 0 1"));
        assert!(dead("8/8/4k3/8/8/3KB3/8/8 w - - 0 1"));
        assert!(dead("8/8/4kn2/8/8/3K4/8/8 w - - 0 1"));
        assert!(dead("8/8/3bk3/8/8/3KB3/8/8 w - - 0 1"));
        assert!(dead("8/8/4k3/8/8/3KB3/8/2B5 w - - 0 1"));
        assert!(!dead("8/8/2bk4/8/8/3KB3/8/8 w - - 0 1"));
        assert!(!dead("8/8/3nk3/8/8/2NK4/8/8 w - - 0 1"));
        assert!(!dead("8/8/4k3/8/8/2NKN3/8/8 w - - 0 1"));
        assert!(!dead("8/8/4k3/8/8/2NKB3/8/8 w - - 0 1"));
        assert!(!dead("8/8/4k3/7p/8/3K4/8/8 w - - 0 1"));
    }
}

#[cfg(test)]
mod uci_tests {
    use super::*;

    #[test]
    fn castling_renders_standard_and_round_trips() {
        let board: Board = "r3k2r/pppqpppp/8/8/8/8/PPPQPPPP/R3K2R w KQkq - 0 1"
            .parse()
            .unwrap();
        for uci in ["e1g1", "e1c1"] {
            let id = uci_to_action(uci, &board).expect(uci);
            let mv = decode_move(id, &board).unwrap();
            assert_eq!(move_to_uci(mv, &board), uci);
        }
        assert!(uci_to_action("e1h1", &board).is_none());
    }

    #[test]
    fn every_legal_move_round_trips_via_uci() {
        let mut board = Board::default();
        for ply in 0..40 {
            let mut moves = Vec::new();
            board.generate_moves(|ms| {
                moves.extend(ms);
                false
            });
            if moves.is_empty() {
                break;
            }
            for &mv in &moves {
                let uci = move_to_uci(mv, &board);
                assert_eq!(
                    uci_to_action(&uci, &board),
                    Some(encode_move(mv, board.side_to_move())),
                    "ply {ply} move {uci}"
                );
            }
            board.play(moves[(ply * 5) % moves.len()]);
        }
    }
}

impl reinfors_core::StateCodec for Chess {
    type State = ChessState;

    fn encode(&self, s: &ChessState) -> Vec<u8> {
        crate::codec_util::serde_encode(2, s)
    }

    fn decode(&self, bytes: &[u8]) -> Result<ChessState, String> {
        let mut s: ChessState = crate::codec_util::serde_decode(2, bytes)?;
        if s.hashes.is_empty() {
            return Err("empty hash history (must contain at least the current position)".into());
        }
        s.finished = position_outcome(&s);
        Ok(s)
    }

    fn validate_decoded_state(&self, state: &ChessState, done: bool) -> Result<(), String> {
        if state.hashes.is_empty() {
            return Err("empty hash history (must contain at least the current position)".into());
        }
        if state.hashes.len() > 100_000 {
            return Err(format!(
                "implausible hash history length {}",
                state.hashes.len()
            ));
        }
        if *state.hashes.last().expect("non-empty") != repetition_hash(&state.board) {
            return Err("hash history does not end at the current position".into());
        }
        if state.recent.len() > 8 {
            return Err(format!(
                "recent-history length {} exceeds the 8-slot ring",
                state.recent.len()
            ));
        }
        if state.finished.is_some() != done {
            return Err(format!(
                "derived outcome {:?} disagrees with envelope done {done}",
                state.finished
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod codec_tests {
    use super::*;
    use reinfors_core::StateCodec;

    fn played() -> (Chess, ChessState) {
        let game = Chess {
            max_ticks: None,
            history_len: 8,
        };
        let mut s = game.initial_state();
        for uci in ["e2e4", "e7e5", "g1f3", "b8c6"] {
            let mv: Move = uci.parse().unwrap();
            let a = encode_move(mv, s.board.side_to_move());
            let mover = s.turn();
            let mut joint = vec![0usize; 2];
            joint[mover] = a;
            s = game.step(&s, &joint).next_state;
        }
        (game, s)
    }

    #[test]
    fn round_trips_canonically_and_validates() {
        let (game, s) = played();
        let bytes = game.encode(&s);
        let back = game
            .decode(&bytes)
            .unwrap_or_else(|e| panic!("decode failed: {e}"));
        assert_eq!(game.encode(&back), bytes);
        game.validate_decoded_state(&back, false).unwrap();
        assert!(game.decode(&bytes[..bytes.len() - 1]).is_err());
        assert!(game.decode(&[1, 0, 0]).unwrap_err().contains("version"));
    }

    #[test]
    fn safety_invariants_reject_tampered_structs() {
        let (game, s) = played();
        let mut no_tail = s.clone();
        no_tail.hashes.pop();
        assert!(game
            .validate_decoded_state(&no_tail, false)
            .unwrap_err()
            .contains("hash history"));
        let mut fat_ring = s.clone();
        fat_ring.recent = vec![(s.board.clone(), 1); 9];
        assert!(game
            .validate_decoded_state(&fat_ring, false)
            .unwrap_err()
            .contains("8-slot"));
        assert!(game
            .validate_decoded_state(&s, true)
            .unwrap_err()
            .contains("disagrees"));
    }

    #[test]
    fn outcome_is_recomputed_at_decode() {
        let game = Chess {
            max_ticks: None,
            history_len: 0,
        };
        let mut s = game.initial_state();
        for uci in ["f2f3", "e7e5", "g2g4", "d8h4"] {
            let mv: Move = uci.parse().unwrap();
            let a = encode_move(mv, s.board.side_to_move());
            let mover = s.turn();
            let mut joint = vec![0usize; 2];
            joint[mover] = a;
            s = game.step(&s, &joint).next_state;
        }
        assert!(s.is_done());
        let back = game.decode(&game.encode(&s)).unwrap();
        assert!(back.is_done(), "mate must be rediscovered at decode");
        game.validate_decoded_state(&back, true).unwrap();
    }
}
