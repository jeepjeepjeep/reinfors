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
//! moves), and an insufficient-material draw: bare kings, a lone knight, or any number of
//! bishops all on same-colored squares. This is the MATERIAL-ONLY sufficient condition of FIDE's
//! dead-position rule (§5.2.2) — the standard practical subset (OpenSpiel and python-chess score
//! identically). Full dead-position detection is semantic ("no series of legal moves can ever
//! mate", including locked pawn fortresses) and is a proof-search problem no engine-adjacent
//! implementation attempts; such positions cannot be won and fall through to the fifty-move or
//! repetition draw — the same result FIDE reaches, later. Helpmate-admitting material (KNvKN,
//! opposite-colored KBvKB) plays on.
//! An illegal action id (impossible through the masked searches; reachable through a raw `Env`)
//! is an immediate loss for the mover — the same posture as connect4's full-column rule.

use cozy_chess::{Board, Color, File, GameStatus, Move, Piece, Rank, Square};
use reinfors_core::{ActionView, Actor, Game, Reward, Rng, StateEncoder, Transition};

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

/// Standard-UCI rendering of a cozy move ("e2e4", "e7e8q"). Cozy represents castling as
/// king-takes-own-rook ("e1h1"); standard UCI wants the king's two-square destination ("e1g1"),
/// so castling is detected against `board` and re-rendered.
pub fn move_to_uci(mv: Move, board: &Board) -> String {
    let stm = board.side_to_move();
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

/// The action id of the legal move whose standard-UCI rendering is `uci`, in `board`'s position.
/// Matching against the LEGAL move list (rather than parsing) disambiguates castling from plain
/// king moves and pins promotions exactly. `None` if no legal move renders as `uci`.
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
    /// The last up-to-8 positions (newest last) with each position's repetition count AS OF that
    /// moment (computed against the then-current threefold window, so history planes are exact even
    /// across irreversible-move resets). Maintained only when the game was built with
    /// a nonzero `history_len` (the AZ-119 encoder's input); empty otherwise, so the minimal-encoder path
    /// pays nothing for it in tree-node clones.
    recent: Vec<(Board, u8)>,
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
    /// Number of recent positions to maintain in the state (the [`ChessPlanesAz119`] encoder's
    /// input). 0 (the default) keeps none — the factory sets this to the selected encoder's
    /// `history` exactly when that encoder is chosen, so the two can't drift apart.
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

/// The material-only sufficient condition of FIDE's dead-position rule (OpenSpiel-aligned; see
/// the module doc for the semantic cases this deliberately does not attempt): with no
/// pawns/rooks/queens, a draw iff kings only, one lone knight, or any number of bishops all on
/// same-colored squares. KNvKN and opposite-colored KBvKB admit helpmates, so they play on.
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
        if self.history_len > 0 {
            let count = next.repetition_count().min(u8::MAX as usize) as u8;
            next.recent.push((next.board.clone(), count));
            if next.recent.len() > self.history_len {
                next.recent.remove(0);
            }
        }

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

impl ActionView for ChessPlanesMinimal {} // absolute: identity action view

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

/// Ray-direction index after rank reflection (`dr -> -dr`): N<->S, NE<->SE, NW<->SW, E/W fixed.
const RAY_REFLECT: [usize; 8] = [4, 3, 2, 1, 0, 7, 6, 5];
/// Knight-direction index after rank reflection.
const KNIGHT_REFLECT: [usize; 8] = [3, 2, 1, 0, 7, 6, 5, 4];

/// The role symmetry sigma on an 8x8x73 action id: reflect the from-square's rank and negate every
/// rank delta. Underpromotion slots encode file direction + piece only ("forward" is
/// side-implicit), so they are fixed points. An involution: applying it twice is the identity.
pub(crate) fn reflect_action(action: usize) -> usize {
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

/// Mover-relative observation planes `(19, 8, 8)`: the position seen from `agent`'s side — for
/// agent 1 (Black) the board is rank-reflected and colors swap roles, so both colors present the
/// net the same "my pieces advance up the board" geometry (role equivariance as an inductive
/// bias — the AlphaZero paper's convention, vs [`ChessPlanesMinimal`]'s absolute frame). Layout
/// mirrors the minimal encoder with my/opp planes in place of White/Black:
///   0..6   my P N B R Q K   6..12  opponent P N B R Q K   (ranks reflected for agent 1)
///   12     my-turn plane (all-ones when `agent` is to move)
///   13..17 castling rights: my short, my long, opp short, opp long
///   17     en-passant file (one-hot column; files are fixed under rank reflection)
///   18     halfmove clock / 100
/// The paired `ActionView` applies the SAME sigma to action indexing (`reflect_action` for
/// agent 1), so observations and the policy/Q head transform together — the seam's coherence
/// contract, pinned by the equivariance tests below.
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
        // sigma is an involution: the map is its own inverse.
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
        for (ci, color) in [persp, !persp].into_iter().enumerate() {
            for (pi, piece) in Piece::ALL.into_iter().enumerate() {
                let bb = board.pieces(piece) & board.colors(color);
                for sq in bb {
                    obs[(ci * 6 + pi) * PLANE + at(sq)] = 1.0;
                }
            }
        }
        if board.side_to_move() == persp {
            obs[12 * PLANE..13 * PLANE].fill(1.0);
        }
        for (i, color) in [persp, !persp].into_iter().enumerate() {
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

/// OpenSpiel's chess observation, replicated exactly `(20, 8, 8)` — the interop/benchmark view
/// (see reinfors-benchmarks: both frameworks' nets consume identical inputs, making throughput,
/// learning curves AND infer-cache hit rates commensurable — including their encoding's
/// en-passant blindness, which merges positions differing only in ep rights into one cache
/// entry). Layout from the pinned source (`chess.cc::ObservationTensor`, commit d15d49f8):
///   0..12  piece occupancy interleaved by TYPE in K Q R B N P order: white then black per type
///   12     empty-square occupancy
///   13     repetition count of the current position, (rep - 1) / 2 over [1, 3]
///   14     side to move as their player id (ColorToPlayer: Black=0, White=1) — all-ONES White
///   15     halfmove clock / 101
///   16..20 castling rights: W queenside, W kingside, B queenside, B kingside
/// Square order matches ours (rank-major, a1 = 0), so planes are straight bitboard writes.
/// Absolute frame: the action view is the identity (native 8x8x73 ids; action encodings are
/// deliberately NOT translated between frameworks — see the benchmarks ledger).
pub struct ChessPlanesOpenSpiel;

impl ActionView for ChessPlanesOpenSpiel {} // absolute: identity action view

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
        // Their player ids are inverted vs intuition: ColorToPlayer maps Black -> 0, White -> 1,
        // so the side-to-move plane is all-ONES when White is to move.
        if board.side_to_move() == Color::White {
            obs[14 * PLANE..15 * PLANE].fill(1.0);
        }
        obs[15 * PLANE..16 * PLANE].fill(f32::from(board.halfmove_clock()) / 101.0);
        for (i, color) in [Color::White, Color::Black].into_iter().enumerate() {
            let rights = board.castle_rights(color);
            if rights.long.is_some() {
                obs[(16 + i * 2) * PLANE..(17 + i * 2) * PLANE].fill(1.0); // queenside first
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

/// AlphaZero's 119-plane chess observation `(119, 8, 8)`, in this framework's ABSOLUTE orientation
/// (the paper flips the board to the mover's perspective; a flipped view belongs together with a
/// flipped action mapping, which is game-level, not encoder-level — so both stay absolute here and
/// plane 112 carries side-to-move instead):
///   0..112   8 history steps, NEWEST FIRST (t=0 = current position), 14 planes each:
///            6 White pieces (P N B R Q K), 6 Black pieces, repeated-once (count >= 2),
///            repeated-twice (count >= 3) — all-ones indicator planes, counted in the
///            threefold window. Steps before the game start are all-zero.
///   112      side to move (all-ones when White)
///   113      fullmove number / 100
///   114..118 castling rights: W short, W long, B short, B long
///   118      halfmove clock / 100
/// Requires a `Chess` with `history_len == self.history` (the factory pairs them); with history off
/// only the t=0 step is populated.
pub struct ChessPlanesAz119 {
    /// History steps encoded (the paper's 8). Plane count = `14 * history + 7`.
    pub history: usize,
}

impl Default for ChessPlanesAz119 {
    fn default() -> Self {
        ChessPlanesAz119 { history: 8 }
    }
}

impl ChessPlanesAz119 {
    /// Piece + repetition planes for one historical position into `obs[base..base + 14*64]`.
    /// `count` is that position's repetition count as of its own moment.
    fn step_planes(obs: &mut [f32], base: usize, board: &Board, count: usize) {
        const PLANE: usize = 64;
        for (ci, color) in [Color::White, Color::Black].into_iter().enumerate() {
            for (pi, piece) in Piece::ALL.into_iter().enumerate() {
                let bb = board.pieces(piece) & board.colors(color);
                for sq in bb {
                    obs[base + (ci * 6 + pi) * PLANE + sq_index(sq)] = 1.0;
                }
            }
        }
        if count >= 2 {
            obs[base + 12 * PLANE..base + 13 * PLANE].fill(1.0);
        }
        if count >= 3 {
            obs[base + 13 * PLANE..base + 14 * PLANE].fill(1.0);
        }
    }
}

impl ActionView for ChessPlanesAz119 {} // absolute: identity action view

impl StateEncoder for ChessPlanesAz119 {
    type State = ChessState;

    fn encode(&self, state: &ChessState, _agent: usize) -> Vec<f32> {
        const PLANE: usize = 64;
        let aux = 14 * self.history; // first auxiliary plane index
        let mut obs = vec![0.0f32; (aux + 7) * PLANE];
        let board = &state.board;
        // History steps, newest first. `recent` ends with the current position when history is on;
        // with history off, synthesize the single current step so t=0 is always populated.
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
        for (i, color) in [Color::White, Color::Black].into_iter().enumerate() {
            let rights = board.castle_rights(color);
            if rights.short.is_some() {
                obs[(aux + 2 + i * 2) * PLANE..(aux + 3 + i * 2) * PLANE].fill(1.0);
            }
            if rights.long.is_some() {
                obs[(aux + 3 + i * 2) * PLANE..(aux + 4 + i * 2) * PLANE].fill(1.0);
            }
        }
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
            recent: Vec::new(),
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

#[cfg(test)]
mod az119_tests {
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

    fn hist_game() -> (Chess, ChessState) {
        let game = Chess {
            max_ticks: Some(512),
            history_len: 8,
        };
        let state = game.initial_state(&mut NoRng);
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
        // t=0 piece planes (0..12) must equal the minimal encoder's piece planes.
        assert_eq!(&az[..12 * PLANE], &minimal[..12 * PLANE]);
    }

    #[test]
    fn history_step_one_is_the_previous_position() {
        let (game, state) = hist_game();
        let before = ChessPlanesAz119::default().encode(&state, 0);
        let after = ChessPlanesAz119::default().encode(&play(&game, state, &["e2e4"]), 0);
        // t=1 of the new obs = t=0 of the old obs (piece planes).
        assert_eq!(&after[14 * PLANE..26 * PLANE], &before[..12 * PLANE]);
        // and t=0 changed (the pawn moved).
        assert_ne!(&after[..12 * PLANE], &before[..12 * PLANE]);
    }

    #[test]
    fn repetition_planes_activate_exactly_when_positions_recur() {
        let (game, state) = hist_game();
        let shuffle = ["g1f3", "g8f6", "f3g1", "f6g8"];
        // After 4 plies the start position has occurred twice -> t=0 repeated-once plane on.
        let s4 = play(&game, state, &shuffle);
        let obs4 = ChessPlanesAz119::default().encode(&s4, 0);
        assert!(obs4[12 * PLANE..13 * PLANE].iter().all(|&v| v == 1.0));
        assert!(obs4[13 * PLANE..14 * PLANE].iter().all(|&v| v == 0.0));
        // The half-shuffled positions in between occurred once -> their history rep planes stay 0.
        assert!(obs4[(14 + 12) * PLANE..(14 + 13) * PLANE]
            .iter()
            .all(|&v| v == 0.0));
        // The start position's FIRST occurrence (t=4 in history) must NOT be flagged — its count
        // was 1 at the time (the exactness the per-step stored counts buy).
        let t4 = 4 * 14 * PLANE;
        assert!(obs4[t4 + 12 * PLANE..t4 + 13 * PLANE]
            .iter()
            .all(|&v| v == 0.0));
    }

    #[test]
    fn aux_planes() {
        let (game, state) = hist_game();
        let obs = ChessPlanesAz119::default().encode(&state, 0);
        assert!(obs[112 * PLANE..113 * PLANE].iter().all(|&v| v == 1.0)); // white to move
        assert!((obs[113 * PLANE] - 0.01).abs() < 1e-6); // fullmove 1 / 100
        for p in 114..118 {
            assert!(obs[p * PLANE..(p + 1) * PLANE].iter().all(|&v| v == 1.0)); // all rights
        }
        let s = play(&game, state, &["e2e4"]);
        let obs = ChessPlanesAz119::default().encode(&s, 0);
        assert!(obs[112 * PLANE..113 * PLANE].iter().all(|&v| v == 0.0)); // black to move
    }

    #[test]
    fn short_history_shapes_and_truncates() {
        let game = Chess {
            max_ticks: Some(512),
            history_len: 2,
        };
        let enc = ChessPlanesAz119 { history: 2 };
        assert_eq!(enc.obs_shape(), (35, 8, 8)); // 14*2 + 7
        let state = game.initial_state(&mut NoRng);
        let s = play(&game, state, &["e2e4", "e7e5", "g1f3"]);
        let obs = enc.encode(&s, 0);
        assert_eq!(obs.len(), 35 * PLANE);
        // Ring truncated to 2: both steps populated, aux planes at 28.. (black to move -> side 0).
        assert!(obs[..12 * PLANE].contains(&1.0));
        assert!(obs[14 * PLANE..26 * PLANE].contains(&1.0));
        assert!(obs[28 * PLANE..29 * PLANE].iter().all(|&v| v == 0.0));
    }

    #[test]
    fn history_off_populates_only_the_current_step() {
        let game = Chess::default(); // history_len: 0
        let state = game.initial_state(&mut NoRng);
        let s = play(&game, state, &["e2e4", "e7e5"]);
        let obs = ChessPlanesAz119::default().encode(&s, 0);
        assert!(obs[..12 * PLANE].contains(&1.0)); // t=0 present
        assert!(obs[14 * PLANE..112 * PLANE].iter().all(|&v| v == 0.0)); // no history steps
    }
}

#[cfg(test)]
mod relative_frame_tests {
    //! The sigma-coherence suite for the mover-relative encoder: `encode` and the `ActionView`
    //! must apply the SAME role symmetry. Any drift — reflecting the board but not en passant,
    //! reflecting observations but not actions, a wrong ray/knight table — fails these.
    use super::*;
    use reinfors_core::check_action_view;

    /// sigma on a FEN: reverse rank rows + swap piece case, flip side to move, swap castling
    /// case, reflect the en-passant rank (3 <-> 6). Clocks unchanged.
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
        // Deterministic pseudo-random walk; at EVERY position s with mover m, against t = sigma(s):
        //   encode(s, m) == encode(t, 1 - m)                                  (obs equivariance)
        //   head_index(id(mv in s), m) == head_index(id(sigma(mv) in t), 1-m) (action coherence)
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
        // The initial position equals its own sigma image, so the two perspectives may differ
        // ONLY in plane 12 (whose turn it is) — a direct, scaffold-free equivariance check.
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
        let s = {
            struct R;
            impl reinfors_core::Rng for R {
                fn below(&mut self, _: usize) -> usize {
                    0
                }
                fn unit(&mut self) -> f64 {
                    0.0
                }
            }
            game.initial_state(&mut R)
        };
        let obs = ChessPlanesOpenSpiel.encode(&s, 0);
        let plane = |p: usize| obs[p * 64..(p + 1) * 64].iter().sum::<f32>();
        // K Q R B N P interleaved white/black
        for (p, n) in [(0, 1.0), (1, 1.0), (2, 1.0), (3, 1.0), (4, 2.0), (5, 2.0)] {
            assert_eq!(plane(p), n, "kings/queens/rooks plane {p}");
        }
        assert_eq!(plane(10), 8.0); // white pawns
        assert_eq!(plane(11), 8.0); // black pawns
        assert_eq!(plane(12), 32.0); // empty squares
        assert_eq!(plane(13), 0.0); // repetition 1 -> (1-1)/2
        assert_eq!(plane(14), 64.0); // White to move -> all ONES (their Black=0/White=1 ids)
        assert_eq!(plane(15), 0.0); // clock 0
        for p in 16..20 {
            assert_eq!(plane(p), 64.0, "castling plane {p}");
        }
        // a1 is index 0: white queenside rook present in the rook plane at slot 0.
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
        assert!(dead("8/8/4k3/8/8/3K4/8/8 w - - 0 1")); // K vs K
        assert!(dead("8/8/4k3/8/8/3KB3/8/8 w - - 0 1")); // K+B vs K
        assert!(dead("8/8/4kn2/8/8/3K4/8/8 w - - 0 1")); // K vs K+N
        assert!(dead("8/8/3bk3/8/8/3KB3/8/8 w - - 0 1")); // KB vs KB same-colored squares
        assert!(dead("8/8/4k3/8/8/3KB3/8/2B5 w - - 0 1")); // K+B+B vs K, same-colored squares
        assert!(!dead("8/8/2bk4/8/8/3KB3/8/8 w - - 0 1")); // KB vs KB opposite colors: helpmate
        assert!(!dead("8/8/3nk3/8/8/2NK4/8/8 w - - 0 1")); // KN vs KN: helpmate
        assert!(!dead("8/8/4k3/8/8/2NKN3/8/8 w - - 0 1")); // KNN vs K
        assert!(!dead("8/8/4k3/8/8/2NKB3/8/8 w - - 0 1")); // knight + bishop
        assert!(!dead("8/8/4k3/7p/8/3K4/8/8 w - - 0 1")); // a pawn plays on
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
        // Standard-UCI castling strings resolve to legal ids; cozy-form strings do NOT exist
        // in the standard rendering.
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
