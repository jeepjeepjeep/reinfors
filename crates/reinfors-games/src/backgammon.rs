//! Backgammon — sequential, zero-sum, with declared dice chance. Rules, geometry, action space,
//! and observation encoding deliberately mirror OpenSpiel's `backgammon` (pinned in the benchmarks
//! repo) so legal-move sets and encodings can be parity-tested position-for-position against it.
//!
//! **Chance modeling**: explicit chance states throughout, and every random element declared
//! (`initial_state` draws nothing). A completed turn hands off at the AwaitingRoll chance state (dice
//! unset → `Actor::Chance`); `chance_node` declares the 21 distinct rolls (non-doubles 1/18,
//! doubles 1/36) and `apply_chance_node(i)` stamps roll `i` in with neutral events. A doubles
//! turn whose first action used both dice re-arms the SAME dice for the same player (an extra
//! turn) — a deterministic transition, no chance state. The opening (who starts + first roll,
//! doubles excluded) is a ROOT chance phase: `initial_state` draws nothing and returns the
//! pre-roll state; its chance node declares 30 uniform outcomes (0–14 = X starts with roll i,
//! 15–29 = O starts — OpenSpiel's `turns_ == -1` node), realized at episode birth.
//!
//! **Action space** (OpenSpiel-compatible, 1352): one turn = one action encoding two half-moves as
//! 2 digits base 26 (source 0–23, bar = 24, pass = 25): `dig1 * 26 + dig0`, plus 676 when the LOW
//! die moves first. Doubles = up to two consecutive actions by the same player. Board geometry:
//! player X (agent 0) moves 0→23 with home 18–23; O (agent 1) moves 23→0 with home 0–5; the bar
//! enters at `die - 1` (X) / `24 - die` (O).
//!
//! **No doubling cube** (fixed stakes). Outcomes carry the margin: win / gammon (loser bore off
//! nothing) / backgammon (…and has a checker on the bar or in the winner's home), scored by the
//! decoupled [`BackgammonReward`] (defaults 1/2/3, zero-sum).

use std::collections::BTreeSet;

#[cfg(test)]
use reinfors_core::Rng;
use reinfors_core::{ActionView, Actor, Game, Reward, StateEncoder, Transition};

pub const NUM_POINTS: usize = 24;
pub const NUM_CHECKERS: u8 = 15;
pub const NUM_ACTIONS: usize = 1352;
pub const BAR: i32 = 100; // sentinel source: the bar (OpenSpiel's kBarPos)
pub const PASS: i32 = -1; // sentinel source: pass (OpenSpiel's kPassPos)
const ENC_BAR: i32 = 24; // bar as an action digit
const ENC_PASS: i32 = 25; // pass as an action digit
const SCORE: i32 = 101; // sentinel destination: borne off

/// The 21 distinct rolls in OpenSpiel's outcome order: 15 non-doubles then 6 doubles.
pub const ROLLS: [[u8; 2]; 21] = [
    [1, 2],
    [1, 3],
    [1, 4],
    [1, 5],
    [1, 6],
    [2, 3],
    [2, 4],
    [2, 5],
    [2, 6],
    [3, 4],
    [3, 5],
    [3, 6],
    [4, 5],
    [4, 6],
    [5, 6],
    [1, 1],
    [2, 2],
    [3, 3],
    [4, 4],
    [5, 5],
    [6, 6],
];

/// One half-move: a source (`0..24`, [`BAR`], or [`PASS`]) played with die `num`.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct CheckerMove {
    pub pos: i32,
    pub num: u8,
}

/// Per-agent tick outcome: the game ended with this margin for/against the agent, or continues.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BackgammonEvent {
    Ongoing,
    /// The agent won with margin 1 (plain), 2 (gammon), or 3 (backgammon).
    Win(u8),
    /// The agent lost with that margin.
    Loss(u8),
}

#[derive(Clone, PartialEq, Eq, Debug, serde::Serialize, serde::Deserialize)]
pub struct BackgammonState {
    /// Checker counts per player per point; X (agent 0) travels 0→23, O (agent 1) 23→0.
    pub board: [[u8; NUM_POINTS]; 2],
    pub bar: [u8; 2],
    pub scores: [u8; 2],
    pub to_move: u8,
    /// The two dice for the mover, 1–6; `die + 6` marks a used die; `[0, 0]` = the AwaitingRoll
    /// chance state (transient — the framework realizes it before anyone moves).
    pub dice: [u8; 2],
    /// This action is the second half of a doubles turn (same dice re-armed, no fresh roll).
    pub double_turn: bool,
    /// The root chance phase: the opening draw (starter + first roll) has not happened yet.
    /// Transient — realized at episode birth; never observable afterwards.
    #[serde(default)]
    pub opening: bool,
}

pub struct Backgammon {
    pub max_ticks: Option<usize>,
}

impl Default for Backgammon {
    fn default() -> Self {
        // Weak nets can shuffle checkers for a long time; a generous cap keeps rollouts finite
        // (the AZ family's truncation tail-bootstrap handles the cut).
        Backgammon {
            max_ticks: Some(1000),
        }
    }
}

impl BackgammonState {
    fn initial_board() -> [[u8; NUM_POINTS]; 2] {
        let mut board = [[0u8; NUM_POINTS]; 2];
        board[0][0] = 2;
        board[0][11] = 5;
        board[0][16] = 3;
        board[0][18] = 5;
        board[1][23] = 2;
        board[1][12] = 5;
        board[1][7] = 3;
        board[1][5] = 5;
        board
    }

    fn opponent(player: usize) -> usize {
        1 - player
    }

    /// Destination of moving `spaces` from `pos` (a point or the bar); [`SCORE`] = off the board.
    fn position_from(player: usize, pos: i32, spaces: u8) -> i32 {
        if pos == BAR {
            return if player == 0 {
                i32::from(spaces) - 1
            } else {
                24 - i32::from(spaces)
            };
        }
        let next = if player == 0 {
            pos + i32::from(spaces)
        } else {
            pos - i32::from(spaces)
        };
        if !(0..24).contains(&next) {
            SCORE
        } else {
            next
        }
    }

    fn usable(die: u8) -> bool {
        (1..=6).contains(&die)
    }

    fn all_in_home(&self, player: usize) -> bool {
        if self.bar[player] > 0 {
            return false;
        }
        let (start, end) = if player == 0 { (0, 18) } else { (6, 24) };
        self.board[player][start..end].iter().all(|&c| c == 0)
    }

    fn furthest_in_home(&self, player: usize) -> Option<i32> {
        // "Furthest" = farthest from bearing off: X scans 18→23 ascending distance-from-off, i.e.
        // the LOWEST occupied home point; O the HIGHEST.
        if player == 0 {
            (18..24).find(|&i| self.board[0][i as usize] > 0)
        } else {
            (0..6).rev().find(|&i| self.board[1][i as usize] > 0)
        }
    }

    /// The legal single half-moves for the mover with the current (usable) dice — bar first.
    fn legal_checker_moves(&self, player: usize) -> BTreeSet<CheckerMove> {
        let mut moves = BTreeSet::new();
        let dice = self.dice;
        if self.bar[player] > 0 {
            for die in dice {
                if Self::usable(die) {
                    let pos = Self::position_from(player, BAR, die);
                    if self.board[Self::opponent(player)][pos as usize] <= 1 {
                        moves.insert(CheckerMove { pos: BAR, num: die });
                    }
                }
            }
            return moves;
        }
        let all_home = self.all_in_home(player);
        for i in 0..NUM_POINTS as i32 {
            if self.board[player][i as usize] == 0 {
                continue;
            }
            for die in dice {
                if !Self::usable(die) {
                    continue;
                }
                let pos = Self::position_from(player, i, die);
                if pos == SCORE {
                    if all_home {
                        let exact = if player == 0 {
                            i + i32::from(die) == 24
                        } else {
                            i - i32::from(die) == -1
                        };
                        if exact || Some(i) == self.furthest_in_home(player) {
                            moves.insert(CheckerMove { pos: i, num: die });
                        }
                    }
                } else if self.board[Self::opponent(player)][pos as usize] <= 1 {
                    moves.insert(CheckerMove { pos: i, num: die });
                }
            }
        }
        moves
    }

    /// Apply one half-move in place (pass does nothing); hits send the opponent to the bar; the
    /// consumed die is marked used (`+6`).
    fn apply_checker_move(&mut self, player: usize, m: CheckerMove) {
        if m.pos == PASS {
            return;
        }
        // Loud guards, not u8 wraparound: an illegal half-move (no checker at the source) must
        // panic with a message, never silently corrupt the board — the searches only produce
        // legal ids and the Env boundary validates, so these are unreachable backstops.
        let next = if m.pos == BAR {
            assert!(
                self.bar[player] > 0,
                "illegal half-move: no checker on the bar"
            );
            self.bar[player] -= 1;
            Self::position_from(player, BAR, m.num)
        } else {
            assert!(
                self.board[player][m.pos as usize] > 0,
                "illegal half-move: no checker at point {}",
                m.pos
            );
            self.board[player][m.pos as usize] -= 1;
            Self::position_from(player, m.pos, m.num)
        };
        for d in &mut self.dice {
            if *d == m.num {
                *d += 6;
                break;
            }
        }
        if next == SCORE {
            self.scores[player] += 1;
        } else {
            let opp = Self::opponent(player);
            if self.board[opp][next as usize] == 1 {
                self.board[opp][next as usize] = 0;
                self.bar[opp] += 1;
            }
            self.board[player][next as usize] += 1;
        }
    }

    /// All maximal move sequences (the movegen recursion): DFS over per-die legal moves to depth 2,
    /// collecting every sequence; the caller filters by the forced-move maxims.
    fn rec_legal_moves(
        &self,
        player: usize,
        seq: &mut Vec<CheckerMove>,
        out: &mut BTreeSet<Vec<CheckerMove>>,
    ) -> usize {
        if seq.len() == 2 {
            out.insert(seq.clone());
            return 2;
        }
        let moves = self.legal_checker_moves(player);
        if moves.is_empty() {
            out.insert(seq.clone());
            return seq.len();
        }
        let mut max_moves = 0;
        for m in moves {
            let mut child = self.clone();
            child.apply_checker_move(player, m);
            seq.push(m);
            max_moves = max_moves.max(child.rec_legal_moves(player, seq, out));
            seq.pop();
        }
        max_moves
    }

    fn high_low(&self) -> (u8, u8) {
        let a = if self.dice[0] > 6 {
            self.dice[0] - 6
        } else {
            self.dice[0]
        };
        let b = if self.dice[1] > 6 {
            self.dice[1] - 6
        } else {
            self.dice[1]
        };
        (a.max(b), a.min(b))
    }

    /// Encode a (≤2)-half-move sequence as the OpenSpiel action id.
    fn encode_action(&self, moves: &[CheckerMove]) -> usize {
        let (high, _low) = self.high_low();
        let mut dig0 = ENC_PASS;
        let mut dig1 = ENC_PASS;
        let mut high_first = false;
        if let Some(m0) = moves.first() {
            if m0.pos != PASS {
                dig0 = if m0.pos == BAR { ENC_BAR } else { m0.pos };
                high_first = m0.num == high;
            }
        }
        if let Some(m1) = moves.get(1) {
            if m1.pos != PASS {
                dig1 = if m1.pos == BAR { ENC_BAR } else { m1.pos };
            }
        }
        let mut action = (dig1 * 26 + dig0) as usize;
        if !high_first {
            action += 676;
        }
        action
    }

    /// Decode an action id into its two half-moves (die assignment per the high/low-first flag).
    fn decode_action(&self, action: usize) -> [CheckerMove; 2] {
        debug_assert!(action < NUM_ACTIONS);
        let high_first = action < 676;
        let a = if high_first { action } else { action - 676 };
        let digits = [(a % 26) as i32, (a / 26) as i32];
        let (high, low) = self.high_low();
        let mut moves = [CheckerMove { pos: PASS, num: 0 }; 2];
        for (i, m) in moves.iter_mut().enumerate() {
            let num = if (i == 0) == high_first { high } else { low };
            if digits[i] != ENC_PASS {
                *m = CheckerMove {
                    pos: if digits[i] == ENC_BAR { BAR } else { digits[i] },
                    num,
                };
            }
        }
        moves
    }

    /// The mover's legal action ids under the forced-move maxims: play both dice when possible;
    /// with only single moves available, only max-die singles; with none, the double-pass.
    fn legal_action_ids(&self, player: usize) -> Vec<usize> {
        let mut movelist = BTreeSet::new();
        let max_moves = self.rec_legal_moves(player, &mut Vec::new(), &mut movelist);
        let mut actions: Vec<usize> = match max_moves {
            0 => vec![self.encode_action(&[])], // dance: the double-pass (id 1351)
            1 => {
                let max_roll = movelist
                    .iter()
                    .filter(|s| s.len() == 1)
                    .map(|s| s[0].num)
                    .max()
                    .unwrap_or(0);
                movelist
                    .iter()
                    .filter(|s| s.len() == 1 && s[0].num == max_roll)
                    .map(|s| self.encode_action(s))
                    .collect()
            }
            _ => movelist
                .iter()
                .filter(|s| s.len() == 2)
                .map(|s| self.encode_action(s))
                .collect(),
        };
        actions.sort_unstable();
        actions.dedup();
        actions
    }

    /// Total checkers a player has anywhere (board + bar + borne off) — conserved at 15.
    pub fn total_checkers(&self, player: usize) -> u8 {
        self.board[player].iter().sum::<u8>() + self.bar[player] + self.scores[player]
    }

    /// The loser's margin category given a finished game: 3 = backgammoned (nothing borne off AND
    /// a checker on the bar or in the winner's home), 2 = gammoned (nothing borne off), 1 = plain.
    fn loss_margin(&self, loser: usize) -> u8 {
        if self.scores[loser] > 0 {
            return 1;
        }
        let in_winner_home = if loser == 0 {
            // X's checkers in O's home (0–5)? No: the WINNER is O, whose home is 0–5.
            self.board[0][0..6].iter().any(|&c| c > 0)
        } else {
            self.board[1][18..24].iter().any(|&c| c > 0)
        };
        if self.bar[loser] > 0 || in_winner_home {
            3
        } else {
            2
        }
    }
}

impl Game for Backgammon {
    type State = BackgammonState;
    type Event = BackgammonEvent;

    fn num_agents(&self) -> usize {
        2
    }

    fn action_count(&self) -> usize {
        NUM_ACTIONS
    }

    fn actor(&self, state: &BackgammonState) -> Actor {
        // Unarmed dice mark the AwaitingRoll chance state between turns: nature rolls before
        // the next player sees the position.
        if state.dice == [0, 0] {
            Actor::Chance
        } else {
            Actor::Agent(usize::from(state.to_move))
        }
    }

    fn legal_actions(&self, state: &BackgammonState, agent: usize) -> Vec<usize> {
        if state.dice == [0, 0] || agent != usize::from(state.to_move) {
            return Vec::new();
        }
        state.legal_action_ids(agent)
    }

    fn step(
        &self,
        state: &BackgammonState,
        actions: &[usize],
    ) -> Transition<BackgammonState, BackgammonEvent> {
        let player = usize::from(state.to_move);
        let mut next = state.clone();
        let moves = next.decode_action(actions[player]);
        for m in moves {
            next.apply_checker_move(player, m);
        }
        // Doubles extra turn: first half of a doubles turn that used BOTH dice re-arms them for
        // the same player — a deterministic transition (no fresh roll).
        let (d0, d1) = (state.dice[0], state.dice[1]);
        let mut extra_turn = false;
        if !state.double_turn && d0 == d1 && next.dice.iter().all(|&d| d > 6) {
            for d in &mut next.dice {
                *d -= 6;
            }
            extra_turn = true;
        }
        let terminal = next.scores[player] == NUM_CHECKERS;
        let mut events = [None, None];
        if terminal {
            let loser = BackgammonState::opponent(player);
            let margin = next.loss_margin(loser);
            events[player] = Some(BackgammonEvent::Win(margin));
            events[loser] = Some(BackgammonEvent::Loss(margin));
        }
        if extra_turn {
            next.double_turn = true;
        } else {
            next.to_move = 1 - state.to_move;
            next.dice = [0, 0]; // the AwaitingRoll chance state — nature rolls next
            next.double_turn = false;
        }
        Transition {
            next_state: next,
            events: events.to_vec(),
            terminal,
        }
    }

    fn chance_node(&self, state: &BackgammonState) -> reinfors_core::ChanceDist {
        debug_assert_eq!(state.dice, [0, 0], "chance only at AwaitingRoll states");
        if state.opening {
            // The opening: a non-double roll and who starts — 0–14 = X with `ROLLS[i]`,
            // 15–29 = O with `ROLLS[i - 15]` (OpenSpiel's `turns_ == -1` node).
            return reinfors_core::ChanceDist::Uniform(30);
        }
        // The 21 distinct rolls in `ROLLS` order: 15 non-doubles at 1/18, 6 doubles at 1/36.
        let mut probs = vec![1.0 / 18.0; 21];
        for p in probs.iter_mut().skip(15) {
            *p = 1.0 / 36.0;
        }
        reinfors_core::ChanceDist::Weighted(probs)
    }

    fn apply_chance_node(
        &self,
        state: &BackgammonState,
        outcome: usize,
    ) -> Transition<BackgammonState, BackgammonEvent> {
        let mut next = state.clone();
        if state.opening {
            let (starter, roll) = if outcome < 15 {
                (0u8, outcome)
            } else {
                (1u8, outcome - 15)
            };
            next.opening = false;
            next.to_move = starter;
            next.dice = ROLLS[roll];
            return Transition {
                next_state: next,
                events: vec![None, None],
                terminal: false,
            };
        }
        next.dice = ROLLS[outcome];
        // The roll settles nothing by itself — outcomes ride the checker plays it enables.
        Transition::silent(next, 2)
    }

    fn initial_state(&self) -> BackgammonState {
        // The opening is a declared ROOT chance phase (30 uniform outcomes — see `chance_node`);
        // `initial_state` draws nothing — structural now that it takes no rng. The
        // framework realizes the draw at episode birth.
        BackgammonState {
            board: BackgammonState::initial_board(),
            bar: [0, 0],
            scores: [0, 0],
            to_move: 0, // placeholder until the opening draw decides the starter
            dice: [0, 0],
            double_turn: false,
            opening: true,
        }
    }

    fn truncation_horizon(&self) -> Option<usize> {
        self.max_ticks
    }
}

/// The margin-aware zero-sum reward: `win`/`gammon`/`backgammon` are the winner's payoffs for
/// margins 1/2/3 (the loser scores the negative). Defaults 1/2/3 (OpenSpiel's `full_scoring`);
/// set `gammon = backgammon = win` for plain win/loss scoring.
pub struct BackgammonReward {
    pub win: f64,
    pub gammon: f64,
    pub backgammon: f64,
}

impl Default for BackgammonReward {
    fn default() -> Self {
        BackgammonReward {
            win: 1.0,
            gammon: 2.0,
            backgammon: 3.0,
        }
    }
}

impl BackgammonReward {
    fn magnitude(&self, margin: u8) -> f64 {
        match margin {
            1 => self.win,
            2 => self.gammon,
            _ => self.backgammon,
        }
    }
}

impl Reward for BackgammonReward {
    type Event = BackgammonEvent;
    fn step_reward(&self, event: &BackgammonEvent, _agent: usize) -> f64 {
        match event {
            BackgammonEvent::Ongoing => 0.0,
            BackgammonEvent::Win(m) => self.magnitude(*m),
            BackgammonEvent::Loss(m) => -self.magnitude(*m),
        }
    }
}

/// The Tesauro state encoding (OpenSpiel's `ObservationTensor`, 200 dims, player-relative): per
/// point 4 features for the requesting agent then 4 for the opponent (1/2/3/overflow encodings of
/// the checker count), then bar/score/is-my-turn for each side, then the two dice values.
pub struct BackgammonTesauro;

impl ActionView for BackgammonTesauro {} // absolute: identity action view

impl StateEncoder for BackgammonTesauro {
    type State = BackgammonState;

    fn encode(&self, s: &BackgammonState, agent: usize) -> Vec<f32> {
        let opp = BackgammonState::opponent(agent);
        let mut v = Vec::with_capacity(200);
        for player in [agent, opp] {
            for count in s.board[player] {
                let c = i32::from(count);
                v.push(f32::from(u8::from(c == 1)));
                v.push(f32::from(u8::from(c == 2)));
                v.push(f32::from(u8::from(c == 3)));
                v.push(if c > 3 { (c - 3) as f32 } else { 0.0 });
            }
        }
        v.push(f32::from(s.bar[agent]));
        v.push(f32::from(s.scores[agent]));
        v.push(f32::from(u8::from(usize::from(s.to_move) == agent)));
        v.push(f32::from(s.bar[opp]));
        v.push(f32::from(s.scores[opp]));
        v.push(f32::from(u8::from(usize::from(s.to_move) == opp)));
        v.push(f32::from(s.dice[0]));
        v.push(f32::from(s.dice[1]));
        v
    }

    fn obs_shape(&self) -> (usize, usize, usize) {
        (200, 1, 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestRng(u64);
    impl Rng for TestRng {
        fn below(&mut self, n: usize) -> usize {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 33) as usize) % n.max(1)
        }
        fn unit(&mut self) -> f64 {
            (self.below(1 << 24) as f64) / f64::from(1 << 24)
        }
    }

    fn game() -> Backgammon {
        Backgammon::default()
    }

    /// A hand-built position: X pieces at `x` points, O at `o` (with off-board scores making both
    /// sides total 15 so the conservation invariant holds in tests that check it).
    fn state(
        x: &[(usize, u8)],
        o: &[(usize, u8)],
        bar: [u8; 2],
        dice: [u8; 2],
        to_move: u8,
    ) -> BackgammonState {
        let mut board = [[0u8; NUM_POINTS]; 2];
        for &(p, c) in x {
            board[0][p] = c;
        }
        for &(p, c) in o {
            board[1][p] = c;
        }
        let on: [u8; 2] = [
            board[0].iter().sum::<u8>() + bar[0],
            board[1].iter().sum::<u8>() + bar[1],
        ];
        BackgammonState {
            board,
            bar,
            scores: [NUM_CHECKERS - on[0], NUM_CHECKERS - on[1]],
            to_move,
            dice,
            double_turn: false,
            opening: false,
        }
    }

    #[test]
    fn opening_covers_thirty_outcomes() {
        // 0-14 -> X starts with non-double roll i; 15-29 -> O with roll i-15 (OpenSpiel's opening).
        let g = game();
        let mut seen = [[false; 2]; 15];
        for _ in 0..2000 {
            let s = reinfors_core::game::realize_initial_state(&g, &mut TestRng(rand_seed()));
            assert_ne!(s.dice[0], s.dice[1], "opening roll is never a double");
            let roll = ROLLS.iter().position(|r| *r == s.dice).unwrap();
            assert!(roll < 15);
            seen[roll][usize::from(s.to_move)] = true;
        }
        assert!(
            seen.iter().all(|r| r[0] && r[1]),
            "every (roll, starter) pair reachable"
        );
    }

    fn rand_seed() -> u64 {
        use std::sync::atomic::{AtomicU64, Ordering};
        static C: AtomicU64 = AtomicU64::new(7);
        C.fetch_add(0x9E37_79B9_7F4A_7C15, Ordering::Relaxed)
    }

    #[test]
    fn encode_decode_roundtrip_on_legal_actions() {
        let g = game();
        let mut rng = TestRng(3);
        let mut s = reinfors_core::game::realize_initial_state(&g, &mut rng);
        for _ in 0..200 {
            let legal = g.legal_actions(&s, usize::from(s.to_move));
            assert!(!legal.is_empty());
            for &a in &legal {
                let moves = s.decode_action(a);
                let non_pass: Vec<CheckerMove> =
                    moves.iter().copied().filter(|m| m.pos != PASS).collect();
                let re = s.encode_action(if non_pass.is_empty() {
                    &[]
                } else {
                    // encode from the SAME slot layout the decode produced
                    &moves[..]
                });
                assert_eq!(re, a, "roundtrip failed for action {a} in {s:?}");
            }
            let a = legal[rng.below(legal.len())];
            let t = g.step(&s, &[a, a]);
            if t.terminal {
                s = reinfors_core::game::realize_initial_state(&g, &mut rng);
            } else {
                s = t.next_state;
                while matches!(g.actor(&s), Actor::Chance) {
                    let reinfors_core::ChanceDist::Weighted(probs) = g.chance_node(&s) else {
                        unreachable!("backgammon declares weighted rolls")
                    };
                    s = g
                        .apply_chance_node(&s, tests_weighted(&mut rng, &probs))
                        .next_state;
                }
            }
        }
    }

    #[test]
    fn random_selfplay_conserves_checkers_and_terminates() {
        let g = game();
        let mut rng = TestRng(11);
        let mut finished = 0;
        for _ in 0..8 {
            let mut s = reinfors_core::game::realize_initial_state(&g, &mut rng);
            for _tick in 0..2000 {
                for p in 0..2 {
                    assert_eq!(
                        s.total_checkers(p),
                        NUM_CHECKERS,
                        "conservation broke: {s:?}"
                    );
                }
                let legal = g.legal_actions(&s, usize::from(s.to_move));
                let a = legal[rng.below(legal.len())];
                let t = g.step(&s, &[a, a]);
                for p in 0..2 {
                    assert_eq!(t.next_state.total_checkers(p), NUM_CHECKERS);
                }
                if t.terminal {
                    finished += 1;
                    let (w, l);
                    if t.events[0].is_some() {
                        match t.events[0].unwrap() {
                            BackgammonEvent::Win(m) => {
                                (w, l) = (BackgammonEvent::Win(m), BackgammonEvent::Loss(m));
                                assert_eq!(t.events[1], Some(l));
                                assert_eq!(t.events[0], Some(w));
                            }
                            BackgammonEvent::Loss(m) => {
                                assert_eq!(t.events[1], Some(BackgammonEvent::Win(m)));
                            }
                            BackgammonEvent::Ongoing => unreachable!(),
                        }
                    }
                    break;
                }
                s = t.next_state;
                while matches!(g.actor(&s), Actor::Chance) {
                    let reinfors_core::ChanceDist::Weighted(probs) = g.chance_node(&s) else {
                        unreachable!("backgammon declares weighted rolls")
                    };
                    s = g
                        .apply_chance_node(&s, tests_weighted(&mut rng, &probs))
                        .next_state;
                }
            }
        }
        assert!(
            finished >= 6,
            "random games should mostly finish inside 2000 ticks: {finished}/8"
        );
    }

    #[test]
    fn single_playable_die_forces_the_higher() {
        // X: one checker on 0, rest borne off. O blocks 11 (= 0+5+6 both orders), 5 and 6 open:
        // either die is playable alone but not both -> only the HIGHER (6) single is legal.
        let s = state(&[(0, 1)], &[(11, 2), (12, 13)], [0, 0], [5, 6], 0);
        let legal = s.legal_action_ids(0);
        let decoded: Vec<[CheckerMove; 2]> = legal.iter().map(|&a| s.decode_action(a)).collect();
        assert!(!legal.is_empty());
        for moves in &decoded {
            let played: Vec<&CheckerMove> = moves.iter().filter(|m| m.pos != PASS).collect();
            assert_eq!(
                played.len(),
                1,
                "only single moves are legal here: {decoded:?}"
            );
            assert_eq!(
                played[0].num, 6,
                "the higher die must be played: {decoded:?}"
            );
            assert_eq!(played[0].pos, 0);
        }
    }

    #[test]
    fn bar_checkers_must_enter_first() {
        // X has a checker on the bar; O blocks entry for die 3 (point 2). Every legal action's
        // first half-move is a bar entry with die 5 (entry at 4).
        let s = state(&[(11, 14)], &[(2, 2), (12, 13)], [1, 0], [3, 5], 0);
        let legal = s.legal_action_ids(0);
        assert!(!legal.is_empty());
        for &a in &legal {
            let moves = s.decode_action(a);
            assert_eq!(moves[0].pos, BAR, "bar first: {moves:?}");
            assert_eq!(moves[0].num, 5, "die-3 entry is blocked: {moves:?}");
        }
    }

    #[test]
    fn dance_yields_only_the_double_pass() {
        // O owns every X entry point (0-5): X on the bar cannot move at all.
        let s = state(
            &[(11, 14)],
            &[(0, 2), (1, 2), (2, 2), (3, 2), (4, 2), (5, 2), (12, 3)],
            [1, 0],
            [3, 5],
            0,
        );
        assert_eq!(
            s.legal_action_ids(0),
            vec![1351],
            "the double-pass is the only action"
        );
        // And it applies as a no-op turn handing over at the AwaitingRoll chance state.
        let g = game();
        let t = g.step(&s, &[1351, 0]);
        assert_eq!(t.next_state.bar[0], 1);
        assert_eq!(t.next_state.to_move, 1);
        assert!(matches!(g.actor(&t.next_state), Actor::Chance));
    }

    #[test]
    fn bear_off_exact_and_furthest_rules() {
        // All X in home: 2@18, 1@20. Dice (6, 3): 18+6=24 exact -> legal; 20+6=26 over -> NOT
        // legal from 20 (18 is further); 20+3=23 regular move; 18+3=21 regular move.
        let s = state(&[(18, 2), (20, 1)], &[(0, 2), (1, 13)], [0, 0], [6, 3], 0);
        let singles: BTreeSet<CheckerMove> = s.legal_checker_moves(0);
        assert!(
            singles.contains(&CheckerMove { pos: 18, num: 6 }),
            "exact bear-off"
        );
        assert!(
            !singles.contains(&CheckerMove { pos: 20, num: 6 }),
            "over-shoot only from furthest"
        );
        // Now clear 18: 20 becomes furthest -> die 6 bears it off despite over-shooting.
        let s2 = state(&[(20, 1), (21, 2)], &[(0, 2), (1, 13)], [0, 0], [6, 3], 0);
        let singles2 = s2.legal_checker_moves(0);
        assert!(
            singles2.contains(&CheckerMove { pos: 20, num: 6 }),
            "furthest may over-shoot"
        );
        assert!(
            !singles2.contains(&CheckerMove { pos: 21, num: 6 }),
            "21 is not furthest"
        );
    }

    #[test]
    fn doubles_grant_an_extra_deterministic_turn() {
        let g = game();
        let s = state(&[(0, 2), (11, 13)], &[(23, 2), (12, 13)], [0, 0], [3, 3], 0);
        let legal = s.legal_action_ids(0);
        let t = g.step(&s, &[legal[0], 0]);
        assert!(!t.terminal);
        assert_eq!(
            t.next_state.to_move, 0,
            "same player moves again on doubles"
        );
        assert!(t.next_state.double_turn);
        assert_eq!(t.next_state.dice, [3, 3], "dice re-armed");
        assert!(
            !matches!(g.actor(&t.next_state), Actor::Chance),
            "extra turn is deterministic"
        );
        // The second half of the doubles turn hands over at the AwaitingRoll chance state.
        let legal2 = t.next_state.legal_action_ids(0);
        let t2 = g.step(&t.next_state, &[legal2[0], 0]);
        assert_eq!(t2.next_state.to_move, 1);
        assert!(matches!(g.actor(&t2.next_state), Actor::Chance));
    }

    #[test]
    fn hitting_a_blot_sends_it_to_the_bar() {
        // X plays 0 -> 3 (die 3) onto O's lone checker.
        let s = state(&[(0, 1), (11, 14)], &[(3, 1), (12, 14)], [0, 0], [3, 5], 0);
        let mut ns = s.clone();
        ns.apply_checker_move(0, CheckerMove { pos: 0, num: 3 });
        assert_eq!(ns.board[1][3], 0);
        assert_eq!(ns.bar[1], 1);
        assert_eq!(ns.board[0][3], 1);
    }

    #[test]
    fn chance_declaration_matches_dice_probabilities() {
        let g = game();
        let s = state(&[(0, 2), (11, 13)], &[(23, 2), (12, 13)], [0, 0], [2, 5], 0);
        let legal = s.legal_action_ids(0);
        let t = g.step(&s, &[legal[0], 0]);
        let next = t.next_state;
        assert!(
            matches!(g.actor(&next), Actor::Chance),
            "a completed turn lands on the AwaitingRoll chance state"
        );
        assert!(g.legal_actions(&next, 0).is_empty() && g.legal_actions(&next, 1).is_empty());
        let reinfors_core::ChanceDist::Weighted(probs) = g.chance_node(&next) else {
            panic!("backgammon declares weighted rolls");
        };
        assert_eq!(probs.len(), 21);
        assert!((probs.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!(probs[..15].iter().all(|&p| (p - 1.0 / 18.0).abs() < 1e-15));
        assert!(probs[15..].iter().all(|&p| (p - 1.0 / 36.0).abs() < 1e-15));
        for (i, roll) in ROLLS.iter().enumerate() {
            let realized = g.apply_chance_node(&next, i);
            assert_eq!(realized.next_state.dice, *roll);
            assert_eq!(realized.next_state.to_move, 1);
            assert!(!realized.terminal);
            assert!(realized.events.iter().all(Option::is_none));
        }
    }

    #[test]
    fn margins_gammon_and_backgammon() {
        // X bears off its last checker; O has borne off nothing and sits in X's home -> backgammon.
        let g = game();
        let s = state(&[(23, 1)], &[(20, 15)], [0, 0], [1, 2], 0);
        let t = g.step(&s, &[s.legal_action_ids(0)[0], 0]);
        assert!(t.terminal);
        assert_eq!(t.events[0], Some(BackgammonEvent::Win(3)));
        assert_eq!(t.events[1], Some(BackgammonEvent::Loss(3)));
        // O out of X's home with nothing borne off -> gammon.
        let s2 = state(&[(23, 1)], &[(10, 15)], [0, 0], [1, 2], 0);
        let t2 = g.step(&s2, &[s2.legal_action_ids(0)[0], 0]);
        assert_eq!(t2.events[0], Some(BackgammonEvent::Win(2)));
        // O has borne off one -> plain win.
        let s3 = state(&[(23, 1)], &[(10, 14)], [0, 0], [1, 2], 0);
        let t3 = g.step(&s3, &[s3.legal_action_ids(0)[0], 0]);
        assert_eq!(t3.events[0], Some(BackgammonEvent::Win(1)));
        let r = BackgammonReward::default();
        assert_eq!(r.step_reward(t.events[0].as_ref().unwrap(), 0), 3.0);
        assert_eq!(r.step_reward(t.events[1].as_ref().unwrap(), 1), -3.0);
    }

    #[test]
    fn tesauro_encoding_shape_and_perspective() {
        let g = game();
        let s = reinfors_core::game::realize_initial_state(&g, &mut TestRng(5));
        let enc = BackgammonTesauro;
        let a = enc.encode(&s, 0);
        let b = enc.encode(&s, 1);
        assert_eq!(a.len(), 200);
        assert_eq!(b.len(), 200);
        assert_ne!(a, b, "egocentric views differ");
        assert_eq!(enc.obs_shape(), (200, 1, 1));
    }
    #[test]
    fn the_opening_is_a_declared_root_chance_phase() {
        let g = game();
        let root = g.initial_state(); // draws nothing: the opening is declared
        assert!(matches!(g.actor(&root), Actor::Chance));
        assert!(g.legal_actions(&root, 0).is_empty() && g.legal_actions(&root, 1).is_empty());
        let dist = g.chance_node(&root);
        assert_eq!(dist, reinfors_core::ChanceDist::Uniform(30));
        for outcome in 0..30 {
            let t = g.apply_chance_node(&root, outcome);
            assert!(!t.terminal);
            assert!(t.events.iter().all(Option::is_none));
            let s = t.next_state;
            assert!(!s.opening);
            let (starter, roll) = if outcome < 15 {
                (0, outcome)
            } else {
                (1, outcome - 15)
            };
            assert_eq!(usize::from(s.to_move), starter);
            assert_eq!(s.dice, ROLLS[roll]);
            assert_ne!(s.dice[0], s.dice[1], "the opening roll is never a double");
        }
        // Birth realization runs the root phase to a realized decision state.
        let born = reinfors_core::game::realize_initial_state(&g, &mut TestRng(3));
        assert!(!born.opening);
        assert!(matches!(g.actor(&born), Actor::Agent(_)));
    }

    fn tests_weighted(rng: &mut dyn Rng, probs: &[f64]) -> usize {
        let total: f64 = probs.iter().sum();
        let mut r = rng.unit() * total;
        for (i, &p) in probs.iter().enumerate() {
            r -= p;
            if r <= 0.0 {
                return i;
            }
        }
        probs.len() - 1
    }
}

impl reinfors_core::StateCodec for Backgammon {
    type State = BackgammonState;

    fn encode(&self, s: &BackgammonState) -> Vec<u8> {
        // Layout 3: the `opening` root-chance flag joined the state.
        crate::codec_util::serde_encode(3, s)
    }

    fn decode(&self, bytes: &[u8]) -> Result<BackgammonState, String> {
        crate::codec_util::serde_decode(3, bytes)
    }

    // Safety per the narrowed contract: the 15-checker sum bounds every count the move logic
    // does arithmetic on; ranges cover the indexing paths. Terminality is derived from the
    // borne-off scores (no state-side flag exists), so the envelope check compares against the
    // single derived source, not a duplicate.
    fn validate_decoded_state(&self, state: &BackgammonState, done: bool) -> Result<(), String> {
        if state.opening {
            // The opening draw is realized at episode birth; positions cannot await it.
            return Err("the opening roll is realized at episode birth".to_string());
        }
        if state.to_move > 1 {
            return Err(format!("to_move {} out of range", state.to_move));
        }
        for (p, (board, (bar, score))) in state
            .board
            .iter()
            .zip(state.bar.iter().zip(state.scores.iter()))
            .enumerate()
        {
            let total = board.iter().map(|&c| u32::from(c)).sum::<u32>()
                + u32::from(*bar)
                + u32::from(*score);
            if total != 15 {
                return Err(format!("player {p} has {total} checkers, expected 15"));
            }
        }
        for d in state.dice {
            if d > 12 {
                return Err(format!(
                    "die value {d} out of range (0 unrolled, 1-6 fresh, 7-12 used)"
                ));
            }
        }
        let finished = state.scores.iter().any(|&s| s >= 15);
        if !finished && state.dice == [0, 0] {
            // AwaitingRoll is a transient chance state the framework realizes inside a tick; a
            // restored live position must be actionable, not stuck awaiting nature.
            return Err("a live position must carry rolled dice".to_string());
        }
        if finished != done {
            return Err(format!(
                "borne-off counts {:?} disagree with done {done}",
                state.scores
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod codec_tests {
    use super::*;
    use reinfors_core::StateCodec;

    fn initial() -> (Backgammon, BackgammonState) {
        struct R(u64);
        impl reinfors_core::Rng for R {
            fn below(&mut self, n: usize) -> usize {
                self.0 = self
                    .0
                    .wrapping_mul(2862933555777941757)
                    .wrapping_add(3037000493);
                (self.0 >> 33) as usize % n.max(1)
            }
            fn unit(&mut self) -> f64 {
                0.5
            }
        }
        let game = Backgammon { max_ticks: None };
        let s = reinfors_core::game::realize_initial_state(&game, &mut R(9));
        (game, s)
    }

    #[test]
    fn round_trips_canonically_and_validates() {
        let (game, s) = initial();
        let bytes = game.encode(&s);
        let back = game.decode(&bytes).unwrap();
        assert_eq!(game.encode(&back), bytes);
        game.validate_decoded_state(&back, false).unwrap();
    }

    #[test]
    fn semantic_invariants_reject_tampered_structs() {
        let (game, s) = initial();
        let mut extra = s.clone();
        extra.board[0][0] += 1;
        assert!(game
            .validate_decoded_state(&extra, false)
            .unwrap_err()
            .contains("checkers"));
        let mut bad_die = s.clone();
        bad_die.dice[0] = 13;
        assert!(game
            .validate_decoded_state(&bad_die, false)
            .unwrap_err()
            .contains("die"));
        let mut won = s.clone();
        won.board[0] = [0; NUM_POINTS];
        won.bar[0] = 0;
        won.scores[0] = 15;
        assert!(game
            .validate_decoded_state(&won, false)
            .unwrap_err()
            .contains("borne-off"));
        game.validate_decoded_state(&won, true).unwrap();
    }
}
