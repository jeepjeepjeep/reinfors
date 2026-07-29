//! Fixed-limit Texas hold'em — reinfors' first HIDDEN-information game (2-9 players, one hand
//! per episode). The game declares `perfect_information() = false`, so the tree-search families
//! reject it at construction (their values would be clairvoyant about hole cards); training runs
//! through the observation-only DQN family, whose per-agent encoder shows each seat only its own
//! holes.
//!
//! **Rules** (standard fixed limit): blinds, then four streets (preflop/flop/turn/river) of
//! betting with a 3-action space — fold (only when facing a bet), check/call, bet/raise — a cap
//! of 4 bets per street (the preflop big blind counts as the first), small bets (one big blind)
//! preflop/flop and big bets (two) on turn/river. Calls short of the amount are all-in and build
//! side pots; showdown resolves per pot level with odd chips to the earliest seat left of the
//! button. Folding is illegal when checking is free — which also guarantees the largest
//! total commitment always belongs to a LIVE player, the invariant the side-pot sweep relies on.
//!
//! **Chance**: hole cards and the button are dealt by `initial_state` from the env's seeded rng;
//! street reveals are the transition's *declared* chance — `ChanceDist::Uniform(C(remaining, k))`
//! with the unordered combination decoded from the index by `apply_chance` (the compact
//! declaration exists for exactly this shape). Per the chance contract, outcomes never carry
//! reward: when everyone is all-in the remaining streets are stepped through FORCED CHECKS (a
//! single live seat holds one legal action), so each reveal is a zero-reward chance transition
//! and the showdown resolves deterministically from already-dealt cards.
//!
//! Hand strength comes from `rs_poker`'s 7-card ranking (the cozy-chess pattern: the crate owns
//! the pure, battle-tested evaluation; reinfors owns the betting machine, which the pyspiel
//! `universal_poker` parity harness validates).

use std::collections::HashSet;

use reinfors_core::game::{Actor, ChanceDist, Game, Rng, Transition};
use reinfors_core::Reward;

/// A card id `rank * 4 + suit`, rank 0 (deuce) to 12 (ace), suit 0-3 — the ACPC-style layout,
/// chosen for the parity harness's card mapping.
pub type Card = u8;

pub const DECK: u8 = 52;

pub fn card_rank(c: Card) -> u8 {
    c / 4
}

pub fn card_suit(c: Card) -> u8 {
    c % 4
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, serde::Serialize, serde::Deserialize)]
pub enum Street {
    Preflop,
    Flop,
    Turn,
    River,
    /// The hand is over (fold-out or showdown) — the state's single terminal marker.
    Done,
}

impl Street {
    /// Board cards a street plays with (Done keeps whatever was dealt).
    fn board_len(self) -> usize {
        match self {
            Street::Preflop => 0,
            Street::Flop => 3,
            Street::Turn => 4,
            Street::River => 5,
            Street::Done => 5,
        }
    }

    fn next(self) -> Street {
        match self {
            Street::Preflop => Street::Flop,
            Street::Flop => Street::Turn,
            Street::Turn => Street::River,
            Street::River | Street::Done => Street::Done,
        }
    }
}

/// The three fixed-limit actions, by id.
pub const FOLD: usize = 0;
pub const CHECK_CALL: usize = 1;
pub const BET_RAISE: usize = 2;
pub const HOLDEM_ACTIONS: usize = 3;

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct HoldemState {
    pub hole: Vec<[Card; 2]>, // [N] — every seat's cards live in the TRUE state; encoders hide
    pub board: Vec<Card>,     // canonical: each street's reveal appended in ascending card order
    pub button: usize,
    pub street: Street,
    pub to_act: usize,
    pub stacks: Vec<u32>,           // [N] chips behind
    pub street_committed: Vec<u32>, // [N] chips put in on the current street
    pub total_committed: Vec<u32>,  // [N] chips put in over the whole hand (side-pot levels)
    pub folded: Vec<bool>,
    /// Who still owes an action this street (reset on street open and on every full raise).
    pub needs_action: Vec<bool>,
    /// Bets + raises this street (the preflop big blind counts as the first bet).
    pub raises: u8,
}

impl HoldemState {
    pub fn is_done(&self) -> bool {
        self.street == Street::Done
    }

    fn live(&self, i: usize) -> bool {
        !self.folded[i]
    }

    fn to_call(&self, i: usize) -> u32 {
        self.street_committed.iter().copied().max().unwrap_or(0) - self.street_committed[i]
    }

    /// Cards not yet dealt, ascending — the canonical enumeration the chance indices decode
    /// against.
    fn remaining_deck(&self) -> Vec<Card> {
        let mut used: HashSet<Card> = self.board.iter().copied().collect();
        for h in &self.hole {
            used.insert(h[0]);
            used.insert(h[1]);
        }
        (0..DECK).filter(|c| !used.contains(c)).collect()
    }
}

/// Passes each seat's terminal chip delta through as its reward (the events are already
/// zero-sum); `scale` rescales chips into whatever unit the net trains on (e.g. big blinds).
pub struct HoldemReward {
    pub scale: f64,
}

impl Reward for HoldemReward {
    type Event = f64;
    fn step_reward(&self, e: &f64, _agent: usize) -> f64 {
        e * self.scale
    }
}

/// Fixed-limit hold'em config. One episode = one hand at fresh stacks; the button is drawn by
/// `initial_state`, so seats rotate positions across self-play episodes.
pub struct TexasHoldem {
    pub num_players: usize,
    /// Starting stack per seat, in chips.
    pub stack: u32,
    pub small_blind: u32,
    pub big_blind: u32,
}

impl TexasHoldem {
    pub fn validate(&self) -> Result<(), String> {
        if !(2..=9).contains(&self.num_players) {
            return Err("num_players must be in 2..=9".to_string());
        }
        if self.big_blind == 0 || self.small_blind == 0 {
            return Err("blinds must be positive".to_string());
        }
        if self.small_blind > self.big_blind {
            return Err("small_blind must not exceed big_blind".to_string());
        }
        if self.stack < self.big_blind {
            return Err("stack must cover the big blind".to_string());
        }
        // Chip arithmetic runs in u32 with i64 deltas; bound the totals far below overflow.
        if self.stack > 1 << 24 {
            return Err("stack must fit 2^24 chips".to_string());
        }
        Ok(())
    }

    /// The street's full bet increment: small bet preflop/flop, big bet turn/river.
    fn bet_size(&self, street: Street) -> u32 {
        match street {
            Street::Preflop | Street::Flop => self.big_blind,
            _ => self.big_blind * 2,
        }
    }

    /// First seat clockwise from `from` (exclusive) satisfying `pred`.
    fn next_seat(&self, from: usize, pred: impl Fn(usize) -> bool) -> Option<usize> {
        (1..=self.num_players)
            .map(|d| (from + d) % self.num_players)
            .find(|&i| pred(i))
    }

    /// Commit up to `amount` chips from seat `i` (short = all-in).
    fn commit(state: &mut HoldemState, i: usize, amount: u32) {
        let paid = amount.min(state.stacks[i]);
        state.stacks[i] -= paid;
        state.street_committed[i] += paid;
        state.total_committed[i] += paid;
    }

    /// Open a street: reset the per-street bookkeeping and seat the first actor. When fewer than
    /// two seats can still bet, betting is moot — a single live seat is given ONE forced check
    /// per street so every reveal stays an action-driven, zero-reward chance transition.
    fn open_street(&self, state: &mut HoldemState) {
        state.street_committed.iter_mut().for_each(|c| *c = 0);
        state.raises = 0;
        let can_bet: Vec<usize> = (0..self.num_players)
            .filter(|&i| state.live(i) && state.stacks[i] > 0)
            .collect();
        for i in 0..self.num_players {
            state.needs_action[i] = false;
        }
        if can_bet.len() >= 2 {
            for &i in &can_bet {
                state.needs_action[i] = true;
            }
        } else {
            // Runout: exactly one forced check from the first live seat.
            let first_live = self
                .next_seat(state.button, |i| state.live(i))
                .expect("a hand always has a live seat");
            state.needs_action[first_live] = true;
        }
        state.to_act = self
            .next_seat(state.button, |i| state.needs_action[i])
            .expect("street opened with someone to act");
    }

    /// Terminal resolution: per-agent chip deltas. Fold-out pays the last live seat; otherwise
    /// side pots resolve level by level (odd chips to the earliest seat left of the button).
    /// The fold-only-facing-a-bet rule guarantees the maximum total commitment is live, so the
    /// level sweep over LIVE totals collects every chip.
    fn payouts(&self, state: &HoldemState) -> Vec<f64> {
        let n = self.num_players;
        let mut deltas: Vec<i64> = (0..n).map(|i| -(state.total_committed[i] as i64)).collect();
        let live: Vec<usize> = (0..n).filter(|&i| state.live(i)).collect();
        if live.len() == 1 {
            let pot: i64 = state.total_committed.iter().map(|&c| c as i64).sum();
            deltas[live[0]] += pot;
            return deltas.iter().map(|&d| d as f64).collect();
        }
        let ranks: Vec<Option<rs_poker::core::Rank>> = (0..n)
            .map(|i| {
                state
                    .live(i)
                    .then(|| seven_card_rank(state.hole[i], &state.board))
            })
            .collect();
        let mut levels: Vec<u32> = live.iter().map(|&i| state.total_committed[i]).collect();
        levels.sort_unstable();
        levels.dedup();
        let mut prev = 0u32;
        for &level in &levels {
            let slice: i64 = state
                .total_committed
                .iter()
                .map(|&c| (c.min(level) - c.min(prev)) as i64)
                .sum();
            let winners = best_of(&live, &ranks, |i| state.total_committed[i] >= level);
            let share = slice / winners.len() as i64;
            let mut odd = slice % winners.len() as i64;
            // Odd chips: earliest winner clockwise from the small blind seat (button + 1).
            let mut order: Vec<usize> = winners.clone();
            order.sort_by_key(|&i| {
                (i + self.num_players - (state.button + 1) % self.num_players) % self.num_players
            });
            for &w in &order {
                deltas[w] += share + if odd > 0 { 1 } else { 0 };
                odd -= 1;
            }
            prev = level;
        }
        deltas.iter().map(|&d| d as f64).collect()
    }
}

/// The best-ranked subset of `live` seats passing `eligible`.
fn best_of(
    live: &[usize],
    ranks: &[Option<rs_poker::core::Rank>],
    eligible: impl Fn(usize) -> bool,
) -> Vec<usize> {
    let mut best: Vec<usize> = Vec::new();
    for &i in live {
        if !eligible(i) {
            continue;
        }
        let r = ranks[i].as_ref().expect("live seats are ranked");
        match best.first() {
            None => best.push(i),
            Some(&b) => {
                let rb = ranks[b].as_ref().expect("live seats are ranked");
                if r > rb {
                    best.clear();
                    best.push(i);
                } else if r == rb {
                    best.push(i);
                }
            }
        }
    }
    best
}

/// The best 5-card rank from 2 hole + up to 5 board cards (rs_poker's 7-card evaluation).
pub fn seven_card_rank(hole: [Card; 2], board: &[Card]) -> rs_poker::core::Rank {
    use rs_poker::core::{Card as RsCard, Hand, Rankable, Suit, Value};
    let cards: Vec<RsCard> = hole
        .iter()
        .chain(board.iter())
        .map(|&c| RsCard {
            value: Value::from_u8(card_rank(c)),
            suit: Suit::from_u8(card_suit(c)),
        })
        .collect();
    Hand::new_with_cards(cards).rank()
}

/// `C(n, k)` for the tiny deck-sized arguments the reveal indexing needs.
fn binomial(n: usize, k: usize) -> usize {
    if k > n {
        return 0;
    }
    let k = k.min(n - k);
    let mut out: usize = 1;
    for i in 0..k {
        out = out * (n - i) / (i + 1);
    }
    out
}

/// The `idx`-th k-subset of `0..m` in lexicographic order, ascending.
fn unrank_combination(mut idx: usize, m: usize, k: usize) -> Vec<usize> {
    let mut out = Vec::with_capacity(k);
    let mut a = 0;
    for remaining in (1..=k).rev() {
        loop {
            let with_a = binomial(m - a - 1, remaining - 1);
            if idx < with_a {
                out.push(a);
                a += 1;
                break;
            }
            idx -= with_a;
            a += 1;
        }
    }
    out
}

impl Game for TexasHoldem {
    type State = HoldemState;
    type Event = f64; // per-seat chip delta at the terminal tick, 0 elsewhere

    fn num_agents(&self) -> usize {
        self.num_players
    }

    fn action_count(&self) -> usize {
        HOLDEM_ACTIONS
    }

    fn perfect_information(&self) -> bool {
        false // hole cards are hidden — search families reject this game at construction
    }

    fn actor(&self, state: &HoldemState) -> Actor {
        Actor::Agent(state.to_act)
    }

    fn legal_actions(&self, state: &HoldemState, agent: usize) -> Vec<usize> {
        if state.is_done() || agent != state.to_act || state.board.len() < state.street.board_len()
        {
            return Vec::new(); // no actor mid-reveal (the env applies chance within the step)
        }
        let mut out = Vec::with_capacity(3);
        if state.to_call(agent) > 0 {
            out.push(FOLD); // folding with a free check available is illegal (ACPC convention)
        }
        out.push(CHECK_CALL);
        // A raise must add chips beyond the call, and the street must be under its bet cap. A
        // runout forced-check seat has an empty stack, so it never sees this arm.
        if state.raises < 4 && state.stacks[agent] > state.to_call(agent) {
            out.push(BET_RAISE);
        }
        out
    }

    fn step(&self, state: &HoldemState, actions: &[usize]) -> Transition<HoldemState, f64> {
        let me = state.to_act;
        let mut next = state.clone();
        let legal = self.legal_actions(state, me);
        // Backstop for direct core callers (the Env boundary validates, engine policies mask):
        // an illegal action folds when facing a bet, else checks.
        let action = if legal.contains(&actions[me]) {
            actions[me]
        } else if state.to_call(me) > 0 {
            FOLD
        } else {
            CHECK_CALL
        };

        match action {
            FOLD => {
                next.folded[me] = true;
                next.needs_action[me] = false;
            }
            CHECK_CALL => {
                let owed = next.to_call(me);
                Self::commit(&mut next, me, owed);
                next.needs_action[me] = false;
            }
            _ => {
                let owed = next.to_call(me);
                let bet = self.bet_size(next.street);
                Self::commit(&mut next, me, owed + bet);
                next.raises += 1;
                // A raise re-opens the action for every other live seat with chips behind.
                for j in 0..self.num_players {
                    next.needs_action[j] = j != me && next.live(j) && next.stacks[j] > 0;
                }
            }
        }

        let live_count = (0..self.num_players).filter(|&i| next.live(i)).count();
        let terminal_events = |s: &HoldemState| self.payouts(s);
        if live_count == 1 {
            next.street = Street::Done;
            let events = terminal_events(&next);
            return Transition {
                next_state: next,
                events,
                terminal: true,
            };
        }
        if next.needs_action.iter().any(|&b| b) {
            next.to_act = self
                .next_seat(next.to_act, |i| next.needs_action[i])
                .expect("someone still owes an action");
            return Transition {
                next_state: next,
                events: vec![0.0; self.num_players],
                terminal: false,
            };
        }
        // The street closes.
        if next.street == Street::River {
            next.street = Street::Done;
            let events = terminal_events(&next);
            return Transition {
                next_state: next,
                events,
                terminal: true,
            };
        }
        next.street = next.street.next();
        self.open_street(&mut next);
        // The board is now SHORT for the new street: the reveal is this transition's declared
        // chance, decoded by `apply_chance` before any actor sees the state.
        Transition {
            next_state: next,
            events: vec![0.0; self.num_players],
            terminal: false,
        }
    }

    fn chance_outcomes(
        &self,
        _state: &HoldemState,
        t: &Transition<HoldemState, f64>,
    ) -> Option<ChanceDist> {
        let next = &t.next_state;
        let missing = next.street.board_len().saturating_sub(next.board.len());
        if t.terminal || missing == 0 {
            return None;
        }
        Some(ChanceDist::Uniform(binomial(
            next.remaining_deck().len(),
            missing,
        )))
    }

    fn apply_chance(
        &self,
        _state: &HoldemState,
        t: &Transition<HoldemState, f64>,
        outcome: usize,
    ) -> HoldemState {
        let mut out = t.next_state.clone();
        let deck = out.remaining_deck();
        let missing = out.street.board_len() - out.board.len();
        for pos in unrank_combination(outcome, deck.len(), missing) {
            out.board.push(deck[pos]);
        }
        out
    }

    fn initial_state(&self, rng: &mut dyn Rng) -> HoldemState {
        let n = self.num_players;
        let button = rng.below(n);
        let mut deck: Vec<Card> = (0..DECK).collect();
        let mut draw = |deck: &mut Vec<Card>| {
            let i = rng.below(deck.len());
            deck.swap_remove(i)
        };
        let hole: Vec<[Card; 2]> = (0..n).map(|_| [draw(&mut deck), draw(&mut deck)]).collect();
        let mut state = HoldemState {
            hole,
            board: Vec::new(),
            button,
            street: Street::Preflop,
            to_act: 0,
            stacks: vec![self.stack; n],
            street_committed: vec![0; n],
            total_committed: vec![0; n],
            folded: vec![false; n],
            needs_action: vec![true; n],
            raises: 1, // the big blind is the street's first bet
        };
        // Blinds: heads-up the button IS the small blind and acts first preflop; otherwise the
        // two seats after the button post and the seat after the big blind opens.
        let (sb, bb) = if n == 2 {
            (button, (button + 1) % n)
        } else {
            ((button + 1) % n, (button + 2) % n)
        };
        Self::commit(&mut state, sb, self.small_blind);
        Self::commit(&mut state, bb, self.big_blind);
        state.to_act = if n == 2 { button } else { (bb + 1) % n };
        state
    }

    fn truncation_horizon(&self) -> Option<usize> {
        None // capped raises and finite streets: a hand always terminates on its own
    }
}

impl reinfors_core::StateCodec for TexasHoldem {
    type State = HoldemState;

    fn encode(&self, s: &HoldemState) -> Vec<u8> {
        crate::codec_util::serde_encode(1, s)
    }

    fn decode(&self, bytes: &[u8]) -> Result<HoldemState, String> {
        crate::codec_util::serde_decode(1, bytes)
    }

    fn validate_decoded_state(&self, state: &HoldemState, done: bool) -> Result<(), String> {
        let n = self.num_players;
        for (name, len) in [
            ("hole", state.hole.len()),
            ("stacks", state.stacks.len()),
            ("street_committed", state.street_committed.len()),
            ("total_committed", state.total_committed.len()),
            ("folded", state.folded.len()),
            ("needs_action", state.needs_action.len()),
        ] {
            if len != n {
                return Err(format!("{name} has {len} seats; this game has {n}"));
            }
        }
        if state.to_act >= n || state.button >= n {
            return Err("seat index out of range".to_string());
        }
        let mut seen = HashSet::new();
        for &c in state.hole.iter().flatten().chain(state.board.iter()) {
            if c >= DECK {
                return Err(format!("card id {c} out of range"));
            }
            if !seen.insert(c) {
                return Err(format!("card {c} dealt twice"));
            }
        }
        if state.board.len() > 5
            || (!state.is_done() && state.board.len() != state.street.board_len())
        {
            return Err("board length inconsistent with the street".to_string());
        }
        // Chip conservation: what's behind plus what's committed is exactly the buy-ins.
        let total: u64 = state.stacks.iter().map(|&c| c as u64).sum::<u64>()
            + state.total_committed.iter().map(|&c| c as u64).sum::<u64>();
        if total != self.stack as u64 * n as u64 {
            return Err(format!(
                "chips do not conserve: {total} != {} x {n}",
                self.stack
            ));
        }
        for i in 0..n {
            if state.street_committed[i] > state.total_committed[i] {
                return Err("street commitment exceeds the hand total".to_string());
            }
        }
        if state.raises > 8 {
            return Err(format!("implausible raise count {}", state.raises));
        }
        if state.is_done() != done {
            return Err(format!(
                "state street {:?} disagrees with envelope done {done}",
                state.street
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use reinfors_core::game::step_env;
    use reinfors_core::StateCodec;

    struct TestRng(u64);
    impl Rng for TestRng {
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

    fn game(n: usize) -> TexasHoldem {
        TexasHoldem {
            num_players: n,
            stack: 200,
            small_blind: 5,
            big_blind: 10,
        }
    }

    fn card(rank: u8, suit: u8) -> Card {
        rank * 4 + suit
    }

    #[test]
    fn validation_bounds_the_config() {
        assert!(game(2).validate().is_ok());
        assert!(game(9).validate().is_ok());
        assert!(game(1).validate().is_err());
        assert!(game(10).validate().is_err());
        let mut g = game(3);
        g.small_blind = 20;
        assert!(g.validate().is_err(), "sb > bb");
        g = game(3);
        g.stack = 5;
        assert!(g.validate().is_err(), "stack under the big blind");
        g = game(3);
        g.stack = 1 << 25;
        assert!(g.validate().is_err(), "stack over the chip bound");
    }

    #[test]
    fn evaluator_orders_the_hand_categories() {
        // One representative per category, each strictly beating the next.
        let board = [card(2, 0), card(3, 1), card(9, 2), card(10, 3), card(12, 0)];
        let straight_flush = seven_card_rank(
            [card(0, 1), card(4, 1)],
            &[card(1, 1), card(2, 1), card(3, 1), card(9, 2), card(12, 0)],
        );
        let quads = seven_card_rank(
            [card(9, 0), card(9, 1)],
            &[card(9, 2), card(9, 3), card(3, 1), card(2, 0), card(12, 0)],
        );
        let boat = seven_card_rank(
            [card(9, 0), card(9, 1)],
            &[card(9, 3), card(3, 1), card(3, 0), card(2, 0), card(12, 0)],
        );
        let flush = seven_card_rank(
            [card(0, 1), card(7, 1)],
            &[card(1, 1), card(2, 1), card(9, 1), card(9, 2), card(12, 0)],
        );
        let straight = seven_card_rank(
            [card(4, 0), card(5, 1)],
            &[card(6, 2), card(7, 3), card(8, 0), card(2, 0), card(12, 0)],
        );
        let trips = seven_card_rank([card(9, 0), card(9, 1)], &board[..5]);
        let two_pair = seven_card_rank([card(2, 1), card(3, 0)], &board[..5]);
        let pair = seven_card_rank([card(12, 1), card(5, 0)], &board[..5]);
        let high = seven_card_rank([card(11, 1), card(5, 0)], &board[..5]);
        let order = [
            straight_flush,
            quads,
            boat,
            flush,
            straight,
            trips,
            two_pair,
            pair,
            high,
        ];
        for w in order.windows(2) {
            assert!(w[0] > w[1], "{:?} must beat {:?}", w[0], w[1]);
        }
        // The wheel: A-2-3-4-5 is a straight but the lowest one.
        let wheel = seven_card_rank(
            [card(12, 0), card(0, 1)],
            &[card(1, 2), card(2, 3), card(3, 0), card(9, 1), card(10, 2)],
        );
        assert!(wheel < straight && wheel > trips);
    }

    #[test]
    fn combination_unranking_is_a_sorted_bijection() {
        for (m, k) in [(5, 3), (7, 2), (10, 1), (48, 3)] {
            let count = binomial(m, k);
            let mut seen = HashSet::new();
            for idx in 0..count.min(20_000) {
                let comb = unrank_combination(idx, m, k);
                assert_eq!(comb.len(), k);
                assert!(comb.windows(2).all(|w| w[0] < w[1]), "ascending");
                assert!(comb.iter().all(|&x| x < m));
                assert!(seen.insert(comb), "distinct");
            }
        }
    }

    #[test]
    fn blinds_positions_and_preflop_order() {
        // 3-handed: SB = button+1, BB = button+2, UTG (= button) opens.
        let g = game(3);
        let mut rng = TestRng(1);
        let s = g.initial_state(&mut rng);
        let (sb, bb) = ((s.button + 1) % 3, (s.button + 2) % 3);
        assert_eq!(s.total_committed[sb], 5);
        assert_eq!(s.total_committed[bb], 10);
        assert_eq!(s.to_act, (bb + 1) % 3);
        assert_eq!(s.raises, 1, "the big blind is the first bet");
        // Heads-up: the button posts the SMALL blind and acts first.
        let g2 = game(2);
        let s2 = g2.initial_state(&mut TestRng(2));
        assert_eq!(s2.total_committed[s2.button], 5);
        assert_eq!(s2.total_committed[1 - s2.button], 10);
        assert_eq!(s2.to_act, s2.button);
    }

    #[test]
    fn fold_requires_a_bet_and_the_cap_stops_raises() {
        let g = game(3);
        let s = g.initial_state(&mut TestRng(3));
        // UTG faces the blind: all three actions.
        assert_eq!(
            g.legal_actions(&s, s.to_act),
            vec![FOLD, CHECK_CALL, BET_RAISE]
        );
        // Raise until the 4-bet cap: raises 1 (blind) + 3 more = 4, then no raise is offered.
        let mut cur = s;
        let mut guard = 0;
        while g.legal_actions(&cur, cur.to_act).contains(&BET_RAISE) {
            let mut joint = vec![0; 3];
            joint[cur.to_act] = BET_RAISE;
            cur = g.step(&cur, &joint).next_state;
            guard += 1;
            assert!(guard < 10);
        }
        assert_eq!(cur.raises, 4);
        let legal = g.legal_actions(&cur, cur.to_act);
        assert_eq!(legal, vec![FOLD, CHECK_CALL]);
        // A seat with no bet to call may not fold.
        let g2 = game(2);
        let mut hu = g2.initial_state(&mut TestRng(4));
        let mut joint = vec![0; 2];
        joint[hu.to_act] = CHECK_CALL; // button completes the small blind
        hu = g2.step(&hu, &joint).next_state;
        assert_eq!(hu.to_call(hu.to_act), 0);
        assert_eq!(
            g2.legal_actions(&hu, hu.to_act),
            vec![CHECK_CALL, BET_RAISE]
        );
    }

    #[test]
    fn fold_out_pays_the_pot_without_showdown() {
        let g = game(3);
        let s = g.initial_state(&mut TestRng(5));
        let (sb, bb) = ((s.button + 1) % 3, (s.button + 2) % 3);
        // UTG folds, SB folds -> BB wins the blinds; nobody's cards matter.
        let mut joint = vec![0; 3];
        joint[s.to_act] = FOLD;
        let t1 = g.step(&s, &joint);
        assert!(!t1.terminal);
        let mut joint = vec![0; 3];
        joint[t1.next_state.to_act] = FOLD;
        let t2 = g.step(&t1.next_state, &joint);
        assert!(t2.terminal);
        assert_eq!(t2.events[bb], 5.0, "BB nets the small blind");
        assert_eq!(t2.events[sb], -5.0);
        assert_eq!(t2.events.iter().sum::<f64>(), 0.0);
    }

    #[test]
    fn side_pots_split_by_commitment_level() {
        // Hand-built: 3 players, stacks 200 but P0 committed 50 all-in, P1 and P2 200 each.
        // P0 holds the best hand, P1 the second best: P0 wins the main pot (3 x 50), P1 the
        // side pot (2 x 150).
        let g = game(3);
        let state = HoldemState {
            hole: vec![
                [card(12, 0), card(12, 1)], // AA - best
                [card(10, 0), card(10, 1)], // QQ
                [card(2, 0), card(3, 1)],   // junk
            ],
            board: vec![card(4, 2), card(6, 3), card(8, 0), card(9, 1), card(11, 2)],
            button: 0,
            street: Street::River,
            to_act: 0,
            stacks: vec![150, 0, 0],
            street_committed: vec![0, 0, 0],
            total_committed: vec![50, 200, 200],
            folded: vec![false, false, false],
            needs_action: vec![false, false, false],
            raises: 0,
        };
        let deltas = g.payouts(&state);
        assert_eq!(deltas[0], 100.0, "main pot 150 minus 50 in");
        assert_eq!(deltas[1], 100.0, "side pot 300 minus 200 in");
        assert_eq!(deltas[2], -200.0);
        assert_eq!(deltas.iter().sum::<f64>(), 0.0);
    }

    #[test]
    fn split_pots_give_odd_chips_to_the_earliest_seat_after_the_button() {
        // Both live seats play the board (identical hands); pot of 21 splits 11/10 with the odd
        // chip to the first seat clockwise from the small blind.
        let g = game(3);
        let state = HoldemState {
            hole: vec![
                [card(0, 0), card(1, 1)],
                [card(0, 2), card(1, 3)],
                [card(5, 0), card(6, 1)],
            ],
            board: vec![
                card(8, 0),
                card(9, 1),
                card(10, 2),
                card(11, 3),
                card(12, 0),
            ],
            button: 2,
            street: Street::River,
            to_act: 0,
            stacks: vec![193, 193, 200],
            street_committed: vec![0, 0, 0],
            total_committed: vec![10, 10, 1],
            folded: vec![false, false, true],
            needs_action: vec![false, false, false],
            raises: 0,
        };
        let deltas = g.payouts(&state);
        // Pot 21: winners split 10 each, odd chip to seat 0 (first after button 2).
        assert_eq!(deltas[0], 1.0);
        assert_eq!(deltas[1], 0.0);
        assert_eq!(deltas[2], -1.0);
    }

    #[test]
    fn street_reveals_declare_the_combination_space() {
        let g = game(2);
        let mut s = g.initial_state(&mut TestRng(6));
        // Button calls, BB checks -> the flop transition declares C(48, 3).
        let mut joint = vec![0; 2];
        joint[s.to_act] = CHECK_CALL;
        s = g.step(&s, &joint).next_state;
        let mut joint = vec![0; 2];
        joint[s.to_act] = CHECK_CALL;
        let t = g.step(&s, &joint);
        assert!(!t.terminal);
        assert_eq!(t.next_state.street, Street::Flop);
        assert_eq!(t.next_state.board.len(), 0, "the reveal is the chance");
        let dist = g.chance_outcomes(&s, &t).expect("flop declares chance");
        assert_eq!(dist.count(), binomial(48, 3));
        let dealt = g.apply_chance(&s, &t, 17_000);
        assert_eq!(dealt.board.len(), 3);
        assert!(dealt.board.windows(2).all(|w| w[0] < w[1]));
        g.validate_decoded_state(&dealt, false).unwrap();
    }

    #[test]
    fn random_hands_conserve_chips_and_terminate() {
        for n in [2, 3, 6, 9] {
            let g = game(n);
            let mut rng = TestRng(7 + n as u64);
            for _ in 0..40 {
                let mut s = g.initial_state(&mut rng);
                let mut guard = 0;
                loop {
                    let legal = g.legal_actions(&s, s.to_act);
                    assert!(!legal.is_empty(), "live states always offer an action");
                    let mut joint = vec![0; n];
                    joint[s.to_act] = legal[rng.below(legal.len())];
                    let t = step_env(&g, &s, &joint, &mut rng);
                    if t.terminal {
                        assert_eq!(t.events.iter().sum::<f64>(), 0.0, "zero-sum");
                        for (i, &d) in t.events.iter().enumerate() {
                            assert!(d >= -(g.stack as f64));
                            let _ = i;
                        }
                        break;
                    }
                    s = t.next_state;
                    g.validate_decoded_state(&s, false).unwrap();
                    guard += 1;
                    assert!(guard < 500, "hands terminate");
                }
            }
        }
    }

    #[test]
    fn all_in_runout_steps_through_forced_checks() {
        // Heads-up with stacks equal to the big blind: both all-in preflop; the remaining
        // streets must run out via single forced checks, then showdown.
        let g = TexasHoldem {
            num_players: 2,
            stack: 10,
            small_blind: 5,
            big_blind: 10,
        };
        let mut rng = TestRng(11);
        let mut s = g.initial_state(&mut rng);
        // Button (SB, 5 behind) calls all-in; BB already all-in from the blind.
        let mut joint = vec![0; 2];
        joint[s.to_act] = CHECK_CALL;
        let mut t = step_env(&g, &s, &joint, &mut rng);
        let mut forced = 0;
        while !t.terminal {
            s = t.next_state;
            let legal = g.legal_actions(&s, s.to_act);
            assert_eq!(
                legal,
                vec![CHECK_CALL],
                "runout offers only the forced check"
            );
            let mut joint = vec![0; 2];
            joint[s.to_act] = CHECK_CALL;
            t = step_env(&g, &s, &joint, &mut rng);
            forced += 1;
            assert!(forced <= 4);
        }
        assert_eq!(t.next_state.board.len(), 5, "full board dealt");
        assert_eq!(t.events.iter().sum::<f64>(), 0.0);
    }

    #[test]
    fn codec_round_trips_and_rejects_unsafe_states() {
        let g = game(3);
        let mut rng = TestRng(9);
        let s = g.initial_state(&mut rng);
        let bytes = g.encode(&s);
        let back = g.decode(&bytes).unwrap();
        assert_eq!(g.encode(&back), bytes);
        g.validate_decoded_state(&back, false).unwrap();

        let mut dup = s.clone();
        dup.board = vec![s.hole[0][0]];
        dup.street = Street::Done; // board length is free-form once done
        assert!(g
            .validate_decoded_state(&dup, true)
            .unwrap_err()
            .contains("dealt twice"));
        let mut leak = s.clone();
        leak.stacks[0] += 1;
        assert!(g
            .validate_decoded_state(&leak, false)
            .unwrap_err()
            .contains("conserve"));
        let mut wrong = s.clone();
        wrong.to_act = 7;
        assert!(g
            .validate_decoded_state(&wrong, false)
            .unwrap_err()
            .contains("seat index"));
        assert!(g
            .validate_decoded_state(&s, true)
            .unwrap_err()
            .contains("disagrees"));
    }
}
