//! Fixed-limit Texas hold'em with declared deals, side pots, and private observations.

use std::collections::HashSet;

use reinfors_core::game::{Actor, ChanceDist, Game, Transition};
use reinfors_core::Reward;
#[cfg(test)]
use reinfors_core::Rng;

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
    Done,
}

impl Street {
    fn board_len(self) -> usize {
        match self {
            Street::Preflop => 0,
            Street::Flop => 3,
            Street::Turn => 4,
            Street::River => 5,
            Street::Done => 5,
        }
    }

    fn index(self) -> usize {
        match self {
            Street::Preflop => 0,
            Street::Flop => 1,
            Street::Turn => 2,
            Street::River => 3,
            Street::Done => unreachable!("no betting happens on a finished hand"),
        }
    }

    fn index_or_done(self) -> usize {
        match self {
            Street::Done => 4,
            s => s.index(),
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

pub const FOLD: usize = 0;
pub const CHECK_CALL: usize = 1;
pub const BET_RAISE: usize = 2;
pub const HOLDEM_ACTIONS: usize = 3;

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct HoldemState {
    pub hole: Vec<[Card; 2]>,
    pub board: Vec<Card>,
    pub button: usize,
    pub street: Street,
    pub to_act: usize,
    pub stacks: Vec<u32>,
    pub street_committed: Vec<u32>,
    pub total_committed: Vec<u32>,
    pub folded: Vec<bool>,
    pub needs_action: Vec<bool>,
    pub raises: u8,
    // Perfect-recall observations need the sequence after street counters reset.
    pub history: Vec<Vec<(u8, u8)>>,
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

    fn remaining_deck(&self) -> Vec<Card> {
        let mut used = 0u64;
        for h in &self.hole {
            used |= (1 << h[0]) | (1 << h[1]);
        }
        for &c in &self.board {
            used |= 1 << c;
        }
        (0..DECK).filter(|&c| used & (1 << c) == 0).collect()
    }
}

pub struct HoldemReward {
    pub scale: f64,
}

impl Reward for HoldemReward {
    type Event = f64;
    fn step_reward(&self, e: &f64, _agent: usize) -> f64 {
        e * self.scale
    }
}

pub struct TexasHoldem {
    pub num_players: usize,
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
        if self.stack <= self.small_blind {
            return Err("stack must exceed the small blind".to_string());
        }
        if self.stack > 1 << 24 {
            return Err("stack must fit 2^24 chips".to_string());
        }
        Ok(())
    }

    fn bet_size(&self, street: Street) -> u32 {
        match street {
            Street::Preflop | Street::Flop => self.big_blind,
            _ => self.big_blind * 2,
        }
    }

    fn next_seat(&self, from: usize, pred: impl Fn(usize) -> bool) -> Option<usize> {
        (1..=self.num_players)
            .map(|d| (from + d) % self.num_players)
            .find(|&i| pred(i))
    }

    fn commit(state: &mut HoldemState, i: usize, amount: u32) {
        let paid = amount.min(state.stacks[i]);
        state.stacks[i] -= paid;
        state.street_committed[i] += paid;
        state.total_committed[i] += paid;
    }

    fn close_street(&self, state: &mut HoldemState) {
        state.street = state.street.next();
        state.street_committed.iter_mut().for_each(|c| *c = 0);
        state.raises = 0;
        state.needs_action.iter_mut().for_each(|b| *b = false);
    }

    fn open_betting(&self, state: &mut HoldemState) {
        let can_bet: Vec<usize> = (0..self.num_players)
            .filter(|&i| state.live(i) && state.stacks[i] > 0)
            .collect();
        // With fewer than two betting seats, leave the state actorless so reveals chain.
        if can_bet.len() >= 2 {
            for &i in &can_bet {
                state.needs_action[i] = true;
            }
            state.to_act = self
                .next_seat(state.button, |i| state.needs_action[i])
                .expect("betting reopened with someone to act");
        }
    }

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
        // Folding is only legal facing a bet, so the largest commitment is always live.
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
            // Odd chips go clockwise from the seat left of the button.
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
    type Event = f64;

    fn num_agents(&self) -> usize {
        self.num_players
    }

    fn action_count(&self) -> usize {
        HOLDEM_ACTIONS
    }

    fn perfect_information(&self) -> bool {
        false
    }

    fn actor(&self, state: &HoldemState) -> Actor {
        if !state.is_done() && !state.needs_action.iter().any(|&b| b) {
            return Actor::Chance;
        }
        Actor::Agent(state.to_act)
    }

    fn legal_actions(&self, state: &HoldemState, agent: usize) -> Vec<usize> {
        if state.is_done() || agent != state.to_act || !state.needs_action.iter().any(|&b| b) {
            return Vec::new();
        }
        let mut out = Vec::with_capacity(3);
        if state.to_call(agent) > 0 {
            out.push(FOLD);
        }
        out.push(CHECK_CALL);
        if state.raises < 4 && state.stacks[agent] > state.to_call(agent) {
            out.push(BET_RAISE);
        }
        out
    }

    fn step(&self, state: &HoldemState, actions: &[usize]) -> Transition<HoldemState, f64> {
        let me = state.to_act;
        let mut next = state.clone();
        let legal = self.legal_actions(state, me);
        let action = if legal.contains(&actions[me]) {
            actions[me]
        } else if state.to_call(me) > 0 {
            FOLD
        } else {
            CHECK_CALL
        };
        next.history[state.street.index()].push((me as u8, action as u8));

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
                for j in 0..self.num_players {
                    next.needs_action[j] = j != me && next.live(j) && next.stacks[j] > 0;
                }
            }
        }

        let live_count = (0..self.num_players).filter(|&i| next.live(i)).count();
        let terminal_events = |s: &HoldemState| self.payouts(s);
        if live_count == 1 {
            next.street = Street::Done;
            let events: Vec<Option<f64>> = terminal_events(&next).into_iter().map(Some).collect();
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
            return Transition::silent(next, self.num_players);
        }
        if next.street == Street::River {
            next.street = Street::Done;
            let events: Vec<Option<f64>> = terminal_events(&next).into_iter().map(Some).collect();
            return Transition {
                next_state: next,
                events,
                terminal: true,
            };
        }
        self.close_street(&mut next);
        Transition::silent(next, self.num_players)
    }

    fn information_states(&self) -> bool {
        true
    }

    fn information_state_key(&self, state: &HoldemState, agent: usize) -> Vec<u8> {
        let mut k = Vec::with_capacity(64);
        k.push(agent as u8);
        k.extend_from_slice(&state.hole[agent]);
        k.push(state.board.len() as u8);
        k.extend_from_slice(&state.board);
        k.push(state.button as u8);
        k.push(state.street.index_or_done() as u8);
        k.push(state.to_act as u8);
        k.push(state.raises);
        for i in 0..self.num_players {
            k.extend_from_slice(&state.stacks[i].to_le_bytes());
            k.extend_from_slice(&state.street_committed[i].to_le_bytes());
            k.push(u8::from(state.folded[i]) | (u8::from(state.needs_action[i]) << 1));
        }
        for street in &state.history {
            k.push(street.len() as u8);
            for &(seat, action) in street {
                k.push(seat);
                k.push(action);
            }
        }
        k
    }

    fn chance_node(&self, state: &HoldemState) -> ChanceDist {
        if state.button == self.num_players {
            return ChanceDist::Uniform(self.num_players);
        }
        if state.hole.len() < self.num_players {
            return ChanceDist::Uniform(binomial(state.remaining_deck().len(), 2));
        }
        let missing = state.street.board_len() - state.board.len();
        ChanceDist::Uniform(binomial(state.remaining_deck().len(), missing))
    }

    fn apply_chance_node(
        &self,
        state: &HoldemState,
        outcome: usize,
    ) -> Transition<HoldemState, f64> {
        let n = self.num_players;
        let mut next = state.clone();
        if next.button == n {
            next.button = outcome;
            return Transition::silent(next, n);
        }
        if next.hole.len() < n {
            let deck = next.remaining_deck();
            let pair = unrank_combination(outcome, deck.len(), 2);
            next.hole.push([deck[pair[0]], deck[pair[1]]]);
            if next.hole.len() == n {
                let (sb, bb) = if n == 2 {
                    (next.button, (next.button + 1) % n)
                } else {
                    ((next.button + 1) % n, (next.button + 2) % n)
                };
                Self::commit(&mut next, sb, self.small_blind);
                Self::commit(&mut next, bb, self.big_blind);
                next.raises = 1;
                for i in 0..n {
                    next.needs_action[i] = next.stacks[i] > 0;
                }
                next.to_act = if n == 2 { next.button } else { (bb + 1) % n };
                debug_assert!(next.needs_action[next.to_act]);
            }
            return Transition::silent(next, n);
        }
        let deck = next.remaining_deck();
        let missing = next.street.board_len() - next.board.len();
        for pos in unrank_combination(outcome, deck.len(), missing) {
            next.board.push(deck[pos]);
        }
        self.open_betting(&mut next);
        if next.needs_action.iter().any(|&b| b) {
            return Transition::silent(next, self.num_players);
        }
        if next.street == Street::River {
            next.street = Street::Done;
            let events: Vec<Option<f64>> = self.payouts(&next).into_iter().map(Some).collect();
            return Transition {
                next_state: next,
                events,
                terminal: true,
            };
        }
        next.street = next.street.next();
        Transition::silent(next, self.num_players)
    }

    fn initial_state(&self) -> HoldemState {
        let n = self.num_players;
        HoldemState {
            hole: Vec::new(),
            board: Vec::new(),
            button: n,
            street: Street::Preflop,
            to_act: 0,
            stacks: vec![self.stack; n],
            street_committed: vec![0; n],
            total_committed: vec![0; n],
            folded: vec![false; n],
            needs_action: vec![false; n],
            raises: 0,
            history: vec![Vec::new(); 4],
        }
    }

    fn truncation_horizon(&self) -> Option<usize> {
        None
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
        if !state.is_done() && !state.needs_action[state.to_act] {
            return Err("to_act owes no action".to_string());
        }
        if state.history.len() != 4 {
            return Err("history must cover the four betting streets".to_string());
        }
        for street in &state.history {
            if street.len() > 52 {
                return Err("implausible street history length".to_string());
            }
            for &(seat, action) in street {
                if seat as usize >= n || action as usize >= HOLDEM_ACTIONS {
                    return Err("history entry out of range".to_string());
                }
            }
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

    pub(super) fn deal(g: &TexasHoldem, rng: &mut dyn Rng) -> HoldemState {
        let mut s = g.initial_state();
        while matches!(g.actor(&s), Actor::Chance) {
            let outcome = g.chance_node(&s).draw(rng);
            let t = g.apply_chance_node(&s, outcome);
            assert!(!t.terminal, "the deal may not decide the game");
            assert!(
                t.events.iter().all(Option::is_none),
                "birth edges emit nothing"
            );
            s = t.next_state;
        }
        s
    }

    #[test]
    fn the_deal_is_a_declared_root_chain() {
        let g = game(3);
        let root = g.initial_state();
        assert!(matches!(g.actor(&root), Actor::Chance));
        assert_eq!(g.chance_node(&root).count(), 3, "button draw first");
        let s = deal(&g, &mut TestRng(41));
        assert_eq!(s.hole.len(), 3);
        assert!(s.button < 3);
        assert_eq!(s.raises, 1);
        assert!(matches!(g.actor(&s), Actor::Agent(_)));
        g.validate_decoded_state(&s, false).unwrap();
    }

    #[test]
    fn information_keys_hide_exactly_the_opponents_holes() {
        let g = game(3);
        let s = deal(&g, &mut TestRng(42));
        assert!(g.information_states());
        let k0 = g.information_state_key(&s, 0);
        let mut swapped = s.clone();
        swapped.hole.swap(1, 2);
        assert_eq!(k0, g.information_state_key(&swapped, 0));
        assert_ne!(
            g.information_state_key(&s, 1),
            g.information_state_key(&swapped, 1)
        );
        let mut own = s.clone();
        own.hole[0] = [51, 50];
        assert_ne!(k0, g.information_state_key(&own, 0));
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
        g = game(3);
        g.small_blind = 10;
        g.big_blind = 10;
        g.stack = 10;
        assert!(g.validate().is_err(), "no chips behind the small blind");
    }

    #[test]
    fn evaluator_orders_the_hand_categories() {
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
        let g = game(3);
        let mut rng = TestRng(1);
        let s = deal(&g, &mut rng);
        let (sb, bb) = ((s.button + 1) % 3, (s.button + 2) % 3);
        assert_eq!(s.total_committed[sb], 5);
        assert_eq!(s.total_committed[bb], 10);
        assert_eq!(s.to_act, (bb + 1) % 3);
        assert_eq!(s.raises, 1, "the big blind is the first bet");
        let g2 = game(2);
        let s2 = deal(&g2, &mut TestRng(2));
        assert_eq!(s2.total_committed[s2.button], 5);
        assert_eq!(s2.total_committed[1 - s2.button], 10);
        assert_eq!(s2.to_act, s2.button);
    }

    #[test]
    fn fold_requires_a_bet_and_the_cap_stops_raises() {
        let g = game(3);
        let s = deal(&g, &mut TestRng(3));
        assert_eq!(
            g.legal_actions(&s, s.to_act),
            vec![FOLD, CHECK_CALL, BET_RAISE]
        );
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
        let g2 = game(2);
        let mut hu = deal(&g2, &mut TestRng(4));
        let mut joint = vec![0; 2];
        joint[hu.to_act] = CHECK_CALL;
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
        let s = deal(&g, &mut TestRng(5));
        let (sb, bb) = ((s.button + 1) % 3, (s.button + 2) % 3);
        let mut joint = vec![0; 3];
        joint[s.to_act] = FOLD;
        let t1 = g.step(&s, &joint);
        assert!(!t1.terminal);
        let mut joint = vec![0; 3];
        joint[t1.next_state.to_act] = FOLD;
        let t2 = g.step(&t1.next_state, &joint);
        assert!(t2.terminal);
        assert_eq!(t2.events[bb], Some(5.0), "BB nets the small blind");
        assert_eq!(t2.events[sb], Some(-5.0));
        assert_eq!(t2.events.iter().flatten().sum::<f64>(), 0.0);
    }

    #[test]
    fn side_pots_split_by_commitment_level() {
        let g = game(3);
        let state = HoldemState {
            hole: vec![
                [card(12, 0), card(12, 1)],
                [card(10, 0), card(10, 1)],
                [card(2, 0), card(3, 1)],
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
            history: vec![Vec::new(); 4],
        };
        let deltas = g.payouts(&state);
        assert_eq!(deltas[0], 100.0, "main pot 150 minus 50 in");
        assert_eq!(deltas[1], 100.0, "side pot 300 minus 200 in");
        assert_eq!(deltas[2], -200.0);
        assert_eq!(deltas.iter().sum::<f64>(), 0.0);
    }

    #[test]
    fn split_pots_give_odd_chips_to_the_earliest_seat_after_the_button() {
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
            history: vec![Vec::new(); 4],
        };
        let deltas = g.payouts(&state);
        assert_eq!(deltas[0], 1.0);
        assert_eq!(deltas[1], 0.0);
        assert_eq!(deltas[2], -1.0);
    }

    #[test]
    fn street_reveals_declare_the_combination_space() {
        let g = game(2);
        let mut s = deal(&g, &mut TestRng(6));
        let mut joint = vec![0; 2];
        joint[s.to_act] = CHECK_CALL;
        s = g.step(&s, &joint).next_state;
        let mut joint = vec![0; 2];
        joint[s.to_act] = CHECK_CALL;
        let t = g.step(&s, &joint);
        assert!(!t.terminal);
        let node = &t.next_state;
        assert_eq!(node.street, Street::Flop);
        assert_eq!(node.board.len(), 0, "the reveal is the chance");
        assert!(matches!(g.actor(node), Actor::Chance));
        assert!(g.legal_actions(node, node.to_act).is_empty());
        assert_eq!(g.chance_node(node).count(), binomial(48, 3));
        let rt = g.apply_chance_node(node, 17_000);
        assert!(!rt.terminal);
        let dealt = rt.next_state;
        assert_eq!(dealt.board.len(), 3);
        assert!(dealt.board.windows(2).all(|w| w[0] < w[1]));
        assert!(
            matches!(g.actor(&dealt), Actor::Agent(_)),
            "betting reopens"
        );
        g.validate_decoded_state(&dealt, false).unwrap();
    }

    #[test]
    fn random_hands_conserve_chips_and_terminate() {
        for n in [2, 3, 6, 9] {
            let g = game(n);
            let mut rng = TestRng(7 + n as u64);
            for _ in 0..40 {
                let mut s = deal(&g, &mut rng);
                let mut guard = 0;
                loop {
                    let legal = g.legal_actions(&s, s.to_act);
                    assert!(!legal.is_empty(), "live states always offer an action");
                    let mut joint = vec![0; n];
                    joint[s.to_act] = legal[rng.below(legal.len())];
                    let t = step_env(&g, &s, &joint, &mut rng);
                    if t.terminal {
                        let deltas: Vec<f64> = t.trace.iter().map(|(_, d)| *d).collect();
                        assert_eq!(deltas.iter().sum::<f64>(), 0.0, "zero-sum");
                        for &d in &deltas {
                            assert!(d >= -(g.stack as f64));
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
    fn all_in_runout_chains_the_reveals_without_actions() {
        let g = TexasHoldem {
            num_players: 2,
            stack: 10,
            small_blind: 5,
            big_blind: 10,
        };
        let mut rng = TestRng(11);
        let s = deal(&g, &mut rng);
        assert!(
            !s.needs_action[(s.button + 1) % 2],
            "the all-in big blind owes no action"
        );
        let mut joint = vec![0; 2];
        joint[s.to_act] = CHECK_CALL;
        let t = step_env(&g, &s, &joint, &mut rng);
        assert!(t.terminal, "the call settles the hand in one transition");
        assert_eq!(t.next_state.board.len(), 5, "full board dealt");
        assert_eq!(t.trace.iter().map(|(_, d)| d).sum::<f64>(), 0.0);
        let all: Vec<_> = t.next_state.history.concat();
        assert_eq!(all, vec![(s.to_act as u8, CHECK_CALL as u8)]);
    }

    #[test]
    fn history_records_the_betting_sequence() {
        let g = game(2);
        let s = deal(&g, &mut TestRng(12));
        let play = |actions: &[usize]| {
            let mut cur = s.clone();
            for &a in actions {
                let mut joint = vec![0; 2];
                joint[cur.to_act] = a;
                cur = g.step(&cur, &joint).next_state;
            }
            cur
        };
        let a = play(&[BET_RAISE, CHECK_CALL]);
        let b = play(&[CHECK_CALL, BET_RAISE, CHECK_CALL]);
        assert_eq!(a.total_committed, b.total_committed);
        assert_eq!(a.street, b.street);
        assert_ne!(a.history, b.history, "the sequences stay distinguishable");
        let sb = s.to_act as u8;
        let bb = 1 - sb;
        assert_eq!(
            a.history[0],
            vec![(sb, BET_RAISE as u8), (bb, CHECK_CALL as u8)]
        );
    }

    #[test]
    fn codec_round_trips_and_rejects_unsafe_states() {
        let g = game(3);
        let mut rng = TestRng(9);
        let s = deal(&g, &mut rng);
        let bytes = g.encode(&s);
        let back = g.decode(&bytes).unwrap();
        assert_eq!(g.encode(&back), bytes);
        g.validate_decoded_state(&back, false).unwrap();

        let mut dup = s.clone();
        dup.board = vec![s.hole[0][0]];
        dup.street = Street::Done;
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
        let mut idle = s.clone();
        idle.needs_action.iter_mut().for_each(|b| *b = false);
        assert!(g
            .validate_decoded_state(&idle, false)
            .unwrap_err()
            .contains("owes no action"));
        let mut hist = s.clone();
        hist.history[0].push((7, 1));
        assert!(g
            .validate_decoded_state(&hist, false)
            .unwrap_err()
            .contains("history entry"));
    }
}

pub struct HoldemEgocentric {
    pub num_players: usize,
    pub stack: u32,
}

impl HoldemEgocentric {
    fn channels(&self) -> usize {
        11 + 2 * (self.num_players - 1) + 10
    }

    fn history_base(&self) -> usize {
        11 + 2 * (self.num_players - 1)
    }
}

const PLANE: usize = 4 * 13;

fn set_card(obs: &mut [f32], ch: usize, c: Card) {
    obs[ch * PLANE + card_suit(c) as usize * 13 + card_rank(c) as usize] = 1.0;
}

fn fill(obs: &mut [f32], ch: usize, v: f32) {
    obs[ch * PLANE..(ch + 1) * PLANE].fill(v.clamp(0.0, 1.0));
}

impl reinfors_core::ActionView for HoldemEgocentric {}

impl reinfors_core::StateEncoder for HoldemEgocentric {
    type State = HoldemState;

    fn encode(&self, s: &HoldemState, agent: usize) -> Vec<f32> {
        let n = self.num_players;
        let total_chips = (self.stack as f32) * n as f32;
        let mut obs = vec![0.0f32; self.channels() * PLANE];
        set_card(&mut obs, 0, s.hole[agent][0]);
        set_card(&mut obs, 0, s.hole[agent][1]);
        for &c in &s.board {
            set_card(&mut obs, 1, c);
        }
        let pot: u32 = s.total_committed.iter().sum();
        fill(&mut obs, 2, pot as f32 / total_chips);
        fill(&mut obs, 3, s.to_call(agent) as f32 / self.stack as f32);
        fill(&mut obs, 4, s.stacks[agent] as f32 / self.stack as f32);
        let street_ch = match s.street {
            Street::Preflop => 5,
            Street::Flop => 6,
            Street::Turn => 7,
            Street::River | Street::Done => 8,
        };
        fill(&mut obs, street_ch, 1.0);
        fill(&mut obs, 9, f32::from(s.raises) / 4.0);
        fill(&mut obs, 10, ((s.button + n - agent) % n) as f32 / n as f32);
        for d in 1..n {
            let seat = (agent + d) % n;
            let base = 11 + 2 * (d - 1);
            fill(&mut obs, base, if s.folded[seat] { 0.0 } else { 1.0 });
            fill(
                &mut obs,
                base + 1,
                s.stacks[seat] as f32 / self.stack as f32,
            );
        }
        for (st, street) in s.history.iter().enumerate() {
            let (seat_ch, act_ch) = (
                self.history_base() + 2 * st,
                self.history_base() + 2 * st + 1,
            );
            for (k, &(seat, action)) in street.iter().enumerate() {
                let rel = (seat as usize + n - agent) % n;
                obs[seat_ch * PLANE + k] = (rel + 1) as f32 / n as f32;
                obs[act_ch * PLANE + k] = (action + 1) as f32 / 3.0;
            }
        }
        if let Some(&turn) = s.board.get(3) {
            set_card(&mut obs, self.history_base() + 8, turn);
        }
        if let Some(&river) = s.board.get(4) {
            set_card(&mut obs, self.history_base() + 9, river);
        }
        obs
    }

    fn obs_shape(&self) -> (usize, usize, usize) {
        (self.channels(), 4, 13)
    }

    fn observation_space(&self) -> reinfors_core::Space {
        let (c, h, w) = self.obs_shape();
        reinfors_core::Space::unit_box(vec![c, h, w])
    }
}

#[cfg(test)]
mod encoder_tests {
    use super::*;
    use reinfors_core::StateEncoder;

    fn card(rank: u8, suit: u8) -> Card {
        rank * 4 + suit
    }

    #[test]
    fn each_seat_sees_only_its_own_holes() {
        let g = TexasHoldem {
            num_players: 3,
            stack: 200,
            small_blind: 5,
            big_blind: 10,
        };
        struct R(u64);
        impl Rng for R {
            fn below(&mut self, n: usize) -> usize {
                self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
                (self.0 >> 33) as usize % n.max(1)
            }
            fn unit(&mut self) -> f64 {
                0.5
            }
        }
        let s = super::tests::deal(&g, &mut R(4));
        let enc = HoldemEgocentric {
            num_players: 3,
            stack: 200,
        };
        for agent in 0..3 {
            let obs = enc.encode(&s, agent);
            assert_eq!(obs.len(), (11 + 4 + 10) * PLANE);
            let holes: f32 = obs[..PLANE].iter().sum();
            assert_eq!(holes, 2.0, "exactly the agent's two cards");
            let own = &s.hole[agent];
            for &c in own {
                let idx = card_suit(c) as usize * 13 + card_rank(c) as usize;
                assert_eq!(obs[idx], 1.0);
            }
            for other in 0..3 {
                if other == agent {
                    continue;
                }
                for &c in &s.hole[other] {
                    let idx = card_suit(c) as usize * 13 + card_rank(c) as usize;
                    if !own.contains(&c) {
                        assert_eq!(obs[idx], 0.0, "hidden card leaked");
                    }
                }
            }
        }
    }

    #[test]
    fn scalars_and_relative_seats_populate() {
        let g = TexasHoldem {
            num_players: 3,
            stack: 200,
            small_blind: 5,
            big_blind: 10,
        };
        struct R;
        impl Rng for R {
            fn below(&mut self, _n: usize) -> usize {
                0
            }
            fn unit(&mut self) -> f64 {
                0.5
            }
        }
        let s = super::tests::deal(&g, &mut R);
        let enc = HoldemEgocentric {
            num_players: 3,
            stack: 200,
        };
        let obs = enc.encode(&s, s.to_act);
        assert!((obs[2 * PLANE] - 15.0 / 600.0).abs() < 1e-6, "pot = blinds");
        assert_eq!(obs[5 * PLANE], 1.0, "preflop one-hot");
        assert_eq!(obs[9 * PLANE], 0.25, "the blind counts as the first bet");
        for d in 0..2 {
            assert_eq!(obs[(11 + 2 * d) * PLANE], 1.0);
            assert!(obs[(11 + 2 * d + 1) * PLANE] > 0.9);
        }
    }

    #[test]
    fn history_planes_distinguish_equal_commitment_sequences() {
        let g = TexasHoldem {
            num_players: 2,
            stack: 200,
            small_blind: 5,
            big_blind: 10,
        };
        struct R;
        impl Rng for R {
            fn below(&mut self, _n: usize) -> usize {
                0
            }
            fn unit(&mut self) -> f64 {
                0.5
            }
        }
        let s = super::tests::deal(&g, &mut R);
        let play = |actions: &[usize]| {
            let mut cur = s.clone();
            for &a in actions {
                let mut joint = vec![0; 2];
                joint[cur.to_act] = a;
                cur = g.step(&cur, &joint).next_state;
            }
            cur
        };
        let a = g
            .apply_chance_node(&play(&[BET_RAISE, CHECK_CALL]), 1234)
            .next_state;
        let b = g
            .apply_chance_node(&play(&[CHECK_CALL, BET_RAISE, CHECK_CALL]), 1234)
            .next_state;
        let enc = HoldemEgocentric {
            num_players: 2,
            stack: 200,
        };
        let viewer = a.to_act;
        assert_eq!(b.to_act, viewer, "same seat opens the flop in both lines");
        let (oa, ob) = (enc.encode(&a, viewer), enc.encode(&b, viewer));
        let base = enc.history_base();
        assert_ne!(
            oa[base * PLANE..(base + 2) * PLANE],
            ob[base * PLANE..(base + 2) * PLANE],
            "the aggressor is visible in the history planes"
        );
        assert_eq!(oa[base * PLANE], 1.0, "relative seat (1+1)/2");
        assert_eq!(oa[(base + 1) * PLANE], 1.0, "raise = (2+1)/3");
        assert_eq!(
            oa[..base * PLANE],
            ob[..base * PLANE],
            "identical commitments outside the history"
        );
    }

    #[test]
    fn reveal_chronology_planes_distinguish_swapped_turn_and_river() {
        let mk = |board: Vec<Card>| HoldemState {
            hole: vec![[card(0, 0), card(1, 1)], [card(0, 2), card(1, 3)]],
            board,
            button: 0,
            street: Street::River,
            to_act: 1,
            stacks: vec![190, 190],
            street_committed: vec![0, 0],
            total_committed: vec![10, 10],
            folded: vec![false, false],
            needs_action: vec![false, true],
            raises: 0,
            history: vec![Vec::new(); 4],
        };
        let (x, y) = (card(7, 0), card(11, 2));
        let flop = [card(2, 1), card(4, 3), card(9, 0)];
        let a = mk(vec![flop[0], flop[1], flop[2], x, y]);
        let b = mk(vec![flop[0], flop[1], flop[2], y, x]);
        let enc = HoldemEgocentric {
            num_players: 2,
            stack: 200,
        };
        let (oa, ob) = (enc.encode(&a, 1), enc.encode(&b, 1));
        assert_ne!(oa, ob, "reveal order is public information");
        assert_eq!(
            oa[PLANE..2 * PLANE],
            ob[PLANE..2 * PLANE],
            "aggregate board plane matches"
        );
        let turn_ch = enc.history_base() + 8;
        let idx = |c: Card| card_suit(c) as usize * 13 + card_rank(c) as usize;
        assert_eq!(oa[turn_ch * PLANE + idx(x)], 1.0);
        assert_eq!(ob[turn_ch * PLANE + idx(y)], 1.0);
    }
}
