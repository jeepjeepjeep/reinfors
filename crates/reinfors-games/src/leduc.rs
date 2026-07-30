//! Leduc hold'em — the standard small imperfect-information benchmark (6 cards, two betting
//! rounds, a few hundred information sets). Rules and action ids match OpenSpiel's
//! `leduc_poker`: both players ante 1; each is dealt one private card from a 6-card deck (3
//! ranks x 2 suits, card id = rank * 2 + suit); round 1 betting (raise size 2), then one
//! public card, then round 2 betting (raise size 4); at most 2 raises per round; player 0
//! opens both rounds; FOLD is legal only when facing a raise. Showdown: a private card
//! pairing the public card wins, otherwise higher rank, equal ranks split.
//!
//! Chance is fully declared (`all_chance_declared`): the two deals are root chance nodes and
//! the public reveal is an interior chance node (`chance_nodes` = true). The state is minimal
//! — cards, public card, per-round histories; pots, the actor, round, and terminal status are
//! all derived, so decode validation is pure grammar checking.

use reinfors_core::game::{Actor, ChanceDist, Game, Rng, Transition};

pub const FOLD: usize = 0;
pub const CALL: usize = 1;
pub const RAISE: usize = 2;

pub const LEDUC_DECK: u8 = 6;

fn rank(card: u8) -> u8 {
    card / 2
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct LeducState {
    /// Dealt private cards, player order; grows 0 -> 2 during the birth chain.
    pub cards: Vec<u8>,
    /// The public card, revealed by the interior chance node between the rounds.
    pub public: Option<u8>,
    /// Per-round public action histories, player 0 first in both.
    pub history: [Vec<u8>; 2],
}

/// A betting round is over once a call closes it: any history of length >= 2 ending in CALL
/// (check-check, raise-call, check-raise-call, ...).
fn round_over(h: &[u8]) -> bool {
    h.len() >= 2 && *h.last().unwrap() == CALL as u8
}

impl LeducState {
    /// Public terminality accessor (bindings).
    pub fn is_terminal_pub(&self) -> bool {
        self.is_terminal()
    }

    /// Public round accessor (bindings).
    pub fn round_pub(&self) -> usize {
        self.round()
    }

    fn folder(&self) -> Option<usize> {
        for h in &self.history {
            if let Some(pos) = h.iter().position(|&a| a == FOLD as u8) {
                return Some(pos % 2);
            }
        }
        None
    }

    fn is_terminal(&self) -> bool {
        self.folder().is_some() || round_over(&self.history[1])
    }

    /// The active betting round (0 or 1). Round 1 begins once round 0's betting closed —
    /// including the reveal-pending window where `public` is still `None`.
    fn round(&self) -> usize {
        usize::from(round_over(&self.history[0]))
    }

    /// Per-player pot contribution, derived by replaying the histories (ante 1; a CALL
    /// matches, a RAISE matches then adds the round's raise size).
    fn contributions(&self) -> [i64; 2] {
        let mut pot = [1i64, 1];
        for (r, h) in self.history.iter().enumerate() {
            let size = if r == 0 { 2 } else { 4 };
            for (i, &a) in h.iter().enumerate() {
                let p = i % 2;
                match a as usize {
                    CALL => pot[p] = pot[1 - p].max(pot[p]),
                    RAISE => pot[p] = pot[1 - p].max(pot[p]) + size,
                    _ => {} // FOLD commits nothing
                }
            }
        }
        pot
    }
}

pub struct LeducPoker;

impl LeducPoker {
    fn remaining(&self, state: &LeducState) -> Vec<u8> {
        (0..LEDUC_DECK)
            .filter(|c| !state.cards.contains(c) && state.public != Some(*c))
            .collect()
    }

    fn payouts(&self, state: &LeducState) -> Vec<f64> {
        let pot = state.contributions();
        let winner = match state.folder() {
            Some(f) => 1 - f,
            None => {
                let public = state.public.expect("showdown requires the public card");
                let pairs = |p: usize| rank(state.cards[p]) == rank(public);
                if pairs(0) != pairs(1) {
                    usize::from(pairs(1))
                } else if rank(state.cards[0]) == rank(state.cards[1]) {
                    return vec![0.0, 0.0]; // split: equal ranks, equal contributions
                } else {
                    usize::from(rank(state.cards[1]) > rank(state.cards[0]))
                }
            }
        };
        let mut deltas = vec![0.0; 2];
        deltas[winner] = pot[1 - winner] as f64;
        deltas[1 - winner] = -(pot[1 - winner] as f64);
        deltas
    }
}

impl Game for LeducPoker {
    type State = LeducState;
    type Event = f64; // per-player chip delta at the terminal tick, 0 elsewhere

    fn num_agents(&self) -> usize {
        2
    }

    fn action_count(&self) -> usize {
        3
    }

    fn perfect_information(&self) -> bool {
        false // the opponent's card is hidden
    }

    fn chance_nodes(&self) -> bool {
        true // the public reveal is an interior chance node
    }

    fn all_chance_declared(&self) -> bool {
        true // deals are root chance nodes, the reveal interior; initial_state draws nothing
    }

    fn information_states(&self) -> bool {
        true
    }

    fn information_state_key(&self, state: &LeducState, agent: usize) -> Vec<u8> {
        let mut k = Vec::with_capacity(8 + state.history[0].len() + state.history[1].len());
        k.push(agent as u8);
        k.push(state.cards[agent]);
        k.push(state.public.map_or(255, |c| c));
        for h in &state.history {
            k.push(h.len() as u8);
            k.extend_from_slice(h);
        }
        k
    }

    fn actor(&self, state: &LeducState) -> Actor {
        if state.cards.len() < 2 {
            return Actor::Chance; // the deal (birth chain)
        }
        if !state.is_terminal() && state.round() == 1 && state.public.is_none() {
            return Actor::Chance; // the public reveal (interior)
        }
        Actor::Agent(state.history[state.round()].len() % 2)
    }

    fn chance_node(&self, state: &LeducState) -> ChanceDist {
        ChanceDist::Uniform(self.remaining(state).len())
    }

    fn apply_chance_node(&self, state: &LeducState, outcome: usize) -> Transition<LeducState, f64> {
        let mut next = state.clone();
        let card = self.remaining(state)[outcome];
        if next.cards.len() < 2 {
            next.cards.push(card);
        } else {
            next.public = Some(card);
        }
        Transition::silent(next, 2)
    }

    fn legal_actions(&self, state: &LeducState, agent: usize) -> Vec<usize> {
        if state.cards.len() < 2 || state.is_terminal() {
            return Vec::new();
        }
        let round = state.round();
        if (round == 1 && state.public.is_none()) || agent != state.history[round].len() % 2 {
            return Vec::new();
        }
        let pot = state.contributions();
        let raises = state.history[round]
            .iter()
            .filter(|&&a| a == RAISE as u8)
            .count();
        let mut out = Vec::with_capacity(3);
        if pot[1 - agent] > pot[agent] {
            out.push(FOLD); // folding with a free check available is illegal
        }
        out.push(CALL);
        if raises < 2 {
            out.push(RAISE);
        }
        out
    }

    fn step(&self, state: &LeducState, actions: &[usize]) -> Transition<LeducState, f64> {
        let round = state.round();
        let me = state.history[round].len() % 2;
        let legal = self.legal_actions(state, me);
        // Backstop for direct core callers: an illegal action folds when facing a raise, else
        // checks.
        let action = if legal.contains(&actions[me]) {
            actions[me]
        } else if legal.contains(&FOLD) {
            FOLD
        } else {
            CALL
        };
        let mut next = state.clone();
        next.history[round].push(action as u8);
        let terminal = next.is_terminal();
        let events = if terminal {
            self.payouts(&next).into_iter().map(Some).collect()
        } else {
            vec![None; 2]
        };
        // A closed round 0 leaves the state at the reveal chance node (public still None) —
        // the framework draws it before any agent sees the state.
        Transition {
            next_state: next,
            events,
            terminal,
        }
    }

    fn initial_state(&self, _rng: &mut dyn Rng) -> LeducState {
        // Draws nothing (`all_chance_declared`): the empty deal is the birth-chain root.
        LeducState {
            cards: Vec::new(),
            public: None,
            history: [Vec::new(), Vec::new()],
        }
    }
}

impl reinfors_core::StateCodec for LeducPoker {
    type State = LeducState;

    fn encode(&self, s: &LeducState) -> Vec<u8> {
        crate::codec_util::serde_encode(1, s)
    }

    fn decode(&self, bytes: &[u8]) -> Result<LeducState, String> {
        crate::codec_util::serde_decode(1, bytes)
    }

    fn validate_decoded_state(&self, state: &LeducState, done: bool) -> Result<(), String> {
        if state.cards.len() != 2 {
            return Err("both cards must be dealt (birth chains are transient)".to_string());
        }
        let mut seen: Vec<u8> = state.cards.clone();
        seen.extend(state.public);
        for &c in &seen {
            if c >= LEDUC_DECK {
                return Err(format!("card id {c} out of range"));
            }
        }
        seen.sort_unstable();
        if seen.windows(2).any(|w| w[0] == w[1]) {
            return Err("a card appears twice".to_string());
        }
        for (r, h) in state.history.iter().enumerate() {
            if h.len() > 4 || h.iter().any(|&a| a > RAISE as u8) {
                return Err("malformed action history".to_string());
            }
            // Every action must have been legal at its point: replay the grammar.
            for (i, &a) in h.iter().enumerate() {
                let prefix = LeducState {
                    cards: state.cards.clone(),
                    public: state.public,
                    history: if r == 0 {
                        [h[..i].to_vec(), Vec::new()]
                    } else {
                        [state.history[0].clone(), h[..i].to_vec()]
                    },
                };
                if prefix.is_terminal() || prefix.round() != r {
                    return Err("actions continue past a closed round".to_string());
                }
                if !self.legal_actions(&prefix, i % 2).contains(&(a as usize)) {
                    return Err("history contains an illegal action".to_string());
                }
            }
        }
        if !state.history[1].is_empty() && !round_over(&state.history[0]) {
            return Err("round 2 actions before round 1 closed".to_string());
        }
        if state.round() == 1 && state.public.is_none() && !state.is_terminal() {
            return Err(
                "live round-2 state without the public card (reveals are transient)".to_string(),
            );
        }
        if state.public.is_some() && state.round() == 0 {
            return Err("public card revealed before round 1 closed".to_string());
        }
        if state.is_terminal() != done {
            return Err(format!(
                "history terminality disagrees with envelope done {done}"
            ));
        }
        Ok(())
    }
}

// ===================== Observation =====================

/// `(21, 1, 1)`: own card one-hot (6) + public card one-hot (6) + round-2 flag (1) + 2 rounds
/// x 4 history slots holding `(action + 1) / 3`. Exactly the information set (pinned by test
/// against the key).
pub struct LeducEncoder;

impl reinfors_core::ActionView for LeducEncoder {}

impl reinfors_core::StateEncoder for LeducEncoder {
    type State = LeducState;

    fn encode(&self, s: &LeducState, agent: usize) -> Vec<f32> {
        let mut obs = vec![0.0f32; 21];
        obs[s.cards[agent] as usize] = 1.0;
        if let Some(p) = s.public {
            obs[6 + p as usize] = 1.0;
        }
        if s.round() == 1 {
            obs[12] = 1.0;
        }
        for (r, h) in s.history.iter().enumerate() {
            for (i, &a) in h.iter().enumerate() {
                obs[13 + 4 * r + i] = (a + 1) as f32 / 3.0;
            }
        }
        obs
    }

    fn obs_shape(&self) -> (usize, usize, usize) {
        (21, 1, 1)
    }

    fn observation_space(&self) -> reinfors_core::Space {
        reinfors_core::Space::unit_box(vec![21, 1, 1])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use reinfors_core::game::step_env;
    use reinfors_core::{StateCodec, StateEncoder};

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

    fn dealt(c0: u8, c1: u8) -> LeducState {
        LeducState {
            cards: vec![c0, c1],
            public: None,
            history: [Vec::new(), Vec::new()],
        }
    }

    fn play(g: &LeducPoker, mut s: LeducState, actions: &[usize]) -> Transition<LeducState, f64> {
        let mut last = None;
        for &a in actions {
            let me = match g.actor(&s) {
                Actor::Agent(p) => p,
                Actor::Chance => panic!("test lines must realize chance explicitly"),
                Actor::Simultaneous => unreachable!(),
            };
            let mut joint = vec![0; 2];
            joint[me] = a;
            let t = g.step(&s, &joint);
            s = t.next_state.clone();
            last = Some(t);
        }
        last.unwrap()
    }

    #[test]
    fn betting_follows_the_pyspiel_grammar() {
        let g = LeducPoker;
        let s = dealt(0, 2);
        // Player 0 opens; no fold with a free check (pinned against pyspiel).
        assert_eq!(g.legal_actions(&s, 0), vec![CALL, RAISE]);
        // Facing a raise: all three actions until the 2-raise cap.
        let r1 = g.step(&s, &[RAISE, 0]).next_state;
        assert_eq!(g.legal_actions(&r1, 1), vec![FOLD, CALL, RAISE]);
        let r2 = g.step(&r1, &[0, RAISE]).next_state;
        assert_eq!(g.legal_actions(&r2, 0), vec![FOLD, CALL], "cap reached");
    }

    #[test]
    fn showdown_and_fold_pay_the_derived_pots() {
        let g = LeducPoker;
        // raise-fold in round 1: p1 folds, p0 wins p1's ante.
        let t = play(&g, dealt(0, 2), &[RAISE, FOLD]);
        assert!(t.terminal);
        assert_eq!(t.events, vec![Some(1.0), Some(-1.0)]);
        // check-check, reveal, check-check: showdown for the antes. Public pairs p1.
        let mut s = play(&g, dealt(0, 2), &[CALL, CALL]).next_state;
        assert!(matches!(g.actor(&s), Actor::Chance), "reveal pending");
        let reveal = g.remaining(&s).iter().position(|&c| c == 3).unwrap();
        s = g.apply_chance_node(&s, reveal).next_state; // public = 3 pairs p1's card 2
        let t = play(&g, s, &[CALL, CALL]);
        assert!(t.terminal);
        assert_eq!(
            t.events,
            vec![Some(-1.0), Some(1.0)],
            "pair beats high card"
        );
        // raise-call round 1, reveal, raise-raise-call round 2 (both raises hit the cap):
        let mut s = play(&g, dealt(4, 0), &[RAISE, CALL]).next_state;
        let reveal = g.remaining(&s).iter().position(|&c| c == 2).unwrap();
        s = g.apply_chance_node(&s, reveal).next_state;
        let t = play(&g, s, &[RAISE, RAISE, CALL]);
        assert!(t.terminal);
        // Contributions: ante 1 + round-1 raise 2 + two round-2 raises of 4 -> 11 each;
        // K high beats J.
        assert_eq!(t.events, vec![Some(11.0), Some(-11.0)]);
        // Split: equal ranks at showdown.
        let mut s = play(&g, dealt(0, 1), &[CALL, CALL]).next_state;
        s = g.apply_chance_node(&s, 0).next_state;
        let t = play(&g, s, &[CALL, CALL]);
        assert_eq!(t.events, vec![Some(0.0), Some(0.0)]);
    }

    #[test]
    fn full_hands_realize_and_conserve_chips() {
        let g = LeducPoker;
        let mut rng = TestRng(7);
        for _ in 0..200 {
            // Birth chain via the realization path (step_env handles interior reveals).
            let mut s = g.initial_state(&mut rng);
            while matches!(g.actor(&s), Actor::Chance) {
                let o = g.chance_node(&s).draw(&mut rng);
                s = g.apply_chance_node(&s, o).next_state;
            }
            let mut guard = 0;
            loop {
                let me = match g.actor(&s) {
                    Actor::Agent(p) => p,
                    _ => unreachable!("post-birth non-terminal states are decision states"),
                };
                let legal = g.legal_actions(&s, me);
                assert!(!legal.is_empty());
                let mut joint = vec![0; 2];
                joint[me] = legal[rng.below(legal.len())];
                let t = step_env(&g, &s, &joint, &mut rng);
                if t.terminal {
                    assert_eq!(t.trace.iter().map(|(_, d)| d).sum::<f64>(), 0.0, "zero-sum");
                    break;
                }
                s = t.next_state;
                g.validate_decoded_state(&s, false).unwrap();
                guard += 1;
                assert!(guard < 20, "hands terminate");
            }
        }
    }

    #[test]
    fn keys_and_observations_carry_the_same_information() {
        let g = LeducPoker;
        let enc = LeducEncoder;
        let line = |c0: u8, c1: u8| {
            let mut s = play(&g, dealt(c0, c1), &[RAISE, CALL]).next_state;
            let reveal = g.remaining(&s).iter().position(|&c| c == 5).unwrap();
            s = g.apply_chance_node(&s, reveal).next_state;
            play(&g, s, &[RAISE]).next_state
        };
        let (a, b) = (line(0, 2), line(0, 3));
        assert_eq!(
            g.information_state_key(&a, 0),
            g.information_state_key(&b, 0)
        );
        assert_eq!(enc.encode(&a, 0), enc.encode(&b, 0));
        assert_ne!(
            g.information_state_key(&a, 1),
            g.information_state_key(&b, 1)
        );
        assert_ne!(enc.encode(&a, 1), enc.encode(&b, 1));
    }

    #[test]
    fn codec_round_trips_and_rejects_unsafe_states() {
        let g = LeducPoker;
        let s = play(&g, dealt(0, 2), &[RAISE]).next_state;
        let back = g.decode(&g.encode(&s)).unwrap();
        assert_eq!(back, s);
        g.validate_decoded_state(&back, false).unwrap();
        let mut dup = s.clone();
        dup.public = Some(0);
        assert!(g.validate_decoded_state(&dup, false).is_err());
        let mut bad = s.clone();
        bad.history[0] = vec![FOLD as u8]; // fold with a free check: illegal in the grammar
        assert!(g
            .validate_decoded_state(&bad, true)
            .unwrap_err()
            .contains("illegal action"));
        let mut early = dealt(0, 2);
        early.history[1] = vec![CALL as u8];
        assert!(g.validate_decoded_state(&early, false).is_err());
    }
}
