//! Kuhn poker — the 3-card analytic testbed for imperfect-information algorithms (12
//! information sets; known Nash family with game value -1/18 for the first player). Rules and
//! action ids match OpenSpiel's `kuhn_poker`: both players ante 1, each is dealt one card from
//! {J, Q, K}, then player 0 acts first with PASS/BET (bet size 1); passing when facing a bet
//! folds. Fully declared chance (`all_chance_declared`): the two deals are root chance nodes
//! realized at episode birth, so solvers can enumerate them. Hidden information
//! (`perfect_information` = false): each player sees only its own card.
//!
//! The state is minimal — dealt cards plus the public action history; pots, the actor, and
//! terminal status are all derived, so no consistency cross-checks are needed at decode.

use reinfors_core::game::{Actor, ChanceDist, Game, Rng, Transition};

/// PASS checks, or folds when facing a bet (OpenSpiel action id 0).
pub const PASS: usize = 0;
/// BET opens for 1, or calls a bet (OpenSpiel action id 1).
pub const BET: usize = 1;

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct KuhnState {
    /// Dealt private cards (0 = J, 1 = Q, 2 = K), player order; grows 0 -> 2 during the birth
    /// chain.
    pub cards: Vec<u8>,
    /// Public action history (PASS/BET ids), player 0 first.
    pub history: Vec<u8>,
}

impl KuhnState {
    /// Public terminality accessor (bindings; `is_terminal` stays crate-internal shorthand).
    pub fn is_terminal_pub(&self) -> bool {
        self.is_terminal()
    }

    fn is_terminal(&self) -> bool {
        matches!(
            self.history.as_slice(),
            [p, q] if !(*p == PASS as u8 && *q == BET as u8)
        ) || self.history.len() == 3
    }

    /// Per-player pot contribution: the ante plus 1 per BET made.
    fn contribution(&self, player: usize) -> i64 {
        1 + self
            .history
            .iter()
            .enumerate()
            .filter(|&(i, &a)| i % 2 == player && a == BET as u8)
            .count() as i64
    }
}

pub struct KuhnPoker;

impl KuhnPoker {
    fn remaining(&self, state: &KuhnState) -> Vec<u8> {
        (0..3).filter(|c| !state.cards.contains(c)).collect()
    }

    /// Terminal chip deltas (zero-sum). A terminal history ending in PASS after a BET is a
    /// fold; otherwise showdown, higher card wins.
    fn payouts(&self, state: &KuhnState) -> Vec<f64> {
        let h = &state.history;
        let folder: Option<usize> = match h.as_slice() {
            [b, p] if *b == BET as u8 && *p == PASS as u8 => Some(1),
            [_, _, p] if *p == PASS as u8 => Some(0),
            _ => None,
        };
        let winner = match folder {
            Some(f) => 1 - f,
            None => usize::from(state.cards[1] > state.cards[0]),
        };
        let loser = 1 - winner;
        let mut deltas = vec![0.0; 2];
        deltas[winner] = state.contribution(loser) as f64;
        deltas[loser] = -(state.contribution(loser) as f64);
        deltas
    }
}

impl Game for KuhnPoker {
    type State = KuhnState;
    type Event = f64; // per-player chip delta at the terminal tick, 0 elsewhere

    fn num_agents(&self) -> usize {
        2
    }

    fn action_count(&self) -> usize {
        2
    }

    fn perfect_information(&self) -> bool {
        false // the opponent's card is hidden
    }

    fn all_chance_declared(&self) -> bool {
        true // both deals are root chance nodes; initial_state draws nothing
    }

    fn information_states(&self) -> bool {
        true
    }

    fn information_state_key(&self, state: &KuhnState, agent: usize) -> Vec<u8> {
        let mut k = Vec::with_capacity(2 + state.history.len());
        k.push(agent as u8);
        k.push(state.cards[agent]);
        k.extend_from_slice(&state.history);
        k
    }

    fn actor(&self, state: &KuhnState) -> Actor {
        if state.cards.len() < 2 {
            return Actor::Chance; // the deal (birth chain)
        }
        Actor::Agent(state.history.len() % 2)
    }

    fn chance_node(&self, state: &KuhnState) -> ChanceDist {
        ChanceDist::Uniform(3 - state.cards.len())
    }

    fn apply_chance_node(&self, state: &KuhnState, outcome: usize) -> Transition<KuhnState, f64> {
        let mut next = state.clone();
        next.cards.push(self.remaining(state)[outcome]);
        Transition {
            next_state: next,
            events: vec![0.0; 2],
            terminal: false,
        }
    }

    fn legal_actions(&self, state: &KuhnState, agent: usize) -> Vec<usize> {
        if state.cards.len() < 2 || state.is_terminal() || agent != state.history.len() % 2 {
            return Vec::new();
        }
        vec![PASS, BET]
    }

    fn step(&self, state: &KuhnState, actions: &[usize]) -> Transition<KuhnState, f64> {
        let me = state.history.len() % 2;
        let mut next = state.clone();
        // Backstop for direct core callers: both actions are always legal, so clamp.
        next.history.push(actions[me].min(BET) as u8);
        let terminal = next.is_terminal();
        let events = if terminal {
            self.payouts(&next)
        } else {
            vec![0.0; 2]
        };
        Transition {
            next_state: next,
            events,
            terminal,
        }
    }

    fn initial_state(&self, _rng: &mut dyn Rng) -> KuhnState {
        // Draws nothing (`all_chance_declared`): the empty deal is the birth-chain root.
        KuhnState {
            cards: Vec::new(),
            history: Vec::new(),
        }
    }
}

impl reinfors_core::StateCodec for KuhnPoker {
    type State = KuhnState;

    fn encode(&self, s: &KuhnState) -> Vec<u8> {
        crate::codec_util::serde_encode(1, s)
    }

    fn decode(&self, bytes: &[u8]) -> Result<KuhnState, String> {
        crate::codec_util::serde_decode(1, bytes)
    }

    fn validate_decoded_state(&self, state: &KuhnState, done: bool) -> Result<(), String> {
        if state.cards.len() != 2 {
            return Err("both cards must be dealt (birth chains are transient)".to_string());
        }
        if state.cards[0] >= 3 || state.cards[1] >= 3 || state.cards[0] == state.cards[1] {
            return Err("cards must be two distinct ids below 3".to_string());
        }
        // History grammar: every proper prefix is non-terminal, length capped by the rules.
        if state.history.len() > 3 || state.history.iter().any(|&a| a > BET as u8) {
            return Err("malformed action history".to_string());
        }
        for cut in 0..state.history.len() {
            let prefix = KuhnState {
                cards: state.cards.clone(),
                history: state.history[..cut].to_vec(),
            };
            if prefix.is_terminal() {
                return Err("actions continue past a terminal history".to_string());
            }
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

/// `(6, 1, 1)`: own card one-hot (3) + one channel per history slot (3), holding
/// `(action + 1) / 2` — 0 empty, 0.5 PASS, 1.0 BET. Together with the seat parity implied by
/// slot count this is exactly the information set (pinned by test against the key).
pub struct KuhnEncoder;

impl reinfors_core::ActionView for KuhnEncoder {}

impl reinfors_core::StateEncoder for KuhnEncoder {
    type State = KuhnState;

    fn encode(&self, s: &KuhnState, agent: usize) -> Vec<f32> {
        let mut obs = vec![0.0f32; 6];
        obs[s.cards[agent] as usize] = 1.0;
        for (i, &a) in s.history.iter().enumerate() {
            obs[3 + i] = (a + 1) as f32 / 2.0;
        }
        obs
    }

    fn obs_shape(&self) -> (usize, usize, usize) {
        (6, 1, 1)
    }

    fn observation_space(&self) -> reinfors_core::Space {
        reinfors_core::Space::unit_box(vec![6, 1, 1])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use reinfors_core::{StateCodec, StateEncoder};

    fn dealt(c0: u8, c1: u8, history: &[u8]) -> KuhnState {
        KuhnState {
            cards: vec![c0, c1],
            history: history.to_vec(),
        }
    }

    #[test]
    fn terminal_lines_pay_the_pyspiel_amounts() {
        let g = KuhnPoker;
        // bet-fold: p0 wins the ante (+1/-1) — pinned against pyspiel.
        let t = g.step(&dealt(0, 2, &[BET as u8]), &[PASS, PASS]);
        assert!(t.terminal);
        assert_eq!(t.events, vec![1.0, -1.0]);
        // pass-pass: showdown for the antes; K beats J.
        let t = g.step(&dealt(0, 2, &[PASS as u8]), &[0, PASS]);
        assert!(t.terminal);
        assert_eq!(t.events, vec![-1.0, 1.0]);
        // bet-call: showdown for 2 each.
        let t = g.step(&dealt(2, 1, &[BET as u8]), &[0, BET]);
        assert!(t.terminal);
        assert_eq!(t.events, vec![2.0, -2.0]);
        // pass-bet-pass: p0 folds, p1 wins the ante.
        let t = g.step(&dealt(2, 0, &[PASS as u8, BET as u8]), &[PASS, 0]);
        assert!(t.terminal);
        assert_eq!(t.events, vec![-1.0, 1.0]);
        // pass-bet-bet: showdown for 2 each.
        let t = g.step(&dealt(2, 0, &[PASS as u8, BET as u8]), &[BET, 0]);
        assert!(t.terminal);
        assert_eq!(t.events, vec![2.0, -2.0]);
    }

    #[test]
    fn the_deal_is_a_declared_root_chain() {
        let g = KuhnPoker;
        struct Poisoned;
        impl Rng for Poisoned {
            fn below(&mut self, _n: usize) -> usize {
                panic!("initial_state drew despite all_chance_declared")
            }
            fn unit(&mut self) -> f64 {
                panic!("initial_state drew despite all_chance_declared")
            }
        }
        let root = g.initial_state(&mut Poisoned);
        assert!(matches!(g.actor(&root), Actor::Chance));
        assert_eq!(g.chance_node(&root).count(), 3);
        let s1 = g.apply_chance_node(&root, 1).next_state; // p0 gets Q
        assert_eq!(g.chance_node(&s1).count(), 2);
        let s2 = g.apply_chance_node(&s1, 1).next_state; // remaining {J, K}[1] = K
        assert_eq!(s2.cards, vec![1, 2]);
        assert!(matches!(g.actor(&s2), Actor::Agent(0)), "player 0 opens");
        assert!(
            !g.chance_nodes(),
            "root-only chance: post-birth states never chance"
        );
    }

    #[test]
    fn keys_and_observations_carry_the_same_information() {
        let g = KuhnPoker;
        let enc = KuhnEncoder;
        // Same public line, different opponent card: same key, same obs for agent 0...
        let a = dealt(1, 0, &[PASS as u8, BET as u8]);
        let b = dealt(1, 2, &[PASS as u8, BET as u8]);
        assert_eq!(
            g.information_state_key(&a, 0),
            g.information_state_key(&b, 0)
        );
        assert_eq!(enc.encode(&a, 0), enc.encode(&b, 0));
        // ...different for agent 1 (their own card changed).
        assert_ne!(
            g.information_state_key(&a, 1),
            g.information_state_key(&b, 1)
        );
        assert_ne!(enc.encode(&a, 1), enc.encode(&b, 1));
        // Different public lines split both.
        let c = dealt(1, 0, &[BET as u8]);
        assert_ne!(
            g.information_state_key(&a, 0),
            g.information_state_key(&c, 0)
        );
        assert_ne!(enc.encode(&a, 0), enc.encode(&c, 0));
    }

    #[test]
    fn codec_round_trips_and_rejects_unsafe_states() {
        let g = KuhnPoker;
        let s = dealt(0, 2, &[PASS as u8]);
        let back = g.decode(&g.encode(&s)).unwrap();
        assert_eq!(back, s);
        g.validate_decoded_state(&back, false).unwrap();
        assert!(g
            .validate_decoded_state(&dealt(1, 1, &[]), false)
            .unwrap_err()
            .contains("distinct"));
        assert!(g
            .validate_decoded_state(&dealt(0, 1, &[BET as u8, PASS as u8, PASS as u8]), true)
            .unwrap_err()
            .contains("past a terminal"));
        assert!(g
            .validate_decoded_state(&dealt(0, 1, &[PASS as u8, PASS as u8]), false)
            .unwrap_err()
            .contains("disagrees"));
        let undealt = KuhnState {
            cards: vec![0],
            history: Vec::new(),
        };
        assert!(g
            .validate_decoded_state(&undealt, false)
            .unwrap_err()
            .contains("birth chains are transient"));
    }
}
