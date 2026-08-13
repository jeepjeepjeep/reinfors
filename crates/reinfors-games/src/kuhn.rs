//! N-player Kuhn poker with declared deals and private observations.

use reinfors_core::game::{Actor, ChanceDist, Game, Transition};

pub const PASS: usize = 0;
pub const BET: usize = 1;

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct KuhnState {
    pub cards: Vec<u8>,
    pub history: Vec<u8>,
}

impl KuhnState {
    pub fn is_terminal_pub(&self) -> bool {
        self.is_terminal()
    }

    fn n(&self) -> usize {
        self.cards.len()
    }

    fn first_bet(&self) -> Option<usize> {
        self.history.iter().position(|&a| a == BET as u8)
    }

    fn is_terminal(&self) -> bool {
        match self.first_bet() {
            None => self.history.len() == self.n(),
            Some(b) => self.history.len() == b + self.n(),
        }
    }

    fn bet_by(&self, player: usize) -> bool {
        self.history
            .iter()
            .enumerate()
            .any(|(i, &a)| i % self.n() == player && a == BET as u8)
    }

    fn contribution(&self, player: usize) -> i64 {
        1 + i64::from(self.bet_by(player))
    }
}

#[derive(Clone)]
pub struct KuhnPoker {
    pub players: usize,
}

impl Default for KuhnPoker {
    fn default() -> Self {
        KuhnPoker { players: 2 }
    }
}

impl KuhnPoker {
    pub fn validate(&self) -> Result<(), String> {
        if !(2..=10).contains(&self.players) {
            return Err(format!(
                "players must be in 2..=10 (OpenSpiel's kuhn_poker range), got {}",
                self.players
            ));
        }
        Ok(())
    }

    fn remaining(&self, state: &KuhnState) -> Vec<u8> {
        (0..(self.players + 1) as u8)
            .filter(|c| !state.cards.contains(c))
            .collect()
    }

    fn payouts(&self, state: &KuhnState) -> Vec<f64> {
        let n = state.n();
        let any_bet = state.first_bet().is_some();
        let winner = (0..n)
            .filter(|&p| !any_bet || state.bet_by(p))
            .max_by_key(|&p| state.cards[p])
            .expect("the bettor is always eligible");
        let pot: i64 = (0..n).map(|p| state.contribution(p)).sum();
        (0..n)
            .map(|p| {
                if p == winner {
                    (pot - state.contribution(p)) as f64
                } else {
                    -(state.contribution(p) as f64)
                }
            })
            .collect()
    }
}

impl Game for KuhnPoker {
    type State = KuhnState;
    type Event = f64;

    fn num_agents(&self) -> usize {
        self.players
    }

    fn action_count(&self) -> usize {
        2
    }

    fn perfect_information(&self) -> bool {
        false
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
        if state.cards.len() < self.players {
            return Actor::Chance;
        }
        Actor::Agent(state.history.len() % self.players)
    }

    fn chance_node(&self, state: &KuhnState) -> ChanceDist {
        ChanceDist::Uniform(self.players + 1 - state.cards.len())
    }

    fn apply_chance_node(&self, state: &KuhnState, outcome: usize) -> Transition<KuhnState, f64> {
        let mut next = state.clone();
        next.cards.push(self.remaining(state)[outcome]);
        Transition::silent(next, self.players)
    }

    fn legal_actions(&self, state: &KuhnState, agent: usize) -> Vec<usize> {
        if state.cards.len() < self.players
            || state.is_terminal()
            || agent != state.history.len() % self.players
        {
            return Vec::new();
        }
        vec![PASS, BET]
    }

    fn step(&self, state: &KuhnState, actions: &[usize]) -> Transition<KuhnState, f64> {
        let me = state.history.len() % self.players;
        let mut next = state.clone();
        next.history.push(actions[me].min(BET) as u8);
        let terminal = next.is_terminal();
        let events = if terminal {
            self.payouts(&next).into_iter().map(Some).collect()
        } else {
            vec![None; self.players]
        };
        Transition {
            next_state: next,
            events,
            terminal,
        }
    }

    fn initial_state(&self) -> KuhnState {
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
        let n = self.players;
        if state.cards.len() != n {
            return Err("every card must be dealt (birth chains are transient)".to_string());
        }
        let deck = (n + 1) as u8;
        let distinct = state
            .cards
            .iter()
            .collect::<std::collections::HashSet<_>>()
            .len();
        if state.cards.iter().any(|&c| c >= deck) || distinct != n {
            return Err(format!("cards must be {n} distinct ids below {deck}"));
        }
        if state.history.len() > 2 * n - 1 || state.history.iter().any(|&a| a > BET as u8) {
            return Err("malformed action history".to_string());
        }
        // Replay proper prefixes so structurally plausible histories cannot continue past terminal.
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

pub struct KuhnEncoder {
    pub players: usize,
}

impl Default for KuhnEncoder {
    fn default() -> Self {
        KuhnEncoder { players: 2 }
    }
}

impl reinfors_core::ActionView for KuhnEncoder {}

impl reinfors_core::StateEncoder for KuhnEncoder {
    type State = KuhnState;

    fn encode(&self, s: &KuhnState, agent: usize) -> Vec<f32> {
        let cards = self.players + 1;
        // (N+1) card slots + (2N-1) history slots = the advertised 3N shape.
        let mut obs = vec![0.0f32; cards + 2 * self.players - 1];
        obs[s.cards[agent] as usize] = 1.0;
        for (i, &a) in s.history.iter().enumerate() {
            obs[cards + i] = (a + 1) as f32 / 2.0;
        }
        obs
    }

    fn obs_shape(&self) -> (usize, usize, usize) {
        (3 * self.players, 1, 1)
    }

    fn observation_space(&self) -> reinfors_core::Space {
        reinfors_core::Space::unit_box(vec![3 * self.players, 1, 1])
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
        let g = KuhnPoker::default();
        let t = g.step(&dealt(0, 2, &[BET as u8]), &[PASS, PASS]);
        assert!(t.terminal);
        assert_eq!(t.events, vec![Some(1.0), Some(-1.0)]);
        let t = g.step(&dealt(0, 2, &[PASS as u8]), &[0, PASS]);
        assert!(t.terminal);
        assert_eq!(t.events, vec![Some(-1.0), Some(1.0)]);
        let t = g.step(&dealt(2, 1, &[BET as u8]), &[0, BET]);
        assert!(t.terminal);
        assert_eq!(t.events, vec![Some(2.0), Some(-2.0)]);
        let t = g.step(&dealt(2, 0, &[PASS as u8, BET as u8]), &[PASS, 0]);
        assert!(t.terminal);
        assert_eq!(t.events, vec![Some(-1.0), Some(1.0)]);
        let t = g.step(&dealt(2, 0, &[PASS as u8, BET as u8]), &[BET, 0]);
        assert!(t.terminal);
        assert_eq!(t.events, vec![Some(2.0), Some(-2.0)]);
    }

    #[test]
    fn three_player_lines_pay_the_pyspiel_amounts() {
        let g = KuhnPoker { players: 3 };
        let dealt3 = |history: &[u8]| KuhnState {
            cards: vec![0, 1, 2],
            history: history.to_vec(),
        };
        let line = |history: &[u8]| {
            let s = dealt3(history);
            assert!(s.is_terminal(), "{history:?} must be terminal");
            g.payouts(&s)
        };
        assert_eq!(line(&[0, 0, 0]), vec![-1.0, -1.0, 2.0]);
        assert_eq!(line(&[1, 0, 0]), vec![2.0, -1.0, -1.0]);
        assert_eq!(line(&[1, 1, 1]), vec![-2.0, -2.0, 4.0]);
        assert_eq!(line(&[1, 0, 1]), vec![-2.0, -1.0, 3.0]);
        assert_eq!(line(&[0, 1, 0, 0]), vec![-1.0, 2.0, -1.0]);
        assert_eq!(line(&[0, 1, 1, 0]), vec![-1.0, -2.0, 3.0]);
        assert_eq!(line(&[0, 1, 0, 1]), vec![-2.0, 3.0, -1.0]);
        assert_eq!(line(&[0, 0, 1, 0, 0]), vec![-1.0, -1.0, 2.0]);
        assert_eq!(line(&[0, 0, 1, 1, 1]), vec![-2.0, -2.0, 4.0]);
        assert!(!dealt3(&[0, 1, 0]).is_terminal());
        assert!(!dealt3(&[0, 0, 1, 0]).is_terminal());
        assert!(matches!(g.actor(&dealt3(&[0, 1])), Actor::Agent(2)));
        assert!(matches!(g.actor(&dealt3(&[0, 1, 0])), Actor::Agent(0)));
    }

    #[test]
    fn three_player_deal_and_construction_bounds() {
        let g = KuhnPoker { players: 3 };
        assert!(g.validate().is_ok());
        assert!(KuhnPoker { players: 1 }.validate().is_err());
        assert!(KuhnPoker { players: 11 }.validate().is_err());
        let root = g.initial_state();
        assert!(matches!(g.actor(&root), Actor::Chance));
        assert_eq!(g.chance_node(&root).count(), 4, "deck is players + 1");
        let s1 = g.apply_chance_node(&root, 0).next_state;
        assert_eq!(g.chance_node(&s1).count(), 3);
        let s2 = g.apply_chance_node(&s1, 0).next_state;
        let s3 = g.apply_chance_node(&s2, 0).next_state;
        assert_eq!(s3.cards, vec![0, 1, 2]);
        assert!(matches!(g.actor(&s3), Actor::Agent(0)), "player 0 opens");
        use reinfors_core::StateEncoder;
        let enc = KuhnEncoder { players: 3 };
        assert_eq!(enc.obs_shape(), (9, 1, 1));
        assert_eq!(enc.encode(&s3, 2).len(), 9);
    }

    #[test]
    fn the_deal_is_a_declared_root_chain() {
        let g = KuhnPoker::default();
        let root = g.initial_state();
        assert!(matches!(g.actor(&root), Actor::Chance));
        assert_eq!(g.chance_node(&root).count(), 3);
        let s1 = g.apply_chance_node(&root, 1).next_state;
        assert_eq!(g.chance_node(&s1).count(), 2);
        let s2 = g.apply_chance_node(&s1, 1).next_state;
        assert_eq!(s2.cards, vec![1, 2]);
        assert!(matches!(g.actor(&s2), Actor::Agent(0)), "player 0 opens");
    }

    #[test]
    fn keys_and_observations_carry_the_same_information() {
        let g = KuhnPoker::default();
        let enc = KuhnEncoder::default();
        let a = dealt(1, 0, &[PASS as u8, BET as u8]);
        let b = dealt(1, 2, &[PASS as u8, BET as u8]);
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
        let c = dealt(1, 0, &[BET as u8]);
        assert_ne!(
            g.information_state_key(&a, 0),
            g.information_state_key(&c, 0)
        );
        assert_ne!(enc.encode(&a, 0), enc.encode(&c, 0));
    }

    #[test]
    fn codec_round_trips_and_rejects_unsafe_states() {
        let g = KuhnPoker::default();
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
