//! Tabular vanilla CFR, CFR+, and external-sampling MCCFR for sequential games. Vanilla and CFR+
//! deliberately mirror OpenSpiel's alternating passes: `current` is materialized after each pass,
//! not regret-matched on demand. The parity harness pins iteration-exact trajectories.

use std::collections::HashMap;

use crate::game::{Actor, Game, Rng, Transition};

/// Player-count ceiling for tabular solvers.
pub const MAX_CFR_PLAYERS: usize = 10;

type Vals = [f64; MAX_CFR_PLAYERS];
type Reach = [f64; MAX_CFR_PLAYERS + 1];
const CHANCE: usize = MAX_CFR_PLAYERS;
use crate::policy::MAX_ENUMERATED_OUTCOMES;
use crate::reward::Reward;
use crate::rng::SplitMix64;
use crate::solvers::best_response;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CfrVariant {
    Vanilla,
    Plus,
    ExternalMccfr,
}

impl CfrVariant {
    fn id(self) -> u8 {
        match self {
            CfrVariant::Vanilla => 0,
            CfrVariant::Plus => 1,
            CfrVariant::ExternalMccfr => 2,
        }
    }
}

struct Node {
    actions: Vec<usize>,
    regrets: Vec<f64>,
    cumulative: Vec<f64>,
    current: Vec<f64>,
}

impl Node {
    fn new(actions: Vec<usize>) -> Self {
        let n = actions.len();
        Node {
            actions,
            regrets: vec![0.0; n],
            cumulative: vec![0.0; n],
            current: vec![1.0 / n as f64; n],
        }
    }

    fn regret_matched(regrets: &[f64]) -> Vec<f64> {
        let positive: f64 = regrets.iter().map(|r| r.max(0.0)).sum();
        if positive > 0.0 {
            regrets.iter().map(|r| r.max(0.0) / positive).collect()
        } else {
            vec![1.0 / regrets.len() as f64; regrets.len()]
        }
    }

    fn average(&self) -> Vec<f64> {
        let total: f64 = self.cumulative.iter().sum();
        if total > 0.0 {
            self.cumulative.iter().map(|c| c / total).collect()
        } else {
            vec![1.0 / self.actions.len() as f64; self.actions.len()]
        }
    }
}

pub struct CfrSolver<G: Game> {
    game: G,
    reward: Box<dyn Reward<Event = G::Event>>,
    variant: CfrVariant,
    nodes: HashMap<Vec<u8>, Node>,
    iterations: u64,
    rng: SplitMix64,
}

impl<G: Game> CfrSolver<G> {
    pub fn new(
        game: G,
        reward: Box<dyn Reward<Event = G::Event>>,
        variant: CfrVariant,
        seed: u64,
    ) -> Self {
        assert!(
            (2..=MAX_CFR_PLAYERS).contains(&game.num_agents()),
            "CFR supports 2..={MAX_CFR_PLAYERS} players; this game has {} agents. NOTE: for \
             more than 2 players the average profile carries NO Nash guarantee — regret \
             minimization is empirical there (see the module docs)",
            game.num_agents()
        );
        assert!(
            game.information_states(),
            "CFR requires information-state keys (Game::information_states)"
        );
        // A chance root does not reveal the game's decision dynamics.
        let realized = crate::game::realize_initial_state(&game, &mut SplitMix64::new(0x0517_B0BE));
        assert!(
            !matches!(game.actor(&realized), Actor::Simultaneous),
            "simultaneous games are not supported by CFR (sequential turn-taking only)"
        );
        CfrSolver {
            game,
            reward,
            variant,
            nodes: HashMap::new(),
            iterations: 0,
            rng: SplitMix64::new(seed ^ 0xC0FF_EE00_5EED_5EED),
        }
    }

    pub fn iterations(&self) -> u64 {
        self.iterations
    }

    pub fn num_infosets(&self) -> usize {
        self.nodes.len()
    }

    pub fn num_players(&self) -> usize {
        self.game.num_agents()
    }

    /// Run `n` full player-pass iterations.
    pub fn iterate(&mut self, n: u64) {
        for _ in 0..n {
            self.iterations += 1;
            match self.variant {
                CfrVariant::Vanilla | CfrVariant::Plus => {
                    for player in 0..self.game.num_agents() {
                        let root = self.game.initial_state();
                        self.enumerate_values(&root, [1.0; MAX_CFR_PLAYERS + 1], player);
                        if self.variant == CfrVariant::Plus {
                            for node in self.nodes.values_mut() {
                                for r in &mut node.regrets {
                                    *r = r.max(0.0);
                                }
                            }
                        }
                        for node in self.nodes.values_mut() {
                            node.current = Node::regret_matched(&node.regrets);
                        }
                    }
                }
                CfrVariant::ExternalMccfr => {
                    for player in 0..self.game.num_agents() {
                        let root = self.game.initial_state();
                        self.sample_values(&root, player);
                    }
                }
            }
        }
    }

    /// Average strategy and action ids at a visited information set.
    pub fn average_strategy(&self, key: &[u8]) -> Option<(Vec<usize>, Vec<f64>)> {
        self.nodes
            .get(key)
            .map(|n| (n.actions.clone(), n.average()))
    }

    /// NashConv divided by the number of players.
    pub fn exploitability(&self) -> Result<f64, best_response::EnumerationCapExceeded> {
        best_response::exploitability(&self.game, &*self.reward, &self.profile())
    }

    /// Sum of every player's unilateral best-response improvement.
    pub fn nash_conv(&self) -> Result<f64, best_response::EnumerationCapExceeded> {
        best_response::nash_conv(&self.game, &*self.reward, &self.profile())
    }

    /// Exact best-response value for every player.
    pub fn best_response_values(&self) -> Result<Vec<f64>, best_response::EnumerationCapExceeded> {
        best_response::best_response_values(&self.game, &*self.reward, &self.profile())
    }

    /// Average profile, uniform at unvisited information sets.
    fn profile(&self) -> impl Fn(&[u8], usize) -> Vec<f64> + '_ {
        |key, legal| {
            self.average_strategy(key)
                .map(|(_, probs)| probs)
                .unwrap_or_else(|| vec![1.0 / legal as f64; legal])
        }
    }

    /// Expected value for `player` under the average profile.
    pub fn expected_value(&self, player: usize) -> f64 {
        assert!(
            player < self.num_players(),
            "player {player} out of range: this game has {} players",
            self.num_players()
        );
        let root = self.game.initial_state();
        self.profile_value(&root)[player]
    }

    /// Serialize the complete solve state for exact continuation.
    pub fn save(&self) -> Vec<u8> {
        let width = |x: usize, what: &str| -> u32 {
            u32::try_from(x).unwrap_or_else(|_| panic!("{what} exceeds the u32 payload bound"))
        };
        let mut out = vec![1u8];
        out.push(self.variant.id());
        out.extend_from_slice(&width(self.game.action_count(), "action_count").to_le_bytes());
        out.extend_from_slice(&self.rng.state().to_le_bytes());
        out.extend_from_slice(&self.iterations.to_le_bytes());
        out.extend_from_slice(&(self.nodes.len() as u64).to_le_bytes());
        let mut keys: Vec<&Vec<u8>> = self.nodes.keys().collect();
        keys.sort();
        for key in keys {
            let node = &self.nodes[key];
            out.extend_from_slice(&width(key.len(), "an information-set key").to_le_bytes());
            out.extend_from_slice(key);
            out.extend_from_slice(&width(node.actions.len(), "a legal-action count").to_le_bytes());
            for (i, &a) in node.actions.iter().enumerate() {
                out.extend_from_slice(&width(a, "an action id").to_le_bytes());
                out.extend_from_slice(&node.regrets[i].to_le_bytes());
                out.extend_from_slice(&node.cumulative[i].to_le_bytes());
                out.extend_from_slice(&node.current[i].to_le_bytes());
            }
        }
        out
    }

    pub fn load(&mut self, bytes: &[u8]) -> Result<(), String> {
        let mut r = ByteReader { bytes, pos: 0 };
        if r.u8()? != 1 {
            return Err("unknown CFR payload version".to_string());
        }
        if r.u8()? != self.variant.id() {
            return Err("payload was saved by a different CFR variant".to_string());
        }
        if r.u32()? as usize != self.game.action_count() {
            return Err("payload was saved for a different action space".to_string());
        }
        let rng_state = r.u64()?;
        let iterations = r.u64()?;
        let n_nodes = r.u64()? as usize;
        if n_nodes > 1 << 32 {
            return Err("implausible node count".to_string());
        }
        let mut nodes = HashMap::with_capacity(n_nodes.min(1 << 20));
        for _ in 0..n_nodes {
            let key_len = r.u32()? as usize;
            let key = r.take(key_len)?.to_vec();
            let n_actions = r.u32()? as usize;
            if n_actions == 0 || n_actions > self.game.action_count() {
                return Err("implausible action count".to_string());
            }
            let mut actions = Vec::with_capacity(n_actions);
            let mut regrets = Vec::with_capacity(n_actions);
            let mut cumulative = Vec::with_capacity(n_actions);
            let mut current = Vec::with_capacity(n_actions);
            for _ in 0..n_actions {
                let a = r.u32()? as usize;
                if a >= self.game.action_count() {
                    return Err("action id out of range".to_string());
                }
                actions.push(a);
                regrets.push(finite(r.f64()?)?);
                cumulative.push(finite(r.f64()?)?);
                current.push(finite(r.f64()?)?);
            }
            nodes.insert(
                key,
                Node {
                    actions,
                    regrets,
                    cumulative,
                    current,
                },
            );
        }
        r.done()?;
        self.rng = SplitMix64::from_state(rng_state);
        self.iterations = iterations;
        self.nodes = nodes;
        Ok(())
    }

    /// Traverse exactly under the current profile and update one player's regrets.
    fn enumerate_values(&mut self, state: &G::State, reach: Reach, player: usize) -> Vals {
        let n = self.game.num_agents();
        match self.game.actor(state) {
            Actor::Chance => {
                let dist = self.game.chance_node(state);
                assert!(
                    dist.count() <= MAX_ENUMERATED_OUTCOMES,
                    "chance fan {} exceeds the enumeration cap; use CfrVariant::ExternalMccfr",
                    dist.count()
                );
                let probs: Vec<f64> = dist.iter_probs().collect();
                let mut v = [0.0; MAX_CFR_PLAYERS];
                for (i, p) in probs.into_iter().enumerate() {
                    let t = self.game.apply_chance_node(state, i);
                    let child = self.transition_values(
                        &t,
                        {
                            let mut r = reach;
                            r[CHANCE] *= p;
                            r
                        },
                        player,
                    );
                    for (vi, ci) in v.iter_mut().zip(child.iter()).take(n) {
                        *vi += p * ci;
                    }
                }
                v
            }
            Actor::Agent(who) => {
                let legal = self.game.legal_actions(state, who);
                let key = self.game.information_state_key(state, who);
                let current = {
                    let node = self
                        .nodes
                        .entry(key.clone())
                        .or_insert_with(|| Node::new(legal.clone()));
                    debug_assert_eq!(node.actions, legal, "one infoset, one action set");
                    node.current.clone()
                };
                let mut child_values = Vec::with_capacity(legal.len());
                let mut v = [0.0; MAX_CFR_PLAYERS];
                for (ai, &a) in legal.iter().enumerate() {
                    let mut joint = vec![0; n];
                    joint[who] = a;
                    let t = self.game.step(state, &joint);
                    let child = self.transition_values(
                        &t,
                        {
                            let mut r = reach;
                            r[who] *= current[ai];
                            r
                        },
                        player,
                    );
                    for (vi, ci) in v.iter_mut().zip(child.iter()).take(n) {
                        *vi += current[ai] * ci;
                    }
                    child_values.push(child);
                }
                if who == player {
                    // Counterfactual reach deliberately excludes the updating player.
                    let mut cf_reach = reach[CHANCE];
                    for (j, r) in reach.iter().enumerate().take(n) {
                        if j != who {
                            cf_reach *= r;
                        }
                    }
                    let weight = if self.variant == CfrVariant::Plus {
                        self.iterations as f64
                    } else {
                        1.0
                    };
                    let node = self.nodes.get_mut(&key).expect("created above");
                    for (ai, child) in child_values.iter().enumerate() {
                        node.regrets[ai] += cf_reach * (child[who] - v[who]);
                        node.cumulative[ai] += weight * reach[who] * current[ai];
                    }
                }
                v
            }
            Actor::Simultaneous => panic!("a simultaneous decision was reached mid-game: solvers support uniformly SEQUENTIAL games (the framework assumes one dynamics per game; mixing violates that contract)"),
        }
    }

    fn transition_values(
        &mut self,
        t: &Transition<G::State, G::Event>,
        reach: Reach,
        player: usize,
    ) -> Vals {
        let mut r = self.edge_rewards(t);
        if t.terminal {
            return r;
        }
        let child = self.enumerate_values(&t.next_state, reach, player);
        for (ri, ci) in r.iter_mut().zip(child.iter()) {
            *ri += ci;
        }
        r
    }

    fn edge_rewards(&self, t: &Transition<G::State, G::Event>) -> Vals {
        let mut r = [0.0; MAX_CFR_PLAYERS];
        for (p, slot) in r.iter_mut().enumerate().take(self.game.num_agents()) {
            *slot = crate::reward::edge_reward(&*self.reward, &t.events, p);
        }
        r
    }

    /// External sampling with the traverser's actions enumerated.
    fn sample_values(&mut self, state: &G::State, player: usize) -> f64 {
        match self.game.actor(state) {
            Actor::Chance => {
                let outcome = self.game.chance_node(state).draw(&mut self.rng);
                let t = self.game.apply_chance_node(state, outcome);
                self.sampled_transition(&t, player)
            }
            Actor::Agent(who) => {
                let legal = self.game.legal_actions(state, who);
                let key = self.game.information_state_key(state, who);
                let sigma = {
                    let node = self
                        .nodes
                        .entry(key.clone())
                        .or_insert_with(|| Node::new(legal.clone()));
                    Node::regret_matched(&node.regrets)
                };
                if who == player {
                    let mut values = Vec::with_capacity(legal.len());
                    let mut v = 0.0;
                    for &a in &legal {
                        let mut joint = vec![0; self.game.num_agents()];
                        joint[who] = a;
                        let t = self.game.step(state, &joint);
                        let value = self.sampled_transition(&t, player);
                        values.push(value);
                        v += sigma[values.len() - 1] * value;
                    }
                    let node = self.nodes.get_mut(&key).expect("created above");
                    for (ai, value) in values.iter().enumerate() {
                        node.regrets[ai] += value - v;
                    }
                    v
                } else {
                    // OpenSpiel's simple average estimator records one successor player.
                    if who == (player + 1) % self.game.num_agents() {
                        let node = self.nodes.get_mut(&key).expect("created above");
                        for (ai, &s) in sigma.iter().enumerate() {
                            node.cumulative[ai] += s;
                        }
                    }
                    let a = legal[sample_index(&sigma, &mut self.rng)];
                    let mut joint = vec![0; self.game.num_agents()];
                    joint[who] = a;
                    let t = self.game.step(state, &joint);
                    self.sampled_transition(&t, player)
                }
            }
            Actor::Simultaneous => panic!("a simultaneous decision was reached mid-game: solvers support uniformly SEQUENTIAL games (the framework assumes one dynamics per game; mixing violates that contract)"),
        }
    }

    fn sampled_transition(&mut self, t: &Transition<G::State, G::Event>, player: usize) -> f64 {
        let r = self.edge_rewards(t)[player];
        if t.terminal {
            return r;
        }
        r + self.sample_values(&t.next_state, player)
    }

    fn profile_value(&self, state: &G::State) -> Vals {
        let n = self.game.num_agents();
        match self.game.actor(state) {
            Actor::Chance => {
                let dist = self.game.chance_node(state);
                let probs: Vec<f64> = dist.iter_probs().collect();
                let mut v = [0.0; MAX_CFR_PLAYERS];
                for (i, p) in probs.into_iter().enumerate() {
                    let t = self.game.apply_chance_node(state, i);
                    let child = self.profile_transition(&t);
                    for (vi, ci) in v.iter_mut().zip(child.iter()).take(n) {
                        *vi += p * ci;
                    }
                }
                v
            }
            Actor::Agent(who) => {
                let legal = self.game.legal_actions(state, who);
                let key = self.game.information_state_key(state, who);
                let sigma = self
                    .average_strategy(&key)
                    .map(|(_, p)| p)
                    .unwrap_or_else(|| vec![1.0 / legal.len() as f64; legal.len()]);
                let mut v = [0.0; MAX_CFR_PLAYERS];
                for (ai, &a) in legal.iter().enumerate() {
                    let mut joint = vec![0; n];
                    joint[who] = a;
                    let t = self.game.step(state, &joint);
                    let child = self.profile_transition(&t);
                    for (vi, ci) in v.iter_mut().zip(child.iter()).take(n) {
                        *vi += sigma[ai] * ci;
                    }
                }
                v
            }
            Actor::Simultaneous => panic!("a simultaneous decision was reached mid-game: solvers support uniformly SEQUENTIAL games (the framework assumes one dynamics per game; mixing violates that contract)"),
        }
    }

    fn profile_transition(&self, t: &Transition<G::State, G::Event>) -> Vals {
        let mut r = self.edge_rewards(t);
        if t.terminal {
            return r;
        }
        let child = self.profile_value(&t.next_state);
        for (ri, ci) in r.iter_mut().zip(child.iter()) {
            *ri += ci;
        }
        r
    }
}

fn finite(x: f64) -> Result<f64, String> {
    if x.is_finite() {
        Ok(x)
    } else {
        Err("non-finite table entry".to_string())
    }
}

fn sample_index(probs: &[f64], rng: &mut SplitMix64) -> usize {
    let u = rng.unit();
    let mut acc = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        acc += p;
        if u < acc {
            return i;
        }
    }
    probs.len() - 1
}

struct ByteReader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> ByteReader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        if self.pos + n > self.bytes.len() {
            return Err("truncated CFR payload".to_string());
        }
        let out = &self.bytes[self.pos..self.pos + n];
        self.pos += n;
        Ok(out)
    }
    fn u8(&mut self) -> Result<u8, String> {
        Ok(self.take(1)?[0])
    }
    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn f64(&mut self) -> Result<f64, String> {
        Ok(f64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn done(&self) -> Result<(), String> {
        if self.pos == self.bytes.len() {
            Ok(())
        } else {
            Err("trailing bytes in CFR payload".to_string())
        }
    }
}
