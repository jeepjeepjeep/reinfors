//! Counterfactual regret minimization (2-player zero-sum): vanilla CFR, CFR+, and
//! external-sampling MCCFR over one traversal core.
//!
//! CFR decomposes overall regret into per-information-set counterfactual regrets and
//! minimizes each locally by regret matching; the time-AVERAGED strategy profile converges to
//! a Nash equilibrium in 2-player zero-sum games (the current strategies oscillate — the
//! average is the output). The vanilla/CFR+ update scheme deliberately mirrors OpenSpiel's
//! `cfr.py` (alternating player passes; the current policy is a table materialized from
//! regrets AFTER each pass, not regret-matched on the fly; CFR+ adds regret-matching+
//! clamping after each pass and linear averaging) so exploitability trajectories are
//! ITERATION-EXACT against pyspiel — the parity harness pins this.
//!
//! Requirements, asserted at construction: 2 players; `Game::information_states` (tables are
//! keyed by information-set bytes — what forces the learned strategy to be measurable with
//! respect to each player's information); `Game::all_chance_declared`, with the root claim
//! verified by calling `initial_state` with an rng that PANICS on any draw — a game that
//! samples privately would be solved against the wrong tree. Chance is consumed through all
//! three declared seams (root/interior chance nodes, transition-attached `chance_outcomes`):
//! enumerated by the exact variants (fan-capped at [`MAX_ENUMERATED_OUTCOMES`]), sampled by
//! MCCFR — full hold'em therefore runs only under MCCFR (the deal fan is astronomical), and
//! at that scale nothing converges anyway; the solver's home ground is Kuhn/Leduc-sized
//! games.

use std::collections::HashMap;

use crate::game::{Actor, Game, Rng, Transition};
use crate::policy::MAX_ENUMERATED_OUTCOMES;
use crate::reward::Reward;
use crate::rng::SplitMix64;
use crate::solvers::best_response;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CfrVariant {
    /// Alternating-update vanilla CFR (matches pyspiel's `CFRSolver`).
    Vanilla,
    /// CFR+: regret-matching+ (regrets clamped at zero after each pass), alternating updates,
    /// linear averaging (matches pyspiel's `CFRPlusSolver`).
    Plus,
    /// External-sampling MCCFR: chance and opponent actions sampled, the traverser's actions
    /// enumerated. Unbiased counterfactual regret estimates; the only variant that scales past
    /// enumerable chance fans.
    ExternalMccfr,
}

impl CfrVariant {
    /// Stable payload identity (part of the serialized checkpoint).
    fn id(self) -> u8 {
        match self {
            CfrVariant::Vanilla => 0,
            CfrVariant::Plus => 1,
            CfrVariant::ExternalMccfr => 2,
        }
    }
}

struct Node {
    /// Legal action ids at this information set (identical for every member state, by the
    /// definition of an information set — debug-asserted on revisit).
    actions: Vec<usize>,
    regrets: Vec<f64>,
    /// Cumulative (reach-weighted) strategy — the AVERAGE strategy is its normalization.
    cumulative: Vec<f64>,
    /// The materialized current policy (regret-matched after each pass; uniform at birth).
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

    /// Regret matching: play proportionally to positive regret, uniform when none is positive.
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

/// An rng that panics on any draw — proves `initial_state` honors `all_chance_declared`.
struct PoisonedRng;

impl Rng for PoisonedRng {
    fn below(&mut self, _n: usize) -> usize {
        panic!("initial_state drew from the rng despite declaring all_chance_declared")
    }
    fn unit(&mut self) -> f64 {
        panic!("initial_state drew from the rng despite declaring all_chance_declared")
    }
}

pub struct CfrSolver<G: Game> {
    game: G,
    reward: Box<dyn Reward<Event = G::Event>>,
    variant: CfrVariant,
    nodes: HashMap<Vec<u8>, Node>,
    iterations: u64,
    rng: SplitMix64, // MCCFR's sampling stream
}

impl<G: Game> CfrSolver<G> {
    pub fn new(
        game: G,
        reward: Box<dyn Reward<Event = G::Event>>,
        variant: CfrVariant,
        seed: u64,
    ) -> Self {
        assert!(
            game.num_agents() == 2,
            "CFR v1 solves 2-player zero-sum games only; this game has {} agents",
            game.num_agents()
        );
        assert!(
            game.information_states(),
            "CFR requires information-state keys (Game::information_states)"
        );
        assert!(
            game.all_chance_declared(),
            "CFR enumerates chance and requires it fully declared (Game::all_chance_declared)"
        );
        let _ = game.initial_state(&mut PoisonedRng); // verifies the root claim loudly
                                                      // Gate on the REALIZED root — the raw root of a declared game is commonly a chance
                                                      // node, which says nothing about decision dynamics (see the same gate in Deep CFR).
        let realized = crate::game::realize_initial_state(&game, &mut SplitMix64::new(0x0517_B0BE));
        assert!(
            !matches!(game.actor(&realized), Actor::Simultaneous),
            "simultaneous games are not supported by CFR v1 (sequential 2-player only)"
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

    /// Run `n` iterations (one iteration = one regret/strategy pass per player).
    pub fn iterate(&mut self, n: u64) {
        for _ in 0..n {
            self.iterations += 1;
            match self.variant {
                CfrVariant::Vanilla | CfrVariant::Plus => {
                    for player in 0..2 {
                        let root = self.game.initial_state(&mut PoisonedRng);
                        self.enumerate_values(&root, [1.0, 1.0, 1.0], player);
                        if self.variant == CfrVariant::Plus {
                            for node in self.nodes.values_mut() {
                                for r in &mut node.regrets {
                                    *r = r.max(0.0); // regret-matching+
                                }
                            }
                        }
                        for node in self.nodes.values_mut() {
                            node.current = Node::regret_matched(&node.regrets);
                        }
                    }
                }
                CfrVariant::ExternalMccfr => {
                    for player in 0..2 {
                        let root = self.game.initial_state(&mut PoisonedRng);
                        self.sample_values(&root, player);
                    }
                }
            }
        }
    }

    /// The average strategy at an information-set key: `(action ids, probabilities)`, or
    /// `None` for a key the solve never visited (play uniform there).
    pub fn average_strategy(&self, key: &[u8]) -> Option<(Vec<usize>, Vec<f64>)> {
        self.nodes
            .get(key)
            .map(|n| (n.actions.clone(), n.average()))
    }

    /// Exploitability of the current AVERAGE profile: mean of both players' exact
    /// best-response improvements (pyspiel's definition — NashConv / num_players). Zero at a
    /// Nash equilibrium.
    pub fn exploitability(&self) -> f64 {
        best_response::exploitability(&self.game, &*self.reward, &|key, legal| {
            self.average_strategy(key)
                .map(|(_, probs)| probs)
                .unwrap_or_else(|| vec![1.0 / legal as f64; legal])
        })
    }

    /// Expected value for `player` when BOTH play the average profile (full enumeration).
    pub fn expected_value(&self, player: usize) -> f64 {
        let root = self.game.initial_state(&mut PoisonedRng);
        self.profile_value(&root)[player]
    }

    /// Serialize the solve state — tables, iteration counter, AND the sampling rng — for
    /// `load`: an exact checkpoint (a restored MCCFR solve continues bit-identically). The
    /// payload identifies the CFR variant and the game's action-space width so it cannot be
    /// silently loaded into an incompatible solver; full composition identity (game
    /// parameters, reward) is the binding's job, mirroring engine/env snapshots.
    pub fn save(&self) -> Vec<u8> {
        let width = |x: usize, what: &str| -> u32 {
            u32::try_from(x).unwrap_or_else(|_| panic!("{what} exceeds the u32 payload bound"))
        };
        let mut out = vec![1u8]; // layout version
        out.push(self.variant.id());
        out.extend_from_slice(&width(self.game.action_count(), "action_count").to_le_bytes());
        out.extend_from_slice(&self.rng.state().to_le_bytes());
        out.extend_from_slice(&self.iterations.to_le_bytes());
        out.extend_from_slice(&(self.nodes.len() as u64).to_le_bytes());
        let mut keys: Vec<&Vec<u8>> = self.nodes.keys().collect();
        keys.sort(); // canonical order: equal solves serialize identically
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

    // ---------------- exact traversal (vanilla / CFR+) ----------------

    /// Per-player expected values of `state` under the CURRENT profile, updating `player`'s
    /// regrets and cumulative strategy. `reach` = [p0 reach, p1 reach, chance reach].
    fn enumerate_values(&mut self, state: &G::State, reach: [f64; 3], player: usize) -> [f64; 2] {
        match self.game.actor(state) {
            Actor::Chance => {
                let dist = self.game.chance_node(state);
                assert!(
                    dist.count() <= MAX_ENUMERATED_OUTCOMES,
                    "chance fan {} exceeds the enumeration cap; use CfrVariant::ExternalMccfr",
                    dist.count()
                );
                let probs: Vec<f64> = dist.iter_probs().collect();
                let mut v = [0.0; 2];
                for (i, p) in probs.into_iter().enumerate() {
                    let t = self.game.apply_chance_node(state, i);
                    let child = self.transition_values(
                        state,
                        &t,
                        {
                            let mut r = reach;
                            r[2] *= p;
                            r
                        },
                        player,
                    );
                    v[0] += p * child[0];
                    v[1] += p * child[1];
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
                let mut v = [0.0; 2];
                for (ai, &a) in legal.iter().enumerate() {
                    let mut joint = vec![0; 2];
                    joint[who] = a;
                    let t = self.game.step(state, &joint);
                    let child = self.transition_values(
                        state,
                        &t,
                        {
                            let mut r = reach;
                            r[who] *= current[ai];
                            r
                        },
                        player,
                    );
                    v[0] += current[ai] * child[0];
                    v[1] += current[ai] * child[1];
                    child_values.push(child);
                }
                if who == player {
                    let cf_reach = reach[1 - who] * reach[2];
                    let weight = if self.variant == CfrVariant::Plus {
                        self.iterations as f64 // linear averaging
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

    /// Fold a realized transition into the recursion: edge rewards + terminal stop +
    /// transition-attached chance enumeration + the child state.
    fn transition_values(
        &mut self,
        state: &G::State,
        t: &Transition<G::State, G::Event>,
        reach: [f64; 3],
        player: usize,
    ) -> [f64; 2] {
        let r = self.edge_rewards(t);
        if t.terminal {
            return r;
        }
        if let Some(dist) = self.game.chance_outcomes(state, t) {
            assert!(
                dist.count() <= MAX_ENUMERATED_OUTCOMES,
                "chance fan {} exceeds the enumeration cap; use CfrVariant::ExternalMccfr",
                dist.count()
            );
            let probs: Vec<f64> = dist.iter_probs().collect();
            let mut v = r;
            for (i, p) in probs.into_iter().enumerate() {
                let child_state = self.game.apply_chance(state, t, i);
                let child = self.enumerate_values(
                    &child_state,
                    {
                        let mut rr = reach;
                        rr[2] *= p;
                        rr
                    },
                    player,
                );
                v[0] += p * child[0];
                v[1] += p * child[1];
            }
            return v;
        }
        let child = self.enumerate_values(&t.next_state, reach, player);
        [r[0] + child[0], r[1] + child[1]]
    }

    fn edge_rewards(&self, t: &Transition<G::State, G::Event>) -> [f64; 2] {
        [
            crate::reward::edge_reward(&*self.reward, &t.events, 0),
            crate::reward::edge_reward(&*self.reward, &t.events, 1),
        ]
    }

    // ---------------- external-sampling MCCFR ----------------

    /// The traverser's sampled value: chance and the opponent sampled, the traverser's actions
    /// enumerated. Regrets updated at traverser infosets, cumulative strategy at the
    /// opponent's sampled path (the standard external-sampling scheme).
    fn sample_values(&mut self, state: &G::State, player: usize) -> f64 {
        match self.game.actor(state) {
            Actor::Chance => {
                let outcome = self.game.chance_node(state).draw(&mut self.rng);
                let t = self.game.apply_chance_node(state, outcome);
                self.sampled_transition(state, &t, player)
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
                        let mut joint = vec![0; 2];
                        joint[who] = a;
                        let t = self.game.step(state, &joint);
                        let value = self.sampled_transition(state, &t, player);
                        values.push(value);
                        v += sigma[values.len() - 1] * value;
                    }
                    let node = self.nodes.get_mut(&key).expect("created above");
                    for (ai, value) in values.iter().enumerate() {
                        node.regrets[ai] += value - v;
                    }
                    v
                } else {
                    {
                        let node = self.nodes.get_mut(&key).expect("created above");
                        for (ai, &s) in sigma.iter().enumerate() {
                            node.cumulative[ai] += s;
                        }
                    }
                    let a = legal[sample_index(&sigma, &mut self.rng)];
                    let mut joint = vec![0; 2];
                    joint[who] = a;
                    let t = self.game.step(state, &joint);
                    self.sampled_transition(state, &t, player)
                }
            }
            Actor::Simultaneous => panic!("a simultaneous decision was reached mid-game: solvers support uniformly SEQUENTIAL games (the framework assumes one dynamics per game; mixing violates that contract)"),
        }
    }

    fn sampled_transition(
        &mut self,
        state: &G::State,
        t: &Transition<G::State, G::Event>,
        player: usize,
    ) -> f64 {
        let r = self.edge_rewards(t)[player];
        if t.terminal {
            return r;
        }
        if let Some(dist) = self.game.chance_outcomes(state, t) {
            let outcome = dist.draw(&mut self.rng);
            let child_state = self.game.apply_chance(state, t, outcome);
            return r + self.sample_values(&child_state, player);
        }
        r + self.sample_values(&t.next_state, player)
    }

    // ---------------- profile evaluation ----------------

    /// Expected values when BOTH players play the average profile.
    fn profile_value(&self, state: &G::State) -> [f64; 2] {
        match self.game.actor(state) {
            Actor::Chance => {
                let dist = self.game.chance_node(state);
                let probs: Vec<f64> = dist.iter_probs().collect();
                let mut v = [0.0; 2];
                for (i, p) in probs.into_iter().enumerate() {
                    let t = self.game.apply_chance_node(state, i);
                    let child = self.profile_transition(state, &t);
                    v[0] += p * child[0];
                    v[1] += p * child[1];
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
                let mut v = [0.0; 2];
                for (ai, &a) in legal.iter().enumerate() {
                    let mut joint = vec![0; 2];
                    joint[who] = a;
                    let t = self.game.step(state, &joint);
                    let child = self.profile_transition(state, &t);
                    v[0] += sigma[ai] * child[0];
                    v[1] += sigma[ai] * child[1];
                }
                v
            }
            Actor::Simultaneous => panic!("a simultaneous decision was reached mid-game: solvers support uniformly SEQUENTIAL games (the framework assumes one dynamics per game; mixing violates that contract)"),
        }
    }

    fn profile_transition(&self, state: &G::State, t: &Transition<G::State, G::Event>) -> [f64; 2] {
        let r = self.edge_rewards(t);
        if t.terminal {
            return r;
        }
        if let Some(dist) = self.game.chance_outcomes(state, t) {
            let probs: Vec<f64> = dist.iter_probs().collect();
            let mut v = r;
            for (i, p) in probs.into_iter().enumerate() {
                let child_state = self.game.apply_chance(state, t, i);
                let child = self.profile_value(&child_state);
                v[0] += p * child[0];
                v[1] += p * child[1];
            }
            return v;
        }
        let child = self.profile_value(&t.next_state);
        [r[0] + child[0], r[1] + child[1]]
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
