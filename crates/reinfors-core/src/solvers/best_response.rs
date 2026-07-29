//! Exact best response and exploitability against a fixed strategy profile.
//!
//! The best responder maximizes per INFORMATION SET, not per state: at each of its infosets
//! the chosen action must be the argmax of the counterfactual-reach-weighted sum over the
//! member histories (the responder cannot act differently in states it cannot distinguish).
//! Implementation: enumerate the full game tree once into an arena (fan-capped — this is an
//! exact instrument for Kuhn/Leduc-sized games, not full hold'em), collect each responder
//! infoset's members with their opponent-and-chance reach weights, then resolve values with
//! the infoset argmax memoized (choices at deeper infosets resolve recursively).
//!
//! `exploitability` follows pyspiel's definition: the mean of both players' best-response
//! improvements (NashConv / num_players) — zero exactly at a Nash equilibrium.

use std::collections::HashMap;

use crate::game::{Actor, Game, Rng};
use crate::policy::MAX_ENUMERATED_OUTCOMES;
use crate::reward::Reward;

/// A strategy profile queried by information-set key: `(key, legal action count) -> probs`
/// (aligned with the game's `legal_actions` order).
pub type Profile<'a> = dyn Fn(&[u8], usize) -> Vec<f64> + 'a;

/// The arena cap — best response is exact enumeration; past this the game is out of this
/// instrument's scope.
const MAX_TREE_NODES: usize = 4_000_000;

enum ArenaNode {
    Terminal,
    /// `(probability, edge rewards, child)` per outcome.
    Chance(Vec<(f64, [f64; 2], usize)>),
    /// A decision: `(actor, infoset key, (edge rewards, child) per legal action)`.
    Decision {
        who: usize,
        key: Vec<u8>,
        children: Vec<([f64; 2], usize)>,
    },
}

struct Arena {
    nodes: Vec<ArenaNode>,
}

fn build_arena<G: Game>(game: &G, reward: &dyn Reward<Event = G::Event>) -> Arena {
    struct Builder<'a, G: Game> {
        game: &'a G,
        reward: &'a dyn Reward<Event = G::Event>,
        nodes: Vec<ArenaNode>,
    }
    impl<G: Game> Builder<'_, G> {
        fn add(&mut self, state: &G::State) -> usize {
            assert!(
                self.nodes.len() < MAX_TREE_NODES,
                "game tree exceeds the exact best-response cap ({MAX_TREE_NODES} nodes)"
            );
            let idx = self.nodes.len();
            self.nodes.push(ArenaNode::Terminal); // placeholder, patched below
            let node = match self.game.actor(state) {
                Actor::Chance => {
                    let dist = self.game.chance_node(state);
                    assert!(
                        dist.count() <= MAX_ENUMERATED_OUTCOMES,
                        "chance fan {} exceeds the enumeration cap",
                        dist.count()
                    );
                    let probs: Vec<f64> = dist.iter_probs().collect();
                    let mut outcomes = Vec::with_capacity(probs.len());
                    for (i, p) in probs.into_iter().enumerate() {
                        let t = self.game.apply_chance_node(state, i);
                        let r = [
                            self.reward.step_reward(&t.events[0], 0),
                            self.reward.step_reward(&t.events[1], 1),
                        ];
                        let child = if t.terminal {
                            self.terminal()
                        } else {
                            self.resolve_transition_chance(&t.next_state)
                        };
                        outcomes.push((p, r, child));
                    }
                    ArenaNode::Chance(outcomes)
                }
                Actor::Agent(who) => {
                    let legal = self.game.legal_actions(state, who);
                    let key = self.game.information_state_key(state, who);
                    let mut children = Vec::with_capacity(legal.len());
                    for &a in &legal {
                        let mut joint = vec![0; 2];
                        joint[who] = a;
                        let t = self.game.step(state, &joint);
                        let r = [
                            self.reward.step_reward(&t.events[0], 0),
                            self.reward.step_reward(&t.events[1], 1),
                        ];
                        // Transition-attached chance folds into an interposed chance node.
                        let child = if t.terminal {
                            self.terminal()
                        } else if let Some(dist) = self.game.chance_outcomes(state, &t) {
                            let probs: Vec<f64> = dist.iter_probs().collect();
                            let mut outcomes = Vec::with_capacity(probs.len());
                            for (i, p) in probs.into_iter().enumerate() {
                                let cs = self.game.apply_chance(state, &t, i);
                                let child = self.add(&cs);
                                outcomes.push((p, [0.0, 0.0], child));
                            }
                            let idx = self.nodes.len();
                            self.nodes.push(ArenaNode::Chance(outcomes));
                            idx
                        } else {
                            self.add(&t.next_state)
                        };
                        children.push((r, child));
                    }
                    ArenaNode::Decision { who, key, children }
                }
                Actor::Simultaneous => unreachable!("2-player sequential games only"),
            };
            self.nodes[idx] = node;
            idx
        }

        fn resolve_transition_chance(&mut self, state: &G::State) -> usize {
            self.add(state)
        }

        fn terminal(&mut self) -> usize {
            let idx = self.nodes.len();
            self.nodes.push(ArenaNode::Terminal);
            idx
        }
    }
    let mut b = Builder {
        game,
        reward,
        nodes: Vec::new(),
    };
    struct Poisoned;
    impl Rng for Poisoned {
        fn below(&mut self, _n: usize) -> usize {
            panic!("best response requires all_chance_declared (initial_state drew)")
        }
        fn unit(&mut self) -> f64 {
            panic!("best response requires all_chance_declared (initial_state drew)")
        }
    }
    let root = game.initial_state(&mut Poisoned);
    b.add(&root);
    Arena { nodes: b.nodes }
}

struct BrPass<'a> {
    arena: &'a Arena,
    profile: &'a Profile<'a>,
    br_player: usize,
    /// Per responder infoset: member arena nodes with their counterfactual reach weights.
    members: HashMap<Vec<u8>, Vec<(usize, f64)>>,
    choice: HashMap<Vec<u8>, usize>,
}

impl BrPass<'_> {
    /// Collect responder-infoset members weighted by opponent-and-chance reach (the
    /// responder's own reach deliberately excluded — counterfactual).
    fn collect(&mut self, idx: usize, weight: f64) {
        match &self.arena.nodes[idx] {
            ArenaNode::Terminal => {}
            ArenaNode::Chance(outcomes) => {
                for (p, _r, child) in outcomes {
                    self.collect(*child, weight * p);
                }
            }
            ArenaNode::Decision { who, key, children } => {
                if *who == self.br_player {
                    self.members
                        .entry(key.clone())
                        .or_default()
                        .push((idx, weight));
                    for (_r, child) in children {
                        self.collect(*child, weight);
                    }
                } else {
                    let sigma = (self.profile)(key, children.len());
                    for (ai, (_r, child)) in children.iter().enumerate() {
                        self.collect(*child, weight * sigma[ai]);
                    }
                }
            }
        }
    }

    /// The responder's value at `idx` with infoset-level argmax choices (memoized; deeper
    /// choices resolve recursively — perfect recall makes the recursion well-founded).
    fn value(&mut self, idx: usize) -> f64 {
        match &self.arena.nodes[idx] {
            ArenaNode::Terminal => 0.0,
            ArenaNode::Chance(outcomes) => {
                let outcomes = outcomes.clone();
                let mut v = 0.0;
                for (p, r, child) in outcomes {
                    v += p * (r[self.br_player] + self.value(child));
                }
                v
            }
            ArenaNode::Decision { who, key, children } => {
                let (who, key, children) = (*who, key.clone(), children.clone());
                if who == self.br_player {
                    let a = self.choose(&key);
                    let (r, child) = children[a];
                    r[self.br_player] + self.value(child)
                } else {
                    let sigma = (self.profile)(&key, children.len());
                    let mut v = 0.0;
                    for (ai, (r, child)) in children.into_iter().enumerate() {
                        v += sigma[ai] * (r[self.br_player] + self.value(child));
                    }
                    v
                }
            }
        }
    }

    fn choose(&mut self, key: &[u8]) -> usize {
        if let Some(&a) = self.choice.get(key) {
            return a;
        }
        let members = self.members.get(key).cloned().unwrap_or_default();
        let n_actions = members
            .first()
            .map(|&(idx, _)| match &self.arena.nodes[idx] {
                ArenaNode::Decision { children, .. } => children.len(),
                _ => unreachable!("members are decision nodes"),
            })
            .expect("a queried infoset has members");
        let mut best = (0, f64::NEG_INFINITY);
        for a in 0..n_actions {
            let mut q = 0.0;
            for &(idx, w) in &members {
                let (r, child) = match &self.arena.nodes[idx] {
                    ArenaNode::Decision { children, .. } => children[a],
                    _ => unreachable!(),
                };
                q += w * (r[self.br_player] + self.value(child));
            }
            if q > best.1 {
                best = (a, q);
            }
        }
        self.choice.insert(key.to_vec(), best.0);
        best.0
    }
}

/// The exact best-response value for `br_player` against `profile` (expected utility from the
/// root when the responder plays the infoset-argmax reply).
pub fn best_response_value<G: Game>(
    game: &G,
    reward: &dyn Reward<Event = G::Event>,
    profile: &Profile<'_>,
    br_player: usize,
) -> f64 {
    let arena = build_arena(game, reward);
    let mut pass = BrPass {
        arena: &arena,
        profile,
        br_player,
        members: HashMap::new(),
        choice: HashMap::new(),
    };
    pass.collect(0, 1.0);
    pass.value(0)
}

/// pyspiel's exploitability: `(br_0 + br_1) / 2` for a 2-player zero-sum game — zero exactly
/// at a Nash equilibrium of `profile`.
pub fn exploitability<G: Game>(
    game: &G,
    reward: &dyn Reward<Event = G::Event>,
    profile: &Profile<'_>,
) -> f64 {
    (best_response_value(game, reward, profile, 0) + best_response_value(game, reward, profile, 1))
        / 2.0
}
