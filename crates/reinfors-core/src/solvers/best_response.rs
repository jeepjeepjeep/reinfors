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

use crate::game::{Actor, Game};
use crate::policy::MAX_ENUMERATED_OUTCOMES;
use crate::reward::Reward;

/// A strategy profile queried by information-set key: `(key, legal action count) -> probs`
/// (aligned with the game's `legal_actions` order).
pub type Profile<'a> = dyn Fn(&[u8], usize) -> Vec<f64> + 'a;

/// The arena cap — best response is exact enumeration; past this the game is out of this
/// instrument's scope.
const MAX_TREE_NODES: usize = 4_000_000;

/// The exact enumeration outgrew a cap — the game is out of exact-metric range. A typed
/// error rather than a panic: a big-but-valid game is expected input at the public boundary,
/// and a panic there would either escape it or force callers to catch unwinds (masking
/// genuine bugs and printing panic diagnostics on the way).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnumerationCapExceeded(pub String);

impl std::fmt::Display for EnumerationCapExceeded {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for EnumerationCapExceeded {}

enum ArenaNode {
    Terminal,
    /// `(probability, per-player edge rewards, child)` per outcome.
    Chance(Vec<(f64, Vec<f64>, usize)>),
    /// A decision: `(actor, infoset key, (per-player edge rewards, child) per legal action)`.
    Decision {
        who: usize,
        key: Vec<u8>,
        children: Vec<(Vec<f64>, usize)>,
    },
}

struct Arena {
    nodes: Vec<ArenaNode>,
}

fn build_arena<G: Game>(
    game: &G,
    reward: &dyn Reward<Event = G::Event>,
) -> Result<Arena, EnumerationCapExceeded> {
    struct Builder<'a, G: Game> {
        game: &'a G,
        reward: &'a dyn Reward<Event = G::Event>,
        nodes: Vec<ArenaNode>,
    }
    impl<G: Game> Builder<'_, G> {
        fn add(&mut self, state: &G::State) -> Result<usize, EnumerationCapExceeded> {
            if self.nodes.len() >= MAX_TREE_NODES {
                return Err(EnumerationCapExceeded(format!(
                    "game tree exceeds the exact best-response cap ({MAX_TREE_NODES} nodes)"
                )));
            }
            let idx = self.nodes.len();
            self.nodes.push(ArenaNode::Terminal); // placeholder, patched below
            let node = match self.game.actor(state) {
                Actor::Chance => {
                    let dist = self.game.chance_node(state);
                    if dist.count() > MAX_ENUMERATED_OUTCOMES {
                        return Err(EnumerationCapExceeded(format!(
                            "chance fan {} exceeds the enumeration cap",
                            dist.count()
                        )));
                    }
                    let probs: Vec<f64> = dist.iter_probs().collect();
                    let mut outcomes = Vec::with_capacity(probs.len());
                    for (i, p) in probs.into_iter().enumerate() {
                        let t = self.game.apply_chance_node(state, i);
                        let n = self.game.num_agents();
                        let r: Vec<f64> = (0..n)
                            .map(|p| crate::reward::edge_reward(self.reward, &t.events, p))
                            .collect();
                        let child = if t.terminal {
                            self.terminal()
                        } else {
                            self.add(&t.next_state)?
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
                        let mut joint = vec![0; self.game.num_agents()];
                        joint[who] = a;
                        let t = self.game.step(state, &joint);
                        let n = self.game.num_agents();
                        let r: Vec<f64> = (0..n)
                            .map(|p| crate::reward::edge_reward(self.reward, &t.events, p))
                            .collect();
                        let child = if t.terminal {
                            self.terminal()
                        } else {
                            self.add(&t.next_state)?
                        };
                        children.push((r, child));
                    }
                    ArenaNode::Decision { who, key, children }
                }
                Actor::Simultaneous => panic!("a simultaneous decision was reached mid-game: solvers support uniformly SEQUENTIAL games (the framework assumes one dynamics per game; mixing violates that contract)"),
            };
            self.nodes[idx] = node;
            Ok(idx)
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
    let root = game.initial_state();
    b.add(&root)?;
    Ok(Arena { nodes: b.nodes })
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
                    let (r, child) = children[a].clone();
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
                    ArenaNode::Decision { children, .. } => children[a].clone(),
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
) -> Result<f64, EnumerationCapExceeded> {
    let arena = build_arena(game, reward)?;
    let mut pass = BrPass {
        arena: &arena,
        profile,
        br_player,
        members: HashMap::new(),
        choice: HashMap::new(),
    };
    pass.collect(0, 1.0);
    Ok(pass.value(0))
}

/// Every reachable information set, with one exemplar state and the acting agent — the
/// enumeration behind the net-policy exploitability instrument (a full capped tree walk over
/// declared chance nodes; enumerable games only). First-visit exemplar per key; the key
/// contract guarantees every member state yields the same features and ordered legal list.
/// `(infoset key, exemplar state, acting agent)` per reachable infoset.
pub type InfosetExemplars<S> = Vec<(Vec<u8>, S, usize)>;

pub fn enumerate_infosets<G: Game>(
    game: &G,
) -> Result<InfosetExemplars<G::State>, EnumerationCapExceeded> {
    struct Walk<'a, G: Game> {
        game: &'a G,
        seen: HashMap<Vec<u8>, ()>,
        out: Vec<(Vec<u8>, G::State, usize)>,
        visited: usize,
    }
    impl<G: Game> Walk<'_, G> {
        fn go(&mut self, state: &G::State) -> Result<(), EnumerationCapExceeded> {
            self.visited += 1;
            if self.visited >= MAX_TREE_NODES {
                return Err(EnumerationCapExceeded(format!(
                    "game tree exceeds the enumeration cap ({MAX_TREE_NODES} nodes) — the \
                     exploitability instrument is for enumerable games"
                )));
            }
            match self.game.actor(state) {
                Actor::Chance => {
                    let dist = self.game.chance_node(state);
                    for outcome in 0..dist.count() {
                        let t = self.game.apply_chance_node(state, outcome);
                        if !t.terminal {
                            self.go(&t.next_state)?;
                        }
                    }
                    Ok(())
                }
                Actor::Agent(who) => {
                    let key = self.game.information_state_key(state, who);
                    if self.seen.insert(key.clone(), ()).is_none() {
                        self.out.push((key, state.clone(), who));
                    }
                    for a in self.game.legal_actions(state, who) {
                        let mut joint = vec![0; self.game.num_agents()];
                        joint[who] = a;
                        let t = self.game.step(state, &joint);
                        if t.terminal {
                            continue;
                        }
                        self.go(&t.next_state)?;
                    }
                    Ok(())
                }
                Actor::Simultaneous => panic!("a simultaneous decision was reached mid-game: solvers support uniformly SEQUENTIAL games (the framework assumes one dynamics per game; mixing violates that contract)"),
            }
        }
    }
    let mut walk = Walk {
        game,
        seen: HashMap::new(),
        out: Vec::new(),
        visited: 0,
    };
    let root = game.initial_state();
    walk.go(&root)?;
    Ok(walk.out)
}

/// Every player's exact best-response value against the others playing `profile`.
pub fn best_response_values<G: Game>(
    game: &G,
    reward: &dyn Reward<Event = G::Event>,
    profile: &Profile<'_>,
) -> Result<Vec<f64>, EnumerationCapExceeded> {
    (0..game.num_agents())
        .map(|p| best_response_value(game, reward, profile, p))
        .collect()
}

/// Every player's expected value when EVERYONE plays `profile` (one arena walk).
pub fn profile_values<G: Game>(
    game: &G,
    reward: &dyn Reward<Event = G::Event>,
    profile: &Profile<'_>,
) -> Result<Vec<f64>, EnumerationCapExceeded> {
    let arena = build_arena(game, reward)?;
    fn go(arena: &Arena, profile: &Profile<'_>, idx: usize, n: usize) -> Vec<f64> {
        match &arena.nodes[idx] {
            ArenaNode::Terminal => vec![0.0; n],
            ArenaNode::Chance(outcomes) => {
                let mut v = vec![0.0; n];
                for (p, r, child) in outcomes {
                    let c = go(arena, profile, *child, n);
                    for i in 0..n {
                        v[i] += p * (r[i] + c[i]);
                    }
                }
                v
            }
            ArenaNode::Decision { key, children, .. } => {
                let sigma = profile(key, children.len());
                let mut v = vec![0.0; n];
                for (ai, (r, child)) in children.iter().enumerate() {
                    let c = go(arena, profile, *child, n);
                    for i in 0..n {
                        v[i] += sigma[ai] * (r[i] + c[i]);
                    }
                }
                v
            }
        }
    }
    Ok(go(&arena, profile, 0, game.num_agents()))
}

/// NashConv: `Σᵢ (brᵢ − vᵢ)` — every player's exact unilateral improvement over `profile`,
/// summed. Zero exactly at a Nash equilibrium; for more than 2 players a positive plateau is
/// the expected outcome (regret minimization carries no equilibrium guarantee there).
pub fn nash_conv<G: Game>(
    game: &G,
    reward: &dyn Reward<Event = G::Event>,
    profile: &Profile<'_>,
) -> Result<f64, EnumerationCapExceeded> {
    let on_policy = profile_values(game, reward, profile)?;
    Ok(best_response_values(game, reward, profile)?
        .into_iter()
        .zip(on_policy)
        .map(|(br, v)| br - v)
        .sum())
}

/// pyspiel's exploitability: NashConv / num_players — for a 2-player zero-sum game exactly
/// the historical `(br_0 + br_1) / 2`.
pub fn exploitability<G: Game>(
    game: &G,
    reward: &dyn Reward<Event = G::Event>,
    profile: &Profile<'_>,
) -> Result<f64, EnumerationCapExceeded> {
    Ok(nash_conv(game, reward, profile)? / game.num_agents() as f64)
}
