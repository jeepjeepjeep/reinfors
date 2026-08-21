//! Exact best response and exploitability against a fixed strategy profile.

use std::collections::HashMap;

use crate::game::{Actor, Game};
use crate::policy::MAX_ENUMERATED_OUTCOMES;
use crate::reward::Reward;

/// Strategy probabilities aligned with the information set's legal-action order.
pub type Profile<'a> = dyn Fn(&[u8], usize) -> Vec<f64> + 'a;

/// Maximum nodes in an exact enumeration.
const MAX_TREE_NODES: usize = 4_000_000;

/// The game tree exceeded the exact-enumeration cap. This is a typed error because an oversized
/// but valid game is expected public input, not an internal invariant failure.
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
    Chance(Vec<(f64, Vec<f64>, usize)>),
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
        // Terminal nodes count because the cap bounds construction work, not recursion depth.
        fn ensure_capacity(&self) -> Result<(), EnumerationCapExceeded> {
            if self.nodes.len() >= MAX_TREE_NODES {
                return Err(EnumerationCapExceeded(format!(
                    "game tree exceeds the exact best-response cap ({MAX_TREE_NODES} nodes)"
                )));
            }
            Ok(())
        }

        fn add(&mut self, state: &G::State) -> Result<usize, EnumerationCapExceeded> {
            self.ensure_capacity()?;
            let idx = self.nodes.len();
            self.nodes.push(ArenaNode::Terminal);
            let node = match self.game.actor(state) {
                Actor::Chance => {
                    let dist = self.game.chance_node(state);
                    let Some(count) = dist.enumerable_count() else {
                        return Err(EnumerationCapExceeded(
                            "chance is sample-only; best response requires enumerable chance"
                                .to_string(),
                        ));
                    };
                    if count > MAX_ENUMERATED_OUTCOMES {
                        return Err(EnumerationCapExceeded(format!(
                            "chance fan {count} exceeds the enumeration cap"
                        )));
                    }
                    let probs: Vec<f64> =
                        dist.iter_probs().expect("enumerable checked above").collect();
                    let mut outcomes = Vec::with_capacity(probs.len());
                    for (i, p) in probs.into_iter().enumerate() {
                        let t = self.game.apply_chance_node(state, i);
                        let n = self.game.num_agents();
                        let r: Vec<f64> = (0..n)
                            .map(|p| crate::reward::edge_reward(self.reward, &t.events, p))
                            .collect();
                        let child = if t.terminal {
                            self.terminal()?
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
                            self.terminal()?
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

        fn terminal(&mut self) -> Result<usize, EnumerationCapExceeded> {
            self.ensure_capacity()?;
            let idx = self.nodes.len();
            self.nodes.push(ArenaNode::Terminal);
            Ok(idx)
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
    members: HashMap<Vec<u8>, Vec<(usize, f64)>>,
    choice: HashMap<Vec<u8>, usize>,
}

impl BrPass<'_> {
    /// Collect infoset members with counterfactual reach weights.
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

    /// Resolve values while sharing one action choice across each information set.
    // Perfect recall makes recursive resolution of deeper infoset choices well-founded.
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

/// Exact best-response value for one player.
pub fn best_response_value<G: Game>(
    game: &G,
    reward: &dyn Reward<Event = G::Event>,
    profile: &Profile<'_>,
    br_player: usize,
) -> Result<f64, EnumerationCapExceeded> {
    let arena = build_arena(game, reward)?;
    Ok(br_value_in(&arena, profile, br_player))
}

fn br_value_in(arena: &Arena, profile: &Profile<'_>, br_player: usize) -> f64 {
    let mut pass = BrPass {
        arena,
        profile,
        br_player,
        members: HashMap::new(),
        choice: HashMap::new(),
    };
    pass.collect(0, 1.0);
    pass.value(0)
}

/// `(infoset key, exemplar state, acting agent)` per reachable infoset.
/// One first-visited state per information key. The information-state contract guarantees every
/// member has the same features and ordered legal actions, so any member is a valid exemplar.
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
        // Terminal transitions count because the cap bounds traversal work.
        fn charge(&mut self) -> Result<(), EnumerationCapExceeded> {
            self.visited += 1;
            if self.visited >= MAX_TREE_NODES {
                return Err(EnumerationCapExceeded(format!(
                    "game tree exceeds the enumeration cap ({MAX_TREE_NODES} nodes) — the \
                     exploitability instrument is for enumerable games"
                )));
            }
            Ok(())
        }

        fn go(&mut self, state: &G::State) -> Result<(), EnumerationCapExceeded> {
            self.charge()?;
            match self.game.actor(state) {
                Actor::Chance => {
                    let dist = self.game.chance_node(state);
                    let Some(count) = dist.enumerable_count() else {
                        return Err(EnumerationCapExceeded(
                            "chance is sample-only; best response requires enumerable chance"
                                .to_string(),
                        ));
                    };
                    for outcome in 0..count {
                        let t = self.game.apply_chance_node(state, outcome);
                        if t.terminal {
                            self.charge()?;
                        } else {
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
                            self.charge()?;
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

/// Exact best-response value for every player.
pub fn best_response_values<G: Game>(
    game: &G,
    reward: &dyn Reward<Event = G::Event>,
    profile: &Profile<'_>,
) -> Result<Vec<f64>, EnumerationCapExceeded> {
    let arena = build_arena(game, reward)?;
    Ok((0..game.num_agents())
        .map(|p| br_value_in(&arena, profile, p))
        .collect())
}

/// Expected value for every player under `profile`.
pub fn profile_values<G: Game>(
    game: &G,
    reward: &dyn Reward<Event = G::Event>,
    profile: &Profile<'_>,
) -> Result<Vec<f64>, EnumerationCapExceeded> {
    let arena = build_arena(game, reward)?;
    Ok(profile_values_in(&arena, profile, game.num_agents()))
}

fn profile_values_in(arena: &Arena, profile: &Profile<'_>, n: usize) -> Vec<f64> {
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
    go(arena, profile, 0, n)
}

/// Sum of every player's unilateral best-response improvement.
pub fn nash_conv<G: Game>(
    game: &G,
    reward: &dyn Reward<Event = G::Event>,
    profile: &Profile<'_>,
) -> Result<f64, EnumerationCapExceeded> {
    let arena = build_arena(game, reward)?;
    let n = game.num_agents();
    let on_policy = profile_values_in(&arena, profile, n);
    Ok((0..n)
        .map(|p| br_value_in(&arena, profile, p) - on_policy[p])
        .sum())
}

/// NashConv divided by the number of players.
pub fn exploitability<G: Game>(
    game: &G,
    reward: &dyn Reward<Event = G::Event>,
    profile: &Profile<'_>,
) -> Result<f64, EnumerationCapExceeded> {
    Ok(nash_conv(game, reward, profile)? / game.num_agents() as f64)
}

#[cfg(test)]
mod cap_tests {
    use super::*;
    use crate::game::Transition;
    use crate::reward::Reward;

    struct WideFan {
        fan: usize,
    }
    impl Game for WideFan {
        type State = u8;
        type Event = f64;
        fn num_agents(&self) -> usize {
            2
        }
        fn action_count(&self) -> usize {
            self.fan
        }
        fn actor(&self, _s: &u8) -> Actor {
            Actor::Agent(0)
        }
        fn legal_actions(&self, _s: &u8, _agent: usize) -> Vec<usize> {
            (0..self.fan).collect()
        }
        fn step(&self, _s: &u8, _actions: &[usize]) -> Transition<u8, f64> {
            Transition {
                next_state: 0,
                events: vec![Some(0.0), None],
                terminal: true,
            }
        }
        fn information_states(&self) -> bool {
            true
        }
        fn information_state_key(&self, _state: &u8, _agent: usize) -> Vec<u8> {
            vec![0]
        }
        fn initial_state(&self) -> u8 {
            0
        }
    }

    struct NoReward;
    impl Reward for NoReward {
        type Event = f64;
        fn step_reward(&self, _event: &f64, _agent: usize) -> f64 {
            0.0
        }
    }

    fn uniform(_key: &[u8], legal: usize) -> Vec<f64> {
        vec![1.0 / legal as f64; legal]
    }

    #[test]
    fn a_wide_terminal_fan_cannot_slip_past_the_arena_cap() {
        let g = WideFan {
            fan: MAX_TREE_NODES + 10,
        };
        let err = best_response_value(&g, &NoReward, &uniform, 0).unwrap_err();
        assert!(err.to_string().contains("cap"), "{err}");
    }

    #[test]
    fn a_wide_terminal_fan_cannot_slip_past_the_infoset_walk_cap() {
        let g = WideFan {
            fan: MAX_TREE_NODES + 10,
        };
        let err = enumerate_infosets(&g).unwrap_err();
        assert!(err.to_string().contains("cap"), "{err}");
    }

    #[test]
    fn a_fan_under_the_cap_still_enumerates() {
        let g = WideFan { fan: 3 };
        assert_eq!(enumerate_infosets(&g).unwrap().len(), 1);
        let vals = nash_conv(&g, &NoReward, &uniform).unwrap();
        assert!(
            vals.abs() < 1e-12,
            "one uniform node over zero payouts: {vals}"
        );
    }
}
