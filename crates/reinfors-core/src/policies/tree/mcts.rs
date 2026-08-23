//! Pooled UCT and PUCT search over sequential, simultaneous, and chance nodes.

use std::collections::HashMap;

use crate::codec::bytes::Reader;
use crate::encoder::{ActionView, StateEncoder};
use crate::game::{Actor, ChanceDist, Game, Rng, Transition};
use crate::policies::tree::expectimax::search::SearchStats;
use crate::policies::tree::expectimax::{decode_search_eval, encode_search_eval, SearchEvaluation};
use crate::policy::{argmax, fold_search_stats, ply_from_u64, Policy, MAX_ENUMERATED_OUTCOMES};
use crate::reward::Reward;
use crate::rng::{dirichlet, SplitMix64};
use crate::rollout::engine::CollectStats;
use crate::rollout::evaluator::{CommittedRows, EvalBatch, Evaluator, Resolve};

/// Search guidance and leaf-output contract.
#[derive(Clone)]
pub(crate) enum Guidance {
    Uct {
        c: f64,
    },
    Puct {
        c: f64,
        noise: Option<(f64, f64, u64)>,
        noise_all: bool,
    },
}

/// Backup scheme for sequential games. Simultaneous games always use DUCT. In a two-seed Connect4
/// comparison, `Auto`'s negamax path scored about 0.60 against forced `MaxN` at about 35% less wall
/// time; `MaxN` remains the remeasurement seam.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum SequentialBackup {
    #[default]
    Auto,
    MaxN,
}

/// Root priors that receive noise in simultaneous search.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum NoiseScope {
    #[default]
    Requester,
    All,
}

use crate::policy::ChanceMode;

fn softmax(logits: &[f64]) -> Vec<f64> {
    let m = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|&l| (l - m).exp()).collect();
    let total: f64 = exps.iter().sum();
    exps.into_iter().map(|e| e / total).collect()
}

/// Rule used to select an action from the finished tree.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ActBy {
    Value,
    Visits,
}

#[derive(Clone, Copy, Debug)]
pub struct MctsConfig {
    pub num_simulations: usize,
    pub uct_c: f64,
    pub gamma: f64,
    pub max_depth: i32,
    /// Acting temperature during the opening moves; zero is greedy.
    pub temperature: f64,
    /// Number of opening plies to which the temperature applies.
    pub temperature_drop: u32,
    pub chance: ChanceMode,
}

pub struct Mcts {
    cfg: MctsConfig,
    act_by: ActBy,
}

pub use crate::policy::MAX_JOINT_SLOTS;

impl Mcts {
    pub fn new(cfg: MctsConfig, act_by: ActBy) -> Self {
        Mcts { cfg, act_by }
    }
}

/// Decision-node arrays are parallel and sparse over legal actions.
struct Node<S> {
    state: S,
    actor: usize,
    depth: i32,
    terminal: bool,
    kind: NodeKind,
    actions: Vec<usize>,
    child: Vec<i64>,
    reward: Vec<f64>,
    visits: Vec<u32>,
    value_sum: Vec<f64>,
    total_visits: u32,
    value: f64,
    obs: Vec<f32>,
    prior: Vec<f64>,
    obs_all: Vec<Vec<f32>>,
    values_all: Vec<f64>,
    rewards_all: Vec<f64>,
    chance_in: Vec<f64>,
}

enum NodeKind {
    Decision,
    Chance {
        dist: ChanceDist,
        committed: Vec<usize>,
        // Combinatorial outcome spaces make a scanned vector quadratic in simulations.
        resampled: HashMap<usize, usize>,
        // Non-empty only when an ExpandAll chance chain has been flattened.
        fan_weights: Vec<f64>,
    },
    Simultaneous(Box<SimNode>),
}

struct AgentTable {
    actions: Vec<usize>,
    visits: Vec<u32>,
    value_sum: Vec<f64>,
    prior: Vec<f64>,
    obs: Vec<f32>,
    value: f64,
    total_visits: u32,
}

/// Dense mixed-radix joint-action storage for a simultaneous node.
struct SimNode {
    tables: Vec<AgentTable>,
    child: Vec<i64>,
    reward: Vec<f64>,
}

impl SimNode {
    fn each_table_slot(&self, slot: usize, mut f: impl FnMut(usize, usize)) {
        let mut s = slot;
        for ag in (0..self.tables.len()).rev() {
            let l = self.tables[ag].actions.len();
            f(ag, s % l);
            s /= l;
        }
    }
    fn evaluated(&self) -> bool {
        !self.tables[0].prior.is_empty()
    }
}

impl<S> Node<S> {
    fn leaf(
        state: S,
        actor: usize,
        depth: i32,
        terminal: bool,
        actions: Vec<usize>,
        obs: Vec<f32>,
    ) -> Node<S> {
        let actions = if terminal { Vec::new() } else { actions };
        debug_assert!(
            terminal || !actions.is_empty(),
            "non-terminal node with no legal actions — the game must mark such states terminal"
        );
        let width = actions.len();
        Node {
            state,
            actor,
            depth,
            terminal,
            kind: NodeKind::Decision,
            actions,
            child: vec![-1; width],
            reward: vec![0.0; width],
            visits: vec![0; width],
            value_sum: vec![0.0; width],
            total_visits: 0,
            value: 0.0,
            obs,
            prior: Vec::new(),
            obs_all: Vec::new(),
            values_all: Vec::new(),
            rewards_all: Vec::new(),
            chance_in: Vec::new(),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum TreeMode {
    // The 2p default; keep its measurement rationale canonical on SequentialBackup.
    SeqNegamax,
    SeqMaxN,
    Sim,
}

enum Reached {
    Eval,
    Terminal,
    DepthCapped,
    Fan,
}

enum Expanded {
    Leaf(usize),
    Chance(usize),
    Fan(usize),
}

enum PendingWork {
    Fan,
    NodeEval(usize),
}

struct Tree<S> {
    arena: Vec<Node<S>>,
    sims: usize,
    path: Vec<(usize, usize)>,
    leaf: usize,
    requester: usize,
    mode: TreeMode,
    n_agents: usize,
    max_depth_seen: i32,
    rng: SplitMix64,
    pending: Option<(PendingWork, usize)>,
    terminal_sims: usize,
    depthcap_sims: usize,
    shared_rows: usize,
    fresh_rows: usize,
    hit_rows: usize,
    extra_eval_rows: usize,
    // Reused because these backup paths run once per simulation.
    g_buf: Vec<f64>,
    val_buf: Vec<f64>,
    pend_buf: Vec<f64>,
    // Chance rewards join the parent decision edge before its discount.
    pend_seed: Vec<f64>,
}

fn per_agent_chance_rewards<G: Game>(
    reward: &dyn Reward<Event = G::Event>,
    n: usize,
    t: &Transition<G::State, G::Event>,
) -> Vec<f64> {
    if t.events.iter().all(Option::is_none) {
        return Vec::new();
    }
    (0..n)
        .map(|ag| crate::reward::edge_reward(reward, &t.events, ag))
        .collect()
}

type FanLeaf<S> = (S, f64, Vec<f64>, bool);

fn merge_chance_rewards(a: &[f64], b: Vec<f64>, n: usize) -> Vec<f64> {
    if b.is_empty() {
        return a.to_vec();
    }
    let mut out = if a.is_empty() {
        vec![0.0; n]
    } else {
        a.to_vec()
    };
    for (o, v) in out.iter_mut().zip(&b) {
        *o += v;
    }
    out
}

fn flatten_chance_fan<G: Game>(
    game: &G,
    reward: &dyn Reward<Event = G::Event>,
    n_agents: usize,
    seed: Vec<FanLeaf<G::State>>,
) -> (Vec<FanLeaf<G::State>>, bool) {
    let mut leaves: Vec<FanLeaf<G::State>> = Vec::with_capacity(seed.len());
    let mut chained = false;
    // One shared stack makes the projected-size check include unprocessed siblings. Reverse seed
    // insertion preserves the original outcome order when the stack is popped.
    let mut stack: Vec<(FanLeaf<G::State>, usize)> =
        seed.into_iter().rev().map(|e| (e, 0)).collect();
    {
        while let Some(((s, p, ci, term), hops)) = stack.pop() {
            if term || !matches!(game.actor(&s), Actor::Chance) {
                leaves.push((s, p, ci, term));
                continue;
            }
            chained = true;
            assert!(
                hops < crate::game::CHANCE_CHAIN_LIMIT,
                "chance-node chain exceeded {} edges — the game cycles through chance states",
                crate::game::CHANCE_CHAIN_LIMIT
            );
            let dist = game.chance_node(&s);
            let count = dist
                .enumerable_count()
                .expect("ExpandAll cannot expand sample-only chance; use a sampling chance mode");
            assert!(
                count <= MAX_ENUMERATED_OUTCOMES,
                "ExpandAll cannot enumerate {count} chance outcomes (bound {}); use a sampling chance mode for combinatorial outcome spaces",
                MAX_ENUMERATED_OUTCOMES
            );
            assert!(
                // The parent has already been popped, so this is the fan size after expansion.
                leaves.len() + stack.len() + count <= MAX_ENUMERATED_OUTCOMES,
                "a chance chain's flattened fan exceeds the enumeration bound ({}); use a narrower sampling mode",
                MAX_ENUMERATED_OUTCOMES
            );
            let probs: Vec<f64> = dist
                .iter_probs()
                .expect("enumerable checked above")
                .collect();
            for (i, q) in probs.into_iter().enumerate() {
                let ct = game.apply_chance_node(&s, i);
                let ci2 = merge_chance_rewards(
                    &ci,
                    per_agent_chance_rewards::<G>(reward, n_agents, &ct),
                    n_agents,
                );
                stack.push(((ct.next_state, p * q, ci2, ct.terminal), hops + 1));
            }
        }
    }
    (leaves, chained)
}

fn agent_table<G: Game>(
    game: &G,
    enc: &dyn StateEncoder<State = G::State>,
    state: &G::State,
    agent: usize,
) -> AgentTable {
    let mut actions = game.legal_actions(state, agent);
    if actions.is_empty() {
        // Inactive simultaneous agents retain one placeholder slot; the game must ignore action 0
        // for them, matching the engine's inactive-agent convention.
        actions = vec![0];
    }
    let width = actions.len();
    AgentTable {
        actions,
        visits: vec![0; width],
        value_sum: vec![0.0; width],
        prior: Vec::new(),
        obs: enc.encode(state, agent),
        value: 0.0,
        total_visits: 0,
    }
}

fn sim_leaf<G: Game>(
    game: &G,
    enc: &dyn StateEncoder<State = G::State>,
    state: G::State,
    depth: i32,
) -> Node<G::State> {
    let n = game.num_agents();
    let tables: Vec<AgentTable> = (0..n)
        .map(|ag| agent_table(game, enc, &state, ag))
        .collect();
    let width = tables
        .iter()
        .try_fold(1usize, |acc, t| {
            acc.checked_mul(t.actions.len())
                .filter(|&w| w <= MAX_JOINT_SLOTS)
        })
        .unwrap_or_else(|| {
            panic!("simultaneous joint space exceeds {MAX_JOINT_SLOTS} slots at one node")
        });
    let mut node = Node::leaf(state, 0, depth, false, vec![0], Vec::new());
    node.actions = Vec::new();
    node.child = Vec::new();
    node.kind = NodeKind::Simultaneous(Box::new(SimNode {
        tables,
        child: vec![-1; width],
        reward: vec![0.0; width * n],
    }));
    node
}

fn maxn_leaf<G: Game>(
    game: &G,
    enc: &dyn StateEncoder<State = G::State>,
    state: G::State,
    actor: usize,
    depth: i32,
) -> Node<G::State> {
    let n = game.num_agents();
    let legal = game.legal_actions(&state, actor);
    let obs_all: Vec<Vec<f32>> = (0..n).map(|ag| enc.encode(&state, ag)).collect();
    let mut node = Node::leaf(state, actor, depth, false, legal, Vec::new());
    node.rewards_all = vec![0.0; node.actions.len() * n];
    node.values_all = vec![0.0; n];
    node.obs_all = obs_all;
    node
}

impl<S: Clone> Tree<S> {
    fn new<G>(
        game: &G,
        enc: &dyn StateEncoder<State = S>,
        state: S,
        requester: usize,
        chance_seed: u64,
        force_maxn: bool,
    ) -> Tree<S>
    where
        G: Game<State = S>,
    {
        let n = game.num_agents();
        let (root, mode) = match game.actor(&state) {
            Actor::Agent(actor) if n > 2 || (force_maxn && n == 2) => {
                (maxn_leaf(game, enc, state, actor, 0), TreeMode::SeqMaxN)
            }
            Actor::Agent(actor) => {
                let obs = enc.encode(&state, actor);
                let legal = game.legal_actions(&state, actor);
                (
                    Node::leaf(state, actor, 0, false, legal, obs),
                    TreeMode::SeqNegamax,
                )
            }
            Actor::Simultaneous => (sim_leaf(game, enc, state, 0), TreeMode::Sim),
            Actor::Chance => panic!(
                "search roots must be realized decision states — the rollout realizes chance \
                 chains before any policy sees the state"
            ),
        };
        Tree {
            arena: vec![root],
            sims: 0,
            path: Vec::new(),
            leaf: 0,
            requester,
            mode,
            n_agents: n,
            max_depth_seen: 0,
            rng: SplitMix64::new(chance_seed),
            pending: None,
            terminal_sims: 0,
            depthcap_sims: 0,
            shared_rows: 0,
            fresh_rows: 0,
            hit_rows: 0,
            extra_eval_rows: 0,
            g_buf: Vec::new(),
            val_buf: Vec::new(),
            pend_buf: Vec::new(),
            pend_seed: Vec::new(),
        }
    }

    fn select_expand<G>(
        &mut self,
        game: &G,
        enc: &dyn StateEncoder<State = S>,
        reward: &dyn Reward<Event = G::Event>,
        max_depth: i32,
        guidance: &Guidance,
        chance: ChanceMode,
    ) -> Reached
    where
        G: Game<State = S>,
    {
        self.path.clear();
        let mut ni = 0;
        let mut chance_hops = 0usize;
        loop {
            if let NodeKind::Chance { .. } = self.arena[ni].kind {
                chance_hops += 1;
                assert!(
                    chance_hops <= crate::game::CHANCE_CHAIN_LIMIT,
                    "chance-node chain exceeded {} edges — the game cycles through chance states",
                    crate::game::CHANCE_CHAIN_LIMIT
                );
                let slot = self.pick_chance_slot(ni, chance);
                self.path.push((ni, slot));
                match self.chance_child(ni, slot) {
                    Some(child) => {
                        ni = child;
                        continue;
                    }
                    None => {
                        let child = self.materialize_outcome(game, enc, reward, ni, slot, chance);
                        if let NodeKind::Chance { .. } = self.arena[child].kind {
                            ni = child;
                            continue;
                        }
                        self.leaf = child;
                        return if self.arena[child].terminal {
                            Reached::Terminal
                        } else {
                            Reached::Eval
                        };
                    }
                }
            }
            chance_hops = 0;
            let node = &self.arena[ni];
            self.max_depth_seen = self.max_depth_seen.max(node.depth);
            if node.terminal {
                self.leaf = ni;
                return Reached::Terminal;
            }
            if let NodeKind::Simultaneous(sim) = &node.kind {
                if matches!(guidance, Guidance::Puct { .. }) && !sim.evaluated() {
                    self.leaf = ni;
                    return Reached::Eval;
                }
                if node.depth >= max_depth {
                    self.leaf = ni;
                    return Reached::DepthCapped;
                }
                let js = sim.tables.iter().fold(0usize, |acc, t| {
                    acc * t.actions.len() + select_table(t, guidance)
                });
                self.path.push((ni, js));
                if sim.child[js] < 0 {
                    match self.expand(game, enc, reward, ni, js, chance) {
                        Expanded::Leaf(child) => {
                            self.leaf = child;
                            return if self.arena[child].terminal {
                                Reached::Terminal
                            } else {
                                Reached::Eval
                            };
                        }
                        Expanded::Chance(cni) => {
                            ni = cni;
                            continue;
                        }
                        Expanded::Fan(cni) => {
                            self.leaf = cni;
                            return Reached::Fan;
                        }
                    }
                }
                let NodeKind::Simultaneous(sim) = &self.arena[ni].kind else {
                    unreachable!()
                };
                ni = sim.child[js] as usize;
                continue;
            }
            if matches!(guidance, Guidance::Puct { .. }) && node.prior.is_empty() {
                self.leaf = ni;
                return Reached::Eval;
            }
            if node.depth >= max_depth {
                self.leaf = ni;
                return Reached::DepthCapped;
            }
            let a = select_edge(node, guidance);
            self.path.push((ni, a));
            if node.child[a] < 0 {
                match self.expand(game, enc, reward, ni, a, chance) {
                    Expanded::Leaf(child) => {
                        self.leaf = child;
                        return if self.arena[child].terminal {
                            Reached::Terminal
                        } else {
                            Reached::Eval
                        };
                    }
                    Expanded::Chance(cni) => {
                        ni = cni;
                        continue;
                    }
                    Expanded::Fan(cni) => {
                        self.leaf = cni;
                        return Reached::Fan;
                    }
                }
            }
            ni = self.arena[ni].child[a] as usize;
        }
    }

    fn pick_chance_slot(&mut self, ni: usize, chance: ChanceMode) -> usize {
        let Tree { arena, rng, .. } = self;
        let NodeKind::Chance {
            dist,
            committed,
            fan_weights,
            ..
        } = &arena[ni].kind
        else {
            unreachable!("pick_chance_slot on a decision node");
        };
        match chance {
            ChanceMode::Committed { .. } => rng.below(committed.len()),
            ChanceMode::AlwaysResample | ChanceMode::ExpandAll => {
                if fan_weights.is_empty() {
                    dist.draw(rng)
                } else {
                    crate::rng::weighted_index(rng, fan_weights)
                }
            }
        }
    }

    fn chance_child(&self, ni: usize, slot: usize) -> Option<usize> {
        let node = &self.arena[ni];
        // Only AlwaysResample has an empty dense child array; it materializes a sparse outcome map.
        if node.child.is_empty() {
            let NodeKind::Chance { resampled, .. } = &node.kind else {
                unreachable!("chance_child on a decision node");
            };
            resampled.get(&slot).copied()
        } else if node.child[slot] >= 0 {
            Some(node.child[slot] as usize)
        } else {
            None
        }
    }

    fn materialize_outcome<G>(
        &mut self,
        game: &G,
        enc: &dyn StateEncoder<State = S>,
        reward: &dyn Reward<Event = G::Event>,
        cni: usize,
        slot: usize,
        chance: ChanceMode,
    ) -> usize
    where
        G: Game<State = S>,
    {
        let NodeKind::Chance { committed, .. } = &self.arena[cni].kind else {
            unreachable!("materialize_outcome on a decision node");
        };
        let outcome = if committed.is_empty() {
            slot
        } else {
            committed[slot]
        };
        let t = game.apply_chance_node(&self.arena[cni].state, outcome);
        let chance_in = per_agent_chance_rewards::<G>(reward, self.n_agents, &t);
        let mover = self.arena[cni].actor;
        let mut child = self.child_leaf(
            game,
            enc,
            t.next_state,
            mover,
            self.arena[cni].depth + 1,
            t.terminal,
            chance,
        );
        child.chance_in = chance_in;
        let idx = self.arena.len();
        self.arena.push(child);
        if self.arena[cni].child.is_empty() {
            let NodeKind::Chance { resampled, .. } = &mut self.arena[cni].kind else {
                unreachable!()
            };
            resampled.insert(slot, idx);
        } else {
            self.arena[cni].child[slot] = idx as i64;
        }
        idx
    }

    fn edge_joint<G>(&self, game: &G, ni: usize, ai: usize) -> (Vec<usize>, usize)
    where
        G: Game<State = S>,
    {
        match &self.arena[ni].kind {
            NodeKind::Simultaneous(sim) => {
                let mut joint = vec![0usize; sim.tables.len()];
                sim.each_table_slot(ai, |ag, s| joint[ag] = sim.tables[ag].actions[s]);
                (joint, 0)
            }
            _ => {
                let mover = self.arena[ni].actor;
                let mut joint = vec![0usize; game.num_agents()];
                joint[mover] = self.arena[ni].actions[ai];
                (joint, mover)
            }
        }
    }

    fn record_edge_reward<G>(
        &mut self,
        reward: &dyn Reward<Event = G::Event>,
        ni: usize,
        ai: usize,
        t: &Transition<S, G::Event>,
    ) where
        G: Game<State = S>,
    {
        match &mut self.arena[ni].kind {
            NodeKind::Simultaneous(sim) => {
                let n = sim.tables.len();
                for ag in 0..n {
                    sim.reward[ai * n + ag] = crate::reward::edge_reward(reward, &t.events, ag);
                }
            }
            _ => {
                let mover = self.arena[ni].actor;
                self.arena[ni].reward[ai] = crate::reward::edge_reward(reward, &t.events, mover);
                if self.mode == TreeMode::SeqMaxN {
                    let n = self.n_agents;
                    for ag in 0..n {
                        self.arena[ni].rewards_all[ai * n + ag] =
                            crate::reward::edge_reward(reward, &t.events, ag);
                    }
                }
            }
        }
    }

    fn set_edge_child(&mut self, ni: usize, ai: usize, idx: usize) {
        match &mut self.arena[ni].kind {
            NodeKind::Simultaneous(sim) => sim.child[ai] = idx as i64,
            _ => self.arena[ni].child[ai] = idx as i64,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn child_leaf<G>(
        &mut self,
        game: &G,
        enc: &dyn StateEncoder<State = S>,
        state: S,
        mover: usize,
        depth: i32,
        terminal: bool,
        chance: ChanceMode,
    ) -> Node<S>
    where
        G: Game<State = S>,
    {
        if terminal {
            return Node::leaf(state, mover, depth, true, Vec::new(), Vec::new());
        }
        match game.actor(&state) {
            Actor::Agent(actor) => {
                assert!(
                    self.mode != TreeMode::Sim,
                    "mixed simultaneous/sequential dynamics are not supported"
                );
                if self.mode == TreeMode::SeqMaxN {
                    return maxn_leaf(game, enc, state, actor, depth);
                }
                let obs = enc.encode(&state, actor);
                let legal = game.legal_actions(&state, actor);
                Node::leaf(state, actor, depth, false, legal, obs)
            }
            Actor::Simultaneous => {
                assert!(
                    self.mode == TreeMode::Sim,
                    "mixed simultaneous/sequential dynamics are not supported"
                );
                sim_leaf(game, enc, state, depth)
            }
            Actor::Chance => {
                let dist = game.chance_node(&state);
                let committed: Vec<usize> = match chance {
                    ChanceMode::Committed { samples } => (0..samples.max(1))
                        .map(|_| dist.draw(&mut self.rng))
                        .collect(),
                    _ => Vec::new(),
                };
                let width = match chance {
                    ChanceMode::Committed { .. } => committed.len(),
                    ChanceMode::ExpandAll => {
                        let count = dist.enumerable_count().expect(
                            "ExpandAll cannot expand sample-only chance; use a sampling chance mode",
                        );
                        assert!(
                            count <= MAX_ENUMERATED_OUTCOMES,
                            "ExpandAll cannot enumerate {count} chance outcomes (bound {}); use \
                             a sampling chance mode for combinatorial outcome spaces",
                            MAX_ENUMERATED_OUTCOMES
                        );
                        count
                    }
                    ChanceMode::AlwaysResample => 0,
                };
                let mut node = Node::leaf(state, mover, depth - 1, false, vec![0], Vec::new());
                node.kind = NodeKind::Chance {
                    dist,
                    committed,
                    resampled: HashMap::new(),
                    fan_weights: Vec::new(),
                };
                node.actions = Vec::new();
                node.child = vec![-1; width];
                node
            }
        }
    }

    fn expand<G>(
        &mut self,
        game: &G,
        enc: &dyn StateEncoder<State = S>,
        reward: &dyn Reward<Event = G::Event>,
        ni: usize,
        ai: usize,
        chance: ChanceMode,
    ) -> Expanded
    where
        G: Game<State = S>,
    {
        let (joint, mover) = self.edge_joint(game, ni, ai);
        let t = game.step(&self.arena[ni].state, &joint);
        self.record_edge_reward::<G>(reward, ni, ai, &t);
        let depth = self.arena[ni].depth + 1;
        let child = self.child_leaf(game, enc, t.next_state, mover, depth, t.terminal, chance);
        let is_chance = matches!(child.kind, NodeKind::Chance { .. });
        let idx = self.arena.len();
        self.arena.push(child);
        self.set_edge_child(ni, ai, idx);
        if is_chance {
            if let ChanceMode::ExpandAll = chance {
                self.materialize_explicit_fan(game, enc, reward, idx, chance);
                return Expanded::Fan(idx);
            }
            return Expanded::Chance(idx);
        }
        Expanded::Leaf(idx)
    }

    fn materialize_explicit_fan<G>(
        &mut self,
        game: &G,
        enc: &dyn StateEncoder<State = S>,
        reward: &dyn Reward<Event = G::Event>,
        cni: usize,
        chance: ChanceMode,
    ) where
        G: Game<State = S>,
    {
        let NodeKind::Chance { dist, .. } = &self.arena[cni].kind else {
            unreachable!("materialize_explicit_fan on a decision node");
        };
        let probs: Vec<f64> = dist
            .iter_probs()
            .expect("explicit fans are only materialized for enumerable chance")
            .collect();
        let mover = self.arena[cni].actor;
        let mut seed = Vec::with_capacity(probs.len());
        for (i, p) in probs.into_iter().enumerate() {
            let t = game.apply_chance_node(&self.arena[cni].state, i);
            let ci = per_agent_chance_rewards::<G>(reward, self.n_agents, &t);
            seed.push((t.next_state, p, ci, t.terminal));
        }
        let (leaves, chained) = flatten_chance_fan(game, reward, self.n_agents, seed);
        self.arena[cni].child = vec![-1; leaves.len()];
        let mut weights = Vec::with_capacity(leaves.len());
        for (slot, (state, p, ci, term)) in leaves.into_iter().enumerate() {
            let mut child = self.child_leaf(
                game,
                enc,
                state,
                mover,
                self.arena[cni].depth + 1,
                term,
                chance,
            );
            child.chance_in = ci;
            let idx = self.arena.len();
            self.arena.push(child);
            self.arena[cni].child[slot] = idx as i64;
            weights.push(p);
        }
        if chained {
            let NodeKind::Chance { fan_weights, .. } = &mut self.arena[cni].kind else {
                unreachable!()
            };
            *fan_weights = weights;
        }
    }

    fn backprop(&mut self, gamma: f64, leaf_value: f64) {
        self.arena[self.leaf].value = leaf_value;
        let mut g = leaf_value; // value from `g_actor`'s perspective
        let mut g_actor = self.arena[self.leaf].actor;
        // This path is selected only for the <=2-agent negamax mode; indexing this at N>2 panics.
        let mut pend = [0.0f64; 2];
        for (p, r) in pend.iter_mut().zip(&self.pend_seed) {
            *p += r;
        }
        self.pend_seed.clear();
        for &(ni, a) in self.path.iter().rev() {
            if let NodeKind::Chance { .. } = self.arena[ni].kind {
                if let Some(child) = self.chance_child(ni, a) {
                    for (p, r) in pend.iter_mut().zip(&self.arena[child].chance_in) {
                        *p += r;
                    }
                }
                continue; // transparent: perspective and value pass through unchanged
            }
            let node_actor = self.arena[ni].actor;
            let child_val = if g_actor == node_actor { g } else { -g };
            let q = self.arena[ni].reward[a] + pend[node_actor] + gamma * child_val;
            pend = [0.0; 2];
            self.arena[ni].value_sum[a] += q;
            self.arena[ni].visits[a] += 1;
            self.arena[ni].total_visits += 1;
            g = q; // now from node_actor's perspective, for the level above
            g_actor = node_actor;
        }
    }

    fn backup(&mut self, gamma: f64, vals: &[f64]) {
        match self.mode {
            TreeMode::Sim => self.backprop_sim(gamma, vals),
            TreeMode::SeqMaxN => self.backprop_maxn(gamma, vals),
            TreeMode::SeqNegamax => self.backprop(gamma, vals[0]),
        }
    }

    fn backup_terminal(&mut self, gamma: f64) {
        match self.mode {
            TreeMode::SeqNegamax => self.backprop(gamma, 0.0),
            _ => {
                let mut z = std::mem::take(&mut self.val_buf);
                z.clear();
                z.resize(self.n_agents, 0.0);
                self.backup(gamma, &z);
                self.val_buf = z;
            }
        }
    }

    fn backup_leaf_values(&mut self, gamma: f64) {
        let ni = self.leaf;
        match &self.arena[ni].kind {
            NodeKind::Simultaneous(_) => {
                let mut vals = std::mem::take(&mut self.val_buf);
                vals.clear();
                let NodeKind::Simultaneous(sim) = &self.arena[ni].kind else {
                    unreachable!()
                };
                vals.extend(sim.tables.iter().map(|t| t.value));
                self.backup(gamma, &vals);
                self.val_buf = vals;
            }
            _ if self.mode == TreeMode::SeqMaxN => {
                let mut vals = std::mem::take(&mut self.val_buf);
                vals.clear();
                vals.extend_from_slice(&self.arena[ni].values_all);
                self.backup(gamma, &vals);
                self.val_buf = vals;
            }
            _ => self.backprop(gamma, self.arena[ni].value),
        }
    }

    fn backprop_sim(&mut self, gamma: f64, leaf_vals: &[f64]) {
        let mut g = std::mem::take(&mut self.g_buf);
        g.clear();
        g.extend_from_slice(leaf_vals);
        let mut pend = std::mem::take(&mut self.pend_buf);
        pend.clear();
        pend.resize(self.n_agents, 0.0);
        for (p, r) in pend.iter_mut().zip(&self.pend_seed) {
            *p += r;
        }
        self.pend_seed.clear();
        for &(ni, slot) in self.path.iter().rev() {
            if matches!(self.arena[ni].kind, NodeKind::Chance { .. }) {
                if let Some(child) = self.chance_child(ni, slot) {
                    for (p, r) in pend.iter_mut().zip(&self.arena[child].chance_in) {
                        *p += r;
                    }
                }
                continue;
            }
            match &mut self.arena[ni].kind {
                NodeKind::Simultaneous(sim) => {
                    let n = sim.tables.len();
                    let mut s = slot;
                    for ag in (0..n).rev() {
                        let l = sim.tables[ag].actions.len();
                        let si = s % l;
                        s /= l;
                        let q = sim.reward[slot * n + ag] + pend[ag] + gamma * g[ag];
                        sim.tables[ag].value_sum[si] += q;
                        sim.tables[ag].visits[si] += 1;
                        sim.tables[ag].total_visits += 1;
                        g[ag] = q;
                    }
                    pend.iter_mut().for_each(|p| *p = 0.0);
                }
                _ => {
                    unreachable!("decision node on a simultaneous tree's path (mixed dynamics)")
                }
            }
        }
        self.g_buf = g;
        self.pend_buf = pend;
    }

    fn backprop_maxn(&mut self, gamma: f64, leaf_vals: &[f64]) {
        let n = self.n_agents;
        let mut g = std::mem::take(&mut self.g_buf);
        g.clear();
        g.extend_from_slice(leaf_vals);
        let mut pend = std::mem::take(&mut self.pend_buf);
        pend.clear();
        pend.resize(n, 0.0);
        for (p, r) in pend.iter_mut().zip(&self.pend_seed) {
            *p += r;
        }
        self.pend_seed.clear();
        for &(ni, a) in self.path.iter().rev() {
            if matches!(self.arena[ni].kind, NodeKind::Chance { .. }) {
                if let Some(child) = self.chance_child(ni, a) {
                    for (p, r) in pend.iter_mut().zip(&self.arena[child].chance_in) {
                        *p += r;
                    }
                }
                continue;
            }
            let node = &mut self.arena[ni];
            for (i, gi) in g.iter_mut().enumerate() {
                *gi = node.rewards_all[a * n + i] + pend[i] + gamma * *gi;
            }
            pend.iter_mut().for_each(|p| *p = 0.0);
            node.value_sum[a] += g[node.actor];
            node.visits[a] += 1;
            node.total_visits += 1;
        }
        self.g_buf = g;
        self.pend_buf = pend;
    }

    fn fan_backprop(&mut self, gamma: f64) {
        let cni = self.leaf;
        let NodeKind::Chance {
            dist, fan_weights, ..
        } = &self.arena[cni].kind
        else {
            unreachable!("fan_backprop on a decision node");
        };
        let fan_weights = fan_weights.clone();
        let dist = dist.clone();
        let n = self.n_agents;
        let mut mix = vec![
            0.0f64;
            match self.mode {
                TreeMode::SeqNegamax => 1,
                _ => n,
            }
        ];
        let mut reward_mix = vec![0.0f64; n];
        let fan_probs: Vec<f64> = if fan_weights.is_empty() {
            dist.iter_probs()
                .expect("weightless fans come from enumerable expansion")
                .take(self.arena[cni].child.len())
                .collect()
        } else {
            fan_weights
        };
        let mut ref_actor = self.arena[cni].actor;
        // Negamax child values use each outcome's mover perspective. Chance may change that mover,
        // so normalize all nonterminal values to one reference under the 2p zero-sum contract.
        if self.mode == TreeMode::SeqNegamax {
            for &c in &self.arena[cni].child {
                if !self.arena[c as usize].terminal {
                    ref_actor = self.arena[c as usize].actor;
                    break;
                }
            }
        }
        for (slot, &p) in fan_probs.iter().enumerate() {
            let child = self.arena[cni].child[slot] as usize;
            for (m, r) in reward_mix.iter_mut().zip(&self.arena[child].chance_in) {
                *m += p * r;
            }
            match self.mode {
                TreeMode::Sim => {
                    for (ag, m) in mix.iter_mut().enumerate() {
                        let v = match &self.arena[child].kind {
                            NodeKind::Simultaneous(sim) => sim.tables[ag].value,
                            _ => 0.0,
                        };
                        *m += p * v;
                    }
                }
                TreeMode::SeqMaxN => {
                    for (ag, m) in mix.iter_mut().enumerate() {
                        let v = self.arena[child].values_all.get(ag).copied().unwrap_or(0.0);
                        *m += p * v;
                    }
                }
                TreeMode::SeqNegamax => {
                    let c = &self.arena[child];
                    let v = if c.terminal {
                        0.0
                    } else if c.actor == ref_actor {
                        c.value
                    } else {
                        -c.value
                    };
                    mix[0] += p * v;
                }
            }
        }
        if self.mode == TreeMode::SeqNegamax {
            self.arena[cni].actor = ref_actor; // fan value: the outcome children's perspective
        }
        if reward_mix.iter().any(|&r| r != 0.0) {
            self.pend_seed = reward_mix;
        }
        self.backup(gamma, &mix);
    }

    fn evaluation(self, actions: usize) -> SearchEvaluation {
        let root = &self.arena[0];
        let mut values = vec![0.0f64; actions];
        let mut visits = vec![0.0f64; actions];
        let (r_actions, r_visits, r_sums) = match &root.kind {
            NodeKind::Simultaneous(sim) => {
                let t = &sim.tables[self.requester];
                (&t.actions, &t.visits, &t.value_sum)
            }
            _ => (&root.actions, &root.visits, &root.value_sum),
        };
        for (slot, &action) in r_actions.iter().enumerate() {
            if r_visits[slot] > 0 {
                values[action] = r_sums[slot] / f64::from(r_visits[slot]);
            }
            visits[action] = f64::from(r_visits[slot]);
        }
        let legal = r_actions.clone();
        let stats = SearchStats {
            max_depth: self.max_depth_seen,
            expansions: self.sims,
            leaves: self.sims,
            rounds: self.sims,
            sigma_sum: 0.0,
            terminal_sims: self.terminal_sims,
            depthcap_sims: self.depthcap_sims,
            shared_rows: self.shared_rows,
            fresh_rows: self.fresh_rows,
            hit_rows: self.hit_rows,
            extra_eval_rows: self.extra_eval_rows,
        };
        SearchEvaluation {
            values: vec![values],
            visits,
            interior: Vec::new(),
            legal,
            stats,
        }
    }
}

fn select_edge<S>(node: &Node<S>, guidance: &Guidance) -> usize {
    select_scored(
        &node.visits,
        &node.value_sum,
        &node.prior,
        node.total_visits,
        guidance,
    )
}

fn select_table(t: &AgentTable, guidance: &Guidance) -> usize {
    select_scored(&t.visits, &t.value_sum, &t.prior, t.total_visits, guidance)
}

fn select_scored(
    visits: &[u32],
    value_sum: &[f64],
    prior: &[f64],
    total_visits: u32,
    guidance: &Guidance,
) -> usize {
    let mut best = 0;
    let mut best_score = f64::NEG_INFINITY;
    for a in 0..visits.len() {
        let score = match guidance {
            Guidance::Uct { c } => {
                if visits[a] == 0 {
                    f64::INFINITY
                } else {
                    let n = f64::from(visits[a]);
                    let ln_n = f64::from(total_visits.max(1)).ln();
                    value_sum[a] / n + c * (ln_n / n).sqrt()
                }
            }
            Guidance::Puct { c, .. } => {
                let n = f64::from(visits[a]);
                let q = if visits[a] > 0 { value_sum[a] / n } else { 0.0 };
                let sqrt_total = f64::from(total_visits.max(1)).sqrt();
                q + c * prior[a] * sqrt_total / (1.0 + n)
            }
        };
        if score > best_score {
            best_score = score;
            best = a;
        }
    }
    best
}

fn leaf_value(
    q: &[f64],
    k: usize,
    a: usize,
    legal: &[usize],
    view: &dyn ActionView,
    agent: usize,
) -> f64 {
    legal
        .iter()
        .map(|&ai| {
            let hi = view.head_index(ai, agent); // q is net output: game id -> head slot
            (0..k).map(|h| q[h * a + hi]).sum::<f64>() / k as f64
        })
        .fold(f64::NEG_INFINITY, f64::max)
}

pub fn mcts_many<G, F>(
    game: &G,
    enc: &dyn StateEncoder<State = G::State>,
    reward: &dyn Reward<Event = G::Event>,
    cfg: &MctsConfig,
    requests: Vec<(G::State, usize)>,
    seed: u64,
    eval: &mut Evaluator<'_, F>,
) -> Vec<SearchEvaluation>
where
    G: Game + Sync,
    G::State: Send,
    F: FnMut(usize, Vec<f32>, usize) -> Vec<f64>,
{
    let guidance = Guidance::Uct { c: cfg.uct_c };
    search_many(
        game,
        enc,
        reward,
        cfg.num_simulations,
        cfg.gamma,
        cfg.max_depth,
        &guidance,
        cfg.chance,
        seed,
        requests,
        eval,
        false,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn search_many<G, F>(
    game: &G,
    enc: &dyn StateEncoder<State = G::State>,
    reward: &dyn Reward<Event = G::Event>,
    num_simulations: usize,
    gamma: f64,
    max_depth: i32,
    guidance: &Guidance,
    chance: ChanceMode,
    seed: u64,
    requests: Vec<(G::State, usize)>,
    eval: &mut Evaluator<'_, F>,
    force_maxn: bool,
) -> Vec<SearchEvaluation>
where
    G: Game + Sync,
    G::State: Send,
    F: FnMut(usize, Vec<f32>, usize) -> Vec<f64>,
{
    let mut pool = PooledSearch::new(
        game,
        enc,
        reward,
        num_simulations,
        gamma,
        max_depth,
        guidance.clone(),
        chance,
        seed,
        requests,
        force_maxn,
    );
    while !pool.finished() {
        let mut batch = eval.batch();
        pool.stage_round(&mut batch);
        let rows = batch.commit();
        pool.apply_rows(&rows);
    }
    pool.into_evaluations()
}

/// A pooled search whose round loop is owned by the caller: `stage_round` drives every tree
/// until it stages rows (or finishes), the caller commits the batch however it likes, and
/// `apply_rows` distributes the results. Schedulers can interleave rounds of several pools.
struct PooledSearch<'c, G: Game> {
    game: &'c G,
    enc: &'c dyn StateEncoder<State = G::State>,
    reward: &'c dyn Reward<Event = G::Event>,
    num_simulations: usize,
    gamma: f64,
    max_depth: i32,
    guidance: Guidance,
    chance: ChanceMode,
    a: usize,
    trees: Vec<Tree<G::State>>,
    consumers: Vec<Vec<(usize, usize, usize)>>,
    awaiting: bool,
}

impl<'c, G: Game> PooledSearch<'c, G> {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        game: &'c G,
        enc: &'c dyn StateEncoder<State = G::State>,
        reward: &'c dyn Reward<Event = G::Event>,
        num_simulations: usize,
        gamma: f64,
        max_depth: i32,
        guidance: Guidance,
        chance: ChanceMode,
        seed: u64,
        requests: Vec<(G::State, usize)>,
        force_maxn: bool,
    ) -> Self {
        assert!(
            game.num_agents() >= 1,
            "a game must have at least one agent"
        );
        assert!(
            game.perfect_information(),
            "tree search on a hidden-information game is clairvoyant: its values condition on state \
             the agents cannot observe; see {}",
            crate::COMPATIBILITY_DOCS
        );
        let a = game.action_count();
        let trees: Vec<Tree<G::State>> = requests
            .into_iter()
            .enumerate()
            .map(|(ti, (state, agent))| {
                let chance_seed =
                    seed ^ 0x53A3_C5A9_1D87_2F6B ^ (ti as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
                Tree::new(game, enc, state, agent, chance_seed, force_maxn)
            })
            .collect();
        assert!(
            // Q-derived UCT leaf values exist only at the evaluated agent's own turns. Sequential MaxN
            // needs every perspective; simultaneous search gives every agent its own table instead.
            !(matches!(&guidance, Guidance::Uct { .. })
                && trees.iter().any(|t| t.mode == TreeMode::SeqMaxN)),
            "UCT does not support this sequential player count; see {}",
            crate::COMPATIBILITY_DOCS
        );
        PooledSearch {
            game,
            enc,
            reward,
            num_simulations,
            gamma,
            max_depth,
            guidance,
            chance,
            a,
            trees,
            consumers: Vec::new(),
            awaiting: false,
        }
    }

    pub(crate) fn finished(&self) -> bool {
        !self.awaiting && !self.trees.iter().any(|t| t.sims < self.num_simulations)
    }

    /// Drive every unfinished tree until it stages rows into `batch` or completes its budget.
    /// Rounds alternate strictly: every `stage_round` must be answered by `apply_rows` before
    /// the next, and the batch must be fresh — ticket ids index the consumer table directly.
    pub(crate) fn stage_round<F>(&mut self, batch: &mut EvalBatch<'_, '_, F>)
    where
        F: FnMut(usize, Vec<f32>, usize) -> Vec<f64>,
    {
        assert!(
            !self.awaiting,
            "stage_round with a round outstanding: apply_rows must consume it first"
        );
        assert!(
            batch.is_empty(),
            "stage_round requires a fresh batch: ticket ids index the consumer table"
        );
        let (game, enc, reward) = (self.game, self.enc, self.reward);
        let (num_simulations, gamma, max_depth, chance, a) = (
            self.num_simulations,
            self.gamma,
            self.max_depth,
            self.chance,
            self.a,
        );
        let guidance = &self.guidance;

        for (ti, tree) in self.trees.iter_mut().enumerate() {
            while tree.sims < num_simulations {
                tree.sims += 1;
                match tree.select_expand(game, enc, reward, max_depth, guidance, chance) {
                    Reached::Terminal => {
                        tree.terminal_sims += 1;
                        tree.backup_terminal(gamma);
                    }
                    Reached::DepthCapped => {
                        tree.depthcap_sims += 1;
                        tree.backup_leaf_values(gamma);
                    }
                    Reached::Eval => {
                        let leaf = tree.leaf;
                        let multi_row = matches!(tree.arena[leaf].kind, NodeKind::Simultaneous(_))
                            || tree.mode == TreeMode::SeqMaxN;
                        if multi_row {
                            let n = tree.n_agents;
                            tree.extra_eval_rows += n - 1;
                            // Set the full count before cache hits can consume rows.
                            tree.pending = Some((PendingWork::NodeEval(leaf), n));
                            for ag in 0..n {
                                let obs = match &mut tree.arena[leaf].kind {
                                    NodeKind::Simultaneous(sim) => {
                                        std::mem::take(&mut sim.tables[ag].obs)
                                    }
                                    _ => std::mem::take(&mut tree.arena[leaf].obs_all[ag]),
                                };
                                match batch.resolve_or_stage(ag, &obs) {
                                    Resolve::Resolved(row) => {
                                        tree.hit_rows += 1;
                                        consume_row(
                                            tree, leaf, ag, &row, guidance, gamma, a, ti, enc,
                                        );
                                    }
                                    Resolve::Staged(ticket) => {
                                        stage(tree, ti, leaf, ag, ticket, &mut self.consumers);
                                    }
                                }
                            }
                            if tree.pending.is_some() {
                                break;
                            }
                        } else {
                            match batch
                                .resolve_or_stage(tree.arena[leaf].actor, &tree.arena[leaf].obs)
                            {
                                Resolve::Resolved(row) => {
                                    tree.hit_rows += 1;
                                    consume_row(tree, leaf, 0, &row, guidance, gamma, a, ti, enc);
                                }
                                Resolve::Staged(ticket) => {
                                    stage(tree, ti, leaf, 0, ticket, &mut self.consumers);
                                    break;
                                }
                            }
                        }
                    }
                    Reached::Fan => {
                        let cni = tree.leaf;
                        let kids: Vec<usize> = tree.arena[cni]
                            .child
                            .iter()
                            .map(|&c| c as usize)
                            .filter(|&c| !tree.arena[c].terminal)
                            .collect();
                        let rows_per_child = match tree.mode {
                            TreeMode::SeqNegamax => 1,
                            _ => tree.n_agents,
                        };
                        let total_rows = kids.len() * rows_per_child;
                        if total_rows == 0 {
                            tree.terminal_sims += 1;
                            tree.fan_backprop(gamma);
                            continue;
                        }
                        tree.extra_eval_rows += total_rows.saturating_sub(1);
                        // Set the full count before cache hits can consume rows.
                        tree.pending = Some((PendingWork::Fan, total_rows));
                        for child in kids {
                            for ag in 0..rows_per_child {
                                let obs = match &mut tree.arena[child].kind {
                                    NodeKind::Simultaneous(sim) => {
                                        std::mem::take(&mut sim.tables[ag].obs)
                                    }
                                    _ if tree.mode == TreeMode::SeqMaxN => {
                                        std::mem::take(&mut tree.arena[child].obs_all[ag])
                                    }
                                    _ => std::mem::take(&mut tree.arena[child].obs),
                                };
                                let row_player = if rows_per_child == 1 {
                                    tree.arena[child].actor
                                } else {
                                    ag
                                };
                                match batch.resolve_or_stage(row_player, &obs) {
                                    Resolve::Resolved(row) => {
                                        tree.hit_rows += 1;
                                        consume_row(
                                            tree, child, ag, &row, guidance, gamma, a, ti, enc,
                                        );
                                    }
                                    Resolve::Staged(ticket) => {
                                        stage(tree, ti, child, ag, ticket, &mut self.consumers);
                                    }
                                }
                            }
                        }
                        if tree.pending.is_some() {
                            break;
                        }
                    }
                }
            }
        }
        self.awaiting = true;
    }

    /// Distribute one committed round to every tree that staged rows in it.
    pub(crate) fn apply_rows(&mut self, rows: &CommittedRows) {
        assert!(self.awaiting, "apply_rows without a staged round");
        self.awaiting = false;
        let consumers = std::mem::take(&mut self.consumers);
        for (ticket, waiting) in consumers.iter().enumerate() {
            for &(ti, node, slot) in waiting {
                consume_row(
                    &mut self.trees[ti],
                    node,
                    slot,
                    rows.row(ticket),
                    &self.guidance,
                    self.gamma,
                    self.a,
                    ti,
                    self.enc,
                );
            }
        }
    }

    pub(crate) fn into_evaluations(self) -> Vec<SearchEvaluation> {
        assert!(
            self.finished(),
            "into_evaluations on an unfinished pool: outstanding rows or unspent budget"
        );
        let a = self.a;
        self.trees.into_iter().map(|t| t.evaluation(a)).collect()
    }
}

fn stage<S>(
    tree: &mut Tree<S>,
    ti: usize,
    node: usize,
    slot: usize,
    ticket: usize,
    consumers: &mut Vec<Vec<(usize, usize, usize)>>,
) {
    if ticket < consumers.len() {
        tree.shared_rows += 1;
        consumers[ticket].push((ti, node, slot));
    } else {
        tree.fresh_rows += 1;
        consumers.push(vec![(ti, node, slot)]);
    }
}

#[allow(clippy::too_many_arguments)]
fn consume_row<S: Clone>(
    tree: &mut Tree<S>,
    ni: usize,
    slot: usize,
    row_data: &[f64],
    guidance: &Guidance,
    gamma: f64,
    a: usize,
    ti: usize,
    view: &dyn ActionView,
) {
    // Fresh forwards, cache hits, and within-batch deduplication all terminate here; keeping one
    // ingestion path prevents cache behavior from changing search semantics.
    let is_sim_node = matches!(tree.arena[ni].kind, NodeKind::Simultaneous(_));
    let is_maxn_node = !is_sim_node && tree.mode == TreeMode::SeqMaxN;
    let row_agent = if is_sim_node || is_maxn_node {
        slot
    } else {
        tree.arena[ni].actor
    };
    let maxn_side_row = is_maxn_node && slot != tree.arena[ni].actor;
    let value = match guidance {
        Guidance::Uct { .. } => {
            let k = row_data.len() / a;
            let actions: &[usize] = match &tree.arena[ni].kind {
                NodeKind::Simultaneous(sim) => &sim.tables[slot].actions,
                _ => &tree.arena[ni].actions,
            };
            leaf_value(row_data, k, a, actions, view, row_agent)
        }
        Guidance::Puct { .. } if maxn_side_row => row_data[a],
        Guidance::Puct {
            noise, noise_all, ..
        } => {
            let (logits, value) = row_data.split_at(a);
            let node_actions: &[usize] = match &tree.arena[ni].kind {
                NodeKind::Simultaneous(sim) => &sim.tables[slot].actions,
                _ => &tree.arena[ni].actions,
            };
            let legal_logits: Vec<f64> = node_actions
                .iter()
                .map(|&act| logits[view.head_index(act, row_agent)])
                .collect();
            let mut prior = softmax(&legal_logits);
            let noised = ni == 0 && (!is_sim_node || slot == tree.requester || *noise_all);
            if noised {
                // Noise is applied after lookup: the cache must contain raw logits so root hits
                // reproduce the same per-tree noise as fresh rows.
                if let Some((eps, alpha, seed)) = noise {
                    if *eps > 0.0 {
                        let mut rng = SplitMix64::new(
                            seed ^ (ti as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
                                ^ (slot as u64).wrapping_mul(0xD1B5_4A32_D192_ED03),
                        );
                        let noise_draw = dirichlet(&mut rng, *alpha, prior.len());
                        for (p, d) in prior.iter_mut().zip(noise_draw) {
                            *p = (1.0 - eps) * *p + eps * d;
                        }
                    }
                }
            }
            match &mut tree.arena[ni].kind {
                NodeKind::Simultaneous(sim) => sim.tables[slot].prior = prior,
                _ => tree.arena[ni].prior = prior,
            }
            value[0]
        }
    };
    match &mut tree.arena[ni].kind {
        NodeKind::Simultaneous(sim) => sim.tables[slot].value = value,
        _ if is_maxn_node => tree.arena[ni].values_all[slot] = value,
        _ => tree.arena[ni].value = value,
    }
    match tree.pending.as_mut() {
        Some((_, missing)) => {
            if *missing > 0 {
                *missing -= 1;
            }
            if matches!(tree.pending, Some((_, 0))) {
                let Some((work, _)) = tree.pending.take() else {
                    unreachable!()
                };
                match work {
                    PendingWork::Fan => tree.fan_backprop(gamma),
                    PendingWork::NodeEval(sni) => {
                        debug_assert_eq!(sni, tree.leaf, "pending eval at a stale leaf");
                        tree.backup_leaf_values(gamma);
                    }
                }
            }
        }
        None => {
            debug_assert_eq!(ni, tree.leaf, "single-leaf row delivered to the wrong node");
            debug_assert!(
                !is_sim_node && !is_maxn_node,
                "multi-row evaluations must flow through a pending eval"
            );
            tree.backprop(gamma, value);
        }
    }
}

pub(crate) fn sample_visits(visits: &[f64], temperature: f64, rng: &mut dyn Rng) -> usize {
    // Max-normalization prevents low temperatures overflowing; unvisited actions retain zero mass.
    let n_max = visits.iter().fold(0.0f64, |m, &v| m.max(v));
    if n_max <= 0.0 {
        return 0;
    }
    let w: Vec<f64> = visits
        .iter()
        .map(|&n| {
            if n > 0.0 {
                (n / n_max).powf(1.0 / temperature)
            } else {
                0.0
            }
        })
        .collect();
    let total: f64 = w.iter().sum();
    let mut r = rng.unit() * total;
    for (a, &wa) in w.iter().enumerate() {
        r -= wa;
        if wa > 0.0 && r <= 0.0 {
            return a;
        }
    }
    argmax(&w)
}

impl Policy for Mcts {
    type Evaluation = SearchEvaluation;
    type PolicyState = u32;

    fn supports_imperfect_information(&self) -> bool {
        false
    }

    fn max_agents(&self, sequential: bool) -> Option<usize> {
        if sequential {
            Some(2)
        } else {
            None
        }
    }

    fn encode_eval(&self, eval: &SearchEvaluation, out: &mut Vec<u8>) {
        encode_search_eval(eval, out);
    }

    fn decode_eval(&self, r: &mut Reader, action_count: usize) -> Result<SearchEvaluation, String> {
        decode_search_eval(r, action_count, 1, true)
    }

    fn policy_state_to_u64(&self, s: &u32) -> u64 {
        u64::from(*s)
    }

    fn policy_state_from_u64(&self, v: u64) -> Result<u32, String> {
        ply_from_u64(v)
    }

    fn begin_episode(&self, _rng: &mut dyn Rng) -> u32 {
        0
    }

    fn evaluate<G, F>(
        &self,
        game: &G,
        enc: &dyn StateEncoder<State = G::State>,
        reward: &dyn Reward<Event = G::Event>,
        requests: Vec<(G::State, usize)>,
        seed: u64,
        _collect_interior: bool,
        eval: &mut Evaluator<'_, F>,
    ) -> Vec<SearchEvaluation>
    where
        G: Game + Sync,
        G::State: Send,
        F: FnMut(usize, Vec<f32>, usize) -> Vec<f64>,
    {
        mcts_many(game, enc, reward, &self.cfg, requests, seed, eval)
    }

    fn select(&self, eval: &SearchEvaluation, state: &mut u32, rng: &mut dyn Rng) -> usize {
        let move_idx = *state;
        *state += 1;
        if self.cfg.temperature > 0.0 && move_idx < self.cfg.temperature_drop {
            return sample_visits(&eval.visits, self.cfg.temperature, rng);
        }
        let row = match self.act_by {
            ActBy::Visits if !eval.visits.is_empty() => &eval.visits,
            _ => &eval.values[0],
        };
        debug_assert!(!eval.legal.is_empty());
        let mut best = eval.legal[0];
        for &a in &eval.legal {
            if row[a] > row[best] {
                best = a;
            }
        }
        best
    }

    fn fold_telemetry(&self, eval: &SearchEvaluation, stats: &mut CollectStats) {
        fold_search_stats(eval, stats);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct FakeRng(Vec<f64>);
    impl Rng for FakeRng {
        fn below(&mut self, _n: usize) -> usize {
            0
        }
        fn unit(&mut self) -> f64 {
            self.0.remove(0)
        }
    }

    fn eval(values: Vec<f64>, visits: Vec<f64>) -> SearchEvaluation {
        SearchEvaluation {
            values: vec![values],
            visits,
            interior: Vec::new(),
            legal: (0..2).collect(),

            stats: SearchStats {
                max_depth: 1,
                expansions: 1,
                leaves: 1,
                rounds: 1,
                sigma_sum: 0.0,
                ..SearchStats::default()
            },
        }
    }

    fn mcts(temperature: f64, temperature_drop: u32) -> Mcts {
        Mcts::new(
            MctsConfig {
                num_simulations: 8,
                uct_c: 1.4,
                gamma: 1.0,
                max_depth: 8,
                temperature,
                temperature_drop,
                chance: ChanceMode::AlwaysResample,
            },
            ActBy::Visits,
        )
    }

    #[test]
    fn sample_visits_is_proportional_at_temperature_one() {
        let visits = vec![1.0, 3.0, 0.0];
        assert_eq!(sample_visits(&visits, 1.0, &mut FakeRng(vec![0.1])), 0);
        assert_eq!(sample_visits(&visits, 1.0, &mut FakeRng(vec![0.9])), 1);
    }

    #[test]
    fn sample_visits_never_picks_unvisited() {
        for u in [0.0, 0.5, 0.999] {
            assert_eq!(
                sample_visits(&[0.0, 5.0, 0.0], 1.0, &mut FakeRng(vec![u])),
                1
            );
        }
    }

    #[test]
    fn low_temperature_approaches_greedy() {
        assert_eq!(sample_visits(&[1.0, 3.0], 0.05, &mut FakeRng(vec![0.5])), 1);
    }

    #[test]
    fn value_acting_never_picks_an_illegal_densified_zero() {
        let p = Mcts::new(
            MctsConfig {
                num_simulations: 8,
                uct_c: 1.4,
                gamma: 1.0,
                max_depth: 8,
                temperature: 0.0,
                temperature_drop: u32::MAX,
                chance: ChanceMode::AlwaysResample,
            },
            ActBy::Value,
        );
        let mut e = eval(vec![0.0, -0.5], vec![0.0, 6.0]);
        e.legal = vec![1];
        let mut moves = 0u32;
        assert_eq!(p.select(&e, &mut moves, &mut FakeRng(vec![])), 1);
    }

    #[test]
    fn zero_temperature_acts_greedily_and_ignores_rng() {
        let p = mcts(0.0, u32::MAX);
        let e = eval(vec![0.9, 0.1], vec![2.0, 6.0]);
        let mut moves = 0u32;
        assert_eq!(p.select(&e, &mut moves, &mut FakeRng(vec![])), 1);
        assert_eq!(moves, 1);
    }

    #[test]
    fn temperature_drop_switches_to_greedy() {
        let p = mcts(1.0, 1);
        let e = eval(vec![0.9, 0.1], vec![6.0, 2.0]);
        let mut moves = 0u32;
        assert_eq!(p.select(&e, &mut moves, &mut FakeRng(vec![0.99])), 1);
        assert_eq!(p.select(&e, &mut moves, &mut FakeRng(vec![])), 0);
    }

    #[test]
    fn softmax_is_a_distribution_and_orders_by_logit() {
        let p = softmax(&[1.0, 3.0, 2.0]);
        assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!(p[1] > p[2] && p[2] > p[0]);
        let u = softmax(&[5.0, 5.0]);
        assert!((u[0] - 0.5).abs() < 1e-12);
    }

    fn puct_node(prior: Vec<f64>, visits: Vec<u32>, value_sum: Vec<f64>) -> Node<()> {
        let a = prior.len();
        let total = visits.iter().sum();
        Node {
            state: (),
            actor: 0,
            depth: 0,
            terminal: false,
            kind: NodeKind::Decision,
            actions: (0..a).collect(),
            child: vec![-1; a],
            reward: vec![0.0; a],
            visits,
            value_sum,
            total_visits: total,
            value: 0.0,
            obs: Vec::new(),
            prior,
            obs_all: Vec::new(),
            values_all: Vec::new(),
            rewards_all: Vec::new(),
            chance_in: Vec::new(),
        }
    }

    #[test]
    fn puct_first_selection_is_prior_ordered() {
        let node = puct_node(vec![0.2, 0.5, 0.3], vec![0, 0, 0], vec![0.0; 3]);
        assert_eq!(
            select_edge(
                &node,
                &Guidance::Puct {
                    c: 1.5,
                    noise: None,
                    noise_all: false
                }
            ),
            1
        );
    }

    #[test]
    fn puct_high_visits_shrink_the_exploration_term() {
        let node = puct_node(vec![0.3, 0.7], vec![1, 99], vec![0.1, 9.9]);
        let g = Guidance::Puct {
            c: 2.0,
            noise: None,
            noise_all: false,
        };
        assert_eq!(select_edge(&node, &g), 0);
    }

    #[test]
    fn puct_exploits_value_at_equal_priors() {
        let node = puct_node(vec![0.5, 0.5], vec![10, 10], vec![9.0, 1.0]);
        assert_eq!(
            select_edge(
                &node,
                &Guidance::Puct {
                    c: 0.1,
                    noise: None,
                    noise_all: false
                }
            ),
            0
        );
    }
}

#[cfg(test)]
mod masking_tests {
    use super::*;
    use crate::encoder::StateEncoder;
    use crate::game::{Actor, Game, Transition};
    use crate::policies::tree::alphazero::{alphazero_many, AlphaZeroConfig};
    use crate::reward::Reward as RewardTrait;

    struct EvenOnly;
    #[derive(Clone)]
    struct St(i32);

    impl Game for EvenOnly {
        type State = St;
        type Event = f64;
        fn num_agents(&self) -> usize {
            1
        }
        fn action_count(&self) -> usize {
            10
        }
        fn actor(&self, _s: &St) -> Actor {
            Actor::Agent(0)
        }
        fn legal_actions(&self, _s: &St, _agent: usize) -> Vec<usize> {
            (0..10).filter(|a| a % 2 == 0).collect()
        }
        fn step(&self, s: &St, actions: &[usize]) -> Transition<St, f64> {
            let a = actions[0] as i32;
            assert!(a % 2 == 0, "search stepped an illegal (odd) action: {a}");
            let total = s.0 + a;
            Transition {
                next_state: St(total),
                events: vec![Some(if total >= 8 { 1.0 } else { 0.0 })],
                terminal: total >= 8,
            }
        }
        fn initial_state(&self) -> St {
            St(0)
        }
    }

    struct Enc;
    impl ActionView for Enc {}
    impl StateEncoder for Enc {
        type State = St;
        fn encode(&self, s: &St, _agent: usize) -> Vec<f32> {
            vec![s.0 as f32]
        }
        fn obs_shape(&self) -> (usize, usize, usize) {
            (1, 1, 1)
        }
    }

    struct Passthrough;
    impl RewardTrait for Passthrough {
        type Event = f64;
        fn step_reward(&self, event: &f64, _agent: usize) -> f64 {
            *event
        }
    }

    fn run(guidance_puct: bool, noise_eps: f64) -> SearchEvaluation {
        let mut infer = |_p: usize, _obs: Vec<f32>, n: usize| -> Vec<f64> {
            if guidance_puct {
                vec![0.0; n * 11]
            } else {
                vec![0.0; n * 10]
            }
        };
        let mut eval = Evaluator::new(
            &mut infer,
            crate::rollout::evaluator::InferMode::Shared,
            None,
        );
        let evals = if guidance_puct {
            let cfg = AlphaZeroConfig {
                num_simulations: 40,
                c_puct: 1.5,
                gamma: 1.0,
                max_depth: 8,
                noise_epsilon: noise_eps,
                noise_alpha: 0.3,
                temperature: 0.0,
                temperature_drop: u32::MAX,
                chance: ChanceMode::AlwaysResample,
                noise_scope: NoiseScope::Requester,
                sequential_backup: Default::default(),
            };
            alphazero_many(
                &EvenOnly,
                &Enc,
                &Passthrough,
                &cfg,
                vec![(St(0), 0)],
                7,
                &mut eval,
            )
        } else {
            let cfg = MctsConfig {
                num_simulations: 40,
                uct_c: 1.4,
                gamma: 1.0,
                max_depth: 8,
                temperature: 0.0,
                temperature_drop: u32::MAX,
                chance: ChanceMode::AlwaysResample,
            };
            mcts_many(
                &EvenOnly,
                &Enc,
                &Passthrough,
                &cfg,
                vec![(St(0), 0)],
                7,
                &mut eval,
            )
        };
        evals.into_iter().next().unwrap()
    }

    #[test]
    fn uct_visits_only_legal_actions() {
        let eval = run(false, 0.0);
        for a in (1..10).step_by(2) {
            assert_eq!(eval.visits[a], 0.0, "illegal action {a} was visited");
            assert_eq!(eval.values[0][a], 0.0);
        }
        assert!(eval.visits.iter().sum::<f64>() > 0.0);
        let best = (0..10).max_by(|&x, &y| eval.visits[x].partial_cmp(&eval.visits[y]).unwrap());
        assert_eq!(best.unwrap() % 2, 0);
    }

    #[test]
    fn puct_visits_and_noise_stay_legal() {
        let eval = run(true, 0.9);
        for a in (1..10).step_by(2) {
            assert_eq!(
                eval.visits[a], 0.0,
                "illegal action {a} was visited under noise"
            );
        }
        assert!((eval.visits.iter().sum::<f64>() - 39.0).abs() < 1e-9);
    }
}

#[cfg(test)]
mod chance_tests {
    use super::*;
    use crate::encoder::StateEncoder;
    use crate::game::{Actor, Game, Transition};
    use crate::policies::tree::alphazero::{alphazero_many, AlphaZeroConfig};
    use crate::reward::Reward as RewardTrait;

    #[derive(Clone)]
    struct St {
        total: i32,
        ply: u8,
    }
    struct RiskyGame;

    impl Game for RiskyGame {
        type State = St;
        type Event = f64;
        fn num_agents(&self) -> usize {
            1
        }
        fn action_count(&self) -> usize {
            2
        }
        fn actor(&self, s: &St) -> Actor {
            if s.ply == 2 {
                Actor::Chance
            } else {
                Actor::Agent(0)
            }
        }
        fn legal_actions(&self, s: &St, _agent: usize) -> Vec<usize> {
            if s.ply == 2 {
                Vec::new()
            } else {
                vec![0, 1]
            }
        }
        fn step(&self, s: &St, actions: &[usize]) -> Transition<St, f64> {
            if s.ply == 0 {
                let (total, ply) = if actions[0] == 0 {
                    (s.total + 1, 1)
                } else {
                    (s.total, 2)
                };
                Transition {
                    next_state: St { total, ply },
                    events: vec![Some(0.0)],
                    terminal: false,
                }
            } else {
                Transition {
                    next_state: St { ..*s },
                    events: vec![Some(f64::from(s.total))],
                    terminal: true,
                }
            }
        }
        fn chance_node(&self, _s: &St) -> ChanceDist {
            ChanceDist::Weighted(vec![0.5, 0.5])
        }
        fn apply_chance_node(&self, s: &St, outcome: usize) -> Transition<St, f64> {
            Transition {
                next_state: St {
                    total: s.total + if outcome == 0 { 0 } else { 3 },
                    ply: 1,
                },
                events: vec![None],
                terminal: false,
            }
        }
        fn initial_state(&self) -> St {
            St { total: 0, ply: 0 }
        }
    }

    struct Enc;
    impl ActionView for Enc {}
    impl StateEncoder for Enc {
        type State = St;
        fn encode(&self, s: &St, _agent: usize) -> Vec<f32> {
            vec![s.total as f32, f32::from(s.ply)]
        }
        fn obs_shape(&self) -> (usize, usize, usize) {
            (1, 1, 2)
        }
    }

    struct Passthrough;
    impl RewardTrait for Passthrough {
        type Event = f64;
        fn step_reward(&self, event: &f64, _agent: usize) -> f64 {
            *event
        }
    }

    fn cfg(sims: usize, chance: ChanceMode) -> MctsConfig {
        MctsConfig {
            num_simulations: sims,
            uct_c: 3.0,
            gamma: 1.0,
            max_depth: 8,
            temperature: 0.0,
            temperature_drop: u32::MAX,
            chance,
        }
    }

    fn run_uct(sims: usize, chance: ChanceMode, seed: u64) -> SearchEvaluation {
        let mut infer = |_p: usize, _obs: Vec<f32>, n: usize| vec![0.0; n * 2];
        let mut eval = Evaluator::new(
            &mut infer,
            crate::rollout::evaluator::InferMode::Shared,
            None,
        );
        mcts_many(
            &RiskyGame,
            &Enc,
            &Passthrough,
            &cfg(sims, chance),
            vec![(St { total: 0, ply: 0 }, 0)],
            seed,
            &mut eval,
        )
        .remove(0)
    }

    #[test]
    fn always_resample_converges_to_the_expectation() {
        let eval = run_uct(600, ChanceMode::AlwaysResample, 7);
        assert!(
            (eval.values[0][1] - 1.5).abs() < 0.25,
            "Q(risky) should approach E = 1.5, got {}",
            eval.values[0][1]
        );
        assert!((eval.values[0][0] - 1.0).abs() < 0.25);
        assert!(
            eval.visits[1] > eval.visits[0],
            "risky (E=1.5) must out-visit safe (1)"
        );
    }

    #[test]
    fn committed_one_freezes_a_single_world() {
        let (mut saw_low, mut saw_high) = (false, false);
        for seed in 0..12 {
            let q = run_uct(200, ChanceMode::Committed { samples: 1 }, seed).values[0][1];
            assert!(
                q.abs() < 0.25 || (q - 3.0).abs() < 0.25,
                "Committed{{1}} must condition on one world (Q ~0 or ~3), got {q}"
            );
            saw_low |= q < 1.5;
            saw_high |= q > 1.5;
        }
        assert!(
            saw_low && saw_high,
            "both worlds should be drawn across seeds"
        );
    }

    #[test]
    fn committed_two_averages_its_frozen_draws() {
        for seed in 0..8 {
            let q = run_uct(400, ChanceMode::Committed { samples: 2 }, seed).values[0][1];
            let near = [0.0, 1.5, 3.0].iter().any(|m| (q - m).abs() < 0.25);
            assert!(
                near,
                "Committed{{2}} Q should sit near a frozen-pair mean, got {q}"
            );
        }
    }

    #[test]
    fn chance_search_is_deterministic_per_seed() {
        for chance in [
            ChanceMode::AlwaysResample,
            ChanceMode::Committed { samples: 2 },
            ChanceMode::ExpandAll,
        ] {
            let a = run_uct(100, chance, 42);
            let b = run_uct(100, chance, 42);
            assert_eq!(a.values, b.values);
            assert_eq!(a.visits, b.visits);
        }
    }

    #[test]
    fn expand_all_seeds_the_exact_expectation_immediately() {
        let mut infer = |_p: usize, obs: Vec<f32>, n: usize| -> Vec<f64> {
            (0..n).flat_map(|i| [f64::from(obs[i * 2]); 2]).collect()
        };
        let mut eval = Evaluator::new(
            &mut infer,
            crate::rollout::evaluator::InferMode::Shared,
            None,
        );
        let out = mcts_many(
            &RiskyGame,
            &Enc,
            &Passthrough,
            &cfg(2, ChanceMode::ExpandAll),
            vec![(St { total: 0, ply: 0 }, 0)],
            3,
            &mut eval,
        )
        .remove(0);
        assert_eq!(
            out.values[0][1], 1.5,
            "fan backup must be the exact weighted expectation"
        );
        assert_eq!(out.stats.extra_eval_rows, 1);
        let s = &out.stats;
        assert_eq!(
            2,
            s.fresh_rows + s.hit_rows + s.shared_rows + s.terminal_sims + s.depthcap_sims
                - s.extra_eval_rows
        );
    }

    #[test]
    fn puct_searches_chance_games() {
        let mut infer = |_p: usize, _obs: Vec<f32>, n: usize| vec![0.0; n * 3];
        let mut eval = Evaluator::new(
            &mut infer,
            crate::rollout::evaluator::InferMode::Shared,
            None,
        );
        let cfg = AlphaZeroConfig {
            num_simulations: 300,
            c_puct: 2.0,
            gamma: 1.0,
            max_depth: 8,
            noise_epsilon: 0.0,
            noise_alpha: 0.3,
            temperature: 0.0,
            temperature_drop: u32::MAX,
            chance: ChanceMode::AlwaysResample,
            noise_scope: NoiseScope::Requester,
            sequential_backup: Default::default(),
        };
        let out = alphazero_many(
            &RiskyGame,
            &Enc,
            &Passthrough,
            &cfg,
            vec![(St { total: 0, ply: 0 }, 0)],
            11,
            &mut eval,
        )
        .remove(0);
        assert!(
            out.visits[1] > out.visits[0],
            "PUCT should favor the E=1.5 risky action"
        );
    }
}

#[cfg(test)]
mod duct_tests {
    use super::*;
    use crate::encoder::StateEncoder;
    use crate::game::{Actor, Game, Transition};
    use crate::policies::tree::alphazero::{alphazero_many, AlphaZeroConfig};
    use crate::reward::Reward as RewardTrait;

    #[derive(Clone)]
    struct St {
        done: bool,
    }
    struct MatrixGame;

    impl Game for MatrixGame {
        type State = St;
        type Event = [f64; 2];
        fn num_agents(&self) -> usize {
            2
        }
        fn action_count(&self) -> usize {
            2
        }
        fn actor(&self, _s: &St) -> Actor {
            Actor::Simultaneous
        }
        fn legal_actions(&self, _s: &St, _agent: usize) -> Vec<usize> {
            vec![0, 1]
        }
        fn step(&self, _s: &St, actions: &[usize]) -> Transition<St, [f64; 2]> {
            let coord = f64::from(u8::from(actions[0] == actions[1]));
            Transition {
                next_state: St { done: true },
                events: vec![Some([coord, 0.0]), Some([f64::from(actions[1] as u8), 0.0])],
                terminal: true,
            }
        }
        fn initial_state(&self) -> St {
            St { done: false }
        }
    }

    struct Enc;
    impl ActionView for Enc {}
    impl StateEncoder for Enc {
        type State = St;
        fn encode(&self, s: &St, agent: usize) -> Vec<f32> {
            vec![f32::from(u8::from(s.done)), agent as f32]
        }
        fn obs_shape(&self) -> (usize, usize, usize) {
            (1, 1, 2)
        }
    }

    struct Passthrough;
    impl RewardTrait for Passthrough {
        type Event = [f64; 2];
        fn step_reward(&self, event: &[f64; 2], _agent: usize) -> f64 {
            event[0]
        }
    }

    fn az_cfg(sims: usize, eps: f64, scope: NoiseScope) -> AlphaZeroConfig {
        AlphaZeroConfig {
            num_simulations: sims,
            c_puct: 2.0,
            gamma: 1.0,
            max_depth: 8,
            noise_epsilon: eps,
            noise_alpha: 0.3,
            temperature: 0.0,
            temperature_drop: u32::MAX,
            chance: ChanceMode::AlwaysResample,
            noise_scope: scope,
            sequential_backup: Default::default(),
        }
    }

    #[test]
    fn uct_finds_each_requesters_dominant_action() {
        let mut infer = |_p: usize, _obs: Vec<f32>, n: usize| vec![0.0; n * 2];
        let mut eval = Evaluator::new(
            &mut infer,
            crate::rollout::evaluator::InferMode::Shared,
            None,
        );
        let cfg = MctsConfig {
            num_simulations: 60,
            uct_c: 2.0,
            gamma: 1.0,
            max_depth: 8,
            temperature: 0.0,
            temperature_drop: u32::MAX,
            chance: ChanceMode::AlwaysResample,
        };
        let evals = mcts_many(
            &MatrixGame,
            &Enc,
            &Passthrough,
            &cfg,
            vec![(St { done: false }, 0), (St { done: false }, 1)],
            5,
            &mut eval,
        );
        for (seat, e) in evals.iter().enumerate() {
            assert!(
                e.visits[1] > e.visits[0],
                "seat {seat} should favor action 1 (dominant / coordinate-with-dominant), visits {:?}",
                e.visits
            );
        }
    }

    #[test]
    fn puct_finds_the_dominant_action_and_is_deterministic() {
        let run = |seed: u64| {
            let mut infer = |_p: usize, _obs: Vec<f32>, n: usize| vec![0.0; n * 3];
            let mut eval = Evaluator::new(
                &mut infer,
                crate::rollout::evaluator::InferMode::Shared,
                None,
            );
            alphazero_many(
                &MatrixGame,
                &Enc,
                &Passthrough,
                &az_cfg(60, 0.25, NoiseScope::Requester),
                vec![(St { done: false }, 0)],
                seed,
                &mut eval,
            )
            .remove(0)
        };
        let a = run(3);
        assert!(a.visits[1] > a.visits[0]);
        let b = run(3);
        assert_eq!(a.visits, b.visits, "same seed, same simultaneous search");
        assert_eq!(a.values, b.values);
    }

    #[test]
    fn noise_scope_changes_only_the_opponent_stream() {
        let run = |eps: f64, scope: NoiseScope| {
            let mut infer = |_p: usize, obs: Vec<f32>, n: usize| -> Vec<f64> {
                (0..n)
                    .flat_map(|i| [f64::from(obs[i * 2 + 1]), 0.5, 0.0])
                    .collect()
            };
            let mut eval = Evaluator::new(
                &mut infer,
                crate::rollout::evaluator::InferMode::Shared,
                None,
            );
            alphazero_many(
                &MatrixGame,
                &Enc,
                &Passthrough,
                &az_cfg(40, eps, scope),
                vec![(St { done: false }, 0)],
                9,
                &mut eval,
            )
            .remove(0)
        };
        assert_eq!(
            run(0.0, NoiseScope::Requester).visits,
            run(0.0, NoiseScope::All).visits,
            "noise off: scope must be inert"
        );
        assert_ne!(
            run(0.6, NoiseScope::Requester).visits,
            run(0.6, NoiseScope::All).visits,
            "noise on: Both must perturb the opponent's selection stream"
        );
    }

    #[test]
    fn simultaneous_eval_rows_close_the_identity() {
        let mut infer = |_p: usize, _obs: Vec<f32>, n: usize| vec![0.0; n * 3];
        let mut eval = Evaluator::new(
            &mut infer,
            crate::rollout::evaluator::InferMode::Shared,
            None,
        );
        let out = alphazero_many(
            &MatrixGame,
            &Enc,
            &Passthrough,
            &az_cfg(20, 0.0, NoiseScope::Requester),
            vec![(St { done: false }, 0)],
            1,
            &mut eval,
        )
        .remove(0);
        let s = &out.stats;
        assert!(
            s.extra_eval_rows > 0,
            "two-perspective evals must record extra rows"
        );
        assert_eq!(
            20,
            s.fresh_rows + s.hit_rows + s.shared_rows + s.terminal_sims + s.depthcap_sims
                - s.extra_eval_rows
        );
    }

    struct SwapFor1;
    impl ActionView for SwapFor1 {
        fn head_index(&self, action: usize, agent: usize) -> usize {
            if agent == 1 {
                1 - action
            } else {
                action
            }
        }
        fn game_action(&self, head: usize, agent: usize) -> usize {
            if agent == 1 {
                1 - head
            } else {
                head
            }
        }
    }
    impl StateEncoder for SwapFor1 {
        type State = St;
        fn encode(&self, s: &St, agent: usize) -> Vec<f32> {
            vec![f32::from(u8::from(s.done)), agent as f32]
        }
        fn obs_shape(&self) -> (usize, usize, usize) {
            (1, 1, 2)
        }
    }

    #[test]
    fn duct_tables_gather_through_each_agents_own_view() {
        let game_logits = |seat: usize| {
            if seat == 0 {
                [1.2, -0.4]
            } else {
                [-0.8, 0.9]
            }
        };
        let states = || vec![(St { done: false }, 0), (St { done: false }, 1)];
        let run = |swapped: bool| {
            let mut infer = move |_p: usize, obs: Vec<f32>, n: usize| -> Vec<f64> {
                (0..n)
                    .flat_map(|i| {
                        let seat = obs[i * 2 + 1] as usize;
                        let l = game_logits(seat);
                        let (l0, l1) = if swapped && seat == 1 {
                            (l[1], l[0])
                        } else {
                            (l[0], l[1])
                        };
                        [l0, l1, 0.1]
                    })
                    .collect()
            };
            let mut eval = Evaluator::new(
                &mut infer,
                crate::rollout::evaluator::InferMode::Shared,
                None,
            );
            let cfg = az_cfg(40, 0.0, NoiseScope::Requester);
            if swapped {
                alphazero_many(
                    &MatrixGame,
                    &SwapFor1,
                    &Passthrough,
                    &cfg,
                    states(),
                    9,
                    &mut eval,
                )
            } else {
                alphazero_many(
                    &MatrixGame,
                    &Enc,
                    &Passthrough,
                    &cfg,
                    states(),
                    9,
                    &mut eval,
                )
            }
        };
        let id = run(false);
        let sw = run(true);
        for (seat, (a, b)) in id.iter().zip(&sw).enumerate() {
            assert_eq!(a.visits, b.visits, "seat {seat}: per-table view broken");
            assert_eq!(a.values, b.values, "seat {seat}: per-table view broken");
        }
    }
}

#[cfg(test)]
mod frame_tests {
    use super::*;
    use crate::encoder::StateEncoder;
    use crate::game::{Actor, Game, Transition};
    use crate::policies::tree::alphazero::{alphazero_many, AlphaZeroConfig};
    use crate::reward::Reward as RewardTrait;

    struct Sparse;
    #[derive(Clone)]
    struct St(i32);
    impl Game for Sparse {
        type State = St;
        type Event = f64;
        fn num_agents(&self) -> usize {
            1
        }
        fn action_count(&self) -> usize {
            4
        }
        fn actor(&self, _: &St) -> Actor {
            Actor::Agent(0)
        }
        fn legal_actions(&self, _: &St, _: usize) -> Vec<usize> {
            vec![0, 2]
        }
        fn step(&self, s: &St, actions: &[usize]) -> Transition<St, f64> {
            let next = s.0 * 3 + actions[0] as i32 + 1;
            Transition {
                next_state: St(next),
                events: vec![Some(0.0)],
                terminal: next > 40,
            }
        }
        fn initial_state(&self) -> St {
            St(0)
        }
    }
    struct Zero;
    impl RewardTrait for Zero {
        type Event = f64;
        fn step_reward(&self, _: &f64, _: usize) -> f64 {
            0.0
        }
    }

    const A: usize = 4;
    fn rot(a: usize) -> usize {
        (a + 1) % A
    }

    struct IdEnc;
    impl ActionView for IdEnc {}
    struct RotEnc;
    impl ActionView for RotEnc {
        fn head_index(&self, action: usize, _: usize) -> usize {
            rot(action)
        }
        fn game_action(&self, head: usize, _: usize) -> usize {
            (head + A - 1) % A
        }
    }
    macro_rules! enc_obs {
        ($t:ty) => {
            impl StateEncoder for $t {
                type State = St;
                fn encode(&self, s: &St, _: usize) -> Vec<f32> {
                    vec![s.0 as f32]
                }
                fn obs_shape(&self) -> (usize, usize, usize) {
                    (1, 1, 1)
                }
            }
        };
    }
    enc_obs!(IdEnc);
    enc_obs!(RotEnc);

    fn v(s: f64, a: usize) -> f64 {
        ((s as i64 * 3 + a as i64 * 7) % 11) as f64 * 0.1
    }

    #[test]
    fn uct_leaf_values_gather_through_the_view() {
        let cfg = MctsConfig {
            num_simulations: 30,
            uct_c: 1.4,
            gamma: 0.95,
            max_depth: 6,
            temperature: 0.0,
            temperature_drop: u32::MAX,
            chance: ChanceMode::AlwaysResample,
        };
        let run = |use_rot: bool| {
            let mut infer = move |_p: usize, obs: Vec<f32>, n: usize| -> Vec<f64> {
                (0..n)
                    .flat_map(|i| {
                        let s = f64::from(obs[i]);
                        (0..A).map(move |slot| {
                            let game_a = if use_rot { (slot + A - 1) % A } else { slot };
                            v(s, game_a)
                        })
                    })
                    .collect()
            };
            let mut eval = Evaluator::new(
                &mut infer,
                crate::rollout::evaluator::InferMode::Shared,
                None,
            );
            let requests = vec![(St(0), 0)];
            if use_rot {
                mcts_many(&Sparse, &RotEnc, &Zero, &cfg, requests, 3, &mut eval).remove(0)
            } else {
                mcts_many(&Sparse, &IdEnc, &Zero, &cfg, requests, 3, &mut eval).remove(0)
            }
        };
        let id = run(false);
        let rot = run(true);
        assert_eq!(id.visits, rot.visits, "UCT leaf gathers must be frame-true");
        assert_eq!(id.values, rot.values);
    }

    #[test]
    fn puct_priors_gather_through_the_view() {
        let cfg = AlphaZeroConfig {
            num_simulations: 30,
            c_puct: 1.5,
            gamma: 1.0,
            max_depth: 6,
            noise_epsilon: 0.0,
            noise_alpha: 0.3,
            temperature: 0.0,
            temperature_drop: u32::MAX,
            chance: ChanceMode::AlwaysResample,
            noise_scope: NoiseScope::Requester,
            sequential_backup: Default::default(),
        };
        let run = |use_rot: bool| {
            let mut infer = move |_p: usize, obs: Vec<f32>, n: usize| -> Vec<f64> {
                (0..n)
                    .flat_map(|i| {
                        let s = f64::from(obs[i]);
                        let mut row: Vec<f64> = (0..A)
                            .map(|slot| {
                                let game_a = if use_rot { (slot + A - 1) % A } else { slot };
                                v(s, game_a) * 4.0
                            })
                            .collect();
                        row.push(0.2);
                        row
                    })
                    .collect()
            };
            let mut eval = Evaluator::new(
                &mut infer,
                crate::rollout::evaluator::InferMode::Shared,
                None,
            );
            let requests = vec![(St(0), 0)];
            if use_rot {
                alphazero_many(&Sparse, &RotEnc, &Zero, &cfg, requests, 3, &mut eval).remove(0)
            } else {
                alphazero_many(&Sparse, &IdEnc, &Zero, &cfg, requests, 3, &mut eval).remove(0)
            }
        };
        let id = run(false);
        let rot = run(true);
        assert_eq!(
            id.visits, rot.visits,
            "PUCT prior gathers must be frame-true"
        );
        assert_eq!(id.values, rot.values);
    }
}

#[cfg(test)]
mod maxn_tests {
    use super::*;
    use crate::encoder::StateEncoder;
    use crate::game::{Actor, Game, Transition};
    use crate::policies::tree::alphazero::{alphazero_many, AlphaZeroConfig};
    use crate::reward::Reward as RewardTrait;
    use crate::rollout::evaluator::Evaluator;

    #[derive(Clone)]
    enum Lr {
        Root,
        AfterL,
        Done,
    }
    struct Discriminator;
    impl Game for Discriminator {
        type State = Lr;
        type Event = f64;
        fn num_agents(&self) -> usize {
            3
        }
        fn action_count(&self) -> usize {
            2
        }
        fn actor(&self, s: &Lr) -> Actor {
            match s {
                Lr::AfterL => Actor::Agent(1),
                _ => Actor::Agent(0),
            }
        }
        fn legal_actions(&self, s: &Lr, agent: usize) -> Vec<usize> {
            match (s, agent) {
                (Lr::Root, 0) | (Lr::AfterL, 1) => vec![0, 1],
                _ => Vec::new(),
            }
        }
        fn step(&self, s: &Lr, actions: &[usize]) -> Transition<Lr, f64> {
            match s {
                Lr::Root if actions[0] == 0 => Transition {
                    next_state: Lr::AfterL,
                    events: vec![Some(0.0); 3],
                    terminal: false,
                },
                Lr::Root => Transition {
                    next_state: Lr::Done,
                    events: vec![Some(3.0), Some(0.0), Some(0.0)],
                    terminal: true,
                },
                Lr::AfterL if actions[1] == 0 => Transition {
                    next_state: Lr::Done,
                    events: vec![Some(5.0), Some(10.0), Some(0.0)],
                    terminal: true,
                },
                Lr::AfterL => Transition {
                    next_state: Lr::Done,
                    events: vec![Some(0.0); 3],
                    terminal: true,
                },
                Lr::Done => unreachable!("stepping a terminal state"),
            }
        }
        fn initial_state(&self) -> Lr {
            Lr::Root
        }
    }

    struct LrEnc;
    impl ActionView for LrEnc {}
    impl StateEncoder for LrEnc {
        type State = Lr;
        fn encode(&self, s: &Lr, agent: usize) -> Vec<f32> {
            let phase = match s {
                Lr::Root => 0.0,
                Lr::AfterL => 1.0,
                Lr::Done => 2.0,
            };
            vec![phase, agent as f32]
        }
        fn obs_shape(&self) -> (usize, usize, usize) {
            (1, 1, 2)
        }
    }

    struct Payout;
    impl RewardTrait for Payout {
        type Event = f64;
        fn step_reward(&self, e: &f64, _agent: usize) -> f64 {
            *e
        }
    }

    #[test]
    fn maxn_models_co_players_as_self_interested() {
        let cfg = AlphaZeroConfig {
            num_simulations: 200,
            c_puct: 1.5,
            gamma: 1.0,
            max_depth: 8,
            noise_epsilon: 0.0,
            noise_alpha: 0.3,
            temperature: 0.0,
            temperature_drop: 0,
            chance: ChanceMode::Committed { samples: 1 },
            noise_scope: NoiseScope::Requester,
            sequential_backup: Default::default(),
        };
        let mut infer = |_p: usize, _obs: Vec<f32>, n: usize| vec![0.0; n * 3];
        let mut eval = Evaluator::new(
            &mut infer,
            crate::rollout::evaluator::InferMode::Shared,
            None,
        );
        let e = alphazero_many(
            &Discriminator,
            &LrEnc,
            &Payout,
            &cfg,
            vec![(Lr::Root, 0)],
            0,
            &mut eval,
        )
        .remove(0);
        assert!(
            e.visits[0] > e.visits[1],
            "Max^N must prefer L (self-interested co-player): visits {:?}",
            e.visits
        );
        assert!(
            (e.values[0][1] - 3.0).abs() < 1e-9,
            "R's value is its exact payoff: {:?}",
            e.values[0]
        );
        assert!(
            e.values[0][0] > 4.0,
            "L's value must converge toward 5 (agent 1 picks a), got {}",
            e.values[0][0]
        );
    }

    struct Dominant;
    #[derive(Clone)]
    struct DSt(bool);
    impl Game for Dominant {
        type State = DSt;
        type Event = f64;
        fn num_agents(&self) -> usize {
            3
        }
        fn action_count(&self) -> usize {
            2
        }
        fn actor(&self, _s: &DSt) -> Actor {
            Actor::Simultaneous
        }
        fn legal_actions(&self, s: &DSt, _agent: usize) -> Vec<usize> {
            if s.0 {
                Vec::new()
            } else {
                vec![0, 1]
            }
        }
        fn step(&self, _s: &DSt, actions: &[usize]) -> Transition<DSt, f64> {
            Transition {
                next_state: DSt(true),
                events: actions.iter().map(|&a| Some(a as f64)).collect(),
                terminal: true,
            }
        }
        fn initial_state(&self) -> DSt {
            DSt(false)
        }
    }

    struct DEnc;
    impl ActionView for DEnc {}
    impl StateEncoder for DEnc {
        type State = DSt;
        fn encode(&self, s: &DSt, agent: usize) -> Vec<f32> {
            vec![f32::from(u8::from(s.0)), agent as f32]
        }
        fn obs_shape(&self) -> (usize, usize, usize) {
            (1, 1, 2)
        }
    }

    #[test]
    fn duct_three_agents_each_find_their_dominant_action() {
        let cfg = MctsConfig {
            num_simulations: 64,
            uct_c: 1.0,
            gamma: 1.0,
            max_depth: 4,
            temperature: 0.0,
            temperature_drop: 0,
            chance: ChanceMode::Committed { samples: 1 },
        };
        for requester in 0..3 {
            let mut infer = |_p: usize, _obs: Vec<f32>, n: usize| vec![0.0; n * 2];
            let mut eval = Evaluator::new(
                &mut infer,
                crate::rollout::evaluator::InferMode::Shared,
                None,
            );
            let e = mcts_many(
                &Dominant,
                &DEnc,
                &Payout,
                &cfg,
                vec![(DSt(false), requester)],
                9,
                &mut eval,
            )
            .remove(0);
            assert!(
                e.visits[1] > e.visits[0],
                "agent {requester} must concentrate on its dominant action: {:?}",
                e.visits
            );
        }
    }
}

#[cfg(test)]
mod forced_maxn_tests {
    use super::*;
    use crate::encoder::StateEncoder;
    use crate::game::{Actor, Game, Transition};
    use crate::policies::tree::alphazero::{alphazero_many, AlphaZeroConfig};
    use crate::reward::Reward as RewardTrait;
    use crate::rollout::evaluator::Evaluator;

    #[derive(Clone)]
    enum Lr {
        Root,
        AfterL,
        Done,
    }
    struct Discriminator2;
    impl Game for Discriminator2 {
        type State = Lr;
        type Event = f64;
        fn num_agents(&self) -> usize {
            2
        }
        fn action_count(&self) -> usize {
            2
        }
        fn actor(&self, s: &Lr) -> Actor {
            match s {
                Lr::AfterL => Actor::Agent(1),
                _ => Actor::Agent(0),
            }
        }
        fn legal_actions(&self, s: &Lr, agent: usize) -> Vec<usize> {
            match (s, agent) {
                (Lr::Root, 0) | (Lr::AfterL, 1) => vec![0, 1],
                _ => Vec::new(),
            }
        }
        fn step(&self, s: &Lr, actions: &[usize]) -> Transition<Lr, f64> {
            match s {
                Lr::Root if actions[0] == 0 => Transition {
                    next_state: Lr::AfterL,
                    events: vec![Some(0.0); 2],
                    terminal: false,
                },
                Lr::Root => Transition {
                    next_state: Lr::Done,
                    events: vec![Some(3.0), Some(0.0)],
                    terminal: true,
                },
                Lr::AfterL if actions[1] == 0 => Transition {
                    next_state: Lr::Done,
                    events: vec![Some(5.0), Some(10.0)],
                    terminal: true,
                },
                Lr::AfterL => Transition {
                    next_state: Lr::Done,
                    events: vec![Some(0.0); 2],
                    terminal: true,
                },
                Lr::Done => unreachable!("stepping a terminal state"),
            }
        }
        fn initial_state(&self) -> Lr {
            Lr::Root
        }
    }

    struct LrEnc;
    impl ActionView for LrEnc {}
    impl StateEncoder for LrEnc {
        type State = Lr;
        fn encode(&self, s: &Lr, agent: usize) -> Vec<f32> {
            let phase = match s {
                Lr::Root => 0.0,
                Lr::AfterL => 1.0,
                Lr::Done => 2.0,
            };
            vec![phase, agent as f32]
        }
        fn obs_shape(&self) -> (usize, usize, usize) {
            (1, 1, 2)
        }
    }

    struct Payout;
    impl RewardTrait for Payout {
        type Event = f64;
        fn step_reward(&self, e: &f64, _agent: usize) -> f64 {
            *e
        }
    }

    #[test]
    fn sequential_backup_maxn_changes_the_two_agent_tree() {
        let run = |backup| {
            let cfg = AlphaZeroConfig {
                num_simulations: 200,
                c_puct: 1.5,
                gamma: 1.0,
                max_depth: 8,
                noise_epsilon: 0.0,
                noise_alpha: 0.3,
                temperature: 0.0,
                temperature_drop: 0,
                chance: ChanceMode::Committed { samples: 1 },
                noise_scope: NoiseScope::Requester,
                sequential_backup: backup,
            };
            let mut infer = |_p: usize, _obs: Vec<f32>, n: usize| vec![0.0; n * 3];
            let mut eval = Evaluator::new(
                &mut infer,
                crate::rollout::evaluator::InferMode::Shared,
                None,
            );
            alphazero_many(
                &Discriminator2,
                &LrEnc,
                &Payout,
                &cfg,
                vec![(Lr::Root, 0)],
                0,
                &mut eval,
            )
            .remove(0)
        };
        let auto = run(SequentialBackup::Auto);
        assert!(
            auto.visits[1] > auto.visits[0],
            "negamax (paranoid at general-sum) must prefer R: {:?}",
            auto.visits
        );
        let maxn = run(SequentialBackup::MaxN);
        assert!(
            maxn.visits[0] > maxn.visits[1],
            "forced Max^N (self-interested opponent) must prefer L: {:?}",
            maxn.visits
        );
        assert!(
            (maxn.values[0][1] - 3.0).abs() < 1e-9 && maxn.values[0][0] > 4.0,
            "Max^N root values converge to the self-interested payoffs: {:?}",
            maxn.values[0]
        );
    }
}

#[cfg(test)]
mod pooled_lifecycle_tests {
    use super::*;
    use crate::encoder::StateEncoder;
    use crate::game::{Actor, Game, Transition};
    use crate::reward::Reward;
    use crate::rollout::evaluator::{Evaluator, InferMode};

    #[derive(Clone)]
    struct St {
        tick: usize,
    }
    struct SimPair;
    impl Game for SimPair {
        type State = St;
        type Event = ();
        fn num_agents(&self) -> usize {
            2
        }
        fn action_count(&self) -> usize {
            2
        }
        fn actor(&self, _s: &St) -> Actor {
            Actor::Simultaneous
        }
        fn legal_actions(&self, _s: &St, _agent: usize) -> Vec<usize> {
            vec![0, 1]
        }
        fn step(&self, s: &St, _actions: &[usize]) -> Transition<St, ()> {
            Transition {
                next_state: St { tick: s.tick + 1 },
                events: vec![None; 2],
                terminal: s.tick + 1 >= 4,
            }
        }
        fn initial_state(&self) -> St {
            St { tick: 0 }
        }
    }
    struct Enc;
    impl crate::encoder::ActionView for Enc {}
    impl StateEncoder for Enc {
        type State = St;
        fn encode(&self, s: &St, agent: usize) -> Vec<f32> {
            vec![s.tick as f32, agent as f32]
        }
        fn obs_shape(&self) -> (usize, usize, usize) {
            (1, 1, 2)
        }
    }
    struct Zero;
    impl Reward for Zero {
        type Event = ();
        fn step_reward(&self, _e: &(), _agent: usize) -> f64 {
            0.0
        }
    }

    fn infer(_player: usize, obs: Vec<f32>, n: usize) -> Vec<f64> {
        assert_eq!(obs.len(), n * 2);
        vec![0.25; n * 2]
    }

    fn new_pool<'c>(
        game: &'c SimPair,
        enc: &'c Enc,
        reward: &'c Zero,
        sims: usize,
    ) -> PooledSearch<'c, SimPair> {
        PooledSearch::new(
            game,
            enc,
            reward,
            sims,
            1.0,
            64,
            Guidance::Uct { c: 1.0 },
            ChanceMode::AlwaysResample,
            7,
            vec![(St { tick: 0 }, 0)],
            false,
        )
    }

    #[test]
    fn final_round_is_outstanding_until_applied() {
        let (game, enc, reward) = (SimPair, Enc, Zero);
        let mut f = infer;
        let mut eval: Evaluator<'_, _> = Evaluator::new(&mut f, InferMode::Shared, None);
        let mut p = new_pool(&game, &enc, &reward, 1);
        let mut batch = eval.batch();
        p.stage_round(&mut batch);
        // The single simulation is spent, but its (multi-row: simultaneous) evaluation is in
        // flight — the pool must not present as finished.
        assert!(!p.finished());
        let rows = batch.commit();
        p.apply_rows(&rows);
        assert!(p.finished());
        let evals = p.into_evaluations();
        assert_eq!(evals.len(), 1);
    }

    #[test]
    #[should_panic(expected = "round outstanding")]
    fn double_stage_panics() {
        let (game, enc, reward) = (SimPair, Enc, Zero);
        let mut f = infer;
        let mut eval: Evaluator<'_, _> = Evaluator::new(&mut f, InferMode::Shared, None);
        let mut p = new_pool(&game, &enc, &reward, 2);
        let mut batch = eval.batch();
        p.stage_round(&mut batch);
        drop(batch);
        let mut batch2 = eval.batch();
        p.stage_round(&mut batch2);
    }

    #[test]
    #[should_panic(expected = "without a staged round")]
    fn apply_without_stage_panics() {
        let (game, enc, reward) = (SimPair, Enc, Zero);
        let mut f = infer;
        let mut eval: Evaluator<'_, _> = Evaluator::new(&mut f, InferMode::Shared, None);
        let empty = eval.batch().commit();
        let mut p = new_pool(&game, &enc, &reward, 1);
        p.apply_rows(&empty);
    }

    #[test]
    #[should_panic(expected = "unfinished pool")]
    fn into_evaluations_while_awaiting_panics() {
        let (game, enc, reward) = (SimPair, Enc, Zero);
        let mut f = infer;
        let mut eval: Evaluator<'_, _> = Evaluator::new(&mut f, InferMode::Shared, None);
        let mut p = new_pool(&game, &enc, &reward, 1);
        let mut batch = eval.batch();
        p.stage_round(&mut batch);
        drop(batch);
        let _ = p.into_evaluations();
    }

    #[test]
    #[should_panic(expected = "fresh batch")]
    fn nonempty_batch_panics() {
        let (game, enc, reward) = (SimPair, Enc, Zero);
        let mut f = infer;
        let mut eval: Evaluator<'_, _> = Evaluator::new(&mut f, InferMode::Shared, None);
        let mut p = new_pool(&game, &enc, &reward, 1);
        let mut batch = eval.batch();
        let _ = batch.resolve_or_stage(0, &[9.0, 9.0]);
        p.stage_round(&mut batch);
    }

    #[test]
    fn alternated_pools_match_search_many() {
        let (game, enc, reward) = (SimPair, Enc, Zero);
        let sims = 6;
        let mut f = infer;
        let mut eval: Evaluator<'_, _> = Evaluator::new(&mut f, InferMode::Shared, None);
        let guidance = Guidance::Uct { c: 1.0 };
        let reference = search_many(
            &game,
            &enc,
            &reward,
            sims,
            1.0,
            64,
            &guidance,
            ChanceMode::AlwaysResample,
            7,
            vec![(St { tick: 0 }, 0)],
            &mut eval,
            false,
        );

        let mut f2 = infer;
        let mut eval2: Evaluator<'_, _> = Evaluator::new(&mut f2, InferMode::Shared, None);
        let mut a = new_pool(&game, &enc, &reward, sims);
        let mut b = new_pool(&game, &enc, &reward, sims);
        while !a.finished() || !b.finished() {
            for p in [&mut a, &mut b] {
                if p.finished() {
                    continue;
                }
                let mut batch = eval2.batch();
                p.stage_round(&mut batch);
                let rows = batch.commit();
                p.apply_rows(&rows);
            }
        }
        for evals in [a.into_evaluations(), b.into_evaluations()] {
            assert_eq!(evals.len(), 1);
            assert_eq!(evals[0].visits, reference[0].visits);
            assert_eq!(evals[0].values, reference[0].values);
        }
    }
}
