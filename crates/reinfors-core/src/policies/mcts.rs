//! Monte-Carlo Tree Search (UCT): a genuine MCTS planner alongside the expectimax family, for a
//! like-for-like comparison with other MCTS engines. It produces the same [`SearchEvaluation`] the
//! `TreeStrap` learner consumes — the training target is the root's backed-up per-action values
//! `values[1][A]` ("MCTS-strap") — and pools its leaf evaluations across games into one `infer` per
//! round, exactly like the expectimax search.
//!
//! **Sequential + single-agent games only.** MCTS here assumes strictly alternating turns (or one
//! agent), so a node's actor is a single [`Actor::Agent`]; `Actor::Simultaneous` and `Actor::Chance`
//! are rejected. Two-player games are treated as zero-sum (negamax backup) — correct for connect4. The
//! binding refuses to pair this policy with a simultaneous/chance game (snake); this module panics as a
//! backstop for direct core use.
//!
//! **Deterministic acting.** v1 acts greedily (`argmax` value or visits), with no root Dirichlet noise
//! or visit-count sampling, and ignores the acting RNG — ideal for evaluation and a like-for-like
//! benchmark, where reproducibility is the point. As a *training* policy it adds no self-play diversity
//! on its own (self-play from a fixed start replays the same game every episode), so training use leans
//! on the reached-state start buffer for coverage; a temperature / root-noise knob (AlphaZero-style) is
//! the natural addition if undiluted self-play exploration is wanted.

use crate::encoder::StateEncoder;
use crate::engine::CollectStats;
use crate::game::{Actor, Game, Rng};
use crate::policies::expectimax::search::SearchStats;
use crate::policies::expectimax::SearchEvaluation;
use crate::policy::{argmax, Policy};
use crate::reward::Reward;

/// How the policy picks the move to play from a finished tree (the training target is the backed-up
/// value either way — this only changes acting, not what `TreeStrap` regresses).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ActBy {
    /// Highest mean backed-up action value `argmax_a Q(a)`.
    Value,
    /// Most-visited action `argmax_a N(a)` — the classic MCTS "robust" choice.
    Visits,
}

#[derive(Clone, Copy, Debug)]
pub struct MctsConfig {
    pub num_simulations: usize,
    pub uct_c: f64,
    pub gamma: f64,
    pub max_depth: i32,
}

pub struct Mcts {
    cfg: MctsConfig,
    act_by: ActBy,
}

impl Mcts {
    pub fn new(cfg: MctsConfig, act_by: ActBy) -> Self {
        Mcts { cfg, act_by }
    }
}

/// The single-agent index of a node's `Actor`; MCTS supports only sequential/single-agent play.
fn sole_actor(actor: Actor) -> usize {
    match actor {
        Actor::Agent(a) => a,
        other => panic!(
            "MCTS supports only sequential/single-agent games (Actor::Agent); got {other:?}. \
             Use SelectiveExpectimax for simultaneous/chance games."
        ),
    }
}

/// One search tree. `child`/`reward`/`visits`/`value_sum` are indexed by action id (every game's
/// `legal_actions` spans the full action space, so action id == slot — no legality indirection).
struct Node<S> {
    state: S,
    actor: usize,
    depth: i32,
    terminal: bool,
    child: Vec<i64>,     // [A] child arena index, -1 if the edge is unexpanded
    reward: Vec<f64>, // [A] immediate reward for the mover taking this action (its own perspective)
    visits: Vec<u32>, // [A] edge visit counts
    value_sum: Vec<f64>, // [A] summed backed-up value (mover's perspective)
    total_visits: u32,
    value: f64, // this node's state value (net leaf eval, or 0 at a terminal) — the backprop source
    obs: Vec<f32>, // staged observation for the pending net eval (empty for terminals)
}

impl<S> Node<S> {
    fn leaf(
        state: S,
        actor: usize,
        depth: i32,
        terminal: bool,
        actions: usize,
        obs: Vec<f32>,
    ) -> Node<S> {
        let width = if terminal { 0 } else { actions };
        Node {
            state,
            actor,
            depth,
            terminal,
            child: vec![-1; width],
            reward: vec![0.0; width],
            visits: vec![0; width],
            value_sum: vec![0.0; width],
            total_visits: 0,
            value: 0.0,
            obs,
        }
    }
}

/// What one simulation reached, dictating how it backs up: a fresh non-terminal leaf whose value comes
/// from the pooled net forward, or a leaf whose value is already known (terminal = 0, or a cached
/// depth-capped node) and can be backed up immediately.
enum Reached {
    Eval,
    Cached(f64),
}

struct Tree<S> {
    arena: Vec<Node<S>>,
    sims: usize,
    path: Vec<(usize, usize)>, // (node idx, action) edges from root to the current leaf
    leaf: usize,
    max_depth_seen: i32,
}

impl<S: Clone> Tree<S> {
    fn new<G>(game: &G, enc: &dyn StateEncoder<State = S>, state: S) -> Tree<S>
    where
        G: Game<State = S>,
    {
        let actor = sole_actor(game.actor(&state));
        let obs = enc.encode(&state, actor);
        let root = Node::leaf(state, actor, 0, false, game.action_count(), obs);
        Tree {
            arena: vec![root],
            sims: 0,
            path: Vec::new(),
            leaf: 0,
            max_depth_seen: 0,
        }
    }

    /// Select from the root by UCT down to an expandable edge, create its child (stepping the game),
    /// and mark it as the leaf to back up. Returns how the leaf's value is obtained.
    fn select_expand<G>(
        &mut self,
        game: &G,
        enc: &dyn StateEncoder<State = S>,
        reward: &dyn Reward<Event = G::Event>,
        cfg: &MctsConfig,
    ) -> Reached
    where
        G: Game<State = S>,
    {
        self.path.clear();
        let mut ni = 0;
        loop {
            let node = &self.arena[ni];
            self.max_depth_seen = self.max_depth_seen.max(node.depth);
            if node.terminal {
                self.leaf = ni;
                return Reached::Cached(0.0);
            }
            if node.depth >= cfg.max_depth {
                self.leaf = ni; // depth cap: use the cached net value evaluated when this node was created
                return Reached::Cached(node.value);
            }
            let a = uct_select(node, cfg.uct_c);
            self.path.push((ni, a));
            if node.child[a] < 0 {
                let child = self.expand(game, enc, reward, ni, a);
                self.leaf = child;
                return if self.arena[child].terminal {
                    Reached::Cached(0.0)
                } else {
                    Reached::Eval // its obs is staged for the pooled forward
                };
            }
            ni = self.arena[ni].child[a] as usize;
        }
    }

    /// Step the game for action `a` at node `ni`, appending the resulting child to the arena.
    fn expand<G>(
        &mut self,
        game: &G,
        enc: &dyn StateEncoder<State = S>,
        reward: &dyn Reward<Event = G::Event>,
        ni: usize,
        a: usize,
    ) -> usize
    where
        G: Game<State = S>,
    {
        let mover = self.arena[ni].actor;
        let mut joint = vec![0usize; game.num_agents()];
        joint[mover] = a;
        let t = game.step(&self.arena[ni].state, &joint);
        self.arena[ni].reward[a] = reward.step_reward(&t.events[mover], mover);
        let depth = self.arena[ni].depth + 1;
        let child = if t.terminal {
            Node::leaf(
                t.next_state,
                mover,
                depth,
                true,
                game.action_count(),
                Vec::new(),
            )
        } else {
            let actor = sole_actor(game.actor(&t.next_state));
            let obs = enc.encode(&t.next_state, actor);
            Node::leaf(t.next_state, actor, depth, false, game.action_count(), obs)
        };
        let idx = self.arena.len();
        self.arena.push(child);
        self.arena[ni].child[a] = idx as i64;
        idx
    }

    /// Back up `leaf_value` (from the leaf actor's perspective) along the selected path, negamax across
    /// turn changes (zero-sum), discounting by gamma and adding each edge's immediate reward.
    fn backprop(&mut self, cfg: &MctsConfig, leaf_value: f64) {
        self.arena[self.leaf].value = leaf_value;
        let mut g = leaf_value; // value from the child's actor perspective
        for &(ni, a) in self.path.iter().rev() {
            let node_actor = self.arena[ni].actor;
            let child_actor = self.arena[self.arena[ni].child[a] as usize].actor;
            let child_val = if child_actor == node_actor { g } else { -g };
            let q = self.arena[ni].reward[a] + cfg.gamma * child_val;
            self.arena[ni].value_sum[a] += q;
            self.arena[ni].visits[a] += 1;
            self.arena[ni].total_visits += 1;
            g = q; // now from node_actor's perspective, for the level above
        }
    }

    /// The finished tree's root evaluation: per-action mean value `values[1][A]` (0 for any unvisited
    /// action) and visit counts, plus telemetry.
    fn evaluation(self, actions: usize) -> SearchEvaluation {
        let root = &self.arena[0];
        let values: Vec<f64> = (0..actions)
            .map(|a| {
                if root.visits[a] > 0 {
                    root.value_sum[a] / root.visits[a] as f64
                } else {
                    0.0
                }
            })
            .collect();
        let visits: Vec<f64> = root.visits.iter().map(|&n| n as f64).collect();
        let stats = SearchStats {
            max_depth: self.max_depth_seen,
            expansions: self.sims,
            leaves: self.sims,
            rounds: self.sims,
            sigma_sum: 0.0,
        };
        SearchEvaluation {
            values: vec![values],
            visits,
            interior: Vec::new(),
            stats,
        }
    }
}

/// UCB1 over a node's actions (mover's perspective); an unvisited action wins outright.
fn uct_select<S>(node: &Node<S>, c: f64) -> usize {
    let ln_n = (node.total_visits.max(1) as f64).ln();
    let mut best = 0;
    let mut best_ucb = f64::NEG_INFINITY;
    for a in 0..node.child.len() {
        let ucb = if node.visits[a] == 0 {
            f64::INFINITY
        } else {
            let n = node.visits[a] as f64;
            node.value_sum[a] / n + c * (ln_n / n).sqrt()
        };
        if ucb > best_ucb {
            best_ucb = ucb;
            best = a;
        }
    }
    best
}

/// A leaf state's value = greedy `max_a` of the head-mean net Q (matches the expectimax bootstrap).
fn leaf_value(q: &[f64], k: usize, a: usize) -> f64 {
    (0..a)
        .map(|ai| (0..k).map(|h| q[h * a + ai]).sum::<f64>() / k as f64)
        .fold(f64::NEG_INFINITY, f64::max)
}

/// Pooled UCT over a batch of `(state, agent)` requests: each round advances every active tree by one
/// simulation, batching the new non-terminal leaves' observations into a single `infer` forward.
pub fn mcts_many<G, F>(
    game: &G,
    enc: &dyn StateEncoder<State = G::State>,
    reward: &dyn Reward<Event = G::Event>,
    cfg: &MctsConfig,
    requests: Vec<(G::State, usize)>,
    infer: &mut F,
) -> Vec<SearchEvaluation>
where
    G: Game + Sync,
    G::State: Send,
    F: FnMut(Vec<f32>, usize) -> Vec<f64>,
{
    let a = game.action_count();
    let mut trees: Vec<Tree<G::State>> = requests
        .into_iter()
        .map(|(state, _agent)| Tree::new(game, enc, state))
        .collect();

    while trees.iter().any(|t| t.sims < cfg.num_simulations) {
        let mut obs_flat: Vec<f32> = Vec::new();
        let mut eval_rows: Vec<usize> = Vec::new(); // trees awaiting this round's forward
        for (ti, tree) in trees.iter_mut().enumerate() {
            if tree.sims >= cfg.num_simulations {
                continue;
            }
            tree.sims += 1;
            match tree.select_expand(game, enc, reward, cfg) {
                Reached::Cached(v) => tree.backprop(cfg, v),
                Reached::Eval => {
                    obs_flat.extend_from_slice(&tree.arena[tree.leaf].obs);
                    eval_rows.push(ti);
                }
            }
        }
        if eval_rows.is_empty() {
            continue;
        }
        let n = eval_rows.len();
        let q = infer(obs_flat, n);
        let k = q.len() / (n * a);
        for (row, &ti) in eval_rows.iter().enumerate() {
            let v = leaf_value(&q[row * k * a..(row + 1) * k * a], k, a);
            trees[ti].backprop(cfg, v);
        }
    }

    trees.into_iter().map(|t| t.evaluation(a)).collect()
}

impl Policy for Mcts {
    type Evaluation = SearchEvaluation;
    type PolicyState = (); // no per-episode state (single value head; UCT is the exploration)

    fn begin_episode(&self, _rng: &mut dyn Rng) {}

    #[allow(clippy::too_many_arguments)]
    fn evaluate<G, F>(
        &self,
        game: &G,
        enc: &dyn StateEncoder<State = G::State>,
        reward: &dyn Reward<Event = G::Event>,
        requests: Vec<(G::State, usize)>,
        _seed: u64,
        _collect_interior: bool,
        infer: &mut F,
    ) -> Vec<SearchEvaluation>
    where
        G: Game + Sync,
        G::State: Send,
        F: FnMut(Vec<f32>, usize) -> Vec<f64>,
    {
        mcts_many(game, enc, reward, &self.cfg, requests, infer)
    }

    fn select(&self, eval: &SearchEvaluation, _state: &mut (), _rng: &mut dyn Rng) -> usize {
        match self.act_by {
            ActBy::Visits if !eval.visits.is_empty() => argmax(&eval.visits),
            _ => argmax(&eval.values[0]),
        }
    }

    fn fold_telemetry(&self, eval: &SearchEvaluation, stats: &mut CollectStats) {
        let s = &eval.stats;
        stats.max_depth = stats.max_depth.max(s.max_depth);
        stats.sum_leaves += s.leaves as f64;
        stats.sum_rounds += s.rounds as f64;
        stats.sum_expansions += s.expansions as f64;
    }
}
