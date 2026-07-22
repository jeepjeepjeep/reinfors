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
//! **Acting.** Greedy by default (`argmax` value or visits) — deterministic, ideal for evaluation and
//! like-for-like benchmarks. For *training*, greedy self-play from a fixed start replays the same game
//! every episode, so an AlphaZero-style acting temperature supplies self-play diversity: with
//! `temperature > 0`, the first `temperature_drop` moves of each episode are sampled `∝
//! visits^(1/temperature)` from the engine's seeded acting RNG (later moves act greedily). Same seed →
//! same games (collects stay reproducible); different episodes → different games. Root Dirichlet noise
//! is deliberately absent: it perturbs *priors*, a PUCT concept — this UCT search has none. The
//! reached-state start buffer remains the complementary coverage source.

use crate::encoder::StateEncoder;
use crate::engine::CollectStats;
use crate::game::{Actor, Game, Rng};
use crate::policies::expectimax::search::SearchStats;
use crate::policies::expectimax::SearchEvaluation;
use crate::policy::{argmax, Policy};
use crate::reward::Reward;
use crate::rng::{dirichlet, SplitMix64};

/// Which rule guides selection, and what the net returns per leaf. The tree machinery below is shared;
/// this is the only axis the UCT (`Mcts`) and PUCT (`AlphaZero`) policies differ on.
pub(crate) enum Guidance {
    /// UCB1 over backed-up values; the net returns per-head Q rows `[K][A]` (leaf value = max of the
    /// head-mean), and unvisited actions win selection outright.
    Uct { c: f64 },
    /// AlphaZero PUCT: the net returns `[A]` policy logits + a value per row (stride `A+1`); each
    /// node stores its softmaxed prior, and selection scores `Q + c·P·√N/(1+n)`. `noise` is the root
    /// Dirichlet mix `(epsilon, alpha, stream_seed)` applied once per tree when the root's eval
    /// returns — `None` (or epsilon 0) searches noise-free.
    Puct {
        c: f64,
        noise: Option<(f64, f64, u64)>,
    },
}

impl Guidance {
    /// Elements per net-output row: `K·A` Q-values (UCT, `k` heads) or `A+1` logits+value (PUCT).
    fn row_stride(&self, n_rows: usize, out_len: usize, actions: usize) -> usize {
        match self {
            Guidance::Uct { .. } => out_len / n_rows.max(1),
            Guidance::Puct { .. } => actions + 1,
        }
    }
}

/// Numerically stable softmax — the PUCT prior over the full action space (every action is legal by
/// framework contract; a bad-but-legal action's prior just learns toward zero).
fn softmax(logits: &[f64]) -> Vec<f64> {
    let m = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|&l| (l - m).exp()).collect();
    let total: f64 = exps.iter().sum();
    exps.into_iter().map(|e| e / total).collect()
}

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
    /// AlphaZero-style acting temperature: `> 0` samples the move `∝ visits^(1/temperature)` for the
    /// first `temperature_drop` moves of each episode; `0` acts greedily everywhere (the default).
    pub temperature: f64,
    /// Number of opening plies per episode the temperature applies to (the counter spans both sides
    /// of a sequential game, like AlphaZero's); `u32::MAX` means the whole episode. Irrelevant when
    /// `temperature == 0`.
    pub temperature_drop: u32,
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
    prior: Vec<f64>, // [A] PUCT prior (softmaxed net logits); empty under UCT, and empty = not yet evaluated under PUCT
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
            prior: Vec::new(),
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

    /// Select from the root down to an expandable edge (scored by `guidance`), create its child
    /// (stepping the game), and mark it as the leaf to back up. Returns how the leaf's value is
    /// obtained. Under PUCT the root itself is the first leaf: it is created without an eval, and
    /// selection needs its prior, so simulation 1 evaluates it in place (empty path).
    fn select_expand<G>(
        &mut self,
        game: &G,
        enc: &dyn StateEncoder<State = S>,
        reward: &dyn Reward<Event = G::Event>,
        max_depth: i32,
        guidance: &Guidance,
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
            if matches!(guidance, Guidance::Puct { .. }) && node.prior.is_empty() {
                self.leaf = ni; // un-evaluated PUCT node (the root on sim 1): evaluate it in place
                return Reached::Eval;
            }
            if node.depth >= max_depth {
                self.leaf = ni; // depth cap: use the cached net value evaluated when this node was created
                return Reached::Cached(node.value);
            }
            let a = select_edge(node, guidance);
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
    fn backprop(&mut self, gamma: f64, leaf_value: f64) {
        self.arena[self.leaf].value = leaf_value;
        let mut g = leaf_value; // value from the child's actor perspective
        for &(ni, a) in self.path.iter().rev() {
            let node_actor = self.arena[ni].actor;
            let child_actor = self.arena[self.arena[ni].child[a] as usize].actor;
            let child_val = if child_actor == node_actor { g } else { -g };
            let q = self.arena[ni].reward[a] + gamma * child_val;
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

/// Score a node's actions (mover's perspective) by the guidance rule and pick the best.
/// UCT: UCB1, an unvisited action wins outright. PUCT: `Q + c·P·√N_total/(1+n)` with unvisited
/// `Q = 0` (the AlphaZero convention) — the prior, not optimism, drives first visits; `N_total` is
/// floored at 1 so the very first selection is prior-ordered rather than degenerate.
fn select_edge<S>(node: &Node<S>, guidance: &Guidance) -> usize {
    let mut best = 0;
    let mut best_score = f64::NEG_INFINITY;
    for a in 0..node.child.len() {
        let score = match guidance {
            Guidance::Uct { c } => {
                if node.visits[a] == 0 {
                    f64::INFINITY
                } else {
                    let n = node.visits[a] as f64;
                    let ln_n = (node.total_visits.max(1) as f64).ln();
                    node.value_sum[a] / n + c * (ln_n / n).sqrt()
                }
            }
            Guidance::Puct { c, .. } => {
                let n = node.visits[a] as f64;
                let q = if node.visits[a] > 0 {
                    node.value_sum[a] / n
                } else {
                    0.0
                };
                let sqrt_total = (node.total_visits.max(1) as f64).sqrt();
                q + c * node.prior[a] * sqrt_total / (1.0 + n)
            }
        };
        if score > best_score {
            best_score = score;
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
    let guidance = Guidance::Uct { c: cfg.uct_c };
    search_many(
        game,
        enc,
        reward,
        cfg.num_simulations,
        cfg.gamma,
        cfg.max_depth,
        &guidance,
        requests,
        infer,
    )
}

/// The shared pooled search loop behind both `mcts_many` (UCT) and `alphazero_many` (PUCT): each round
/// advances every active tree by one simulation, batching the new leaves' observations into a single
/// `infer` forward, then consumes each returned row per the guidance mode (UCT: Q rows → leaf value;
/// PUCT: logits+value → node prior + backed-up value, with root noise mixed in once per tree).
#[allow(clippy::too_many_arguments)]
pub(crate) fn search_many<G, F>(
    game: &G,
    enc: &dyn StateEncoder<State = G::State>,
    reward: &dyn Reward<Event = G::Event>,
    num_simulations: usize,
    gamma: f64,
    max_depth: i32,
    guidance: &Guidance,
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

    while trees.iter().any(|t| t.sims < num_simulations) {
        let mut obs_flat: Vec<f32> = Vec::new();
        let mut eval_rows: Vec<usize> = Vec::new(); // trees awaiting this round's forward
        for (ti, tree) in trees.iter_mut().enumerate() {
            if tree.sims >= num_simulations {
                continue;
            }
            tree.sims += 1;
            match tree.select_expand(game, enc, reward, max_depth, guidance) {
                Reached::Cached(v) => tree.backprop(gamma, v),
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
        let out = infer(obs_flat, n);
        let stride = guidance.row_stride(n, out.len(), a);
        for (row, &ti) in eval_rows.iter().enumerate() {
            let tree = &mut trees[ti];
            let row_data = &out[row * stride..(row + 1) * stride];
            match guidance {
                Guidance::Uct { .. } => {
                    let v = leaf_value(row_data, stride / a, a);
                    tree.backprop(gamma, v);
                }
                Guidance::Puct { noise, .. } => {
                    let (logits, value) = row_data.split_at(a);
                    let mut prior = softmax(logits);
                    if tree.leaf == 0 {
                        // The root's one eval: mix in the Dirichlet exploration noise (per-tree
                        // stream, so pooled searches stay deterministic and independent).
                        if let Some((eps, alpha, seed)) = noise {
                            if *eps > 0.0 {
                                let mut rng = SplitMix64::new(
                                    seed ^ (ti as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15),
                                );
                                let noise_draw = dirichlet(&mut rng, *alpha, a);
                                for (p, d) in prior.iter_mut().zip(noise_draw) {
                                    *p = (1.0 - eps) * *p + eps * d;
                                }
                            }
                        }
                    }
                    tree.arena[tree.leaf].prior = prior;
                    tree.backprop(gamma, value[0]);
                }
            }
        }
    }

    trees.into_iter().map(|t| t.evaluation(a)).collect()
}

/// Sample an action `∝ visits^(1/temperature)`. Weights are max-normalized before the power so a
/// small temperature cannot overflow; unvisited actions keep weight 0 and are never picked.
/// Shared with the `AlphaZero` policy (same acting-temperature semantics).
pub(crate) fn sample_visits(visits: &[f64], temperature: f64, rng: &mut dyn Rng) -> usize {
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
    argmax(&w) // numeric fallback (r exhausted by rounding): the modal action
}

impl Policy for Mcts {
    type Evaluation = SearchEvaluation;
    type PolicyState = u32; // moves acted this episode — drives the temperature_drop cutoff

    fn begin_episode(&self, _rng: &mut dyn Rng) -> u32 {
        0
    }

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

    fn select(&self, eval: &SearchEvaluation, state: &mut u32, rng: &mut dyn Rng) -> usize {
        let move_idx = *state;
        *state += 1;
        if self.cfg.temperature > 0.0 && move_idx < self.cfg.temperature_drop {
            return sample_visits(&eval.visits, self.cfg.temperature, rng);
        }
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
            stats: SearchStats {
                max_depth: 1,
                expansions: 1,
                leaves: 1,
                rounds: 1,
                sigma_sum: 0.0,
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
            },
            ActBy::Visits,
        )
    }

    #[test]
    fn sample_visits_is_proportional_at_temperature_one() {
        // weights (max-normalized): [1/3, 1, 0] -> cumulative thresholds 0.25 / 1.0 of the total
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
    fn zero_temperature_acts_greedily_and_ignores_rng() {
        let p = mcts(0.0, u32::MAX);
        let e = eval(vec![0.9, 0.1], vec![2.0, 6.0]);
        let mut moves = 0u32;
        assert_eq!(p.select(&e, &mut moves, &mut FakeRng(vec![])), 1); // argmax visits, no draw
        assert_eq!(moves, 1);
    }

    #[test]
    fn temperature_drop_switches_to_greedy() {
        let p = mcts(1.0, 1);
        let e = eval(vec![0.9, 0.1], vec![6.0, 2.0]);
        let mut moves = 0u32;
        // move 0: sampled — a high draw lands on the minority action
        assert_eq!(p.select(&e, &mut moves, &mut FakeRng(vec![0.99])), 1);
        // move 1: past the drop — greedy argmax visits, rng untouched
        assert_eq!(p.select(&e, &mut moves, &mut FakeRng(vec![])), 0);
    }

    #[test]
    fn softmax_is_a_distribution_and_orders_by_logit() {
        let p = softmax(&[1.0, 3.0, 2.0]);
        assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!(p[1] > p[2] && p[2] > p[0]);
        let u = softmax(&[5.0, 5.0]);
        assert!((u[0] - 0.5).abs() < 1e-12); // equal logits -> uniform
    }

    fn puct_node(prior: Vec<f64>, visits: Vec<u32>, value_sum: Vec<f64>) -> Node<()> {
        let a = prior.len();
        let total = visits.iter().sum();
        Node {
            state: (),
            actor: 0,
            depth: 0,
            terminal: false,
            child: vec![-1; a],
            reward: vec![0.0; a],
            visits,
            value_sum,
            total_visits: total,
            value: 0.0,
            obs: Vec::new(),
            prior,
        }
    }

    #[test]
    fn puct_first_selection_is_prior_ordered() {
        // No visits anywhere: scores reduce to c·P(a) (N_total floored at 1) — the prior decides,
        // unlike UCT where any unvisited action wins outright.
        let node = puct_node(vec![0.2, 0.5, 0.3], vec![0, 0, 0], vec![0.0; 3]);
        assert_eq!(
            select_edge(
                &node,
                &Guidance::Puct {
                    c: 1.5,
                    noise: None
                }
            ),
            1
        );
    }

    #[test]
    fn puct_high_visits_shrink_the_exploration_term() {
        // Action 1 has the bigger prior but many visits and a mediocre Q; action 0's small-n
        // exploration term wins — the 1/(1+n) decay working as intended.
        let node = puct_node(vec![0.3, 0.7], vec![1, 99], vec![0.1, 9.9]); // Q = 0.1 both
        let g = Guidance::Puct {
            c: 2.0,
            noise: None,
        };
        assert_eq!(select_edge(&node, &g), 0);
    }

    #[test]
    fn puct_exploits_value_at_equal_priors() {
        let node = puct_node(vec![0.5, 0.5], vec![10, 10], vec![9.0, 1.0]); // Q: 0.9 vs 0.1
        assert_eq!(
            select_edge(
                &node,
                &Guidance::Puct {
                    c: 0.1,
                    noise: None
                }
            ),
            0
        );
    }
}
