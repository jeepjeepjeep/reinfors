//! Best-first selective expectimax, ported to match `snake_RL`'s `SelectiveExpectimaxPlanner`.
//!
//! Values back up per ensemble head (K coherent worlds; K=1 for a single-head net), so the
//! expansion priority is value-of-information: expand where the heads disagree (sigma) and the leaf
//! most affects the root (path weight). Leaf values come from an injected evaluator (`infer`) — the
//! Python network forward in production. The opponent is either `Uniform` (eager 1/3 weights) or
//! `Distributional` (deferred: opponent observations are batched with the leaves through `infer`,
//! then their head-mean Q is turned into chance weights via softmax-with-floor).
//!
//! The tree is an arena (`Vec<Node>` indexed by `usize`) so the frontier can hold node references
//! without fighting the borrow checker.
//!
//! The engine is generic over the [`Game`] trait: a `Node<S>` carries an opaque game state `S` and
//! successors come from `Game::step` + `Game::chance_outcomes`. Snake is the only game today, so the
//! public `selective_search`/`selective_search_many` are thin wrappers that build a `SnakeGame` +
//! `SnakeState` from `SearchParams` and call the generic [`search_many`].
//!
//! When a move eats an apple in-tree, a replacement is spawned at the first unoccupied cell
//! (row-major). This deterministic spawn belief is bit-reproducible across Rust and Python, so the
//! differential test injects the same rule into the oracle — unlike the env's true RNG spawn, which
//! Option B treats as injected input. `food_samples > 1` fans each eating branch into that many
//! equally-weighted sub-branches (the Monte-Carlo spawn structure, matching the oracle); under the
//! deterministic spawn the sub-branches are identical, so it only adds value once the spawn belief is
//! stochastic — the production config uses a single sample.

use std::collections::HashSet;

use rayon::prelude::*;

use crate::game::{Game, SnakeGame, SnakeState};
use crate::reward::Reward;
use crate::snake::{Cell, Snake};

/// The agent's belief about the opponent's move distribution.
#[derive(Clone, Copy)]
pub enum Opponent {
    /// Equal weight on each opponent action (no net dependency).
    Uniform,
    /// Softmax over the opponent's (head-mean) Q with a uniform floor: deferred, so opponent
    /// observations ride the same batched forward as the leaves. `p = (1-floor)*softmax(q/temp) + floor/n`.
    Distributional { temperature: f64, floor: f64 },
}

pub struct SearchParams {
    pub grid_size: i32,
    pub initial_length: usize,
    pub play_to_last: bool,
    pub win_food_lead: Option<usize>,
    pub gamma: f64,
    pub beta: f64,
    pub expansion_budget: usize,
    pub top_k: usize,
    pub max_depth: i32,
    /// Monte-Carlo apple-spawn samples per eaten-apple branch (>= 1). With the deterministic
    /// first-empty spawn belief the samples are identical, so this is the fan-out structure a
    /// stochastic spawn would populate; 1 disables it.
    pub food_samples: usize,
    pub reward: Reward,
    pub opponent: Opponent,
}

/// Game-agnostic search knobs, derived from `SearchParams` once the game-specific config has been
/// split off onto the `Game` itself.
#[derive(Clone, Copy)]
pub struct SearchConfig {
    pub gamma: f64,
    pub beta: f64,
    pub expansion_budget: usize,
    pub top_k: usize,
    pub max_depth: i32,
    pub food_samples: usize,
    pub opponent: Opponent,
}

#[derive(Default, Clone, Copy, Debug)]
pub struct SearchStats {
    pub max_depth: i32,
    pub expansions: usize,
    pub leaves: usize,
    pub rounds: usize,
    /// Sum of per-leaf sigma (head-disagreement std) over all expanded leaves; `sigma_sum / leaves`
    /// is the search's mean leaf epistemic uncertainty (matches snake_RL's `mean_sigma`).
    pub sigma_sum: f64,
}

/// An interior MAX node's TreeStrap target: its observation and per-head backed-up action values
/// `[K][A]`. Collected (when requested) for every expanded non-terminal node below the root.
pub type InteriorTarget = (Vec<f32>, Vec<Vec<f64>>);

/// Per-request search output: root per-head action values `[K][A]`, interior TreeStrap targets (empty
/// unless `collect_interior`), and the search diagnostics.
pub type SearchResult = (Vec<Vec<f64>>, Vec<InteriorTarget>, SearchStats);

/// A committed agent action's chance branch. `weight` is the resolved chance probability; for a
/// deferred (distributional) opponent it is filled in during evaluation from `deferred = (opp obs
/// index this round, opponent action index)`. `scale` (1/food_samples on a fanned-out branch, else 1)
/// multiplies the deferred weight at resolution; fixed weights are pre-scaled at construction.
struct Branch {
    weight: f64,
    deferred: Option<(usize, usize)>,
    scale: f64,
    reward: f64,
    child: usize,
}

struct Edge {
    branches: Vec<Branch>,
}

struct Node<S> {
    state: S,
    obs: Vec<f32>, // leaf observation (empty for terminal nodes)
    depth: i32,
    terminal: bool,
    edges: Option<Vec<Edge>>, // None => unexpanded frontier leaf
    bootstrap: Vec<f64>,      // per-head net leaf value (empty until evaluated)
    value: Vec<f64>,          // per-head backed-up value (empty at terminals; treated as 0)
    sigma: f64,               // std over heads of the bootstrap — the VOI signal
    path_weight: f64,
}

/// What the opponent branching produced for one node, before evaluation resolves deferred weights.
enum BranchWeight {
    Fixed(f64),
    Deferred(usize, usize),
}

/// Per-request search state, advanced one round at a time so several searches can run in lockstep
/// and pool their per-round observations into a single `infer` call.
struct Search<S> {
    arena: Vec<Node<S>>,
    frontier: Vec<usize>,
    agent: usize,
    opp: usize,
    n_heads: usize,
    stats: SearchStats,
    batch: Vec<usize>,
    opp_obs: Vec<Vec<f32>>,
    new_leaves: Vec<usize>,
}

impl<S> Search<S> {
    fn new(state: S, agent: usize) -> Search<S> {
        let root = Node {
            state,
            obs: Vec::new(),
            depth: 0,
            terminal: false,
            edges: None,
            bootstrap: Vec::new(),
            value: Vec::new(),
            sigma: 0.0,
            path_weight: 1.0,
        };
        Search {
            arena: vec![root],
            frontier: vec![0],
            agent,
            opp: 1 - agent,
            n_heads: 1,
            stats: SearchStats::default(),
            batch: Vec::new(),
            opp_obs: Vec::new(),
            new_leaves: Vec::new(),
        }
    }

    fn active(&self, budget: usize) -> bool {
        !self.frontier.is_empty() && self.stats.expansions < budget
    }
}

/// One round's build phase for a single search: sort the frontier (after the root round), expand its
/// top-k nodes one ply each, and stage the new leaves + opponent observations for the pooled forward.
fn expand_round<G: Game>(s: &mut Search<G::State>, game: &G, cfg: &SearchConfig, first: bool) {
    if !first {
        sort_frontier(&s.arena, &mut s.frontier, cfg);
    }
    let take = cfg
        .top_k
        .min(cfg.expansion_budget - s.stats.expansions)
        .min(s.frontier.len());
    s.batch = s.frontier.drain(..take).collect();
    s.opp_obs.clear();
    s.new_leaves.clear();
    let (agent, opp) = (s.agent, s.opp);
    for ni in s.batch.clone() {
        expand_node(
            &mut s.arena,
            game,
            cfg,
            ni,
            agent,
            opp,
            &mut s.opp_obs,
            &mut s.new_leaves,
        );
        s.stats.expansions += 1;
        s.stats.max_depth = s.stats.max_depth.max(s.arena[ni].depth + 1);
    }
    s.stats.rounds += 1;
}

/// Apply `f(search_index, &mut search)` to every search, in parallel (rayon) when `parallel`, else
/// serially. The per-search work is independent, so this is value-neutral either way; the serial path
/// avoids rayon's per-dispatch cost when the active pool is too small to win from it.
fn for_each_search<S, F>(searches: &mut [Search<S>], parallel: bool, f: F)
where
    S: Send,
    F: Fn(usize, &mut Search<S>) + Sync,
{
    if parallel {
        searches
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, s)| f(i, s));
    } else {
        for (i, s) in searches.iter_mut().enumerate() {
            f(i, s);
        }
    }
}

/// Run several best-first searches over (possibly different) states in lockstep, pooling each round's
/// opponent + leaf observations across all active searches into ONE `infer` call. Pooling does not
/// change any individual search's result — each reads only its own slice of the batched output — so
/// it is a pure throughput win when `infer` is a GPU network. Returns per-request (values, stats).
///
/// The per-search CPU work (expand, evaluate, back up) is independent across searches, so it runs in
/// parallel via rayon; only the pooled-observation gather and the single `infer` call per round are
/// serial. This is value-neutral: every search is deterministic and reads only its own state, so the
/// result is bit-identical to a sequential run regardless of thread count. `infer` is never called
/// off the calling thread, so a Python `infer` callback keeps the GIL on one thread.
fn search_many<G: Game + Sync, F>(
    game: &G,
    cfg: &SearchConfig,
    requests: Vec<(G::State, usize)>,
    collect_interior: bool,
    mut infer: F,
) -> Vec<SearchResult>
where
    // `infer(obs_flat, n_rows) -> values_flat`: obs is one contiguous row-major `[n_rows, dim]` buffer
    // (moved in, so the binding hands it to numpy with no copy); values come back as one contiguous
    // row-major `[n_rows, K, A]` buffer (K inferred from its length). Flat on both sides avoids the
    // per-row obs clones and the per-leaf nested-`Vec` allocations the boundary would otherwise incur.
    F: FnMut(Vec<f32>, usize) -> Vec<f64>,
    G::State: Send,
{
    debug_assert_eq!(game.num_agents(), 2);
    let a = game.action_count();
    let mut searches: Vec<Search<G::State>> = requests
        .into_iter()
        .map(|(s, agent)| Search::new(s, agent))
        .collect();
    let budget = cfg.expansion_budget;
    let mut first = true;
    loop {
        let active: Vec<usize> = (0..searches.len())
            .filter(|&i| searches[i].active(budget))
            .collect();
        if active.is_empty() {
            break;
        }
        // Parallelize across searches only when the active pool is large enough to amortize rayon's
        // per-dispatch cost; a solo search (e.g. `selective_search`) or a near-dead pool runs serially.
        let parallel = active.len() >= 2;

        // Build phase: each active search expands its top-k one ply (staging leaves + opp observations)
        // and enqueues its surviving leaves for a future round. Folding the frontier push in here —
        // rather than a separate parallel pass — drops a barrier per round; the push needs only the
        // leaves' depths (known after expansion), not the not-yet-computed leaf values. Re-checking
        // `active` inside is equivalent to the list above (nothing mutates between).
        for_each_search(&mut searches, parallel, |_, s| {
            if s.active(budget) {
                expand_round(s, game, cfg, first);
                for k in 0..s.new_leaves.len() {
                    let li = s.new_leaves[k];
                    if s.arena[li].depth < cfg.max_depth {
                        s.frontier.push(li);
                    }
                }
            }
        });
        first = false;

        // Pool this round's observations across all active searches into one contiguous row-major
        // buffer (serial: order matters), recording each search's row span. `span_by_idx[si].is_some()`
        // marks "active this round". `row_start` is a row index into the batch / the returned values.
        let mut obs_flat: Vec<f32> = Vec::new();
        let mut span_by_idx: Vec<Option<(usize, usize)>> = vec![None; searches.len()]; // (row_start, n_opp)
        let mut n_rows = 0;
        for &si in &active {
            let s = &searches[si];
            let row_start = n_rows;
            for o in &s.opp_obs {
                obs_flat.extend_from_slice(o);
            }
            for &li in &s.new_leaves {
                obs_flat.extend_from_slice(&s.arena[li].obs);
            }
            n_rows += s.opp_obs.len() + s.new_leaves.len();
            span_by_idx[si] = Some((row_start, s.opp_obs.len()));
        }
        if n_rows > 0 {
            let q = infer(obs_flat, n_rows); // serial: the (GPU/Python) network forward, flat in/out
            let n_heads = q.len() / (n_rows * a);
            // Evaluate phase: each active search resolves its own row span of the batch.
            for_each_search(&mut searches, parallel, |si, s| {
                if let Some((row_start, n_opp)) = span_by_idx[si] {
                    let rows = n_opp + s.new_leaves.len();
                    let slice = &q[row_start * n_heads * a..(row_start + rows) * n_heads * a];
                    s.n_heads = n_heads;
                    evaluate(
                        &mut s.arena,
                        &s.batch,
                        &s.new_leaves,
                        slice,
                        n_opp,
                        n_heads,
                        a,
                        cfg,
                        &mut s.stats,
                    );
                }
            });
        }
    }

    searches
        .into_par_iter()
        .map(|mut s| {
            resolve(&mut s.arena, 0, cfg.gamma, s.n_heads);
            let values = node_action_values(&s.arena, 0, cfg.gamma, s.n_heads, a);
            let mut interior: Vec<InteriorTarget> = Vec::new();
            if collect_interior && s.arena[0].edges.is_some() {
                // Walk the expanded tree below the root (the root itself is the decision recorded as
                // `values`), DFS in edge-then-branch order to match the oracle.
                let root_edges = take_edges(&s.arena, 0);
                for edge in &root_edges {
                    for &(_, _, child) in edge {
                        collect_interior_targets(
                            &s.arena,
                            child,
                            cfg.gamma,
                            s.n_heads,
                            a,
                            &mut interior,
                        );
                    }
                }
            }
            (values, interior, s.stats)
        })
        .collect()
}

/// Build a `SnakeGame` + `SearchConfig` from `SearchParams`. The snake-specific config splits onto
/// the game; the search keeps the game-agnostic knobs.
fn snake_game_and_config(p: &SearchParams) -> (SnakeGame, SearchConfig) {
    let game = SnakeGame {
        grid_size: p.grid_size,
        initial_length: p.initial_length,
        play_to_last: p.play_to_last,
        win_food_lead: p.win_food_lead,
        reward: p.reward,
    };
    let cfg = SearchConfig {
        gamma: p.gamma,
        beta: p.beta,
        expansion_budget: p.expansion_budget,
        top_k: p.top_k,
        max_depth: p.max_depth,
        food_samples: p.food_samples,
        opponent: p.opponent,
    };
    (game, cfg)
}

/// Snake wrapper over the generic [`search_many`]: maps each `([Snake;2], HashSet<Cell>)` request to a
/// `SnakeState` and runs the generic engine. The public API is unchanged.
pub fn selective_search_many<F>(
    p: &SearchParams,
    requests: Vec<([Snake; 2], HashSet<Cell>, usize)>,
    collect_interior: bool,
    infer: F,
) -> Vec<SearchResult>
where
    F: FnMut(Vec<f32>, usize) -> Vec<f64>,
{
    let (game, cfg) = snake_game_and_config(p);
    let requests: Vec<(SnakeState, usize)> = requests
        .into_iter()
        .map(|(snakes, food, agent)| (SnakeState { snakes, food }, agent))
        .collect();
    search_many(&game, &cfg, requests, collect_interior, infer)
}

/// Single-request convenience wrapper over [`selective_search_many`].
pub fn selective_search<F>(
    p: &SearchParams,
    snakes: [Snake; 2],
    food: HashSet<Cell>,
    agent: usize,
    collect_interior: bool,
    infer: F,
) -> SearchResult
where
    F: FnMut(Vec<f32>, usize) -> Vec<f64>,
{
    selective_search_many(p, vec![(snakes, food, agent)], collect_interior, infer)
        .pop()
        .unwrap()
}

/// Resolve a round's batched forward: opponent rows -> chance weights, leaf rows -> per-head
/// bootstrap + sigma, then write branch weights and child path-weights. `q` is this search's flat
/// row-major slice `[rows, k, A]`; row `r`'s `[k, A]` block is `q[r*k*A .. (r+1)*k*A]`.
#[allow(clippy::too_many_arguments)]
fn evaluate<S>(
    arena: &mut [Node<S>],
    batch: &[usize],
    new_leaves: &[usize],
    q: &[f64],
    n_opp: usize,
    k: usize,
    a: usize,
    cfg: &SearchConfig,
    stats: &mut SearchStats,
) {
    let row = |r: usize| -> &[f64] { &q[r * k * a..(r + 1) * k * a] }; // [k, A], head-major

    // Opponent move probabilities from the head-mean Q (shared across heads, so chance weights stay
    // scalar and sigma reflects only the agent's own value disagreement).
    let opp_probs: Vec<Vec<f64>> = (0..n_opp)
        .map(|i| match cfg.opponent {
            Opponent::Distributional { temperature, floor } => {
                softmax_floor(&head_mean(row(i), k, a), temperature, floor)
            }
            Opponent::Uniform => Vec::new(), // uniform registers no opponent observations
        })
        .collect();

    for (j, &li) in new_leaves.iter().enumerate() {
        let leaf_q = row(n_opp + j); // [k, A]
        let boot: Vec<f64> = (0..k)
            .map(|h| {
                leaf_q[h * a..(h + 1) * a]
                    .iter()
                    .copied()
                    .fold(f64::NEG_INFINITY, f64::max)
            })
            .collect();
        arena[li].sigma = std(&boot);
        arena[li].bootstrap = boot;
        stats.leaves += 1;
        stats.sigma_sum += arena[li].sigma;
    }

    // Resolve weights and child path-weights. Collected first, then applied, to avoid borrowing two
    // arena entries (a node and its child) at once.
    let mut weight_updates: Vec<(usize, usize, usize, f64)> = Vec::new();
    let mut path_updates: Vec<(usize, f64)> = Vec::new();
    for &ni in batch {
        let parent_pw = arena[ni].path_weight;
        if let Some(edges) = &arena[ni].edges {
            for (ei, edge) in edges.iter().enumerate() {
                for (bi, b) in edge.branches.iter().enumerate() {
                    let w = match b.deferred {
                        Some((oi, si)) => opp_probs[oi][si] * b.scale,
                        None => b.weight,
                    };
                    weight_updates.push((ni, ei, bi, w));
                    path_updates.push((b.child, parent_pw * w * cfg.gamma));
                }
            }
        }
    }
    for (ni, ei, bi, w) in weight_updates {
        arena[ni].edges.as_mut().unwrap()[ei].branches[bi].weight = w;
    }
    for (c, pw) in path_updates {
        arena[c].path_weight = pw;
    }
}

fn sort_frontier<S>(arena: &[Node<S>], frontier: &mut [usize], cfg: &SearchConfig) {
    let max_voi = frontier
        .iter()
        .map(|&i| arena[i].path_weight * arena[i].sigma)
        .fold(0.0, f64::max);
    let max_voi = if max_voi > 0.0 { max_voi } else { 1.0 };
    let key = |i: usize| -> (f64, f64, f64) {
        let n = &arena[i];
        let voi = cfg.beta * (n.path_weight * n.sigma) / max_voi;
        let depth_term = (1.0 - cfg.beta) * (n.depth as f64) / (cfg.max_depth as f64);
        (-(voi + depth_term), -(n.depth as f64), -n.path_weight)
    };
    frontier.sort_by(|&a, &b| key(a).partial_cmp(&key(b)).unwrap());
}

#[allow(clippy::too_many_arguments)]
fn expand_node<G: Game>(
    arena: &mut Vec<Node<G::State>>,
    game: &G,
    cfg: &SearchConfig,
    ni: usize,
    agent: usize,
    opp: usize,
    opp_obs: &mut Vec<Vec<f32>>,
    new_leaves: &mut Vec<usize>,
) {
    let state = arena[ni].state.clone();
    let depth = arena[ni].depth;

    // Opponent branching, shared across the agent's edges at this node. `opp_action` indexes into
    // `0..action_count`; for a dead opponent the single branch carries a placeholder index (ignored).
    let opp_legal = game.legal_actions(&state, opp);
    let branching: Vec<(usize, BranchWeight)> = if opp_legal.is_empty() {
        vec![(0, BranchWeight::Fixed(1.0))]
    } else {
        match cfg.opponent {
            Opponent::Uniform => {
                let prob = 1.0 / opp_legal.len() as f64;
                opp_legal
                    .iter()
                    .map(|&oa| (oa, BranchWeight::Fixed(prob)))
                    .collect()
            }
            Opponent::Distributional { .. } => {
                let oi = opp_obs.len();
                opp_obs.push(game.observe(&state, opp));
                opp_legal
                    .iter()
                    .map(|&oa| (oa, BranchWeight::Deferred(oi, oa)))
                    .collect()
            }
        }
    };

    let agent_legal = game.legal_actions(&state, agent);
    let mut edges: Vec<Edge> = Vec::with_capacity(agent_legal.len());
    for &agent_action in agent_legal.iter() {
        let mut branches: Vec<Branch> = Vec::with_capacity(branching.len());
        for (opp_action, bw) in &branching {
            let mut joint = vec![0usize; game.num_agents()];
            joint[agent] = agent_action;
            joint[opp] = *opp_action;

            let t = game.step(&state, &joint);
            let reward = t.rewards[agent];
            let chance = game.chance_outcomes(&state, &t);
            let stochastic = !chance.is_empty();
            let child_state = if stochastic {
                chance[0].1.clone()
            } else {
                t.next_state
            };
            let terminal = t.terminal || game.legal_actions(&child_state, agent).is_empty();

            // food_samples Monte-Carlo fan-out: a stochastic (eaten-apple) transition splits into
            // `food_samples` equally-weighted sub-branches (matching the oracle). Under the
            // deterministic first-empty spawn belief the sub-branches are identical (see module docs) —
            // this is the structure a stochastic spawn belief would populate. The 1/n weight goes onto
            // fixed branches directly; for deferred (distributional) ones it rides `scale`, applied to
            // the resolved opponent probability during evaluation.
            let n = if stochastic && cfg.food_samples > 1 {
                cfg.food_samples
            } else {
                1
            };
            let (weight, deferred, scale) = match *bw {
                BranchWeight::Fixed(f) => (f / n as f64, None, 1.0),
                BranchWeight::Deferred(oi, si) => (0.0, Some((oi, si)), 1.0 / n as f64),
            };
            if n == 1 {
                // common path: move the single child's state in (no clone)
                let child = if terminal {
                    push_node(arena, child_state, Vec::new(), depth + 1, true)
                } else {
                    let obs = game.observe(&child_state, agent);
                    let idx = push_node(arena, child_state, obs, depth + 1, false);
                    new_leaves.push(idx);
                    idx
                };
                branches.push(Branch {
                    weight,
                    deferred,
                    scale,
                    reward,
                    child,
                });
            } else {
                let obs = if terminal {
                    Vec::new()
                } else {
                    game.observe(&child_state, agent)
                };
                for _ in 0..n {
                    let child =
                        push_node(arena, child_state.clone(), obs.clone(), depth + 1, terminal);
                    if !terminal {
                        new_leaves.push(child);
                    }
                    branches.push(Branch {
                        weight,
                        deferred,
                        scale,
                        reward,
                        child,
                    });
                }
            }
        }
        edges.push(Edge { branches });
    }
    arena[ni].edges = Some(edges);
}

fn push_node<S>(
    arena: &mut Vec<Node<S>>,
    state: S,
    obs: Vec<f32>,
    depth: i32,
    terminal: bool,
) -> usize {
    arena.push(Node {
        state,
        obs,
        depth,
        terminal,
        edges: None,
        bootstrap: Vec::new(),
        value: Vec::new(),
        sigma: 0.0,
        path_weight: 1.0,
    });
    arena.len() - 1
}

/// One bottom-up pass caching per-head `node.value`: 0 at terminals, the bootstrap at frontier
/// leaves, the per-head max over agent actions of the chance-averaged edge value at decision nodes.
fn resolve<S>(arena: &mut Vec<Node<S>>, idx: usize, gamma: f64, k: usize) {
    if arena[idx].terminal {
        return; // value stays empty; treated as 0 in edge_value
    }
    if arena[idx].edges.is_none() {
        arena[idx].value = arena[idx].bootstrap.clone();
        return;
    }
    let edges = take_edges(arena, idx);
    for edge in &edges {
        for &(_, _, child) in edge {
            resolve(arena, child, gamma, k);
        }
    }
    let mut value = vec![f64::NEG_INFINITY; k];
    for edge in &edges {
        let ev = edge_value(arena, edge, gamma, k);
        for h in 0..k {
            value[h] = value[h].max(ev[h]);
        }
    }
    arena[idx].value = value;
}

/// Snapshot a node's edges as (weight, reward, child) so we can recurse with `&mut arena`.
fn take_edges<S>(arena: &[Node<S>], idx: usize) -> Vec<Vec<(f64, f64, usize)>> {
    arena[idx]
        .edges
        .as_ref()
        .unwrap()
        .iter()
        .map(|e| {
            e.branches
                .iter()
                .map(|b| (b.weight, b.reward, b.child))
                .collect()
        })
        .collect()
}

/// Per-head chance-averaged value of one edge: sum over branches of `w * (r + gamma * child.value)`,
/// with terminal children contributing `w * r` (their continuation value is 0).
fn edge_value<S>(
    arena: &[Node<S>],
    branches: &[(f64, f64, usize)],
    gamma: f64,
    k: usize,
) -> Vec<f64> {
    let mut acc = vec![0.0; k];
    for &(w, r, c) in branches {
        if arena[c].terminal {
            for a in acc.iter_mut() {
                *a += w * r;
            }
        } else {
            for (h, a) in acc.iter_mut().enumerate() {
                *a += w * (r + gamma * arena[c].value[h]);
            }
        }
    }
    acc
}

/// A decision node's action values as `[K][A]` (per head, per relative action) — the chance-averaged
/// value of each agent action. Used for both the root target and interior TreeStrap targets.
fn node_action_values<S>(
    arena: &[Node<S>],
    idx: usize,
    gamma: f64,
    k: usize,
    a: usize,
) -> Vec<Vec<f64>> {
    let edges = match &arena[idx].edges {
        None => return vec![vec![0.0; a]; k],
        Some(_) => take_edges(arena, idx),
    };
    let per_action: Vec<Vec<f64>> = edges
        .iter()
        .map(|e| edge_value(arena, e, gamma, k))
        .collect(); // [A][K]
    (0..k)
        .map(|h| per_action.iter().map(|ev| ev[h]).collect())
        .collect() // -> [K][A]
}

/// DFS-collect every expanded non-terminal MAX node at or below `idx` as `(obs, [K][A] values)` —
/// true TreeStrap data. Terminal and unexpanded-frontier nodes are skipped (no backed-up values).
fn collect_interior_targets<S>(
    arena: &[Node<S>],
    idx: usize,
    gamma: f64,
    k: usize,
    a: usize,
    out: &mut Vec<InteriorTarget>,
) {
    if arena[idx].terminal || arena[idx].edges.is_none() {
        return;
    }
    out.push((
        arena[idx].obs.clone(),
        node_action_values(arena, idx, gamma, k, a),
    ));
    let edges = take_edges(arena, idx);
    for edge in &edges {
        for &(_, _, child) in edge {
            collect_interior_targets(arena, child, gamma, k, a, out);
        }
    }
}

/// Per-action mean over heads of one node's flat `[k, A]` (head-major) Q block.
fn head_mean(row: &[f64], k: usize, a: usize) -> Vec<f64> {
    (0..a)
        .map(|j| (0..k).map(|h| row[h * a + j]).sum::<f64>() / k as f64)
        .collect()
}

fn std(values: &[f64]) -> f64 {
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let var = values.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / n;
    var.sqrt()
}

/// `p = (1 - floor) * softmax(q / temperature) + floor / n`, matching `DistributionalSelfPlayOpponent`.
fn softmax_floor(q: &[f64], temperature: f64, floor: f64) -> Vec<f64> {
    let qmax = q.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let z: Vec<f64> = q
        .iter()
        .map(|&v| ((v - qmax) / temperature).exp())
        .collect();
    let zsum: f64 = z.iter().sum();
    let n = q.len() as f64;
    z.iter()
        .map(|&zi| (1.0 - floor) * zi / zsum + floor / n)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::action::Action;
    use crate::snake::Snake;

    fn snake(cells: &[Cell], dir: Action) -> Snake {
        Snake {
            body: cells.iter().copied().collect(),
            direction: dir,
            alive: true,
        }
    }

    fn params() -> SearchParams {
        SearchParams {
            grid_size: 12,
            initial_length: 3,
            play_to_last: true,
            win_food_lead: None,
            gamma: 0.99,
            beta: 1.0,
            expansion_budget: 30,
            top_k: 4,
            max_depth: 6,
            food_samples: 1,
            reward: Reward {
                step: 0.0,
                food: 0.0,
                loss: -10.0,
                draw: -6.0,
                kill: 20.0,
                win: 20.0,
                survival: 0.0,
            },
            opponent: Opponent::Uniform,
        }
    }

    #[test]
    fn fatal_actions_score_the_loss_and_survivable_turns_score_higher() {
        // A in the top-left corner heading Left: Forward (Left) and Right (Up) both run off-grid;
        // only Left (Down) survives. With a zero value function the only signal is the death penalty.
        let snakes = [
            snake(&[(0, 0), (0, 1), (0, 2)], Action::Left),
            snake(&[(6, 6), (6, 7), (6, 8)], Action::Left), // opponent, far away
        ];
        let p = params();
        let (values, _interior, stats) =
            selective_search(&p, snakes, HashSet::new(), 0, false, |_obs, n| {
                vec![0.0; n * 3]
            });
        let v = &values[0]; // single head
        assert!(
            (v[0] - (-10.0)).abs() < 1e-9,
            "fatal Forward should score the loss: {v:?}"
        );
        assert!(
            (v[2] - (-10.0)).abs() < 1e-9,
            "fatal Right should score the loss: {v:?}"
        );
        assert!(
            v[1] > v[0] && v[1] > v[2],
            "survivable Left should win: {v:?}"
        );
        assert!(stats.expansions > 0 && stats.rounds > 0);
    }

    #[test]
    fn ensemble_bootstrap_sigma_drives_voi_priority() {
        // Two heads that disagree -> nonzero sigma -> beta=1 priority can steer on it (smoke check
        // that multi-head search runs and produces per-head root values).
        let snakes = [
            snake(&[(6, 5), (6, 4), (6, 3)], Action::Right),
            snake(&[(2, 8), (2, 9), (1, 9)], Action::Left),
        ];
        let mut p = params();
        p.expansion_budget = 24;
        let (values, _interior, stats) =
            selective_search(&p, snakes, HashSet::new(), 0, false, two_head_infer);
        assert_eq!(values.len(), 2); // two heads
        assert_eq!(values[0].len(), 3);
        assert!(stats.expansions > 0);
    }

    // Two disagreeing heads, sum-dependent — exercises sigma + the VOI priority under pooling. Flat
    // `(obs[n*dim], n) -> values[n*2*3]` (head-major rows), matching the new infer interface.
    fn two_head_infer(obs: Vec<f32>, n: usize) -> Vec<f64> {
        let dim = obs.len() / n;
        let mut out = Vec::with_capacity(n * 2 * 3);
        for i in 0..n {
            let s = obs[i * dim..(i + 1) * dim].iter().sum::<f32>() as f64;
            out.extend_from_slice(&[
                s.sin(),
                s.cos(),
                (s * 0.5).sin(), // head 0
                (s + 1.0).sin(),
                (s * 0.3).cos(),
                (s * 0.2).sin(), // head 1
            ]);
        }
        out
    }

    type Request = ([Snake; 2], HashSet<Cell>, usize);

    fn two_requests() -> (Request, Request) {
        let a = (
            [
                snake(&[(6, 5), (6, 4), (6, 3)], Action::Right),
                snake(&[(2, 8), (2, 9), (1, 9)], Action::Left),
            ],
            HashSet::new(),
            0usize,
        );
        let b = (
            [
                snake(&[(3, 3), (3, 2), (3, 1)], Action::Right),
                snake(&[(8, 8), (8, 9), (9, 9)], Action::Left),
            ],
            HashSet::new(),
            1usize,
        );
        (a, b)
    }

    #[test]
    fn pooling_matches_solo_searches_bit_for_bit() {
        let (a, b) = two_requests();
        let mut p = params();
        p.expansion_budget = 24;
        let many = selective_search_many(&p, vec![a.clone(), b.clone()], false, two_head_infer);
        let solo_a = selective_search(&p, a.0.clone(), a.1.clone(), a.2, false, two_head_infer);
        let solo_b = selective_search(&p, b.0.clone(), b.1.clone(), b.2, false, two_head_infer);
        assert_eq!(
            many[0].0, solo_a.0,
            "pooled values must equal the solo search"
        );
        assert_eq!(many[1].0, solo_b.0);
        assert_eq!(many[0].2.expansions, solo_a.2.expansions);
        assert_eq!(many[1].2.expansions, solo_b.2.expansions);
        assert_eq!(many[0].2.rounds, solo_a.2.rounds);
    }

    #[test]
    fn parallel_search_is_thread_count_independent() {
        // The rayon-parallel per-search work is value-neutral: running the same pooled search inside a
        // 1-thread pool and a 4-thread pool must give bit-identical values, interior, and stats.
        let (a, b) = two_requests();
        let mut p = params();
        p.expansion_budget = 24;
        let run = |threads: usize| {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            pool.install(|| {
                selective_search_many(&p, vec![a.clone(), b.clone()], true, two_head_infer)
            })
        };
        let one = run(1);
        let many = run(4);
        for i in 0..2 {
            assert_eq!(
                one[i].0, many[i].0,
                "values must not depend on thread count"
            );
            assert_eq!(
                one[i].1, many[i].1,
                "interior targets must not depend on thread count"
            );
            assert_eq!(
                (
                    one[i].2.max_depth,
                    one[i].2.expansions,
                    one[i].2.leaves,
                    one[i].2.rounds
                ),
                (
                    many[i].2.max_depth,
                    many[i].2.expansions,
                    many[i].2.leaves,
                    many[i].2.rounds
                ),
            );
        }
    }

    #[test]
    fn pooling_issues_fewer_forwards_than_solo() {
        use std::cell::Cell as Counter;
        let (a, b) = two_requests();
        let mut p = params();
        p.expansion_budget = 24;

        let pooled = Counter::new(0usize);
        selective_search_many(&p, vec![a.clone(), b.clone()], false, |o, n| {
            pooled.set(pooled.get() + 1);
            two_head_infer(o, n)
        });
        let solo = Counter::new(0usize);
        selective_search(&p, a.0.clone(), a.1.clone(), a.2, false, |o, n| {
            solo.set(solo.get() + 1);
            two_head_infer(o, n)
        });
        selective_search(&p, b.0.clone(), b.1.clone(), b.2, false, |o, n| {
            solo.set(solo.get() + 1);
            two_head_infer(o, n)
        });
        assert!(
            pooled.get() < solo.get(),
            "pooled forwards {} should be fewer than solo {}",
            pooled.get(),
            solo.get()
        );
    }

    #[test]
    fn all_terminal_root_returns_single_head_without_calling_infer() {
        // Agent boxed in heading Left at the top-left: Forward (Left) and Right (Up) hit the wall, and
        // Left (Down) moves onto its own neck (self-collision). Every root child is terminal, so the
        // round produces no observations -> infer is never called and n_heads falls back to 1.
        let snakes = [
            snake(&[(0, 0), (1, 0), (2, 0)], Action::Left),
            snake(&[(5, 5), (5, 6), (5, 7)], Action::Left),
        ];
        let p = params(); // uniform opponent, loss = -10
        let mut calls = 0usize;
        let results =
            selective_search_many(&p, vec![(snakes, HashSet::new(), 0)], false, |_obs, n| {
                calls += 1;
                vec![0.0; n * 3]
            });
        let (values, _interior, stats) = &results[0];
        assert_eq!(
            calls, 0,
            "no observations this round -> infer must not be called"
        );
        assert_eq!(
            values.len(),
            1,
            "n_heads falls back to 1 when nothing was evaluated"
        );
        for v in &values[0] {
            assert!(
                (v - (-10.0)).abs() < 1e-9,
                "every action is fatal -> the loss: {values:?}"
            );
        }
        assert_eq!(stats.leaves, 0);
        assert_eq!(stats.expansions, 1);
    }

    #[test]
    fn food_samples_fans_out_only_eating_branches() {
        // Agent mid-grid heading Right with an apple directly ahead; opponent dead (one opp branch per
        // edge). In a single root expansion only Forward eats, so food_samples=3 turns its one child
        // into three while Left/Right keep one each: 3 leaves at k=1 -> 5 at k=3.
        let snakes = [
            snake(&[(6, 5), (6, 4), (6, 3)], Action::Right),
            Snake {
                body: [(0, 0), (1, 0)].into_iter().collect(),
                direction: Action::Down,
                alive: false,
            },
        ];
        let food: HashSet<Cell> = [(6, 6)].into_iter().collect();
        let infer = |_o: Vec<f32>, n: usize| vec![0.0; n * 3]; // single head, zero values
        let leaves = |samples: usize| {
            let mut p = params();
            p.expansion_budget = 1; // one expansion: just the root
            p.food_samples = samples;
            selective_search(&p, snakes.clone(), food.clone(), 0, false, infer)
                .2
                .leaves
        };
        assert_eq!(leaves(1), 3);
        assert_eq!(
            leaves(3),
            5,
            "the eating (Forward) branch fans 1 -> 3; the others are unchanged"
        );
    }

    #[test]
    fn generic_search_many_matches_snake_wrapper() {
        // The generic path (SnakeGame + SnakeState fed straight into search_many) must produce
        // bit-identical results to the public snake wrapper on the same state.
        let snakes = [
            snake(&[(6, 5), (6, 4), (6, 3)], Action::Right),
            snake(&[(2, 8), (2, 9), (1, 9)], Action::Left),
        ];
        let food: HashSet<Cell> = [(4, 4)].into_iter().collect();
        let mut p = params();
        p.expansion_budget = 24;

        let (game, cfg) = snake_game_and_config(&p);
        let state = SnakeState {
            snakes: snakes.clone(),
            food: food.clone(),
        };
        let generic = search_many(&game, &cfg, vec![(state, 0usize)], true, two_head_infer)
            .pop()
            .unwrap();
        let wrapped = selective_search(&p, snakes, food, 0, true, two_head_infer);

        assert_eq!(generic.0, wrapped.0, "root values must match");
        assert_eq!(generic.1, wrapped.1, "interior targets must match");
        assert_eq!(
            (
                generic.2.max_depth,
                generic.2.expansions,
                generic.2.leaves,
                generic.2.rounds
            ),
            (
                wrapped.2.max_depth,
                wrapped.2.expansions,
                wrapped.2.leaves,
                wrapped.2.rounds
            ),
        );
    }
}
