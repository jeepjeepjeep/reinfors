//! Batched best-first selective expectimax.

use rayon::prelude::*;

use crate::encoder::{ActionView, StateEncoder};
use crate::game::{Actor, Game, Rng};
use crate::policy::{ChanceMode, MAX_ENUMERATED_OUTCOMES, MAX_JOINT_SLOTS};
use crate::reward::Reward;
use crate::rng::SplitMix64;

/// Belief model for each co-mover's action.
#[derive(Clone, Copy)]
pub enum Opponent {
    Uniform,
    Distributional {
        temperature: f64,
        floor: f64,
    },
    /// Minimax: opponent decisions back up as a minimum over their moves; `move_cap` beams
    /// non-root nodes by their own leaf evaluation (`usize::MAX` = full width).
    Adversarial {
        move_cap: usize,
    },
}

/// Game-independent selective-search parameters.
#[derive(Clone, Copy)]
pub struct SearchConfig {
    pub gamma: f64,
    pub beta: f64,
    pub expansion_budget: usize,
    pub top_k: usize,
    pub max_depth: i32,
    pub chance: ChanceMode,
    pub opponent: Opponent,
}

/// Shared search accounting. MCTS maintains the exact identity
/// `expansions = fresh + hit + shared + terminal + depthcap - extra_eval_rows`; the subtraction
/// removes auxiliary perspective rows from simulation fate.
#[derive(Default, Clone, Copy, Debug)]
pub struct SearchStats {
    pub max_depth: i32,
    pub expansions: usize,
    pub leaves: usize,
    pub rounds: usize,
    pub sigma_sum: f64,
    pub terminal_sims: usize,
    pub depthcap_sims: usize,
    pub shared_rows: usize,
    pub fresh_rows: usize,
    pub hit_rows: usize,
    pub extra_eval_rows: usize,
}

pub type InteriorTarget = (Vec<f32>, Vec<Vec<f64>>);

pub type SearchResult = (Vec<Vec<f64>>, Vec<InteriorTarget>, SearchStats);

// Deferred factors are resolved from routed co-mover rows, then multiplied by `scale`.
struct Branch {
    weight: f64,
    deferred: Vec<(usize, usize)>,
    scale: f64,
    reward: f64,
    child: usize,
}

struct Edge {
    branches: Vec<Branch>,
}

struct Node<S> {
    state: S,
    obs: Vec<f32>,
    depth: i32,
    terminal: bool,
    edges: Option<Vec<Edge>>,
    bootstrap: Vec<f64>,
    value: Vec<f64>,
    sigma: f64,
    path_weight: f64,
    // True only at the searcher's decisions; opponent nodes must not become TreeStrap targets.
    max_node: bool,
    // Minimum over edges; values stay in the searcher's perspective — no sign flip anywhere.
    min_node: bool,
    // Adversarial beam only: this node's own leaf evaluation, kept to order a later expansion.
    q_row: Vec<f64>,
}

#[derive(Clone, Copy)]
enum BranchWeight {
    Fixed(f64),
    Deferred(usize, usize),
}

struct MoveWeight {
    fixed: f64,
    deferred: Vec<(usize, usize)>,
}

impl From<BranchWeight> for MoveWeight {
    fn from(bw: BranchWeight) -> MoveWeight {
        match bw {
            BranchWeight::Fixed(f) => MoveWeight {
                fixed: f,
                deferred: Vec::new(),
            },
            BranchWeight::Deferred(oi, si) => MoveWeight {
                fixed: 1.0,
                deferred: vec![(oi, si)],
            },
        }
    }
}

struct Search<S> {
    arena: Vec<Node<S>>,
    frontier: Vec<usize>,
    agent: usize,
    n_heads: usize,
    stats: SearchStats,
    batch: Vec<usize>,
    opp_obs: Vec<Vec<f32>>,
    // Each routed row must retain the mover whose action frame encoded it.
    opp_legal: Vec<(usize, Vec<usize>)>,
    new_leaves: Vec<usize>,
    rng: SplitMix64,
}

impl<S> Search<S> {
    fn new(state: S, agent: usize, seed: u64) -> Search<S> {
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
            max_node: false,
            min_node: false,
            q_row: Vec::new(),
        };
        Search {
            arena: vec![root],
            frontier: vec![0],
            agent,
            n_heads: 1,
            stats: SearchStats::default(),
            batch: Vec::new(),
            opp_obs: Vec::new(),
            opp_legal: Vec::new(),
            new_leaves: Vec::new(),
            rng: SplitMix64::new(seed),
        }
    }

    fn active(&self, budget: usize) -> bool {
        !self.frontier.is_empty() && self.stats.expansions < budget
    }
}

/// Build one lockstep round: expand the selected frontier and fuse every resulting network row
/// into the next pooled forward.
fn expand_round<G: Game>(
    s: &mut Search<G::State>,
    game: &G,
    enc: &dyn StateEncoder<State = G::State>,
    reward: &dyn Reward<Event = G::Event>,
    cfg: &SearchConfig,
    first: bool,
) {
    if !first {
        sort_frontier(&s.arena, &mut s.frontier, cfg);
    }
    let take = cfg
        .top_k
        .min(cfg.expansion_budget - s.stats.expansions)
        .min(s.frontier.len());
    s.batch = s.frontier.drain(..take).collect();
    s.opp_obs.clear();
    s.opp_legal.clear();
    s.new_leaves.clear();
    let agent = s.agent;
    for ni in s.batch.clone() {
        expand_node(
            &mut s.arena,
            game,
            enc,
            reward,
            cfg,
            ni,
            agent,
            &mut s.opp_obs,
            &mut s.opp_legal,
            &mut s.new_leaves,
            &mut s.rng,
        );
        s.stats.expansions += 1;
        s.stats.max_depth = s.stats.max_depth.max(s.arena[ni].depth + 1);
    }
    s.stats.rounds += 1;
}

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

/// Run searches in lockstep and pool each round's network rows.
#[allow(clippy::too_many_arguments)]
pub fn search_many<G: Game + Sync, F>(
    game: &G,
    enc: &dyn StateEncoder<State = G::State>,
    reward: &dyn Reward<Event = G::Event>,
    cfg: &SearchConfig,
    requests: Vec<(G::State, usize)>,
    collect_interior: bool,
    seed: u64,
    mut infer: F,
) -> Vec<SearchResult>
where
    F: FnMut(&[usize], Vec<f32>, usize) -> Vec<f64>,
    G::State: Send,
{
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
    // Per-request RNG streams make results independent of the parallel schedule.
    let mut searches: Vec<Search<G::State>> = requests
        .into_iter()
        .enumerate()
        .map(|(i, (s, agent))| Search::new(s, agent, seed.wrapping_add(i as u64)))
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
        let parallel = active.len() >= 2;

        for_each_search(&mut searches, parallel, |_, s| {
            if s.active(budget) {
                expand_round(s, game, enc, reward, cfg, first);
                for k in 0..s.new_leaves.len() {
                    let li = s.new_leaves[k];
                    if s.arena[li].depth < cfg.max_depth {
                        s.frontier.push(li);
                    }
                }
            }
        });
        first = false;

        let mut obs_flat: Vec<f32> = Vec::new();
        let mut players: Vec<usize> = Vec::new();
        let mut span_by_idx: Vec<Option<(usize, usize)>> = vec![None; searches.len()];
        let mut n_rows = 0;
        for &si in &active {
            let s = &searches[si];
            let row_start = n_rows;
            for (o, (mover, _)) in s.opp_obs.iter().zip(&s.opp_legal) {
                obs_flat.extend_from_slice(o);
                players.push(*mover);
            }
            for &li in &s.new_leaves {
                obs_flat.extend_from_slice(&s.arena[li].obs);
                players.push(s.agent);
            }
            n_rows += s.opp_obs.len() + s.new_leaves.len();
            span_by_idx[si] = Some((row_start, s.opp_obs.len()));
        }
        if n_rows > 0 {
            let q = infer(&players, obs_flat, n_rows);
            let n_heads = q.len() / (n_rows * a);
            for_each_search(&mut searches, parallel, |si, s| {
                if let Some((row_start, n_opp)) = span_by_idx[si] {
                    let rows = n_opp + s.new_leaves.len();
                    let slice = &q[row_start * n_heads * a..(row_start + rows) * n_heads * a];
                    s.n_heads = n_heads;
                    let agent = s.agent;
                    // At a sequential opponent leaf, available actions belong to the mover even
                    // though the value row is evaluated from the searcher's perspective.
                    let legal_of = |state: &G::State| match game.actor(state) {
                        Actor::Agent(mover) => game.legal_actions(state, mover),
                        Actor::Simultaneous => game.legal_actions(state, agent),
                        Actor::Chance => unreachable!("chance actors are not searched"),
                    };
                    let adversarial = matches!(cfg.opponent, Opponent::Adversarial { .. });
                    let min_leaf = |state: &G::State| {
                        adversarial && matches!(game.actor(state), Actor::Agent(m) if m != agent)
                    };
                    evaluate(
                        &mut s.arena,
                        &s.batch,
                        &s.new_leaves,
                        &s.opp_legal,
                        enc,
                        s.agent,
                        &legal_of,
                        &min_leaf,
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
            let agent = s.agent;
            // Resolve-phase MAX values are in the searcher's action frame; this deliberately differs
            // from bootstrap leaves, whose legal set belongs to the leaf mover.
            let legal_of = |state: &G::State| game.legal_actions(state, agent);
            let values = node_action_values(&s.arena, 0, cfg.gamma, s.n_heads, a, &legal_of);
            let mut interior: Vec<InteriorTarget> = Vec::new();
            if collect_interior && s.arena[0].edges.is_some() {
                let root_edges = take_edges(&s.arena, 0);
                for edge in &root_edges {
                    for &(_, _, child) in edge {
                        collect_interior_targets(
                            &s.arena,
                            child,
                            cfg.gamma,
                            s.n_heads,
                            a,
                            &legal_of,
                            &mut interior,
                        );
                    }
                }
            }
            (values, interior, s.stats)
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn evaluate<S, L, M>(
    arena: &mut [Node<S>],
    batch: &[usize],
    new_leaves: &[usize],
    opp_legal: &[(usize, Vec<usize>)],
    // Leaf and co-mover rows use different perspectives and action frames.
    view: &dyn ActionView,
    searcher: usize,
    legal_of: &L,
    min_leaf: &M,
    q: &[f64],
    n_opp: usize,
    k: usize,
    a: usize,
    cfg: &SearchConfig,
    stats: &mut SearchStats,
) where
    L: Fn(&S) -> Vec<usize>,
    M: Fn(&S) -> bool,
{
    let row = |r: usize| -> &[f64] { &q[r * k * a..(r + 1) * k * a] };

    // Softmax only legal actions, gathered through the mover's frame.
    let opp_probs: Vec<Vec<f64>> = (0..n_opp)
        .map(|i| match cfg.opponent {
            Opponent::Distributional { temperature, floor } => {
                let mean = head_mean(row(i), k, a);
                let (mover, legal) = &opp_legal[i];
                let gathered: Vec<f64> = legal
                    .iter()
                    .map(|&aid| mean[view.head_index(aid, *mover)])
                    .collect();
                let probs = softmax_floor(&gathered, temperature, floor);
                let mut full = vec![0.0; a];
                for (&aid, p) in legal.iter().zip(probs) {
                    full[aid] = p;
                }
                full
            }
            Opponent::Uniform | Opponent::Adversarial { .. } => Vec::new(),
        })
        .collect();

    for (j, &li) in new_leaves.iter().enumerate() {
        let leaf_q = row(n_opp + j);
        let legal = legal_of(&arena[li].state);
        debug_assert!(!legal.is_empty(), "non-terminal leaf with no legal actions");
        // Adversarial opponent-to-move horizons collapse with min: max would value the
        // opponent's best reply as ours.
        let fold_min = min_leaf(&arena[li].state);
        let boot: Vec<f64> = (0..k)
            .map(|h| {
                let head = &leaf_q[h * a..(h + 1) * a];
                let it = legal
                    .iter()
                    .map(|&aid| head[view.head_index(aid, searcher)]);
                if fold_min {
                    it.fold(f64::INFINITY, f64::min)
                } else {
                    it.fold(f64::NEG_INFINITY, f64::max)
                }
            })
            .collect();
        arena[li].sigma = std(&boot);
        arena[li].bootstrap = boot;
        if let Opponent::Adversarial { move_cap } = cfg.opponent {
            if move_cap != usize::MAX {
                // Game-frame head-mean, retained to order this node's later beamed expansion.
                let mut q_row = vec![f64::NEG_INFINITY; a];
                for &aid in &legal {
                    q_row[aid] = (0..k)
                        .map(|h| leaf_q[h * a + view.head_index(aid, searcher)])
                        .sum::<f64>()
                        / k as f64;
                }
                arena[li].q_row = q_row;
            }
        }
        stats.leaves += 1;
        stats.sigma_sum += arena[li].sigma;
    }

    // Collect updates first to avoid borrowing parent and child arena entries together.
    let mut weight_updates: Vec<(usize, usize, usize, f64)> = Vec::new();
    let mut path_updates: Vec<(usize, f64)> = Vec::new();
    for &ni in batch {
        let parent_pw = arena[ni].path_weight;
        if let Some(edges) = &arena[ni].edges {
            for (ei, edge) in edges.iter().enumerate() {
                for (bi, b) in edge.branches.iter().enumerate() {
                    let w = if b.deferred.is_empty() {
                        b.weight
                    } else {
                        let mut w = b.scale;
                        for &(oi, si) in &b.deferred {
                            w *= opp_probs[oi][si];
                        }
                        w
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

fn agent_branching<G: Game>(
    game: &G,
    enc: &dyn StateEncoder<State = G::State>,
    cfg: &SearchConfig,
    state: &G::State,
    mover: usize,
    opp_obs: &mut Vec<Vec<f32>>,
    opp_legal: &mut Vec<(usize, Vec<usize>)>,
) -> Vec<(usize, BranchWeight)> {
    let legal = game.legal_actions(state, mover);
    if legal.is_empty() {
        return vec![(0, BranchWeight::Fixed(1.0))];
    }
    match cfg.opponent {
        Opponent::Uniform => {
            let prob = 1.0 / legal.len() as f64;
            legal
                .iter()
                .map(|&a| (a, BranchWeight::Fixed(prob)))
                .collect()
        }
        Opponent::Distributional { .. } => {
            let oi = opp_obs.len();
            opp_obs.push(enc.encode(state, mover));
            let branches = legal
                .iter()
                .map(|&a| (a, BranchWeight::Deferred(oi, a)))
                .collect();
            opp_legal.push((mover, legal));
            branches
        }
        Opponent::Adversarial { .. } => unreachable!(
            "adversarial opponent decisions build per-move edges in expand_node, never weighted \
             branches"
        ),
    }
}

#[allow(clippy::too_many_arguments)]
fn push_branches<G: Game>(
    arena: &mut Vec<Node<G::State>>,
    game: &G,
    enc: &dyn StateEncoder<State = G::State>,
    reward: &dyn Reward<Event = G::Event>,
    cfg: &SearchConfig,
    state: &G::State,
    joint: &[usize],
    mw: MoveWeight,
    agent: usize,
    // No legal action means death in simultaneous play, but merely "not your turn" sequentially.
    agent_out_terminal: bool,
    depth: i32,
    new_leaves: &mut Vec<usize>,
    branches: &mut Vec<Branch>,
    rng: &mut dyn Rng,
) {
    let t = game.step(state, joint);
    let step_reward = crate::reward::edge_reward(reward, &t.events, agent);
    // A whole chance chain remains one ply; compound probabilities and edge rewards.
    let mut resolved: Vec<(G::State, f64, f64, bool)> = Vec::with_capacity(1);
    let mut work: Vec<(G::State, f64, f64, bool, usize)> =
        vec![(t.next_state, 1.0, 0.0, t.terminal, 0)];
    while let Some((s, p, r, term, hops)) = work.pop() {
        if term || !matches!(game.actor(&s), Actor::Chance) {
            resolved.push((s, p, r, term));
            continue;
        }
        assert!(
            hops < crate::game::CHANCE_CHAIN_LIMIT,
            "chance-node chain exceeded {} edges — the game cycles through chance states",
            crate::game::CHANCE_CHAIN_LIMIT
        );
        let dist = game.chance_node(&s);
        match cfg.chance {
            ChanceMode::Committed { samples } => {
                let k = samples.max(1);
                // The parent is already popped: resolved + pending + children is the projected fan.
                assert!(
                    resolved.len() + work.len() + k <= MAX_ENUMERATED_OUTCOMES,
                    "a chance chain's flattened fan exceeds the enumeration bound ({}); use a \
                     narrower sampling mode",
                    MAX_ENUMERATED_OUTCOMES
                );
                for _ in 0..k {
                    let idx = dist.draw(rng);
                    let ct = game.apply_chance_node(&s, idx);
                    let er = crate::reward::edge_reward(reward, &ct.events, agent);
                    work.push((ct.next_state, p / k as f64, r + er, ct.terminal, hops + 1));
                }
            }
            ChanceMode::ExpandAll => {
                let count = dist.count();
                assert!(
                    count <= MAX_ENUMERATED_OUTCOMES,
                    "ExpandAll cannot enumerate {count} chance outcomes (bound {}); use a \
                     sampling chance mode for combinatorial outcome spaces",
                    MAX_ENUMERATED_OUTCOMES
                );
                assert!(
                    // As above, guard the projected size after this outcome fan is inserted.
                    resolved.len() + work.len() + count <= MAX_ENUMERATED_OUTCOMES,
                    "a chance chain's flattened fan exceeds the enumeration bound ({}); use a \
                     narrower sampling mode",
                    MAX_ENUMERATED_OUTCOMES
                );
                let probs: Vec<f64> = dist.iter_probs().collect();
                for (idx, pr) in probs.into_iter().enumerate() {
                    let ct = game.apply_chance_node(&s, idx);
                    let er = crate::reward::edge_reward(reward, &ct.events, agent);
                    work.push((ct.next_state, p * pr, r + er, ct.terminal, hops + 1));
                }
            }
            ChanceMode::AlwaysResample => unreachable!(
                "rejected at SelectiveExpectimax construction (no traversal event to redraw on)"
            ),
        }
    }
    for (child_state, p, chain_reward, chain_terminal) in resolved {
        let (weight, scale) = if mw.deferred.is_empty() {
            (mw.fixed * p, 1.0)
        } else {
            (0.0, mw.fixed * p)
        };
        let terminal = chain_terminal
            || (agent_out_terminal && game.legal_actions(&child_state, agent).is_empty());
        let child = if terminal {
            push_node(arena, child_state, Vec::new(), depth + 1, true)
        } else {
            let obs = enc.encode(&child_state, agent);
            let idx = push_node(arena, child_state, obs, depth + 1, false);
            new_leaves.push(idx);
            idx
        };
        branches.push(Branch {
            weight,
            deferred: mw.deferred.clone(),
            scale,
            reward: step_reward + chain_reward,
            child,
        });
    }
}

#[allow(clippy::too_many_arguments)]
fn expand_node<G: Game>(
    arena: &mut Vec<Node<G::State>>,
    game: &G,
    enc: &dyn StateEncoder<State = G::State>,
    reward: &dyn Reward<Event = G::Event>,
    cfg: &SearchConfig,
    ni: usize,
    agent: usize,
    opp_obs: &mut Vec<Vec<f32>>,
    opp_legal: &mut Vec<(usize, Vec<usize>)>,
    new_leaves: &mut Vec<usize>,
    rng: &mut dyn Rng,
) {
    let state = arena[ni].state.clone();
    let depth = arena[ni].depth;
    let num_agents = game.num_agents();

    let (edges, max_node) = match game.actor(&state) {
        Actor::Simultaneous => {
            assert!(
                !matches!(cfg.opponent, Opponent::Adversarial { .. }),
                "adversarial (minimax) backup is undefined for simultaneous decisions; see {}",
                crate::COMPATIBILITY_DOCS
            );
            let co_movers: Vec<usize> = (0..num_agents).filter(|&i| i != agent).collect();
            let co_b: Vec<Vec<(usize, BranchWeight)>> = co_movers
                .iter()
                .map(|&m| agent_branching(game, enc, cfg, &state, m, opp_obs, opp_legal))
                .collect();
            // Overflow could turn an oversized joint fan into an empty search.
            let combos: usize = co_b
                .iter()
                .try_fold(1usize, |acc, b| {
                    acc.checked_mul(b.len()).filter(|&c| c <= MAX_JOINT_SLOTS)
                })
                .unwrap_or_else(|| {
                    panic!(
                        "simultaneous co-mover fan exceeds {} joint branches at one node",
                        MAX_JOINT_SLOTS
                    )
                });
            let agent_legal = game.legal_actions(&state, agent);
            let mut edges = Vec::with_capacity(agent_legal.len());
            for &agent_action in &agent_legal {
                let mut branches = Vec::with_capacity(combos);
                for combo in 0..combos {
                    let mut joint = vec![0usize; num_agents];
                    joint[agent] = agent_action;
                    let mut mw = MoveWeight {
                        fixed: 1.0,
                        deferred: Vec::new(),
                    };
                    let mut rem = combo;
                    for ci in (0..co_b.len()).rev() {
                        let options = &co_b[ci];
                        let (action, bw) = options[rem % options.len()];
                        rem /= options.len();
                        joint[co_movers[ci]] = action;
                        match bw {
                            BranchWeight::Fixed(f) => mw.fixed *= f,
                            BranchWeight::Deferred(oi, si) => mw.deferred.push((oi, si)),
                        }
                    }
                    push_branches(
                        arena,
                        game,
                        enc,
                        reward,
                        cfg,
                        &state,
                        &joint,
                        mw,
                        agent,
                        true,
                        depth,
                        new_leaves,
                        &mut branches,
                        &mut *rng,
                    );
                }
                edges.push(Edge { branches });
            }
            (edges, true)
        }
        Actor::Agent(a) if a == agent => {
            let agent_legal = match cfg.opponent {
                Opponent::Adversarial { move_cap } => {
                    beamed_legal(arena, game, &state, ni, agent, depth, move_cap, true)
                }
                _ => game.legal_actions(&state, agent),
            };
            let mut edges = Vec::with_capacity(agent_legal.len());
            for &action in &agent_legal {
                let mut branches = Vec::new();
                let mut joint = vec![0usize; num_agents];
                joint[agent] = action;
                push_branches(
                    arena,
                    game,
                    enc,
                    reward,
                    cfg,
                    &state,
                    &joint,
                    MoveWeight::from(BranchWeight::Fixed(1.0)),
                    agent,
                    false,
                    depth,
                    new_leaves,
                    &mut branches,
                    &mut *rng,
                );
                edges.push(Edge { branches });
            }
            (edges, true)
        }
        Actor::Agent(mover) => match cfg.opponent {
            // One edge per opponent move: min of per-move expectations, not a weighted sum.
            Opponent::Adversarial { move_cap } => {
                let legal = beamed_legal(arena, game, &state, ni, mover, depth, move_cap, false);
                let mut edges = Vec::with_capacity(legal.len().max(1));
                if legal.is_empty() {
                    let mut branches = Vec::new();
                    let joint = vec![0usize; num_agents];
                    push_branches(
                        arena,
                        game,
                        enc,
                        reward,
                        cfg,
                        &state,
                        &joint,
                        MoveWeight::from(BranchWeight::Fixed(1.0)),
                        agent,
                        false,
                        depth,
                        new_leaves,
                        &mut branches,
                        &mut *rng,
                    );
                    edges.push(Edge { branches });
                } else {
                    for &action in &legal {
                        let mut branches = Vec::new();
                        let mut joint = vec![0usize; num_agents];
                        joint[mover] = action;
                        push_branches(
                            arena,
                            game,
                            enc,
                            reward,
                            cfg,
                            &state,
                            &joint,
                            MoveWeight::from(BranchWeight::Fixed(1.0)),
                            agent,
                            false,
                            depth,
                            new_leaves,
                            &mut branches,
                            &mut *rng,
                        );
                        edges.push(Edge { branches });
                    }
                }
                arena[ni].min_node = true;
                (edges, false)
            }
            _ => {
                let mover_b = agent_branching(game, enc, cfg, &state, mover, opp_obs, opp_legal);
                let mut branches = Vec::with_capacity(mover_b.len());
                for &(action, bw) in &mover_b {
                    let mut joint = vec![0usize; num_agents];
                    joint[mover] = action;
                    push_branches(
                        arena,
                        game,
                        enc,
                        reward,
                        cfg,
                        &state,
                        &joint,
                        MoveWeight::from(bw),
                        agent,
                        false,
                        depth,
                        new_leaves,
                        &mut branches,
                        &mut *rng,
                    );
                }
                (vec![Edge { branches }], false)
            }
        },
        Actor::Chance => unimplemented!("explicit Actor::Chance nodes are not yet supported"),
    };
    arena[ni].edges = Some(edges);
    arena[ni].max_node = max_node;
}

/// Beam-capped legal moves: descending in the searcher's leaf evaluation at its own decisions,
/// ascending at the opponent's. The root, and nodes without a retained `q_row`, keep full width.
#[allow(clippy::too_many_arguments)]
fn beamed_legal<G: Game>(
    arena: &[Node<G::State>],
    game: &G,
    state: &G::State,
    ni: usize,
    mover: usize,
    depth: i32,
    move_cap: usize,
    descending: bool,
) -> Vec<usize> {
    let mut legal = game.legal_actions(state, mover);
    let q_row = &arena[ni].q_row;
    if depth == 0 || move_cap >= legal.len() || q_row.is_empty() {
        return legal;
    }
    legal.sort_by(|&x, &y| {
        let (a, b) = (q_row[x], q_row[y]);
        if descending {
            b.partial_cmp(&a).unwrap()
        } else {
            a.partial_cmp(&b).unwrap()
        }
    });
    legal.truncate(move_cap);
    // Restore canonical order so edges stay parallel to a sorted legal gather downstream.
    legal.sort_unstable();
    legal
}

fn push_node<S>(
    arena: &mut Vec<Node<S>>,
    state: S,
    obs: Vec<f32>,
    depth: i32,
    terminal: bool,
) -> usize {
    // Full-width modes have no expansion budget; every insertion checks the realized bound so a
    // single wide final ply cannot allocate past it.
    assert!(
        arena.len() < MAX_ENUMERATED_OUTCOMES,
        "search tree exceeds {} nodes; lower depth or set top_k",
        MAX_ENUMERATED_OUTCOMES
    );
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
        max_node: false,
        min_node: false,
        q_row: Vec::new(),
    });
    arena.len() - 1
}

fn resolve<S>(arena: &mut Vec<Node<S>>, idx: usize, gamma: f64, k: usize) {
    if arena[idx].terminal {
        return;
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
    let min_node = arena[idx].min_node;
    let init = if min_node {
        f64::INFINITY
    } else {
        f64::NEG_INFINITY
    };
    let mut value = vec![init; k];
    for edge in &edges {
        let ev = edge_value(arena, edge, gamma, k);
        for h in 0..k {
            value[h] = if min_node {
                value[h].min(ev[h])
            } else {
                value[h].max(ev[h])
            };
        }
    }
    arena[idx].value = value;
}

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

fn node_action_values<S, L>(
    arena: &[Node<S>],
    idx: usize,
    gamma: f64,
    k: usize,
    a: usize,
    legal_of: &L,
) -> Vec<Vec<f64>>
where
    L: Fn(&S) -> Vec<usize>,
{
    // Edges follow legal-action order; scatter them into the full action vocabulary.
    let edges = match &arena[idx].edges {
        None => return vec![vec![0.0; a]; k],
        Some(_) => take_edges(arena, idx),
    };
    let legal = legal_of(&arena[idx].state);
    debug_assert_eq!(legal.len(), edges.len(), "edges parallel the legal set");
    let mut values = vec![vec![0.0; a]; k];
    for (slot, e) in edges.iter().enumerate() {
        let ev = edge_value(arena, e, gamma, k);
        let action = legal.get(slot).copied().unwrap_or(slot);
        for (h, row) in values.iter_mut().enumerate() {
            row[action] = ev[h];
        }
    }
    values
}

fn collect_interior_targets<S, L>(
    arena: &[Node<S>],
    idx: usize,
    gamma: f64,
    k: usize,
    a: usize,
    legal_of: &L,
    out: &mut Vec<InteriorTarget>,
) where
    L: Fn(&S) -> Vec<usize>,
{
    if arena[idx].terminal || arena[idx].edges.is_none() {
        return;
    }
    // Only searcher-choice nodes yield policy targets; opponent expectations are internal.
    // Beamed nodes emit nothing: a dense row would train unsearched actions toward zero.
    if arena[idx].max_node
        && arena[idx].edges.as_ref().unwrap().len() == legal_of(&arena[idx].state).len()
    {
        out.push((
            arena[idx].obs.clone(),
            node_action_values(arena, idx, gamma, k, a, legal_of),
        ));
    }
    let edges = take_edges(arena, idx);
    for edge in &edges {
        for &(_, _, child) in edge {
            collect_interior_targets(arena, child, gamma, k, a, legal_of, out);
        }
    }
}

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
    use crate::encoder::StateEncoder;
    use crate::game::{Actor, Game, Transition};
    use crate::reward::Reward;
    use std::cell::Cell;

    struct Line;
    impl Game for Line {
        type State = i32;
        type Event = f64;
        fn num_agents(&self) -> usize {
            1
        }
        fn action_count(&self) -> usize {
            2
        }
        fn actor(&self, _: &i32) -> Actor {
            Actor::Agent(0)
        }
        fn legal_actions(&self, _: &i32, agent: usize) -> Vec<usize> {
            if agent == 0 {
                vec![0, 1]
            } else {
                Vec::new()
            }
        }
        fn step(&self, pos: &i32, actions: &[usize]) -> Transition<i32, f64> {
            let a = actions[0] as i32;
            Transition {
                next_state: pos + a,
                events: vec![Some(a as f64)],
                terminal: false,
            }
        }
        fn initial_state(&self) -> i32 {
            0
        }
    }

    struct LineReward;
    impl Reward for LineReward {
        type Event = f64;
        fn step_reward(&self, event: &f64, _agent: usize) -> f64 {
            *event
        }
    }

    struct LineEnc;
    impl ActionView for LineEnc {}
    impl StateEncoder for LineEnc {
        type State = i32;
        fn encode(&self, pos: &i32, _: usize) -> Vec<f32> {
            vec![*pos as f32]
        }
        fn obs_shape(&self) -> (usize, usize, usize) {
            (1, 1, 1)
        }
    }

    fn cfg(expansion_budget: usize) -> SearchConfig {
        SearchConfig {
            gamma: 0.9,
            beta: 1.0,
            expansion_budget,
            top_k: 2,
            max_depth: 4,
            chance: ChanceMode::Committed { samples: 1 },
            opponent: Opponent::Uniform,
        }
    }

    fn counting_infer(
        calls: &Cell<usize>,
    ) -> impl FnMut(&[usize], Vec<f32>, usize) -> Vec<f64> + '_ {
        move |_players: &[usize], obs: Vec<f32>, n: usize| {
            calls.set(calls.get() + 1);
            let dim = obs.len() / n;
            let mut out = Vec::with_capacity(n * 2 * 2);
            for i in 0..n {
                let p = obs[i * dim] as f64;
                out.extend_from_slice(&[
                    (p * 0.5).sin(),
                    (p * 0.3).cos(),
                    (p * 0.2).sin(),
                    (p * 0.7).cos(),
                ]);
            }
            out
        }
    }

    #[test]
    fn pooled_search_matches_solo_and_issues_fewer_forwards() {
        let states = [0i32, 5, 10];
        let pooled_calls = Cell::new(0);
        let pooled = search_many(
            &Line,
            &LineEnc,
            &LineReward,
            &cfg(8),
            states.iter().map(|&s| (s, 0)).collect(),
            false,
            0,
            counting_infer(&pooled_calls),
        );

        let mut solo_calls = 0usize;
        for (i, &s) in states.iter().enumerate() {
            let c = Cell::new(0);
            let solo = search_many(
                &Line,
                &LineEnc,
                &LineReward,
                &cfg(8),
                vec![(s, 0)],
                false,
                0,
                counting_infer(&c),
            );
            assert_eq!(
                pooled[i].0, solo[0].0,
                "pooled values must match solo for state {s}"
            );
            let (p, q) = (pooled[i].2, solo[0].2);
            assert_eq!(
                (p.max_depth, p.expansions, p.leaves, p.rounds),
                (q.max_depth, q.expansions, q.leaves, q.rounds),
                "pooled stats must match solo for state {s}"
            );
            solo_calls += c.get();
        }
        assert!(
            pooled_calls.get() < solo_calls,
            "pooling should issue fewer forwards: {} vs {}",
            pooled_calls.get(),
            solo_calls
        );
    }

    #[test]
    fn backed_up_root_values_match_a_hand_computed_tree() {
        let infer = |_players: &[usize], obs: Vec<f32>, n: usize| {
            let dim = obs.len() / n;
            let mut out = Vec::with_capacity(n * 2);
            for i in 0..n {
                let p = obs[i * dim] as f64;
                out.extend_from_slice(&[p * 10.0, p * 10.0 + 1.0]);
            }
            out
        };
        let results = search_many(
            &Line,
            &LineEnc,
            &LineReward,
            &cfg(1),
            vec![(0, 0)],
            false,
            0,
            infer,
        );
        let values = &results[0].0;
        assert_eq!(values.len(), 1);
        assert!((values[0][0] - 0.9).abs() < 1e-9, "{values:?}");
        assert!((values[0][1] - 10.9).abs() < 1e-9, "{values:?}");
    }
}

#[cfg(test)]
mod chance_mode_tests {
    use super::*;
    use crate::encoder::StateEncoder;
    use crate::game::{Actor, ChanceDist, Game, Transition};
    use crate::policies::tree::expectimax::SelectiveExpectimax;
    use crate::reward::Reward as RewardTrait;

    #[derive(Clone)]
    struct St {
        total: i32,
        ply: u8,
    }
    struct Risky;
    impl Game for Risky {
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
        fn legal_actions(&self, s: &St, agent: usize) -> Vec<usize> {
            if agent == 0 && s.ply != 2 {
                vec![0, 1]
            } else {
                Vec::new()
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
        fn encode(&self, s: &St, _: usize) -> Vec<f32> {
            vec![s.total as f32, f32::from(s.ply)]
        }
        fn obs_shape(&self) -> (usize, usize, usize) {
            (1, 1, 2)
        }
    }
    struct Passthrough;
    impl RewardTrait for Passthrough {
        type Event = f64;
        fn step_reward(&self, e: &f64, _: usize) -> f64 {
            *e
        }
    }

    fn run(chance: ChanceMode, seed: u64) -> Vec<Vec<f64>> {
        let cfg = SearchConfig {
            gamma: 1.0,
            beta: 1.0,
            expansion_budget: 16,
            top_k: 4,
            max_depth: 4,
            chance,
            opponent: Opponent::Uniform,
        };
        let mut infer = |_players: &[usize], _obs: Vec<f32>, n: usize| vec![0.0; n * 2];
        search_many(
            &Risky,
            &Enc,
            &Passthrough,
            &cfg,
            vec![(St { total: 0, ply: 0 }, 0)],
            false,
            seed,
            &mut infer,
        )
        .remove(0)
        .0
    }

    #[test]
    fn expand_all_is_the_exact_expectation() {
        let values = run(ChanceMode::ExpandAll, 7);
        assert_eq!(
            values[0][1], 1.5,
            "risky = 0.5*0 + 0.5*3 exactly, no sampling"
        );
        assert_eq!(values[0][0], 1.0);
    }

    #[test]
    fn committed_one_freezes_one_world() {
        let (mut lo, mut hi) = (false, false);
        for seed in 0..12 {
            let q = run(ChanceMode::Committed { samples: 1 }, seed)[0][1];
            assert!(
                q == 0.0 || q == 3.0,
                "one frozen world: q in {{0, 3}}, got {q}"
            );
            lo |= q == 0.0;
            hi |= q == 3.0;
        }
        assert!(lo && hi, "both worlds should occur across seeds");
    }

    #[test]
    #[should_panic(expected = "per-traversal chance modes")]
    fn selective_expectimax_rejects_always_resample() {
        let cfg = SearchConfig {
            gamma: 1.0,
            beta: 1.0,
            expansion_budget: 8,
            top_k: 2,
            max_depth: 4,
            chance: ChanceMode::AlwaysResample,
            opponent: Opponent::Uniform,
        };
        let _ = SelectiveExpectimax::new(cfg, 1, 0.0);
    }
}

#[cfg(test)]
mod frame_tests {
    use super::*;
    use crate::encoder::{ActionView, StateEncoder};
    use crate::game::{Actor, Game, Transition};
    use crate::reward::Reward as RewardTrait;

    #[derive(Clone)]
    struct St {
        id: i32,
        turn: usize,
    }
    struct Duel;
    impl Game for Duel {
        type State = St;
        type Event = f64;
        fn num_agents(&self) -> usize {
            2
        }
        fn action_count(&self) -> usize {
            2
        }
        fn actor(&self, s: &St) -> Actor {
            Actor::Agent(s.turn)
        }
        fn legal_actions(&self, s: &St, agent: usize) -> Vec<usize> {
            if agent == s.turn {
                vec![0, 1]
            } else {
                Vec::new()
            }
        }
        fn step(&self, s: &St, actions: &[usize]) -> Transition<St, f64> {
            Transition {
                next_state: St {
                    id: s.id * 2 + actions[s.turn] as i32 + 1,
                    turn: 1 - s.turn,
                },
                events: vec![Some(0.0), Some(0.0)],
                terminal: false,
            }
        }
        fn initial_state(&self) -> St {
            St { id: 0, turn: 0 }
        }
    }
    struct Zero;
    impl RewardTrait for Zero {
        type Event = f64;
        fn step_reward(&self, _: &f64, _: usize) -> f64 {
            0.0
        }
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
    struct IdEnc;
    impl ActionView for IdEnc {}
    macro_rules! enc_obs {
        ($t:ty) => {
            impl StateEncoder for $t {
                type State = St;
                fn encode(&self, s: &St, agent: usize) -> Vec<f32> {
                    vec![s.id as f32, agent as f32]
                }
                fn obs_shape(&self) -> (usize, usize, usize) {
                    (1, 1, 2)
                }
            }
        };
    }
    enc_obs!(IdEnc);
    enc_obs!(SwapFor1);

    fn v(id: f64, a: usize, h: usize, g: usize) -> f64 {
        ((id as i64 * 3 + a as i64 * 7 + h as i64 * 5 + g as i64 * 2) % 11) as f64 * 0.1
    }

    #[test]
    fn leaf_and_opponent_gathers_cross_via_each_rows_view() {
        let cfg = SearchConfig {
            gamma: 0.9,
            beta: 1.0,
            expansion_budget: 5,
            top_k: 2,
            max_depth: 3,
            chance: ChanceMode::Committed { samples: 1 },
            opponent: Opponent::Distributional {
                temperature: 1.0,
                floor: 0.1,
            },
        };
        let run = |swapped: bool| {
            let infer = move |_players: &[usize], obs: Vec<f32>, n: usize| -> Vec<f64> {
                (0..n)
                    .flat_map(|i| {
                        let id = f64::from(obs[i * 2]);
                        let g = obs[i * 2 + 1] as usize;
                        (0..2).flat_map(move |h| {
                            (0..2).map(move |slot| {
                                let game_a = if swapped && g == 1 { 1 - slot } else { slot };
                                v(id, game_a, h, g)
                            })
                        })
                    })
                    .collect()
            };
            let requests = vec![(St { id: 0, turn: 0 }, 0)];
            if swapped {
                search_many(&Duel, &SwapFor1, &Zero, &cfg, requests, true, 0, infer).remove(0)
            } else {
                search_many(&Duel, &IdEnc, &Zero, &cfg, requests, true, 0, infer).remove(0)
            }
        };
        let id = run(false);
        let sw = run(true);
        assert_eq!(id.0, sw.0, "root values: a gather skipped its row's view");
        assert_eq!(
            id.1, sw.1,
            "interior targets are game-frame search products: must be frame-invariant"
        );
    }
}

#[cfg(test)]
mod n_player_tests {
    use super::*;
    use crate::encoder::StateEncoder;
    use crate::game::Transition;
    use crate::reward::Reward as RewardTrait;

    struct Payout;
    impl RewardTrait for Payout {
        type Event = f64;
        fn step_reward(&self, e: &f64, _agent: usize) -> f64 {
            *e
        }
    }

    fn cfg(opponent: Opponent) -> SearchConfig {
        SearchConfig {
            gamma: 1.0,
            beta: 1.0,
            expansion_budget: 16,
            top_k: 8,
            max_depth: 8,
            chance: ChanceMode::Committed { samples: 1 },
            opponent,
        }
    }

    #[derive(Clone)]
    struct SimSt(bool);
    struct SimSum;
    impl Game for SimSum {
        type State = SimSt;
        type Event = f64;
        fn num_agents(&self) -> usize {
            3
        }
        fn action_count(&self) -> usize {
            2
        }
        fn actor(&self, _s: &SimSt) -> Actor {
            Actor::Simultaneous
        }
        fn legal_actions(&self, s: &SimSt, _agent: usize) -> Vec<usize> {
            if s.0 {
                Vec::new()
            } else {
                vec![0, 1]
            }
        }
        fn step(&self, _s: &SimSt, actions: &[usize]) -> Transition<SimSt, f64> {
            let others: f64 = actions[1] as f64 + actions[2] as f64;
            let mine = if actions[0] == 1 { others } else { 0.0 };
            Transition {
                next_state: SimSt(true),
                events: vec![Some(mine), Some(0.0), Some(0.0)],
                terminal: true,
            }
        }
        fn initial_state(&self) -> SimSt {
            SimSt(false)
        }
    }

    struct SimEnc;
    impl ActionView for SimEnc {}
    impl StateEncoder for SimEnc {
        type State = SimSt;
        fn encode(&self, s: &SimSt, agent: usize) -> Vec<f32> {
            vec![f32::from(u8::from(s.0)), agent as f32]
        }
        fn obs_shape(&self) -> (usize, usize, usize) {
            (1, 1, 2)
        }
    }

    #[test]
    fn factored_uniform_co_movers_average_exactly() {
        let results = search_many(
            &SimSum,
            &SimEnc,
            &Payout,
            &cfg(Opponent::Uniform),
            vec![(SimSt(false), 0)],
            false,
            0,
            |_players: &[usize], _obs: Vec<f32>, n: usize| vec![0.0; n * 2],
        );
        let values = &results[0].0;
        for head in values {
            assert!(
                (head[1] - 1.0).abs() < 1e-12,
                "E[sum of 2 uniform bits] = 1"
            );
            assert_eq!(head[0], 0.0);
        }
    }

    #[derive(Clone)]
    struct ChainSt {
        phase: usize,
        a1: usize,
    }
    struct Chain;
    impl Game for Chain {
        type State = ChainSt;
        type Event = f64;
        fn num_agents(&self) -> usize {
            3
        }
        fn action_count(&self) -> usize {
            2
        }
        fn actor(&self, s: &ChainSt) -> Actor {
            Actor::Agent(s.phase.min(2))
        }
        fn legal_actions(&self, s: &ChainSt, agent: usize) -> Vec<usize> {
            if s.phase < 3 && agent == s.phase {
                vec![0, 1]
            } else {
                Vec::new()
            }
        }
        fn step(&self, s: &ChainSt, actions: &[usize]) -> Transition<ChainSt, f64> {
            match s.phase {
                0 if actions[0] == 1 => Transition {
                    next_state: ChainSt { phase: 3, a1: 0 },
                    events: vec![Some(1.0), Some(0.0), Some(0.0)],
                    terminal: true,
                },
                0 => Transition {
                    next_state: ChainSt { phase: 1, a1: 0 },
                    events: vec![Some(0.0); 3],
                    terminal: false,
                },
                1 => Transition {
                    next_state: ChainSt {
                        phase: 2,
                        a1: actions[1],
                    },
                    events: vec![Some(0.0); 3],
                    terminal: false,
                },
                2 => {
                    let mine =
                        2.0 * f64::from(u8::from(s.a1 == 0)) + f64::from(u8::from(actions[2] == 0));
                    Transition {
                        next_state: ChainSt { phase: 3, a1: s.a1 },
                        events: vec![Some(mine), Some(0.0), Some(0.0)],
                        terminal: true,
                    }
                }
                _ => unreachable!("stepping a terminal state"),
            }
        }
        fn initial_state(&self) -> ChainSt {
            ChainSt { phase: 0, a1: 0 }
        }
    }

    struct ChainEnc;
    impl ActionView for ChainEnc {
        fn head_index(&self, action: usize, agent: usize) -> usize {
            (action + agent) % 2
        }
        fn game_action(&self, head: usize, agent: usize) -> usize {
            (head + agent) % 2
        }
    }
    impl StateEncoder for ChainEnc {
        type State = ChainSt;
        fn encode(&self, s: &ChainSt, agent: usize) -> Vec<f32> {
            vec![s.phase as f32, agent as f32, s.a1 as f32]
        }
        fn obs_shape(&self) -> (usize, usize, usize) {
            (1, 1, 3)
        }
    }

    #[test]
    fn sequential_uniform_co_movers_average_exactly() {
        let results = search_many(
            &Chain,
            &ChainEnc,
            &Payout,
            &cfg(Opponent::Uniform),
            vec![(ChainSt { phase: 0, a1: 0 }, 0)],
            false,
            0,
            |_players: &[usize], _obs: Vec<f32>, n: usize| vec![0.0; n * 2],
        );
        let values = &results[0].0;
        for head in values {
            assert!(
                (head[0] - 1.5).abs() < 1e-12,
                "L = E[2·b1 + b2] = 1.5: {head:?}"
            );
            assert!((head[1] - 1.0).abs() < 1e-12, "R pays exactly 1");
        }
    }

    #[test]
    fn distributional_weights_come_from_each_movers_own_row_and_view() {
        let (temperature, floor) = (1.0, 0.05);
        let results = search_many(
            &Chain,
            &ChainEnc,
            &Payout,
            &cfg(Opponent::Distributional { temperature, floor }),
            vec![(ChainSt { phase: 0, a1: 0 }, 0)],
            false,
            0,
            |_players: &[usize], obs: Vec<f32>, n: usize| {
                let mut out = Vec::with_capacity(n * 2);
                for r in 0..n {
                    let agent = obs[r * 3 + 1];
                    if agent == 0.0 {
                        out.extend([0.0, 0.0]);
                    } else {
                        out.extend([0.0, 10.0]);
                    }
                }
                out
            },
        );
        let ph = softmax_floor(&[10.0, 0.0], temperature, floor)[0];
        let expect_l = 2.0 * ph + (1.0 - ph);
        let values = &results[0].0;
        for head in values {
            assert!(
                (head[0] - expect_l).abs() < 1e-9,
                "L must mix each mover's own distribution: got {}, want {expect_l}",
                head[0]
            );
            assert!((head[1] - 1.0).abs() < 1e-12);
        }
    }
}

#[cfg(test)]
mod joint_fan_bound_tests {
    use super::*;
    use crate::encoder::StateEncoder;
    use crate::game::Transition;
    use crate::reward::Reward as RewardTrait;

    #[derive(Clone)]
    struct WSt(bool);
    struct Wide;
    impl Game for Wide {
        type State = WSt;
        type Event = f64;
        fn num_agents(&self) -> usize {
            65
        }
        fn action_count(&self) -> usize {
            2
        }
        fn actor(&self, _s: &WSt) -> Actor {
            Actor::Simultaneous
        }
        fn legal_actions(&self, s: &WSt, _agent: usize) -> Vec<usize> {
            if s.0 {
                Vec::new()
            } else {
                vec![0, 1]
            }
        }
        fn step(&self, _s: &WSt, _actions: &[usize]) -> Transition<WSt, f64> {
            Transition {
                next_state: WSt(true),
                events: vec![Some(0.0); 65],
                terminal: true,
            }
        }
        fn initial_state(&self) -> WSt {
            WSt(false)
        }
    }

    struct WEnc;
    impl ActionView for WEnc {}
    impl StateEncoder for WEnc {
        type State = WSt;
        fn encode(&self, s: &WSt, agent: usize) -> Vec<f32> {
            vec![f32::from(u8::from(s.0)), agent as f32]
        }
        fn obs_shape(&self) -> (usize, usize, usize) {
            (1, 1, 2)
        }
    }

    struct Zero;
    impl RewardTrait for Zero {
        type Event = f64;
        fn step_reward(&self, e: &f64, _agent: usize) -> f64 {
            *e
        }
    }

    #[test]
    #[should_panic(expected = "joint branches")]
    fn oversized_co_mover_fans_panic_instead_of_wrapping() {
        let cfg = SearchConfig {
            gamma: 1.0,
            beta: 1.0,
            expansion_budget: 1,
            top_k: 1,
            max_depth: 2,
            chance: ChanceMode::Committed { samples: 1 },
            opponent: Opponent::Uniform,
        };
        let _ = search_many(
            &Wide,
            &WEnc,
            &Zero,
            &cfg,
            vec![(WSt(false), 0)],
            false,
            0,
            |_players: &[usize], _obs: Vec<f32>, n: usize| vec![0.0; n * 2],
        );
    }
}
