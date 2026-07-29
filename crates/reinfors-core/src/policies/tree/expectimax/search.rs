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
//! successors come from `Game::step` (+ the declared chance). The public [`search_many`] +
//! [`SearchConfig`] are game-agnostic; concrete games (e.g. the snake `selective_search` wrappers in
//! reinfors-games) build a `SearchConfig` and a `Game` and call [`search_many`].
//!
//! Chance comes from the game's *declared* distribution (`Game::chance_outcomes` +
//! `apply_chance` — the game's only chance seam; the env realizes from the same declaration),
//! fanned per the configured [`ChanceMode`]: `Committed{k}` draws k
//! equal-weight realizations (the historical `food_samples` estimator), `ExpandAll` fans every
//! outcome at its true probability (exact). A deterministic transition keeps a single child. Each
//! search owns a seeded RNG, so results are reproducible from a seed.

use rayon::prelude::*;

use crate::encoder::{ActionView, StateEncoder};
use crate::game::{Actor, Game, Rng};
use crate::policy::ChanceMode;
use crate::reward::Reward;
use crate::rng::SplitMix64;

/// The agent's belief about each co-mover's move distribution (applied per co-mover — at N>2
/// every non-searching agent gets an independent copy of this model; a simultaneous joint is
/// their factored product).
#[derive(Clone, Copy)]
pub enum Opponent {
    /// Equal weight on each of the mover's actions (no net dependency).
    Uniform,
    /// Softmax over that mover's OWN (head-mean) Q — evaluated from its own observation, which
    /// rides the same batched forward as the leaves — with a uniform floor:
    /// `p = (1-floor)*softmax(q/temp) + floor/n`.
    Distributional { temperature: f64, floor: f64 },
}

/// Game-agnostic search knobs. A concrete game's wrapper (e.g. snake's `SearchParams`) builds one of
/// these directly from its public fields and passes it to [`search_many`].
#[derive(Clone, Copy)]
pub struct SearchConfig {
    pub gamma: f64,
    pub beta: f64,
    pub expansion_budget: usize,
    pub top_k: usize,
    pub max_depth: i32,
    /// How stochastic transitions' declared chance (`Game::chance_outcomes`) fans into branches
    /// (see [`ChanceMode`](crate::ChanceMode)): `Committed{k}` draws k realizations at equal weight
    /// (the historical `food_samples` estimator), `ExpandAll` fans every outcome at its true
    /// probability — the exact exhaustive-expectimax treatment. `AlwaysResample` is rejected at
    /// construction: an expand-once search has no traversal event to redraw on. Inert for
    /// deterministic transitions.
    pub chance: ChanceMode,
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
    /// Tree-search sim fates (0 for the expectimax family). Every simulation ends in exactly one
    /// bucket, counted by the tree at the moment it resolves — a fresh forwarded row, an infer-cache
    /// hit, a within-batch shared (deduped) row, an in-tree terminal, or a depth cap — so the
    /// per-collect identity `decisions × num_simulations =
    ///   fresh_rows + hit_rows + shared_rows + terminal_sims + depthcap_sims`
    /// is exact by construction, assembled from these search-local counters alone (the Evaluator's
    /// global row counts play no part in it, so non-search forwards can never unbalance it).
    pub terminal_sims: usize,
    pub depthcap_sims: usize,
    pub shared_rows: usize,
    pub fresh_rows: usize,
    pub hit_rows: usize,
    /// Rows an `ExpandAll` chance fan consumed *beyond* the one its simulation already accounts
    /// for (fan width − 1 per expanded chance edge; 0 in the other chance modes), so the identity
    /// above generalizes by subtracting this term from the row-bucket side.
    pub extra_eval_rows: usize,
}

/// An interior MAX node's TreeStrap target: its observation and per-head backed-up action values
/// `[K][A]`. Collected (when requested) for every expanded non-terminal node below the root.
pub type InteriorTarget = (Vec<f32>, Vec<Vec<f64>>);

/// Per-request search output: root per-head action values `[K][A]`, interior TreeStrap targets (empty
/// unless `collect_interior`), and the search diagnostics.
pub type SearchResult = (Vec<Vec<f64>>, Vec<InteriorTarget>, SearchStats);

/// A committed agent action's chance branch. `weight` is the resolved chance probability; for
/// deferred (distributional) co-movers it is filled in during evaluation as the product of every
/// `(co-mover obs index this round, co-mover action index)` factor in `deferred` — one factor per
/// distributional co-mover on the branch (a factored joint). `scale` (the branch's chance
/// probability times any fixed move factors) multiplies the deferred product at resolution; fully
/// fixed weights are pre-scaled at construction (`deferred` empty).
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
    obs: Vec<f32>, // leaf observation (empty for terminal nodes)
    depth: i32,
    terminal: bool,
    edges: Option<Vec<Edge>>, // None => unexpanded frontier leaf
    bootstrap: Vec<f64>,      // per-head net leaf value (empty until evaluated)
    value: Vec<f64>,          // per-head backed-up value (empty at terminals; treated as 0)
    sigma: f64,               // std over heads of the bootstrap — the VOI signal
    path_weight: f64,
    max_node: bool, // a searching-agent decision node (vs an opponent/chance node)
}

/// What a node's move branching produced, before evaluation resolves deferred weights.
#[derive(Clone, Copy)]
enum BranchWeight {
    Fixed(f64),
    Deferred(usize, usize),
}

/// One (joint) co-mover assignment's combined move probability: the product of its fixed factors,
/// and the deferred (distributional) factors still awaiting the pooled forward. A single co-mover
/// reduces to exactly the old `BranchWeight` semantics; N co-movers compose factored-independent.
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

/// Per-request search state, advanced one round at a time so several searches can run in lockstep
/// and pool their per-round observations into a single `infer` call.
struct Search<S> {
    arena: Vec<Node<S>>,
    frontier: Vec<usize>,
    agent: usize,
    n_heads: usize,
    stats: SearchStats,
    batch: Vec<usize>,
    opp_obs: Vec<Vec<f32>>,
    // (mover, its legal set) for each registered co-mover obs — the mover tags which agent's
    // perspective the row was encoded for (its gathers cross the view through that agent).
    opp_legal: Vec<(usize, Vec<usize>)>,
    new_leaves: Vec<usize>,
    rng: SplitMix64, // this search's chance-sampling stream (apple respawns), seeded per request
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
            max_node: false, // set when the node is expanded (the root is always a MAX node)
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

/// One round's build phase for a single search: sort the frontier (after the root round), expand its
/// top-k nodes one ply each, and stage the new leaves + opponent observations for the pooled forward.
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
    // `infer(obs_flat, n_rows) -> values_flat`: obs is one contiguous row-major `[n_rows, dim]` buffer
    // (moved in, so the binding hands it to numpy with no copy); values come back as one contiguous
    // row-major `[n_rows, K, A]` buffer (K inferred from its length). Flat on both sides avoids the
    // per-row obs clones and the per-leaf nested-`Vec` allocations the boundary would otherwise incur.
    F: FnMut(Vec<f32>, usize) -> Vec<f64>,
    G::State: Send,
{
    assert!(
        game.num_agents() >= 1,
        "a game must have at least one agent"
    );
    assert!(
        game.perfect_information(),
        "tree search on a hidden-information game is clairvoyant: its values condition on state \
         the agents cannot observe; use an observation-only policy family"
    );
    assert!(
        !game.chance_nodes(),
        "tree search does not realize chance-node states (outcome-dependent payouts)"
    );
    let a = game.action_count();
    // Each search gets its own chance-sampling stream, seeded deterministically from the request
    // index, so results are reproducible and independent of the parallel-vs-serial expansion schedule.
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
                    // The leaf-bootstrap legality convention: the MOVER'S legal set — the actions
                    // that exist at the state. At the searching agent's own decisions that is its
                    // legal set exactly; at sequential opponent-to-move leaves it is the
                    // opponent's (the agent's Q row masked to the moves actually available there —
                    // the Q-family's heuristic for foreign-turn values, made legality-consistent).
                    let agent = s.agent;
                    let legal_of = |state: &G::State| match game.actor(state) {
                        Actor::Agent(mover) => game.legal_actions(state, mover),
                        Actor::Simultaneous => game.legal_actions(state, agent),
                        Actor::Chance => unreachable!("chance actors are not searched"),
                    };
                    evaluate(
                        &mut s.arena,
                        &s.batch,
                        &s.new_leaves,
                        &s.opp_legal,
                        enc,
                        s.agent,
                        &legal_of,
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
            // MAX nodes are the searching agent's decisions, so their legal sets come from its
            // perspective (the closure densifies value rows back to the full action space).
            let agent = s.agent;
            let legal_of = |state: &G::State| game.legal_actions(state, agent);
            let values = node_action_values(&s.arena, 0, cfg.gamma, s.n_heads, a, &legal_of);
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

/// Resolve a round's batched forward: opponent rows -> chance weights, leaf rows -> per-head
/// bootstrap + sigma, then write branch weights and child path-weights. `q` is this search's flat
/// row-major slice `[rows, k, A]`; row `r`'s `[k, A]` block is `q[r*k*A .. (r+1)*k*A]`.
#[allow(clippy::too_many_arguments)]
fn evaluate<S, L>(
    arena: &mut [Node<S>],
    batch: &[usize],
    new_leaves: &[usize],
    opp_legal: &[(usize, Vec<usize>)],
    // The encoder's action view + the perspectives the rows were encoded for: every leaf row is
    // the SEARCHER's observation, every co-mover row that mover's own (tagged per row in
    // `opp_legal` — sequential N-player interleaves rows from different movers). Gathers from raw
    // net rows cross into the head frame through the matching perspective.
    view: &dyn ActionView,
    searcher: usize,
    legal_of: &L,
    q: &[f64],
    n_opp: usize,
    k: usize,
    a: usize,
    cfg: &SearchConfig,
    stats: &mut SearchStats,
) where
    L: Fn(&S) -> Vec<usize>,
{
    let row = |r: usize| -> &[f64] { &q[r * k * a..(r + 1) * k * a] }; // [k, A], head-major

    // Opponent move probabilities from the head-mean Q, GATHERED to the mover's legal set before
    // the softmax (the PUCT-prior convention) and scattered back by action id — so the legal
    // branch weights sum to 1 and illegal logits cannot distort the distribution.
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
            Opponent::Uniform => Vec::new(), // uniform registers no opponent observations
        })
        .collect();

    for (j, &li) in new_leaves.iter().enumerate() {
        let leaf_q = row(n_opp + j); // [k, A]
                                     // Leaf bootstrap = per-head max over the LEGAL actions of the leaf state (the MOVER'S
                                     // legal set — the actions that exist there; see the `legal_of` convention at the call
                                     // site). A dense max would let a full column's phantom Q leak into every backed-up value
                                     // above this leaf, steering even legal root choices.
        let legal = legal_of(&arena[li].state);
        debug_assert!(!legal.is_empty(), "non-terminal leaf with no legal actions");
        let boot: Vec<f64> = (0..k)
            .map(|h| {
                let head = &leaf_q[h * a..(h + 1) * a];
                legal
                    .iter()
                    .map(|&aid| head[view.head_index(aid, searcher)])
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

/// A mover's believed move distribution at `state`: uniform, or distributional from its observed
/// head-mean Q (deferred via `opp_obs`, resolved in `evaluate`). An inactive mover (no legal actions)
/// contributes a single placeholder branch.
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
    }
}

/// Build the child node(s) for one `joint` action and append the resulting branch(es) to `branches`.
/// `bw` is the move's chance weight. `agent_out_terminal` decides whether "the searching agent has no
/// legal actions in the child" counts as terminal: true for simultaneous play (it means the agent
/// died), false for a sequential turn (it just means it is not the agent's move next). A stochastic
/// transition fans per the configured `ChanceMode` over the game's declared distribution — the
/// same declaration the env realizes from, so search and env cannot diverge.
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
    agent_out_terminal: bool,
    depth: i32,
    new_leaves: &mut Vec<usize>,
    branches: &mut Vec<Branch>,
    rng: &mut dyn Rng,
) {
    let t = game.step(state, joint);
    let step_reward = reward.step_reward(&t.events[agent], agent);
    // Fan the chance children from the game's DECLARED distribution (`chance_outcomes` +
    // `apply_chance` — the same seam the tree searches consume), per the configured mode:
    // `Committed{k}` draws k outcome indices proportionally to the declared probabilities (equal
    // 1/k branch weights — the historical `food_samples` Monte-Carlo estimator); `ExpandAll` fans
    // every outcome at its true probability (exact). `None` = deterministic: one child.
    let children: Vec<(G::State, f64)> = match game.chance_outcomes(state, &t) {
        None => vec![(t.next_state, 1.0)],
        Some(dist) => match cfg.chance {
            ChanceMode::Committed { samples } => {
                let k = samples.max(1);
                (0..k)
                    .map(|_| {
                        let idx = dist.draw(rng);
                        (game.apply_chance(state, &t, idx), 1.0 / k as f64)
                    })
                    .collect()
            }
            ChanceMode::ExpandAll => {
                // Exhaustive fan: exact, so an outcome space past the enumeration bound is an
                // error rather than an approximation — sample (Committed) instead.
                let count = dist.count();
                assert!(
                    count <= crate::policy::MAX_ENUMERATED_OUTCOMES,
                    "ExpandAll cannot enumerate {count} chance outcomes (bound {}); use a \
                     sampling chance mode for combinatorial outcome spaces",
                    crate::policy::MAX_ENUMERATED_OUTCOMES
                );
                dist.iter_probs()
                    .enumerate()
                    .map(|(idx, pr)| (game.apply_chance(state, &t, idx), pr))
                    .collect()
            }
            ChanceMode::AlwaysResample => unreachable!(
                "rejected at SelectiveExpectimax construction (no traversal event to redraw on)"
            ),
        },
    };
    // Each chance child is a distinct state, so its terminality and observation are computed per
    // child (the food position differs across outcomes).
    for (child_state, p) in children {
        // Fully fixed moves pre-scale into `weight`; any deferred factor leaves the weight to
        // resolution (scale carries the fixed part x the chance probability).
        let (weight, scale) = if mw.deferred.is_empty() {
            (mw.fixed * p, 1.0)
        } else {
            (0.0, mw.fixed * p)
        };
        let terminal = t.terminal
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
            reward: step_reward,
            child,
        });
    }
}

/// Expand `ni` one ply, dispatching on whose turn it is:
///  - `Simultaneous` (snake): the searching agent's MAX edges, each fanned over the FACTORED joint
///    distribution of every co-mover acting this ply (each co-mover an independent Uniform /
///    Distributional model; joint probability = product). A MAX node. Per-expansion branch width
///    is the product of the co-movers' legal counts — exact w.r.t. the factored model; a sampled
///    co-mover fan is a future knob if agent counts outgrow game-typical widths.
///  - `Agent(me)` (single-agent, or our turn in a sequential game): our MAX edges, one deterministic
///    branch each, no co-mover. A MAX node.
///  - `Agent(other)` (another agent's turn in a sequential game): a single chance edge over that
///    mover's modeled move distribution — at N>2 each other agent's turn is simply its own such
///    node. The searching agent does not choose here, so it is NOT a MAX node — it backs up as the
///    expectation over the mover's moves (MAX over its one edge) and is never collected as a
///    TreeStrap target.
///
/// `max_node` marks the searching agent's decision points — the only nodes whose `[K][A]` action
/// values are valid training targets.
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
            let co_movers: Vec<usize> = (0..num_agents).filter(|&i| i != agent).collect();
            let co_b: Vec<Vec<(usize, BranchWeight)>> = co_movers
                .iter()
                .map(|&m| agent_branching(game, enc, cfg, &state, m, opp_obs, opp_legal))
                .collect();
            // Checked: an unchecked product wraps (65 two-action co-movers -> 0 combos -> a
            // silently empty, zero-valued fan) — and exact expectimax has no sparse escape,
            // every joint transition is evaluated, so an oversized fan is an error (the binding
            // pre-checks the static worst case; a sampled co-mover mode is the eventual answer
            // for genuinely large fans).
            let combos: usize = co_b
                .iter()
                .try_fold(1usize, |acc, b| {
                    acc.checked_mul(b.len())
                        .filter(|&c| c <= crate::policy::MAX_JOINT_SLOTS)
                })
                .unwrap_or_else(|| {
                    panic!(
                        "simultaneous co-mover fan exceeds {} joint branches at one node",
                        crate::policy::MAX_JOINT_SLOTS
                    )
                });
            let agent_legal = game.legal_actions(&state, agent);
            let mut edges = Vec::with_capacity(agent_legal.len());
            for &agent_action in &agent_legal {
                let mut branches = Vec::with_capacity(combos);
                // Mixed-radix walk over the co-mover assignments (last co-mover fastest — the
                // single-co-mover order is unchanged), composing each joint's factored weight.
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
            let agent_legal = game.legal_actions(&state, agent);
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
        Actor::Agent(mover) => {
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
        Actor::Chance => unimplemented!("explicit Actor::Chance nodes are not yet supported"),
    };
    arena[ni].edges = Some(edges);
    arena[ni].max_node = max_node;
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
        max_node: false, // set if/when this leaf is later expanded
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
    // Densified back to the full action space: a node's edges run over its LEGAL actions (in
    // `legal_actions` order), so the per-edge values scatter into their action ids — illegal
    // columns carry 0, mirroring the tree searches' densification, and acting masks to the legal
    // set so the zeros are never selectable.
    let edges = match &arena[idx].edges {
        None => return vec![vec![0.0; a]; k],
        Some(_) => take_edges(arena, idx),
    };
    let legal = legal_of(&arena[idx].state);
    debug_assert_eq!(legal.len(), edges.len(), "edges parallel the legal set");
    let mut values = vec![vec![0.0; a]; k];
    for (slot, e) in edges.iter().enumerate() {
        let ev = edge_value(arena, e, gamma, k); // [K]
        let action = legal.get(slot).copied().unwrap_or(slot);
        for (h, row) in values.iter_mut().enumerate() {
            row[action] = ev[h];
        }
    }
    values
}

/// DFS-collect every expanded non-terminal **MAX** node at or below `idx` as `(obs, [K][A] values)` —
/// true TreeStrap data, valid only at the searching agent's decision points. Terminal and
/// unexpanded-frontier nodes are skipped (no backed-up values); opponent/chance nodes are recursed
/// through but not emitted (their `[K][A]` would be over the opponent's actions, not a training target).
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
    if arena[idx].max_node {
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
    use crate::encoder::StateEncoder;
    use crate::game::{Actor, Game, Rng, Transition};
    use crate::reward::Reward;
    use std::cell::Cell;

    // A 1-agent deterministic line walk: action 0 stays, 1 advances; the event is the action, and
    // `LineReward` turns it back into reward = action. Never terminal (so the frontier is always real
    // leaves). No chance node, so a search is seed-independent — which makes pooled-vs-solo bit-identity
    // trivial to assert and the backup hand-computable.
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
                events: vec![a as f64],
                terminal: false,
            }
        }
        fn initial_state(&self, _: &mut dyn Rng) -> i32 {
            0
        }
    }

    // Reward = the event (the action), so the search backs up reward = action as before.
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

    // Two-head infer, deterministic in the position; counts its calls so pooling can be observed.
    fn counting_infer(calls: &Cell<usize>) -> impl FnMut(Vec<f32>, usize) -> Vec<f64> + '_ {
        move |obs: Vec<f32>, n: usize| {
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
            out // K=2, A=2
        }
    }

    #[test]
    fn pooled_search_matches_solo_and_issues_fewer_forwards() {
        // Pooling batches each round's leaf evaluations across all active searches. It must not change
        // any search's result (the throughput optimisation is value-neutral) — only cut `infer` calls.
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
        // Budget 1 expands only the root; its two children are evaluated leaves. With K=1 and a known
        // Q, the backup is exactly `reward_a + gamma * max_a' Q(child_a)`. Root at pos 0:
        //   action 0 -> child pos 0, reward 0;  action 1 -> child pos 1, reward 1.
        //   Q(pos) = [pos*10, pos*10 + 1]  ->  max Q(0) = 1, max Q(1) = 11.
        //   root[0][0] = 0 + 0.9*1 = 0.9;   root[0][1] = 1 + 0.9*11 = 10.9.
        let infer = |obs: Vec<f32>, n: usize| {
            let dim = obs.len() / n;
            let mut out = Vec::with_capacity(n * 2);
            for i in 0..n {
                let p = obs[i * dim] as f64;
                out.extend_from_slice(&[p * 10.0, p * 10.0 + 1.0]); // K=1, A=2
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
        let values = &results[0].0; // [K=1][A=2]
        assert_eq!(values.len(), 1);
        assert!((values[0][0] - 0.9).abs() < 1e-9, "{values:?}");
        assert!((values[0][1] - 10.9).abs() < 1e-9, "{values:?}");
    }
}

#[cfg(test)]
mod chance_mode_tests {
    //! The declared-chance modes in the expectimax fan: `ExpandAll` = exact probability-weighted
    //! branches (the exhaustive treatment this search previously could not express), `Committed{k}`
    //! = the historical `food_samples` estimator over the same declared distribution. Game: one
    //! risky action whose value is pure expectation (outcomes +0/+3 at p=[.5,.5], E=1.5) vs a
    //! certain +1, rewards landing on the NEXT ply so everything flows through chance branches.
    use super::*;
    use crate::encoder::StateEncoder;
    use crate::game::{Actor, Game, Rng, Transition};
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
        fn actor(&self, _: &St) -> Actor {
            Actor::Agent(0)
        }
        fn legal_actions(&self, _: &St, agent: usize) -> Vec<usize> {
            if agent == 0 {
                vec![0, 1]
            } else {
                Vec::new()
            }
        }
        fn step(&self, s: &St, actions: &[usize]) -> Transition<St, f64> {
            if s.ply == 0 {
                let total = if actions[0] == 0 {
                    s.total + 1
                } else {
                    s.total
                };
                Transition {
                    next_state: St { total, ply: 1 },
                    events: vec![0.0],
                    terminal: false,
                }
            } else {
                Transition {
                    next_state: St { ..*s },
                    events: vec![f64::from(s.total)],
                    terminal: true,
                }
            }
        }
        fn chance_outcomes(
            &self,
            s: &St,
            t: &Transition<St, f64>,
        ) -> Option<crate::game::ChanceDist> {
            (s.ply == 0 && t.next_state.total == s.total)
                .then(|| crate::game::ChanceDist::Weighted(vec![0.5, 0.5]))
        }
        fn apply_chance(&self, _s: &St, t: &Transition<St, f64>, outcome: usize) -> St {
            St {
                total: t.next_state.total + if outcome == 0 { 0 } else { 3 },
                ply: t.next_state.ply,
            }
        }
        fn initial_state(&self, _: &mut dyn Rng) -> St {
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
        let mut infer = |_obs: Vec<f32>, n: usize| vec![0.0; n * 2]; // K=1 zeros
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
        let _ = crate::policies::tree::expectimax::SelectiveExpectimax::new(cfg, 1, 0.0);
    }
}

#[cfg(test)]
mod frame_tests {
    //! The two expectimax gathers — leaf bootstraps (searcher's frame) and distributional-opponent
    //! rows (opponent's frame) — must cross via each row's own encoding perspective. Same
    //! equivalence scheme as the mcts frame tests: identity encoder + game-frame rows vs an
    //! agent-dependent view + rows permuted into each agent's head frame; results must be equal.
    use super::*;
    use crate::encoder::{ActionView, StateEncoder};
    use crate::game::{Actor, Game, Rng, Transition};
    use crate::reward::Reward as RewardTrait;

    /// Two agents alternating by an explicit turn flag; searcher = agent 0, so leaf rows are
    /// encoded for agent 0 and (distributional) opponent rows for agent 1.
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
                events: vec![0.0, 0.0],
                terminal: false,
            }
        }
        fn initial_state(&self, _: &mut dyn Rng) -> St {
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

    /// Agent 0 keeps the identity; agent 1's head frame swaps the two actions.
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
                    vec![s.id as f32, agent as f32] // the encoding perspective rides the obs
                }
                fn obs_shape(&self) -> (usize, usize, usize) {
                    (1, 1, 2)
                }
            }
        };
    }
    enc_obs!(IdEnc);
    enc_obs!(SwapFor1);

    /// Deterministic game-frame value for (state id, game action, head, encode agent).
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
            let infer = move |obs: Vec<f32>, n: usize| -> Vec<f64> {
                (0..n)
                    .flat_map(|i| {
                        let id = f64::from(obs[i * 2]);
                        let g = obs[i * 2 + 1] as usize;
                        (0..2).flat_map(move |h| {
                            (0..2).map(move |slot| {
                                // identity: slot is the game action; swapped view: agent 1's
                                // slots are exchanged (agent 0 unchanged).
                                let game_a = if swapped && g == 1 { 1 - slot } else { slot };
                                v(id, game_a, h, g)
                            })
                        })
                    })
                    .collect() // K=2, A=2 rows
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
    //! N>2 co-mover modeling: exact values under factored Uniform co-movers (simultaneous and
    //! sequential), and the per-row mover perspective — each co-mover's Distributional weights
    //! come from ITS OWN net row, gathered through ITS OWN action view (the single-`opp` era
    //! would have gathered every foreign row through `1 - agent`).
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

    /// One-shot 3-agent simultaneous: my action 1 pays the SUM of the co-movers' actions, action 0
    /// pays nothing. Under uniform co-movers the exact root values are 1.0 and 0.0.
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
                events: vec![mine, 0.0, 0.0],
                terminal: true,
            }
        }
        fn initial_state(&self, _rng: &mut dyn Rng) -> SimSt {
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
            |_obs: Vec<f32>, n: usize| vec![0.0; n * 2],
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

    /// 3-agent sequential chain: I move (L=0 keeps playing, R=1 ends at exactly 1.0), then agent 1,
    /// then agent 2, and my terminal payoff is `2·[a1==0] + [a2==0]`.
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
                    events: vec![1.0, 0.0, 0.0],
                    terminal: true,
                },
                0 => Transition {
                    next_state: ChainSt { phase: 1, a1: 0 },
                    events: vec![0.0; 3],
                    terminal: false,
                },
                1 => Transition {
                    next_state: ChainSt {
                        phase: 2,
                        a1: actions[1],
                    },
                    events: vec![0.0; 3],
                    terminal: false,
                },
                2 => {
                    let mine =
                        2.0 * f64::from(u8::from(s.a1 == 0)) + f64::from(u8::from(actions[2] == 0));
                    Transition {
                        next_state: ChainSt { phase: 3, a1: s.a1 },
                        events: vec![mine, 0.0, 0.0],
                        terminal: true,
                    }
                }
                _ => unreachable!("stepping a terminal state"),
            }
        }
        fn initial_state(&self, _rng: &mut dyn Rng) -> ChainSt {
            ChainSt { phase: 0, a1: 0 }
        }
    }

    /// Per-agent action view: agent `g`'s head frame is the game frame rotated by `g` (mod 2) — a
    /// bijection per agent, so gathers from a co-mover's row MUST use that co-mover's view.
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
            |_obs: Vec<f32>, n: usize| vec![0.0; n * 2],
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
        // Both co-mover rows are the SAME bytes ([0, 10] in their own head frames), but their
        // views differ: agent 1's frame is rotated (so its row favors GAME action 0), agent 2's
        // is identity-parity (favoring GAME action 1). p1(0) = ph, p2(0) = 1 - ph, so
        // L = 2·ph + (1 - ph) — a single shared foreign perspective would compute 3·ph instead.
        let (temperature, floor) = (1.0, 0.05);
        let results = search_many(
            &Chain,
            &ChainEnc,
            &Payout,
            &cfg(Opponent::Distributional { temperature, floor }),
            vec![(ChainSt { phase: 0, a1: 0 }, 0)],
            false,
            0,
            |obs: Vec<f32>, n: usize| {
                let mut out = Vec::with_capacity(n * 2);
                for r in 0..n {
                    let agent = obs[r * 3 + 1];
                    if agent == 0.0 {
                        out.extend([0.0, 0.0]); // searcher leaf bootstraps: inert
                    } else {
                        out.extend([0.0, 10.0]); // co-mover Q in its OWN head frame
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

    /// 65 two-action agents: the unchecked co-mover product would wrap to 0 (a silently empty,
    /// zero-valued fan); the checked fold must panic at the bound instead.
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
                events: vec![0.0; 65],
                terminal: true,
            }
        }
        fn initial_state(&self, _rng: &mut dyn Rng) -> WSt {
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
            |_obs: Vec<f32>, n: usize| vec![0.0; n * 2],
        );
    }
}
