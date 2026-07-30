//! Monte-Carlo Tree Search: the shared arena tree + pooled simulation loop (`search_many`), guided
//! either by UCB1 (the `Mcts` policy here) or by net priors under PUCT (the `AlphaZero` policy in
//! `alphazero` — see [`Guidance`] for the exact axis of difference). The UCT policy produces the same
//! [`SearchEvaluation`] the `TreeStrap` learner consumes — the training target is the root's backed-up
//! per-action values `values[1][A]` ("MCTS-strap") — and pools its leaf evaluations across games into
//! one `infer` per round, exactly like the expectimax search.
//!
//! **Sequential, single-agent, and simultaneous games, at any agent count.** Sequential games
//! with ≤2 agents are treated as zero-sum (negamax backup — correct for connect4/chess; kept as
//! the measured-fast path until a 2p Max^N comparison justifies deleting it). Sequential games
//! with N>2 agents back up **Max^N**: per-agent value vectors propagate absolutely (general-sum,
//! no sign flips), each mover's edge keeps its own component for selection, and every leaf is
//! evaluated from all N perspectives (pooled rows) — which requires a value head, so N>2
//! sequential is PUCT-only (asserted). Simultaneous games use **decoupled statistics (DUCT)** at
//! any N: each node keeps one [`AgentTable`] per agent, each agent independently applies the
//! ordinary UCT/PUCT rule over its OWN table, the tuple steps the game as a joint action, and
//! backup is per-agent own-perspective. A game must be uniformly one dynamics or the other
//! (asserted); `Actor::Chance` states are traversed as fixed-probability chance nodes —
//! never UCB/PUCT arms, never net-evaluated, transparent to depth/discount/perspective.
//!
//! **Chance** comes from the game's *declared* [`Game::chance_node`] distributions (chains
//! descended lazily, or flattened under `ExpandAll`),
//! consumed per the configured [`ChanceMode`]: chance nodes sit between a
//! decision edge and its outcome children, transparent to backup (the decision edge carries the
//! reward and the discount). Games that declare no chance build byte-identical trees to before.
//!
//! **Acting.** Greedy by default (`argmax` value or visits) — deterministic, ideal for evaluation and
//! like-for-like benchmarks. For *training*, greedy self-play from a fixed start replays the same game
//! every episode, so an AlphaZero-style acting temperature supplies self-play diversity: with
//! `temperature > 0`, the first `temperature_drop` moves of each episode are sampled `∝
//! visits^(1/temperature)` from the engine's seeded acting RNG (later moves act greedily). Same seed →
//! same games (collects stay reproducible); different episodes → different games. Root Dirichlet noise
//! is deliberately absent from the UCT policy: it perturbs *priors*, a PUCT concept — for a
//! prior-guided, noise-capable search use the `AlphaZero` policy. The reached-state start buffer
//! remains the complementary coverage source.

use std::collections::HashMap;

use crate::codec::bytes::Reader;
use crate::encoder::{ActionView, StateEncoder};
use crate::game::{Actor, ChanceDist, Game, Rng, Transition};
use crate::policies::tree::expectimax::search::SearchStats;
use crate::policies::tree::expectimax::{decode_search_eval, encode_search_eval, SearchEvaluation};
use crate::policy::{argmax, Policy, SearchPolicy, MAX_ENUMERATED_OUTCOMES};
use crate::reward::Reward;
use crate::rng::{dirichlet, SplitMix64};
use crate::rollout::engine::CollectStats;
use crate::rollout::evaluator::{Evaluator, Resolve};

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
        // Simultaneous roots: noise every agent's prior, or only the requester's (see NoiseScope).
        noise_all: bool,
    },
}

/// Which backup scheme sequential games use (simultaneous games always run DUCT). `Auto` is the
/// production default: scalar negamax at <=2 agents (zero-sum exact, one value row per leaf),
/// Max^N past that. `MaxN` forces the Max^N vector backup at 2 agents too — the measurement
/// seam for the negamax-deletion decision (PUCT only: Max^N consumes per-perspective leaf
/// VALUES). Under a forced Max^N the engine also emits per-perspective value rows, keeping
/// supervised perspectives = consumed perspectives.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum SequentialBackup {
    #[default]
    Auto,
    MaxN,
}

/// Which root prior(s) receive Dirichlet exploration noise in a *simultaneous* search tree.
/// Sequential trees have one root table (the requester's), so the scope is irrelevant there.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum NoiseScope {
    /// Noise the requesting agent's root prior only: exploration for the agent whose pi target
    /// this tree produces, while the in-tree co-mover models keep the net's honest beliefs.
    #[default]
    Requester,
    /// Noise every agent's root prior: more joint-space exploration, at the cost of deliberately
    /// perturbed co-mover beliefs baked into the search.
    All,
}

use crate::policy::ChanceMode;

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
    /// How the search consumes the game's declared chance states (see [`ChanceMode`]).
    /// Inert for deterministic games.
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

/// One search tree node, sparse over the mover's LEGAL actions: `actions` holds the legal action ids
/// (from `Game::legal_actions`), and `child`/`reward`/`visits`/`value_sum`/`prior` are parallel to it.
/// For the always-fully-legal games (snake, connect4, gridworld) `actions` is `0..A` and behavior is
/// bit-identical to the former dense layout; for wide, mostly-illegal action spaces (chess: 4672 ids,
/// ~35 legal) nodes stay ~legal-sized and selection scans only real moves. Priors and root Dirichlet
/// noise likewise live on the legal set only (the AlphaZero convention).
struct Node<S> {
    state: S,
    actor: usize,
    depth: i32,
    terminal: bool,
    kind: NodeKind,
    actions: Vec<usize>, // [L] the mover's legal action ids at this node
    child: Vec<i64>,     // [L] child arena index, -1 if the edge is unexpanded
    reward: Vec<f64>, // [L] immediate reward for the mover taking this action (its own perspective)
    visits: Vec<u32>, // [L] edge visit counts
    value_sum: Vec<f64>, // [L] summed backed-up value (mover's perspective)
    total_visits: u32,
    value: f64, // this node's state value (net leaf eval, or 0 at a terminal) — the backprop source
    obs: Vec<f32>, // staged observation for the pending net eval (empty for terminals)
    prior: Vec<f64>, // [L] PUCT prior (softmaxed net logits over the legal set); empty under UCT, and empty = not yet evaluated under PUCT
    // Max^N (sequential N>2) extras, empty in the other modes: every agent's staged observation
    // and per-perspective net value ([N]; the mover's row also supplies `prior`), and per-agent
    // edge rewards ([L·N] row-major) — general-sum games pay non-movers on a mover's edge.
    obs_all: Vec<Vec<f32>>,
    values_all: Vec<f64>,
    rewards_all: Vec<f64>,
    // Per-agent rewards emitted on the CHANCE edge that produced this node (empty = the edge
    // settled nothing, the common case). Backups fold these into the tick above, undiscounted.
    chance_in: Vec<f64>,
}

/// A node is a decision (an agent to move — the layout above), a **chance node** sitting between a
/// decision edge and its outcome children, or a **simultaneous node** where both agents move at
/// once (decoupled DUCT statistics — see [`SimNode`]). Chance nodes reuse `child` for their outcome
/// children (`-1` = not yet materialized) and are transparent to backup: no reward, no discount, no
/// stats — the action edge above them carries all of those. Simultaneous nodes keep everything in
/// their `SimNode`; the flat `Node` arrays are unused.
enum NodeKind {
    Decision,
    Chance {
        /// The declared outcome distribution (`Game::chance_node` at this `Actor::Chance`
        /// state) — outcomes are full transitions that may emit events, end the game, or land
        /// on another chance state.
        dist: ChanceDist,
        /// `Committed` mode: the frozen outcome draws (with replacement; `child` is parallel to
        /// this, so duplicated draws keep separate equal-weight branches, like `food_samples`).
        /// Empty in the other modes. Under `ExpandAll`, `child` is parallel to the (bounded)
        /// outcome space instead.
        committed: Vec<usize>,
        /// `AlwaysResample`: sparse outcome -> child map — the outcome space can be
        /// combinatorial (`Uniform(count)`), so children materialize per DISTINCT drawn outcome
        /// (at most one per simulation; a map keeps S simulations O(S), where a scanned vec
        /// would be O(S^2) since combinatorial draws are almost always distinct).
        /// `node.child` is empty in this mode, which is how the descent recognizes it.
        resampled: HashMap<usize, usize>,
        /// Non-empty ONLY for an `ExpandAll` fan whose chance CHAIN was flattened: `child` then
        /// parallels these compound leaf probabilities instead of the node's own `dist` (descent
        /// draws ∝ these; `fan_backprop` mixes by them). Empty = children parallel the dist.
        fan_weights: Vec<f64>,
    },
    Simultaneous(Box<SimNode>),
}

/// One agent's decoupled statistics at a simultaneous node — the same sparse legal-set layout as a
/// decision node's flat arrays, one copy per agent. Each agent selects over its OWN table with the
/// ordinary UCT/PUCT rule (Decoupled UCT): the opponent inside this tree is *searched*, not modeled
/// as a fixed distribution the way expectimax's `Opponent` is — its statistics sharpen as
/// simulations accumulate.
struct AgentTable {
    actions: Vec<usize>, // [L] this agent's legal action ids here
    visits: Vec<u32>,
    value_sum: Vec<f64>, // own-perspective backed-up values (no negamax — simultaneous ≠ zero-sum)
    prior: Vec<f64>,     // PUCT prior over the legal set; empty = not yet evaluated
    obs: Vec<f32>,       // this agent's staged observation (per-agent egocentric encoding)
    value: f64,          // this agent's net value here (leaf eval / depth-cap source)
    total_visits: u32,
}

/// A simultaneous node's state: one decoupled [`AgentTable`] per agent plus joint-action
/// children. A node evaluation consumes one net row per agent's observation — the sim-fate
/// identity counts the extras under `extra_eval_rows`. Children and per-agent edge rewards are
/// keyed by the mixed-radix joint slot `(((s0)·L1 + s1)·L2 + s2)…` over each table's legal-set
/// width (identical to the former `s0·L1 + s1` at two agents). The joint space is stored dense —
/// at game-typical widths (snake: ≤4 legal per agent) the product stays small; a sparse keying is
/// a measured follow-up if a game demands it.
struct SimNode {
    tables: Vec<AgentTable>, // [N]
    child: Vec<i64>,         // [∏ Lᵢ]
    reward: Vec<f64>,        // [∏ Lᵢ · N] flat per-agent immediate rewards for the joint action
}

impl SimNode {
    /// Decompose a joint slot into each agent's table slot, visiting agents LAST to FIRST (the
    /// mixed-radix digits come off least-significant first). Per-agent updates are independent,
    /// so visiting order never affects results.
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

/// How this tree backs values up, fixed by the root's dynamics and agent count.
#[derive(Clone, Copy, PartialEq, Eq)]
enum TreeMode {
    /// Sequential, ≤2 agents: scalar negamax (zero-sum sign flip across turn changes) — the
    /// legacy path, bit-identical for existing games. MEASURED against forced Max^N (connect4
    /// AZ, equal decisions, 2 seeds): negamax ahead head-to-head (~0.60 mean) at ~35% less wall
    /// (one leaf row vs two) — the zero-sum opponent model comes free here, where Max^N must
    /// learn both value functions. Negamax therefore stays the 2p default;
    /// `SequentialBackup::MaxN` remains the re-measurement seam.
    SeqNegamax,
    /// Sequential, N>2 agents (Max^N): per-agent value vectors propagate absolutely (no
    /// perspective flips); each mover's edge keeps its OWN component for selection. Leaves are
    /// evaluated from every agent's perspective (N pooled rows).
    SeqMaxN,
    /// Simultaneous: decoupled per-agent tables (DUCT), own-perspective vector backup.
    Sim,
}

/// What one simulation reached, dictating how it backs up: a fresh non-terminal leaf whose value comes
/// from the pooled net forward, a leaf whose value is already known (terminal = 0, or a cached
/// depth-capped node) and can be backed up immediately, or an `ExpandAll` chance fan whose outcome
/// children all await the pooled forward at once.
enum Reached {
    Eval,     // fresh leaf (or un-evaluated PUCT node) at `leaf`: 1 row (negamax) or N (sim / Max^N)
    Terminal, // in-tree terminal: exact value 0 (per agent) — no forward
    DepthCapped, // depth cap: back up the leaf's cached net value(s) — no forward
    Fan,      // ExpandAll chance node at `leaf`: every outcome child staged for evaluation
}

/// What expanding a decision edge produced: a plain child leaf, a fresh chance node to descend
/// (lazy-outcome modes), or an `ExpandAll` chance node with every outcome child materialized.
enum Expanded {
    Leaf(usize),
    Chance(usize),
    Fan(usize),
}

/// What a completed multi-row evaluation does once its last row lands: an `ExpandAll` fan backs up
/// the exact outcome expectation; a simultaneous-node evaluation backs up both agents' leaf values.
enum PendingWork {
    Fan,             // the chance node at `leaf` whose outcome children await rows
    NodeEval(usize), // a sim node (or Max^N decision leaf) awaiting its N per-agent rows
}

struct Tree<S> {
    arena: Vec<Node<S>>,
    sims: usize,
    path: Vec<(usize, usize)>, // (node idx, slot) edges from root to the current leaf
    leaf: usize,
    // The agent this tree answers for (the engine's request): a simultaneous root densifies THIS
    // agent's table into the returned evaluation. Decision roots ignore it (the mover's arrays are
    // the evaluation, as before).
    requester: usize,
    // The backup scheme (fixed by the root's dynamics + agent count; v1 forbids mixing dynamics).
    mode: TreeMode,
    n_agents: usize,
    max_depth_seen: i32,
    // This search's chance stream (outcome draws), seeded per request — disjoint from the PUCT
    // noise stream, never drawn for games that declare no chance (deterministic games bit-identical).
    rng: SplitMix64,
    // An in-flight multi-row evaluation: (what completes, rows still awaited) — see `PendingWork`.
    pending: Option<(PendingWork, usize)>,
    // Per-search sim-fate counters, one bucket per simulation (the per-move identity `sims =
    // fresh rows + cache hits + shared rows + terminal + depth-capped − extra_eval_rows` — see
    // `SearchStats`). The tree counts them at the moment each sim resolves, so the identity is
    // search-local and exact.
    terminal_sims: usize,
    depthcap_sims: usize,
    shared_rows: usize,
    fresh_rows: usize,
    hit_rows: usize,
    extra_eval_rows: usize,
    // Reused scratch for the vector backups (sim / Max^N) — per-simulation allocations showed up
    // in the 2p snake regression measurement, so these paths stay allocation-free.
    g_buf: Vec<f64>,
    val_buf: Vec<f64>,
    // Undiscounted chance-edge rewards accumulated while a backup walks chance hops, consumed by
    // the first decision edge above them (they belong to that edge's tick).
    pend_buf: Vec<f64>,
    // A fan's expected chance-edge rewards (per agent), seeded by `fan_backprop` so the decision
    // edge above adds them UNDISCOUNTED — folding them into the mixed value would wrongly put
    // them behind that edge's gamma. Consumed (cleared) by the next backup.
    pend_seed: Vec<f64>,
}

/// The per-agent rewards a chance-edge transition emitted — empty when it emitted nothing (the
/// common case, kept empty so backups skip the fold entirely).
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

/// One flattened fan leaf: `(state, compound probability, accumulated per-agent chance
/// rewards, terminal)`.
type FanLeaf<S> = (S, f64, Vec<f64>, bool);

/// Merge two per-agent chance-reward vectors (either may be empty = no emissions).
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

/// Flatten chance CHAINS under `ExpandAll`: run every seeded `(state, prob, chance rewards,
/// terminal)` entry to its final decision/terminal leaves, compounding probabilities and
/// accumulating each chain edge's emitted rewards — the whole chain is one ply, exactly as the
/// expectimax fan resolves it. The aggregate flattened width is capped at
/// [`MAX_ENUMERATED_OUTCOMES`], checked BEFORE each expansion so the bound is real, not
/// retrospective. Returns the leaves and whether any chain expansion happened (an unchained fan
/// keeps its outcome-parallel layout and weights).
fn flatten_chance_fan<G: Game>(
    game: &G,
    reward: &dyn Reward<Event = G::Event>,
    n_agents: usize,
    seed: Vec<FanLeaf<G::State>>,
) -> (Vec<FanLeaf<G::State>>, bool) {
    let mut leaves: Vec<FanLeaf<G::State>> = Vec::with_capacity(seed.len());
    let mut chained = false;
    // ONE stack holding every seed entry (plus hop counts — the chain-cycle backstop), so the
    // projected-size check below sees unprocessed seeds too: a per-seed stack would let one
    // outcome flatten to the full cap and later outcomes append past it. Seeds are pushed in
    // reverse so pops preserve outcome order (an unchained fan must stay outcome-parallel).
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
            let count = dist.count();
            assert!(
                count <= MAX_ENUMERATED_OUTCOMES,
                "ExpandAll cannot enumerate {count} chance outcomes (bound {}); use a sampling chance mode for combinatorial outcome spaces",
                MAX_ENUMERATED_OUTCOMES
            );
            // Projected size BEFORE pushing (the popped parent is already off the list): the cap
            // bounds the fan that will actually exist, not the fan minus its last expansion.
            assert!(
                leaves.len() + stack.len() + count <= MAX_ENUMERATED_OUTCOMES,
                "a chance chain's flattened fan exceeds the enumeration bound ({}); use a narrower sampling mode",
                MAX_ENUMERATED_OUTCOMES
            );
            let probs: Vec<f64> = dist.iter_probs().collect();
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

/// Build one agent's decoupled table at a simultaneous state.
fn agent_table<G: Game>(
    game: &G,
    enc: &dyn StateEncoder<State = G::State>,
    state: &G::State,
    agent: usize,
) -> AgentTable {
    let mut actions = game.legal_actions(state, agent);
    if actions.is_empty() {
        // An inactive agent (e.g. a dead snake in a play-to-last game) has no legal moves but the
        // node is still simultaneous: give it a single placeholder slot — the engine substitutes
        // action 0 for non-actors, and the game ignores an inactive agent's move by contract.
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

/// A fresh simultaneous node (every agent's table + empty joint-child arrays).
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

/// A fresh Max^N decision leaf: the mover's ordinary sparse arrays plus every agent's staged
/// observation (per-perspective leaf values) and per-agent edge-reward storage.
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

    /// Select from the root down to an expandable edge (scored by `guidance`; chance nodes are
    /// descended per `chance` mode), create its child (stepping the game and materializing a chance
    /// outcome where the game declares one), and mark it as the leaf to back up. Returns how the
    /// leaf's value is obtained. Under PUCT the root itself is the first leaf: it is created
    /// without an eval, and selection needs its prior, so simulation 1 evaluates it in place
    /// (empty path).
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
                // Chance descent: pick an outcome slot per mode; materialize its child on first
                // visit (a fresh leaf to evaluate), else keep descending. Explicit chance states
                // may chain — a materialized child that is itself a chance node keeps descending.
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
                            ni = child; // the outcome landed on another chance state
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
                // Decoupled (DUCT) descent: each agent independently scores its OWN table with the
                // ordinary rule; the pair is the joint edge.
                if matches!(guidance, Guidance::Puct { .. }) && !sim.evaluated() {
                    self.leaf = ni; // un-evaluated PUCT node: both agents' rows, in place
                    return Reached::Eval;
                }
                if node.depth >= max_depth {
                    self.leaf = ni;
                    return Reached::DepthCapped;
                }
                // Decoupled selection per agent, composed into the mixed-radix joint slot.
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
                self.leaf = ni; // un-evaluated PUCT node (the root on sim 1): evaluate it in place
                return Reached::Eval;
            }
            if node.depth >= max_depth {
                self.leaf = ni; // depth cap: back up the cached net value(s) evaluated at creation
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
                            Reached::Eval // its obs is staged for the pooled forward
                        };
                    }
                    Expanded::Chance(cni) => {
                        ni = cni; // fresh chance node: descend it (picks + materializes an outcome)
                        continue;
                    }
                    Expanded::Fan(cni) => {
                        self.leaf = cni; // every outcome child staged; backup runs when all arrive
                        return Reached::Fan;
                    }
                }
            }
            ni = self.arena[ni].child[a] as usize;
        }
    }

    /// The outcome slot a descent takes through the chance node `ni`, per mode: `Committed` picks
    /// uniformly among its frozen draws; the resampling modes draw fresh ∝ probability.
    fn pick_chance_slot(&mut self, ni: usize, chance: ChanceMode) -> usize {
        // Destructure for disjoint borrows: `dist`/`committed` borrow the arena while the draw
        // needs the rng.
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
                    // A flattened chain fan: slots carry compound leaf probabilities.
                    crate::rng::weighted_index(rng, fan_weights)
                }
            }
        }
    }

    /// The already-materialized child for chance node `ni`'s `slot`, if any — dense (`Committed`
    /// keyed by draw, `ExpandAll` by outcome) or sparse (`AlwaysResample` pairs).
    fn chance_child(&self, ni: usize, slot: usize) -> Option<usize> {
        let node = &self.arena[ni];
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

    /// Build the child for outcome `slot` of chance node `cni`: realize the outcome's full
    /// transition — it may emit events (folded into the tick above via `chance_in`), end the
    /// game, or chain to another chance state (the caller keeps descending).
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
            resampled.insert(slot, idx); // sparse: AlwaysResample keys by distinct outcome
        } else {
            self.arena[cni].child[slot] = idx as i64;
        }
        idx
    }

    /// The joint action (and mover hint) for edge `ai` of node `ni` — a one-hot joint for a
    /// decision node, both agents' actions for a simultaneous one.
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

    /// Record edge `ai`'s immediate reward(s) on node `ni` from the transition's events.
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
                    // General-sum: a mover's edge can pay every agent (Max^N backs the full
                    // vector up); the mover-own copy above still drives selection.
                    let n = self.n_agents;
                    for ag in 0..n {
                        self.arena[ni].rewards_all[ai * n + ag] =
                            crate::reward::edge_reward(reward, &t.events, ag);
                    }
                }
            }
        }
    }

    /// Store edge `ai`'s child index on node `ni` (dispatching on the node's child storage).
    fn set_edge_child(&mut self, ni: usize, ai: usize, idx: usize) {
        match &mut self.arena[ni].kind {
            NodeKind::Simultaneous(sim) => sim.child[ai] = idx as i64,
            _ => self.arena[ni].child[ai] = idx as i64,
        }
    }

    /// A fresh child leaf for `state` — terminal, sequential decision, simultaneous, or an
    /// explicit CHANCE node, per the game's actor there. v1 forbids mixing sequential and
    /// simultaneous dynamics in one game. `depth` is the would-be decision depth: a chance state
    /// sits at `depth - 1` (transparent — its decision children land at `depth`, the same tick).
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
                // An explicit chance state: a fixed-probability node — never an arm of UCB/PUCT,
                // never evaluated by the net (no observation staged), transparent to depth.
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
                        let count = dist.count();
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

    /// Step the game for the edge at slot `ai` (an index into the node's legal `actions`). A
    /// child at a decision state appends directly; one at a chance STATE appends a chance node
    /// (drawing `Committed` outcomes now, or materializing every outcome for `ExpandAll`).
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

    /// Materialize the explicit chance node `cni`'s outcomes now (`ExpandAll`), so the caller
    /// can stage them all for one pooled evaluation. Chained chance states flatten into the fan
    /// (compound probabilities, accumulated chance rewards — `fan_weights` then carries the leaf
    /// distribution); terminal outcomes need no row (their value is exact).
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
        let probs: Vec<f64> = dist.iter_probs().collect();
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

    /// Back up `leaf_value` (from the leaf actor's perspective) along the selected path: negamax
    /// across turn changes (zero-sum), discounting by gamma and adding each decision edge's
    /// immediate reward. Chance hops are transparent — no reward, no discount, no stats; the
    /// decision edge above the chance node carries all of those, so a stochastic edge costs exactly
    /// one reward + one discount, like a deterministic one.
    fn backprop(&mut self, gamma: f64, leaf_value: f64) {
        self.arena[self.leaf].value = leaf_value;
        let mut g = leaf_value; // value from `g_actor`'s perspective
        let mut g_actor = self.arena[self.leaf].actor;
        // Undiscounted chance-edge rewards awaiting the first decision edge above — they belong
        // to that edge's tick, so they join BEFORE its gamma (negamax caps at two agents, so a
        // stack pair suffices). Seeded by a completed fan, fed by chance hops on the path.
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

    /// Back a simulation up the path with the right scheme for this tree: scalar negamax for
    /// sequential ≤2-agent games (`vals[0]` from the leaf actor's perspective), per-agent
    /// own-perspective vector backup for simultaneous (DUCT) and sequential Max^N trees — both
    /// general-sum: each agent's statistics take its own reward stream, no sign flips.
    fn backup(&mut self, gamma: f64, vals: &[f64]) {
        match self.mode {
            TreeMode::Sim => self.backprop_sim(gamma, vals),
            TreeMode::SeqMaxN => self.backprop_maxn(gamma, vals),
            TreeMode::SeqNegamax => self.backprop(gamma, vals[0]),
        }
    }

    /// Terminal backup: exact value 0 for every agent.
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

    /// Back up the cached net value(s) sitting at `leaf` (depth caps and completed multi-row
    /// evaluations): the sim tables', the Max^N per-perspective vector, or the scalar.
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

    /// The simultaneous (DUCT) backup: per path edge, each agent's table takes
    /// `qᵢ = rewardᵢ + γ·gᵢ` on its OWN selected slot. Chance hops stay transparent.
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
                    // Mixed-radix digits come off least-significant (last agent) first; per-agent
                    // updates are independent, so the visiting order never affects results.
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

    /// The Max^N backup (sequential N>2): the per-agent value vector propagates absolutely — no
    /// perspective flips — with each decision edge adding its full per-agent reward vector; the
    /// mover's edge statistics keep the mover's OWN component (what its selection maximizes).
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
                continue; // transparent: values pass through unchanged
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

    /// Complete an `ExpandAll` fan once every outcome child's evaluation has arrived: the sim backs
    /// up the exact probability-weighted expectation of the children's values (each from its own
    /// actor's perspective — the contract fixes one actor across a transition's outcomes; per-agent
    /// values mix independently on a simultaneous tree).
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
        // The fan's expected chance-edge rewards, kept SEPARATE from the mixed value: they
        // belong to the decision edge's tick above, so they must join before its gamma —
        // `pend_seed` hands them to the backup undiscounted.
        let mut reward_mix = vec![0.0f64; n];
        // A flattened chain fan carries its own compound leaf probabilities.
        let fan_probs: Vec<f64> = if fan_weights.is_empty() {
            dist.iter_probs()
                .take(self.arena[cni].child.len())
                .collect()
        } else {
            fan_weights
        };
        // Negamax: children's values are each from their OWN mover's perspective, and explicit
        // chance outcomes may hand the turn to different movers — normalize to a reference
        // perspective (the first non-terminal child's; sound under this mode's 2p zero-sum
        // contract). Terminal outcomes contribute their exact value, 0.
        let mut ref_actor = self.arena[cni].actor;
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
                            _ => 0.0, // terminal outcome: exact value 0
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

    /// The finished tree's root evaluation: per-action mean value `values[1][A]` (0 for any unvisited
    /// action) and visit counts, plus telemetry.
    /// The finished tree's root evaluation, densified back to the full action space: illegal (and
    /// unvisited) actions carry value 0 and visit count 0 — so π targets naturally put zero mass on
    /// illegal moves, and by-visits acting can never pick one.
    fn evaluation(self, actions: usize) -> SearchEvaluation {
        let root = &self.arena[0];
        let mut values = vec![0.0f64; actions];
        let mut visits = vec![0.0f64; actions];
        // A simultaneous root densifies the REQUESTING agent's decoupled table; a decision root's
        // flat arrays are the evaluation, as before.
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

/// Score a node's actions (mover's perspective) by the guidance rule and pick the best.
/// UCT: UCB1, an unvisited action wins outright. PUCT: `Q + c·P·√N_total/(1+n)` with unvisited
/// `Q = 0` (the AlphaZero convention) — the prior, not optimism, drives first visits; `N_total` is
/// floored at 1 so the very first selection is prior-ordered rather than degenerate.
fn select_edge<S>(node: &Node<S>, guidance: &Guidance) -> usize {
    select_scored(
        &node.visits,
        &node.value_sum,
        &node.prior,
        node.total_visits,
        guidance,
    )
}

/// One agent's decoupled selection at a simultaneous node — the identical rule over its own table.
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

/// A leaf state's value = greedy max over the LEGAL actions of the head-mean net Q (matches the
/// expectimax bootstrap; restricting to legal keeps an illegal move's phantom Q out of the bootstrap).
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

/// Pooled UCT over a batch of `(state, agent)` requests: each round advances every active tree by one
/// simulation, batching the new non-terminal leaves' observations into a single `infer` forward.
/// `seed` drives the per-tree chance streams (inert for games without chance states).
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
        false, // UCT never forces Max^N (its leaf values need own-turn decision points)
    )
}

/// The shared pooled search loop behind both `mcts_many` (UCT) and `alphazero_many` (PUCT): each round
/// advances every active tree by one simulation, staging the new leaves' observations on one
/// [`Evaluator`] batch (which dedupes identical positions and serves infer-cache hits inline), then
/// consumes each committed row per the guidance mode (UCT: Q rows → leaf value; PUCT: logits+value →
/// node prior + backed-up value, with root noise mixed in once per tree).
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
    assert!(
        game.num_agents() >= 1,
        "a game must have at least one agent"
    );
    assert!(
        game.perfect_information(),
        "tree search on a hidden-information game is clairvoyant: its values condition on state \
         the agents cannot observe; use an observation-only policy family"
    );
    let a = game.action_count();
    let mut trees: Vec<Tree<G::State>> = requests
        .into_iter()
        .enumerate()
        .map(|(ti, (state, agent))| {
            // Per-tree chance stream, disjoint from the PUCT root-noise stream (different salt).
            let chance_seed =
                seed ^ 0x53A3_C5A9_1D87_2F6B ^ (ti as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            Tree::new(game, enc, state, agent, chance_seed, force_maxn)
        })
        .collect();
    // Sequential N>2 (Max^N) needs per-perspective leaf VALUES; a Q-derived (UCT) leaf value is
    // only defined at the evaluated agent's own decision points, which sequential games give
    // non-movers none of. Simultaneous games are fine at any N (every agent owns its table).
    assert!(
        !(matches!(guidance, Guidance::Uct { .. })
            && trees.iter().any(|t| t.mode == TreeMode::SeqMaxN)),
        "UCT supports sequential games only up to 2 agents; N-player sequential search needs a \
         value head (PUCT / AlphaZero)"
    );

    while trees.iter().any(|t| t.sims < num_simulations) {
        let mut batch = eval.batch();
        // Per ticket: the (tree, node, table-slot) triples awaiting the row — the node is the
        // tree's pending leaf, an ExpandAll fan's outcome child, or a simultaneous node (slot =
        // which agent's table the row feeds; 0 for decision rows).
        let mut consumers: Vec<Vec<(usize, usize, usize)>> = Vec::new();

        for (ti, tree) in trees.iter_mut().enumerate() {
            // Advance-until-miss: keep simulating this tree — terminals, depth caps, and cache
            // hits all resolve synchronously — until it needs a real forward (or its budget ends).
            // Hits therefore reduce the number of pooled calls, not just their width (which is
            // what matters on latency-bound devices), and terminal-heavy rounds no longer idle.
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
                            // A sim node (or Max^N decision leaf) evaluates EVERY agent's
                            // observation — N rows for one simulation (the extras land in
                            // `extra_eval_rows`). The count is set before any row is consumed,
                            // so an early cache hit cannot complete the evaluation prematurely.
                            let n = tree.n_agents;
                            tree.extra_eval_rows += n - 1;
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
                                        stage(tree, ti, leaf, ag, ticket, &mut consumers);
                                    }
                                }
                            }
                            if tree.pending.is_some() {
                                break; // one or more rows await the pooled forward
                            }
                        } else {
                            // A negamax row is the leaf MOVER's perspective (and network).
                            match batch
                                .resolve_or_stage(tree.arena[leaf].actor, &tree.arena[leaf].obs)
                            {
                                Resolve::Resolved(row) => {
                                    tree.hit_rows += 1;
                                    consume_row(tree, leaf, 0, &row, guidance, gamma, a, ti, enc);
                                    // resolved from cache — keep advancing this tree
                                }
                                Resolve::Staged(ticket) => {
                                    stage(tree, ti, leaf, 0, ticket, &mut consumers);
                                    break; // this tree waits on the pooled forward
                                }
                            }
                        }
                    }
                    Reached::Fan => {
                        // ExpandAll: stage every outcome child of the chance node at `leaf` — one
                        // row per negamax decision child, N per sim/Max^N child. The sim's backup
                        // (`fan_backprop`) runs once the last row arrives — which may be
                        // immediately, if the cache serves them all. Terminal outcomes (explicit
                        // chance can end the game) need no row: their value is exactly 0.
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
                            // Every outcome is terminal: the fan resolves exactly, in place.
                            tree.terminal_sims += 1;
                            tree.fan_backprop(gamma);
                            continue;
                        }
                        tree.extra_eval_rows += total_rows.saturating_sub(1);
                        // The count covers EVERY row before any is consumed, so an early cache hit
                        // cannot trigger the fan's backup prematurely; `consume_row` decrements
                        // per delivered row and fires `fan_backprop` on the last one.
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
                                // Multi-perspective fans tag per perspective; a negamax fan's
                                // single row is the CHILD mover's.
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
                                        stage(tree, ti, child, ag, ticket, &mut consumers);
                                    }
                                }
                            }
                        }
                        if tree.pending.is_some() {
                            break; // some rows await the pooled forward
                        }
                        // fully cache-served: the fan completed in place — keep advancing
                    }
                }
            }
        }
        let rows = batch.commit();
        for (ticket, waiting) in consumers.iter().enumerate() {
            for &(ti, node, slot) in waiting {
                consume_row(
                    &mut trees[ti],
                    node,
                    slot,
                    rows.row(ticket),
                    guidance,
                    gamma,
                    a,
                    ti,
                    enc,
                );
            }
        }
    }

    trees.into_iter().map(|t| t.evaluation(a)).collect()
}

/// Record a staged row's consumer and its sim-fate bucket: a fresh ticket is this tree's forwarded
/// row; an existing ticket is within-batch dedup (an identical position staged by an earlier tree).
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

/// Deliver one net-output row to node `ni` of `tree`: UCT derives the legal-max head-mean Q value,
/// PUCT the legal-set prior (with per-tree root Dirichlet noise) plus value. A single pending leaf
/// backs up immediately; an `ExpandAll` fan child only stores its value, and the sim backs up the
/// exact weighted expectation when the fan's last row lands. One code path for fresh forwards,
/// cache hits, and deduped rows — so caching cannot change search behavior.
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
    let is_sim_node = matches!(tree.arena[ni].kind, NodeKind::Simultaneous(_));
    let is_maxn_node = !is_sim_node && tree.mode == TreeMode::SeqMaxN;
    // The perspective this row was encoded for — a negamax decision node's obs is the mover's; a
    // sim table's (or Max^N per-agent) obs is that slot's agent. All gathers from the raw row
    // cross into the head frame through this agent's view.
    let row_agent = if is_sim_node || is_maxn_node {
        slot
    } else {
        tree.arena[ni].actor
    };
    // A Max^N non-mover row supplies only its agent's leaf value: the prior belongs to the mover
    // (whose legal set the node's edges are), and UCT is unreachable here (asserted at entry —
    // Q-derived leaf values are only defined at the evaluated agent's own decision points).
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
            // Prior over the LEGAL set only: gather the node's legal actions' logits and softmax
            // those — illegal moves get zero mass by construction (the AlphaZero convention), and
            // the net never needs to learn to suppress them. At a simultaneous node the row feeds
            // ONE agent's table (its own legal set and egocentric observation).
            let node_actions: &[usize] = match &tree.arena[ni].kind {
                NodeKind::Simultaneous(sim) => &sim.tables[slot].actions,
                _ => &tree.arena[ni].actions,
            };
            let legal_logits: Vec<f64> = node_actions
                .iter()
                .map(|&act| logits[view.head_index(act, row_agent)])
                .collect();
            let mut prior = softmax(&legal_logits);
            // Root noise scope: a decision root's one table always gets it; a simultaneous root's
            // tables get it per `noise_scope` — the requester's always, the co-movers' only
            // under `All` (deliberately perturbed co-mover models are opt-in).
            let noised = ni == 0 && (!is_sim_node || slot == tree.requester || *noise_all);
            if noised {
                // Mix in the Dirichlet exploration noise (per-tree stream, so pooled searches stay
                // deterministic and independent — and cached root rows renoise identically, since
                // the cache stores raw logits), drawn over the legal set. A simultaneous root's
                // two tables draw disjoint streams (slot-salted).
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
    // Store the value where completion will read it.
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

    fn supports_imperfect_information(&self) -> bool {
        false // the tree branches on the true state (clairvoyant past hidden information)
    }

    fn max_agents(&self, sequential: bool) -> Option<usize> {
        // Simultaneous games: any N (decoupled per-agent tables). UCT over a SEQUENTIAL game
        // caps at 2 agents — Q-derived leaf values exist only at the evaluated agent's own
        // decision points, which sequential games give non-movers none of. (The search entry
        // keeps an assert as the direct-core backstop.)
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
        // the tree's root evaluation is always one value row plus full-width visits (both acting
        // modes; `Tree::evaluation` densifies)
        decode_search_eval(r, action_count, 1, true)
    }

    fn policy_state_to_u64(&self, s: &u32) -> u64 {
        u64::from(*s)
    }

    fn policy_state_from_u64(&self, v: u64) -> Result<u32, String> {
        u32::try_from(v).map_err(|_| format!("acting-ply counter {v} out of range"))
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
        // Argmax over the LEGAL set: densified rows carry 0 on illegal slots, and a 0 can
        // out-argmax all-negative legal values in a losing position.
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
        Self::fold_search_stats(eval, stats);
    }
}

impl SearchPolicy for Mcts {
    fn supports_chance(&self, _mode: ChanceMode) -> bool {
        true // sampled-trajectory search: every mode is expressible
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
    fn value_acting_never_picks_an_illegal_densified_zero() {
        // Losing position: all LEGAL values negative; the densified illegal slot carries 0 and
        // must not win the argmax (the sparse-game act_by="value" bug class).
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
        e.legal = vec![1]; // slot 0 is illegal — its 0.0 is a densification artifact
        let mut moves = 0u32;
        assert_eq!(p.select(&e, &mut moves, &mut FakeRng(vec![])), 1);
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
        // No visits anywhere: scores reduce to c·P(a) (N_total floored at 1) — the prior decides,
        // unlike UCT where any unvisited action wins outright.
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
        // Action 1 has the bigger prior but many visits and a mediocre Q; action 0's small-n
        // exploration term wins — the 1/(1+n) decay working as intended.
        let node = puct_node(vec![0.3, 0.7], vec![1, 99], vec![0.1, 9.9]); // Q = 0.1 both
        let g = Guidance::Puct {
            c: 2.0,
            noise: None,
            noise_all: false,
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
    //! The legal-action masking path, on a synthetic game whose legal set is a strict subset of the
    //! action space (every real game today is fully legal, so only this covers sparse nodes).
    use super::*;
    use crate::encoder::StateEncoder;
    use crate::game::{Actor, Game, Rng, Transition};
    use crate::policies::tree::alphazero::{alphazero_many, AlphaZeroConfig};
    use crate::reward::Reward as RewardTrait;

    /// A 1-player counting game over A=10 actions where only EVEN action ids are ever legal.
    /// Action `a` adds `a` to a running total; reaching total >= 8 is a terminal win (+1).
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
        fn initial_state(&self, _rng: &mut dyn Rng) -> St {
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
                vec![0.0; n * 11] // A logits + value
            } else {
                vec![0.0; n * 10] // K=1 Q rows
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
        // The winning line exists (e.g. 8 outright, or 4+4): argmax visits must be even.
        let best = (0..10).max_by(|&x, &y| eval.visits[x].partial_cmp(&eval.visits[y]).unwrap());
        assert_eq!(best.unwrap() % 2, 0);
    }

    #[test]
    fn puct_visits_and_noise_stay_legal() {
        // Strong noise: even fully-noised priors must keep zero mass on illegal moves.
        let eval = run(true, 0.9);
        for a in (1..10).step_by(2) {
            assert_eq!(
                eval.visits[a], 0.0,
                "illegal action {a} was visited under noise"
            );
        }
        // π normalization over the dense vector still holds (root visits sum = sims - 1).
        assert!((eval.visits.iter().sum::<f64>() - 39.0).abs() < 1e-9);
    }
}

#[cfg(test)]
mod chance_tests {
    //! The chance-mode semantics on a two-ply synthetic game where the risky action's value is a
    //! pure expectation: safe pays 1 for certain; risky triggers declared chance — outcome 0
    //! pays 0, outcome 1 pays 3, at p = [0.5, 0.5] (E = 1.5 > 1). Rewards land on the second
    //! (terminal) ply, so everything the root learns about chance flows through the tree's
    //! chance-node machinery, not the immediate edge reward.
    use super::*;
    use crate::encoder::StateEncoder;
    use crate::game::{Actor, Game, Rng, Transition};
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
                Actor::Chance // the risky action's unresolved bonus
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
                // Safe (+1) resolves now; risky enters the chance state.
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
        fn initial_state(&self, _rng: &mut dyn Rng) -> St {
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
        let mut infer = |_p: usize, _obs: Vec<f32>, n: usize| vec![0.0; n * 2]; // K=1 zeros net
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
        // Each seed's search lives entirely inside its one drawn world: Q(risky) is ~0 or ~3,
        // never the expectation — and across seeds both worlds occur.
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
        // With two committed draws, Q(risky) is the mean over the frozen pair: ~0, ~1.5, or
        // ~3 depending on the draw — {0,3} recovers the expectation, doubles don't.
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
        // A value-bearing net (Q = state total) makes the fan's first backup exact: after the two
        // sims that expand safe then risky, Q(risky) is 0.5·0 + 0.5·3 = 1.5 with no sampling.
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
        assert_eq!(out.stats.extra_eval_rows, 1); // 2 outcomes, 1 row beyond the sim's own
                                                  // Sim-fate identity with the fan term: sims = fresh + hit + shared + term + cap − extra.
        let s = &out.stats;
        assert_eq!(
            2,
            s.fresh_rows + s.hit_rows + s.shared_rows + s.terminal_sims + s.depthcap_sims
                - s.extra_eval_rows
        );
    }

    #[test]
    fn puct_searches_chance_games() {
        // AlphaZero guidance over the same game: stride A+1 rows, uniform prior, zeros value —
        // the visit distribution must still find the risky action's expectation edge.
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
    //! Decoupled simultaneous (DUCT) search on a synthetic one-shot matrix game with a dominant
    //! action per agent: both agents pick between "good" (+1 for self) and "bad" (0), payoffs
    //! independent — so each agent's own table must discover its dominant action regardless of the
    //! opponent's behavior, and the two per-request trees must agree from either seat.
    use super::*;
    use crate::encoder::StateEncoder;
    use crate::game::{Actor, Game, Rng, Transition};
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
            // Agent 1's action 1 is dominant (+1 to itself); agent 0 is paid for COORDINATING with
            // agent 1 — so agent 0's values depend on the in-tree opponent model, which is what the
            // noise-scope test needs, while both seats still converge on action 1 in equilibrium.
            let coord = f64::from(u8::from(actions[0] == actions[1]));
            Transition {
                next_state: St { done: true },
                events: vec![Some([coord, 0.0]), Some([f64::from(actions[1] as u8), 0.0])],
                terminal: true,
            }
        }
        fn initial_state(&self, _rng: &mut dyn Rng) -> St {
            St { done: false }
        }
    }

    struct Enc;
    impl ActionView for Enc {}
    impl StateEncoder for Enc {
        type State = St;
        fn encode(&self, s: &St, agent: usize) -> Vec<f32> {
            vec![f32::from(u8::from(s.done)), agent as f32] // per-agent egocentric obs
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
        // Both seats' requests pooled — each tree answers for its own agent.
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
        // With noise on, Requester vs Both must differ (the opponent's prior is perturbed under
        // Both), while noise-off searches are scope-independent.
        let run = |eps: f64, scope: NoiseScope| {
            let mut infer = |_p: usize, obs: Vec<f32>, n: usize| -> Vec<f64> {
                // mildly obs-dependent logits so priors are not uniform
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

    /// Each simultaneous table's row must cross via ITS OWN agent's view: agent 1's logits arrive
    /// swapped into its head frame while agent 0 keeps the identity — the search must reproduce
    /// the all-identity run exactly (a slot-for-slot read of either table would break it).
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
                            (l[1], l[0]) // head frame for agent 1: swapped
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
    //! Search gathers must cross raw net rows through the encoder's `ActionView`. Each test runs
    //! the SAME search twice — an identity encoder against game-frame infer rows, and a rotated
    //! view against rows permuted into its head frame — and asserts identical search results:
    //! any gather that skipped (or double-applied) the view would break the equality.
    use super::*;
    use crate::encoder::StateEncoder;
    use crate::game::{Actor, Game, Rng, Transition};
    use crate::policies::tree::alphazero::{alphazero_many, AlphaZeroConfig};
    use crate::reward::Reward as RewardTrait;

    /// A=4 but only {0, 2} legal (sparse) — the gathers iterate the legal set, so a wrong frame
    /// reads a slot the net never associated with that move.
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
        fn initial_state(&self, _: &mut dyn Rng) -> St {
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

    /// Deterministic game-frame value for (state, action).
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
                            // identity: slot IS the game action; rot: slot holds the game
                            // action whose head index it is.
                            let game_a = if use_rot { (slot + A - 1) % A } else { slot };
                            v(s, game_a)
                        })
                    })
                    .collect() // K=1 Q rows
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
            noise_epsilon: 0.0, // noise off: priors are the only stochastic-free signal
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
                                v(s, game_a) * 4.0 // logits
                            })
                            .collect();
                        row.push(0.2); // value slot: layout, not an action — never permuted
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
    //! N>2 backup correctness: Max^N (sequential) must model co-players as SELF-interested — the
    //! discriminator game punishes the paranoid (all-vs-me) assumption — and DUCT-N
    //! (simultaneous) must let every agent find its own dominant action.
    use super::*;
    use crate::encoder::StateEncoder;
    use crate::game::{Actor, Game, Rng, Transition};
    use crate::policies::tree::alphazero::{alphazero_many, AlphaZeroConfig};
    use crate::reward::Reward as RewardTrait;
    use crate::rollout::evaluator::Evaluator;

    /// Agent 0 picks L or R; R ends at payoffs (3, 0, 0). L hands the move to agent 1: action a
    /// ends at (5, 10, 0), action b at (0, 0, 0). Agent 2 exists (forcing the N>2 path) but never
    /// moves. A SELF-interested agent 1 picks a (10 > 0), making L worth 5 > 3 to agent 0; the
    /// paranoid model would predict b (0 < 5 for agent 0) and choose R — so a preference for L is
    /// exactly the Max^N-vs-paranoid discriminator.
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
        fn initial_state(&self, _rng: &mut dyn Rng) -> Lr {
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
        // Uniform logits, zero value: the terminal payoffs are the only signal.
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

    /// 3-agent simultaneous one-shot: action 1 pays every agent +1 regardless of the others.
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
        fn initial_state(&self, _rng: &mut dyn Rng) -> DSt {
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
    //! The `SequentialBackup::MaxN` seam must actually change the TREE's backup at 2 agents —
    //! pinned with a general-sum discriminator where negamax (opponent minimizes MY value) and
    //! Max^N (opponent maximizes ITS OWN) pick different root actions. (An engine-level test
    //! can't see this: the value-row plumbing looks identical either way.)
    use super::*;
    use crate::encoder::StateEncoder;
    use crate::game::{Actor, Game, Rng, Transition};
    use crate::policies::tree::alphazero::{alphazero_many, AlphaZeroConfig};
    use crate::reward::Reward as RewardTrait;
    use crate::rollout::evaluator::Evaluator;

    /// Agent 0: L (0) hands the move to agent 1 — (a) pays (5, 10), (b) pays (0, 0); R (1) ends
    /// at (3, 0). A self-interested agent 1 picks (a), so Max^N values L at 5 and picks L; the
    /// paranoid negamax model assumes (b) and picks R.
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
        fn initial_state(&self, _rng: &mut dyn Rng) -> Lr {
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
