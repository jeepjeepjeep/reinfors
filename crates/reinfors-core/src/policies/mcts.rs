//! Monte-Carlo Tree Search: the shared arena tree + pooled simulation loop (`search_many`), guided
//! either by UCB1 (the `Mcts` policy here) or by net priors under PUCT (the `AlphaZero` policy in
//! `alphazero` — see [`Guidance`] for the exact axis of difference). The UCT policy produces the same
//! [`SearchEvaluation`] the `TreeStrap` learner consumes — the training target is the root's backed-up
//! per-action values `values[1][A]` ("MCTS-strap") — and pools its leaf evaluations across games into
//! one `infer` per round, exactly like the expectimax search.
//!
//! **Sequential + single-agent games only** (simultaneous support is planned; see the binding's
//! rejection). MCTS here assumes strictly alternating turns (or one agent), so a node's actor is a
//! single [`Actor::Agent`]; `Actor::Simultaneous` and `Actor::Chance` (the explicit chance *player*)
//! are rejected — this module panics as a backstop for direct core use. Two-player games are treated
//! as zero-sum (negamax backup) — correct for connect4.
//!
//! **Stochastic transitions** (post-move environment chance, e.g. a spawn after a move) are
//! supported through the game's *declared* distribution — [`Game::chance_outcomes`] +
//! [`Game::apply_chance`] — consumed per the configured [`ChanceMode`]: chance nodes sit between a
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

use crate::encoder::StateEncoder;
use crate::engine::CollectStats;
use crate::evaluator::{Evaluator, Resolve};
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

/// How the tree search consumes a stochastic transition's declared distribution
/// ([`Game::chance_outcomes`] + [`Game::apply_chance`]). The *game seam* is one thing — the
/// distribution, declared; this enum is the *search policy* over it, and the right mode is decided
/// by one ratio: **simulations-per-chance-edge vs. fan width** (the number of outcomes).
///
/// Worked contrast (snake, 3 free cells A/B/C for the respawn, V ≈ +0.9 / +0.3 / −0.3, 30 sims
/// through the "eat" edge):
/// - `AlwaysResample`: ~10 sims land in each world; Q(eat) → +0.3, the true expectation — but no
///   single world gets enough visits to plan a deep route.
/// - `Committed{1}`: one draw (say C) at expansion; all 30 sims plan deeply inside the C-world and
///   conclude Q(eat) = −0.3. The other worlds *never exist* for this search. Deep but biased, and
///   which bias is a coin flip.
/// - `Committed{2}`: two of three worlds get 15 sims each; the missing world's value is simply
///   absent from Q. k controls how tight that lottery is.
/// - `ExpandAll`: all three outcome states are built and net-evaluated at expansion; the edge is
///   seeded with the exact weighted expectation immediately. Three rows here — three *hundred* on
///   a real snake board.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub enum ChanceMode {
    /// Draw a **fresh** outcome ∝ probability on *every* descent through the edge (children keyed
    /// by outcome index, so repeat draws accumulate). Visits per outcome ≈ `p·edge_visits`: with a
    /// narrow or concentrated fan (backgammon's 21 rolls at 1000+ sims, 2048's spawn mass on 2s)
    /// every relevant outcome grows a genuinely searched subtree. Two properties make it the
    /// default: it is the only **asymptotically correct** mode (Q converges to the true expectation
    /// as sims grow; `Committed` converges to the expectation over its frozen draws — wrong forever,
    /// at any budget), and its training-target bias shrinks with budget instead of freezing in.
    /// Its weakness is the wide-fan/small-budget regime, where visits scatter so thin that no
    /// outcome's subtree deepens (snake: ~300 respawn cells at 64 sims ⇒ ~1 visit per world below
    /// the eat edge — value right in expectation, plan nonexistent).
    ///
    /// NOT equivalent to `Committed{samples: 1}` — that mode freezes its single draw for the whole
    /// search (a one-determinization search of a different, deterministic game); this mode
    /// re-consults the true distribution every descent. `Committed{k}` only approaches this mode
    /// as k → ∞.
    #[default]
    AlwaysResample,
    /// Draw `samples` outcomes ∝ probability **once at edge expansion** (independent draws, WITH
    /// replacement — the same estimator as expectimax's `food_samples`, snake's validated
    /// treatment) and thereafter pick uniformly among those frozen realizations. Trades bias for
    /// depth: the search conditions on k concrete futures and plans real routes inside them —
    /// the right trade when the fan is wide relative to the budget. Duplicated draws keep separate
    /// (equal-weight) branches, exactly like `food_samples`.
    Committed { samples: usize },
    /// Materialize and net-evaluate **every** outcome when the edge expands (one pooled batch —
    /// the expectimax exhaustive treatment): the edge's first backup is the exact
    /// probability-weighted expectation of the outcome values, and later descents pick ∝
    /// probability to deepen. Exact and immediate for narrow fans (dice); the per-expansion cost
    /// is one net row per outcome, which is ruinous on wide fans.
    ExpandAll,
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
    /// How the search consumes stochastic transitions' declared chance (see [`ChanceMode`]).
    /// Inert for games that declare no `chance_outcomes`.
    pub chance: ChanceMode,
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
}

/// A node is either a decision (an agent to move — the layout above) or a **chance node** sitting
/// between a decision edge and its outcome children. A chance node reuses `child` for its outcome
/// children (`-1` = not yet materialized) and is transparent to backup: no reward, no discount, no
/// stats — the action edge above it carries all of those.
enum NodeKind {
    Decision,
    Chance {
        /// The declared outcome distribution (`Game::chance_outcomes`), indexing `apply_chance`.
        probs: Vec<f64>,
        /// `Committed` mode: the frozen outcome draws (with replacement; `child` is parallel to
        /// this, so duplicated draws keep separate equal-weight branches, like `food_samples`).
        /// Empty in the other modes, where `child` is parallel to `probs`.
        committed: Vec<usize>,
    },
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
        }
    }
}

/// What one simulation reached, dictating how it backs up: a fresh non-terminal leaf whose value comes
/// from the pooled net forward, a leaf whose value is already known (terminal = 0, or a cached
/// depth-capped node) and can be backed up immediately, or an `ExpandAll` chance fan whose outcome
/// children all await the pooled forward at once.
enum Reached {
    Eval,
    Terminal,         // in-tree terminal: exact value 0 for the mover-to-be (no forward)
    DepthCapped(f64), // depth cap: use the node's cached net value (no forward)
    Fan,              // ExpandAll chance node at `leaf`: every outcome child staged for evaluation
}

/// What expanding a decision edge produced: a plain child leaf, a fresh chance node to descend
/// (lazy-outcome modes), or an `ExpandAll` chance node with every outcome child materialized.
enum Expanded {
    Leaf(usize),
    Chance(usize),
    Fan(usize),
}

struct Tree<S> {
    arena: Vec<Node<S>>,
    sims: usize,
    path: Vec<(usize, usize)>, // (node idx, slot) edges from root to the current leaf
    leaf: usize,
    max_depth_seen: i32,
    // This search's chance stream (outcome draws), seeded per request — disjoint from the PUCT
    // noise stream, never drawn for games that declare no chance (deterministic games bit-identical).
    rng: SplitMix64,
    // An in-flight ExpandAll fan: (chance node, rows still awaited). The sim's backup runs when the
    // count reaches zero (see `fan_backprop`).
    pending_fan: Option<(usize, usize)>,
    // Per-search sim-fate counters, one bucket per simulation (the per-move identity `sims =
    // fresh rows + cache hits + shared rows + terminal + depth-capped − fan_extra_rows` — see
    // `SearchStats`). The tree counts them at the moment each sim resolves, so the identity is
    // search-local and exact.
    terminal_sims: usize,
    depthcap_sims: usize,
    shared_rows: usize,
    fresh_rows: usize,
    hit_rows: usize,
    fan_extra_rows: usize,
}

impl<S: Clone> Tree<S> {
    fn new<G>(game: &G, enc: &dyn StateEncoder<State = S>, state: S, chance_seed: u64) -> Tree<S>
    where
        G: Game<State = S>,
    {
        let actor = sole_actor(game.actor(&state));
        let obs = enc.encode(&state, actor);
        let legal = game.legal_actions(&state, actor);
        let root = Node::leaf(state, actor, 0, false, legal, obs);
        Tree {
            arena: vec![root],
            sims: 0,
            path: Vec::new(),
            leaf: 0,
            max_depth_seen: 0,
            rng: SplitMix64::new(chance_seed),
            pending_fan: None,
            terminal_sims: 0,
            depthcap_sims: 0,
            shared_rows: 0,
            fresh_rows: 0,
            hit_rows: 0,
            fan_extra_rows: 0,
        }
    }

    /// Draw an index ∝ `probs` from this tree's chance stream.
    fn draw_outcome(&mut self, probs: &[f64]) -> usize {
        let total: f64 = probs.iter().sum();
        let mut r = self.rng.unit() * total;
        let mut last = 0;
        for (i, &p) in probs.iter().enumerate() {
            if p > 0.0 {
                last = i;
                r -= p;
                if r <= 0.0 {
                    return i;
                }
            }
        }
        last // numeric fallback (r exhausted by rounding): the last positive-mass outcome
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
        loop {
            if let NodeKind::Chance { .. } = self.arena[ni].kind {
                // Chance descent: pick an outcome slot per mode; materialize its child on first
                // visit (a fresh leaf to evaluate), else keep descending.
                let slot = self.pick_chance_slot(ni, chance);
                self.path.push((ni, slot));
                if self.arena[ni].child[slot] < 0 {
                    let child = self.materialize_outcome(game, enc, ni, slot);
                    self.leaf = child;
                    return if self.arena[child].terminal {
                        Reached::Terminal
                    } else {
                        Reached::Eval
                    };
                }
                ni = self.arena[ni].child[slot] as usize;
                continue;
            }
            let node = &self.arena[ni];
            self.max_depth_seen = self.max_depth_seen.max(node.depth);
            if node.terminal {
                self.leaf = ni;
                return Reached::Terminal;
            }
            if matches!(guidance, Guidance::Puct { .. }) && node.prior.is_empty() {
                self.leaf = ni; // un-evaluated PUCT node (the root on sim 1): evaluate it in place
                return Reached::Eval;
            }
            if node.depth >= max_depth {
                self.leaf = ni; // depth cap: use the cached net value evaluated when this node was created
                return Reached::DepthCapped(node.value);
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
        let NodeKind::Chance { probs, committed } = &self.arena[ni].kind else {
            unreachable!("pick_chance_slot on a decision node");
        };
        match chance {
            ChanceMode::Committed { .. } => self.rng.below(committed.len()),
            ChanceMode::AlwaysResample | ChanceMode::ExpandAll => {
                let probs = probs.clone();
                self.draw_outcome(&probs)
            }
        }
    }

    /// Build the decision child for outcome `slot` of chance node `cni`, re-deriving the parent
    /// edge's transition (chance nodes store no transition; one extra `Game::step` per
    /// materialization, never per descent).
    fn materialize_outcome<G>(
        &mut self,
        game: &G,
        enc: &dyn StateEncoder<State = S>,
        cni: usize,
        slot: usize,
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
        // The chance node's parent decision edge is the path entry just above it.
        let &(pni, pa) = self
            .path
            .iter()
            .rev()
            .find(|&&(n, _)| n != cni)
            .expect("chance node with no parent decision edge on the path");
        let mover = self.arena[pni].actor;
        let mut joint = vec![0usize; game.num_agents()];
        joint[mover] = self.arena[pni].actions[pa];
        let t = game.step(&self.arena[pni].state, &joint);
        let state = game.apply_chance(&self.arena[pni].state, &t, outcome);
        let child = self.decision_leaf(game, enc, state, mover, self.arena[cni].depth + 1, false);
        let idx = self.arena.len();
        self.arena.push(child);
        self.arena[cni].child[slot] = idx as i64;
        idx
    }

    /// A fresh decision leaf for `state` (terminal or with its obs staged for evaluation).
    fn decision_leaf<G>(
        &self,
        game: &G,
        enc: &dyn StateEncoder<State = S>,
        state: S,
        mover: usize,
        depth: i32,
        terminal: bool,
    ) -> Node<S>
    where
        G: Game<State = S>,
    {
        if terminal {
            Node::leaf(state, mover, depth, true, Vec::new(), Vec::new())
        } else {
            let actor = sole_actor(game.actor(&state));
            let obs = enc.encode(&state, actor);
            let legal = game.legal_actions(&state, actor);
            Node::leaf(state, actor, depth, false, legal, obs)
        }
    }

    /// Step the game for the edge at slot `ai` (an index into the node's legal `actions`). A
    /// deterministic transition appends the child directly; a declared-chance transition appends a
    /// chance node instead (drawing `Committed` outcomes now, or materializing every outcome for
    /// `ExpandAll`).
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
        let mover = self.arena[ni].actor;
        let action = self.arena[ni].actions[ai];
        let mut joint = vec![0usize; game.num_agents()];
        joint[mover] = action;
        let t = game.step(&self.arena[ni].state, &joint);
        self.arena[ni].reward[ai] = reward.step_reward(&t.events[mover], mover);
        let depth = self.arena[ni].depth + 1;
        if let Some(probs) = game.chance_outcomes(&self.arena[ni].state, &t) {
            debug_assert!(!t.terminal, "chance_outcomes on a terminal transition");
            debug_assert!(!probs.is_empty());
            return self.expand_chance(game, enc, ni, ai, &t, probs, chance);
        }
        let child = self.decision_leaf(game, enc, t.next_state, mover, depth, t.terminal);
        let idx = self.arena.len();
        self.arena.push(child);
        self.arena[ni].child[ai] = idx as i64;
        Expanded::Leaf(idx)
    }

    /// Append the chance node for a declared-chance edge, shaped per mode: `Committed` freezes its
    /// draws now (children parallel to them, materialized lazily); the resampling modes key children
    /// by outcome index; `ExpandAll` additionally materializes every outcome child immediately so the
    /// caller can stage them all for one pooled evaluation.
    #[allow(clippy::too_many_arguments)]
    fn expand_chance<G>(
        &mut self,
        game: &G,
        enc: &dyn StateEncoder<State = S>,
        ni: usize,
        ai: usize,
        t: &crate::game::Transition<S, G::Event>,
        probs: Vec<f64>,
        chance: ChanceMode,
    ) -> Expanded
    where
        G: Game<State = S>,
    {
        let mover = self.arena[ni].actor;
        let depth = self.arena[ni].depth;
        let committed = match chance {
            ChanceMode::Committed { samples } => {
                debug_assert!(samples >= 1, "ChanceMode::Committed requires samples >= 1");
                (0..samples.max(1))
                    .map(|_| self.draw_outcome(&probs))
                    .collect()
            }
            _ => Vec::new(),
        };
        let width = if committed.is_empty() {
            probs.len()
        } else {
            committed.len()
        };
        let mut chance_node = Node::leaf(
            t.next_state.clone(),
            mover,
            depth,
            false,
            vec![0],
            Vec::new(),
        );
        chance_node.kind = NodeKind::Chance { probs, committed };
        chance_node.actions = Vec::new();
        chance_node.child = vec![-1; width];
        let cni = self.arena.len();
        self.arena.push(chance_node);
        self.arena[ni].child[ai] = cni as i64;
        if let ChanceMode::ExpandAll = chance {
            // Materialize every outcome now; the caller stages all their observations at once and
            // `fan_backprop` seeds the edge with the exact weighted expectation when they arrive.
            for slot in 0..width {
                let state = game.apply_chance(&self.arena[ni].state, t, slot);
                let child = self.decision_leaf(game, enc, state, mover, depth + 1, false);
                let idx = self.arena.len();
                self.arena.push(child);
                self.arena[cni].child[slot] = idx as i64;
            }
            return Expanded::Fan(cni);
        }
        Expanded::Chance(cni)
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
        for &(ni, a) in self.path.iter().rev() {
            if let NodeKind::Chance { .. } = self.arena[ni].kind {
                continue; // transparent: perspective and value pass through unchanged
            }
            let node_actor = self.arena[ni].actor;
            let child_val = if g_actor == node_actor { g } else { -g };
            let q = self.arena[ni].reward[a] + gamma * child_val;
            self.arena[ni].value_sum[a] += q;
            self.arena[ni].visits[a] += 1;
            self.arena[ni].total_visits += 1;
            g = q; // now from node_actor's perspective, for the level above
            g_actor = node_actor;
        }
    }

    /// Complete an `ExpandAll` fan once every outcome child's evaluation has arrived: the sim backs
    /// up the exact probability-weighted expectation of the children's values (each from its own
    /// actor's perspective — the contract fixes one actor across a transition's outcomes).
    fn fan_backprop(&mut self, gamma: f64) {
        let cni = self.leaf;
        let NodeKind::Chance { probs, .. } = &self.arena[cni].kind else {
            unreachable!("fan_backprop on a decision node");
        };
        let total: f64 = probs.iter().sum();
        let mut mix = 0.0;
        let mut child_actor = self.arena[cni].actor;
        for (slot, &p) in probs.clone().iter().enumerate() {
            let child = self.arena[cni].child[slot] as usize;
            mix += p / total * self.arena[child].value;
            child_actor = self.arena[child].actor;
        }
        self.arena[cni].actor = child_actor; // fan value is from the outcome children's perspective
        self.backprop(gamma, mix);
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
        for (slot, &action) in root.actions.iter().enumerate() {
            if root.visits[slot] > 0 {
                values[action] = root.value_sum[slot] / root.visits[slot] as f64;
            }
            visits[action] = root.visits[slot] as f64;
        }
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
            fan_extra_rows: self.fan_extra_rows,
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

/// A leaf state's value = greedy max over the LEGAL actions of the head-mean net Q (matches the
/// expectimax bootstrap; restricting to legal keeps an illegal move's phantom Q out of the bootstrap).
fn leaf_value(q: &[f64], k: usize, a: usize, legal: &[usize]) -> f64 {
    legal
        .iter()
        .map(|&ai| (0..k).map(|h| q[h * a + ai]).sum::<f64>() / k as f64)
        .fold(f64::NEG_INFINITY, f64::max)
}

/// Pooled UCT over a batch of `(state, agent)` requests: each round advances every active tree by one
/// simulation, batching the new non-terminal leaves' observations into a single `infer` forward.
/// `seed` drives the per-tree chance streams (inert for games that declare no `chance_outcomes`).
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
        cfg.chance,
        seed,
        requests,
        eval,
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
) -> Vec<SearchEvaluation>
where
    G: Game + Sync,
    G::State: Send,
    F: FnMut(Vec<f32>, usize) -> Vec<f64>,
{
    let a = game.action_count();
    let mut trees: Vec<Tree<G::State>> = requests
        .into_iter()
        .enumerate()
        .map(|(ti, (state, _agent))| {
            // Per-tree chance stream, disjoint from the PUCT root-noise stream (different salt).
            let chance_seed =
                seed ^ 0x53A3_C5A9_1D87_2F6B ^ (ti as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            Tree::new(game, enc, state, chance_seed)
        })
        .collect();

    while trees.iter().any(|t| t.sims < num_simulations) {
        let mut batch = eval.batch();
        // Per ticket: the (tree, node) pairs awaiting the row — the node is the tree's single
        // pending leaf, or one outcome child of an ExpandAll fan.
        let mut consumers: Vec<Vec<(usize, usize)>> = Vec::new();

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
                        tree.backprop(gamma, 0.0);
                    }
                    Reached::DepthCapped(v) => {
                        tree.depthcap_sims += 1;
                        tree.backprop(gamma, v);
                    }
                    Reached::Eval => {
                        let leaf = tree.leaf;
                        match batch.resolve_or_stage(&tree.arena[leaf].obs) {
                            Resolve::Resolved(row) => {
                                tree.hit_rows += 1;
                                consume_row(tree, leaf, &row, guidance, gamma, a, ti);
                                // resolved from cache — keep advancing this tree
                            }
                            Resolve::Staged(ticket) => {
                                stage(tree, ti, leaf, ticket, &mut consumers);
                                break; // this tree waits on the pooled forward
                            }
                        }
                    }
                    Reached::Fan => {
                        // ExpandAll: stage every outcome child of the chance node at `leaf`; the
                        // sim's backup (`fan_backprop`) runs once the last row arrives — which may
                        // be immediately, if the cache serves them all.
                        let cni = tree.leaf;
                        let kids: Vec<usize> =
                            tree.arena[cni].child.iter().map(|&c| c as usize).collect();
                        tree.fan_extra_rows += kids.len().saturating_sub(1);
                        // The count covers EVERY child before any row is consumed, so an early
                        // cache hit cannot trigger the fan's backup prematurely; `consume_row`
                        // decrements per delivered row and fires `fan_backprop` on the last one.
                        tree.pending_fan = Some((cni, kids.len()));
                        for child in kids {
                            let obs = std::mem::take(&mut tree.arena[child].obs);
                            match batch.resolve_or_stage(&obs) {
                                Resolve::Resolved(row) => {
                                    tree.hit_rows += 1;
                                    consume_row(tree, child, &row, guidance, gamma, a, ti);
                                }
                                Resolve::Staged(ticket) => {
                                    stage(tree, ti, child, ticket, &mut consumers);
                                }
                            }
                        }
                        if tree.pending_fan.is_some() {
                            break; // some rows await the pooled forward
                        }
                        // fully cache-served: the fan completed in place — keep advancing
                    }
                }
            }
        }
        let rows = batch.commit();
        for (ticket, waiting) in consumers.iter().enumerate() {
            for &(ti, node) in waiting {
                consume_row(
                    &mut trees[ti],
                    node,
                    rows.row(ticket),
                    guidance,
                    gamma,
                    a,
                    ti,
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
    ticket: usize,
    consumers: &mut Vec<Vec<(usize, usize)>>,
) {
    if ticket < consumers.len() {
        tree.shared_rows += 1;
        consumers[ticket].push((ti, node));
    } else {
        tree.fresh_rows += 1;
        consumers.push(vec![(ti, node)]);
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
    row_data: &[f64],
    guidance: &Guidance,
    gamma: f64,
    a: usize,
    ti: usize,
) {
    let value = match guidance {
        Guidance::Uct { .. } => {
            let k = row_data.len() / a;
            leaf_value(row_data, k, a, &tree.arena[ni].actions)
        }
        Guidance::Puct { noise, .. } => {
            let (logits, value) = row_data.split_at(a);
            // Prior over the LEGAL set only: gather the node's legal actions' logits and softmax
            // those — illegal moves get zero mass by construction (the AlphaZero convention), and
            // the net never needs to learn to suppress them.
            let node_actions = &tree.arena[ni].actions;
            let legal_logits: Vec<f64> = node_actions.iter().map(|&act| logits[act]).collect();
            let mut prior = softmax(&legal_logits);
            if ni == 0 {
                // The root's one eval: mix in the Dirichlet exploration noise (per-tree stream, so
                // pooled searches stay deterministic and independent — and cached root rows renoise
                // identically, since the cache stores raw logits), drawn over the legal set.
                if let Some((eps, alpha, seed)) = noise {
                    if *eps > 0.0 {
                        let mut rng =
                            SplitMix64::new(seed ^ (ti as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
                        let noise_draw = dirichlet(&mut rng, *alpha, prior.len());
                        for (p, d) in prior.iter_mut().zip(noise_draw) {
                            *p = (1.0 - eps) * *p + eps * d;
                        }
                    }
                }
            }
            tree.arena[ni].prior = prior;
            value[0]
        }
    };
    match tree.pending_fan.as_mut() {
        Some((_, missing)) => {
            tree.arena[ni].value = value;
            if *missing > 0 {
                *missing -= 1;
            }
            if matches!(tree.pending_fan, Some((_, 0))) {
                tree.pending_fan = None;
                tree.fan_backprop(gamma);
            }
        }
        None => {
            debug_assert_eq!(ni, tree.leaf, "single-leaf row delivered to the wrong node");
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
        F: FnMut(Vec<f32>, usize) -> Vec<f64>,
    {
        mcts_many(game, enc, reward, &self.cfg, requests, seed, eval)
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
        stats.sum_terminal_sims += s.terminal_sims;
        stats.sum_depthcap_sims += s.depthcap_sims;
        stats.sum_shared_rows += s.shared_rows;
        stats.sum_fresh_rows += s.fresh_rows;
        stats.sum_hit_rows += s.hit_rows;
        stats.sum_fan_extra_rows += s.fan_extra_rows;
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

#[cfg(test)]
mod masking_tests {
    //! The legal-action masking path, on a synthetic game whose legal set is a strict subset of the
    //! action space (every real game today is fully legal, so only this covers sparse nodes).
    use super::*;
    use crate::encoder::StateEncoder;
    use crate::game::{Actor, Game, Rng, Transition};
    use crate::policies::alphazero::{alphazero_many, AlphaZeroConfig};
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
                events: vec![if total >= 8 { 1.0 } else { 0.0 }],
                terminal: total >= 8,
            }
        }
        fn initial_state(&self, _rng: &mut dyn Rng) -> St {
            St(0)
        }
    }

    struct Enc;
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
        let mut infer = |_obs: Vec<f32>, n: usize| -> Vec<f64> {
            if guidance_puct {
                vec![0.0; n * 11] // A logits + value
            } else {
                vec![0.0; n * 10] // K=1 Q rows
            }
        };
        let mut eval = Evaluator::new(&mut infer, None);
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
    use crate::policies::alphazero::{alphazero_many, AlphaZeroConfig};
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
        fn actor(&self, _s: &St) -> Actor {
            Actor::Agent(0)
        }
        fn legal_actions(&self, _s: &St, _agent: usize) -> Vec<usize> {
            vec![0, 1]
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
        fn chance_outcomes(&self, s: &St, t: &Transition<St, f64>) -> Option<Vec<f64>> {
            // Only the risky first-ply action (total unchanged by the deterministic part) is
            // stochastic.
            (s.ply == 0 && t.next_state.total == s.total).then(|| vec![0.5, 0.5])
        }
        fn apply_chance(&self, _s: &St, t: &Transition<St, f64>, outcome: usize) -> St {
            St {
                total: t.next_state.total + if outcome == 0 { 0 } else { 3 },
                ply: t.next_state.ply,
            }
        }
        fn initial_state(&self, _rng: &mut dyn Rng) -> St {
            St { total: 0, ply: 0 }
        }
    }

    struct Enc;
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
        let mut infer = |_obs: Vec<f32>, n: usize| vec![0.0; n * 2]; // K=1 zeros net
        let mut eval = Evaluator::new(&mut infer, None);
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
        let mut infer = |obs: Vec<f32>, n: usize| -> Vec<f64> {
            (0..n).flat_map(|i| [f64::from(obs[i * 2]); 2]).collect()
        };
        let mut eval = Evaluator::new(&mut infer, None);
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
        assert_eq!(out.stats.fan_extra_rows, 1); // 2 outcomes, 1 row beyond the sim's own
                                                 // Sim-fate identity with the fan term: sims = fresh + hit + shared + term + cap − extra.
        let s = &out.stats;
        assert_eq!(
            2,
            s.fresh_rows + s.hit_rows + s.shared_rows + s.terminal_sims + s.depthcap_sims
                - s.fan_extra_rows
        );
    }

    #[test]
    fn puct_searches_chance_games() {
        // AlphaZero guidance over the same game: stride A+1 rows, uniform prior, zeros value —
        // the visit distribution must still find the risky action's expectation edge.
        let mut infer = |_obs: Vec<f32>, n: usize| vec![0.0; n * 3];
        let mut eval = Evaluator::new(&mut infer, None);
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
