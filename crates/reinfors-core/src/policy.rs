//! The acting seam: a `Policy` *evaluates* the options (search / forward / MCTS) into an `Evaluation`,
//! then *selects* an action from it. Concrete policies live in `crate::policies`; the `Engine` drives
//! any of them, and a `Learner` consuming the matching `Evaluation` produces the training records.

use crate::codec::bytes::Reader;
use crate::encoder::StateEncoder;
use crate::game::{Game, Rng};
use crate::policies::tree::expectimax::SearchEvaluation;
use crate::reward::Reward;
use crate::rollout::engine::CollectStats;
use crate::rollout::evaluator::Evaluator;

/// Upper bound on one node's simultaneous joint fan — MCTS/AZ's dense joint-slot arrays
/// (`∏ per-agent legal widths`) and expectimax's per-edge co-mover branch product
/// (`∏ co-mover legal widths`) both honor it. The binding rejects compositions whose static
/// worst case exceeds it (a config error); the searches check the realized, state-dependent
/// products as backstops — unchecked, the product overflows `usize` (65 two-action agents wrap
/// to a silently empty fan) or attempts absurd allocations long before that.
pub const MAX_JOINT_SLOTS: usize = 1 << 20;

/// Upper bound on the outcomes an `ExpandAll` chance fan will ENUMERATE. Sampling modes draw
/// single indices from a [`ChanceDist`](crate::ChanceDist) at any size; only exhaustive fanning
/// pays per-outcome cost, and an exact fan past this bound is an error (ExpandAll's contract is
/// exactness — use a sampling mode for combinatorial outcome spaces).
pub const MAX_ENUMERATED_OUTCOMES: usize = 1 << 20;

/// How an algorithm evaluates states and acts.
pub trait Policy {
    type Evaluation;

    type PolicyState;

    /// The largest `Game::num_agents` this policy can plan for under the game's dynamics
    /// (`sequential` = the game's decisions are `Actor::Agent` turns, probed from the initial
    /// state; `false` = simultaneous), or `None` if agent-count-agnostic there. Capability can
    /// depend on dynamics — UCT-guided MCTS searches simultaneous games at any N but sequential
    /// ones only to 2 (Q-derived leaf values need the evaluated agent's own decision points).
    /// Checked at construction (the binding turns a violation into a config error; `Engine::new`
    /// asserts as the direct-core backstop) — never mid-collect. Required rather than defaulted:
    /// a capability claim must be deliberate, and running an unsupported agent count doesn't fail
    /// loudly, it silently computes wrong values (e.g. negamax past two players).
    fn max_agents(&self, sequential: bool) -> Option<usize>;

    /// Whether this policy's search consumes EVERY agent's perspective at sequential decision
    /// points (Max^N). The engine then buffers value-only steps for non-movers (when the paired
    /// learner opts in) and bootstraps every perspective's truncation tail — the emission
    /// principle: supervised perspectives ≡ consumed perspectives. Default: no (negamax and
    /// non-search policies read only the mover's row).
    fn evaluates_all_perspectives(&self, sequential: bool, num_agents: usize) -> bool {
        let _ = (sequential, num_agents);
        false
    }

    /// Whether this policy is sound on games with HIDDEN state (`Game::perfect_information()`
    /// = false). True only for policies that consume nothing beyond each agent's own
    /// observation (the DQN family); tree searches branch on the true state and are clairvoyant
    /// there, so they must say false. Required rather than defaulted: a soundness claim must be
    /// deliberate. Checked at construction — never mid-collect.
    fn supports_imperfect_information(&self) -> bool;

    fn begin_episode(&self, rng: &mut dyn Rng) -> Self::PolicyState;

    /// Pooled evaluation of a batch of active `(state, agent)` requests against the engine's
    /// [`Evaluator`] — the sole route to the net, which pools each round's requests into one
    /// batched forward and transparently applies the optional infer cache. `reward` lets a
    /// searching policy value the in-tree immediate rewards (the engine's per-step reward source);
    /// non-search policies ignore it.
    #[allow(clippy::too_many_arguments)]
    /// Serialize/deserialize this policy's per-decision evaluation — buffered `Step`s carry them,
    /// so exact engine snapshots need them portable. Decode validates (untrusted-bytes boundary).
    fn encode_eval(&self, eval: &Self::Evaluation, out: &mut Vec<u8>);
    fn decode_eval(&self, r: &mut Reader, action_count: usize) -> Result<Self::Evaluation, String>;

    /// The per-episode acting state as a plain integer (Thompson head / temperature ply) — every
    /// current policy's state fits; a future richer state would widen this seam.
    fn policy_state_to_u64(&self, s: &Self::PolicyState) -> u64;
    fn policy_state_from_u64(&self, v: u64) -> Result<Self::PolicyState, String>;

    #[allow(clippy::too_many_arguments)]
    fn evaluate<G, F>(
        &self,
        game: &G,
        enc: &dyn StateEncoder<State = G::State>,
        reward: &dyn Reward<Event = G::Event>,
        requests: Vec<(G::State, usize)>,
        seed: u64,
        collect_interior: bool,
        eval: &mut Evaluator<'_, F>,
    ) -> Vec<Self::Evaluation>
    where
        G: Game + Sync,
        G::State: Send,
        F: FnMut(usize, Vec<f32>, usize) -> Vec<f64>;

    /// Choose an action from an evaluation, using the game's per-episode state and acting RNG.
    fn select(
        &self,
        eval: &Self::Evaluation,
        state: &mut Self::PolicyState,
        rng: &mut dyn Rng,
    ) -> usize;

    /// Fold this decision's diagnostics into the rollout telemetry.
    fn fold_telemetry(&self, eval: &Self::Evaluation, stats: &mut CollectStats) {
        let _ = (eval, stats);
    }
}

/// How the tree search consumes a chance state's declared [`Game::chance_node`]
/// distribution. The *game seam* is one thing — the distribution, declared;
/// this enum is the *search policy* over it, and the right mode is decided by one ratio:
/// **simulations-per-chance-edge vs. fan width** (the number of outcomes).
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

impl ChanceMode {
    /// Whether this mode is defined by *repeated traversal* — redrawing on every pass through the
    /// edge. Sampled-trajectory searches (MCTS/PUCT) walk root-to-leaf once per simulation and can
    /// express it; expand-once searches (best-first expectimax) build each node exactly once and
    /// cannot. Policies reject by THIS property, not by mode name, so new modes classify
    /// themselves and existing validations stay correct.
    pub fn requires_repeated_traversal(&self) -> bool {
        matches!(self, ChanceMode::AlwaysResample)
    }
}

/// The search subset of [`Policy`]: simulation/search-based policies, distinguished by the
/// compiler-checked contract that they produce [`SearchEvaluation`]s (root values, visits, search
/// stats). Non-search policies (e.g. `EpsilonGreedyQ`) simply do not implement it — the absence of
/// the capability IS the distinction; there is no `NonSearchPolicy`.
pub trait SearchPolicy: Policy<Evaluation = SearchEvaluation> {
    /// Whether this search paradigm can express `mode` (see
    /// [`ChanceMode::requires_repeated_traversal`]). Checked when a configuration is built — the
    /// binding turns a `false` into a construction error — never mid-collect.
    fn supports_chance(&self, mode: ChanceMode) -> bool;

    /// Fold the search diagnostics common to every search family into the collect telemetry;
    /// policies layer their extras on top in their `Policy::fold_telemetry`.
    fn fold_search_stats(eval: &SearchEvaluation, stats: &mut CollectStats) {
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
        stats.sum_extra_eval_rows += s.extra_eval_rows;
    }
}

pub(crate) fn argmax(values: &[f64]) -> usize {
    let mut best = 0;
    for (i, &v) in values.iter().enumerate() {
        if v > values[best] {
            best = i;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argmax_takes_the_first_max() {
        assert_eq!(argmax(&[1.0, 3.0, 3.0, 2.0]), 1); // ties go to the earliest index
        assert_eq!(argmax(&[5.0]), 0);
    }
}
