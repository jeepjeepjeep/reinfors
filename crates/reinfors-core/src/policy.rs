//! State evaluation and action selection.

use crate::codec::bytes::Reader;
use crate::encoder::StateEncoder;
use crate::game::{Game, Rng};
use crate::policies::tree::expectimax::SearchEvaluation;
use crate::reward::Reward;
use crate::rollout::engine::CollectStats;
use crate::rollout::evaluator::Evaluator;

/// Maximum simultaneous joint-action fan. Bindings reject statically oversized compositions;
/// search repeats the check against each realized legal-action product.
pub const MAX_JOINT_SLOTS: usize = 1 << 20;

/// Maximum chance fan materialized by exhaustive search modes.
pub const MAX_ENUMERATED_OUTCOMES: usize = 1 << 20;

/// How an algorithm evaluates states and acts.
pub trait Policy {
    type Evaluation;

    type PolicyState;

    /// Largest supported agent count for sequential or simultaneous dynamics. This has no default
    /// so every policy must make its capability claim deliberately.
    fn max_agents(&self, sequential: bool) -> Option<usize>;

    /// Whether sequential search consumes every agent's perspective.
    fn evaluates_all_perspectives(&self, sequential: bool, num_agents: usize) -> bool {
        let _ = (sequential, num_agents);
        false
    }

    /// Whether the policy is sound when the game state contains hidden information. This has no
    /// default so a new policy cannot acquire that soundness claim accidentally.
    fn supports_imperfect_information(&self) -> bool;

    fn begin_episode(&self, rng: &mut dyn Rng) -> Self::PolicyState;

    #[allow(clippy::too_many_arguments)]
    /// Serialize a buffered policy evaluation.
    fn encode_eval(&self, eval: &Self::Evaluation, out: &mut Vec<u8>);
    fn decode_eval(&self, r: &mut Reader, action_count: usize) -> Result<Self::Evaluation, String>;

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

    /// Choose an action from an evaluation.
    fn select(
        &self,
        eval: &Self::Evaluation,
        state: &mut Self::PolicyState,
        rng: &mut dyn Rng,
    ) -> usize;

    fn fold_telemetry(&self, eval: &Self::Evaluation, stats: &mut CollectStats) {
        let _ = (eval, stats);
    }
}

/// How tree search consumes declared chance distributions.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub enum ChanceMode {
    /// Draw a fresh outcome on every traversal. Unlike `Committed { samples: 1 }`, repeated draws
    /// converge to the chance distribution instead of freezing one biased sample.
    #[default]
    AlwaysResample,
    /// Draw and freeze `samples` outcomes at edge expansion.
    Committed { samples: usize },
    /// Materialize every outcome at edge expansion.
    ExpandAll,
}

impl ChanceMode {
    /// Whether the mode redraws on every traversal rather than only at expansion. Policies should
    /// reject unsupported modes through this property rather than matching the variant name.
    pub fn requires_repeated_traversal(&self) -> bool {
        matches!(self, ChanceMode::AlwaysResample)
    }
}

/// A policy that produces search evaluations.
pub trait SearchPolicy: Policy<Evaluation = SearchEvaluation> {
    /// Whether this search paradigm supports the chance mode.
    fn supports_chance(&self, mode: ChanceMode) -> bool;

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
        assert_eq!(argmax(&[1.0, 3.0, 3.0, 2.0]), 1);
        assert_eq!(argmax(&[5.0]), 0);
    }
}
