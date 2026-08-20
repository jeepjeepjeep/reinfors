//! Policies that search the full game state.

pub mod alphazero;
pub mod expectimax;
pub mod mcts;
pub mod minimax;

use crate::stats::CollectStats;
use expectimax::SearchEvaluation;

pub(crate) fn fold_search_stats(eval: &SearchEvaluation, stats: &mut CollectStats) {
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
