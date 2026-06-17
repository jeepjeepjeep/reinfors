//! The expectimax policy family: a shared search engine (`search`) and the evaluation type it
//! produces (`SearchEvaluation`), with one expansion strategy today — `SelectiveExpectimaxPolicy`
//! (best-first, budget-limited). A future exhaustive `ExpectimaxPolicy` would live alongside,
//! reusing `search`'s primitives and the same `SearchEvaluation` (at which point the shared vs
//! selective-specific split inside `search` gets drawn — driven by that second consumer).

pub mod search;

use crate::engine::CollectStats;
use crate::game::{Game, Rng};
use crate::policy::{argmax, Policy};
use search::{search_many, InteriorTarget, SearchConfig, SearchStats};

/// A search's per-decision evaluation: root per-head values (for acting and the z-mix target),
/// interior MAX-node targets (immediate records), and search stats (telemetry). Produced by every
/// expectimax policy and consumed by `TreeStrapLearner`.
pub struct SearchEvaluation {
    pub values: Vec<Vec<f64>>, // [K][A]
    pub interior: Vec<InteriorTarget>,
    pub stats: SearchStats,
}

/// Selective expectimax + Thompson/epsilon acting. Holds the search config, the interior-target flag,
/// the ensemble head count (for Thompson sampling + the all-terminal broadcast), and the epsilon.
pub struct SelectiveExpectimaxPolicy {
    cfg: SearchConfig,
    collect_interior: bool,
    n_heads: usize,
    epsilon: f64,
}

impl SelectiveExpectimaxPolicy {
    pub fn new(cfg: SearchConfig, collect_interior: bool, n_heads: usize, epsilon: f64) -> Self {
        SelectiveExpectimaxPolicy {
            cfg,
            collect_interior,
            n_heads: n_heads.max(1),
            epsilon,
        }
    }
}

impl Policy for SelectiveExpectimaxPolicy {
    type Evaluation = SearchEvaluation;
    type PolicyState = usize; // the Thompson head for the current episode

    fn begin_episode(&self, rng: &mut dyn Rng) -> usize {
        rng.below(self.n_heads)
    }

    fn evaluate<G, F>(
        &self,
        game: &G,
        requests: Vec<(G::State, usize)>,
        seed: u64,
        infer: &mut F,
    ) -> Vec<SearchEvaluation>
    where
        G: Game + Sync,
        G::State: Send,
        F: FnMut(Vec<f32>, usize) -> Vec<f64>,
    {
        search_many(
            game,
            &self.cfg,
            requests,
            self.collect_interior,
            seed,
            infer,
        )
        .into_iter()
        .map(|(values, interior, stats)| {
            // A search whose root children are all terminal evaluates no leaves, so it cannot infer
            // the head count and returns a single (head-agnostic) row. Broadcast it to `n_heads` so
            // every emitted target is `[n_heads][A]`. Searches that evaluated leaves already return
            // `[n_heads][A]`, so this is a no-op for them.
            let values = if values.len() < self.n_heads {
                vec![values[0].clone(); self.n_heads]
            } else {
                values
            };
            SearchEvaluation {
                values,
                interior,
                stats,
            }
        })
        .collect()
    }

    fn select(&self, eval: &SearchEvaluation, head: &mut usize, rng: &mut dyn Rng) -> usize {
        let k = eval.values.len();
        let mut rel = argmax(&eval.values[(*head).min(k - 1)]);
        if self.epsilon > 0.0 && rng.unit() < self.epsilon {
            rel = rng.below(eval.values[0].len()); // uniform over the action space
        }
        rel
    }

    fn fold_telemetry(&self, eval: &SearchEvaluation, stats: &mut CollectStats) {
        let s = &eval.stats;
        stats.max_depth = stats.max_depth.max(s.max_depth);
        stats.sum_leaves += s.leaves as f64;
        stats.sum_rounds += s.rounds as f64;
        stats.sum_expansions += s.expansions as f64;
        if s.leaves > 0 {
            stats.sum_sigma += s.sigma_sum / s.leaves as f64;
        }
        stats.sum_disagreement += root_disagreement(&eval.values);
    }
}

/// Root head-disagreement: the per-action population std across heads of the root values `[K][A]`,
/// averaged over actions (`values.std(axis=0).mean()` in snake_RL). 0 with fewer than two heads.
fn root_disagreement(values: &[Vec<f64>]) -> f64 {
    let k = values.len();
    if k < 2 || values[0].is_empty() {
        return 0.0;
    }
    let a = values[0].len();
    let inv_k = 1.0 / k as f64;
    let total: f64 = (0..a)
        .map(|ai| {
            let mean = values.iter().map(|h| h[ai]).sum::<f64>() * inv_k;
            let var = values.iter().map(|h| (h[ai] - mean).powi(2)).sum::<f64>() * inv_k;
            var.sqrt()
        })
        .sum();
    total / a as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn root_disagreement_matches_population_std_definition() {
        // Single action so the per-action std is the whole metric: heads [0, 2] -> mean 1, std 1.
        assert!((root_disagreement(&[vec![0.0], vec![2.0]]) - 1.0).abs() < 1e-12);
        // Identical heads disagree by zero; a single head has no spread.
        assert_eq!(root_disagreement(&[vec![5.0, 5.0], vec![5.0, 5.0]]), 0.0);
        assert_eq!(root_disagreement(&[vec![1.0, 2.0, 3.0]]), 0.0);
    }
}
