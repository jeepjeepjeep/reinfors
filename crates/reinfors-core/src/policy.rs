//! The `Policy` trait — the acting seam: *evaluate* the options (search / forward / MCTS) into an
//! `Evaluation`, then *select* an action from it. `SelectiveExpectimaxPolicy` (selective expectimax +
//! Thompson-head/epsilon-greedy acting) is the first impl. The Engine owns the generic rollout
//! substrate, the Policy owns evaluation + action choice, and the `Learner` owns record production.

use crate::algo::SearchEvaluation;
use crate::engine::CollectStats;
use crate::game::{Game, Rng};
use crate::search::{search_many, SearchConfig};

/// How an algorithm evaluates states and acts. Non-generic so `evaluate` can be method-generic over
/// the game, avoiding inference ambiguity.
pub trait Policy {
    /// The per-decision evaluation `evaluate` produces (the assessment of the options — *not* a chosen
    /// action). Consumed by `select` and by the paired `Learner`.
    type Evaluation;
    /// The policy's transient per-episode state, freshly drawn at episode start (e.g. the Thompson
    /// head it acts greedily under). `()` for stateless policies.
    type PolicyState;

    /// Fresh per-episode policy state at a game's episode start (redrawn on reset).
    fn begin_episode(&self, rng: &mut dyn Rng) -> Self::PolicyState;

    /// Pooled evaluation of a batch of active `(state, agent)` requests with the live net (`infer`):
    /// one batched forward per round, shared across games (the throughput win). `seed` seeds the
    /// search's environment-chance sampling so it is reproducible; the per-game *acting* RNG is not
    /// touched here (it is used only in `select`). Returns one `Evaluation` per request.
    fn evaluate<G, F>(
        &self,
        game: &G,
        requests: Vec<(G::State, usize)>,
        seed: u64,
        infer: &mut F,
    ) -> Vec<Self::Evaluation>
    where
        G: Game + Sync,
        G::State: Send,
        F: FnMut(Vec<f32>, usize) -> Vec<f64>;

    /// Choose an action from an evaluation, using the game's per-episode state and acting RNG.
    fn select(
        &self,
        eval: &Self::Evaluation,
        state: &mut Self::PolicyState,
        rng: &mut dyn Rng,
    ) -> usize;

    /// Fold this decision's diagnostics into the rollout telemetry. Default: nothing (a policy with no
    /// search stats); a search policy contributes depth/leaves/sigma/disagreement.
    fn fold_telemetry(&self, eval: &Self::Evaluation, stats: &mut CollectStats) {
        let _ = (eval, stats);
    }
}

/// Selective expectimax + Thompson/epsilon acting — today's algorithm. Holds the search config, the
/// interior-target flag, the ensemble head count (for Thompson sampling + the all-terminal broadcast),
/// and the exploration epsilon.
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

fn argmax(values: &[f64]) -> usize {
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
    fn root_disagreement_matches_population_std_definition() {
        // Single action so the per-action std is the whole metric: heads [0, 2] -> mean 1, std 1.
        assert!((root_disagreement(&[vec![0.0], vec![2.0]]) - 1.0).abs() < 1e-12);
        // Identical heads disagree by zero; a single head has no spread.
        assert_eq!(root_disagreement(&[vec![5.0, 5.0], vec![5.0, 5.0]]), 0.0);
        assert_eq!(root_disagreement(&[vec![1.0, 2.0, 3.0]]), 0.0);
    }

    #[test]
    fn argmax_takes_the_first_max() {
        assert_eq!(argmax(&[1.0, 3.0, 3.0, 2.0]), 1); // ties go to the earliest index
        assert_eq!(argmax(&[5.0]), 0);
    }
}
