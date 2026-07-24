//! The expectimax policy family: a shared search engine (`search`) and the evaluation type it
//! produces (`SearchEvaluation`), with one expansion strategy today — `SelectiveExpectimax`
//! (best-first, budget-limited). A future exhaustive `ExpectimaxPolicy` would live alongside,
//! reusing `search`'s primitives and the same `SearchEvaluation` (at which point the shared vs
//! selective-specific split inside `search` gets drawn — driven by that second consumer).

pub mod search;

use crate::encoder::StateEncoder;
use crate::engine::CollectStats;
use crate::evaluator::Evaluator;
use crate::game::{Game, Rng};
use crate::policy::{ChanceMode, Policy, SearchPolicy};
use crate::reward::Reward;
use search::{search_many, InteriorTarget, SearchConfig, SearchStats};

/// A search's per-decision evaluation: root per-head values (for acting and the z-mix target),
/// interior MAX-node targets, and search stats (telemetry). Produced by every expectimax policy and
/// consumed by `TreeStrap` (the `learners` → `policies` edge: the producer owns the type).
pub struct SearchEvaluation {
    pub values: Vec<Vec<f64>>, // [K][A]
    /// Root per-action visit counts `[A]`, for a policy that *acts* by visit count (MCTS). Empty for
    /// searches that act by value (expectimax) — `select` falls back to `values`. Never a training
    /// target: `TreeStrap` regresses `values` (backed-up value), not visits.
    pub visits: Vec<f64>,
    /// Interior MAX-node targets — a payload for the *consuming* `TreeStrap` (it drains them
    /// into immediate records), produced here only because the search is what generates them. Empty
    /// unless the learner asked for them via `needs_interior` (threaded into `evaluate`).
    pub interior: Vec<InteriorTarget>,
    /// The root's legal action ids — acting masks to this set. `values`/`visits` are densified
    /// over the FULL action space with 0 on illegal slots, and a 0 can out-argmax all-negative
    /// legal values in a losing position, so a dense argmax is not merely wasteful but wrong.
    pub legal: Vec<usize>,
    pub stats: SearchStats,
}

/// Selective expectimax + Thompson/epsilon acting. Holds the search config, the ensemble head count
/// (for Thompson sampling + the all-terminal broadcast), and the epsilon. Whether to collect interior
/// TreeStrap targets is the paired learner's call (`needs_interior`), threaded in via `evaluate`.
pub struct SelectiveExpectimax {
    cfg: SearchConfig,
    n_heads: usize,
    epsilon: f64,
}

impl SelectiveExpectimax {
    pub fn new(cfg: SearchConfig, n_heads: usize, epsilon: f64) -> Self {
        assert!(
            Self::supports_chance_mode(cfg.chance),
            "SelectiveExpectimax expands each node exactly once (best-first) and cannot express \
             per-traversal chance modes; use Committed or ExpandAll"
        );
        SelectiveExpectimax {
            cfg,
            n_heads: n_heads.max(1),
            epsilon,
        }
    }

    /// Paradigm capability, queryable without an instance (the binding validates handles with it):
    /// an expand-once search cannot express modes that redraw per traversal.
    pub fn supports_chance_mode(mode: ChanceMode) -> bool {
        !mode.requires_repeated_traversal()
    }
}

impl Policy for SelectiveExpectimax {
    type Evaluation = SearchEvaluation;
    type PolicyState = usize; // the Thompson head for the current episode

    fn begin_episode(&self, rng: &mut dyn Rng) -> usize {
        rng.below(self.n_heads)
    }

    fn evaluate<G, F>(
        &self,
        game: &G,
        enc: &dyn StateEncoder<State = G::State>,
        reward: &dyn Reward<Event = G::Event>,
        requests: Vec<(G::State, usize)>,
        seed: u64,
        collect_interior: bool,
        eval: &mut Evaluator<'_, F>,
    ) -> Vec<SearchEvaluation>
    where
        G: Game + Sync,
        G::State: Send,
        F: FnMut(Vec<f32>, usize) -> Vec<f64>,
    {
        // Root legal sets for acting (the search densifies its values over the full space).
        let legal: Vec<Vec<usize>> = requests
            .iter()
            .map(|(state, agent)| game.legal_actions(state, *agent))
            .collect();
        // The expectimax search pools per round through its own loop; routing each pooled call
        // through the Evaluator gives it the same caching/dedup/telemetry as every other consumer.
        search_many(
            game,
            enc,
            reward,
            &self.cfg,
            requests,
            collect_interior,
            seed,
            &mut |obs, n| eval.forward(obs, n),
        )
        .into_iter()
        .zip(legal)
        .map(|((values, interior, stats), legal)| {
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
                visits: Vec::new(), // expectimax acts by value
                interior,
                legal,
                stats,
            }
        })
        .collect()
    }

    /// Thompson-head argmax over the LEGAL set (epsilon explores uniformly over it) — the
    /// densified rows carry 0 on illegal slots, which must never win the argmax.
    fn select(&self, eval: &SearchEvaluation, head: &mut usize, rng: &mut dyn Rng) -> usize {
        let k = eval.values.len();
        let row = &eval.values[(*head).min(k - 1)];
        debug_assert!(!eval.legal.is_empty());
        let mut rel = eval.legal[0];
        for &a in &eval.legal {
            if row[a] > row[rel] {
                rel = a;
            }
        }
        if self.epsilon > 0.0 && rng.unit() < self.epsilon {
            rel = eval.legal[rng.below(eval.legal.len())]; // uniform over the legal set
        }
        rel
    }

    fn fold_telemetry(&self, eval: &SearchEvaluation, stats: &mut CollectStats) {
        Self::fold_search_stats(eval, stats);
        // The expectimax extras: leaf epistemic uncertainty and root head-disagreement.
        let s = &eval.stats;
        if s.leaves > 0 {
            stats.sum_sigma += s.sigma_sum / s.leaves as f64;
        }
        stats.sum_disagreement += root_disagreement(&eval.values);
    }
}

impl SearchPolicy for SelectiveExpectimax {
    fn supports_chance(&self, mode: ChanceMode) -> bool {
        Self::supports_chance_mode(mode)
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

#[cfg(test)]
mod select_masking_tests {
    use super::*;
    use crate::rng::SplitMix64;
    use search::SearchStats;

    #[test]
    fn select_never_picks_an_illegal_zero_in_a_losing_position() {
        // Densified rows carry 0 on illegal slots. In a losing position every LEGAL value is
        // negative, so a dense argmax would "prefer" the illegal 0 — the bug class this masks.
        let policy = SelectiveExpectimax::new(
            SearchConfig {
                gamma: 1.0,
                beta: 1.0,
                expansion_budget: 4,
                top_k: 2,
                max_depth: 2,
                chance: crate::policy::ChanceMode::Committed { samples: 1 },
                opponent: search::Opponent::Uniform,
            },
            1,
            0.0,
        );
        let eval = SearchEvaluation {
            values: vec![vec![0.0, -0.6, -0.9]], // slot 0 illegal (densified zero)
            visits: Vec::new(),
            interior: Vec::new(),
            legal: vec![1, 2],
            stats: SearchStats::default(),
        };
        let mut head = 0;
        assert_eq!(
            policy.select(&eval, &mut head, &mut SplitMix64::new(0)),
            1,
            "the best LEGAL action, not the illegal densified zero"
        );
    }
}
