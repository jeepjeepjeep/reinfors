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

    fn supports_imperfect_information(&self) -> bool {
        false // the search branches on the true state (clairvoyant past hidden information)
    }

    fn max_agents(&self, _sequential: bool) -> Option<usize> {
        // Single-perspective search at any N under either dynamics: each other agent is modeled
        // chance (sequential — a node per foreign turn; simultaneous — a factored co-mover
        // joint), and only the searcher's values ever back up, so no per-agent value plumbing
        // is needed.
        None
    }

    fn encode_eval(&self, eval: &SearchEvaluation, out: &mut Vec<u8>) {
        crate::policies::expectimax::encode_search_eval(eval, out);
    }

    fn decode_eval(
        &self,
        r: &mut crate::codec::bytes::Reader,
        action_count: usize,
    ) -> Result<SearchEvaluation, String> {
        // expectimax evaluations are always [n_heads][A] (broadcast at the search seam) and
        // carry no visits (acting is by value)
        crate::policies::expectimax::decode_search_eval(r, action_count, self.n_heads, false)
    }

    fn policy_state_to_u64(&self, s: &usize) -> u64 {
        *s as u64
    }

    fn policy_state_from_u64(&self, v: u64) -> Result<usize, String> {
        if v as usize >= self.n_heads {
            return Err(format!(
                "Thompson head {v} out of range for {} heads",
                self.n_heads
            ));
        }
        Ok(v as usize)
    }

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
        stats.sum_disagreement += root_disagreement(&eval.values, &eval.legal);
    }
}

impl SearchPolicy for SelectiveExpectimax {
    fn supports_chance(&self, mode: ChanceMode) -> bool {
        Self::supports_chance_mode(mode)
    }
}

/// Root head-disagreement: the per-action population std across heads of the root values `[K][A]`,
/// averaged over actions (`values.std(axis=0).mean()` in snake_RL). 0 with fewer than two heads.
fn root_disagreement(values: &[Vec<f64>], legal: &[usize]) -> f64 {
    // Averaged over the LEGAL actions only: the densified rows carry identical zeros on illegal
    // slots (std 0 across heads), which would dilute the mean — on chess by ~133x (35 of 4672).
    let k = values.len();
    if k < 2 || values[0].is_empty() || legal.is_empty() {
        return 0.0;
    }
    let inv_k = 1.0 / k as f64;
    let total: f64 = legal
        .iter()
        .map(|&ai| {
            let mean = values.iter().map(|h| h[ai]).sum::<f64>() * inv_k;
            let var = values.iter().map(|h| (h[ai] - mean).powi(2)).sum::<f64>() * inv_k;
            var.sqrt()
        })
        .sum();
    total / legal.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn root_disagreement_matches_population_std_definition() {
        // Single action so the per-action std is the whole metric: heads [0, 2] -> mean 1, std 1.
        assert!((root_disagreement(&[vec![0.0], vec![2.0]], &[0]) - 1.0).abs() < 1e-12);
        // Identical heads disagree by zero; a single head has no spread.
        assert_eq!(
            root_disagreement(&[vec![5.0, 5.0], vec![5.0, 5.0]], &[0, 1]),
            0.0
        );
        assert_eq!(root_disagreement(&[vec![1.0, 2.0, 3.0]], &[0, 1, 2]), 0.0);
        // Legal-only averaging: a densified illegal zero (std 0) must not dilute the mean —
        // heads disagree by std 1 on the single legal action; the illegal slot is excluded.
        let heads = [vec![0.0, 0.0], vec![0.0, 2.0]];
        assert!((root_disagreement(&heads, &[1]) - 1.0).abs() < 1e-12);
        assert!((root_disagreement(&heads, &[0, 1]) - 0.5).abs() < 1e-12); // the dilution, shown
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

/// Shared `SearchEvaluation` (de)serialization for the search-family policies' snapshot seams.
/// `interior` is always drained at decision time (`eval_records` moves it out), so buffered
/// evaluations never carry it — encoding asserts that and decoding restores it empty.
pub(crate) fn encode_search_eval(e: &SearchEvaluation, out: &mut Vec<u8>) {
    use crate::codec::bytes::*;
    debug_assert!(
        e.interior.is_empty(),
        "interior is drained before buffering"
    );
    put_u32(out, e.values.len() as u32);
    for row in &e.values {
        put_f64s(out, row);
    }
    put_f64s(out, &e.visits);
    put_usizes(out, &e.legal);
    let st = &e.stats;
    put_i64(out, i64::from(st.max_depth));
    for v in [st.expansions, st.leaves, st.rounds] {
        put_u64(out, v as u64);
    }
    put_f64(out, st.sigma_sum);
    for v in [
        st.terminal_sims,
        st.depthcap_sims,
        st.shared_rows,
        st.fresh_rows,
        st.hit_rows,
        st.extra_eval_rows,
    ] {
        put_u64(out, v as u64);
    }
}

/// Decoding is POLICY-SPECIFIC: each policy passes the evaluation shape its learners consume —
/// `expected_heads` (TreeStrap indexes `values[0]` and z-mixes `z[h]` across every step, so head
/// counts must be exact and uniform) and whether `visits` is full-width (the AZ family's π
/// source) or empty (expectimax acts by value and buffers none). A permissive shared decoder let
/// restored evaluations violate those assumptions and panic at flush time.
pub(crate) fn decode_search_eval(
    r: &mut crate::codec::bytes::Reader,
    action_count: usize,
    expected_heads: usize,
    expect_visits: bool,
) -> Result<SearchEvaluation, String> {
    use crate::codec::bytes::*;
    let k = r.u32()? as usize;
    if k != expected_heads {
        return Err(format!(
            "evaluation has {k} value heads; this policy requires {expected_heads}"
        ));
    }
    let values = (0..k).map(|_| f64s(r)).collect::<Result<Vec<_>, _>>()?;
    for row in &values {
        if row.len() != action_count {
            return Err(format!(
                "value row width {} != action count {action_count}",
                row.len()
            ));
        }
        if row.iter().any(|v| !v.is_finite()) {
            return Err("non-finite search value in evaluation".into());
        }
    }
    let visits = f64s(r)?;
    if expect_visits {
        if visits.len() != action_count {
            return Err(format!(
                "visit vector width {} != action count {action_count}",
                visits.len()
            ));
        }
    } else if !visits.is_empty() {
        return Err(format!(
            "this policy buffers no visit counts, got {}",
            visits.len()
        ));
    }
    if visits.iter().any(|v| !v.is_finite() || *v < 0.0) {
        return Err("invalid visit count".into());
    }
    let legal = usizes(r)?;
    if legal.iter().any(|&a| a >= action_count) {
        return Err("legal action id out of range".into());
    }
    let mut stats = crate::policies::expectimax::search::SearchStats {
        max_depth: i32::try_from(r.i64()?).map_err(|_| "max_depth out of range".to_string())?,
        ..Default::default()
    };
    stats.expansions = r.u64()? as usize;
    stats.leaves = r.u64()? as usize;
    stats.rounds = r.u64()? as usize;
    stats.sigma_sum = r.f64()?;
    stats.terminal_sims = r.u64()? as usize;
    stats.depthcap_sims = r.u64()? as usize;
    stats.shared_rows = r.u64()? as usize;
    stats.fresh_rows = r.u64()? as usize;
    stats.hit_rows = r.u64()? as usize;
    stats.extra_eval_rows = r.u64()? as usize;
    Ok(SearchEvaluation {
        values,
        visits,
        interior: Vec::new(),
        legal,
        stats,
    })
}

#[cfg(test)]
mod eval_codec_tests {
    use super::*;
    use crate::codec::bytes::Reader;

    fn encoded(heads: usize, visits: bool, actions: usize) -> Vec<u8> {
        let e = SearchEvaluation {
            values: vec![vec![0.0; actions]; heads],
            visits: if visits {
                vec![0.0; actions]
            } else {
                Vec::new()
            },
            interior: Vec::new(),
            legal: vec![0, 1],
            stats: Default::default(),
        };
        let mut out = Vec::new();
        encode_search_eval(&e, &mut out);
        out
    }

    #[test]
    fn decoding_enforces_the_policy_shape() {
        let a = 4;
        // the expectimax shape round-trips only under expectimax expectations
        let ex = encoded(2, false, a);
        decode_search_eval(&mut Reader::new(&ex), a, 2, false)
            .map(|_| ())
            .unwrap();
        let err = |bytes: &[u8], heads: usize, visits: bool| {
            decode_search_eval(&mut Reader::new(bytes), a, heads, visits)
                .map(|_| ())
                .unwrap_err()
        };
        assert!(err(&ex, 1, true).contains("value heads"));
        assert!(err(&ex, 3, false).contains("value heads"));
        // zero heads can never decode (TreeStrap indexes values[0] at flush)
        assert!(err(&encoded(0, true, a), 1, true).contains("value heads"));
        // the AZ/MCTS shape requires full-width visits (the learner's pi source)...
        assert!(err(&encoded(1, false, a), 1, true).contains("visit vector width"));
        // ...while expectimax buffers none
        assert!(err(&encoded(1, true, a), 1, false).contains("no visit counts"));
        decode_search_eval(&mut Reader::new(&encoded(1, true, a)), a, 1, true)
            .map(|_| ())
            .unwrap();
    }
}
