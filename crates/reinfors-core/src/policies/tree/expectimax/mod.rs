//! Selective expectimax policy and shared search evaluation.

pub mod search;

use crate::codec::bytes::Reader;
use crate::game::{Game, Rng};
use crate::policies::tree::fold_search_stats;
use crate::policy::{thompson_head_from_u64, ChanceMode, Policy};
use crate::rollout::engine::CollectStats;
use search::{SearchConfig, SearchStats};

/// Per-decision values, visits, auxiliary targets, legality, and telemetry.
pub struct SearchEvaluation {
    pub values: Vec<Vec<f64>>,
    pub visits: Vec<f64>,
    pub legal: Vec<usize>,
    pub stats: SearchStats,
}

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

    pub fn supports_chance_mode(mode: ChanceMode) -> bool {
        !mode.requires_repeated_traversal()
    }
}

impl Policy for SelectiveExpectimax {
    type Evaluation = SearchEvaluation;
    type PolicyState = usize;

    fn supports_imperfect_information(&self) -> bool {
        false
    }

    fn max_agents(&self, _sequential: bool) -> Option<usize> {
        None
    }

    fn encode_eval(&self, eval: &SearchEvaluation, out: &mut Vec<u8>) {
        encode_search_eval(eval, out);
    }

    fn decode_eval(&self, r: &mut Reader, action_count: usize) -> Result<SearchEvaluation, String> {
        decode_search_eval(r, action_count, self.n_heads, false)
    }

    fn policy_state_to_u64(&self, s: &usize) -> u64 {
        *s as u64
    }

    fn policy_state_from_u64(&self, v: u64) -> Result<usize, String> {
        thompson_head_from_u64(v, self.n_heads)
    }

    fn begin_episode(&self, rng: &mut dyn Rng) -> usize {
        rng.below(self.n_heads)
    }

    type Search<S: Send> = search::MultiStepper<S>;

    fn begin_search<G: Game + Sync>(
        &self,
        ctx: crate::policy::SearchCtx<'_, G>,
        state: &G::State,
        perspectives: &[usize],
    ) -> Self::Search<G::State>
    where
        G::State: Send,
    {
        search::guard_game(ctx.game);
        search::MultiStepper::new(
            perspectives
                .iter()
                .map(|&agent| search::Stepper::new(state.clone(), agent, ctx.rng.next_u64()))
                .collect(),
        )
    }

    fn round<G: Game + Sync>(
        &self,
        ctx: crate::policy::SearchCtx<'_, G>,
        search: &mut Self::Search<G::State>,
        out: &mut crate::policy::RequestSink,
    ) -> crate::policy::RoundStatus
    where
        G::State: Send,
    {
        search::multi_round(search, ctx.game, ctx.enc, ctx.reward, &self.cfg, out)
    }

    fn absorb<S: Send>(
        &self,
        search: &mut Self::Search<S>,
        rows: crate::policy::RowsView<'_>,
        _rng: &mut dyn Rng,
    ) {
        search.absorb(rows);
    }

    fn finish<G: Game + Sync>(
        &self,
        ctx: crate::policy::SearchCtx<'_, G>,
        search: Self::Search<G::State>,
    ) -> Vec<(SearchEvaluation, Vec<crate::learner::InteriorTarget>)>
    where
        G::State: Send,
    {
        search
            .steppers
            .into_iter()
            .map(|st| {
                let agent = st.agent();
                let legal = ctx.game.legal_actions(st.root_state(), agent);
                let (values, interior, stats) =
                    search::stepper_finish(st, ctx.game, ctx.enc, &self.cfg, ctx.collect_interior);
                let values = if values.len() < self.n_heads {
                    vec![values[0].clone(); self.n_heads]
                } else {
                    values
                };
                (
                    SearchEvaluation {
                        values,
                        visits: Vec::new(),
                        legal,
                        stats,
                    },
                    interior,
                )
            })
            .collect()
    }

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
            rel = eval.legal[rng.below(eval.legal.len())];
        }
        rel
    }

    fn fold_telemetry(&self, eval: &SearchEvaluation, stats: &mut CollectStats) {
        fold_search_stats(eval, stats);
        let s = &eval.stats;
        if s.leaves > 0 {
            stats.sum_sigma += s.sigma_sum / s.leaves as f64;
        }
        stats.sum_disagreement += root_disagreement(&eval.values, &eval.legal);
    }
}

fn root_disagreement(values: &[Vec<f64>], legal: &[usize]) -> f64 {
    // Illegal densified zeros would otherwise dilute disagreement on sparse-action games.
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
        assert!((root_disagreement(&[vec![0.0], vec![2.0]], &[0]) - 1.0).abs() < 1e-12);
        assert_eq!(
            root_disagreement(&[vec![5.0, 5.0], vec![5.0, 5.0]], &[0, 1]),
            0.0
        );
        assert_eq!(root_disagreement(&[vec![1.0, 2.0, 3.0]], &[0, 1, 2]), 0.0);
        let heads = [vec![0.0, 0.0], vec![0.0, 2.0]];
        assert!((root_disagreement(&heads, &[1]) - 1.0).abs() < 1e-12);
        assert!((root_disagreement(&heads, &[0, 1]) - 0.5).abs() < 1e-12);
    }
}

#[cfg(test)]
mod select_masking_tests {
    use super::*;
    use crate::rng::SplitMix64;
    use search::SearchStats;

    #[test]
    fn select_never_picks_an_illegal_zero_in_a_losing_position() {
        let policy = SelectiveExpectimax::new(
            SearchConfig {
                gamma: 1.0,
                beta: 1.0,
                expansion_budget: 4,
                top_k: 2,
                max_depth: 2,
                chance: ChanceMode::Committed { samples: 1 },
                opponent: search::Opponent::Uniform,
            },
            1,
            0.0,
        );
        let eval = SearchEvaluation {
            values: vec![vec![0.0, -0.6, -0.9]],
            visits: Vec::new(),
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

/// Serialize a buffered evaluation after immediate interior targets have been drained.
pub(crate) fn encode_search_eval(e: &SearchEvaluation, out: &mut Vec<u8>) {
    use crate::codec::bytes::*;
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

/// Decode and validate the shape required by the consuming policy and learner.
pub(crate) fn decode_search_eval(
    r: &mut Reader,
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
    let mut stats = SearchStats {
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
        legal,
        stats,
    })
}

#[cfg(test)]
mod eval_codec_tests {
    use super::*;
    use Reader;

    fn encoded(heads: usize, visits: bool, actions: usize) -> Vec<u8> {
        let e = SearchEvaluation {
            values: vec![vec![0.0; actions]; heads],
            visits: if visits {
                vec![0.0; actions]
            } else {
                Vec::new()
            },
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
        assert!(err(&encoded(0, true, a), 1, true).contains("value heads"));
        assert!(err(&encoded(1, false, a), 1, true).contains("visit vector width"));
        assert!(err(&encoded(1, true, a), 1, false).contains("no visit counts"));
        decode_search_eval(&mut Reader::new(&encoded(1, true, a)), a, 1, true)
            .map(|_| ())
            .unwrap();
    }
}
