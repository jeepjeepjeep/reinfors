//! Epsilon-greedy acting on ensemble Q-values.

use crate::codec::bytes::Reader;
use crate::game::{Game, Rng};
use crate::policy::{thompson_head_from_u64, Policy};

/// Per-head Q-values and legal actions for one decision.
pub struct QEvaluation {
    pub values: Vec<Vec<f64>>,
    pub legal: Vec<usize>,
}

pub struct EpsilonGreedyQ {
    n_heads: usize,
    epsilon: f64,
}

impl EpsilonGreedyQ {
    pub fn new(n_heads: usize, epsilon: f64) -> Self {
        EpsilonGreedyQ {
            n_heads: n_heads.max(1),
            epsilon,
        }
    }
}

impl Policy for EpsilonGreedyQ {
    type Evaluation = QEvaluation;
    type PolicyState = usize;

    fn max_agents(&self, _sequential: bool) -> Option<usize> {
        None
    }

    fn supports_imperfect_information(&self) -> bool {
        true
    }

    fn begin_episode(&self, rng: &mut dyn Rng) -> usize {
        rng.below(self.n_heads)
    }

    fn encode_eval(&self, eval: &QEvaluation, out: &mut Vec<u8>) {
        use crate::codec::bytes::*;
        put_u32(out, eval.values.len() as u32);
        for row in &eval.values {
            put_f64s(out, row);
        }
        put_usizes(out, &eval.legal);
    }

    fn decode_eval(&self, r: &mut Reader, action_count: usize) -> Result<QEvaluation, String> {
        use crate::codec::bytes::*;
        let k = r.u32()? as usize;
        if k != self.n_heads {
            return Err(format!(
                "evaluation has {k} heads, policy has {}",
                self.n_heads
            ));
        }
        let values = (0..k).map(|_| f64s(r)).collect::<Result<Vec<_>, _>>()?;
        for row in &values {
            if row.len() != action_count {
                return Err(format!(
                    "Q row width {} != action count {action_count}",
                    row.len()
                ));
            }
            if row.iter().any(|v| !v.is_finite()) {
                return Err("non-finite Q value in evaluation".into());
            }
        }
        let legal = usizes(r)?;
        if legal.iter().any(|&a| a >= action_count) {
            return Err("legal action id out of range".into());
        }
        Ok(QEvaluation { values, legal })
    }

    fn policy_state_to_u64(&self, s: &usize) -> u64 {
        *s as u64
    }

    fn policy_state_from_u64(&self, v: u64) -> Result<usize, String> {
        thompson_head_from_u64(v, self.n_heads)
    }

    type Search<S: Send> = super::OneShot<S>;

    fn begin_search<G: Game + Sync>(
        &self,
        ctx: crate::policy::SearchCtx<'_, G>,
        state: &G::State,
        perspectives: &[usize],
    ) -> Self::Search<G::State>
    where
        G::State: Send,
    {
        super::one_shot_begin(&ctx, state, perspectives)
    }

    fn round<G: Game + Sync>(
        &self,
        _ctx: crate::policy::SearchCtx<'_, G>,
        search: &mut Self::Search<G::State>,
        out: &mut crate::policy::RequestSink,
    ) -> crate::policy::RoundStatus
    where
        G::State: Send,
    {
        super::one_shot_round(search, out)
    }

    fn absorb<S: Send>(
        &self,
        search: &mut Self::Search<S>,
        rows: crate::policy::RowsView<'_>,
        _rng: &mut dyn Rng,
    ) {
        super::one_shot_absorb(search, rows);
    }

    fn finish<G: Game + Sync>(
        &self,
        ctx: crate::policy::SearchCtx<'_, G>,
        search: Self::Search<G::State>,
    ) -> Vec<(QEvaluation, Vec<crate::learner::InteriorTarget>)>
    where
        G::State: Send,
    {
        let a = ctx.game.action_count();
        let k = if search.agents.is_empty() {
            0
        } else {
            search.stride / a
        };
        search
            .agents
            .iter()
            .zip(search.legal)
            .enumerate()
            .map(|(i, (&agent, legal))| {
                let (perm, identity) = ctx.perms.get(agent);
                let values = (0..k)
                    .map(|h| {
                        let start = i * search.stride + h * a;
                        if identity {
                            search.rows[start..start + a].to_vec()
                        } else {
                            perm.iter().map(|&p| search.rows[start + p]).collect()
                        }
                    })
                    .collect();
                (QEvaluation { values, legal }, Vec::new())
            })
            .collect()
    }

    fn select(&self, eval: &QEvaluation, head: &mut usize, rng: &mut dyn Rng) -> usize {
        let k = eval.values.len();
        let row = &eval.values[(*head).min(k - 1)];
        debug_assert!(
            !eval.legal.is_empty(),
            "select on a state with no legal actions"
        );
        let mut best = eval.legal[0];
        for &a in &eval.legal {
            if row[a] > row[best] {
                best = a;
            }
        }
        if self.epsilon > 0.0 && rng.unit() < self.epsilon {
            best = eval.legal[rng.below(eval.legal.len())];
        }
        best
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::SplitMix64;

    #[test]
    fn select_is_thompson_head_argmax_then_epsilon() {
        let policy = EpsilonGreedyQ::new(2, 0.0);
        let eval = QEvaluation {
            values: vec![vec![3.0, 1.0, 2.0], vec![0.0, 1.0, 5.0]],
            legal: vec![0, 1, 2],
        };
        let mut head = 1;
        assert_eq!(policy.select(&eval, &mut head, &mut SplitMix64::new(0)), 2);
        let mut head0 = 0;
        assert_eq!(policy.select(&eval, &mut head0, &mut SplitMix64::new(0)), 0);
    }
}
