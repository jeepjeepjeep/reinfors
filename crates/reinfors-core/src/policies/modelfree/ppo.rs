//! Stochastic actor over policy logits and a state value.

use crate::codec::bytes::Reader;
use crate::game::{Game, Rng};
use crate::policy::Policy;

/// Masked-softmax log-probabilities parallel to `legal`, the critic's state value, and the
/// game-frame legal action ids for one decision.
pub struct PpoEvaluation {
    pub log_probs: Vec<f64>,
    pub value: f64,
    pub legal: Vec<usize>,
}

/// Log-probabilities of the masked softmax over `legal`, parallel to `legal`. Sampling and
/// the recorded behavior log-prob both come from here so they cannot drift.
pub fn masked_log_probs(logits: &[f64], legal: &[usize]) -> Vec<f64> {
    let max = legal
        .iter()
        .map(|&a| logits[a])
        .fold(f64::NEG_INFINITY, f64::max);
    let log_sum = legal
        .iter()
        .map(|&a| (logits[a] - max).exp())
        .sum::<f64>()
        .ln();
    legal.iter().map(|&a| logits[a] - max - log_sum).collect()
}

pub struct PpoActor;

impl PpoActor {
    pub fn new() -> Self {
        PpoActor
    }
}

impl Default for PpoActor {
    fn default() -> Self {
        PpoActor::new()
    }
}

impl Policy for PpoActor {
    type Evaluation = PpoEvaluation;
    type PolicyState = ();

    fn max_agents(&self, _sequential: bool) -> Option<usize> {
        None
    }

    fn supports_imperfect_information(&self) -> bool {
        true
    }

    fn begin_episode(&self, _rng: &mut dyn Rng) {}

    fn encode_eval(&self, eval: &PpoEvaluation, out: &mut Vec<u8>) {
        use crate::codec::bytes::*;
        put_f64s(out, &eval.log_probs);
        put_f64(out, eval.value);
        put_usizes(out, &eval.legal);
    }

    fn decode_eval(&self, r: &mut Reader, action_count: usize) -> Result<PpoEvaluation, String> {
        use crate::codec::bytes::*;
        let log_probs = f64s(r)?;
        let value = r.f64()?;
        if log_probs.iter().any(|v| !v.is_finite()) || !value.is_finite() {
            return Err("non-finite log-prob or value in evaluation".into());
        }
        let legal = usizes(r)?;
        if legal.len() != log_probs.len() {
            return Err(format!(
                "log-prob row width {} != legal count {}",
                log_probs.len(),
                legal.len()
            ));
        }
        if legal.iter().any(|&a| a >= action_count) {
            return Err("legal action id out of range".into());
        }
        Ok(PpoEvaluation {
            log_probs,
            value,
            legal,
        })
    }

    fn policy_state_to_u64(&self, _s: &()) -> u64 {
        0
    }

    fn policy_state_from_u64(&self, v: u64) -> Result<(), String> {
        if v != 0 {
            return Err(format!("the PPO actor carries no policy state, got {v}"));
        }
        Ok(())
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
    ) -> Vec<(PpoEvaluation, Vec<crate::learner::InteriorTarget>)>
    where
        G::State: Send,
    {
        let a = ctx.game.action_count();
        debug_assert!(
            search.agents.is_empty() || search.stride == a + 1,
            "PolicyValue row width {} != {}",
            search.stride,
            a + 1
        );
        search
            .agents
            .iter()
            .zip(search.legal)
            .enumerate()
            .map(|(i, (&agent, legal))| {
                let (perm, identity) = ctx.perms.get(agent);
                let row = &search.rows[i * search.stride..(i + 1) * search.stride];
                let log_probs = if identity {
                    masked_log_probs(&row[..a], &legal)
                } else {
                    let head_legal: Vec<usize> = legal.iter().map(|&g| perm[g]).collect();
                    masked_log_probs(&row[..a], &head_legal)
                };
                (
                    PpoEvaluation {
                        log_probs,
                        value: row[a],
                        legal,
                    },
                    Vec::new(),
                )
            })
            .collect()
    }

    fn select(&self, eval: &PpoEvaluation, _state: &mut (), rng: &mut dyn Rng) -> usize {
        debug_assert!(
            !eval.legal.is_empty(),
            "select on a state with no legal actions"
        );
        let log_probs = &eval.log_probs;
        let mut r = rng.unit();
        for (i, lp) in log_probs.iter().enumerate() {
            r -= lp.exp();
            if r <= 0.0 {
                return eval.legal[i];
            }
        }
        // Rounding exhausted the unit draw: fall back to the modal legal action.
        let mut best = 0;
        for (i, lp) in log_probs.iter().enumerate() {
            if *lp > log_probs[best] {
                best = i;
            }
        }
        eval.legal[best]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::SplitMix64;

    #[test]
    fn masked_log_probs_normalize_over_legal_only() {
        // The huge illegal logit must not influence the distribution.
        let lp = masked_log_probs(&[1.0, 50.0, 1.0, 0.0], &[0, 2, 3]);
        let total: f64 = lp.iter().map(|l| l.exp()).sum();
        assert!((total - 1.0).abs() < 1e-12);
        assert!((lp[0] - lp[1]).abs() < 1e-12, "equal logits, equal mass");
        assert!(lp[2] < lp[0]);
    }

    #[test]
    fn selection_frequencies_follow_the_masked_softmax() {
        let eval = PpoEvaluation {
            // exp(2)/[exp(2)+exp(0)] ~ 0.881 on action 0 among legal {0, 2}.
            log_probs: masked_log_probs(&[2.0, 9.0, 0.0], &[0, 2]),
            value: 0.0,
            legal: vec![0, 2],
        };
        let policy = PpoActor::new();
        let mut rng = SplitMix64::new(7);
        let mut counts = [0usize; 3];
        for _ in 0..20_000 {
            counts[policy.select(&eval, &mut (), &mut rng)] += 1;
        }
        assert_eq!(counts[1], 0, "illegal action never sampled");
        let frac = counts[0] as f64 / 20_000.0;
        assert!((frac - 0.8808).abs() < 0.01, "got {frac}");
    }

    #[test]
    fn eval_codec_round_trips_and_validates() {
        let policy = PpoActor::new();
        let eval = PpoEvaluation {
            log_probs: masked_log_probs(&[0.5, -1.0, 2.0], &[0, 2]),
            value: 0.25,
            legal: vec![0, 2],
        };
        let mut buf = Vec::new();
        policy.encode_eval(&eval, &mut buf);
        let back = policy.decode_eval(&mut Reader::new(&buf), 3).unwrap();
        assert_eq!(back.log_probs, eval.log_probs);
        assert_eq!(back.value, eval.value);
        assert_eq!(back.legal, eval.legal);
        assert!(policy.decode_eval(&mut Reader::new(&buf), 2).is_err());
    }
}
