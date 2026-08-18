//! Stochastic actor over policy logits and a state value.

use std::collections::HashMap;

use crate::codec::bytes::Reader;
use crate::encoder::{head_permutation, StateEncoder};
use crate::game::{Game, Rng};
use crate::policy::Policy;
use crate::reward::Reward;
use crate::rollout::evaluator::Evaluator;

/// Game-frame policy logits, the critic's state value, and legal actions for one decision.
pub struct PpoEvaluation {
    pub logits: Vec<f64>,
    pub value: f64,
    pub legal: Vec<usize>,
}

/// Log-probabilities of the masked softmax over `legal`, parallel to `legal`.
/// Both action sampling and the recorded behavior log-probability derive from this one
/// function so the acting distribution and the stored ratio denominator cannot drift.
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
        put_f64s(out, &eval.logits);
        put_f64(out, eval.value);
        put_usizes(out, &eval.legal);
    }

    fn decode_eval(&self, r: &mut Reader, action_count: usize) -> Result<PpoEvaluation, String> {
        use crate::codec::bytes::*;
        let logits = f64s(r)?;
        if logits.len() != action_count {
            return Err(format!(
                "logit row width {} != action count {action_count}",
                logits.len()
            ));
        }
        let value = r.f64()?;
        if logits.iter().any(|v| !v.is_finite()) || !value.is_finite() {
            return Err("non-finite logit or value in evaluation".into());
        }
        let legal = usizes(r)?;
        if legal.iter().any(|&a| a >= action_count) {
            return Err("legal action id out of range".into());
        }
        Ok(PpoEvaluation {
            logits,
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

    fn evaluate<G, F>(
        &self,
        game: &G,
        enc: &dyn StateEncoder<State = G::State>,
        _reward: &dyn Reward<Event = G::Event>,
        requests: Vec<(G::State, usize)>,
        _seed: u64,
        _collect_interior: bool,
        eval: &mut Evaluator<'_, F>,
    ) -> Vec<PpoEvaluation>
    where
        G: Game + Sync,
        G::State: Send,
        F: FnMut(usize, Vec<f32>, usize) -> Vec<f64>,
    {
        let n = requests.len();
        if n == 0 {
            return Vec::new();
        }
        let a = game.action_count();
        let mut obs_flat: Vec<f32> = Vec::new();
        for (state, agent) in &requests {
            obs_flat.extend(enc.encode(state, *agent));
        }
        let players: Vec<usize> = requests.iter().map(|(_, agent)| *agent).collect();
        // PolicyValue rows: `a` head-frame logits then the state value.
        let rows = eval.forward(&players, obs_flat, n);
        let width = rows.len() / n;
        debug_assert!(width == a + 1, "PolicyValue row width {width} != {}", a + 1);
        let mut perms: HashMap<usize, (Vec<usize>, bool)> = HashMap::new();
        requests
            .iter()
            .enumerate()
            .map(|(i, (state, agent))| {
                let (perm, identity) = perms
                    .entry(*agent)
                    .or_insert_with(|| head_permutation(enc, a, *agent));
                let row = &rows[i * width..(i + 1) * width];
                let logits = if *identity {
                    row[..a].to_vec()
                } else {
                    perm.iter().map(|&p| row[p]).collect()
                };
                PpoEvaluation {
                    logits,
                    value: row[a],
                    legal: game.legal_actions(state, *agent),
                }
            })
            .collect()
    }

    fn select(&self, eval: &PpoEvaluation, _state: &mut (), rng: &mut dyn Rng) -> usize {
        debug_assert!(
            !eval.legal.is_empty(),
            "select on a state with no legal actions"
        );
        let log_probs = masked_log_probs(&eval.logits, &eval.legal);
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
        // Illegal slot 1 carries a huge logit that must not influence the distribution.
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
            logits: vec![2.0, 9.0, 0.0],
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
            logits: vec![0.5, -1.0, 2.0],
            value: 0.25,
            legal: vec![0, 2],
        };
        let mut buf = Vec::new();
        policy.encode_eval(&eval, &mut buf);
        let back = policy.decode_eval(&mut Reader::new(&buf), 3).unwrap();
        assert_eq!(back.logits, eval.logits);
        assert_eq!(back.value, eval.value);
        assert_eq!(back.legal, eval.legal);
        assert!(policy.decode_eval(&mut Reader::new(&buf), 4).is_err());
    }
}
