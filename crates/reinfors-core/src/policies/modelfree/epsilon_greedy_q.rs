//! Epsilon-greedy acting on ensemble Q-values.

use std::collections::HashMap;

use crate::codec::bytes::Reader;
use crate::encoder::{head_permutation, StateEncoder};
use crate::game::{Game, Rng};
use crate::policy::Policy;
use crate::reward::Reward;
use crate::rollout::evaluator::Evaluator;

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
        if v as usize >= self.n_heads {
            return Err(format!(
                "Thompson head {v} out of range for {} heads",
                self.n_heads
            ));
        }
        Ok(v as usize)
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
    ) -> Vec<QEvaluation>
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
        let q = eval.forward(&players, obs_flat, n);
        let k = q.len() / (n * a);
        // Build each action-frame permutation once rather than dispatching per Q-value.
        let mut perms: HashMap<usize, (Vec<usize>, bool)> = HashMap::new();
        requests
            .iter()
            .enumerate()
            .map(|(i, (state, agent))| {
                let (perm, identity) = perms
                    .entry(*agent)
                    .or_insert_with(|| head_permutation(enc, a, *agent));
                let values = (0..k)
                    .map(|h| {
                        let start = (i * k + h) * a;
                        if *identity {
                            q[start..start + a].to_vec()
                        } else {
                            perm.iter().map(|&p| q[start + p]).collect()
                        }
                    })
                    .collect();
                QEvaluation {
                    values,
                    legal: game.legal_actions(state, *agent),
                }
            })
            .collect()
    }

    fn evaluate_seeded<G, F>(
        &self,
        game: &G,
        enc: &dyn StateEncoder<State = G::State>,
        reward: &dyn Reward<Event = G::Event>,
        requests: Vec<(G::State, usize)>,
        _seeds: &[u64],
        collect_interior: bool,
        eval: &mut Evaluator<'_, F>,
    ) -> Result<Vec<QEvaluation>, String>
    where
        G: Game + Sync,
        G::State: Send,
        F: FnMut(usize, Vec<f32>, usize) -> Vec<f64>,
    {
        Ok(self.evaluate(game, enc, reward, requests, 0, collect_interior, eval))
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
