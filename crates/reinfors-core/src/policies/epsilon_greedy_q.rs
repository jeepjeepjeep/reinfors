//! Epsilon-greedy acting on per-head Q-values. `evaluate` is a plain batched network forward (no
//! search), and `select` is a Thompson-head epsilon-greedy choice. Its `QEvaluation` is the seam's
//! non-search case; the matching `Dqn` (in `crate::learners::dqn`) consumes it into transitions.

use crate::encoder::StateEncoder;
use crate::evaluator::Evaluator;
use crate::game::{Game, Rng};
use crate::policy::Policy;
use crate::reward::Reward;

/// DQN's per-decision evaluation: just the per-head Q-values `[K][A]` from one network forward (no
/// search tree, interior targets, or stats — the seam's non-search case).
pub struct QEvaluation {
    pub values: Vec<Vec<f64>>,
    /// The state's legal action ids — acting and the learner's TD targets must stay inside this
    /// set on sparse-action games (an illegal argmax acts on a phantom Q value).
    pub legal: Vec<usize>,
}

/// Bootstrapped-DQN acting: a batched forward, then a Thompson-head epsilon-greedy choice.
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
    type PolicyState = usize; // the Thompson head for the current episode

    fn begin_episode(&self, rng: &mut dyn Rng) -> usize {
        rng.below(self.n_heads)
    }

    fn evaluate<G, F>(
        &self,
        game: &G,
        enc: &dyn StateEncoder<State = G::State>,
        _reward: &dyn Reward<Event = G::Event>, // model-free: rewards come from the env, not the search
        requests: Vec<(G::State, usize)>,
        _seed: u64,
        _collect_interior: bool, // DQN has no interior targets — a plain forward, nothing to collect
        eval: &mut Evaluator<'_, F>,
    ) -> Vec<QEvaluation>
    where
        G: Game + Sync,
        G::State: Send,
        F: FnMut(Vec<f32>, usize) -> Vec<f64>,
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
        let q = eval.forward(obs_flat, n); // flat [n, K, A]
        let k = q.len() / (n * a);
        requests
            .iter()
            .enumerate()
            .map(|(i, (state, agent))| {
                let values = (0..k)
                    .map(|h| {
                        let start = (i * k + h) * a;
                        q[start..start + a].to_vec()
                    })
                    .collect();
                QEvaluation {
                    values,
                    legal: game.legal_actions(state, *agent),
                }
            })
            .collect()
    }

    /// Thompson-head argmax over the LEGAL set (epsilon explores uniformly over it): identical to
    /// the dense behavior on all-legal games, correct instead of phantom-Q-driven on sparse ones.
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
        // No epsilon: pick the argmax of the chosen head. Head 1's best action is index 2.
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
