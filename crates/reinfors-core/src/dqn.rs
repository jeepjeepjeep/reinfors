//! Model-free bootstrapped DQN — a second algorithm validating the `Policy`/`Learner` seam. Unlike
//! selective-expectimax TreeStrap it does no search (`evaluate` is a plain batched network forward),
//! acts epsilon-greedily under a Thompson-sampled head, and its records are off-policy transitions
//! `(s, a, r, s', done)` — bootstrapped by the (Python) learner against a target net, not episode-end
//! z-mixed targets. This exercises the seam's two hardest generalizations: a non-search `Evaluation`
//! and a transition `Record` (a different shape from TreeStrap's `(obs, target, mask)`).

use crate::algo::{sample_mask, Learner, Step};
use crate::game::{Game, Rng};
use crate::policy::Policy;

/// DQN's per-decision evaluation: just the per-head Q-values `[K][A]` from one network forward (no
/// search tree, interior targets, or stats — the seam's non-search case).
pub struct QEvaluation {
    pub values: Vec<Vec<f64>>,
}

/// Bootstrapped-DQN acting: a batched forward, then a Thompson-head epsilon-greedy choice.
pub struct DqnPolicy {
    n_heads: usize,
    epsilon: f64,
}

impl DqnPolicy {
    pub fn new(n_heads: usize, epsilon: f64) -> Self {
        DqnPolicy {
            n_heads: n_heads.max(1),
            epsilon,
        }
    }
}

impl Policy for DqnPolicy {
    type Evaluation = QEvaluation;
    type PolicyState = usize; // the Thompson head for the current episode

    fn begin_episode(&self, rng: &mut dyn Rng) -> usize {
        rng.below(self.n_heads)
    }

    fn evaluate<G, F>(
        &self,
        game: &G,
        requests: Vec<(G::State, usize)>,
        _seed: u64,
        infer: &mut F,
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
            obs_flat.extend(game.observe(state, *agent));
        }
        let q = infer(obs_flat, n); // flat [n, K, A]
        let k = q.len() / (n * a);
        (0..n)
            .map(|i| {
                let values = (0..k)
                    .map(|h| {
                        let start = (i * k + h) * a;
                        q[start..start + a].to_vec()
                    })
                    .collect();
                QEvaluation { values }
            })
            .collect()
    }

    fn select(&self, eval: &QEvaluation, head: &mut usize, rng: &mut dyn Rng) -> usize {
        let k = eval.values.len();
        let mut rel = argmax(&eval.values[(*head).min(k - 1)]);
        if self.epsilon > 0.0 && rng.unit() < self.epsilon {
            rel = rng.below(eval.values[0].len());
        }
        rel
    }
}

/// One off-policy transition: `(obs, action, reward, next_obs, terminal, mask[K])`. The TD target is
/// computed in the (Python) learner from `next_obs`/`terminal` against a target net, so the engine
/// emits raw transitions rather than precomputed targets. `terminal` is true only at a real terminal
/// (a horizon truncation keeps it false, so the learner bootstraps from `next_obs`).
pub struct Transition {
    pub obs: Vec<f32>,
    pub action: usize,
    pub reward: f64,
    pub next_obs: Vec<f32>,
    pub terminal: bool,
    pub mask: Vec<f32>,
}

/// Bootstrapped-DQN record production: one per-head-masked transition per step, no episode-end mixing.
pub struct DqnLearner {
    n_heads: usize,
    bootstrap_p: f64,
}

impl DqnLearner {
    pub fn new(n_heads: usize, bootstrap_p: f64) -> Self {
        DqnLearner {
            n_heads: n_heads.max(1),
            bootstrap_p,
        }
    }
}

impl Learner<QEvaluation> for DqnLearner {
    type Record = Transition;

    fn needs_next_obs(&self) -> bool {
        true
    }

    fn eval_records(&self, _eval: &mut QEvaluation, _rng: &mut dyn Rng) -> Vec<Transition> {
        Vec::new() // nothing at decision time; transitions need the post-step s', formed at episode end
    }

    fn episode_records(
        &self,
        trajectory: &[Step<QEvaluation>],
        _tail: &[f64],
        rng: &mut dyn Rng,
    ) -> Vec<Transition> {
        trajectory
            .iter()
            .map(|s| Transition {
                obs: s.obs.clone(),
                action: s.action,
                reward: s.reward,
                next_obs: s.next_obs.clone(),
                terminal: s.terminal,
                mask: sample_mask(rng, self.n_heads, self.bootstrap_p),
            })
            .collect()
    }
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
    use crate::rng::SplitMix64;

    fn step(
        obs: Vec<f32>,
        action: usize,
        reward: f64,
        next_obs: Vec<f32>,
        terminal: bool,
    ) -> Step<QEvaluation> {
        Step {
            obs,
            evaluation: QEvaluation {
                values: vec![vec![0.0; 2]],
            },
            action,
            reward,
            next_obs,
            terminal,
        }
    }

    #[test]
    fn episode_records_emit_one_masked_transition_per_step() {
        let learner = DqnLearner::new(3, 1.0); // bootstrap_p = 1 -> all-ones masks
        let traj = vec![
            step(vec![1.0], 0, 0.5, vec![2.0], false),
            step(vec![2.0], 1, -1.0, vec![], true), // terminal: next_obs unused
        ];
        let recs = learner.episode_records(&traj, &[], &mut SplitMix64::new(1));
        assert_eq!(recs.len(), 2);
        assert_eq!(recs[0].obs, vec![1.0]);
        assert_eq!(recs[0].action, 0);
        assert_eq!(recs[0].reward, 0.5);
        assert_eq!(recs[0].next_obs, vec![2.0]);
        assert!(!recs[0].terminal);
        assert_eq!(recs[0].mask, vec![1.0, 1.0, 1.0]);
        assert!(recs[1].terminal);
    }

    #[test]
    fn learner_declares_transition_needs() {
        let learner = DqnLearner::new(2, 0.5);
        assert!(learner.needs_next_obs());
        assert!(!learner.uses_episode_tail());
        assert!(learner
            .eval_records(
                &mut QEvaluation {
                    values: vec![vec![0.0; 2]]
                },
                &mut SplitMix64::new(0)
            )
            .is_empty());
    }

    #[test]
    fn select_is_thompson_head_argmax_then_epsilon() {
        // No epsilon: pick the argmax of the chosen head. Head 1's best action is index 2.
        let policy = DqnPolicy::new(2, 0.0);
        let eval = QEvaluation {
            values: vec![vec![3.0, 1.0, 2.0], vec![0.0, 1.0, 5.0]],
        };
        let mut head = 1;
        assert_eq!(policy.select(&eval, &mut head, &mut SplitMix64::new(0)), 2);
        let mut head0 = 0;
        assert_eq!(policy.select(&eval, &mut head0, &mut SplitMix64::new(0)), 0);
    }

    // Compile-only: a DQN policy + learner share the QEvaluation, so they satisfy the engine coupling.
    fn _assert_seam_composes() -> Option<crate::engine::Engine<DummyGame, DqnPolicy, DqnLearner>> {
        None
    }

    #[allow(dead_code)] // used only as a type parameter in the compile-only assertion above
    struct DummyGame;
    impl Game for DummyGame {
        type State = ();
        fn num_agents(&self) -> usize {
            1
        }
        fn action_count(&self) -> usize {
            2
        }
        fn obs_shape(&self) -> (usize, usize, usize) {
            (1, 1, 1)
        }
        fn actor(&self, _: &()) -> crate::game::Actor {
            crate::game::Actor::Agent(0)
        }
        fn legal_actions(&self, _: &(), _: usize) -> Vec<usize> {
            vec![0, 1]
        }
        fn step(&self, _: &(), _: &[usize]) -> crate::game::Transition<()> {
            unimplemented!()
        }
        fn observe(&self, _: &(), _: usize) -> Vec<f32> {
            vec![0.0]
        }
        fn initial_state(&self, _: &mut dyn Rng) {}
    }
}
