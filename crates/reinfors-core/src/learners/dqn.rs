//! Model-free bootstrapped DQN — records half. Emits off-policy transitions `(s, a, r, s', done)`,
//! bootstrapped by the (Python) learner against a target net rather than episode-end z-mixed. Consumes
//! `QEvaluation` from `crate::policies::dqn`. Together these exercise the seam's two hardest cases: a
//! non-search evaluation and a transition record (a different shape from TreeStrap's targets).

use crate::game::Rng;
use crate::learner::{sample_mask, Learner, Step};
use crate::policies::dqn::QEvaluation;

/// One off-policy transition: `(obs, action, reward, next_obs, terminal, mask[K])`. The TD target is
/// computed in the (Python) learner from `next_obs`/`terminal` against a target net, so the engine
/// emits raw transitions rather than precomputed targets. `terminal` is true only at a real terminal
/// (a horizon truncation keeps it false, so the learner bootstraps from `next_obs`).
pub struct DqnRecord {
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
    type Record = DqnRecord;

    fn needs_next_obs(&self) -> bool {
        true
    }

    fn eval_records(&self, _eval: &mut QEvaluation, _rng: &mut dyn Rng) -> Vec<DqnRecord> {
        Vec::new() // nothing at decision time; transitions need the post-step s', formed at episode end
    }

    fn episode_records(
        &self,
        trajectory: &[Step<QEvaluation>],
        _tail: &[f64],
        rng: &mut dyn Rng,
    ) -> Vec<DqnRecord> {
        trajectory
            .iter()
            .map(|s| DqnRecord {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policies::dqn::DqnPolicy;
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

    // Compile-only: a DQN policy + learner share QEvaluation, so they satisfy the engine coupling.
    fn _assert_seam_composes() -> Option<crate::engine::Engine<DummyGame, DqnPolicy, DqnLearner>> {
        None
    }

    #[allow(dead_code)] // used only as a type parameter in the compile-only assertion above
    struct DummyGame;
    impl crate::game::Game for DummyGame {
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
