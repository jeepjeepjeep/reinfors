//! DQN training-record production.

use crate::encoder::ActionView;
use crate::game::Rng;
use crate::learner::{sample_mask, Learner, Step};
use crate::policies::modelfree::epsilon_greedy_q::QEvaluation;

pub struct DqnRecord {
    pub player: usize,
    pub obs: Vec<f32>,
    pub action: usize,
    pub reward: f64,
    pub next_obs: Vec<f32>,
    pub terminal: bool,
    /// gamma^k for the k own-decision steps this record's window spans; 0 when it cannot
    /// bootstrap. The caller's TD target is `reward + discount * next_value` — no caller gamma.
    pub discount: f64,
    pub mask: Vec<f32>,
    pub legal: Vec<usize>,
    pub next_legal: Vec<usize>,
}

pub struct Dqn {
    n_heads: usize,
    bootstrap_p: f64,
    n_step: usize,
    gamma: f64,
}

impl Dqn {
    pub fn new(n_heads: usize, bootstrap_p: f64, n_step: usize, gamma: f64) -> Self {
        Dqn {
            n_heads: n_heads.max(1),
            bootstrap_p,
            n_step: n_step.max(1),
            gamma,
        }
    }
}

impl Learner<QEvaluation> for Dqn {
    type Record = DqnRecord;

    fn needs_next_obs(&self) -> bool {
        true
    }

    fn eval_records(
        &self,
        _eval: &mut QEvaluation,
        _view: &dyn ActionView,
        _agent: usize,
        _rng: &mut dyn Rng,
    ) -> Vec<DqnRecord> {
        Vec::new()
    }

    fn episode_records(
        &self,
        trajectory: &[Step<QEvaluation>],
        _tail: &[f64],
        view: &dyn ActionView,
        agent: usize,
        rng: &mut dyn Rng,
    ) -> Vec<DqnRecord> {
        let to_head = |ids: &[usize]| ids.iter().map(|&x| view.head_index(x, agent)).collect();
        let len = trajectory.len();
        (0..len)
            .map(|t| {
                let s = &trajectory[t];
                // Window over this agent's OWN decisions t..e, shortened at the episode tail.
                let e = (t + self.n_step).min(len);
                let mut reward = 0.0;
                let mut discount = 1.0;
                for step in &trajectory[t..e] {
                    reward += discount * step.reward;
                    discount *= self.gamma;
                }
                let last = &trajectory[e - 1];
                // Bootstrap from this agent's next decision, not an intervening opponent turn.
                let (next_obs, next_legal): (Vec<f32>, Vec<usize>) = match trajectory.get(e) {
                    Some(succ) => (succ.obs.clone(), to_head(&succ.evaluation.legal)),
                    None => (last.next_obs.clone(), to_head(&last.next_legal)),
                };
                let next_legal = if last.terminal {
                    Vec::new()
                } else {
                    next_legal
                };
                DqnRecord {
                    player: agent,
                    obs: s.obs.clone(),
                    action: view.head_index(s.action, agent),
                    reward,
                    next_obs,
                    terminal: last.terminal,
                    discount: if next_legal.is_empty() { 0.0 } else { discount },
                    mask: sample_mask(rng, self.n_heads, self.bootstrap_p),
                    legal: to_head(&s.evaluation.legal),
                    next_legal,
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoder::IdentityView;
    use crate::game::{Actor, Game, Transition};
    use crate::policies::modelfree::epsilon_greedy_q::EpsilonGreedyQ;
    use crate::rng::SplitMix64;
    use crate::rollout::engine::Engine;

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
                legal: vec![0, 1],
            },
            action,
            reward,
            next_obs,
            next_legal: Vec::new(),
            terminal,
        }
    }

    #[test]
    fn episode_records_emit_one_masked_transition_per_step() {
        let learner = Dqn::new(3, 1.0, 1, 0.99);
        let traj = vec![
            step(vec![1.0], 0, 0.5, vec![2.0], false),
            step(vec![2.0], 1, -1.0, vec![], true),
        ];
        let recs = learner.episode_records(&traj, &[], &IdentityView, 0, &mut SplitMix64::new(1));
        assert_eq!(recs.len(), 2);
        assert_eq!(recs[0].obs, vec![1.0]);
        assert_eq!(recs[0].action, 0);
        assert_eq!(recs[0].reward, 0.5);
        assert_eq!(recs[0].next_obs, vec![2.0]);
        assert!(!recs[0].terminal);
        assert_eq!(recs[0].mask, vec![1.0, 1.0, 1.0]);
        assert!(recs[1].terminal);
    }

    fn trunc_step(obs: Vec<f32>, reward: f64, next_obs: Vec<f32>) -> Step<QEvaluation> {
        let mut s = step(obs, 0, reward, next_obs, false);
        s.next_legal = vec![0, 1];
        s
    }

    #[test]
    fn nstep_windows_sum_hand_derived_returns() {
        // gamma 0.5, n 2, rewards [1, 2, 4], terminal end:
        //   t0: 1 + 0.5*2 = 2, bootstraps from t2's obs with discount 0.25
        //   t1: 2 + 0.5*4 = 4, terminal tail -> no bootstrap
        //   t2: 4, terminal -> no bootstrap
        let learner = Dqn::new(1, 1.0, 2, 0.5);
        let traj = vec![
            step(vec![1.0], 0, 1.0, vec![2.0], false),
            step(vec![2.0], 1, 2.0, vec![3.0], false),
            step(vec![3.0], 0, 4.0, vec![], true),
        ];
        let recs = learner.episode_records(&traj, &[], &IdentityView, 0, &mut SplitMix64::new(1));
        assert_eq!(recs[0].reward, 2.0);
        assert_eq!(recs[0].next_obs, vec![3.0]);
        assert_eq!(recs[0].discount, 0.25);
        assert!(!recs[0].terminal);
        assert_eq!(recs[1].reward, 4.0);
        assert!(recs[1].terminal);
        assert_eq!(recs[1].discount, 0.0);
        assert!(recs[1].next_legal.is_empty());
        assert_eq!(recs[2].reward, 4.0);
        assert_eq!(recs[2].discount, 0.0);
    }

    #[test]
    fn nstep_truncation_tails_shorten_with_matching_discounts() {
        // gamma 0.5, n 3, truncated after 3 steps: every window ends at the truncated state,
        // bootstrapping with gamma^k for the k own decisions actually spanned.
        let learner = Dqn::new(1, 1.0, 3, 0.5);
        let traj = vec![
            trunc_step(vec![1.0], 1.0, vec![9.0]),
            trunc_step(vec![2.0], 2.0, vec![9.0]),
            trunc_step(vec![3.0], 4.0, vec![9.0]),
        ];
        let recs = learner.episode_records(&traj, &[], &IdentityView, 0, &mut SplitMix64::new(1));
        assert_eq!(recs[0].reward, 1.0 + 0.5 * 2.0 + 0.25 * 4.0);
        assert_eq!(recs[0].discount, 0.125);
        assert_eq!(recs[1].reward, 2.0 + 0.5 * 4.0);
        assert_eq!(recs[1].discount, 0.25);
        assert_eq!(recs[2].reward, 4.0);
        assert_eq!(recs[2].discount, 0.5);
        for r in &recs {
            assert_eq!(
                r.next_obs,
                vec![9.0],
                "all tails bootstrap from the truncated state"
            );
            assert_eq!(r.next_legal, vec![0, 1]);
        }
    }

    #[test]
    fn nstep_larger_than_the_episode_degrades_to_monte_carlo() {
        let learner = Dqn::new(1, 1.0, 10, 0.5);
        let traj = vec![
            step(vec![1.0], 0, 1.0, vec![2.0], false),
            step(vec![2.0], 1, 2.0, vec![], true),
        ];
        let recs = learner.episode_records(&traj, &[], &IdentityView, 0, &mut SplitMix64::new(1));
        assert_eq!(recs[0].reward, 1.0 + 0.5 * 2.0);
        assert_eq!(recs[0].discount, 0.0);
        assert_eq!(recs[1].reward, 2.0);
    }

    #[test]
    fn one_step_matches_the_original_semantics_with_discount_gamma() {
        let learner = Dqn::new(1, 1.0, 1, 0.9);
        let traj = vec![
            trunc_step(vec![1.0], 1.0, vec![9.0]),
            trunc_step(vec![2.0], 2.0, vec![9.0]),
        ];
        let recs = learner.episode_records(&traj, &[], &IdentityView, 0, &mut SplitMix64::new(1));
        assert_eq!(recs[0].reward, 1.0);
        assert_eq!(
            recs[0].next_obs,
            vec![2.0],
            "interior bootstraps from the next decision"
        );
        assert_eq!(recs[0].discount, 0.9);
        assert_eq!(recs[1].discount, 0.9);
    }

    #[test]
    fn learner_declares_transition_needs() {
        let learner = Dqn::new(2, 0.5, 1, 0.99);
        assert!(learner.needs_next_obs());
        assert!(!learner.uses_episode_tail());
        assert!(learner
            .eval_records(
                &mut QEvaluation {
                    values: vec![vec![0.0; 2]],
                    legal: vec![0, 1],
                },
                &IdentityView,
                0,
                &mut SplitMix64::new(0)
            )
            .is_empty());
    }

    // Compile-time policy/learner compatibility check.
    fn _assert_seam_composes() -> Option<Engine<DummyGame, EpsilonGreedyQ, Dqn>> {
        None
    }

    #[allow(dead_code)]
    struct DummyGame;
    impl Game for DummyGame {
        type State = ();
        type Event = ();
        fn num_agents(&self) -> usize {
            1
        }
        fn action_count(&self) -> usize {
            2
        }
        fn actor(&self, _: &()) -> Actor {
            Actor::Agent(0)
        }
        fn legal_actions(&self, _: &(), _: usize) -> Vec<usize> {
            vec![0, 1]
        }
        fn step(&self, _: &(), _: &[usize]) -> Transition<(), ()> {
            unimplemented!()
        }
        fn initial_state(&self) {}
    }
}
