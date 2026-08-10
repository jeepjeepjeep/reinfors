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
    pub mask: Vec<f32>,
    pub legal: Vec<usize>,
    pub next_legal: Vec<usize>,
}

pub struct Dqn {
    n_heads: usize,
    bootstrap_p: f64,
}

impl Dqn {
    pub fn new(n_heads: usize, bootstrap_p: f64) -> Self {
        Dqn {
            n_heads: n_heads.max(1),
            bootstrap_p,
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
        trajectory
            .iter()
            .enumerate()
            .map(|(t, s)| {
                // Bootstrap from this agent's next decision, not an intervening opponent turn.
                let (next_obs, next_legal) = match trajectory.get(t + 1) {
                    Some(succ) => (succ.obs.clone(), to_head(&succ.evaluation.legal)),
                    None => (s.next_obs.clone(), to_head(&s.next_legal)),
                };
                DqnRecord {
                    player: agent,
                    obs: s.obs.clone(),
                    action: view.head_index(s.action, agent),
                    reward: s.reward,
                    next_obs,
                    terminal: s.terminal,
                    mask: sample_mask(rng, self.n_heads, self.bootstrap_p),
                    legal: to_head(&s.evaluation.legal),
                    next_legal: if s.terminal { Vec::new() } else { next_legal },
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
        let learner = Dqn::new(3, 1.0);
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

    #[test]
    fn learner_declares_transition_needs() {
        let learner = Dqn::new(2, 0.5);
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
