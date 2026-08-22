//! AlphaZero training-record production.

use crate::encoder::{head_permutation, ActionView};
use crate::game::Rng;
use crate::learner::{InteriorTarget, Learner, Step};
use crate::policies::tree::expectimax::SearchEvaluation;

/// `(observation, policy target, value target, policy weight, player, legal head-frame actions)`.
pub type AlphaZeroRecord = (Vec<f32>, Vec<f64>, f64, f64, usize, Vec<usize>);

pub struct AlphaZeroLearner {
    pub gamma: f64,
}

impl AlphaZeroLearner {
    pub fn new(gamma: f64) -> Self {
        AlphaZeroLearner { gamma }
    }
}

fn normalized_visits(visits: &[f64]) -> Vec<f64> {
    let total: f64 = visits.iter().sum();
    // A real root totals num_simulations - 1 visits, positive because construction requires at
    // least two simulations. Zero therefore unambiguously marks a value-only record.
    if total <= 0.0 {
        return vec![0.0; visits.len()];
    }
    visits.iter().map(|&v| v / total).collect()
}

impl Learner<SearchEvaluation> for AlphaZeroLearner {
    type Record = AlphaZeroRecord;

    fn uses_episode_tail(&self) -> bool {
        true
    }

    fn tail_from_row(
        &self,
        row: &[f64],
        action_count: usize,
        _legal: &[usize],
        _view: &dyn ActionView,
        _agent: usize,
    ) -> Vec<f64> {
        crate::learner::policy_value_tail(row, action_count)
    }

    fn value_only_evaluation(&self, action_count: usize) -> Option<SearchEvaluation> {
        Some(SearchEvaluation {
            values: vec![vec![0.0; action_count]],
            visits: vec![0.0; action_count],
            legal: Vec::new(),
            stats: Default::default(),
        })
    }

    fn eval_records(
        &self,
        _evaluation: &SearchEvaluation,
        _targets: Vec<InteriorTarget>,
        _view: &dyn ActionView,
        _agent: usize,
        _rng: &mut dyn Rng,
    ) -> Vec<Self::Record> {
        Vec::new()
    }

    fn episode_records(
        &self,
        trajectory: &[Step<SearchEvaluation>],
        tail: &[f64],
        view: &dyn ActionView,
        agent: usize,
        _rng: &mut dyn Rng,
    ) -> Vec<Self::Record> {
        let mut z = tail.first().copied().unwrap_or(0.0);
        // Search results use game action ids; records supervise encoder head ids.
        let a = trajectory.first().map_or(0, |s| s.evaluation.visits.len());
        let (perm, identity) = head_permutation(view, a, agent);
        let mut out: Vec<AlphaZeroRecord> = Vec::with_capacity(trajectory.len());
        for step in trajectory.iter().rev() {
            z = step.reward + self.gamma * z;
            let weight = if step.evaluation.visits.iter().sum::<f64>() > 0.0 {
                1.0
            } else {
                0.0
            };
            let visits_game = normalized_visits(&step.evaluation.visits);
            let pi = if identity {
                visits_game
            } else {
                let mut pi = vec![0.0; visits_game.len()];
                for (a, v) in visits_game.into_iter().enumerate() {
                    pi[perm[a]] = v;
                }
                pi
            };
            let legal: Vec<usize> = if identity {
                step.evaluation.legal.clone()
            } else {
                step.evaluation.legal.iter().map(|&a| perm[a]).collect()
            };
            out.push((step.obs.clone(), pi, z, weight, agent, legal));
        }
        out.reverse();
        out
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::encoder::IdentityView;
    use crate::policies::tree::expectimax::search::SearchStats;
    use crate::rng::SplitMix64;

    fn eval(visits: Vec<f64>) -> SearchEvaluation {
        let n = visits.len();
        SearchEvaluation {
            values: vec![vec![0.0; n]],
            visits,
            legal: (0..n).collect(),
            stats: SearchStats::default(),
        }
    }

    pub(crate) fn step(visits: Vec<f64>, action: usize, reward: f64) -> Step<SearchEvaluation> {
        Step {
            obs: vec![reward as f32; 2],
            evaluation: eval(visits),
            action,
            reward,
            next_obs: Vec::new(),
            next_legal: Vec::new(),
            terminal: false,
        }
    }

    #[test]
    fn pi_is_the_normalized_visit_distribution() {
        let learner = AlphaZeroLearner::new(1.0);
        let steps = vec![step(vec![6.0, 2.0, 0.0], 0, 0.0)];
        let recs = learner.episode_records(&steps, &[], &IdentityView, 0, &mut SplitMix64::new(0));
        assert_eq!(recs[0].1, vec![0.75, 0.25, 0.0]);
        assert!((recs[0].1.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn z_is_the_discounted_return_to_go() {
        let learner = AlphaZeroLearner::new(0.5);
        let steps = vec![
            step(vec![1.0, 1.0], 0, 0.0),
            step(vec![1.0, 1.0], 1, 0.0),
            step(vec![1.0, 1.0], 0, 1.0),
        ];
        let recs = learner.episode_records(&steps, &[], &IdentityView, 0, &mut SplitMix64::new(0));
        let zs: Vec<f64> = recs.iter().map(|r| r.2).collect();
        assert_eq!(zs, vec![0.25, 0.5, 1.0]);
    }

    #[test]
    fn truncation_tail_seeds_z() {
        let learner = AlphaZeroLearner::new(1.0);
        let steps = vec![step(vec![1.0], 0, 0.0), step(vec![1.0], 0, 0.0)];
        let recs =
            learner.episode_records(&steps, &[0.8], &IdentityView, 0, &mut SplitMix64::new(0));
        assert_eq!(recs.iter().map(|r| r.2).collect::<Vec<_>>(), vec![0.8, 0.8]);
    }

    #[test]
    fn tail_from_row_reads_the_value_slot() {
        let learner = AlphaZeroLearner::new(1.0);
        assert_eq!(
            learner.tail_from_row(&[9.0, 9.0, 9.0, 0.4], 3, &[0, 1, 2], &IdentityView, 0),
            vec![0.4]
        );
    }

    #[test]
    fn value_only_steps_emit_weight_zero_and_inert_pi() {
        let learner = AlphaZeroLearner::new(1.0);
        let vo = Step {
            obs: vec![7.0; 2],
            evaluation: learner.value_only_evaluation(3).unwrap(),
            action: 0,
            reward: 2.0,
            next_obs: Vec::new(),
            next_legal: Vec::new(),
            terminal: true,
        };
        let steps = vec![step(vec![6.0, 2.0, 0.0], 0, 0.0), vo];
        let recs = learner.episode_records(&steps, &[], &IdentityView, 1, &mut SplitMix64::new(0));
        assert_eq!(recs[0].3, 1.0);
        assert_eq!(recs[1].3, 0.0);
        assert_eq!(recs[1].1, vec![0.0; 3], "value-only pi is inert zeros");
        assert_eq!(recs[1].2, 2.0, "the value row carries its own return");
        assert_eq!(recs[0].2, 2.0, "gamma 1: the mover's z includes it");
        assert_eq!(
            recs[0].5,
            vec![0, 1, 2],
            "mover rows carry the state's legal set"
        );
        assert!(
            recs[1].5.is_empty(),
            "value-only rows have an empty legal set"
        );
    }

    #[test]
    fn no_immediate_records() {
        let learner = AlphaZeroLearner::new(1.0);
        let e = eval(vec![1.0, 2.0]);
        assert!(learner
            .eval_records(&e, Vec::new(), &IdentityView, 0, &mut SplitMix64::new(0))
            .is_empty());
    }
}

#[cfg(test)]
mod frame_tests {
    use super::*;
    use crate::encoder::ActionView;
    use crate::rng::SplitMix64;

    struct Rot;
    impl ActionView for Rot {
        fn head_index(&self, action: usize, _: usize) -> usize {
            (action + 1) % 3
        }
        fn game_action(&self, head: usize, _: usize) -> usize {
            (head + 2) % 3
        }
    }

    #[test]
    fn pi_scatters_into_the_head_frame() {
        let learner = AlphaZeroLearner::new(1.0);
        let steps = vec![super::tests::step(vec![6.0, 2.0, 0.0], 0, 0.0)];
        let recs = learner.episode_records(&steps, &[], &Rot, 0, &mut SplitMix64::new(0));
        assert_eq!(recs[0].1, vec![0.0, 0.75, 0.25]);
    }

    #[test]
    fn legal_maps_into_the_head_frame_with_pi() {
        let learner = AlphaZeroLearner::new(1.0);
        let mut steps = vec![super::tests::step(vec![6.0, 0.0, 2.0], 0, 0.0)];
        // A proper subset catches mappings that accidentally work only for full-width legality.
        steps[0].evaluation.legal = vec![0, 2];
        let recs = learner.episode_records(&steps, &[], &Rot, 0, &mut SplitMix64::new(0));
        assert_eq!(recs[0].5, vec![1, 0]);
    }
}
