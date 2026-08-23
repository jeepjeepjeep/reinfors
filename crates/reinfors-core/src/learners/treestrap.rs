//! TreeStrap training-record production.

use crate::encoder::{head_permutation, ActionView};
use crate::game::Rng;
use crate::learner::{sample_mask, InteriorTarget, Learner, Step};
use crate::policies::tree::expectimax::SearchEvaluation;

/// `(observation, per-head action targets, bootstrap mask, player)`.
pub type TreeStrapRecord = (Vec<f32>, Vec<Vec<f64>>, Vec<f32>, usize);

pub struct TreeStrap {
    pub gamma: f64,
    pub outcome_weight: f64,
    pub bootstrap_p: f64,
    pub interior_targets: bool,
}

impl TreeStrap {
    pub fn new(gamma: f64, outcome_weight: f64, bootstrap_p: f64, interior_targets: bool) -> Self {
        TreeStrap {
            gamma,
            outcome_weight,
            bootstrap_p,
            interior_targets,
        }
    }

    /// Blend discounted returns into each head's executed-action target. This remains an associated
    /// function because the out-of-file differential parity caller supplies free-standing params.
    pub fn blend_outcome_targets(
        trajectory: &[(Vec<Vec<f64>>, usize, f64)],
        gamma: f64,
        outcome_weight: f64,
        tail: &[f64],
    ) -> Vec<Vec<Vec<f64>>> {
        let mut z: Vec<f64> = tail.to_vec();
        let mut out: Vec<Vec<Vec<f64>>> = Vec::with_capacity(trajectory.len());
        for (values, action, reward) in trajectory.iter().rev() {
            for zi in z.iter_mut() {
                *zi = reward + gamma * *zi;
            }
            let mut blended = values.clone();
            if outcome_weight > 0.0 {
                for (h, row) in blended.iter_mut().enumerate() {
                    row[*action] = (1.0 - outcome_weight) * row[*action] + outcome_weight * z[h];
                }
            }
            out.push(blended);
        }
        out.reverse();
        out
    }
}

impl Learner<SearchEvaluation> for TreeStrap {
    type Record = TreeStrapRecord;

    fn uses_episode_tail(&self) -> bool {
        self.outcome_weight > 0.0
    }

    fn needs_interior(&self) -> bool {
        self.interior_targets
    }

    fn eval_records(
        &self,
        evaluation: &SearchEvaluation,
        targets: Vec<InteriorTarget>,
        view: &dyn ActionView,
        agent: usize,
        rng: &mut dyn Rng,
    ) -> Vec<Self::Record> {
        let k = evaluation.values.len();
        // Search targets use game action ids; records supervise encoder head ids.
        let a = targets.first().map_or(0, |(_, v)| v[0].len());
        let (perm, identity) = head_permutation(view, a, agent);
        targets
            .into_iter()
            .map(|(obs, values)| {
                let mask = sample_mask(rng, k, self.bootstrap_p);
                (obs, to_head_frame(values, &perm, identity), mask, agent)
            })
            .collect()
    }

    fn episode_records(
        &self,
        trajectory: &[Step<SearchEvaluation>],
        tail: &[f64],
        view: &dyn ActionView,
        agent: usize,
        rng: &mut dyn Rng,
    ) -> Vec<Self::Record> {
        if trajectory.is_empty() {
            return Vec::new();
        }
        let k = trajectory[0].evaluation.values.len();
        let tail = if tail.is_empty() {
            vec![0.0; k]
        } else {
            tail.to_vec()
        };
        let traj: Vec<(Vec<Vec<f64>>, usize, f64)> = trajectory
            .iter()
            .map(|s| (s.evaluation.values.clone(), s.action, s.reward))
            .collect();
        // Blend before translating game action ids into encoder head ids.
        let a = trajectory[0].evaluation.values[0].len();
        let (perm, identity) = head_permutation(view, a, agent);
        let blended = Self::blend_outcome_targets(&traj, self.gamma, self.outcome_weight, &tail);
        trajectory
            .iter()
            .zip(blended)
            .map(|(step, target)| {
                let mask = sample_mask(rng, k, self.bootstrap_p);
                (
                    step.obs.clone(),
                    to_head_frame(target, &perm, identity),
                    mask,
                    agent,
                )
            })
            .collect()
    }
}

fn to_head_frame(values: Vec<Vec<f64>>, perm: &[usize], identity: bool) -> Vec<Vec<f64>> {
    if identity {
        return values;
    }
    values
        .into_iter()
        .map(|row| {
            let mut out = vec![0.0; row.len()];
            for (a, v) in row.into_iter().enumerate() {
                out[perm[a]] = v;
            }
            out
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoder::IdentityView;
    use crate::policies::tree::expectimax::search::SearchStats;
    use crate::rng::SplitMix64;

    fn eval(values: Vec<Vec<f64>>) -> SearchEvaluation {
        SearchEvaluation {
            values,
            visits: Vec::new(),
            legal: (0..3).collect(),
            stats: SearchStats::default(),
        }
    }

    #[test]
    fn eval_records_drains_interior_and_masks_each() {
        let learner = TreeStrap::new(0.99, 0.3, 0.7, true);
        let interior = vec![
            (
                vec![1.0f32, 2.0],
                vec![vec![0.1, 0.2, 0.3], vec![0.4, 0.5, 0.6]],
            ),
            (
                vec![3.0f32, 4.0],
                vec![vec![0.7, 0.8, 0.9], vec![1.0, 1.1, 1.2]],
            ),
        ];
        let e = eval(vec![vec![0.0; 3], vec![0.0; 3]]);
        let recs = learner.eval_records(
            &e,
            interior.clone(),
            &IdentityView,
            0,
            &mut SplitMix64::new(5),
        );
        assert_eq!(recs.len(), 2);
        let mut rng = SplitMix64::new(5);
        for (i, (obs, values, mask, _player)) in recs.iter().enumerate() {
            assert_eq!(*obs, interior[i].0);
            assert_eq!(*values, interior[i].1);
            assert_eq!(*mask, sample_mask(&mut rng, 2, 0.7));
        }
    }

    #[test]
    fn episode_records_match_blend_plus_mask() {
        let learner = TreeStrap::new(0.99, 0.3, 0.8, false);
        let steps: Vec<Step<SearchEvaluation>> = (0..3)
            .map(|t| Step {
                obs: vec![t as f32; 4],
                evaluation: eval(vec![vec![0.1 * t as f64; 3], vec![0.2 * t as f64; 3]]),
                action: t % 3,
                reward: t as f64,
                next_obs: Vec::new(),
                next_legal: Vec::new(),
                terminal: false,
            })
            .collect();
        let tail = [0.5, -0.5];
        let recs =
            learner.episode_records(&steps, &tail, &IdentityView, 0, &mut SplitMix64::new(9));

        let traj: Vec<(Vec<Vec<f64>>, usize, f64)> = steps
            .iter()
            .map(|s| (s.evaluation.values.clone(), s.action, s.reward))
            .collect();
        let blended = TreeStrap::blend_outcome_targets(&traj, 0.99, 0.3, &tail);
        let mut rng = SplitMix64::new(9);
        assert_eq!(recs.len(), 3);
        for ((obs, target, mask, _player), (step, exp_target)) in
            recs.iter().zip(steps.iter().zip(blended))
        {
            assert_eq!(*obs, step.obs);
            assert_eq!(*target, exp_target);
            assert_eq!(*mask, sample_mask(&mut rng, 2, 0.8));
        }
    }

    #[test]
    fn blend_outcome_targets_mixes_only_the_executed_action() {
        let values = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let traj = vec![(values.clone(), 1usize, 10.0)];
        let blended = TreeStrap::blend_outcome_targets(&traj, 0.9, 0.25, &[0.0, 0.0]);
        assert!((blended[0][0][1] - (0.75 * 2.0 + 0.25 * 10.0)).abs() < 1e-12);
        assert!((blended[0][1][1] - (0.75 * 5.0 + 0.25 * 10.0)).abs() < 1e-12);
        assert_eq!(blended[0][0][0], 1.0);
        assert_eq!(blended[0][1][2], 6.0);
    }

    #[test]
    fn uses_episode_tail_tracks_outcome_weight() {
        assert!(TreeStrap::new(0.99, 0.3, 1.0, false).uses_episode_tail());
        assert!(!TreeStrap::new(0.99, 0.0, 1.0, false).uses_episode_tail());
    }

    #[test]
    fn needs_interior_tracks_the_flag() {
        assert!(TreeStrap::new(0.99, 0.3, 1.0, true).needs_interior());
        assert!(!TreeStrap::new(0.99, 0.3, 1.0, false).needs_interior());
    }
}

#[cfg(test)]
mod frame_tests {
    use super::*;
    use crate::learner::Step;
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

    fn eval(values: Vec<Vec<f64>>) -> SearchEvaluation {
        SearchEvaluation {
            values,
            visits: Vec::new(),
            legal: (0..3).collect(),
            stats: Default::default(),
        }
    }

    #[test]
    fn interior_targets_scatter_into_the_head_frame() {
        let learner = TreeStrap::new(0.99, 0.0, 1.0, true);
        let interior = vec![(vec![1.0f32], vec![vec![0.1, 0.2, 0.3], vec![0.4, 0.5, 0.6]])];
        let e = eval(vec![vec![0.0; 3]; 2]);
        let recs = learner.eval_records(&e, interior, &Rot, 0, &mut SplitMix64::new(0));
        assert_eq!(recs[0].1, vec![vec![0.3, 0.1, 0.2], vec![0.6, 0.4, 0.5]]);
    }

    #[test]
    fn episode_targets_blend_in_game_frame_then_scatter() {
        let learner = TreeStrap::new(1.0, 1.0, 1.0, false);
        let steps = vec![Step {
            obs: vec![0.0f32],
            evaluation: eval(vec![vec![0.1, 0.2, 0.3]]),
            action: 1,
            reward: 2.0,
            next_obs: Vec::new(),
            next_legal: Vec::new(),
            terminal: true,
        }];
        let recs = learner.episode_records(&steps, &[], &Rot, 0, &mut SplitMix64::new(0));
        // w=1 replaces executed game action 1 with z=reward+gamma*tail=2; rotating game slots
        // [0.1, 2.0, 0.3] then yields head slots [0.3, 0.1, 2.0].
        assert_eq!(recs[0].1, vec![vec![0.3, 0.1, 2.0]]);
    }
}
