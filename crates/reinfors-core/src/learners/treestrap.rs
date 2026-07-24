//! Ensemble-TreeStrap record production: episode-end z-mixing of the realized return into the executed
//! action, interior MAX-node targets, and a per-head bootstrap mask on every record. Consumes the
//! expectimax family's `SearchEvaluation`, so it pairs with any expectimax policy (selective today).

use crate::game::Rng;
use crate::learner::{sample_mask, Learner, Step};
use crate::policies::expectimax::SearchEvaluation;

/// One collected TreeStrap record: observation, per-head `[K][A]` target, and per-head bootstrap mask.
pub type TreeStrapRecord = (Vec<f32>, Vec<Vec<f64>>, Vec<f32>);

/// The TreeStrap learner: z-mix `outcome_weight`, gamma, the bootstrap masking probability, and whether
/// to collect interior MAX-node targets (`interior_targets`, reported via `needs_interior` so the
/// policy collects them iff this learner wants them). The head count `K` is read from the evaluation's
/// value matrix, matching the rollout engine.
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

    /// AlphaGo-style z-mixing: blend each step's realized discounted return-to-go into the executed
    /// action's entry of every head, `(1 - w) * V + w * z`. `trajectory` is time-ordered
    /// `(searched values [K][A], executed action, reward)`; `tail` (len K) seeds z past the last step
    /// (0 at a terminal, the net's per-head state value at a truncation). Unexecuted entries keep their
    /// pure searched value. Returns the per-step blended `[K][A]` targets in time order.
    ///
    /// Associated rather than `&self` because it is pure in its args (gamma + outcome_weight, not the
    /// whole learner): the differential parity binding calls it with free-floating params and no learner.
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
        evaluation: &mut SearchEvaluation,
        rng: &mut dyn Rng,
    ) -> Vec<Self::Record> {
        let k = evaluation.values.len();
        // Move the interior nodes out so they are emitted now and never buffered with the step.
        std::mem::take(&mut evaluation.interior)
            .into_iter()
            .map(|(obs, values)| {
                let mask = sample_mask(rng, k, self.bootstrap_p);
                (obs, values, mask)
            })
            .collect()
    }

    fn episode_records(
        &self,
        trajectory: &[Step<SearchEvaluation>],
        tail: &[f64],
        rng: &mut dyn Rng,
    ) -> Vec<Self::Record> {
        if trajectory.is_empty() {
            return Vec::new();
        }
        let k = trajectory[0].evaluation.values.len();
        // An empty tail (a terminal episode) seeds z at zero, per head.
        let tail = if tail.is_empty() {
            vec![0.0; k]
        } else {
            tail.to_vec()
        };
        let traj: Vec<(Vec<Vec<f64>>, usize, f64)> = trajectory
            .iter()
            .map(|s| (s.evaluation.values.clone(), s.action, s.reward))
            .collect();
        let blended = Self::blend_outcome_targets(&traj, self.gamma, self.outcome_weight, &tail);
        trajectory
            .iter()
            .zip(blended)
            .map(|(step, target)| {
                let mask = sample_mask(rng, k, self.bootstrap_p);
                (step.obs.clone(), target, mask)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policies::expectimax::search::{InteriorTarget, SearchStats};
    use crate::rng::SplitMix64;

    fn eval(values: Vec<Vec<f64>>, interior: Vec<InteriorTarget>) -> SearchEvaluation {
        SearchEvaluation {
            values,
            visits: Vec::new(),
            interior,
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
        let mut e = eval(vec![vec![0.0; 3], vec![0.0; 3]], interior.clone());
        let recs = learner.eval_records(&mut e, &mut SplitMix64::new(5));
        assert!(
            e.interior.is_empty(),
            "interior is moved out, never buffered"
        );
        assert_eq!(recs.len(), 2);
        let mut rng = SplitMix64::new(5);
        for (i, (obs, values, mask)) in recs.iter().enumerate() {
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
                evaluation: eval(
                    vec![vec![0.1 * t as f64; 3], vec![0.2 * t as f64; 3]],
                    Vec::new(),
                ),
                action: t % 3,
                reward: t as f64,
                next_obs: Vec::new(),
                terminal: false,
            })
            .collect();
        let tail = [0.5, -0.5];
        let recs = learner.episode_records(&steps, &tail, &mut SplitMix64::new(9));

        let traj: Vec<(Vec<Vec<f64>>, usize, f64)> = steps
            .iter()
            .map(|s| (s.evaluation.values.clone(), s.action, s.reward))
            .collect();
        let blended = TreeStrap::blend_outcome_targets(&traj, 0.99, 0.3, &tail);
        let mut rng = SplitMix64::new(9);
        assert_eq!(recs.len(), 3);
        for ((obs, target, mask), (step, exp_target)) in recs.iter().zip(steps.iter().zip(blended))
        {
            assert_eq!(*obs, step.obs);
            assert_eq!(*target, exp_target);
            assert_eq!(*mask, sample_mask(&mut rng, 2, 0.8));
        }
    }

    #[test]
    fn blend_outcome_targets_mixes_only_the_executed_action() {
        // Two heads, three actions, action 1 executed; one step, terminal tail (z = reward).
        let values = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let traj = vec![(values.clone(), 1usize, 10.0)];
        let blended = TreeStrap::blend_outcome_targets(&traj, 0.9, 0.25, &[0.0, 0.0]);
        // z = 10 + 0.9*0 = 10; executed entry -> 0.75*v + 0.25*10.
        assert!((blended[0][0][1] - (0.75 * 2.0 + 0.25 * 10.0)).abs() < 1e-12);
        assert!((blended[0][1][1] - (0.75 * 5.0 + 0.25 * 10.0)).abs() < 1e-12);
        // unexecuted entries unchanged.
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
        // The learner is authoritative over interior collection; the engine threads this to the policy.
        assert!(TreeStrap::new(0.99, 0.3, 1.0, true).needs_interior());
        assert!(!TreeStrap::new(0.99, 0.3, 1.0, false).needs_interior());
    }
}
