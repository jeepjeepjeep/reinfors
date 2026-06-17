//! The algorithm seam: a `Learner` turns a policy's per-decision evaluations and finished trajectories
//! into training records. This is the record/target half of the engine generalization (PR2); the
//! acting half (a `Policy` trait) follows in PR3. `TreeStrapLearner` is the first impl — the
//! ensemble-TreeStrap target production lifted out of the rollout engine, unchanged.

use crate::game::Rng;
use crate::search::{InteriorTarget, SearchStats};

/// One buffered decision in a trajectory, held until its episode ends so the realized return is known.
/// `E` is the policy's per-decision evaluation; only the part needed for episode-end records is kept
/// here (the immediate-only part — e.g. TreeStrap interior nodes — is taken by `eval_records` first).
pub struct Step<E> {
    pub obs: Vec<f32>,
    pub evaluation: E,
    pub action: usize,
    pub reward: f64,
}

/// Turns evaluations and finished trajectories into training records. Parameterized by the evaluation
/// type `E` it consumes (the paired policy's evaluation), so the link to the policy is a direct bound.
pub trait Learner<E> {
    /// The training record this algorithm emits (TreeStrap `(obs, [K][A] target, [K] mask)` today).
    type Record;

    /// Whether `episode_records` consumes the per-head bootstrap value of the final state (the z-tail).
    /// When false the engine skips computing it (a forward).
    fn uses_episode_tail(&self) -> bool {
        false
    }

    /// Records emitted immediately for one decision (TreeStrap interior MAX nodes). Takes `&mut E` so it
    /// can move out the immediate-only payload (interior nodes), leaving `E` lean enough to buffer for
    /// the whole episode — interior is never retained past the decision that produced it.
    fn eval_records(&self, evaluation: &mut E, rng: &mut dyn Rng) -> Vec<Self::Record>;

    /// Records from a finished episode's buffered trajectory (TreeStrap z-mixing). `tail` is the final
    /// state's per-head bootstrap on a truncation, or **empty** for a terminal episode (the learner
    /// then seeds a zero tail of the head count it reads from its own evaluation).
    fn episode_records(
        &self,
        trajectory: &[Step<E>],
        tail: &[f64],
        rng: &mut dyn Rng,
    ) -> Vec<Self::Record>;
}

/// One collected TreeStrap record: observation, per-head `[K][A]` target, and per-head bootstrap mask.
pub type TreeStrapRecord = (Vec<f32>, Vec<Vec<f64>>, Vec<f32>);

/// A selective-expectimax search's per-decision evaluation: root per-head values (for acting and the
/// z-mix target), interior MAX-node targets (immediate records), and search stats (telemetry).
pub struct SearchEvaluation {
    pub values: Vec<Vec<f64>>, // [K][A]
    pub interior: Vec<InteriorTarget>,
    pub stats: SearchStats,
}

/// Per-head Bernoulli bootstrap mask (`rng < p` per head), so ensemble heads train on different
/// subsets and stay diverse. Shared by every TreeStrap record.
pub(crate) fn sample_mask(rng: &mut dyn Rng, n_heads: usize, p: f64) -> Vec<f32> {
    (0..n_heads)
        .map(|_| if rng.unit() < p { 1.0 } else { 0.0 })
        .collect()
}

/// AlphaGo-style z-mixing: blend each step's realized discounted return-to-go into the executed
/// action's entry of every head, `(1 - w) * V + w * z`. `trajectory` is time-ordered
/// `(searched values [K][A], executed action, reward)`; `tail` (len K) seeds z past the last step
/// (0 at a terminal, the net's per-head state value at a truncation). Unexecuted entries keep their
/// pure searched value. Returns the per-step blended `[K][A]` targets in time order.
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

/// Ensemble-TreeStrap target production: episode-end z-mixing of the realized return into the executed
/// action (`outcome_weight`), interior MAX-node targets, and a per-head bootstrap mask on every record.
/// The head count `K` is read from the evaluation's value matrix, matching the rollout engine.
pub struct TreeStrapLearner {
    pub gamma: f64,
    pub outcome_weight: f64,
    pub bootstrap_p: f64,
}

impl TreeStrapLearner {
    pub fn new(gamma: f64, outcome_weight: f64, bootstrap_p: f64) -> Self {
        TreeStrapLearner {
            gamma,
            outcome_weight,
            bootstrap_p,
        }
    }
}

impl Learner<SearchEvaluation> for TreeStrapLearner {
    type Record = TreeStrapRecord;

    fn uses_episode_tail(&self) -> bool {
        self.outcome_weight > 0.0
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
        let blended = blend_outcome_targets(&traj, self.gamma, self.outcome_weight, &tail);
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
    use crate::rng::SplitMix64;

    fn eval(values: Vec<Vec<f64>>, interior: Vec<InteriorTarget>) -> SearchEvaluation {
        SearchEvaluation {
            values,
            interior,
            stats: SearchStats::default(),
        }
    }

    #[test]
    fn eval_records_drains_interior_and_masks_each() {
        let learner = TreeStrapLearner {
            gamma: 0.99,
            outcome_weight: 0.3,
            bootstrap_p: 0.7,
        };
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
        // Masks match an independent same-seed draw, in node order (k = 2 heads).
        let mut rng = SplitMix64::new(5);
        for (i, (obs, values, mask)) in recs.iter().enumerate() {
            assert_eq!(*obs, interior[i].0);
            assert_eq!(*values, interior[i].1);
            assert_eq!(*mask, sample_mask(&mut rng, 2, 0.7));
        }
    }

    #[test]
    fn episode_records_match_blend_plus_mask() {
        let learner = TreeStrapLearner {
            gamma: 0.99,
            outcome_weight: 0.3,
            bootstrap_p: 0.8,
        };
        let steps: Vec<Step<SearchEvaluation>> = (0..3)
            .map(|t| Step {
                obs: vec![t as f32; 4],
                evaluation: eval(
                    vec![vec![0.1 * t as f64; 3], vec![0.2 * t as f64; 3]],
                    Vec::new(),
                ),
                action: t % 3,
                reward: t as f64,
            })
            .collect();
        let tail = [0.5, -0.5];
        let recs = learner.episode_records(&steps, &tail, &mut SplitMix64::new(9));

        // Reference: blend_outcome_targets then a per-step same-seed mask, exactly as the engine flushes.
        let traj: Vec<(Vec<Vec<f64>>, usize, f64)> = steps
            .iter()
            .map(|s| (s.evaluation.values.clone(), s.action, s.reward))
            .collect();
        let blended = blend_outcome_targets(&traj, 0.99, 0.3, &tail);
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
        let blended = blend_outcome_targets(&traj, 0.9, 0.25, &[0.0, 0.0]);
        // z = 10 + 0.9*0 = 10; executed entry -> 0.75*v + 0.25*10.
        assert!((blended[0][0][1] - (0.75 * 2.0 + 0.25 * 10.0)).abs() < 1e-12);
        assert!((blended[0][1][1] - (0.75 * 5.0 + 0.25 * 10.0)).abs() < 1e-12);
        // unexecuted entries unchanged.
        assert_eq!(blended[0][0][0], 1.0);
        assert_eq!(blended[0][1][2], 6.0);
    }

    #[test]
    fn uses_episode_tail_tracks_outcome_weight() {
        let on = TreeStrapLearner {
            gamma: 0.99,
            outcome_weight: 0.3,
            bootstrap_p: 1.0,
        };
        let off = TreeStrapLearner {
            gamma: 0.99,
            outcome_weight: 0.0,
            bootstrap_p: 1.0,
        };
        assert!(on.uses_episode_tail());
        assert!(!off.uses_episode_tail());
    }
}
