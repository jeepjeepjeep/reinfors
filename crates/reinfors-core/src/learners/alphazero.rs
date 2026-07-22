//! AlphaZero record production: each decision becomes one `(obs, π, z)` training example at episode
//! end — π is the root visit distribution (τ=1 normalized counts, the policy head's cross-entropy
//! target), z the realized discounted return from that step (the value head's regression target).
//! Classic AlphaZero: no interior targets, no bootstrap masks (single head), pure-outcome values —
//! γ=1 with win/loss rewards reproduces the paper's z ∈ {−1, 0, 1}.

use crate::game::Rng;
use crate::learner::{Learner, Step};
use crate::policies::expectimax::SearchEvaluation;

/// One collected AlphaZero record: observation, policy target `π [A]`, value target `z`.
pub type AlphaZeroRecord = (Vec<f32>, Vec<f64>, f64);

/// The AlphaZero learner. Pairs with the `AlphaZero` (PUCT) policy: it reads the search's root visit
/// counts as `π`, so the policy must produce visit-bearing evaluations.
pub struct AlphaZeroLearner {
    pub gamma: f64,
}

impl AlphaZeroLearner {
    pub fn new(gamma: f64) -> Self {
        AlphaZeroLearner { gamma }
    }
}

/// τ=1 policy target: visit counts normalized to a distribution. The root's counts sum to
/// `num_simulations − 1` (sim 1 evaluates the root itself), so the sum is positive for any
/// `num_simulations ≥ 2` (enforced by the binding).
fn normalized_visits(visits: &[f64]) -> Vec<f64> {
    let total: f64 = visits.iter().sum();
    if total <= 0.0 {
        return vec![1.0 / visits.len().max(1) as f64; visits.len()];
    }
    visits.iter().map(|&v| v / total).collect()
}

impl Learner<SearchEvaluation> for AlphaZeroLearner {
    type Record = AlphaZeroRecord;

    /// On truncation, z past the last step seeds from the net's value of the final state.
    fn uses_episode_tail(&self) -> bool {
        true
    }

    /// The net row is `[A]` policy logits + the state value — the tail is that single value.
    fn tail_from_row(&self, row: &[f64], action_count: usize) -> Vec<f64> {
        vec![row[action_count]]
    }

    fn eval_records(
        &self,
        _evaluation: &mut SearchEvaluation,
        _rng: &mut dyn Rng,
    ) -> Vec<Self::Record> {
        Vec::new() // all records are episode-end (z needs the realized outcome)
    }

    fn episode_records(
        &self,
        trajectory: &[Step<SearchEvaluation>],
        tail: &[f64],
        _rng: &mut dyn Rng,
    ) -> Vec<Self::Record> {
        // z = discounted realized return-to-go, from each step's own (per-agent) rewards; an empty
        // tail (terminal episode) seeds z at 0 — the final step's reward carries the outcome.
        let mut z = tail.first().copied().unwrap_or(0.0);
        let mut out: Vec<AlphaZeroRecord> = Vec::with_capacity(trajectory.len());
        for step in trajectory.iter().rev() {
            z = step.reward + self.gamma * z;
            out.push((
                step.obs.clone(),
                normalized_visits(&step.evaluation.visits),
                z,
            ));
        }
        out.reverse();
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policies::expectimax::search::SearchStats;
    use crate::rng::SplitMix64;

    fn eval(visits: Vec<f64>) -> SearchEvaluation {
        SearchEvaluation {
            values: vec![vec![0.0; visits.len()]],
            visits,
            interior: Vec::new(),
            stats: SearchStats::default(),
        }
    }

    fn step(visits: Vec<f64>, action: usize, reward: f64) -> Step<SearchEvaluation> {
        Step {
            obs: vec![reward as f32; 2],
            evaluation: eval(visits),
            action,
            reward,
            next_obs: Vec::new(),
            terminal: false,
        }
    }

    #[test]
    fn pi_is_the_normalized_visit_distribution() {
        let learner = AlphaZeroLearner::new(1.0);
        let steps = vec![step(vec![6.0, 2.0, 0.0], 0, 0.0)];
        let recs = learner.episode_records(&steps, &[], &mut SplitMix64::new(0));
        assert_eq!(recs[0].1, vec![0.75, 0.25, 0.0]);
        assert!((recs[0].1.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn z_is_the_discounted_return_to_go() {
        // Rewards [0, 0, 1] (win at the end), gamma 0.5: z = [0.25, 0.5, 1.0].
        let learner = AlphaZeroLearner::new(0.5);
        let steps = vec![
            step(vec![1.0, 1.0], 0, 0.0),
            step(vec![1.0, 1.0], 1, 0.0),
            step(vec![1.0, 1.0], 0, 1.0),
        ];
        let recs = learner.episode_records(&steps, &[], &mut SplitMix64::new(0));
        let zs: Vec<f64> = recs.iter().map(|r| r.2).collect();
        assert_eq!(zs, vec![0.25, 0.5, 1.0]);
    }

    #[test]
    fn truncation_tail_seeds_z() {
        // gamma 1, rewards [0, 0], tail value 0.8: z = [0.8, 0.8].
        let learner = AlphaZeroLearner::new(1.0);
        let steps = vec![step(vec![1.0], 0, 0.0), step(vec![1.0], 0, 0.0)];
        let recs = learner.episode_records(&steps, &[0.8], &mut SplitMix64::new(0));
        assert_eq!(recs.iter().map(|r| r.2).collect::<Vec<_>>(), vec![0.8, 0.8]);
    }

    #[test]
    fn tail_from_row_reads_the_value_slot() {
        let learner = AlphaZeroLearner::new(1.0);
        // row = [logit, logit, logit, value]
        assert_eq!(learner.tail_from_row(&[9.0, 9.0, 9.0, 0.4], 3), vec![0.4]);
    }

    #[test]
    fn no_immediate_records() {
        let learner = AlphaZeroLearner::new(1.0);
        let mut e = eval(vec![1.0, 2.0]);
        assert!(learner
            .eval_records(&mut e, &mut SplitMix64::new(0))
            .is_empty());
    }
}
