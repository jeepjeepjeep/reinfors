//! AlphaZero record production: each decision becomes one `(obs, π, z)` training example at episode
//! end — π is the root visit distribution (τ=1 normalized counts, the policy head's cross-entropy
//! target), z the realized discounted return from that step (the value head's regression target).
//! Classic AlphaZero: no interior targets, no bootstrap masks (single head), pure-outcome values —
//! γ=1 with win/loss rewards reproduces the paper's z ∈ {−1, 0, 1}.

use crate::encoder::{head_permutation, ActionView};
use crate::game::Rng;
use crate::learner::{Learner, Step};
use crate::policies::tree::expectimax::SearchEvaluation;

/// One collected AlphaZero record: observation, policy target `π [A]`, value target `z`, and the
/// policy weight (1.0 for the acting agent's row, 0.0 for a value-only row).
///
/// **Value-only records** exist for sequential N>2 games: every non-mover perspective of a real
/// self-play state is buffered at the decision tick, so the per-perspective leaf values Max^N
/// consumes are supervised (their π rows are inert zeros). The training loss must weight the
/// policy term: `(w * cross_entropy(logits, π)).sum() / w.sum()` — every row trains the value
/// head, only weight-1 rows train the policy head. 2p-sequential and simultaneous games emit
/// weight-1 rows only (supervised perspectives ≡ the perspectives their searches consume).
pub type AlphaZeroRecord = (Vec<f32>, Vec<f64>, f64, f64);

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
/// `num_simulations ≥ 2` (enforced by the binding). All-zero visits mark a VALUE-ONLY step
/// (a non-mover perspective) and yield the all-zero π mask — never a fabricated uniform.
fn normalized_visits(visits: &[f64]) -> Vec<f64> {
    let total: f64 = visits.iter().sum();
    if total <= 0.0 {
        return vec![0.0; visits.len()];
    }
    visits.iter().map(|&v| v / total).collect()
}

impl Learner<SearchEvaluation> for AlphaZeroLearner {
    type Record = AlphaZeroRecord;

    /// On truncation, z past the last step seeds from the net's value of the final state.
    fn uses_episode_tail(&self) -> bool {
        true
    }

    /// The net row is `[A]` policy logits + the state value — the tail is that single value (a
    /// layout slot, not an action: no frame crossing).
    fn tail_from_row(
        &self,
        row: &[f64],
        action_count: usize,
        _legal: &[usize],
        _view: &dyn ActionView,
        _agent: usize,
    ) -> Vec<f64> {
        vec![row[action_count]]
    }

    /// The value-only placeholder: zero values, full-width zero visits (the codec's AZ shape),
    /// empty legal set. Its record is `(obs, all-zero π, z, weight 0)` — see [`AlphaZeroRecord`].
    fn value_only_evaluation(&self, action_count: usize) -> Option<SearchEvaluation> {
        Some(SearchEvaluation {
            values: vec![vec![0.0; action_count]],
            visits: vec![0.0; action_count],
            interior: Vec::new(),
            legal: Vec::new(),
            stats: Default::default(),
        })
    }

    fn eval_records(
        &self,
        _evaluation: &mut SearchEvaluation,
        _view: &dyn ActionView,
        _agent: usize,
        _rng: &mut dyn Rng,
    ) -> Vec<Self::Record> {
        Vec::new() // all records are episode-end (z needs the realized outcome)
    }

    fn episode_records(
        &self,
        trajectory: &[Step<SearchEvaluation>],
        tail: &[f64],
        view: &dyn ActionView,
        agent: usize,
        _rng: &mut dyn Rng,
    ) -> Vec<Self::Record> {
        // z = discounted realized return-to-go, from each step's own (per-agent) rewards; an empty
        // tail (terminal episode) seeds z at 0 — the final step's reward carries the outcome.
        let mut z = tail.first().copied().unwrap_or(0.0);
        // π trains against the net's raw logits, so it is written in the HEAD frame: the dense
        // game-frame visit vector scatters through a permutation table computed once per episode
        // (no per-scalar dynamic dispatch; identity views skip the scatter entirely).
        let a = trajectory.first().map_or(0, |s| s.evaluation.visits.len());
        let (perm, identity) = head_permutation(view, a, agent);
        let mut out: Vec<AlphaZeroRecord> = Vec::with_capacity(trajectory.len());
        for step in trajectory.iter().rev() {
            z = step.reward + self.gamma * z;
            // Zero visit sum = a value-only step (non-mover perspective): weight 0, inert π.
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
            out.push((step.obs.clone(), pi, z, weight));
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
            interior: Vec::new(),
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
        // Rewards [0, 0, 1] (win at the end), gamma 0.5: z = [0.25, 0.5, 1.0].
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
        // gamma 1, rewards [0, 0], tail value 0.8: z = [0.8, 0.8].
        let learner = AlphaZeroLearner::new(1.0);
        let steps = vec![step(vec![1.0], 0, 0.0), step(vec![1.0], 0, 0.0)];
        let recs =
            learner.episode_records(&steps, &[0.8], &IdentityView, 0, &mut SplitMix64::new(0));
        assert_eq!(recs.iter().map(|r| r.2).collect::<Vec<_>>(), vec![0.8, 0.8]);
    }

    #[test]
    fn tail_from_row_reads_the_value_slot() {
        let learner = AlphaZeroLearner::new(1.0);
        // row = [logit, logit, logit, value]
        assert_eq!(
            learner.tail_from_row(&[9.0, 9.0, 9.0, 0.4], 3, &[0, 1, 2], &IdentityView, 0),
            vec![0.4]
        );
    }

    #[test]
    fn value_only_steps_emit_weight_zero_and_inert_pi() {
        // A trajectory of [mover step, value-only step]: the value row carries this agent's own
        // z, an all-zero pi, and policy weight 0 — the mover row weight 1.
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
    }

    #[test]
    fn no_immediate_records() {
        let learner = AlphaZeroLearner::new(1.0);
        let mut e = eval(vec![1.0, 2.0]);
        assert!(learner
            .eval_records(&mut e, &IdentityView, 0, &mut SplitMix64::new(0))
            .is_empty());
    }
}

#[cfg(test)]
mod frame_tests {
    use super::*;
    use crate::encoder::ActionView;
    use crate::rng::SplitMix64;

    struct Rot; // head = (game + 1) % 3
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
        // game-frame π [0.75, 0.25, 0.0] lands at heads [1, 2, 0] -> [0.0, 0.75, 0.25]
        assert_eq!(recs[0].1, vec![0.0, 0.75, 0.25]);
    }
}
