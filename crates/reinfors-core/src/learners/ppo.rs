//! PPO training-record production: GAE(lambda) advantages over own-decision trajectories.

use crate::encoder::{head_permutation, ActionView};
use crate::game::Rng;
use crate::learner::{policy_value_tail, Learner, Step};
use crate::policies::modelfree::ppo::{masked_log_probs, PpoEvaluation};

pub struct PpoRecord {
    pub player: usize,
    pub obs: Vec<f32>,
    pub action: usize,
    pub behavior_log_prob: f64,
    pub advantage: f64,
    pub ret: f64,
    pub value: f64,
    pub legal: Vec<usize>,
}

pub struct Ppo {
    pub gamma: f64,
    pub lam: f64,
}

impl Ppo {
    pub fn new(gamma: f64, lam: f64) -> Self {
        Ppo { gamma, lam }
    }
}

impl Learner<PpoEvaluation> for Ppo {
    type Record = PpoRecord;

    fn uses_episode_tail(&self) -> bool {
        true
    }

    fn tails_all_trajectories(&self) -> bool {
        true
    }

    fn bootstraps_fragments(&self) -> bool {
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
        policy_value_tail(row, action_count)
    }

    fn eval_records(
        &self,
        _evaluation: &mut PpoEvaluation,
        _view: &dyn ActionView,
        _agent: usize,
        _rng: &mut dyn Rng,
    ) -> Vec<PpoRecord> {
        Vec::new()
    }

    fn episode_records(
        &self,
        trajectory: &[Step<PpoEvaluation>],
        tail: &[f64],
        view: &dyn ActionView,
        agent: usize,
        _rng: &mut dyn Rng,
    ) -> Vec<PpoRecord> {
        let a = trajectory.first().map_or(0, |s| s.evaluation.logits.len());
        let (perm, identity) = head_permutation(view, a, agent);
        // Discounting is per own-decision step (the DQN convention): rewards between this
        // agent's decisions are already accumulated onto the step that caused them.
        let mut next_value = tail.first().copied().unwrap_or(0.0);
        let mut gae = 0.0;
        let mut out: Vec<PpoRecord> = Vec::with_capacity(trajectory.len());
        for step in trajectory.iter().rev() {
            let value = step.evaluation.value;
            let delta = step.reward + self.gamma * next_value - value;
            gae = delta + self.gamma * self.lam * gae;
            let log_probs = masked_log_probs(&step.evaluation.logits, &step.evaluation.legal);
            let slot = step
                .evaluation
                .legal
                .iter()
                .position(|&x| x == step.action)
                .expect("the executed action is legal");
            let legal: Vec<usize> = if identity {
                step.evaluation.legal.clone()
            } else {
                step.evaluation.legal.iter().map(|&x| perm[x]).collect()
            };
            out.push(PpoRecord {
                player: agent,
                obs: step.obs.clone(),
                action: if identity {
                    step.action
                } else {
                    perm[step.action]
                },
                behavior_log_prob: log_probs[slot],
                advantage: gae,
                ret: gae + value,
                value,
                legal,
            });
            next_value = value;
        }
        out.reverse();
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoder::IdentityView;
    use crate::rng::SplitMix64;

    fn step(reward: f64, value: f64, action: usize) -> Step<PpoEvaluation> {
        Step {
            obs: vec![value as f32],
            evaluation: PpoEvaluation {
                logits: vec![0.0, 1.0, -1.0],
                value,
                legal: vec![0, 1, 2],
            },
            action,
            reward,
            next_obs: Vec::new(),
            next_legal: Vec::new(),
            terminal: false,
        }
    }

    fn records(gamma: f64, lam: f64, tail: &[f64]) -> Vec<PpoRecord> {
        // Three decisions: V = [1.0, 2.0, 0.5], r = [0.0, 1.0, 2.0].
        let trajectory = vec![step(0.0, 1.0, 0), step(1.0, 2.0, 1), step(2.0, 0.5, 2)];
        Ppo::new(gamma, lam).episode_records(
            &trajectory,
            tail,
            &IdentityView,
            0,
            &mut SplitMix64::new(0),
        )
    }

    #[test]
    fn gae_matches_the_hand_derivation_with_a_truncation_tail() {
        // gamma 0.9, lam 0.8, bootstrap V_T = 3:
        //   d2 = 2 + 0.9*3 - 0.5 = 4.2
        //   d1 = 1 + 0.9*0.5 - 2 = -0.55
        //   d0 = 0 + 0.9*2 - 1 = 0.8
        //   A2 = 4.2; A1 = -0.55 + 0.72*4.2 = 2.474; A0 = 0.8 + 0.72*2.474 = 2.58128
        let r = records(0.9, 0.8, &[3.0]);
        let adv: Vec<f64> = r.iter().map(|x| x.advantage).collect();
        for (got, want) in adv.iter().zip([2.58128, 2.474, 4.2]) {
            assert!((got - want).abs() < 1e-9, "{adv:?}");
        }
        // Returns are advantage + V_old.
        for x in &r {
            assert!((x.ret - (x.advantage + x.value)).abs() < 1e-12);
        }
    }

    #[test]
    fn lambda_endpoints_reduce_to_td_error_and_monte_carlo() {
        // lam = 0: A_t is exactly the one-step TD error.
        let td = records(0.9, 0.0, &[]);
        let want_td = [0.0 + 0.9 * 2.0 - 1.0, 1.0 + 0.9 * 0.5 - 2.0, 2.0 - 0.5];
        for (x, want) in td.iter().zip(want_td) {
            assert!((x.advantage - want).abs() < 1e-12);
        }
        // lam = 1: A_t telescopes to the full discounted return minus V(s_t).
        let mc = records(0.9, 1.0, &[]);
        let g2 = 2.0;
        let g1 = 1.0 + 0.9 * g2;
        let g0 = 0.0 + 0.9 * g1;
        for (x, g) in mc.iter().zip([g0, g1, g2]) {
            assert!(
                (x.advantage - (g - x.value)).abs() < 1e-12,
                "{}",
                x.advantage
            );
        }
    }

    #[test]
    fn behavior_log_prob_matches_the_actors_sampling_distribution() {
        let r = records(1.0, 1.0, &[]);
        // Action 1 with logits [0, 1, -1] over full legality.
        let lp = masked_log_probs(&[0.0, 1.0, -1.0], &[0, 1, 2]);
        assert!((r[1].behavior_log_prob - lp[1]).abs() < 1e-12);
        let total: f64 = lp.iter().map(|l| l.exp()).sum();
        assert!((total - 1.0).abs() < 1e-12);
    }

    #[test]
    fn terminal_and_truncation_seed_the_recursion_differently() {
        let terminal = records(0.9, 0.8, &[]);
        let truncated = records(0.9, 0.8, &[3.0]);
        assert!(
            (terminal[2].advantage - 1.5).abs() < 1e-12,
            "terminal: d2 = 2 - 0.5"
        );
        assert!(
            (truncated[2].advantage - 4.2).abs() < 1e-12,
            "truncated: d2 = 2 + 0.9*3 - 0.5"
        );
    }
}
