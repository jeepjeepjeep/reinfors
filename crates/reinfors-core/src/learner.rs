//! Policy-evaluation to training-record conversion.

use crate::encoder::ActionView;
use crate::game::Rng;

/// One buffered decision. Action ids remain in the game frame until record construction.
/// Auxiliary search target: `(observation, per-head value rows)` from an interior node.
pub type InteriorTarget = (Vec<f32>, Vec<Vec<f64>>);

pub struct Step<E> {
    pub obs: Vec<f32>,
    pub evaluation: E,
    pub action: usize,
    pub reward: f64,
    pub next_obs: Vec<f32>,
    pub next_legal: Vec<usize>,
    pub terminal: bool,
}

pub trait Learner<E> {
    type Record;

    /// Whether episode records consume a bootstrap value for a truncated final state.
    fn uses_episode_tail(&self) -> bool {
        false
    }

    /// Extract final-state bootstrap values from one network-output row.
    fn tail_from_row(
        &self,
        row: &[f64],
        action_count: usize,
        legal: &[usize],
        view: &dyn ActionView,
        agent: usize,
    ) -> Vec<f64> {
        let k = row.len() / action_count;
        (0..k)
            .map(|h| {
                let head = &row[h * action_count..(h + 1) * action_count];
                legal
                    .iter()
                    .map(|&aid| head[view.head_index(aid, agent)])
                    .fold(f64::NEG_INFINITY, f64::max)
            })
            .collect()
    }

    /// An optional non-mover evaluation for policies that consume every perspective. Implementors
    /// must also encode a policy-mask convention for these value-only rows (AlphaZero uses zero π).
    fn value_only_evaluation(&self, action_count: usize) -> Option<E> {
        let _ = action_count;
        None
    }

    /// Whether buffered steps require the post-transition observation.
    fn needs_next_obs(&self) -> bool {
        false
    }

    /// Whether this learner consumes auxiliary search targets.
    fn needs_interior(&self) -> bool {
        false
    }

    /// Whether every non-empty trajectory receives a truncation tail, not only the
    /// perspectives active at the truncated state. A sequential non-mover's tail row is an
    /// off-turn query of the network — the same approximation the DQN tail already accepts.
    fn tails_all_trajectories(&self) -> bool {
        false
    }

    /// Whether collection is windowed: `collect(n)` advances complete rounds under frozen
    /// weights until the record floor is met, then bootstraps and emits every live trajectory
    /// fragment so no record ever spans two collect calls.
    fn bootstraps_fragments(&self) -> bool {
        false
    }

    /// Records emitted immediately for one evaluation. The mutable evaluation lets a learner move
    /// out immediate-only payloads instead of retaining them in the episode trajectory.
    fn eval_records(
        &self,
        evaluation: &mut E,
        view: &dyn ActionView,
        agent: usize,
        rng: &mut dyn Rng,
    ) -> Vec<Self::Record>;

    /// Records emitted from a completed trajectory; `tail` is empty at a terminal state.
    fn episode_records(
        &self,
        trajectory: &[Step<E>],
        tail: &[f64],
        view: &dyn ActionView,
        agent: usize,
        rng: &mut dyn Rng,
    ) -> Vec<Self::Record>;
}

/// The final-state bootstrap for PolicyValue rows, where the state value follows the logits.
pub(crate) fn policy_value_tail(row: &[f64], action_count: usize) -> Vec<f64> {
    vec![row[action_count]]
}

/// Sample an independent Bernoulli bootstrap mask for each ensemble head.
pub(crate) fn sample_mask(rng: &mut dyn Rng, n_heads: usize, p: f64) -> Vec<f32> {
    (0..n_heads)
        .map(|_| if rng.unit() < p { 1.0 } else { 0.0 })
        .collect()
}
