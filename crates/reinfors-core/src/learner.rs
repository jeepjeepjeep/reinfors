//! The records seam: a `Learner` turns a policy's per-decision evaluations and finished trajectories
//! into training records. Parameterized by the evaluation type it consumes (`Learner<E>`), so any
//! policy producing a compatible `E` can pair with it. Concrete learners live in `crate::learners`.

use crate::game::Rng;

/// One buffered decision in a trajectory, held until its episode ends so the realized return is known.
/// `E` is the policy's per-decision evaluation; only the part needed for episode-end records is kept
/// here (the immediate-only part — e.g. TreeStrap interior nodes — is taken by `eval_records` first).
pub struct Step<E> {
    pub obs: Vec<f32>,
    pub evaluation: E,
    pub action: usize,
    pub reward: f64,
    /// The post-transition observation `s'` — filled by the engine only when the learner sets
    /// `needs_next_obs` (e.g. a DQN transition learner); empty otherwise.
    pub next_obs: Vec<f32>,
    /// Whether this step's transition ended the episode by reaching a terminal state (false for a
    /// horizon truncation, where `s'` is still a real state to bootstrap from).
    pub terminal: bool,
}

/// Turns evaluations and finished trajectories into training records. Parameterized by the evaluation
/// type `E` it consumes (the paired policy's evaluation), so the link to the policy is a direct bound.
pub trait Learner<E> {
    /// The training record this algorithm emits (TreeStrap `(obs, target, mask)`; a DQN transition; …).
    type Record;

    /// Whether `episode_records` consumes the per-head bootstrap value of the final state (the z-tail).
    /// When false the engine skips computing it (a forward).
    fn uses_episode_tail(&self) -> bool {
        false
    }

    /// Whether the engine should fill each buffered `Step`'s `next_obs` (the post-transition `s'`).
    /// True for transition learners (DQN); false (default) for return-based learners (TreeStrap), so
    /// they pay no per-step observation cost.
    fn needs_next_obs(&self) -> bool {
        false
    }

    /// Whether this learner consumes the policy's auxiliary per-decision targets (TreeStrap interior
    /// MAX nodes). The engine threads it into `Policy::evaluate` so the *consumer* decides whether they
    /// are produced — a policy can't independently collect-or-not and silently mismatch the learner.
    /// Default false (a learner that emits nothing in `eval_records`).
    fn needs_interior(&self) -> bool {
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

/// Per-head Bernoulli bootstrap mask (`rng < p` per head), so ensemble heads train on different
/// subsets and stay diverse. Shared by the bootstrapped-ensemble learners (TreeStrap, DQN).
pub(crate) fn sample_mask(rng: &mut dyn Rng, n_heads: usize, p: f64) -> Vec<f32> {
    (0..n_heads)
        .map(|_| if rng.unit() < p { 1.0 } else { 0.0 })
        .collect()
}
