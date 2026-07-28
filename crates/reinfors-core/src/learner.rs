//! The records seam: a `Learner` turns a policy's per-decision evaluations and finished trajectories
//! into training records. Parameterized by the evaluation type it consumes (`Learner<E>`), so any
//! policy producing a compatible `E` can pair with it. Concrete learners live in `crate::learners`.

use crate::encoder::ActionView;
use crate::game::Rng;

/// One buffered decision in a trajectory, held until its episode ends so the realized return is known.
/// Frame note: `action` and `next_legal` are GAME-frame ids (they come from `select` and
/// `Game::legal_actions`); a learner whose records index net output maps them through the
/// `ActionView` it is handed at record-construction time.
pub struct Step<E> {
    pub obs: Vec<f32>,
    pub evaluation: E, // the policy's per-decision evaluation
    pub action: usize,
    pub reward: f64,
    pub next_obs: Vec<f32>,
    /// The next state's legal actions (filled with `next_obs` when the learner asks for it) — the
    /// TD `max_a Q(s', a)` must range over these only; illegal actions' phantom Q values would
    /// inflate the bootstrap on sparse-action games (chess, backgammon).
    pub next_legal: Vec<usize>,
    pub terminal: bool,
}

pub trait Learner<E> {
    type Record;

    /// Whether `episode_records` consumes the per-head bootstrap value of the final state (the z-tail).
    /// When false the engine skips computing it (a forward).
    fn uses_episode_tail(&self) -> bool {
        false
    }

    /// Extract the z-tail values from one net-output row for the final state. The row layout is the
    /// policy family's infer contract, which the paired learner knows: the default reads `[K][A]`
    /// Q-rows — per-head `max_a` over the state's LEGAL actions (`legal`, the mover-convention set
    /// the engine supplies in GAME-frame ids; the row is net output, so each read crosses into the
    /// head frame via `view`; a dense max would bootstrap a phantom illegal Q on sparse-action
    /// games). The AlphaZero learner overrides to read the value slot of its `[A]-logits + value`
    /// row, ignoring `legal`.
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

    /// Sequential N>2 general-sum games: the placeholder evaluation for a VALUE-ONLY step — a
    /// non-mover perspective buffered at another agent's decision tick, so the per-perspective
    /// leaf values an N-player search consumes (e.g. Max^N) are supervised by the collection
    /// pipeline rather than left out-of-distribution. `None` (the default) opts out: the engine
    /// buffers steps only at the agent's own decisions, exactly as before. An opting-in learner
    /// must emit a policy-mask convention for these steps (AlphaZero: an all-zero π row).
    fn value_only_evaluation(&self, action_count: usize) -> Option<E> {
        let _ = action_count;
        None
    }

    /// Whether the engine should fill each buffered `Step`'s `next_obs` (the post-transition `s'`).
    /// True for transition learners (DQN); false (default) for return-based learners (TreeStrap), so
    /// they pay no per-step observation cost.
    fn needs_next_obs(&self) -> bool {
        false
    }

    /// Whether this learner consumes the policy's auxiliary per-decision targets (TreeStrap interior
    /// MAX nodes).
    fn needs_interior(&self) -> bool {
        false
    }

    /// Records emitted immediately for one decision (TreeStrap interior MAX nodes). Takes `&mut E` so it
    /// can move out the immediate-only payload (interior nodes), leaving `E` lean enough to buffer for
    /// the whole episode — interior is never retained past the decision that produced it.
    /// `view`/`agent`: the encoder's action view and the trajectory's agent — anything the record
    /// carries that indexes NET output (π targets, Q masks, action indices) must be written in the
    /// head frame via `view`; everything else stays in game-frame ids.
    fn eval_records(
        &self,
        evaluation: &mut E,
        view: &dyn ActionView,
        agent: usize,
        rng: &mut dyn Rng,
    ) -> Vec<Self::Record>;

    /// Records from a finished episode's buffered trajectory (TreeStrap z-mixing). `tail` is the final
    /// state's per-head bootstrap on a truncation, or **empty** for a terminal episode. `view`/`agent`
    /// as in [`eval_records`](Self::eval_records).
    fn episode_records(
        &self,
        trajectory: &[Step<E>],
        tail: &[f64],
        view: &dyn ActionView,
        agent: usize,
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
