//! The acting seam: a `Policy` *evaluates* the options (search / forward / MCTS) into an `Evaluation`,
//! then *selects* an action from it. Concrete policies live in `crate::policies`; the `Engine` drives
//! any of them, and a `Learner` consuming the matching `Evaluation` produces the training records.

use crate::encoder::StateEncoder;
use crate::engine::CollectStats;
use crate::game::{Game, Rng};
use crate::reward::Reward;

/// How an algorithm evaluates states and acts.
pub trait Policy {
    type Evaluation;

    type PolicyState;

    fn begin_episode(&self, rng: &mut dyn Rng) -> Self::PolicyState;

    /// Pooled evaluation of a batch of active `(state, agent)` requests with the live net (`infer`):
    /// one batched forward per round, shared across games. `reward` lets a searching policy value the
    /// in-tree immediate rewards (the engine's per-step reward source); non-search policies ignore it.
    #[allow(clippy::too_many_arguments)]
    fn evaluate<G, F>(
        &self,
        game: &G,
        enc: &dyn StateEncoder<State = G::State>,
        reward: &dyn Reward<Event = G::Event>,
        requests: Vec<(G::State, usize)>,
        seed: u64,
        collect_interior: bool,
        infer: &mut F,
    ) -> Vec<Self::Evaluation>
    where
        G: Game + Sync,
        G::State: Send,
        F: FnMut(Vec<f32>, usize) -> Vec<f64>;

    /// Choose an action from an evaluation, using the game's per-episode state and acting RNG.
    fn select(
        &self,
        eval: &Self::Evaluation,
        state: &mut Self::PolicyState,
        rng: &mut dyn Rng,
    ) -> usize;

    /// Fold this decision's diagnostics into the rollout telemetry.
    fn fold_telemetry(&self, eval: &Self::Evaluation, stats: &mut CollectStats) {
        let _ = (eval, stats);
    }
}

pub(crate) fn argmax(values: &[f64]) -> usize {
    let mut best = 0;
    for (i, &v) in values.iter().enumerate() {
        if v > values[best] {
            best = i;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argmax_takes_the_first_max() {
        assert_eq!(argmax(&[1.0, 3.0, 3.0, 2.0]), 1); // ties go to the earliest index
        assert_eq!(argmax(&[5.0]), 0);
    }
}
