//! The acting seam: a `Policy` *evaluates* the options (search / forward / MCTS) into an `Evaluation`,
//! then *selects* an action from it. Concrete policies live in `crate::policies`; the `Engine` drives
//! any of them, and a `Learner` consuming the matching `Evaluation` produces the training records.

use crate::encoder::StateEncoder;
use crate::engine::CollectStats;
use crate::game::{Game, Rng};

/// How an algorithm evaluates states and acts. Non-generic so `evaluate` can be method-generic over
/// the game, avoiding inference ambiguity.
pub trait Policy {
    /// The per-decision evaluation `evaluate` produces (the assessment of the options — *not* a chosen
    /// action). Consumed by `select` and by the paired `Learner`.
    type Evaluation;
    /// The policy's transient per-episode state, freshly drawn at episode start (e.g. the Thompson
    /// head it acts greedily under). `()` for stateless policies.
    type PolicyState;

    /// Fresh per-episode policy state at a game's episode start (redrawn on reset).
    fn begin_episode(&self, rng: &mut dyn Rng) -> Self::PolicyState;

    /// Pooled evaluation of a batch of active `(state, agent)` requests with the live net (`infer`):
    /// one batched forward per round, shared across games (the throughput win). `seed` seeds the
    /// search's environment-chance sampling so it is reproducible; the per-game *acting* RNG is not
    /// touched here (it is used only in `select`). `collect_interior` is the paired learner's
    /// `needs_interior()` — whether to produce its auxiliary per-decision targets (TreeStrap interior
    /// MAX nodes); policies without such targets (e.g. a plain forward) ignore it. Returns one
    /// `Evaluation` per request.
    fn evaluate<G, F>(
        &self,
        game: &G,
        enc: &dyn StateEncoder<State = G::State>,
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

    /// Fold this decision's diagnostics into the rollout telemetry. Default: nothing (a policy with no
    /// search stats); a search policy contributes depth/leaves/sigma/disagreement.
    fn fold_telemetry(&self, eval: &Self::Evaluation, stats: &mut CollectStats) {
        let _ = (eval, stats);
    }
}

/// First index of the maximum value (ties to the earliest). Shared by the policies' `select`.
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
