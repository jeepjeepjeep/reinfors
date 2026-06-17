//! The `Planner` trait — the acting/evaluation seam. A planner turns the live value network into
//! per-decision evaluations (here: per-head root action values + interior TreeStrap targets + search
//! stats) for the rollout. `SelectiveTreeStrap` (selective expectimax) is the first impl; a model-free
//! planner (e.g. ensemble DQN) is another, slotting into the same rollout `Engine`. Training-record
//! production (z-mixing, masks) is the `Learner`'s job (see `algo`); the Engine owns the rest of the
//! framework (parallel rollout, Thompson + epsilon action choice, telemetry).
//!
//! (PR3 renames this seam to `Policy` and folds the action choice into it; for now the Engine still
//! does the action choice on top of the returned values.)

use crate::game::Game;
use crate::search::{search_many, SearchConfig, SearchResult};

/// How an algorithm evaluates states, driving the rollout `Engine`. The trait is game-agnostic:
/// `evaluate` is generic over the game.
pub trait Planner {
    /// Pooled evaluation of a batch of `(state, agent)` requests with the live net (`infer`): per
    /// request, the per-head root action values `[K][A]`, any extra training records to emit
    /// immediately (TreeStrap interior nodes), and search diagnostics. (Thompson-head/epsilon action
    /// choice on top of the returned values is the Engine's job, not the planner's.) `seed` seeds any
    /// stochastic chance sampling the search does, so a caller controls reproducibility.
    fn evaluate<G, F>(
        &self,
        game: &G,
        requests: Vec<(G::State, usize)>,
        seed: u64,
        infer: &mut F,
    ) -> Vec<SearchResult>
    where
        G: Game + Sync,
        G::State: Send,
        F: FnMut(Vec<f32>, usize) -> Vec<f64>;
}

/// Selective expectimax — today's search. Holds the (game-agnostic) search config and whether to
/// collect interior TreeStrap targets. (The z-mix `outcome_weight` now lives on the `Learner`.)
pub struct SelectiveTreeStrap {
    cfg: SearchConfig,
    collect_interior: bool,
}

impl SelectiveTreeStrap {
    pub fn new(cfg: SearchConfig, collect_interior: bool) -> Self {
        SelectiveTreeStrap {
            cfg,
            collect_interior,
        }
    }
}

impl Planner for SelectiveTreeStrap {
    fn evaluate<G, F>(
        &self,
        game: &G,
        requests: Vec<(G::State, usize)>,
        seed: u64,
        infer: &mut F,
    ) -> Vec<SearchResult>
    where
        G: Game + Sync,
        G::State: Send,
        F: FnMut(Vec<f32>, usize) -> Vec<f64>,
    {
        search_many(
            game,
            &self.cfg,
            requests,
            self.collect_interior,
            seed,
            &mut *infer,
        )
    }
}
