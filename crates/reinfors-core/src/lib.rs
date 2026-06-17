//! reinfors-core: the pure-Rust generic simulation + search + rollout framework (no Python, no game).
//!
//! It defines the `Game` trait, the `Policy` (acting) + `Learner` (records) seams the rollout `Engine`
//! drives, and concrete algorithm impls under `policies`/`learners`. Concrete games (e.g. snake) live
//! in the `reinfors-games` crate and implement `Game`; the framework drives them through the trait only.

pub mod engine;
pub mod game;
pub mod learner;
pub mod learners;
pub mod policies;
pub mod policy;
pub(crate) mod rng;

pub use engine::{CollectStats, Engine, EngineParams, EpisodeSummary};
pub use game::{Actor, Game, Rng, Transition};
pub use learner::{Learner, Step};
pub use learners::dqn::{DqnLearner, DqnRecord};
pub use learners::treestrap::{blend_outcome_targets, TreeStrapLearner, TreeStrapRecord};
pub use policies::dqn::{DqnPolicy, QEvaluation};
pub use policies::expectimax::search::{
    search_many, InteriorTarget, Opponent, SearchConfig, SearchResult, SearchStats,
};
pub use policies::expectimax::{SearchEvaluation, SelectiveExpectimaxPolicy};
pub use policy::Policy;

pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}
