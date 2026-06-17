//! reinfors-core: the pure-Rust generic simulation + search + rollout framework (no Python, no game).
//!
//! It defines the `Game` trait, a best-first selective-expectimax `Search`, the swappable `Policy`
//! (acting) + `Learner` (records) seams, and the parallel rollout `Engine`. Concrete games (e.g.
//! snake) live in the `reinfors-games` crate and implement `Game`; the framework drives them through
//! the trait only.

pub mod algo;
pub mod dqn;
pub mod engine;
pub mod game;
pub mod policy;
pub(crate) mod rng;
pub mod search;

pub use algo::{
    blend_outcome_targets, Learner, SearchEvaluation, Step, TreeStrapLearner, TreeStrapRecord,
};
pub use dqn::{DqnLearner, DqnPolicy, QEvaluation}; // dqn::Transition stays module-qualified (game::Transition owns the root name)
pub use engine::{CollectStats, Engine, EngineParams, EpisodeSummary};
pub use game::{Actor, Game, Rng, Transition};
pub use policy::{Policy, SelectiveExpectimaxPolicy};
pub use search::{search_many, InteriorTarget, Opponent, SearchConfig, SearchResult, SearchStats};

pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}
