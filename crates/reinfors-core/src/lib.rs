//! reinfors-core: the pure-Rust generic simulation + search + rollout framework (no Python, no game).
//!
//! It defines the `Game` trait, a best-first selective-expectimax `Search`, the swappable `Planner`
//! seam (`SelectiveTreeStrap`), and the parallel rollout `Engine`. Concrete games (e.g. snake) live in
//! the `reinfors-games` crate and implement `Game`; the framework drives them through the trait only.

pub mod algo;
pub mod engine;
pub mod game;
pub mod planner;
pub(crate) mod rng;
pub mod search;

pub use algo::{
    blend_outcome_targets, Learner, SearchEvaluation, Step, TreeStrapLearner, TreeStrapRecord,
};
pub use engine::{CollectStats, Engine, EngineParams, EpisodeSummary};
pub use game::{Actor, Game, Rng, Transition};
pub use planner::{Planner, SelectiveTreeStrap};
pub use search::{search_many, InteriorTarget, Opponent, SearchConfig, SearchResult, SearchStats};

pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}
