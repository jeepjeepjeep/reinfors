//! reinfors-core: the pure-Rust generic simulation + search + rollout framework (no Python, no game).
//!
//! It defines the `Game` trait, the `Policy` (acting) + `Learner` (records) seams the rollout `Engine`
//! drives, and concrete algorithm impls under `policies`/`learners`.

pub mod encoder;
pub mod engine;
pub mod env;
pub(crate) mod episode;
pub mod evaluator;
pub mod game;
pub mod infer_cache;
pub mod learner;
pub mod learners;
pub mod policies;
pub mod policy;
pub mod reward;
pub(crate) mod rng;
pub mod space;
pub mod start;

pub use encoder::StateEncoder;
pub use engine::{CollectStats, Engine, EngineParams, EpisodeSummary};
pub use env::Env;
pub use evaluator::{CommittedRows, EvalBatch, Evaluator, Resolve};
pub use game::{Actor, Game, Rng, Transition};
pub use infer_cache::InferCache;
pub use learner::{Learner, Step};
pub use learners::alphazero::{AlphaZeroLearner, AlphaZeroRecord};
pub use learners::dqn::{Dqn, DqnRecord};
pub use learners::treestrap::{TreeStrap, TreeStrapRecord};
pub use policies::alphazero::{alphazero_many, AlphaZero, AlphaZeroConfig};
pub use policies::epsilon_greedy_q::{EpsilonGreedyQ, QEvaluation};
pub use policies::expectimax::search::{
    search_many, InteriorTarget, Opponent, SearchConfig, SearchResult, SearchStats,
};
pub use policies::expectimax::{SearchEvaluation, SelectiveExpectimax};
pub use policies::mcts::{mcts_many, ActBy, ChanceMode, Mcts, MctsConfig};
pub use policy::Policy;
pub use reward::Reward;
pub use space::Space;
pub use start::{AlwaysInitialState, ReachedStateBuffer, Start, StartDistribution};

pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}
