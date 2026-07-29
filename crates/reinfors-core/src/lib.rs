//! reinfors-core: the pure-Rust generic simulation + search + rollout framework (no Python, no game).
//!
//! It defines the `Game` trait, the `Policy` (acting) + `Learner` (records) seams the rollout `Engine`
//! drives, and concrete algorithm impls under `policies`/`learners`.

pub mod codec;
pub mod encoder;
pub mod game;
pub mod learner;
pub mod learners;
pub mod policies;
pub mod policy;
pub mod reward;
pub(crate) mod rng;
pub mod rollout;
pub mod space;

pub use codec::StateCodec;
pub use encoder::{check_action_view, ActionView, IdentityView, StateEncoder};
pub use game::{realize_initial_state, Actor, ChanceDist, Game, Rng, Transition};
pub use learner::{Learner, Step};
pub use learners::alphazero::{AlphaZeroLearner, AlphaZeroRecord};
pub use learners::dqn::{Dqn, DqnRecord};
pub use learners::treestrap::{TreeStrap, TreeStrapRecord};
pub use policies::modelfree::epsilon_greedy_q::{EpsilonGreedyQ, QEvaluation};
pub use policies::tree::alphazero::{alphazero_many, AlphaZero, AlphaZeroConfig};
pub use policies::tree::expectimax::search::{
    search_many, InteriorTarget, Opponent, SearchConfig, SearchResult, SearchStats,
};
pub use policies::tree::expectimax::{SearchEvaluation, SelectiveExpectimax};
pub use policies::tree::mcts::{
    mcts_many, ActBy, Mcts, MctsConfig, NoiseScope, SequentialBackup, MAX_JOINT_SLOTS,
};
pub use policy::{ChanceMode, Policy, SearchPolicy, MAX_ENUMERATED_OUTCOMES};
pub use reward::Reward;
pub use rollout::engine::{CollectStats, Engine, EngineParams, EpisodeSummary};
pub use rollout::env::Env;
pub use rollout::evaluator::{CommittedRows, EvalBatch, Evaluator, Resolve};
pub use rollout::infer_cache::InferCache;
pub use rollout::start::{AlwaysInitialState, ReachedStateBuffer, Start, StartDistribution};
pub use space::Space;

pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}
