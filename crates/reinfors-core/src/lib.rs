//! Generic game, search, solver, and rollout primitives.

/// Canonical built-in game/algorithm compatibility documentation.
pub const COMPATIBILITY_DOCS: &str =
    "https://github.com/jeepjeepjeep/reinfors/blob/main/docs/catalogue/compatibility.md";

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
pub mod solvers;
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
pub use policies::tree::minimax::Minimax;
pub use policy::{ChanceMode, Policy, SearchPolicy, MAX_ENUMERATED_OUTCOMES};
pub use reward::Reward;
pub use rng::SplitMix64;
pub use rollout::engine::{CollectStats, Engine, EngineParams, EpisodeSummary};
pub use rollout::env::Env;
pub use rollout::evaluator::{CommittedRows, EvalBatch, Evaluator, InferMode, Resolve};
pub use rollout::infer_cache::{InferCache, ShardedInferCache};
pub use rollout::start::{AlwaysInitialState, ReachedStateBuffer, Start, StartDistribution};
pub use solvers::best_response::{
    best_response_value, enumerate_infosets, exploitability, EnumerationCapExceeded,
};
pub use solvers::cfr::{CfrSolver, CfrVariant};
pub use solvers::deep_cfr::{AdvantageSample, DeepCfrSolver, DeepCfrStats, StrategySample};
pub use space::Space;

pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}
