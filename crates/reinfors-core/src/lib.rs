//! reinfors-core: the pure-Rust simulation engine (no Python dependency).
//!
//! Phase 1 is a concrete snake game (dynamics + egocentric observation) built to match
//! `snake_RL`'s `CleanSnakeEnv`, so it can be differential-tested against it. Generic game
//! abstractions come later (Phase 5), once the concrete slice is proven.

pub mod action;
pub mod obs;
pub mod reward;
pub mod search;
pub mod snake;

pub use action::{Action, RelativeAction};
pub use obs::egocentric;
pub use reward::Reward;
pub use search::{selective_search, selective_search_many, Opponent, SearchParams, SearchStats};
pub use snake::{Cell, DeathCause, Snake, SnakeEnv, StepEvent};

pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}
