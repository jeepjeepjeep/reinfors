//! reinfors-games: concrete games implementing reinfors-core's `Game` trait. Snake is the first —
//! its dynamics (`SnakeEnv`), egocentric observation, reward shaping, and the `Snake` adapter plus
//! the selective-search wrappers all live here, leaving reinfors-core a game-free generic framework.

pub mod action;
pub mod connect4;
pub mod gridworld;
pub mod obs;
pub mod reward;
pub mod search;
pub mod snake;
pub mod snake_game;

pub use action::{relative_to_absolute, Action, RelativeAction, RELATIVE_ACTIONS};
pub use connect4::{Connect4, Connect4Reward, Connect4State};
pub use gridworld::{GridState, GridWorld, GridWorldReward};
pub use obs::{egocentric, N_CHANNELS};
pub use reward::SnakeReward;
pub use search::{selective_search, selective_search_many, SearchParams};
pub use snake::{Cell, DeathCause, SnakeBody, SnakeEnv, StepEvent};
pub use snake_game::{Snake, SnakeState};
