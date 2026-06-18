//! reinfors-games: concrete games implementing reinfors-core's `Game` trait, leaving reinfors-core a
//! game-free generic framework. Each game is one self-contained module (`snake`, `connect4`,
//! `gridworld`): its dynamics, observation encoder, reward, actions, and `Game` adapter.

pub mod connect4;
pub mod gridworld;
pub mod snake;

pub use connect4::{Connect4, Connect4Planes, Connect4Reward, Connect4State};
pub use gridworld::{GridState, GridWorld, GridWorldPlanes, GridWorldReward};
pub use snake::{
    egocentric_parts, relative_to_absolute, Action, Cell, DeathCause, EgocentricSnake,
    RelativeAction, Snake, SnakeBody, SnakeReward, SnakeState, StepEvent, N_CHANNELS,
    RELATIVE_ACTIONS,
};
