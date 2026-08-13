//! Built-in games, encoders, and rewards.

pub mod backgammon;
pub mod chess;
pub mod connect4;
pub mod gridworld;
pub mod holdem;
pub mod kuhn;
pub mod leduc;
pub mod snake;

pub use backgammon::{
    Backgammon, BackgammonEvent, BackgammonReward, BackgammonState, BackgammonTesauro,
};
pub use chess::{
    decode_move as chess_decode_move, move_to_uci as chess_move_to_uci,
    uci_to_action as chess_uci_to_action, Chess, ChessEvent, ChessPlanesAz119, ChessPlanesMinimal,
    ChessPlanesOpenSpiel, ChessPlanesRelative, ChessReward, ChessState, CHESS_ACTIONS,
};
pub use connect4::{Connect4, Connect4Event, Connect4Planes, Connect4Reward, Connect4State};
pub use holdem::{
    HoldemEgocentric, HoldemReward, HoldemState, Street, TexasHoldem, HOLDEM_ACTIONS,
};
pub use kuhn::{KuhnEncoder, KuhnPoker, KuhnState};
pub use leduc::{LeducEncoder, LeducPoker, LeducState};
pub(crate) mod codec_util;

pub use cozy_chess::Board as ChessBoard;
pub use gridworld::{GridEvent, GridState, GridWorld, GridWorldPlanes, GridWorldReward};
pub use snake::{
    snake_length_cell, Action, Cell, DeathCause, EgocentricSnake, Snake, SnakeReward, SnakeState,
    StepEvent,
};
