//! Searches that plan over the TRUE state: UCT ([`mcts`]), PUCT/[`alphazero`], and best-first
//! [`expectimax`]. All reject hidden-information and chance-node games at construction — their
//! backed-up values would otherwise condition on state the agents cannot observe.

pub mod alphazero;
pub mod expectimax;
pub mod mcts;
