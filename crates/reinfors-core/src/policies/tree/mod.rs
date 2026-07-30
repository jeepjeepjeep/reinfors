//! Searches that plan over the TRUE state: UCT ([`mcts`]), PUCT/[`alphazero`], and best-first
//! [`expectimax`]. All reject hidden-information games at construction — their backed-up values
//! would otherwise condition on state the agents cannot observe. Chance is fully supported:
//! transition-attached distributions and explicit chance states alike, consumed per the
//! configured [`ChanceMode`](crate::ChanceMode).

pub mod alphazero;
pub mod expectimax;
pub mod mcts;
