//! Concrete `Policy` implementations. Each module is one policy family: `epsilon_greedy_q` (a lone
//! model-free policy acting on Q-values) is a single file; `expectimax` is a family with shared search
//! machinery + room for variants (selective today, exhaustive later), so it is a directory.

pub mod epsilon_greedy_q;
pub mod expectimax;
pub mod mcts;
