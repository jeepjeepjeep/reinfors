//! Concrete `Learner` implementations. `treestrap` consumes the expectimax family's `SearchEvaluation`
//! (so it pairs with any expectimax policy); `dqn` consumes `QEvaluation`.

pub mod dqn;
pub mod treestrap;
