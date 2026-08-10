//! Concrete `Learner` implementations. Each learner consumes a policy-family evaluation type;
//! built-in pairings are documented in the generated compatibility catalogue.
//!
//! Dependency direction: `learners` → `policies` is a **one-way** edge. Each learner imports the
//! `Evaluation` type it consumes from the policy family that *produces* it ("producer owns the type"),
//! so `policies/` and `learners/` are not peers. Do not add a reverse import (e.g. a policy naming a
//! learner's record type) — that turns the edge into a cycle.

pub mod alphazero;
pub mod dqn;
pub mod treestrap;
