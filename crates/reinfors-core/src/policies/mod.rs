//! Policies grouped by their INFORMATION CONDITIONING — which projection of the state each
//! algorithm is allowed to read, the axis the capability seams
//! ([`Policy::supports_imperfect_information`](crate::Policy::supports_imperfect_information),
//! [`Policy::supports_chance_nodes`](crate::Policy::supports_chance_nodes)) enforce:
//!
//! - [`tree`]: plan over the TRUE state — sound only on perfect-information games (rejected at
//!   construction); chance is fully supported, explicit chance states included.
//! - [`modelfree`]: act on each agent's OWN observation only — sound anywhere.
//! - `infoset` (future): plan over information sets — sound on imperfect information by
//!   construction (IS-MCTS; tabular strategies from the `solvers` family).

pub mod modelfree;
pub mod tree;
