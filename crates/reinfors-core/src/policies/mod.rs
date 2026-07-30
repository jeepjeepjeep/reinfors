//! Policies grouped by their INFORMATION CONDITIONING — which projection of the state each
//! algorithm is allowed to read, the axis the capability seam
//! ([`Policy::supports_imperfect_information`](crate::Policy::supports_imperfect_information))
//! enforces:
//!
//! - [`tree`]: plan over the TRUE state — sound only on perfect-information games (rejected at
//!   construction); chance states are fully supported.
//! - [`modelfree`]: act on each agent's OWN observation only — sound anywhere.
//! - `infoset` (future): plan over information sets — sound on imperfect information by
//!   construction (IS-MCTS; tabular strategies from the `solvers` family).

pub mod modelfree;
pub mod tree;
