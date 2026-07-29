//! The data-generation runtime: an autonomous N-game [`Engine`](engine::Engine) and a
//! caller-driven [`Env`](env::Env), both realizing episodes through the shared
//! [`Episode`](episode::Episode) machinery, with pooled inference
//! ([`Evaluator`](evaluator::Evaluator) + cache) and start-state distributions.
//! Algorithm-agnostic: policies act, learners train, this module only rolls out.

pub mod engine;
pub mod env;
pub(crate) mod episode;
pub mod evaluator;
pub mod infer_cache;
pub mod start;
