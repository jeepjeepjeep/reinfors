//! Offline SOLVERS — the third execution shape next to policies (act) and learners (train):
//! a solver owns its own traversal of the game, consumes no rollout and no `infer`, and
//! produces a strategy artifact as output. First family: counterfactual regret minimization
//! ([`cfr`]) with exact [`best_response`] / exploitability as its convergence metric —
//! two-player zero-sum equilibrium computation over games with declared chance and
//! information-state keys.

pub mod best_response;
pub mod cfr;
