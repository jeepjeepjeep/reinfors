//! Offline SOLVERS — the third execution shape next to policies (act) and learners (train):
//! a solver OWNS ITS OWN CONTROL FLOW over the game — its traversal is the data generator,
//! nothing is played, no episode exists — and produces artifacts (strategies, training
//! samples) rather than actions. A solver may consume `infer` (Deep CFR does; the tabular
//! case happens not to): the axis is who drives, not whether nets are involved. First family: counterfactual regret minimization
//! ([`cfr`]) with exact [`best_response`] / exploitability as its convergence metric —
//! two-player zero-sum equilibrium computation over games with declared chance and
//! information-state keys.

pub mod best_response;
pub mod cfr;
pub mod deep_cfr;
