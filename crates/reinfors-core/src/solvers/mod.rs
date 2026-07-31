//! Offline SOLVERS — the third execution shape next to policies (act) and learners (train):
//! a solver OWNS ITS OWN CONTROL FLOW over the game — its traversal is the data generator,
//! nothing is played, no episode exists — and produces artifacts (strategies, training
//! samples) rather than actions. A solver may consume `infer` (Deep CFR does; the tabular
//! case happens not to): the axis is who drives, not whether nets are involved. First family:
//! counterfactual regret minimization ([`cfr`]) over games with declared chance and
//! information-state keys, with exact [`best_response`] / NashConv metrics. The equilibrium
//! convergence guarantee applies to the two-player zero-sum setting; N-player solving remains a
//! useful measured regret-minimization procedure without that guarantee.

pub mod best_response;
pub mod cfr;
pub mod deep_cfr;
