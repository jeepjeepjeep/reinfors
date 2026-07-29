//! Policies that act on each agent's OWN observation, never planning over transitions — sound
//! on any game (hidden information, chance nodes) because chance realization and opponent
//! state are entirely the env's business.

pub mod epsilon_greedy_q;
