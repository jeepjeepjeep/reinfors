//! Policies that act directly from agent observations.

pub mod epsilon_greedy_q;
pub mod ppo;

use crate::game::Game;
use crate::policy::{RequestSink, RoundStatus, SearchCtx};

/// The degenerate one-round search machine shared by model-free policies: encode each
/// perspective's observation at begin, emit them all in one round, integrate the rows
/// into finished evaluations at absorb (no row buffering).
pub struct OneShot<S, E> {
    pub(crate) agents: Vec<usize>,
    pub(crate) legal: Vec<Vec<usize>>,
    obs: Vec<Vec<f32>>,
    pub(crate) results: Vec<E>,
    emitted: bool,
    _state: core::marker::PhantomData<S>,
}

pub(crate) fn one_shot_begin<G: Game + Sync, E>(
    ctx: &SearchCtx<'_, G>,
    state: &G::State,
    perspectives: &[usize],
) -> OneShot<G::State, E>
where
    G::State: Send,
{
    OneShot {
        agents: perspectives.to_vec(),
        legal: perspectives
            .iter()
            .map(|&a| ctx.game.legal_actions(state, a))
            .collect(),
        obs: perspectives
            .iter()
            .map(|&a| ctx.enc.encode(state, a))
            .collect(),
        emitted: false,
        results: Vec::new(),
        _state: core::marker::PhantomData,
    }
}

pub(crate) fn one_shot_round<S, E>(
    search: &mut OneShot<S, E>,
    out: &mut RequestSink,
) -> RoundStatus {
    if search.emitted || search.agents.is_empty() {
        return RoundStatus::Done;
    }
    // Drain: blocked searches must not hold a second copy of every observation.
    for (agent, obs) in search.agents.iter().zip(search.obs.drain(..)) {
        out.push(*agent, &obs);
    }
    search.emitted = true;
    RoundStatus::Pending
}
