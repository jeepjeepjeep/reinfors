//! Policies that act directly from agent observations.

pub mod epsilon_greedy_q;
pub mod ppo;

use crate::game::Game;
use crate::policy::{RequestSink, RoundStatus, RowsView, SearchCtx};

/// The degenerate one-round search machine shared by model-free policies: encode each
/// perspective's observation at begin, emit them all in one round, buffer the rows.
pub struct OneShot<S> {
    pub(crate) agents: Vec<usize>,
    pub(crate) legal: Vec<Vec<usize>>,
    obs: Vec<Vec<f32>>,
    pub(crate) rows: Vec<f64>,
    pub(crate) stride: usize,
    emitted: bool,
    _state: core::marker::PhantomData<S>,
}

pub(crate) fn one_shot_begin<G: Game + Sync>(
    ctx: &SearchCtx<'_, G>,
    state: &G::State,
    perspectives: &[usize],
) -> OneShot<G::State>
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
        rows: Vec::new(),
        stride: 0,
        _state: core::marker::PhantomData,
    }
}

pub(crate) fn one_shot_round<S>(search: &mut OneShot<S>, out: &mut RequestSink) -> RoundStatus {
    if search.emitted {
        return RoundStatus::Done;
    }
    for (agent, obs) in search.agents.iter().zip(&search.obs) {
        out.push(*agent, obs);
    }
    search.emitted = true;
    RoundStatus::Pending
}

pub(crate) fn one_shot_absorb<S>(search: &mut OneShot<S>, rows: RowsView<'_>) {
    search.stride = rows.stride();
    search.rows = (0..rows.len())
        .flat_map(|i| rows.row(i).iter().copied())
        .collect();
}
