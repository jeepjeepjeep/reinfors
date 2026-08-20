//! Lockstep driver for the stepped policy machine: Stage A compatibility path and the
//! engine-less decision driver (choose/Env/arena).

use crate::encoder::{PermTable, StateEncoder};
use crate::game::{Game, Rng};
use crate::learner::InteriorTarget;
use crate::policy::{Policy, RequestSink, RoundStatus, RowsView, SearchCtx};
use crate::reward::Reward;
use crate::rollout::evaluator::Evaluator;

/// Drive one decision per entry to completion, firing one pooled infer call per barrier
/// round. Returns per-decision `(evaluation, interior targets)` pairs in input order.
#[allow(clippy::too_many_arguments)]
pub fn drive_to_completion<G, P, F>(
    policy: &P,
    game: &G,
    enc: &dyn StateEncoder<State = G::State>,
    reward: &dyn Reward<Event = G::Event>,
    perms: &PermTable,
    collect_interior: bool,
    decisions: &[(G::State, Vec<usize>)],
    rng: &mut dyn Rng,
    eval: &mut Evaluator<'_, F>,
) -> Vec<Vec<(P::Evaluation, Vec<InteriorTarget>)>>
where
    G: Game + Sync,
    G::State: Send,
    P: Policy,
    F: FnMut(usize, Vec<f32>, usize) -> Vec<f64>,
{
    let mut searches: Vec<_> = decisions
        .iter()
        .map(|(state, perspectives)| {
            let ctx = SearchCtx {
                game,
                enc,
                reward,
                rng,
                perms,
                collect_interior,
            };
            policy.begin_search(ctx, state, perspectives)
        })
        .collect();
    let mut done = vec![false; searches.len()];
    loop {
        let mut sink = RequestSink::default();
        let mut spans: Vec<(usize, usize, usize)> = Vec::new();
        for (i, search) in searches.iter_mut().enumerate() {
            if done[i] {
                continue;
            }
            let start = sink.len();
            let ctx = SearchCtx {
                game,
                enc,
                reward,
                rng,
                perms,
                collect_interior,
            };
            let status = policy.round(ctx, search, &mut sink);
            let count = sink.len() - start;
            if count > 0 {
                spans.push((i, start, count));
            } else if status == RoundStatus::Done {
                done[i] = true;
            }
        }
        if spans.is_empty() {
            break;
        }
        let n = sink.len();
        let rows = eval.forward(&sink.players, sink.obs, n);
        let stride = rows.len() / n;
        for &(i, start, count) in &spans {
            let view = RowsView {
                data: &rows[start * stride..(start + count) * stride],
                stride,
            };
            policy.absorb(&mut searches[i], view, rng);
        }
    }
    searches
        .into_iter()
        .map(|search| {
            let ctx = SearchCtx {
                game,
                enc,
                reward,
                rng,
                perms,
                collect_interior,
            };
            policy.finish(ctx, search)
        })
        .collect()
}
