//! State evaluation and action selection.

use crate::codec::bytes::Reader;
use crate::encoder::StateEncoder;
use crate::game::{Game, Rng};
use crate::reward::Reward;
use crate::stats::CollectStats;

/// Maximum simultaneous joint-action fan. Bindings reject statically oversized compositions;
/// search repeats the check against each realized legal-action product.
pub const MAX_JOINT_SLOTS: usize = 1 << 20;

/// Maximum chance fan materialized by exhaustive search modes.
pub const MAX_ENUMERATED_OUTCOMES: usize = 1 << 20;

/// Per-call context for the stepped search machine: everything a search consults but must
/// never store. `rng` is the owning game's stream, mutably borrowed for this call only —
/// policies never construct or hold a generator.
pub struct SearchCtx<'a, G: Game> {
    pub game: &'a G,
    pub enc: &'a dyn StateEncoder<State = G::State>,
    pub reward: &'a dyn Reward<Event = G::Event>,
    pub rng: &'a mut dyn Rng,
    pub perms: &'a crate::encoder::PermTable,
    pub collect_interior: bool,
}

/// Whether a search needs more inference rounds.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RoundStatus {
    Pending,
    Done,
}

/// The zero-copy engine's state-backed emission seam: rows born in arena spans,
/// cache identity resolved before any reservation.
pub(crate) trait StateSink<S> {
    fn push_state(
        &mut self,
        enc: &dyn crate::encoder::StateEncoder<State = S>,
        player: usize,
        state: &S,
        pos: u32,
        scratch: &mut Vec<f32>,
    );

    /// Classify an already-encoded row straight into backend storage.
    fn push_row(&mut self, player: usize, row: &[f32], pos: u32);
}

/// Collects one round's evaluation requests: `(player, encoded obs)` rows, or
/// state-backed requests (`push_state`) that the zero-copy engine encodes straight
/// into arena storage. Root marks are retained only in capture mode.
pub struct RequestSink<'e, S> {
    pub(crate) players: Vec<usize>,
    pub(crate) obs: Vec<f32>,
    pub(crate) buffered_pos: Vec<u32>,
    pub(crate) roots: Vec<(usize, Vec<f32>)>,
    capture_roots: bool,
    n: usize,
    pub(crate) backend: Option<&'e mut dyn StateSink<S>>,
    scratch: Vec<f32>,
}

impl<S> Default for RequestSink<'_, S> {
    fn default() -> Self {
        RequestSink {
            players: Vec::new(),
            obs: Vec::new(),
            buffered_pos: Vec::new(),
            roots: Vec::new(),
            capture_roots: false,
            n: 0,
            backend: None,
            scratch: Vec::new(),
        }
    }
}

impl<'e, S> RequestSink<'e, S> {
    pub(crate) fn with_backend(backend: &'e mut dyn StateSink<S>) -> Self {
        RequestSink {
            capture_roots: true,
            backend: Some(backend),
            ..Default::default()
        }
    }

    pub fn push(&mut self, player: usize, obs: &[f32]) {
        let pos = self.n as u32;
        self.n += 1;
        if let Some(backend) = self.backend.as_mut() {
            backend.push_row(player, obs, pos);
            return;
        }
        self.players.push(player);
        self.obs.extend_from_slice(obs);
        self.buffered_pos.push(pos);
    }

    /// Push a request whose row is the canonical current-state observation for
    /// `perspective`, retained (in capture mode) so training records need not
    /// re-encode it. `player` routes inference and is deliberately separate. At most
    /// one mark per perspective per search.
    pub fn push_root(&mut self, player: usize, obs: Vec<f32>, perspective: usize) {
        let pos = self.n as u32;
        self.n += 1;
        if let Some(backend) = self.backend.as_mut() {
            // one copy: classify into the arena, move the original into roots
            backend.push_row(player, &obs, pos);
            self.roots.push((perspective, obs));
            return;
        }
        self.players.push(player);
        self.obs.extend_from_slice(&obs);
        self.buffered_pos.push(pos);
        if self.capture_roots {
            self.roots.push((perspective, obs));
        }
    }

    /// Push a request identified by its state: the zero-copy engine resolves the
    /// cache before reserving and encodes misses directly into arena storage; on
    /// other paths this encodes and buffers exactly like `push`.
    pub fn push_state(
        &mut self,
        enc: &dyn crate::encoder::StateEncoder<State = S>,
        player: usize,
        state: &S,
    ) {
        let pos = self.n as u32;
        if let Some(backend) = self.backend.as_mut() {
            backend.push_state(enc, player, state, pos, &mut self.scratch);
            self.n += 1;
            return;
        }
        let row = enc.encode(state, player);
        self.players.push(player);
        self.obs.extend_from_slice(&row);
        self.buffered_pos.push(pos);
        self.n += 1;
    }

    pub fn len(&self) -> usize {
        self.n
    }

    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    pub fn into_parts(self) -> (Vec<usize>, Vec<f32>) {
        (self.players, self.obs)
    }

    /// Retained root rows — engine only; backend mode never buffers rows here.
    pub(crate) fn into_roots(self) -> Vec<(usize, Vec<f32>)> {
        self.roots
    }
}

/// The complete rows answering one search round, in emission order.
pub struct RowsView<'a> {
    pub(crate) data: &'a [f64],
    pub(crate) stride: usize,
}

impl<'a> RowsView<'a> {
    pub fn row(&self, i: usize) -> &'a [f64] {
        &self.data[i * self.stride..(i + 1) * self.stride]
    }

    pub fn stride(&self) -> usize {
        self.stride
    }

    /// The rows as one contiguous row-major slice.
    pub fn flat(&self) -> &'a [f64] {
        self.data
    }

    pub fn len(&self) -> usize {
        self.data.len().checked_div(self.stride).unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn from_slice(data: &'a [f64], stride: usize) -> RowsView<'a> {
        RowsView { data, stride }
    }

    /// Sub-view over `count` rows starting at `start`.
    pub fn slice(&self, start: usize, count: usize) -> RowsView<'a> {
        RowsView {
            data: &self.data[start * self.stride..(start + count) * self.stride],
            stride: self.stride,
        }
    }
}

/// How an algorithm evaluates states and acts.
pub trait Policy {
    type Evaluation;

    type PolicyState;

    /// Largest supported agent count for sequential or simultaneous dynamics. This has no default
    /// so every policy must make its capability claim deliberately.
    fn max_agents(&self, sequential: bool) -> Option<usize>;

    /// Whether sequential search consumes every agent's perspective.
    fn evaluates_all_perspectives(&self, sequential: bool, num_agents: usize) -> bool {
        let _ = (sequential, num_agents);
        false
    }

    /// Whether the policy is sound when the game state contains hidden information. This has no
    /// default so a new policy cannot acquire that soundness claim accidentally.
    fn supports_imperfect_information(&self) -> bool;

    fn begin_episode(&self, rng: &mut dyn Rng) -> Self::PolicyState;

    #[allow(clippy::too_many_arguments)]
    /// Serialize a buffered policy evaluation.
    fn encode_eval(&self, eval: &Self::Evaluation, out: &mut Vec<u8>);
    fn decode_eval(&self, r: &mut Reader, action_count: usize) -> Result<Self::Evaluation, String>;

    fn policy_state_to_u64(&self, s: &Self::PolicyState) -> u64;
    fn policy_state_from_u64(&self, v: u64) -> Result<Self::PolicyState, String>;

    /// Per-decision search state (GAT over the STATE type): stored by the engine between
    /// rounds, so it owns what persists and borrows nothing. Never outlives a collect call.
    type Search<S: Send>: Send;

    fn begin_search<G: Game + Sync>(
        &self,
        ctx: SearchCtx<'_, G>,
        state: &G::State,
        perspectives: &[usize],
    ) -> Self::Search<G::State>
    where
        G::State: Send;

    /// Emit this round's evaluation requests into `out`. `Done` means `finish` may be
    /// called without further inference.
    fn round<G: Game + Sync>(
        &self,
        ctx: SearchCtx<'_, G>,
        search: &mut Self::Search<G::State>,
        out: &mut RequestSink<'_, G::State>,
    ) -> RoundStatus
    where
        G::State: Send;

    /// Consume the complete rows for this search's last round, in emission order — called
    /// once per round, only when every request is answered. `rows` borrows the caller's
    /// assembly buffer: integrate them here instead of re-buffering a copy.
    fn absorb<G: Game + Sync>(
        &self,
        ctx: SearchCtx<'_, G>,
        search: &mut Self::Search<G::State>,
        rows: RowsView<'_>,
    ) where
        G::State: Send;

    /// One `(evaluation, its interior targets)` per perspective, in `perspectives` order.
    fn finish<G: Game + Sync>(
        &self,
        ctx: SearchCtx<'_, G>,
        search: Self::Search<G::State>,
    ) -> Vec<(Self::Evaluation, Vec<crate::learner::InteriorTarget>)>
    where
        G::State: Send;

    /// Choose an action from an evaluation.
    fn select(
        &self,
        eval: &Self::Evaluation,
        state: &mut Self::PolicyState,
        rng: &mut dyn Rng,
    ) -> usize;

    fn fold_telemetry(&self, eval: &Self::Evaluation, stats: &mut CollectStats) {
        let _ = (eval, stats);
    }
}

/// How tree search consumes declared chance distributions.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub enum ChanceMode {
    /// Draw a fresh outcome on every traversal. Unlike `Committed { samples: 1 }`, repeated draws
    /// converge to the chance distribution instead of freezing one biased sample.
    #[default]
    AlwaysResample,
    /// Draw and freeze `samples` outcomes at edge expansion.
    Committed { samples: usize },
    /// Materialize every outcome at edge expansion.
    ExpandAll,
}

impl ChanceMode {
    /// Whether the mode redraws on every traversal rather than only at expansion. Policies should
    /// reject unsupported modes through this property rather than matching the variant name.
    pub fn requires_repeated_traversal(&self) -> bool {
        matches!(self, ChanceMode::AlwaysResample)
    }
}

// select() advances the counter, so u32::MAX must stay unreachable
pub(crate) fn ply_from_u64(v: u64) -> Result<u32, String> {
    match u32::try_from(v) {
        Ok(x) if x < u32::MAX => Ok(x),
        _ => Err(format!("acting-ply counter {v} out of range")),
    }
}

pub(crate) fn thompson_head_from_u64(v: u64, n_heads: usize) -> Result<usize, String> {
    if v as usize >= n_heads {
        return Err(format!(
            "Thompson head {v} out of range for {n_heads} heads"
        ));
    }
    Ok(v as usize)
}

pub(crate) fn argmax(values: &[f64]) -> usize {
    let mut best = 0;
    for (i, &v) in values.iter().enumerate() {
        if v > values[best] {
            best = i;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argmax_takes_the_first_max() {
        assert_eq!(argmax(&[1.0, 3.0, 3.0, 2.0]), 1);
        assert_eq!(argmax(&[5.0]), 0);
    }
}
