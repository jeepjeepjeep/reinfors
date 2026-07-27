//! The start-state seam: a [`StartDistribution`] decides where each fresh episode begins. It is a
//! training-only injection into the [`Engine`](crate::Engine) — the caller-driven [`Env`](crate::Env)
//! never resets inside a training loop, so it holds none. The default [`AlwaysInitialState`] yields the
//! game's `initial_state` (bit-identical to having no seam at all); alternatives seed some episodes
//! from previously-reached states to flatten start-state coverage.
//!
//! Why coverage, not bias: value-based training with bootstrapping (TreeStrap/DQN) regresses each
//! state toward its own successor targets, so *where* episodes start changes the data distribution, not
//! the targets — off-`d₀` starts are unbiased. The lever is [`ReachedStateBuffer`], which coarsens
//! states into cells and samples cells ~uniformly (Go-Explore style): storing everything and sampling
//! uniformly would just reproduce the front-loaded visitation distribution.
//!
//! Determinism: the methods take an injected `&mut dyn Rng` (the engine passes a *disjoint* buffer
//! stream), so the seam never perturbs the env-chance draw order. `AlwaysInitialState` draws nothing.

use std::collections::BTreeMap;

use crate::game::Rng;

/// Where a fresh episode starts.
pub enum Start<S> {
    /// Use the game's `initial_state` — the engine calls it (so the per-game env RNG is consumed
    /// exactly as with no seam).
    Fresh,
    /// Start from this specific state.
    Restore(S),
}

/// Chooses each episode's start state. `Send + Sync` so a composed `Engine` stays `Send + Sync`
/// (the binding erases it behind a `Send + Sync` engine); the seam itself is only touched serially.
pub trait StartDistribution<S>: Send + Sync {
    /// Pick this episode's start: `Fresh` (the engine uses `initial_state`) or a specific `Restore`d
    /// state. Draws only from the injected buffer RNG.
    fn choose(&mut self, rng: &mut dyn Rng) -> Start<S>;

    /// Observe a reached (non-terminal) state each rollout tick, so state-buffer distributions can
    /// fill. Default: ignore (draws nothing).
    fn observe(&mut self, state: &S, rng: &mut dyn Rng) {
        let _ = (state, rng);
    }

    /// Snapshot this distribution's mutable state (`encode_state` = the game codec). The default
    /// (stateless distributions, e.g. `AlwaysInitialState`) is an empty payload.
    fn snapshot_bytes(&self, encode_state: &dyn Fn(&S) -> Vec<u8>) -> Vec<u8> {
        let _ = encode_state;
        Vec::new()
    }

    /// Restore a payload produced by [`snapshot_bytes`](Self::snapshot_bytes).
    fn restore_bytes(
        &mut self,
        bytes: &[u8],
        decode_state: &dyn Fn(&[u8]) -> Result<S, String>,
    ) -> Result<(), String> {
        let _ = decode_state;
        if bytes.is_empty() {
            Ok(())
        } else {
            Err("this start distribution carries no snapshot state".into())
        }
    }
}

/// The default distribution: every episode starts fresh from the game's `initial_state`. Draws no
/// buffer RNG and observes nothing, so an engine using it is bit-identical to one with no seam.
pub struct AlwaysInitialState;

impl<S> StartDistribution<S> for AlwaysInitialState {
    fn choose(&mut self, _rng: &mut dyn Rng) -> Start<S> {
        Start::Fresh
    }
}

/// Coarsens a state into a difficulty cell for the reached-state buffer, or `None` to skip it. The one
/// game-specific piece, supplied at buffer construction (e.g. snake's `snake_length_cell`).
pub type CellKey<S> = Box<dyn Fn(&S) -> Option<u64> + Send + Sync>;

/// A bounded per-cell reservoir: uniform-within-cell samples with rolling eviction of stale states.
struct Reservoir<S> {
    items: Vec<S>,
    count: usize, // total observed (drives reservoir sampling)
}

impl<S> Default for Reservoir<S> {
    fn default() -> Self {
        Reservoir {
            items: Vec::new(),
            count: 0,
        }
    }
}

/// A cell-stratified reservoir of reached states. `cell_key` coarsens a state into a difficulty cell
/// (or `None` to skip it — the one game-specific piece, injected at construction); each cell keeps a
/// bounded reservoir. Sampling picks a non-empty cell ~uniformly then a state within, so start coverage
/// spreads across difficulty instead of tracking the visitation distribution. `p_fresh` is a safety
/// valve: that fraction of resets ignore the buffer and start from `initial_state`.
pub struct ReachedStateBuffer<S> {
    // BTreeMap (not HashMap) so cell iteration order is deterministic — HashMap's randomized order
    // would make the sampled cell depend on process state, breaking reproducibility.
    cells: BTreeMap<u64, Reservoir<S>>,
    cell_key: CellKey<S>,
    capacity: usize,
    p_fresh: f64,
}

impl<S: Clone + Send + Sync> ReachedStateBuffer<S> {
    /// `capacity` is the per-cell reservoir size; `p_fresh` the fraction of resets that ignore the
    /// buffer; `cell_key` the game-specific coarsening (`None` = don't buffer this state).
    pub fn new(
        capacity: usize,
        p_fresh: f64,
        cell_key: impl Fn(&S) -> Option<u64> + Send + Sync + 'static,
    ) -> Self {
        ReachedStateBuffer {
            cells: BTreeMap::new(),
            cell_key: Box::new(cell_key),
            capacity: capacity.max(1),
            p_fresh,
        }
    }

    /// The non-empty cell keys, in deterministic (sorted) order.
    fn occupied_cells(&self) -> Vec<u64> {
        self.cells
            .iter()
            .filter(|(_, r)| !r.items.is_empty())
            .map(|(&k, _)| k)
            .collect()
    }
}

impl<S: Clone + Send + Sync> StartDistribution<S> for ReachedStateBuffer<S> {
    fn choose(&mut self, rng: &mut dyn Rng) -> Start<S> {
        let occupied = self.occupied_cells();
        // Warm-up: an empty buffer always falls back, drawing nothing.
        if occupied.is_empty() {
            return Start::Fresh;
        }
        if rng.unit() < self.p_fresh {
            return Start::Fresh;
        }
        let cell = occupied[rng.below(occupied.len())];
        let items = &self.cells[&cell].items;
        Start::Restore(items[rng.below(items.len())].clone())
    }

    fn observe(&mut self, state: &S, rng: &mut dyn Rng) {
        let Some(cell) = (self.cell_key)(state) else {
            return;
        };
        let capacity = self.capacity;
        let res = self.cells.entry(cell).or_default();
        res.count += 1;
        if res.items.len() < capacity {
            res.items.push(state.clone()); // still filling
        } else {
            // Reservoir sampling: the k-th item lands in a random slot with probability capacity/k, so
            // the reservoir stays a uniform sample of all states seen for this cell (rolling eviction).
            // The state is cloned only when accepted, bounding the per-tick cost.
            let j = rng.below(res.count);
            if j < capacity {
                res.items[j] = state.clone();
            }
        }
    }

    fn snapshot_bytes(&self, encode_state: &dyn Fn(&S) -> Vec<u8>) -> Vec<u8> {
        use crate::codec::bytes::*;
        let mut out = Vec::new();
        put_u32(&mut out, self.cells.len() as u32);
        for (key, res) in &self.cells {
            put_u64(&mut out, *key);
            put_u64(&mut out, res.count as u64);
            put_u32(&mut out, res.items.len() as u32);
            for item in &res.items {
                put_blob(&mut out, &encode_state(item));
            }
        }
        out
    }

    fn restore_bytes(
        &mut self,
        bytes: &[u8],
        decode_state: &dyn Fn(&[u8]) -> Result<S, String>,
    ) -> Result<(), String> {
        use crate::codec::bytes::*;
        let mut r = Reader::new(bytes);
        let n_cells = r.u32()? as usize;
        let mut cells = std::collections::BTreeMap::new();
        for _ in 0..n_cells {
            let key = r.u64()?;
            let count = r.u64()? as usize;
            let n_items = r.u32()? as usize;
            if n_items > self.capacity {
                return Err(format!(
                    "reservoir holds {n_items} items over capacity {}",
                    self.capacity
                ));
            }
            let mut items = Vec::with_capacity(n_items);
            for _ in 0..n_items {
                items.push(decode_state(r.blob()?)?);
            }
            if count < items.len() {
                return Err("reservoir count below item count".into());
            }
            cells.insert(key, Reservoir { items, count });
        }
        r.done()?;
        self.cells = cells;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::SplitMix64;

    #[test]
    fn always_initial_state_is_fresh_and_draws_nothing() {
        let mut d: AlwaysInitialState = AlwaysInitialState;
        let mut rng = SplitMix64::new(0);
        let before = rng.clone();
        assert!(matches!(
            StartDistribution::<i32>::choose(&mut d, &mut rng),
            Start::Fresh
        ));
        StartDistribution::<i32>::observe(&mut d, &1, &mut rng);
        assert_eq!(rng, before, "AlwaysInitialState must not advance the RNG");
    }

    #[test]
    fn empty_buffer_always_falls_back_without_drawing() {
        let mut buf = ReachedStateBuffer::<i32>::new(4, 0.0, |&x| Some(x as u64));
        let mut rng = SplitMix64::new(1);
        let before = rng.clone();
        assert!(matches!(buf.choose(&mut rng), Start::Fresh));
        assert_eq!(rng, before, "an empty buffer must draw nothing (warm-up)");
    }

    #[test]
    fn p_fresh_one_never_restores() {
        let mut buf = ReachedStateBuffer::<i32>::new(4, 1.0, |&x| Some(x as u64));
        let mut rng = SplitMix64::new(2);
        for i in 0..20 {
            buf.observe(&i, &mut rng);
        }
        for _ in 0..50 {
            assert!(matches!(buf.choose(&mut rng), Start::Fresh));
        }
    }

    #[test]
    fn samples_a_valid_state_and_reservoir_is_bounded() {
        // Two cells (even/odd); capacity 8. After many observations each cell holds <= capacity states,
        // and a sampled state is always one that was observed.
        let mut buf = ReachedStateBuffer::<i32>::new(8, 0.0, |&x| Some((x % 2) as u64));
        let mut rng = SplitMix64::new(3);
        for i in 0..500 {
            buf.observe(&i, &mut rng);
        }
        assert_eq!(buf.occupied_cells(), vec![0, 1], "both parity cells fill");
        for r in buf.cells.values() {
            assert!(r.items.len() <= 8, "reservoir capped at capacity");
        }
        for _ in 0..100 {
            match buf.choose(&mut rng) {
                Start::Restore(s) => assert!((0..500).contains(&s)),
                Start::Fresh => {}
            }
        }
    }

    #[test]
    fn cell_key_none_skips_the_state() {
        let mut buf = ReachedStateBuffer::<i32>::new(4, 0.0, |&x| (x > 0).then_some(x as u64));
        let mut rng = SplitMix64::new(4);
        buf.observe(&-1, &mut rng); // skipped
        buf.observe(&-2, &mut rng); // skipped
        assert!(buf.occupied_cells().is_empty());
        buf.observe(&5, &mut rng);
        assert_eq!(buf.occupied_cells(), vec![5]);
    }

    #[test]
    fn sampling_is_reproducible_for_a_seed() {
        let build = || {
            let mut buf = ReachedStateBuffer::<i32>::new(4, 0.2, |&x| Some((x % 5) as u64));
            let mut rng = SplitMix64::new(7);
            for i in 0..200 {
                buf.observe(&i, &mut rng);
            }
            let mut out = Vec::new();
            for _ in 0..30 {
                out.push(match buf.choose(&mut rng) {
                    Start::Restore(s) => Some(s),
                    Start::Fresh => None,
                });
            }
            out
        };
        assert_eq!(build(), build(), "same seed -> identical choose sequence");
    }
}
