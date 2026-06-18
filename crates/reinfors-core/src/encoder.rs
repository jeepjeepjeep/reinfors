//! The observation seam: a `StateEncoder` maps a game's native `State` into the flat observation
//! tensor a value network consumes. This is *representation*, deliberately split from `Game` (the
//! *rules*): a game can be trained or played under different encodings without touching its dynamics,
//! and the same native state can serve a net (encoded) and a human (raw) at once.
//!
//! The rollout `Engine` and the search hold an encoder as `dyn StateEncoder` (it returns a concrete
//! tensor, so it is object-safe), making the representation swappable at run time without threading a
//! type parameter through the hot path. An encoder is keyed to one game's `State`.

use crate::space::Space;

pub trait StateEncoder: Send + Sync {
    type State;

    /// The observation for `agent` as a flat `[C*H*W]` f32 buffer (row-major, channel-major).
    fn encode(&self, state: &Self::State, agent: usize) -> Vec<f32>;

    /// Observation tensor shape `(C, H, W)` — sizes the value network's input.
    fn obs_shape(&self) -> (usize, usize, usize);

    /// The observation `Space`. Defaults to an unbounded `Box` of `obs_shape`; an encoder may override
    /// to advertise tighter bounds (e.g. one-hot planes in `[0, 1]`).
    fn observation_space(&self) -> Space {
        let (c, h, w) = self.obs_shape();
        Space::Box {
            shape: vec![c, h, w],
            low: f32::NEG_INFINITY,
            high: f32::INFINITY,
        }
    }
}
