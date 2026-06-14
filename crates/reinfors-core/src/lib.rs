//! reinfors-core: the pure-Rust simulation engine (no Python dependency).
//!
//! Phase 1 fills this in with a concrete snake game (dynamics + egocentric observation), built to
//! match `snake_RL`'s `CleanSnakeEnv` so it can be differential-tested against it. Generic game
//! abstractions come later (Phase 5), once the concrete slice is proven.

pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}
