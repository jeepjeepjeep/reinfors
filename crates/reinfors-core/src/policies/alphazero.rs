//! AlphaZero-style planner: PUCT over the shared MCTS tree (see `mcts`), guided by a two-headed net.
//!
//! Differs from the UCT `Mcts` policy on exactly one axis — the guidance (see `mcts::Guidance`):
//! the net returns `[A]` policy logits plus a value per row (stride `A+1`, vs UCT's `[K][A]` Q rows),
//! each node stores its softmaxed prior, selection scores `Q + c·P·√N/(1+n)`, and the root prior is
//! mixed with Dirichlet exploration noise (`(1-ε)P + ε·Dir(α)`) drawn from the search seed — so
//! collects stay bit-reproducible. Everything else (arena tree, pooled per-round `infer` batching,
//! negamax backup, acting temperature) is the shared machinery.
//!
//! **Sequential + single-agent games only**, like `Mcts` — the binding enforces it, the tree panics as
//! a backstop. The produced [`SearchEvaluation`] carries the root's per-action mean values and visit
//! counts; the AlphaZero learner reads the visits as its policy target `π`.

use crate::encoder::StateEncoder;
use crate::engine::CollectStats;
use crate::game::{Game, Rng};
use crate::policies::expectimax::SearchEvaluation;
use crate::policies::mcts::{sample_visits, search_many, Guidance};
use crate::policy::{argmax, Policy};
use crate::reward::Reward;

#[derive(Clone, Copy, Debug)]
pub struct AlphaZeroConfig {
    pub num_simulations: usize,
    /// PUCT exploration constant (the `c` in `Q + c·P·√N/(1+n)`).
    pub c_puct: f64,
    pub gamma: f64,
    pub max_depth: i32,
    /// Root Dirichlet mix weight ε — `(1-ε)·P + ε·Dir(α)` at each search root; 0 disables the noise.
    pub noise_epsilon: f64,
    /// Dirichlet concentration α (AlphaZero convention: ~10/branching-factor).
    pub noise_alpha: f64,
    /// AlphaZero acting temperature — same semantics as `MctsConfig`: `> 0` samples the move
    /// `∝ visits^(1/temperature)` for the first `temperature_drop` plies, 0 acts greedily.
    pub temperature: f64,
    pub temperature_drop: u32,
}

pub struct AlphaZero {
    cfg: AlphaZeroConfig,
}

impl AlphaZero {
    pub fn new(cfg: AlphaZeroConfig) -> Self {
        AlphaZero { cfg }
    }
}

/// Pooled PUCT over a batch of `(state, agent)` requests — the AlphaZero counterpart of `mcts_many`.
/// `infer` must return `n·(A+1)` values: per row, `A` policy logits then the state value. `seed`
/// drives the root-noise Dirichlet draws (disjoint per tree).
pub fn alphazero_many<G, F>(
    game: &G,
    enc: &dyn StateEncoder<State = G::State>,
    reward: &dyn Reward<Event = G::Event>,
    cfg: &AlphaZeroConfig,
    requests: Vec<(G::State, usize)>,
    seed: u64,
    infer: &mut F,
) -> Vec<SearchEvaluation>
where
    G: Game + Sync,
    G::State: Send,
    F: FnMut(Vec<f32>, usize) -> Vec<f64>,
{
    let guidance = Guidance::Puct {
        c: cfg.c_puct,
        noise: Some((cfg.noise_epsilon, cfg.noise_alpha, seed)),
    };
    search_many(
        game,
        enc,
        reward,
        cfg.num_simulations,
        cfg.gamma,
        cfg.max_depth,
        &guidance,
        requests,
        infer,
    )
}

impl Policy for AlphaZero {
    type Evaluation = SearchEvaluation;
    type PolicyState = u32; // plies acted this episode — drives the temperature_drop cutoff

    fn begin_episode(&self, _rng: &mut dyn Rng) -> u32 {
        0
    }

    #[allow(clippy::too_many_arguments)]
    fn evaluate<G, F>(
        &self,
        game: &G,
        enc: &dyn StateEncoder<State = G::State>,
        reward: &dyn Reward<Event = G::Event>,
        requests: Vec<(G::State, usize)>,
        seed: u64,
        _collect_interior: bool,
        infer: &mut F,
    ) -> Vec<SearchEvaluation>
    where
        G: Game + Sync,
        G::State: Send,
        F: FnMut(Vec<f32>, usize) -> Vec<f64>,
    {
        alphazero_many(game, enc, reward, &self.cfg, requests, seed, infer)
    }

    /// Classic AlphaZero acting: by visit count — sampled under the opening temperature, greedy after.
    fn select(&self, eval: &SearchEvaluation, state: &mut u32, rng: &mut dyn Rng) -> usize {
        let ply = *state;
        *state += 1;
        if self.cfg.temperature > 0.0 && ply < self.cfg.temperature_drop {
            return sample_visits(&eval.visits, self.cfg.temperature, rng);
        }
        argmax(&eval.visits)
    }

    fn fold_telemetry(&self, eval: &SearchEvaluation, stats: &mut CollectStats) {
        let s = &eval.stats;
        stats.max_depth = stats.max_depth.max(s.max_depth);
        stats.sum_leaves += s.leaves as f64;
        stats.sum_rounds += s.rounds as f64;
        stats.sum_expansions += s.expansions as f64;
    }
}
