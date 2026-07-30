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

use crate::codec::bytes::Reader;
use crate::encoder::StateEncoder;
use crate::game::{Game, Rng};
use crate::policies::tree::expectimax::{decode_search_eval, encode_search_eval, SearchEvaluation};
use crate::policies::tree::mcts::{
    sample_visits, search_many, Guidance, NoiseScope, SequentialBackup,
};
use crate::policy::{argmax, ChanceMode, Policy, SearchPolicy};
use crate::reward::Reward;
use crate::rollout::engine::CollectStats;
use crate::rollout::evaluator::Evaluator;

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
    /// How the search consumes the game's declared chance — explicit chance states (and the
    /// deprecated transition-attached seam) — see [`ChanceMode`](crate::ChanceMode). Inert for
    /// deterministic games.
    pub chance: ChanceMode,
    /// Simultaneous games: which root priors the Dirichlet noise perturbs — the requester's
    /// only, or every agent's. Irrelevant for sequential games (one root table).
    pub noise_scope: NoiseScope,
    /// Sequential backup scheme: `Auto` (negamax at <=2 agents, Max^N past) or `MaxN` forced at
    /// 2 — the negamax-deletion measurement seam (see [`SequentialBackup`]).
    pub sequential_backup: SequentialBackup,
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
    eval: &mut Evaluator<'_, F>,
) -> Vec<SearchEvaluation>
where
    G: Game + Sync,
    G::State: Send,
    F: FnMut(usize, Vec<f32>, usize) -> Vec<f64>,
{
    let guidance = Guidance::Puct {
        c: cfg.c_puct,
        noise: Some((cfg.noise_epsilon, cfg.noise_alpha, seed)),
        noise_all: matches!(cfg.noise_scope, NoiseScope::All),
    };
    search_many(
        game,
        enc,
        reward,
        cfg.num_simulations,
        cfg.gamma,
        cfg.max_depth,
        &guidance,
        cfg.chance,
        seed,
        requests,
        eval,
        matches!(cfg.sequential_backup, SequentialBackup::MaxN),
    )
}

impl Policy for AlphaZero {
    type Evaluation = SearchEvaluation;
    type PolicyState = u32; // plies acted this episode — drives the temperature_drop cutoff

    fn supports_chance_nodes(&self) -> bool {
        true // fixed-probability chance plies: sampled/committed/enumerated per ChanceMode
    }

    fn supports_imperfect_information(&self) -> bool {
        false // rides the MCTS tree: branches on the true state
    }

    fn max_agents(&self, _sequential: bool) -> Option<usize> {
        None // rides the MCTS tree: negamax at ≤2 sequential, Max^N past that, DUCT-N for sim
    }

    fn evaluates_all_perspectives(&self, sequential: bool, num_agents: usize) -> bool {
        // Max^N consumes every perspective's leaf values — at N>2 always, and at 2 when forced.
        sequential
            && (num_agents > 2
                || (num_agents == 2
                    && matches!(self.cfg.sequential_backup, SequentialBackup::MaxN)))
    }

    fn encode_eval(&self, eval: &SearchEvaluation, out: &mut Vec<u8>) {
        encode_search_eval(eval, out);
    }

    fn decode_eval(&self, r: &mut Reader, action_count: usize) -> Result<SearchEvaluation, String> {
        // one value row plus full-width visits (the π source the AZ learner requires)
        decode_search_eval(r, action_count, 1, true)
    }

    fn policy_state_to_u64(&self, s: &u32) -> u64 {
        u64::from(*s)
    }

    fn policy_state_from_u64(&self, v: u64) -> Result<u32, String> {
        u32::try_from(v).map_err(|_| format!("acting-ply counter {v} out of range"))
    }

    fn begin_episode(&self, _rng: &mut dyn Rng) -> u32 {
        0
    }

    fn evaluate<G, F>(
        &self,
        game: &G,
        enc: &dyn StateEncoder<State = G::State>,
        reward: &dyn Reward<Event = G::Event>,
        requests: Vec<(G::State, usize)>,
        seed: u64,
        _collect_interior: bool,
        eval: &mut Evaluator<'_, F>,
    ) -> Vec<SearchEvaluation>
    where
        G: Game + Sync,
        G::State: Send,
        F: FnMut(usize, Vec<f32>, usize) -> Vec<f64>,
    {
        alphazero_many(game, enc, reward, &self.cfg, requests, seed, eval)
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
        Self::fold_search_stats(eval, stats);
    }
}

impl SearchPolicy for AlphaZero {
    fn supports_chance(&self, _mode: ChanceMode) -> bool {
        true // sampled-trajectory search: every mode is expressible
    }
}
