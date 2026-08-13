//! PUCT search guided by policy logits and state values.

use crate::codec::bytes::Reader;
use crate::encoder::StateEncoder;
use crate::game::{Game, Rng};
use crate::policies::tree::expectimax::{decode_search_eval, encode_search_eval, SearchEvaluation};
use crate::policies::tree::mcts::{
    sample_visits, search_many, Guidance, NoiseScope, SequentialBackup,
};
use crate::policy::{argmax, fold_search_stats, ChanceMode, Policy};
use crate::reward::Reward;
use crate::rollout::engine::CollectStats;
use crate::rollout::evaluator::Evaluator;

#[derive(Clone, Copy, Debug)]
pub struct AlphaZeroConfig {
    pub num_simulations: usize,
    pub c_puct: f64,
    pub gamma: f64,
    pub max_depth: i32,
    pub noise_epsilon: f64,
    /// Dirichlet concentration; the AlphaZero heuristic is roughly `10 / branching_factor`.
    pub noise_alpha: f64,
    pub temperature: f64,
    pub temperature_drop: u32,
    pub chance: ChanceMode,
    pub noise_scope: NoiseScope,
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

/// Run pooled PUCT over `(state, agent)` requests.
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
    type PolicyState = u32;

    fn supports_imperfect_information(&self) -> bool {
        false
    }

    fn max_agents(&self, _sequential: bool) -> Option<usize> {
        None
    }

    fn evaluates_all_perspectives(&self, sequential: bool, num_agents: usize) -> bool {
        sequential
            && (num_agents > 2
                || (num_agents == 2
                    && matches!(self.cfg.sequential_backup, SequentialBackup::MaxN)))
    }

    fn encode_eval(&self, eval: &SearchEvaluation, out: &mut Vec<u8>) {
        encode_search_eval(eval, out);
    }

    fn decode_eval(&self, r: &mut Reader, action_count: usize) -> Result<SearchEvaluation, String> {
        decode_search_eval(r, action_count, 1, true)
    }

    fn policy_state_to_u64(&self, s: &u32) -> u64 {
        u64::from(*s)
    }

    fn policy_state_from_u64(&self, v: u64) -> Result<u32, String> {
        // select() advances the counter, so the maximum value must stay unreachable.
        match u32::try_from(v) {
            Ok(x) if x < u32::MAX => Ok(x),
            _ => Err(format!("acting-ply counter {v} out of range")),
        }
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

    fn select(&self, eval: &SearchEvaluation, state: &mut u32, rng: &mut dyn Rng) -> usize {
        let ply = *state;
        *state += 1;
        if self.cfg.temperature > 0.0 && ply < self.cfg.temperature_drop {
            return sample_visits(&eval.visits, self.cfg.temperature, rng);
        }
        argmax(&eval.visits)
    }

    fn fold_telemetry(&self, eval: &SearchEvaluation, stats: &mut CollectStats) {
        fold_search_stats(eval, stats);
    }
}
