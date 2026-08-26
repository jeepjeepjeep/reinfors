//! PUCT search guided by policy logits and state values.

use crate::codec::bytes::Reader;
use crate::encoder::StateEncoder;
use crate::game::{Game, Rng};
use crate::policies::tree::expectimax::{decode_search_eval, encode_search_eval, SearchEvaluation};
use crate::policies::tree::fold_search_stats;
use crate::policies::tree::mcts::{
    sample_visits, search_many, Guidance, NoiseScope, SequentialBackup,
};
use crate::policy::{argmax, ply_from_u64, ChanceMode, Policy};
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
        noise: Some(crate::policies::tree::mcts::RootNoise {
            epsilon: cfg.noise_epsilon,
            alpha: cfg.noise_alpha,
        }),
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
        ply_from_u64(v)
    }

    fn begin_episode(&self, _rng: &mut dyn Rng) -> u32 {
        0
    }

    type Search<S: Send> = crate::policies::tree::mcts::MctsMulti<S>;

    fn begin_search<G: Game + Sync>(
        &self,
        ctx: crate::policy::SearchCtx<'_, G>,
        state: &G::State,
        perspectives: &[usize],
    ) -> Self::Search<G::State>
    where
        G::State: Send,
    {
        crate::policies::tree::mcts::MctsMulti::new(
            perspectives
                .iter()
                .map(|&agent| {
                    let guidance = Guidance::Puct {
                        c: self.cfg.c_puct,
                        noise: Some(crate::policies::tree::mcts::RootNoise {
                            epsilon: self.cfg.noise_epsilon,
                            alpha: self.cfg.noise_alpha,
                        }),
                        noise_all: matches!(self.cfg.noise_scope, NoiseScope::All),
                    };
                    crate::policies::tree::mcts::mcts_stepper_new(
                        ctx.game,
                        ctx.enc,
                        state.clone(),
                        agent,
                        guidance,
                        matches!(self.cfg.sequential_backup, SequentialBackup::MaxN),
                    )
                })
                .collect(),
        )
    }

    fn round<G: Game + Sync>(
        &self,
        ctx: crate::policy::SearchCtx<'_, G>,
        search: &mut Self::Search<G::State>,
        out: &mut crate::policy::RequestSink<'_, G::State>,
    ) -> crate::policy::RoundStatus
    where
        G::State: Send,
    {
        crate::policies::tree::mcts::mcts_multi_round(
            search,
            ctx.game,
            ctx.enc,
            ctx.reward,
            self.cfg.num_simulations,
            self.cfg.gamma,
            self.cfg.max_depth,
            self.cfg.chance,
            out,
            ctx.rng,
        )
    }

    fn absorb<G: Game + Sync>(
        &self,
        ctx: crate::policy::SearchCtx<'_, G>,
        search: &mut Self::Search<G::State>,
        rows: crate::policy::RowsView<'_>,
    ) where
        G::State: Send,
    {
        crate::policies::tree::mcts::mcts_multi_absorb(
            search,
            ctx.game.action_count(),
            self.cfg.gamma,
            ctx.enc,
            rows,
            ctx.rng,
        );
    }

    fn finish<G: Game + Sync>(
        &self,
        ctx: crate::policy::SearchCtx<'_, G>,
        search: Self::Search<G::State>,
    ) -> Vec<(SearchEvaluation, Vec<crate::learner::InteriorTarget>)>
    where
        G::State: Send,
    {
        let a = ctx.game.action_count();
        search
            .steppers
            .into_iter()
            .map(|st| {
                (
                    crate::policies::tree::mcts::mcts_stepper_finish(st, a),
                    Vec::new(),
                )
            })
            .collect()
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
