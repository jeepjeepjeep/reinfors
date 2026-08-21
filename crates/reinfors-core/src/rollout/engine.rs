//! Batched rollout and training-record collection.

use std::collections::HashMap;

use crate::codec::StateCodec;
use crate::encoder::StateEncoder;
use crate::game::{Actor, Game};
use crate::learner::{Learner, Step};
use crate::policy::Policy;
use crate::reward::Reward;
use crate::rng::SplitMix64;
use crate::rollout::episode::Episode;
use crate::rollout::evaluator::{Evaluator, InferMode};
use crate::rollout::infer_cache::InferCache;
use crate::rollout::start::{AlwaysInitialState, Start, StartDistribution};

pub use crate::stats::EpisodeSummary;

pub use crate::stats::CollectStats;

pub(crate) struct StartParts<'a, S> {
    pub dist: &'a mut dyn StartDistribution<S>,
    pub rng: &'a mut SplitMix64,
}

impl<S> StartParts<'_, S> {
    fn observe(&mut self, state: &S) {
        self.dist.observe(state, &mut *self.rng);
    }

    fn choose(&mut self) -> Start<S> {
        self.dist.choose(&mut *self.rng)
    }
}

/// Engine-level rollout parameters.
pub struct EngineParams {
    pub n_games: usize,
    pub seed: u64,
    /// Fixed call shape in rows (None = off); see the inference contract.
    pub pad_rows_to: Option<usize>,
    /// Scheduler firing threshold in rows (None = max(1, n_games / 2)).
    pub batch_size: Option<usize>,
    /// CPU fan-out width for per-game work (None = 1 for now; threads land with the fan).
    pub n_threads: Option<usize>,
}

impl Default for EngineParams {
    fn default() -> Self {
        EngineParams {
            n_games: 1,
            seed: 0,
            pad_rows_to: None,
            batch_size: None,
            n_threads: None,
        }
    }
}

pub struct Engine<G: Game + Sync, P: Policy, L: Learner<P::Evaluation>> {
    game: G,
    encoder: Box<dyn StateEncoder<State = G::State>>,
    reward: Box<dyn Reward<Event = G::Event>>,
    policy: P,
    learner: L,
    episodes: Vec<Episode<G>>,
    search_rng: SplitMix64,
    start_dist: Box<dyn StartDistribution<G::State>>,
    // Slot 0 is shared; slots 1..=N isolate per-player networks.
    infer_caches: Option<Vec<InferCache>>,
    learn_mask: Vec<bool>,
    // Returns cannot be derived from steps: an agent may be rewarded before ever acting.
    episode_returns: Vec<Vec<f64>>,
    sequential: bool,
    pad_rows_to: Option<usize>,
    buffer_rng: SplitMix64,
    seeded: Vec<bool>,
    policy_states: Vec<P::PolicyState>,
    ticks: Vec<usize>,
    traj: Vec<Vec<Vec<Step<P::Evaluation>>>>,
    perms: crate::encoder::PermTable,
    batch_size: usize,
    sweep_cursor: usize,
    thread_pool: Option<rayon::ThreadPool>,
}

impl<G: Game + Sync, P: Policy, L: Learner<P::Evaluation>> Engine<G, P, L>
where
    G::State: Send,
{
    pub fn new(
        game: G,
        encoder: Box<dyn StateEncoder<State = G::State>>,
        reward: Box<dyn Reward<Event = G::Event>>,
        policy: P,
        learner: L,
        params: EngineParams,
    ) -> Self {
        let n = game.num_agents();
        assert!(n >= 1, "a game must have at least one agent");
        if let Some(pad) = params.pad_rows_to {
            assert!(pad >= 1, "pad_rows_to must be >= 1");
        }
        let mut episodes: Vec<Episode<G>> = (0..params.n_games)
            .map(|i| {
                Episode::new(
                    &game,
                    params
                        .seed
                        .wrapping_add((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)),
                )
            })
            .collect();
        let sequential = episodes
            .first()
            .is_some_and(|ep| matches!(game.actor(&ep.state), Actor::Agent(_)));
        assert!(
            game.perfect_information() || policy.supports_imperfect_information(),
            "this policy searches the true state and would be clairvoyant on a \
             hidden-information game; see {}",
            crate::COMPATIBILITY_DOCS
        );
        if let Some(cap) = policy.max_agents(sequential) {
            assert!(
                n <= cap,
                "this policy supports at most {cap} agents for this game's dynamics; the game has {n}"
            );
        }
        let policy_states: Vec<P::PolicyState> = episodes
            .iter_mut()
            .map(|ep| policy.begin_episode(&mut ep.rng))
            .collect();
        let ticks = vec![0; params.n_games];
        let num_agents = game.num_agents();
        let traj = (0..params.n_games)
            .map(|_| (0..num_agents).map(|_| Vec::new()).collect())
            .collect();
        let search_rng = SplitMix64::new(params.seed ^ 0xD1B5_4A32_D192_ED03);
        let buffer_rng = SplitMix64::new(params.seed ^ 0x2545_F491_4F6C_DD1D);
        let seeded = vec![false; params.n_games];
        let perms = crate::encoder::PermTable::build(&*encoder, game.action_count(), num_agents);
        Engine {
            game,
            encoder,
            reward,
            policy,
            learner,
            episodes,
            search_rng,
            start_dist: Box::new(AlwaysInitialState),
            infer_caches: None,
            learn_mask: vec![true; num_agents],
            episode_returns: vec![vec![0.0; num_agents]; params.n_games],
            sequential,
            pad_rows_to: params.pad_rows_to,
            buffer_rng,
            seeded,
            policy_states,
            ticks,
            traj,
            perms,
            batch_size: params
                .batch_size
                .unwrap_or_else(|| (params.n_games / 2).max(1)),
            sweep_cursor: 0,
            thread_pool: params.n_threads.filter(|&n| n > 1).map(|n| {
                rayon::ThreadPoolBuilder::new()
                    .num_threads(n)
                    .build()
                    .expect("engine thread pool")
            }),
        }
    }

    /// Override the start-state distribution.
    pub fn with_start_distribution(
        mut self,
        start_dist: Box<dyn StartDistribution<G::State>>,
    ) -> Self {
        self.start_dist = start_dist;
        self
    }

    /// Restrict training-record emission to selected players.
    pub fn with_learn_players(mut self, players: &[usize]) -> Self {
        assert!(!players.is_empty(), "at least one player must learn");
        let n = self.game.num_agents();
        let mut mask = vec![false; n];
        for &p in players {
            assert!(p < n, "learn player {p} out of range (game has {n} agents)");
            mask[p] = true;
        }
        self.learn_mask = mask;
        self
    }

    /// Install one shared cache followed by one cache per player.
    pub fn with_infer_caches(mut self, caches: Vec<InferCache>) -> Self {
        assert_eq!(
            caches.len(),
            self.game.num_agents() + 1,
            "one cache per slot: shared + one per player"
        );
        self.infer_caches = Some(caches);
        self
    }

    /// Collect at least `n_records` using one shared network callback.
    pub fn collect<F>(&mut self, n_records: usize, mut infer: F) -> (Vec<L::Record>, CollectStats)
    where
        F: FnMut(Vec<f32>, usize) -> Vec<f64>,
        P: Sync,
    {
        self.collect_routed(n_records, InferMode::Shared, move |_player, obs, n| {
            infer(obs, n)
        })
    }

    /// Collect with shared or per-player inference routing.
    pub fn collect_routed<F>(
        &mut self,
        n_records: usize,
        mode: InferMode,
        mut infer: F,
    ) -> (Vec<L::Record>, CollectStats)
    where
        F: FnMut(usize, Vec<f32>, usize) -> Vec<f64>,
        P: Sync,
    {
        let mut out: Vec<L::Record> = Vec::new();
        let mut stats = CollectStats::default();
        let num_agents = self.game.num_agents();
        let collect_interior = self.learner.needs_interior();
        // Move caches out so the long-lived evaluator does not borrow all of `self`.
        let mut caches = self.infer_caches.take();
        if let Some(c) = caches.as_mut() {
            for cache in c.iter_mut() {
                cache.begin_collect();
            }
        }
        let cache_slice = caches.as_mut().map(|c| match mode {
            InferMode::Shared => &mut c[..1],
            InferMode::PerPlayer => &mut c[1..],
        });
        assert!(
            self.pad_rows_to.is_none() || matches!(mode, InferMode::Shared),
            "pad_rows_to supports a single shared infer callback"
        );
        let mut evaluator =
            Evaluator::new(&mut infer, mode, cache_slice).with_pad_rows_to(self.pad_rows_to);

        let fragments = self.learner.bootstraps_fragments();
        if fragments {
            discard_fragments(&mut self.traj);
        }
        // Threshold scheduler: per-slot decision phases, one queue, fire at batch_size or
        // on drain; games progress unevenly (the documented scheduler contract). Round
        // bodies fan across the engine's thread pool at n_threads > 1 with slot-order
        // merges, so results are schedule-invariant; episode tails and the fragment flush
        // ride the same evaluator directly for now (padding and cache shared) and become
        // queue jobs when the flush unifies.
        {
            enum Phase<SE> {
                Idle,
                Deciding {
                    search: SE,
                    perspectives: Vec<usize>,
                    outstanding: usize,
                    rows: Vec<f64>,
                    stride: usize,
                },
            }
            enum Outcome {
                None,
                Emitted(Vec<usize>, Vec<f32>, usize),
                Done,
            }
            let n_games = self.episodes.len();
            let batch_size = self.batch_size;
            let base_cursor = self.sweep_cursor % n_games.max(1);
            let mut phases: Vec<Phase<P::Search<G::State>>> =
                (0..n_games).map(|_| Phase::Idle).collect();
            let mut q_players: Vec<usize> = Vec::new();
            let mut q_obs: Vec<f32> = Vec::new();
            let mut q_dest: Vec<usize> = Vec::new();
            let mut cutting = false;
            loop {
                let collected = if fragments {
                    fragment_potential(out.len(), &self.traj, &self.learn_mask)
                } else {
                    out.len()
                };
                if collected >= n_records {
                    cutting = true;
                }

                // Phase 1: admission + absorb + round for every runnable slot. Fanned when a
                // pool exists; per-slot outputs land in slots, merged in rotation order below.
                let game = &self.game;
                let encoder = &*self.encoder;
                let reward = &*self.reward;
                let policy = &self.policy;
                let perms = &self.perms;
                let run_slot =
                    |phase: &mut Phase<P::Search<G::State>>, ep: &mut Episode<G>| -> Outcome {
                        if matches!(phase, Phase::Idle) {
                            if cutting {
                                return Outcome::None;
                            }
                            let perspectives: Vec<usize> = (0..num_agents)
                                .filter(|&si| ep.agent_active(game, si))
                                .collect();
                            if perspectives.is_empty() {
                                return Outcome::None;
                            }
                            let ctx = crate::policy::SearchCtx {
                                game,
                                enc: encoder,
                                reward,
                                rng: &mut ep.rng,
                                perms,
                                collect_interior,
                            };
                            let search = policy.begin_search(ctx, &ep.state, &perspectives);
                            *phase = Phase::Deciding {
                                search,
                                perspectives,
                                outstanding: 0,
                                rows: Vec::new(),
                                stride: 0,
                            };
                        }
                        let Phase::Deciding {
                            search,
                            outstanding,
                            rows,
                            stride,
                            ..
                        } = phase
                        else {
                            return Outcome::None;
                        };
                        if *outstanding != 0 {
                            return Outcome::None;
                        }
                        if !rows.is_empty() {
                            let view = crate::policy::RowsView::from_slice(rows, *stride);
                            policy.absorb(search, view, &mut ep.rng);
                            rows.clear();
                        }
                        let ctx = crate::policy::SearchCtx {
                            game,
                            enc: encoder,
                            reward,
                            rng: &mut ep.rng,
                            perms,
                            collect_interior,
                        };
                        let mut sink = crate::policy::RequestSink::default();
                        let status = policy.round(ctx, search, &mut sink);
                        if sink.is_empty() {
                            if status == crate::policy::RoundStatus::Done {
                                Outcome::Done
                            } else {
                                Outcome::None
                            }
                        } else {
                            let n = sink.len();
                            let (players, obs) = sink.into_parts();
                            Outcome::Emitted(players, obs, n)
                        }
                    };
                let mut outcomes: Vec<Outcome> = if let Some(pool) = self.thread_pool.as_ref() {
                    pool.install(|| {
                        use rayon::prelude::*;
                        phases
                            .par_iter_mut()
                            .zip(self.episodes.par_iter_mut())
                            .map(|(phase, ep)| run_slot(phase, ep))
                            .collect()
                    })
                } else {
                    phases
                        .iter_mut()
                        .zip(self.episodes.iter_mut())
                        .map(|(phase, ep)| run_slot(phase, ep))
                        .collect()
                };

                // Phase 2: merge emissions into the queue in rotation order; run completions
                // (finish + select + advance + flush) sequentially in the same order.
                let mut progressed = false;
                for k in 0..n_games {
                    let gi = (base_cursor + k) % n_games;
                    match std::mem::replace(&mut outcomes[gi], Outcome::None) {
                        Outcome::None => {}
                        Outcome::Emitted(players, obs, n) => {
                            q_players.extend(players);
                            q_obs.extend(obs);
                            q_dest.extend(std::iter::repeat_n(gi, n));
                            if let Phase::Deciding { outstanding, .. } = &mut phases[gi] {
                                *outstanding = n;
                            }
                            progressed = true;
                        }
                        Outcome::Done => {
                            let taken = std::mem::replace(&mut phases[gi], Phase::Idle);
                            let Phase::Deciding {
                                search,
                                perspectives,
                                ..
                            } = taken
                            else {
                                unreachable!()
                            };
                            let ep = &mut self.episodes[gi];
                            let ctx = crate::policy::SearchCtx {
                                game: &self.game,
                                enc: &*self.encoder,
                                reward: &*self.reward,
                                rng: &mut ep.rng,
                                perms: &self.perms,
                                collect_interior,
                            };
                            let results = self.policy.finish(ctx, search);
                            let meta: Vec<(usize, usize)> =
                                perspectives.iter().map(|&si| (gi, si)).collect();
                            let finished = {
                                let mut start = StartParts {
                                    dist: &mut *self.start_dist,
                                    rng: &mut self.buffer_rng,
                                };
                                process_tick(
                                    &self.game,
                                    &*self.encoder,
                                    &*self.reward,
                                    &self.policy,
                                    &self.learner,
                                    &self.learn_mask,
                                    self.sequential,
                                    gi..gi + 1,
                                    results,
                                    &meta,
                                    &mut self.episodes,
                                    &mut self.traj,
                                    &mut self.ticks,
                                    &mut self.policy_states,
                                    &mut self.episode_returns,
                                    &mut start,
                                    &mut out,
                                    &mut stats,
                                )
                            };
                            self.flush_finished(&finished, &mut out, &mut stats, &mut evaluator);
                            progressed = true;
                        }
                    }
                }

                // Phase 3: fire full batches; drain when nothing else can move.
                while q_players.len() >= batch_size {
                    fire_batch(
                        &mut q_players,
                        &mut q_obs,
                        &mut q_dest,
                        batch_size,
                        &mut phases,
                        &mut evaluator,
                        |ph| match ph {
                            Phase::Deciding {
                                outstanding,
                                rows,
                                stride,
                                ..
                            } => (outstanding, rows, stride),
                            Phase::Idle => unreachable!("row routed to an idle slot"),
                        },
                    );
                    progressed = true;
                }
                if !progressed {
                    if !q_players.is_empty() {
                        let n = q_players.len();
                        fire_batch(
                            &mut q_players,
                            &mut q_obs,
                            &mut q_dest,
                            n,
                            &mut phases,
                            &mut evaluator,
                            |ph| match ph {
                                Phase::Deciding {
                                    outstanding,
                                    rows,
                                    stride,
                                    ..
                                } => (outstanding, rows, stride),
                                Phase::Idle => unreachable!("row routed to an idle slot"),
                            },
                        );
                    } else if phases.iter().all(|p| matches!(p, Phase::Idle)) {
                        break;
                    }
                }
            }
            self.sweep_cursor = (base_cursor + 1) % n_games.max(1);
        }
        if fragments {
            flush_fragments_parts(
                &self.game,
                &self.policy,
                &self.learner,
                &*self.encoder,
                &self.learn_mask,
                self.sequential,
                &mut self.episodes,
                &mut self.traj,
                &mut evaluator,
                &mut out,
            );
        }
        (stats.infer_seconds, stats.infer_calls, stats.infer_rows) =
            (evaluator.seconds, evaluator.calls, evaluator.rows);
        stats.padded_rows = evaluator.padded_rows;
        (stats.cache_lookups, stats.cache_hits) =
            (evaluator.cache_lookups(), evaluator.cache_hits());
        self.infer_caches = caches;
        (out, stats)
    }

    fn flush_finished<F>(
        &mut self,
        finished: &[(usize, bool)],
        out: &mut Vec<L::Record>,
        stats: &mut CollectStats,
        evaluator: &mut Evaluator<'_, F>,
    ) where
        F: FnMut(usize, Vec<f32>, usize) -> Vec<f64>,
    {
        let tails = self.tail_values(finished, evaluator);
        let mut start = StartParts {
            dist: &mut *self.start_dist,
            rng: &mut self.buffer_rng,
        };
        flush_finished_parts(
            finished,
            &tails,
            &self.game,
            &self.policy,
            &self.learner,
            &*self.encoder,
            &mut self.episodes,
            &mut self.traj,
            &mut self.ticks,
            &mut self.policy_states,
            &mut self.episode_returns,
            &mut self.seeded,
            &mut start,
            out,
            stats,
        );
    }

    fn tail_values<F>(
        &mut self,
        finished: &[(usize, bool)],
        evaluator: &mut Evaluator<'_, F>,
    ) -> HashMap<(usize, usize), Vec<f64>>
    where
        F: FnMut(usize, Vec<f32>, usize) -> Vec<f64>,
    {
        tail_values_parts(
            finished,
            &self.game,
            &self.policy,
            &self.learner,
            &*self.encoder,
            self.sequential,
            &mut self.episodes,
            &self.traj,
            evaluator,
        )
    }
}

/// Snapshot and restore mutable collection state.
impl<G: Game + Sync, P: Policy, L: Learner<P::Evaluation>> Engine<G, P, L>
where
    G::State: Send,
{
    pub fn snapshot_bytes(
        &self,
        codec: &dyn StateCodec<State = G::State>,
    ) -> Result<Vec<u8>, String> {
        use crate::codec::bytes::*;
        let mut out = vec![4u8];
        let n_games = self.episodes.len();
        let num_agents = self.game.num_agents();
        put_u32(&mut out, n_games as u32);
        put_u32(&mut out, num_agents as u32);
        put_u64(&mut out, self.search_rng.state());
        put_u64(&mut out, self.buffer_rng.state());
        // The sweep cursor is result-bearing: it sets rotation start, hence window
        // composition; restore-continuation identity needs it.
        put_u64(&mut out, self.sweep_cursor as u64);
        for gi in 0..n_games {
            put_blob(&mut out, &codec.encode(&self.episodes[gi].state));
            put_u64(&mut out, self.episodes[gi].rng.state());
            put_u64(&mut out, self.ticks[gi] as u64);
            put_u8(&mut out, u8::from(self.seeded[gi]));
            put_f64s(&mut out, &self.episode_returns[gi]);
            put_u64(
                &mut out,
                self.policy.policy_state_to_u64(&self.policy_states[gi]),
            );
            for si in 0..num_agents {
                let steps = &self.traj[gi][si];
                put_u32(&mut out, steps.len() as u32);
                for step in steps {
                    put_f32s(&mut out, &step.obs);
                    self.policy.encode_eval(&step.evaluation, &mut out);
                    put_u64(&mut out, step.action as u64);
                    put_f64(&mut out, step.reward);
                    put_f32s(&mut out, &step.next_obs);
                    put_usizes(&mut out, &step.next_legal);
                    put_u8(&mut out, u8::from(step.terminal));
                }
            }
        }
        put_blob(
            &mut out,
            &self.start_dist.snapshot_bytes(&|s| codec.encode(s)),
        );
        Ok(out)
    }

    pub fn restore_bytes(
        &mut self,
        codec: &dyn StateCodec<State = G::State>,
        bytes: &[u8],
    ) -> Result<(), String> {
        use crate::codec::bytes::*;
        let mut r = Reader::new(bytes);
        let version = r.u8()?;
        if version != 4 {
            return Err("unsupported engine snapshot layout version".into());
        }
        let n_games = r.u32()? as usize;
        let num_agents = r.u32()? as usize;
        if n_games != self.episodes.len() || num_agents != self.game.num_agents() {
            return Err(format!(
                "snapshot shape ({n_games} games, {num_agents} agents) does not match the engine"
            ));
        }
        let search_rng = r.u64()?;
        let buffer_rng = r.u64()?;
        let sweep_cursor = r.u64()? as usize;
        let (c, h, w) = self.encoder.obs_shape();
        let obs_dim = c * h * w;
        let action_count = self.game.action_count();
        let horizon = self.game.truncation_horizon();
        let bool_byte = |b: u8| -> Result<bool, String> {
            match b {
                0 => Ok(false),
                1 => Ok(true),
                other => Err(format!("byte {other} is not a bool")),
            }
        };
        // Decode completely before mutation so restore is transactional.
        struct GameSlice<S, E, PS> {
            state: S,
            rng: u64,
            tick: usize,
            seeded: bool,
            returns: Vec<f64>,
            policy_state: PS,
            traj: Vec<Vec<Step<E>>>,
        }
        let mut slices: Vec<GameSlice<G::State, P::Evaluation, P::PolicyState>> =
            Vec::with_capacity(n_games);
        for _ in 0..n_games {
            let state = codec.decode(r.blob()?)?;
            codec.validate_decoded_state(&state, false)?;
            let rng = r.u64()?;
            let tick = r.u64()? as usize;
            if horizon.is_some_and(|hz| tick >= hz) {
                return Err(format!("tick {tick} at or past the truncation horizon"));
            }
            if tick > (1 << 48) {
                return Err(format!("implausible tick count {tick}"));
            }
            let seeded = bool_byte(r.u8()?)?;
            let returns = f64s(&mut r)?;
            if returns.len() != num_agents || returns.iter().any(|v| !v.is_finite()) {
                return Err("malformed episode-return vector".into());
            }
            let policy_state = self.policy.policy_state_from_u64(r.u64()?)?;
            let mut traj = Vec::with_capacity(num_agents);
            for _ in 0..num_agents {
                let n_steps = r.u32()? as usize;
                if n_steps > 1_000_000 {
                    return Err(format!("implausible trajectory length {n_steps}"));
                }
                let mut steps = Vec::with_capacity(n_steps);
                if n_steps > tick {
                    return Err(format!(
                        "{n_steps} buffered decisions exceed tick count {tick}"
                    ));
                }
                for _ in 0..n_steps {
                    let obs = f32s(&mut r)?;
                    if obs.len() != obs_dim {
                        return Err(format!("obs width {} != {obs_dim}", obs.len()));
                    }
                    let evaluation = self.policy.decode_eval(&mut r, action_count)?;
                    let action = r.u64()? as usize;
                    if action >= action_count {
                        return Err(format!("action {action} out of range"));
                    }
                    let reward = r.f64()?;
                    if !reward.is_finite() {
                        return Err("non-finite reward in trajectory".into());
                    }
                    let next_obs = f32s(&mut r)?;
                    if !(next_obs.is_empty() || next_obs.len() == obs_dim) {
                        return Err(format!("next_obs width {} != {obs_dim}", next_obs.len()));
                    }
                    let next_legal = usizes(&mut r)?;
                    if next_legal.iter().any(|&a| a >= action_count) {
                        return Err("next_legal action id out of range".into());
                    }
                    let terminal = bool_byte(r.u8()?)?;
                    if terminal {
                        return Err(
                            "buffered trajectories never hold terminal steps (they flush at episode end)"
                                .into(),
                        );
                    }
                    steps.push(Step {
                        obs,
                        evaluation,
                        action,
                        reward,
                        next_obs,
                        next_legal,
                        terminal,
                    });
                }
                traj.push(steps);
            }
            slices.push(GameSlice {
                state,
                rng,
                tick,
                seeded,
                returns,
                policy_state,
                traj,
            });
        }
        let start_blob = r.blob()?.to_vec();
        r.done()?;
        self.start_dist.restore_bytes(&start_blob, &|b| {
            let s = codec.decode(b)?;
            codec.validate_decoded_state(&s, false)?;
            Ok(s)
        })?;
        self.search_rng = SplitMix64::from_state(search_rng);
        self.buffer_rng = SplitMix64::from_state(buffer_rng);
        self.sweep_cursor = sweep_cursor;
        for (gi, slice) in slices.into_iter().enumerate() {
            self.episodes[gi].state = slice.state;
            self.episodes[gi].rng = SplitMix64::from_state(slice.rng);
            self.ticks[gi] = slice.tick;
            self.seeded[gi] = slice.seeded;
            self.episode_returns[gi] = slice.returns;
            self.policy_states[gi] = slice.policy_state;
            self.traj[gi] = slice.traj;
        }
        // Numeric generations do not identify weights across restored processes.
        if let Some(caches) = self.infer_caches.as_mut() {
            for cache in caches.iter_mut() {
                cache.force_clear();
            }
        }
        Ok(())
    }
}

/// One decision tick for `games`: record emission, action selection, stepping, and
/// reward attribution. Field-split from `Engine` so group workers can run it.
#[allow(clippy::too_many_arguments)]
fn process_tick<G, P, L>(
    game: &G,
    encoder: &dyn StateEncoder<State = G::State>,
    reward: &dyn Reward<Event = G::Event>,
    policy: &P,
    learner: &L,
    learn_mask: &[bool],
    sequential: bool,
    games: std::ops::Range<usize>,
    evals: Vec<(P::Evaluation, Vec<crate::learner::InteriorTarget>)>,
    meta: &[(usize, usize)],
    episodes: &mut [Episode<G>],
    traj: &mut [Vec<Vec<Step<P::Evaluation>>>],
    ticks: &mut [usize],
    policy_states: &mut [P::PolicyState],
    episode_returns: &mut [Vec<f64>],
    start: &mut StartParts<'_, G::State>,
    out: &mut Vec<L::Record>,
    stats: &mut CollectStats,
) -> Vec<(usize, bool)>
where
    G: Game + Sync,
    G::State: Send,
    P: Policy,
    L: Learner<P::Evaluation>,
{
    let num_agents = game.num_agents();
    assert_eq!(
        evals.len(),
        meta.len(),
        "policy returned {} evaluations for {} requests — one per request, in order",
        evals.len(),
        meta.len()
    );
    let mut acted: Vec<Vec<Option<usize>>> = vec![vec![None; num_agents]; episodes.len()];
    for ((eval, targets), &(gi, si)) in evals.into_iter().zip(meta.iter()) {
        stats.decisions += 1;
        policy.fold_telemetry(&eval, stats);
        if !learn_mask[si] {
            let rel = policy.select(&eval, &mut policy_states[gi], &mut episodes[gi].rng);
            acted[gi][si] = Some(rel);
            continue;
        }
        out.extend(learner.eval_records(&eval, targets, encoder, si, &mut episodes[gi].rng));
        let rel = policy.select(&eval, &mut policy_states[gi], &mut episodes[gi].rng);
        acted[gi][si] = Some(rel);
        traj[gi][si].push(Step {
            obs: episodes[gi].observe(encoder, si),
            evaluation: eval,
            action: rel,
            reward: 0.0,
            next_obs: Vec::new(),
            next_legal: Vec::new(),
            terminal: false,
        });
    }

    // MaxN consumers require value supervision for non-mover perspectives too.
    if policy.evaluates_all_perspectives(sequential, num_agents) {
        let action_count = game.action_count();
        for (gi, agents) in acted.iter().enumerate() {
            if agents.iter().all(|s| s.is_none()) {
                continue;
            }
            for (si, slot) in agents.iter().enumerate() {
                if slot.is_none() && learn_mask[si] {
                    if let Some(evaluation) = learner.value_only_evaluation(action_count) {
                        traj[gi][si].push(Step {
                            obs: episodes[gi].observe(encoder, si),
                            evaluation,
                            action: 0,
                            reward: 0.0,
                            next_obs: Vec::new(),
                            next_legal: Vec::new(),
                            terminal: false,
                        });
                    }
                }
            }
        }
    }

    let horizon = game.truncation_horizon();
    let mut finished: Vec<(usize, bool)> = Vec::new();
    for gi in games.clone() {
        let agents = std::mem::take(&mut acted[gi]);
        let joint: Vec<usize> = agents.iter().map(|a| a.unwrap_or(0)).collect();
        let (mut trace, terminal) = episodes[gi].advance(game, &joint);
        ticks[gi] += 1;
        let truncated = horizon.is_some_and(|h| ticks[gi] >= h) && !terminal;
        if truncated {
            game.mark_truncation(&episodes[gi].state, &mut trace);
            assert!(
                trace.iter().all(|(agent, _)| *agent < num_agents),
                "mark_truncation pushed an event for an out-of-range agent"
            );
        }
        let mut tick_rewards = vec![0.0; num_agents];
        for (agent, e) in &trace {
            tick_rewards[*agent] += reward.step_reward(e, *agent);
        }
        let needs_next_obs = learner.needs_next_obs();
        for (si, action) in agents.iter().enumerate() {
            let reward = tick_rewards[si];
            episode_returns[gi][si] += reward;
            if action.is_some() {
                let (next_obs, next_legal) = if needs_next_obs {
                    (
                        episodes[gi].observe(encoder, si),
                        game.legal_actions(&episodes[gi].state, si),
                    )
                } else {
                    (Vec::new(), Vec::new())
                };
                if let Some(step) = traj[gi][si].last_mut() {
                    step.reward = reward;
                    step.next_obs = next_obs;
                    step.next_legal = next_legal;
                    step.terminal = terminal;
                }
            } else if let Some(step) = traj[gi][si].last_mut() {
                // Sequential terminal events may reward an agent that did not act this tick.
                step.reward += reward;
                step.terminal |= terminal;
            }
        }
        if !terminal {
            start.observe(&episodes[gi].state);
        }
        if terminal || truncated {
            finished.push((gi, terminal));
        }
    }

    finished
}

/// Emit episode records and respawn finished games. Field-split like [`process_tick`].
#[allow(clippy::too_many_arguments)]
fn flush_finished_parts<G, P, L>(
    finished: &[(usize, bool)],
    tails: &HashMap<(usize, usize), Vec<f64>>,
    game: &G,
    policy: &P,
    learner: &L,
    encoder: &dyn StateEncoder<State = G::State>,
    episodes: &mut [Episode<G>],
    traj: &mut [Vec<Vec<Step<P::Evaluation>>>],
    ticks: &mut [usize],
    policy_states: &mut [P::PolicyState],
    episode_returns: &mut [Vec<f64>],
    seeded: &mut [bool],
    start: &mut StartParts<'_, G::State>,
    out: &mut Vec<L::Record>,
    stats: &mut CollectStats,
) where
    G: Game + Sync,
    G::State: Send,
    P: Policy,
    L: Learner<P::Evaluation>,
{
    let num_agents = game.num_agents();

    for &(gi, _) in finished {
        let mut ep_reward = vec![0.0; num_agents];
        for (si, ep_slot) in ep_reward.iter_mut().enumerate() {
            let steps = std::mem::take(&mut traj[gi][si]);
            *ep_slot = std::mem::take(&mut episode_returns[gi][si]);
            if steps.is_empty() {
                continue;
            }
            let tail = tails.get(&(gi, si)).cloned().unwrap_or_default();
            out.extend(learner.episode_records(&steps, &tail, encoder, si, &mut episodes[gi].rng));
        }
        stats.episodes.push(EpisodeSummary {
            reward: ep_reward,
            length: ticks[gi],
            seeded: seeded[gi],
        });
        let choice = start.choose();
        match choice {
            Start::Restore(state) => {
                Episode::assert_decision_state(game, &state);
                episodes[gi].state = state;
                seeded[gi] = true;
            }
            Start::Fresh => {
                episodes[gi].reset(game);
                seeded[gi] = false;
            }
        }
        ticks[gi] = 0;
        policy_states[gi] = policy.begin_episode(&mut episodes[gi].rng);
    }
}

/// Records a fragment cut would emit now: collected records plus buffered learning-player
/// steps — the floor comparison must count buffered steps or the loop would never stop.
fn fragment_potential<E>(out_len: usize, traj: &[Vec<Vec<Step<E>>>], learn_mask: &[bool]) -> usize {
    out_len
        + traj
            .iter()
            .map(|g| {
                g.iter()
                    .enumerate()
                    .filter(|&(si, _)| learn_mask[si])
                    .map(|(_, steps)| steps.len())
                    .sum::<usize>()
            })
            .sum::<usize>()
}

/// Forward the first `take` queued rows in one evaluator call and route the results to the
/// destination slots' row buffers, decrementing their outstanding counts.
fn fire_batch<SE, F>(
    q_players: &mut Vec<usize>,
    q_obs: &mut Vec<f32>,
    q_dest: &mut Vec<usize>,
    take: usize,
    phases: &mut [SE],
    evaluator: &mut Evaluator<'_, F>,
    mut route: impl FnMut(&mut SE) -> (&mut usize, &mut Vec<f64>, &mut usize),
) where
    F: FnMut(usize, Vec<f32>, usize) -> Vec<f64>,
{
    let dim = if q_players.is_empty() {
        0
    } else {
        q_obs.len() / q_players.len()
    };
    let players: Vec<usize> = q_players.drain(..take).collect();
    let obs: Vec<f32> = q_obs.drain(..take * dim).collect();
    let dests: Vec<usize> = q_dest.drain(..take).collect();
    let rows = evaluator.forward(&players, obs, take);
    let stride = rows.len() / take;
    for (i, &gi) in dests.iter().enumerate() {
        let (outstanding, buf, st) = route(&mut phases[gi]);
        *st = stride;
        buf.extend_from_slice(&rows[i * stride..(i + 1) * stride]);
        *outstanding -= 1;
    }
}

/// Roll back an aborted window: a failed collect (callback error, cancellation) leaves its
/// buffered steps in `traj`, and a retry must not flush another version's steps into its batch.
/// A successful window flushes everything, so this is a no-op except after an abort.
fn discard_fragments<E>(traj: &mut [Vec<Vec<Step<E>>>]) {
    for game in traj.iter_mut() {
        for steps in game.iter_mut() {
            steps.clear();
        }
    }
}

/// Cut the window: bootstrap every live learning trajectory from its own tail and emit its
/// records; episode state, ticks, and telemetry persist into the next window.
#[allow(clippy::too_many_arguments)]
fn flush_fragments_parts<G, P, L, F>(
    game: &G,
    policy: &P,
    learner: &L,
    encoder: &dyn StateEncoder<State = G::State>,
    learn_mask: &[bool],
    sequential: bool,
    episodes: &mut [Episode<G>],
    traj: &mut [Vec<Vec<Step<P::Evaluation>>>],
    evaluator: &mut Evaluator<'_, F>,
    out: &mut Vec<L::Record>,
) where
    G: Game + Sync,
    G::State: Send,
    P: Policy,
    L: Learner<P::Evaluation>,
    F: FnMut(usize, Vec<f32>, usize) -> Vec<f64>,
{
    let live: Vec<(usize, bool)> = traj
        .iter()
        .enumerate()
        .filter(|(_, g)| g.iter().any(|steps| !steps.is_empty()))
        .map(|(gi, _)| (gi, false))
        .collect();
    if live.is_empty() {
        return;
    }
    let tails = tail_values_parts(
        &live, game, policy, learner, encoder, sequential, episodes, traj, evaluator,
    );
    for &(gi, _) in &live {
        for si in 0..game.num_agents() {
            if !learn_mask[si] {
                traj[gi][si].clear();
                continue;
            }
            let steps = std::mem::take(&mut traj[gi][si]);
            if steps.is_empty() {
                continue;
            }
            let tail = tails.get(&(gi, si)).cloned().unwrap_or_default();
            out.extend(learner.episode_records(&steps, &tail, encoder, si, &mut episodes[gi].rng));
        }
    }
}

/// Truncation-tail bootstrapping, field-split like [`process_tick`].
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_range_loop)]
fn tail_values_parts<G, P, L, F>(
    finished: &[(usize, bool)],
    game: &G,
    policy: &P,
    learner: &L,
    encoder: &dyn StateEncoder<State = G::State>,
    sequential: bool,
    episodes: &mut [Episode<G>],
    traj: &[Vec<Vec<Step<P::Evaluation>>>],
    evaluator: &mut Evaluator<'_, F>,
) -> HashMap<(usize, usize), Vec<f64>>
where
    G: Game + Sync,
    G::State: Send,
    P: Policy,
    L: Learner<P::Evaluation>,
    F: FnMut(usize, Vec<f32>, usize) -> Vec<f64>,
{
    let mut tails: HashMap<(usize, usize), Vec<f64>> = HashMap::new();
    if !learner.uses_episode_tail() {
        return tails;
    }
    let a = game.action_count();
    let num_agents = game.num_agents();
    let all_perspectives = (policy.evaluates_all_perspectives(sequential, num_agents)
        && learner.value_only_evaluation(a).is_some())
        || learner.tails_all_trajectories();
    let mut obs_flat: Vec<f32> = Vec::new();
    let mut meta: Vec<(usize, usize)> = Vec::new();
    for &(gi, terminal) in finished {
        if terminal {
            continue;
        }
        for si in 0..num_agents {
            if (all_perspectives || episodes[gi].agent_active(game, si)) && !traj[gi][si].is_empty()
            {
                obs_flat.extend(episodes[gi].observe(encoder, si));
                meta.push((gi, si));
            }
        }
    }
    if !meta.is_empty() {
        let players: Vec<usize> = meta.iter().map(|&(_, si)| si).collect();
        let q = evaluator.forward(&players, obs_flat, meta.len());
        let stride = q.len() / meta.len();
        // Cancellation yields zero-width rows; empty tails degrade to the terminal path and
        // the aborted collect's records are discarded by the caller.
        if stride == 0 {
            return tails;
        }
        for (i, &(gi, si)) in meta.iter().enumerate() {
            let row = &q[i * stride..(i + 1) * stride];
            let state = &episodes[gi].state;
            // Sequential non-mover rows still bootstrap over the mover's available actions;
            // using `si` here would turn a valid sparse-action tail into an empty one.
            let legal = match game.actor(state) {
                Actor::Agent(mover) => game.legal_actions(state, mover),
                Actor::Simultaneous => game.legal_actions(state, si),
                Actor::Chance => unreachable!("chance actors are not searched"),
            };
            tails.insert((gi, si), learner.tail_from_row(row, a, &legal, encoder, si));
        }
    }
    tails
}
