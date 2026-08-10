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

/// Summary of one finished episode.
#[derive(Clone)]
pub struct EpisodeSummary {
    pub reward: Vec<f64>,
    pub length: usize,
    pub seeded: bool,
}

/// Telemetry for one collection call.
#[derive(Default, Clone)]
pub struct CollectStats {
    pub episodes: Vec<EpisodeSummary>,
    pub decisions: usize,
    pub max_depth: i32,
    pub sum_leaves: f64,
    pub sum_rounds: f64,
    pub sum_expansions: f64,
    pub sum_sigma: f64,
    pub sum_disagreement: f64,
    pub infer_seconds: f64,
    pub infer_calls: usize,
    pub infer_rows: usize,
    pub cache_lookups: usize,
    pub cache_hits: usize,
    pub sum_terminal_sims: usize,
    pub sum_depthcap_sims: usize,
    pub sum_shared_rows: usize,
    pub sum_fresh_rows: usize,
    pub sum_hit_rows: usize,
    pub sum_extra_eval_rows: usize,
}

/// Engine-level rollout parameters.
pub struct EngineParams {
    pub n_games: usize,
    pub seed: u64,
    /// 1 = the classic lockstep collect; 2 = double-buffered collect (two fixed game groups
    /// alternating rounds so tree work overlaps inference). Gated by the binding on policies
    /// with a pooled search.
    pub n_groups: usize,
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
    n_groups: usize,
    buffer_rng: SplitMix64,
    seeded: Vec<bool>,
    policy_states: Vec<P::PolicyState>,
    ticks: Vec<usize>,
    traj: Vec<Vec<Vec<Step<P::Evaluation>>>>,
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
        assert!(
            matches!(params.n_groups, 1 | 2),
            "n_groups must be 1 or 2 (got {})",
            params.n_groups
        );
        assert!(
            params.n_groups == 1 || params.n_games >= 2,
            "n_groups=2 needs at least 2 games to split"
        );
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
            n_groups: params.n_groups,
            buffer_rng,
            seeded,
            policy_states,
            ticks,
            traj,
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
        let mut evaluator = Evaluator::new(&mut infer, mode, cache_slice);

        while out.len() < n_records {
            let mut requests: Vec<(G::State, usize)> = Vec::new();
            let mut meta: Vec<(usize, usize)> = Vec::new();
            for (gi, ep) in self.episodes.iter().enumerate() {
                for si in 0..num_agents {
                    if ep.agent_active(&self.game, si) {
                        requests.push((ep.state.clone(), si));
                        meta.push((gi, si));
                    }
                }
            }
            if requests.is_empty() {
                break;
            }

            let search_seed = self.search_rng.next_u64();
            let evals = self.policy.evaluate(
                &self.game,
                &*self.encoder,
                &*self.reward,
                requests,
                search_seed,
                collect_interior,
                &mut evaluator,
            );

            let finished = process_tick(
                &self.game,
                &*self.encoder,
                &*self.reward,
                &self.policy,
                &self.learner,
                &self.learn_mask,
                self.sequential,
                0..self.episodes.len(),
                evals,
                &meta,
                &mut self.episodes,
                &mut self.traj,
                &mut self.ticks,
                &mut self.policy_states,
                &mut self.episode_returns,
                &mut *self.start_dist,
                &mut self.buffer_rng,
                &mut out,
                &mut stats,
            );
            self.flush_finished(&finished, &mut out, &mut stats, &mut evaluator);
        }
        (stats.infer_seconds, stats.infer_calls, stats.infer_rows) =
            (evaluator.seconds, evaluator.calls, evaluator.rows);
        (stats.cache_lookups, stats.cache_hits) =
            (evaluator.cache_lookups(), evaluator.cache_hits());
        self.infer_caches = caches;
        (out, stats)
    }

    /// Double-buffered collect: games split into two fixed groups whose search rounds
    /// alternate, so one group's tree work overlaps the other's inference (which runs on a
    /// submitter thread owning the callback). Deterministic: static membership, strict
    /// alternation, game-index row order. Once a batch boundary observes a new weights
    /// generation, older rows are cleared and never served (see [`Evaluator::ingest`]).
    /// Requires a policy with a pooled search (binding-gated; panics
    /// here as the backstop), shared inference, and no truncation-tail bootstrapping.
    pub fn collect_grouped<F>(
        &mut self,
        n_records: usize,
        mode: InferMode,
        infer: F,
    ) -> (Vec<L::Record>, CollectStats)
    where
        F: FnMut(usize, Vec<f32>, usize) -> Vec<f64> + Send,
    {
        assert_eq!(self.n_groups, 2, "collect_grouped requires n_groups=2");
        assert!(
            matches!(mode, InferMode::Shared),
            "grouped collect supports shared inference only"
        );
        assert!(
            self.game.truncation_horizon().is_none() || !self.learner.uses_episode_tail(),
            "grouped collect does not support truncation-tail bootstrapping"
        );
        if n_records == 0 {
            return (Vec::new(), CollectStats::default());
        }
        let n_games = self.episodes.len();
        let half = n_games / 2;
        let ranges = [0..half, half..n_games];

        let mut out: Vec<L::Record> = Vec::new();
        let mut stats = CollectStats::default();
        let collect_interior = self.learner.needs_interior();
        let mut caches = self.infer_caches.take();
        if let Some(c) = caches.as_mut() {
            for cache in c.iter_mut() {
                cache.begin_collect();
            }
        }
        let cache_slice = caches.as_mut().map(|c| &mut c[..1]);
        let mut noop_infer = |_: usize, _: Vec<f32>, _: usize| -> Vec<f64> {
            unreachable!("grouped collect forwards through the submitter thread")
        };
        let mut evaluator = Evaluator::new(&mut noop_infer, mode, cache_slice);

        let Engine {
            game,
            encoder,
            reward,
            policy,
            learner,
            episodes,
            search_rng,
            start_dist,
            learn_mask,
            episode_returns,
            sequential,
            buffer_rng,
            seeded,
            policy_states,
            ticks,
            traj,
            ..
        } = self;
        let game: &G = game;
        let encoder: &dyn StateEncoder<State = G::State> = &**encoder;
        let reward: &dyn Reward<Event = G::Event> = &**reward;
        let policy: &P = &*policy;
        let learner: &L = &*learner;
        let learn_mask: &[bool] = learn_mask;
        let sequential = *sequential;
        let num_agents = game.num_agents();
        let no_tails: HashMap<(usize, usize), Vec<f64>> = HashMap::new();

        std::thread::scope(|scope| {
            let (job_tx, job_rx) =
                std::sync::mpsc::channel::<(usize, crate::rollout::evaluator::StagedBatch)>();
            let (res_tx, res_rx) = std::sync::mpsc::channel::<(
                usize,
                crate::rollout::evaluator::StagedBatch,
                Vec<f64>,
                f64,
                usize,
            )>();
            let mut infer = infer;
            scope.spawn(move || {
                while let Ok((gid, mut staged)) = job_rx.recv() {
                    let obs = std::mem::take(&mut staged.obs_flat);
                    let t = std::time::Instant::now();
                    let (rows, calls) = crate::rollout::evaluator::run_infer(
                        &mut infer,
                        InferMode::Shared,
                        &staged.players,
                        obs,
                        staged.n,
                        staged.dim,
                    );
                    let secs = t.elapsed().as_secs_f64();
                    if res_tx.send((gid, staged, rows, secs, calls)).is_err() {
                        break;
                    }
                }
            });

            type Slot<'c, G> = Option<(
                crate::policies::tree::mcts::PooledSearch<'c, G>,
                Vec<(usize, usize)>,
            )>;
            let mut slots: [Slot<'_, G>; 2] = [None, None];
            let mut pending: std::collections::VecDeque<usize> = [0usize, 1].into();
            let mut inflight = 0usize;

            'outer: loop {
                while let Some(gid) = pending.pop_front() {
                    loop {
                        if slots[gid].is_none() {
                            let mut requests: Vec<(G::State, usize)> = Vec::new();
                            let mut meta: Vec<(usize, usize)> = Vec::new();
                            for gi in ranges[gid].clone() {
                                for si in 0..num_agents {
                                    if episodes[gi].agent_active(game, si) {
                                        requests.push((episodes[gi].state.clone(), si));
                                        meta.push((gi, si));
                                    }
                                }
                            }
                            if requests.is_empty() {
                                break;
                            }
                            let seed = search_rng.next_u64();
                            let pool = policy
                                .begin_pooled(
                                    game,
                                    encoder,
                                    reward,
                                    requests,
                                    seed,
                                    collect_interior,
                                )
                                .expect("grouped collect needs a pooled-search policy");
                            slots[gid] = Some((pool, meta));
                        }
                        let (pool, _) = slots[gid].as_mut().expect("slot filled above");
                        if pool.finished() {
                            let (pool, meta) = slots[gid].take().expect("slot filled above");
                            let evals = policy.pooled_into_evals(pool.into_evaluations());
                            let finished = process_tick(
                                game,
                                encoder,
                                reward,
                                policy,
                                learner,
                                learn_mask,
                                sequential,
                                ranges[gid].clone(),
                                evals,
                                &meta,
                                episodes,
                                traj,
                                ticks,
                                policy_states,
                                episode_returns,
                                &mut **start_dist,
                                buffer_rng,
                                &mut out,
                                &mut stats,
                            );
                            flush_finished_parts(
                                &finished,
                                &no_tails,
                                game,
                                policy,
                                learner,
                                encoder,
                                episodes,
                                traj,
                                ticks,
                                policy_states,
                                episode_returns,
                                seeded,
                                &mut **start_dist,
                                buffer_rng,
                                &mut out,
                                &mut stats,
                            );
                            if out.len() >= n_records {
                                break 'outer;
                            }
                            continue;
                        }
                        let mut batch = evaluator.batch();
                        pool.stage_round(&mut batch);
                        let staged = batch.into_staged();
                        if staged.n == 0 {
                            let rows = evaluator.ingest(staged, Vec::new(), 0.0, 0);
                            let (pool, _) = slots[gid].as_mut().expect("slot filled above");
                            pool.apply_rows(&rows);
                            continue;
                        }
                        if job_tx.send((gid, staged)).is_err() {
                            break 'outer;
                        }
                        inflight += 1;
                        break;
                    }
                }
                if inflight == 0 {
                    break;
                }
                let (gid, staged, rows_out, secs, calls) =
                    res_rx.recv().expect("submitter alive while jobs in flight");
                inflight -= 1;
                let rows = evaluator.ingest(staged, rows_out, secs, calls);
                if let Some((pool, _)) = slots[gid].as_mut() {
                    pool.apply_rows(&rows);
                }
                pending.push_back(gid);
            }
            drop(job_tx);
            while inflight > 0 {
                match res_rx.recv() {
                    Ok((_, staged, rows_out, secs, calls)) => {
                        let _ = evaluator.ingest(staged, rows_out, secs, calls);
                        inflight -= 1;
                    }
                    Err(_) => break,
                }
            }
        });

        (stats.infer_seconds, stats.infer_calls, stats.infer_rows) =
            (evaluator.seconds, evaluator.calls, evaluator.rows);
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
            &mut *self.start_dist,
            &mut self.buffer_rng,
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
        let mut tails: HashMap<(usize, usize), Vec<f64>> = HashMap::new();
        if !self.learner.uses_episode_tail() {
            return tails;
        }
        let a = self.game.action_count();
        let num_agents = self.game.num_agents();
        // Value-only trajectories need a tail for every consumed perspective, not only the mover.
        let all_perspectives = self
            .policy
            .evaluates_all_perspectives(self.sequential, num_agents)
            && self.learner.value_only_evaluation(a).is_some();
        let mut obs_flat: Vec<f32> = Vec::new();
        let mut meta: Vec<(usize, usize)> = Vec::new();
        for &(gi, terminal) in finished {
            if terminal {
                continue;
            }
            for si in 0..num_agents {
                if (all_perspectives || self.episodes[gi].agent_active(&self.game, si))
                    && !self.traj[gi][si].is_empty()
                {
                    obs_flat.extend(self.episodes[gi].observe(&*self.encoder, si));
                    meta.push((gi, si));
                }
            }
        }
        if !meta.is_empty() {
            let players: Vec<usize> = meta.iter().map(|&(_, si)| si).collect();
            let q = evaluator.forward(&players, obs_flat, meta.len());
            let stride = q.len() / meta.len();
            for (i, &(gi, si)) in meta.iter().enumerate() {
                let row = &q[i * stride..(i + 1) * stride];
                let state = &self.episodes[gi].state;
                // Sequential non-mover rows still bootstrap over the mover's available actions;
                // using `si` here would turn a valid sparse-action tail into an empty one.
                let legal = match self.game.actor(state) {
                    Actor::Agent(mover) => self.game.legal_actions(state, mover),
                    Actor::Simultaneous => self.game.legal_actions(state, si),
                    Actor::Chance => unreachable!("chance actors are not searched"),
                };
                tails.insert(
                    (gi, si),
                    self.learner
                        .tail_from_row(row, a, &legal, &*self.encoder, si),
                );
            }
        }
        tails
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
        let mut out = vec![2u8];
        let n_games = self.episodes.len();
        let num_agents = self.game.num_agents();
        put_u32(&mut out, n_games as u32);
        put_u32(&mut out, num_agents as u32);
        put_u64(&mut out, self.search_rng.state());
        put_u64(&mut out, self.buffer_rng.state());
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
        if r.u8()? != 2 {
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
/// reward attribution. Field-split from `Engine` so grouped collects can run it while
/// pooled searches hold shared borrows of the game context. Returns finished episodes.
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
    evals: Vec<P::Evaluation>,
    meta: &[(usize, usize)],
    episodes: &mut [Episode<G>],
    traj: &mut [Vec<Vec<Step<P::Evaluation>>>],
    ticks: &mut [usize],
    policy_states: &mut [P::PolicyState],
    episode_returns: &mut [Vec<f64>],
    start_dist: &mut dyn StartDistribution<G::State>,
    buffer_rng: &mut SplitMix64,
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
    let mut acted: Vec<Vec<Option<usize>>> = vec![vec![None; num_agents]; episodes.len()];
    for (mut eval, &(gi, si)) in evals.into_iter().zip(meta.iter()) {
        stats.decisions += 1;
        policy.fold_telemetry(&eval, stats);
        if !learn_mask[si] {
            let rel = policy.select(&eval, &mut policy_states[gi], &mut episodes[gi].rng);
            acted[gi][si] = Some(rel);
            continue;
        }
        out.extend(learner.eval_records(&mut eval, encoder, si, &mut episodes[gi].rng));
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
            start_dist.observe(&episodes[gi].state, &mut *buffer_rng);
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
    start_dist: &mut dyn StartDistribution<G::State>,
    buffer_rng: &mut SplitMix64,
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
        match start_dist.choose(&mut *buffer_rng) {
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
