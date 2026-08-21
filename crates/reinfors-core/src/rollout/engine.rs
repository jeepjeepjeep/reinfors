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
use crate::rollout::infer_service::ServiceHost;
use crate::rollout::start::{AlwaysInitialState, Start, StartDistribution};

pub use crate::stats::EpisodeSummary;

pub use crate::stats::CollectStats;

/// Start-distribution access: exclusive on the classic path, mutex-shared across group
/// workers.
pub(crate) struct StartParts<'a, S> {
    pub dist: &'a mut dyn StartDistribution<S>,
    pub rng: &'a mut SplitMix64,
}

pub(crate) enum StartAccess<'a, 'b, S> {
    Shared(&'b std::sync::Mutex<StartParts<'a, S>>),
}

impl<S> StartAccess<'_, '_, S> {
    fn observe(&mut self, state: &S) {
        let StartAccess::Shared(shared) = self;
        let mut parts = shared.lock().expect("start access poisoned");
        let StartParts { dist, rng } = &mut *parts;
        dist.observe(state, &mut **rng);
    }

    fn choose(&mut self) -> Start<S> {
        let StartAccess::Shared(shared) = self;
        let mut parts = shared.lock().expect("start access poisoned");
        let StartParts { dist, rng } = &mut *parts;
        dist.choose(&mut **rng)
    }
}

/// Engine-level rollout parameters.
pub struct EngineParams {
    pub n_games: usize,
    pub seed: u64,
    /// 1 = the classic lockstep collect; 2 = grouped collect (two game groups on worker
    /// threads so search overlaps inference).
    pub n_groups: usize,
    /// Fixed call shape in rows (None = off); see the inference contract.
    pub pad_rows_to: Option<usize>,
    /// Scheduler firing threshold in rows (None = max(1, n_games / 2)).
    pub batch_size: Option<usize>,
    /// Worker threads running search rounds (None = 1).
    pub n_threads: Option<usize>,
}

impl Default for EngineParams {
    fn default() -> Self {
        EngineParams {
            n_games: 1,
            seed: 0,
            n_groups: 1,
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
    n_groups: usize,
    pad_rows_to: Option<usize>,
    group_rngs: Vec<SplitMix64>,
    sharded_caches: Option<Vec<crate::rollout::infer_cache::ShardedInferCache>>,
    buffer_rng: SplitMix64,
    seeded: Vec<bool>,
    policy_states: Vec<P::PolicyState>,
    ticks: Vec<usize>,
    traj: Vec<Vec<Vec<Step<P::Evaluation>>>>,
    perms: crate::encoder::PermTable,
    batch_size: usize,
    sweep_cursor: usize,
    thread_pool: rayon::ThreadPool,
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
        assert!(
            params.batch_size.is_none_or(|b| b >= 1),
            "batch_size must be >= 1"
        );
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
        // Persistent per-group streams: results must not depend on collection chunking.
        let group_rngs = (0..params.n_groups)
            .map(|gi| {
                SplitMix64::new(
                    params.seed
                        ^ 0x7F4A_7C15_9E37_79B9_u64.wrapping_mul(gi as u64 + 1)
                        ^ 0xA076_1D64_78BD_642F,
                )
            })
            .collect();
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
            n_groups: params.n_groups,
            pad_rows_to: params.pad_rows_to,
            group_rngs,
            sharded_caches: None,
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
            thread_pool: rayon::ThreadPoolBuilder::new()
                .num_threads(params.n_threads.unwrap_or(1).max(1))
                .build()
                .expect("engine thread pool"),
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

    /// Install shared sharded caches for grouped collection (slot layout as
    /// [`Self::with_infer_caches`]: shared slot then one per player).
    pub fn with_sharded_infer_caches(
        mut self,
        caches: Vec<crate::rollout::infer_cache::ShardedInferCache>,
    ) -> Self {
        assert_eq!(self.n_groups, 2, "sharded caches require n_groups=2");
        assert_eq!(
            caches.len(),
            self.game.num_agents() + 1,
            "one cache per slot: shared + one per player"
        );
        self.sharded_caches = Some(caches);
        self
    }

    /// Install one shared cache followed by one cache per player.
    pub fn with_infer_caches(mut self, caches: Vec<InferCache>) -> Self {
        assert_eq!(
            self.n_groups, 1,
            "exclusive caches require n_groups=1; grouped engines use sharded caches"
        );
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
        P::Evaluation: Send,
        P::PolicyState: Send,
        L: Sync,
        L::Record: Send,
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
        P::Evaluation: Send,
        P::PolicyState: Send,
        L: Sync,
        L::Record: Send,
    {
        assert_eq!(
            self.n_groups, 1,
            "n_groups=2 engines collect via collect_grouped_hosted"
        );
        let mut out: Vec<L::Record> = Vec::new();
        let mut stats = CollectStats::default();
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
        // Free-running scheduler: worker tasks run search rounds AND completions
        // (finish/select/advance/records; terminal flushes) and feed per-player queues
        // through a channel; the callback fires on this thread the moment a queue holds
        // `batch_size` rows, overlapping inference with rounds and completions. The
        // scheduler thread keeps only evaluator work (fires, truncation tails), floor
        // accounting, and admission.
        {
            let n_games = self.episodes.len();
            let batch_size = self.batch_size;
            let base_cursor = self.sweep_cursor % n_games.max(1);
            let game = &self.game;
            let encoder = &*self.encoder;
            let reward = &*self.reward;
            let policy = &self.policy;
            let learner = &self.learner;
            let learn_mask = &self.learn_mask;
            let sequential = self.sequential;
            let perms = &self.perms;
            let slots: Vec<std::sync::Mutex<SlotCtx<'_, G, P>>> = self
                .episodes
                .iter_mut()
                .zip(self.traj.iter_mut())
                .zip(self.ticks.iter_mut())
                .zip(self.policy_states.iter_mut())
                .zip(self.episode_returns.iter_mut())
                .zip(self.seeded.iter_mut())
                .map(|(((((ep, traj), tick), policy_state), returns), seeded)| {
                    std::sync::Mutex::new(SlotCtx {
                        ep,
                        traj,
                        tick,
                        policy_state,
                        returns,
                        seeded,
                    })
                })
                .collect();
            let start_parts = std::sync::Mutex::new(StartParts {
                dist: &mut *self.start_dist,
                rng: &mut self.buffer_rng,
            });
            let (tx, rx) = std::sync::mpsc::channel::<Msg<P::Search<G::State>, L::Record>>();

            let task = |gi: usize,
                        work: Work<P::Search<G::State>>,
                        tx: MsgSender<P::Search<G::State>, L::Record>| {
                let run = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let mut guard = slots[gi].lock().expect("slot lock");
                    let slot: &mut SlotCtx<'_, G, P> = &mut guard;
                    let (mut search, perspectives) = match work {
                        Work::Begin => {
                            let perspectives: Vec<usize> = (0..game.num_agents())
                                .filter(|&si| slot.ep.agent_active(game, si))
                                .collect();
                            if perspectives.is_empty() {
                                return TaskOut::Skip;
                            }
                            let ctx = crate::policy::SearchCtx {
                                game,
                                enc: encoder,
                                reward,
                                rng: &mut slot.ep.rng,
                                perms,
                                collect_interior,
                            };
                            (
                                policy.begin_search(ctx, &slot.ep.state, &perspectives),
                                perspectives,
                            )
                        }
                        Work::Resume {
                            mut search,
                            perspectives,
                            rows,
                            stride,
                        } => {
                            if !rows.is_empty() {
                                let view = crate::policy::RowsView::from_slice(&rows, stride);
                                let ctx = crate::policy::SearchCtx {
                                    game,
                                    enc: encoder,
                                    reward,
                                    rng: &mut slot.ep.rng,
                                    perms,
                                    collect_interior,
                                };
                                policy.absorb(ctx, &mut search, view);
                            }
                            (search, perspectives)
                        }
                    };
                    let ctx = crate::policy::SearchCtx {
                        game,
                        enc: encoder,
                        reward,
                        rng: &mut slot.ep.rng,
                        perms,
                        collect_interior,
                    };
                    let mut sink = crate::policy::RequestSink::default();
                    let status = policy.round(ctx, &mut search, &mut sink);
                    assert!(
                        (!sink.is_empty()) == (status == crate::policy::RoundStatus::Pending),
                        "round contract: Pending emits at least one request, Done emits none"
                    );
                    if !sink.is_empty() {
                        let n = sink.len();
                        let (players, obs) = sink.into_parts();
                        return TaskOut::Emitted {
                            search,
                            perspectives,
                            players,
                            obs,
                            n,
                        };
                    }
                    // The search is done: run the whole completion here — finish, select,
                    // advance, record assembly, and (terminal episodes) the flush.
                    let ctx = crate::policy::SearchCtx {
                        game,
                        enc: encoder,
                        reward,
                        rng: &mut slot.ep.rng,
                        perms,
                        collect_interior,
                    };
                    let results = policy.finish(ctx, search);
                    assert_eq!(
                        results.len(),
                        perspectives.len(),
                        "finish must return one evaluation per perspective"
                    );
                    let mut records: Vec<L::Record> = Vec::new();
                    let mut tstats = CollectStats::default();
                    let mut start = StartAccess::Shared(&start_parts);
                    let finished = process_game_tick(
                        game,
                        encoder,
                        reward,
                        policy,
                        learner,
                        learn_mask,
                        sequential,
                        results,
                        &perspectives,
                        slot,
                        &mut start,
                        &mut records,
                        &mut tstats,
                    );
                    let pending_tail = match finished {
                        Some(true) => {
                            flush_finished_game(
                                &HashMap::new(),
                                game,
                                policy,
                                learner,
                                encoder,
                                slot,
                                &mut start,
                                &mut records,
                                &mut tstats,
                            );
                            false
                        }
                        Some(false) => true,
                        None => false,
                    };
                    TaskOut::Completed {
                        records,
                        stats: tstats,
                        pending_tail,
                    }
                }));
                let out = match run {
                    Ok(out) => out,
                    Err(payload) => TaskOut::Panicked(payload),
                };
                let _ = tx.send(Msg { gi, out });
            };
            let task = &task;

            let n_queues = match mode {
                InferMode::Shared => 1,
                InferMode::PerPlayer => self.game.num_agents(),
            };
            let mut phases: Vec<SlotPhase<P::Search<G::State>>> =
                (0..n_games).map(|_| SlotPhase::Idle).collect();
            let mut queues: Vec<RequestQueue> =
                (0..n_queues).map(|_| RequestQueue::default()).collect();
            let mut in_flight = 0usize;
            let fragment_backlog = |slots: &[std::sync::Mutex<SlotCtx<'_, G, P>>]| {
                slots
                    .iter()
                    .map(|m| {
                        let slot = m.lock().expect("slot lock");
                        slot.traj
                            .iter()
                            .enumerate()
                            .filter(|&(si, _)| learn_mask[si])
                            .map(|(_, steps)| steps.len())
                            .sum::<usize>()
                    })
                    .sum::<usize>()
            };
            let mut cutting = if fragments {
                out.len() + fragment_backlog(&slots)
            } else {
                out.len()
            } >= n_records;
            let admitted = !cutting;

            self.thread_pool.in_place_scope(|s| {
                let spawn = |gi: usize,
                             work: Work<P::Search<G::State>>,
                             phases: &mut Vec<SlotPhase<P::Search<G::State>>>,
                             in_flight: &mut usize| {
                    phases[gi] = SlotPhase::Running;
                    *in_flight += 1;
                    let txc = tx.clone();
                    s.spawn(move |_| task(gi, work, txc));
                };
                // Fire `take` rows from one queue and hand freed slots back to the pool.
                macro_rules! fire_settled {
                    ($qi:expr, $take:expr) => {{
                        fire_batch(&mut queues[$qi], $take, &mut phases, &mut evaluator, |ph| {
                            match ph {
                                SlotPhase::Blocked {
                                    outstanding,
                                    rows,
                                    stride,
                                    total,
                                    ..
                                } => (outstanding, rows, stride, *total),
                                _ => unreachable!("row routed to a slot with no outstanding round"),
                            }
                        });
                        for gi in 0..n_games {
                            if matches!(phases[gi], SlotPhase::Blocked { outstanding: 0, .. }) {
                                let taken = std::mem::replace(&mut phases[gi], SlotPhase::Idle);
                                let SlotPhase::Blocked {
                                    search,
                                    perspectives,
                                    rows,
                                    stride,
                                    ..
                                } = taken
                                else {
                                    unreachable!()
                                };
                                spawn(
                                    gi,
                                    Work::Resume {
                                        search,
                                        perspectives,
                                        rows,
                                        stride,
                                    },
                                    &mut phases,
                                    &mut in_flight,
                                );
                            }
                        }
                    }};
                }
                if !cutting {
                    for k in 0..n_games {
                        let gi = (base_cursor + k) % n_games;
                        spawn(gi, Work::Begin, &mut phases, &mut in_flight);
                    }
                }
                loop {
                    // Each queue fires independently at the full batch_size.
                    while let Some(qi) =
                        (0..n_queues).find(|&qi| queues[qi].pending() >= batch_size)
                    {
                        fire_settled!(qi, batch_size);
                    }

                    if in_flight == 0 {
                        if queues.iter().any(|q| q.pending() > 0) {
                            // Drain: no round can progress without these rows.
                            #[allow(clippy::needless_range_loop)]
                            for qi in 0..n_queues {
                                let n = queues[qi].pending();
                                if n > 0 {
                                    fire_settled!(qi, n);
                                }
                            }
                            continue;
                        }
                        break;
                    }

                    let msg = rx.recv().expect("scheduler channel closed");
                    in_flight -= 1;
                    match msg.out {
                        TaskOut::Panicked(payload) => std::panic::resume_unwind(payload),
                        TaskOut::Skip => phases[msg.gi] = SlotPhase::Parked,
                        TaskOut::Emitted {
                            search,
                            perspectives,
                            players,
                            obs,
                            n,
                        } => {
                            match mode {
                                InferMode::Shared => {
                                    queues[0].push(players.clone(), obs, msg.gi, n)
                                }
                                InferMode::PerPlayer => {
                                    let dim = obs.len() / n;
                                    for (pos, &p) in players.iter().enumerate() {
                                        queues[p].push_row(
                                            p,
                                            &obs[pos * dim..(pos + 1) * dim],
                                            msg.gi,
                                            pos,
                                        );
                                    }
                                }
                            }
                            phases[msg.gi] = SlotPhase::Blocked {
                                search,
                                perspectives,
                                outstanding: n,
                                total: n,
                                rows: Vec::new(),
                                stride: 0,
                            };
                        }
                        TaskOut::Completed {
                            records,
                            stats: tstats,
                            pending_tail,
                        } => {
                            let gi = msg.gi;
                            phases[gi] = SlotPhase::Idle;
                            out.extend(records);
                            stats = fold_stats(std::mem::take(&mut stats), tstats);
                            if pending_tail {
                                let mut guard = slots[gi].lock().expect("slot lock");
                                let slot: &mut SlotCtx<'_, G, P> = &mut guard;
                                let tails = tail_values_game(
                                    game,
                                    policy,
                                    learner,
                                    encoder,
                                    sequential,
                                    slot,
                                    &mut evaluator,
                                );
                                let mut start = StartAccess::Shared(&start_parts);
                                flush_finished_game(
                                    &tails, game, policy, learner, encoder, slot, &mut start,
                                    &mut out, &mut stats,
                                );
                            }
                            let collected = if fragments {
                                out.len() + fragment_backlog(&slots)
                            } else {
                                out.len()
                            };
                            if collected >= n_records {
                                cutting = true;
                            }
                            if !cutting {
                                spawn(gi, Work::Begin, &mut phases, &mut in_flight);
                            }
                        }
                    }
                }
            });
            // A no-op collect (floor already met) must not mutate engine state.
            if admitted {
                self.sweep_cursor = (base_cursor + 1) % n_games.max(1);
            }
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

    /// Grouped collect: each group runs the classic collect loop on its own worker
    /// thread, with inference forwarded to the [`ServiceHost`]'s resident thread —
    /// every callback invocation across every collect using the same host arrives on
    /// that one thread. Run-to-run nondeterministic when shared state is live;
    /// reproduce anomalies with `n_groups=1`.
    pub fn collect_grouped_hosted(
        &mut self,
        n_records: usize,
        mode: InferMode,
        host: &ServiceHost,
    ) -> (Vec<L::Record>, CollectStats)
    where
        P: Sync,
        L: Sync,
        P::PolicyState: Send,
        P::Evaluation: Send,
        L::Record: Send,
    {
        use crate::rollout::infer_service::{InferRequest, ServiceState};

        assert_eq!(
            self.n_groups, 2,
            "collect_grouped_hosted requires n_groups=2"
        );
        assert!(
            matches!(mode, InferMode::Shared),
            "collect_grouped_hosted supports a single shared infer callback"
        );
        if n_records == 0 {
            return (Vec::new(), CollectStats::default());
        }
        let n_games = self.episodes.len();
        let half = n_games / 2;
        // Proportional per-group floors: finish order never changes what is collected.
        let floor = |size: usize| (n_records * size).div_ceil(n_games);
        let floors = [floor(half), floor(n_games - half)];

        if let Some(caches) = self.sharded_caches.as_ref() {
            for cache in caches {
                cache.begin_collect();
            }
        }
        let collect_interior = self.learner.needs_interior();
        let pad_rows_to = self.pad_rows_to;
        let Engine {
            game,
            encoder,
            reward,
            policy,
            learner,
            episodes,
            start_dist,
            learn_mask,
            episode_returns,
            sequential,
            buffer_rng,
            seeded,
            policy_states,
            ticks,
            traj,
            group_rngs,
            sharded_caches,
            perms,
            ..
        } = self;
        let game: &G = game;
        let encoder: &dyn StateEncoder<State = G::State> = &**encoder;
        let reward: &dyn Reward<Event = G::Event> = &**reward;
        let policy: &P = &*policy;
        let learner: &L = &*learner;
        let learn_mask: &[bool] = learn_mask;
        let perms: &crate::encoder::PermTable = perms;
        let sequential = *sequential;
        let slots = sharded_caches.as_deref();
        let start = std::sync::Mutex::new(StartParts {
            dist: &mut **start_dist,
            rng: buffer_rng,
        });

        let (ep_a, ep_b) = episodes.split_at_mut(half);
        let (tr_a, tr_b) = traj.split_at_mut(half);
        let (tk_a, tk_b) = ticks.split_at_mut(half);
        let (ps_a, ps_b) = policy_states.split_at_mut(half);
        let (er_a, er_b) = episode_returns.split_at_mut(half);
        let (sd_a, sd_b) = seeded.split_at_mut(half);
        let (rng_a, rng_b) = group_rngs.split_at_mut(1);

        let service_state = std::sync::Arc::new(ServiceState::new());
        let (req_tx, req_rx) = std::sync::mpsc::sync_channel::<InferRequest>(2);
        let tx_a = req_tx.clone();
        let tx_b = req_tx;
        host.begin(req_rx, service_state.clone());

        let group_results: [Option<(Vec<L::Record>, CollectStats)>; 2];
        let first_panic: std::sync::Mutex<Option<Box<dyn std::any::Any + Send>>> =
            std::sync::Mutex::new(None);
        {
            let svc: &ServiceState = &service_state;
            let start = &start;
            let panic_slot = &first_panic;
            let (res_a, res_b) = std::thread::scope(|scope| {
                let handle_a = scope.spawn(move || {
                    let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        run_group_worker(
                            game,
                            encoder,
                            reward,
                            policy,
                            learner,
                            learn_mask,
                            perms,
                            sequential,
                            collect_interior,
                            mode,
                            floors[0],
                            ep_a,
                            tr_a,
                            tk_a,
                            ps_a,
                            er_a,
                            sd_a,
                            &mut rng_a[0],
                            slots,
                            pad_rows_to,
                            tx_a,
                            svc,
                            start,
                        )
                    }));
                    match res {
                        Ok(v) => Some(v),
                        Err(payload) => {
                            // record before cancelling: collateral panics (empty rows
                            // after cancellation) must not mask the root cause
                            panic_slot
                                .lock()
                                .expect("panic slot poisoned")
                                .get_or_insert(payload);
                            svc.cancel.store(true, std::sync::atomic::Ordering::Relaxed);
                            None
                        }
                    }
                });
                let handle_b = scope.spawn(move || {
                    let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        run_group_worker(
                            game,
                            encoder,
                            reward,
                            policy,
                            learner,
                            learn_mask,
                            perms,
                            sequential,
                            collect_interior,
                            mode,
                            floors[1],
                            ep_b,
                            tr_b,
                            tk_b,
                            ps_b,
                            er_b,
                            sd_b,
                            &mut rng_b[0],
                            slots,
                            pad_rows_to,
                            tx_b,
                            svc,
                            start,
                        )
                    }));
                    match res {
                        Ok(v) => Some(v),
                        Err(payload) => {
                            // record before cancelling: collateral panics (empty rows
                            // after cancellation) must not mask the root cause
                            panic_slot
                                .lock()
                                .expect("panic slot poisoned")
                                .get_or_insert(payload);
                            svc.cancel.store(true, std::sync::atomic::Ordering::Relaxed);
                            None
                        }
                    }
                });
                (
                    handle_a.join().unwrap_or(None),
                    handle_b.join().unwrap_or(None),
                )
            });
            group_results = [res_a, res_b];
        }
        // quiesce before reading the service's stats/error writes below
        host.wait_done();

        if let Some(err) = service_state
            .error
            .lock()
            .expect("service error poisoned")
            .take()
        {
            panic!("grouped collect infer callback failed: {err}");
        }
        if let Some(payload) = first_panic.into_inner().expect("panic slot poisoned") {
            std::panic::resume_unwind(payload);
        }

        let [a, b] = group_results;
        let (out_a, stats_a) = a.expect("worker result present");
        let (out_b, stats_b) = b.expect("worker result present");
        let mut out = out_a;
        out.extend(out_b);
        let mut stats = fold_stats(stats_a, stats_b);
        {
            // rows fold from the worker evaluators (real rows only, excluding padding);
            // wall-clock seconds and call counts are the service's to report
            let svc_stats = service_state.stats.lock().expect("service stats poisoned");
            stats.infer_seconds = svc_stats.seconds;
            stats.infer_calls = svc_stats.calls;
        }
        if let Some(caches) = self.sharded_caches.as_ref() {
            stats.cache_lookups = caches.iter().map(|c| c.lookups()).sum();
            stats.cache_hits = caches.iter().map(|c| c.hits()).sum();
        }
        (out, stats)
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
        // Result-bearing: restores must resume the rotation, not restart it.
        put_u64(&mut out, self.sweep_cursor as u64);
        put_u32(&mut out, self.group_rngs.len() as u32);
        for rng in &self.group_rngs {
            put_u64(&mut out, rng.state());
        }
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
        let mut group_rng_states: Vec<u64> = Vec::new();
        if version >= 3 {
            let n = r.u32()? as usize;
            if n != self.group_rngs.len() {
                return Err(format!(
                    "snapshot has {n} group rng streams; the engine has {}",
                    self.group_rngs.len()
                ));
            }
            for _ in 0..n {
                group_rng_states.push(r.u64()?);
            }
        }
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
        // Version-2 snapshots predate group streams; construction defaults stay in place.
        for (rng, state) in self.group_rngs.iter_mut().zip(&group_rng_states) {
            *rng = SplitMix64::from_state(*state);
        }
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
        if let Some(caches) = self.sharded_caches.as_ref() {
            for cache in caches {
                cache.force_clear();
            }
        }
        Ok(())
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

type DecisionParts = (Vec<(usize, Vec<usize>)>, Vec<(usize, usize)>);

/// Gather every active game's requests for one round.
fn gather_decisions<G: Game>(
    game: &G,
    episodes: &[Episode<G>],
    num_agents: usize,
) -> DecisionParts {
    let mut groups: Vec<(usize, Vec<usize>)> = Vec::new();
    let mut meta: Vec<(usize, usize)> = Vec::new();
    for (gi, ep) in episodes.iter().enumerate() {
        let perspectives: Vec<usize> = (0..num_agents)
            .filter(|&si| ep.agent_active(game, si))
            .collect();
        if !perspectives.is_empty() {
            for &si in &perspectives {
                meta.push((gi, si));
            }
            groups.push((gi, perspectives));
        }
    }
    (groups, meta)
}

/// Drive every gathered decision's stepped search to completion, one pooled infer call per
/// barrier round (the Stage A lockstep driver over engine state — per-game ctx borrows).
#[allow(clippy::too_many_arguments)]
fn drive_decisions<G, P, F>(
    policy: &P,
    game: &G,
    enc: &dyn StateEncoder<State = G::State>,
    reward: &dyn Reward<Event = G::Event>,
    perms: &crate::encoder::PermTable,
    collect_interior: bool,
    groups: &[(usize, Vec<usize>)],
    episodes: &mut [Episode<G>],
    eval: &mut Evaluator<'_, F>,
) -> Vec<(P::Evaluation, Vec<crate::learner::InteriorTarget>)>
where
    G: Game + Sync,
    G::State: Send,
    P: Policy,
    F: FnMut(usize, Vec<f32>, usize) -> Vec<f64>,
{
    use crate::policy::{RequestSink, RoundStatus, RowsView, SearchCtx};
    let mut searches: Vec<_> = groups
        .iter()
        .map(|(gi, perspectives)| {
            let ep = &mut episodes[*gi];
            let ctx = SearchCtx {
                game,
                enc,
                reward,
                rng: &mut ep.rng,
                perms,
                collect_interior,
            };
            policy.begin_search(ctx, &ep.state, perspectives)
        })
        .collect();
    let mut done = vec![false; searches.len()];
    loop {
        let mut sink = RequestSink::default();
        let mut spans: Vec<(usize, usize, usize)> = Vec::new();
        for (i, search) in searches.iter_mut().enumerate() {
            if done[i] {
                continue;
            }
            let ep = &mut episodes[groups[i].0];
            let start = sink.len();
            let ctx = SearchCtx {
                game,
                enc,
                reward,
                rng: &mut ep.rng,
                perms,
                collect_interior,
            };
            let status = policy.round(ctx, search, &mut sink);
            let count = sink.len() - start;
            assert!(
                (count > 0) == (status == RoundStatus::Pending),
                "round contract: Pending emits at least one request, Done emits none"
            );
            if count > 0 {
                spans.push((i, start, count));
            } else if status == RoundStatus::Done {
                done[i] = true;
            }
        }
        if spans.is_empty() {
            break;
        }
        let n = sink.len();
        let rows = eval.forward(&sink.players, sink.obs, n);
        let stride = rows.len() / n;
        for &(i, start, count) in &spans {
            let view = RowsView {
                data: &rows[start * stride..(start + count) * stride],
                stride,
            };
            let ctx = SearchCtx {
                game,
                enc,
                reward,
                rng: &mut episodes[groups[i].0].rng,
                perms,
                collect_interior,
            };
            policy.absorb(ctx, &mut searches[i], view);
        }
    }
    let mut out = Vec::new();
    for (search, (gi, perspectives)) in searches.into_iter().zip(groups) {
        let ep = &mut episodes[*gi];
        let ctx = SearchCtx {
            game,
            enc,
            reward,
            rng: &mut ep.rng,
            perms,
            collect_interior,
        };
        let results = policy.finish(ctx, search);
        assert_eq!(
            results.len(),
            perspectives.len(),
            "finish must return one evaluation per perspective"
        );
        out.extend(results);
    }
    out
}

/// A game slot's scheduler state.
enum SlotPhase<SE> {
    Idle,
    Running,
    Blocked {
        search: SE,
        perspectives: Vec<usize>,
        outstanding: usize,
        total: usize,
        rows: Vec<f64>,
        stride: usize,
    },
    Parked,
}

/// One game's mutable collection state, lockable per slot so a worker task can run
/// the whole completion while other slots proceed.
struct SlotCtx<'a, G: Game, P: Policy> {
    ep: &'a mut Episode<G>,
    traj: &'a mut Vec<Vec<Step<P::Evaluation>>>,
    tick: &'a mut usize,
    policy_state: &'a mut P::PolicyState,
    returns: &'a mut Vec<f64>,
    seeded: &'a mut bool,
}

/// Work handed to a slot task: start a fresh search, or absorb routed rows and round.
enum Work<SE> {
    Begin,
    Resume {
        search: SE,
        perspectives: Vec<usize>,
        rows: Vec<f64>,
        stride: usize,
    },
}

/// A slot task's result. `Completed` covers finish/select/advance and (for terminal
/// episodes) the flush, all run on the worker; a truncated episode leaves
/// `pending_tail` for the scheduler, which owns the evaluator.
enum TaskOut<SE, R> {
    Skip,
    Emitted {
        search: SE,
        perspectives: Vec<usize>,
        players: Vec<usize>,
        obs: Vec<f32>,
        n: usize,
    },
    Completed {
        records: Vec<R>,
        stats: CollectStats,
        pending_tail: bool,
    },
    Panicked(Box<dyn std::any::Any + Send>),
}

struct Msg<SE, R> {
    gi: usize,
    out: TaskOut<SE, R>,
}

type MsgSender<SE, R> = std::sync::mpsc::Sender<Msg<SE, R>>;

/// Forward the first `take` queued rows in one evaluator call and route the results to the
/// destination slots' row buffers, decrementing their outstanding counts.
/// FIFO request queue with an amortized-O(1) head: fires consume from `head`, appends
/// push to the back, and the storage resets once fully drained — no per-fire shifting.
#[derive(Default)]
struct RequestQueue {
    players: Vec<usize>,
    obs: Vec<f32>,
    dest: Vec<usize>,
    pos: Vec<usize>,
    head: usize,
    dim: usize,
}

impl RequestQueue {
    fn pending(&self) -> usize {
        self.players.len() - self.head
    }

    /// Queue a whole round's rows (shared mode): positions run 0..n.
    fn push(&mut self, players: Vec<usize>, obs: Vec<f32>, gi: usize, n: usize) {
        debug_assert_eq!(players.len(), n);
        self.dim = obs.len().checked_div(n).unwrap_or(self.dim);
        self.players.extend(players);
        self.obs.extend(obs);
        self.dest.extend(std::iter::repeat_n(gi, n));
        self.pos.extend(0..n);
    }

    /// Queue one row at a known position within its round (per-player split).
    fn push_row(&mut self, player: usize, obs: &[f32], gi: usize, pos: usize) {
        self.dim = obs.len();
        self.players.push(player);
        self.obs.extend_from_slice(obs);
        self.dest.push(gi);
        self.pos.push(pos);
    }

    /// Consume `take` rows from the head; compact once the consumed prefix outweighs
    /// the pending tail, so a long collection never retains fired observations.
    fn advance(&mut self, take: usize) {
        self.head += take;
        if self.head == self.players.len() {
            self.players.clear();
            self.obs.clear();
            self.dest.clear();
            self.pos.clear();
            self.head = 0;
        } else if self.head * 2 >= self.players.len() {
            let h = self.head;
            self.players.drain(..h);
            self.dest.drain(..h);
            self.pos.drain(..h);
            self.obs.drain(..h * self.dim);
            self.head = 0;
        }
    }
}

fn fire_batch<SE, F>(
    queue: &mut RequestQueue,
    take: usize,
    phases: &mut [SE],
    evaluator: &mut Evaluator<'_, F>,
    mut route: impl FnMut(&mut SE) -> (&mut usize, &mut Vec<f64>, &mut usize, usize),
) where
    F: FnMut(usize, Vec<f32>, usize) -> Vec<f64>,
{
    let (start, dim) = (queue.head, queue.dim);
    let players = &queue.players[start..start + take];
    let obs = queue.obs[start * dim..(start + take) * dim].to_vec();
    let rows = evaluator.forward(players, obs, take);
    let stride = rows.len() / take;
    for (i, (&gi, &pos)) in queue.dest[start..start + take]
        .iter()
        .zip(&queue.pos[start..start + take])
        .enumerate()
    {
        let (outstanding, buf, st, total) = route(&mut phases[gi]);
        if buf.is_empty() {
            buf.resize(total * stride, 0.0);
            *st = stride;
        } else {
            assert_eq!(
                *st, stride,
                "infer returned a different row width for one round's requests"
            );
        }
        buf[pos * stride..(pos + 1) * stride].copy_from_slice(&rows[i * stride..(i + 1) * stride]);
        *outstanding -= 1;
    }
    queue.advance(take);
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

fn fold_stats(mut a: CollectStats, b: CollectStats) -> CollectStats {
    a.decisions += b.decisions;
    a.max_depth = a.max_depth.max(b.max_depth);
    a.sum_leaves += b.sum_leaves;
    a.sum_rounds += b.sum_rounds;
    a.sum_expansions += b.sum_expansions;
    a.sum_sigma += b.sum_sigma;
    a.sum_disagreement += b.sum_disagreement;
    a.sum_terminal_sims += b.sum_terminal_sims;
    a.sum_depthcap_sims += b.sum_depthcap_sims;
    a.sum_shared_rows += b.sum_shared_rows;
    a.sum_fresh_rows += b.sum_fresh_rows;
    a.sum_hit_rows += b.sum_hit_rows;
    a.sum_extra_eval_rows += b.sum_extra_eval_rows;
    a.infer_rows += b.infer_rows;
    a.padded_rows += b.padded_rows;
    a.episodes.extend(b.episodes);
    a
}

/// One group's collect loop over its own state slice (local indices). Stops at its
/// floor, when its games are exhausted, or on cancellation.
#[allow(clippy::too_many_arguments)]
fn run_group_worker<G, P, L>(
    game: &G,
    encoder: &dyn StateEncoder<State = G::State>,
    reward: &dyn Reward<Event = G::Event>,
    policy: &P,
    learner: &L,
    learn_mask: &[bool],
    perms: &crate::encoder::PermTable,
    sequential: bool,
    collect_interior: bool,
    mode: InferMode,
    floor: usize,
    episodes: &mut [Episode<G>],
    traj: &mut [Vec<Vec<Step<P::Evaluation>>>],
    ticks: &mut [usize],
    policy_states: &mut [P::PolicyState],
    episode_returns: &mut [Vec<f64>],
    seeded: &mut [bool],
    _rng: &mut SplitMix64,
    slots: Option<&[crate::rollout::infer_cache::ShardedInferCache]>,
    pad_rows_to: Option<usize>,
    req_tx: std::sync::mpsc::SyncSender<crate::rollout::infer_service::InferRequest>,
    svc: &crate::rollout::infer_service::ServiceState,
    start: &std::sync::Mutex<StartParts<'_, G::State>>,
) -> (Vec<L::Record>, CollectStats)
where
    G: Game + Sync,
    G::State: Send,
    P: Policy,
    L: Learner<P::Evaluation>,
{
    use std::sync::atomic::Ordering;
    let mut start = StartAccess::Shared(start);
    let cancel = &svc.cancel;
    // run_service answers every accepted request exactly once, so one reply channel
    // serves the worker's lifetime.
    let (reply_tx, reply_rx) = std::sync::mpsc::sync_channel(1);
    let mut service_infer = |player: usize, obs: Vec<f32>, n: usize| -> Vec<f64> {
        if cancel.load(Ordering::Relaxed) {
            return Vec::new();
        }
        let request = crate::rollout::infer_service::InferRequest {
            player,
            obs,
            n,
            reply: reply_tx.clone(),
        };
        if req_tx.send(request).is_err() {
            return Vec::new();
        }
        match reply_rx.recv() {
            Ok(Ok(rows)) => rows,
            _ => Vec::new(),
        }
    };
    let mut evaluator = match slots {
        Some(slots) => Evaluator::with_shared_cache(&mut service_infer, mode, slots),
        None => Evaluator::new(&mut service_infer, mode, None),
    }
    .with_pad_rows_to(pad_rows_to);

    let mut out: Vec<L::Record> = Vec::new();
    let mut stats = CollectStats::default();
    let num_agents = game.num_agents();
    let fragments = learner.bootstraps_fragments();
    if fragments {
        discard_fragments(traj);
    }
    while !cancel.load(Ordering::Relaxed) {
        let collected = if fragments {
            fragment_potential(out.len(), traj, learn_mask)
        } else {
            out.len()
        };
        if collected >= floor {
            break;
        }
        let (groups, meta) = gather_decisions(game, episodes, num_agents);
        if groups.is_empty() {
            break;
        }
        let evals = drive_decisions(
            policy,
            game,
            encoder,
            reward,
            perms,
            collect_interior,
            &groups,
            episodes,
            &mut evaluator,
        );
        if cancel.load(Ordering::Relaxed) {
            break;
        }
        let finished = process_tick(
            game,
            encoder,
            reward,
            policy,
            learner,
            learn_mask,
            sequential,
            0..episodes.len(),
            evals,
            &meta,
            episodes,
            traj,
            ticks,
            policy_states,
            episode_returns,
            seeded,
            &mut start,
            &mut out,
            &mut stats,
        );
        let tails = tail_values_parts(
            &finished,
            game,
            policy,
            learner,
            encoder,
            sequential,
            episodes,
            traj,
            &mut evaluator,
        );
        // Finish the tick even when cancelled: skipping the flush would strand finished
        // episodes in a terminal or over-horizon state that the next collect would advance.
        flush_finished_parts(
            &finished,
            &tails,
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
            &mut start,
            &mut out,
            &mut stats,
        );
    }
    if fragments && !cancel.load(Ordering::Relaxed) {
        flush_fragments_parts(
            game,
            policy,
            learner,
            encoder,
            learn_mask,
            sequential,
            episodes,
            traj,
            &mut evaluator,
            &mut out,
        );
    }
    stats.infer_rows = evaluator.rows;
    stats.padded_rows = evaluator.padded_rows;
    (out, stats)
}

/// One game's decision tick: record emission, action selection, stepping. Returns
/// `Some(terminal)` when the episode ended (terminal or truncated) this tick.
#[allow(clippy::too_many_arguments)]
fn process_game_tick<G, P, L>(
    game: &G,
    encoder: &dyn StateEncoder<State = G::State>,
    reward: &dyn Reward<Event = G::Event>,
    policy: &P,
    learner: &L,
    learn_mask: &[bool],
    sequential: bool,
    evals: Vec<(P::Evaluation, Vec<crate::learner::InteriorTarget>)>,
    perspectives: &[usize],
    slot: &mut SlotCtx<'_, G, P>,
    start: &mut StartAccess<'_, '_, G::State>,
    out: &mut Vec<L::Record>,
    stats: &mut CollectStats,
) -> Option<bool>
where
    G: Game + Sync,
    G::State: Send,
    P: Policy,
    L: Learner<P::Evaluation>,
{
    let num_agents = game.num_agents();
    assert_eq!(
        evals.len(),
        perspectives.len(),
        "policy returned {} evaluations for {} requests — one per request, in order",
        evals.len(),
        perspectives.len()
    );
    let mut acted: Vec<Option<usize>> = vec![None; num_agents];
    for ((eval, targets), &si) in evals.into_iter().zip(perspectives.iter()) {
        stats.decisions += 1;
        policy.fold_telemetry(&eval, stats);
        if !learn_mask[si] {
            let rel = policy.select(&eval, &mut *slot.policy_state, &mut slot.ep.rng);
            acted[si] = Some(rel);
            continue;
        }
        out.extend(learner.eval_records(&eval, targets, encoder, si, &mut slot.ep.rng));
        let rel = policy.select(&eval, &mut *slot.policy_state, &mut slot.ep.rng);
        acted[si] = Some(rel);
        slot.traj[si].push(Step {
            obs: slot.ep.observe(encoder, si),
            evaluation: eval,
            action: rel,
            reward: 0.0,
            next_obs: Vec::new(),
            next_legal: Vec::new(),
            terminal: false,
        });
    }

    // MaxN consumers require value supervision for non-mover perspectives too.
    if policy.evaluates_all_perspectives(sequential, num_agents)
        && acted.iter().any(|s| s.is_some())
    {
        let action_count = game.action_count();
        for si in 0..num_agents {
            if acted[si].is_none() && learn_mask[si] {
                if let Some(evaluation) = learner.value_only_evaluation(action_count) {
                    slot.traj[si].push(Step {
                        obs: slot.ep.observe(encoder, si),
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

    let horizon = game.truncation_horizon();
    let joint: Vec<usize> = acted.iter().map(|a| a.unwrap_or(0)).collect();
    let (mut trace, terminal) = slot.ep.advance(game, &joint);
    *slot.tick += 1;
    let truncated = horizon.is_some_and(|h| *slot.tick >= h) && !terminal;
    if truncated {
        game.mark_truncation(&slot.ep.state, &mut trace);
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
    for (si, action) in acted.iter().enumerate() {
        let reward = tick_rewards[si];
        slot.returns[si] += reward;
        if action.is_some() {
            let (next_obs, next_legal) = if needs_next_obs {
                (
                    slot.ep.observe(encoder, si),
                    game.legal_actions(&slot.ep.state, si),
                )
            } else {
                (Vec::new(), Vec::new())
            };
            if let Some(step) = slot.traj[si].last_mut() {
                step.reward = reward;
                step.next_obs = next_obs;
                step.next_legal = next_legal;
                step.terminal = terminal;
            }
        } else if let Some(step) = slot.traj[si].last_mut() {
            // Sequential terminal events may reward an agent that did not act this tick.
            step.reward += reward;
            step.terminal |= terminal;
        }
    }
    if !terminal {
        start.observe(&slot.ep.state);
    }
    if terminal || truncated {
        Some(terminal)
    } else {
        None
    }
}

/// One decision tick for `games` (grouped path): per-game processing in game order.
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
    seeded: &mut [bool],
    start: &mut StartAccess<'_, '_, G::State>,
    out: &mut Vec<L::Record>,
    stats: &mut CollectStats,
) -> Vec<(usize, bool)>
where
    G: Game + Sync,
    G::State: Send,
    P: Policy,
    L: Learner<P::Evaluation>,
{
    assert_eq!(
        evals.len(),
        meta.len(),
        "policy returned {} evaluations for {} requests — one per request, in order",
        evals.len(),
        meta.len()
    );
    let mut finished: Vec<(usize, bool)> = Vec::new();
    let mut evals = evals.into_iter();
    let mut idx = 0;
    for gi in games {
        let mut perspectives = Vec::new();
        while idx < meta.len() && meta[idx].0 == gi {
            perspectives.push(meta[idx].1);
            idx += 1;
        }
        let game_evals: Vec<_> = evals.by_ref().take(perspectives.len()).collect();
        let mut slot = SlotCtx {
            ep: &mut episodes[gi],
            traj: &mut traj[gi],
            tick: &mut ticks[gi],
            policy_state: &mut policy_states[gi],
            returns: &mut episode_returns[gi],
            seeded: &mut seeded[gi],
        };
        if let Some(terminal) = process_game_tick(
            game,
            encoder,
            reward,
            policy,
            learner,
            learn_mask,
            sequential,
            game_evals,
            &perspectives,
            &mut slot,
            start,
            out,
            stats,
        ) {
            finished.push((gi, terminal));
        }
    }
    finished
}

/// One finished game's truncation-tail bootstraps, keyed by perspective.
#[allow(clippy::too_many_arguments)]
fn tail_values_game<G, P, L, F>(
    game: &G,
    policy: &P,
    learner: &L,
    encoder: &dyn StateEncoder<State = G::State>,
    sequential: bool,
    slot: &mut SlotCtx<'_, G, P>,
    evaluator: &mut Evaluator<'_, F>,
) -> HashMap<usize, Vec<f64>>
where
    G: Game + Sync,
    G::State: Send,
    P: Policy,
    L: Learner<P::Evaluation>,
    F: FnMut(usize, Vec<f32>, usize) -> Vec<f64>,
{
    let mut tails: HashMap<usize, Vec<f64>> = HashMap::new();
    if !learner.uses_episode_tail() {
        return tails;
    }
    let a = game.action_count();
    let num_agents = game.num_agents();
    let all_perspectives = (policy.evaluates_all_perspectives(sequential, num_agents)
        && learner.value_only_evaluation(a).is_some())
        || learner.tails_all_trajectories();
    let mut obs_flat: Vec<f32> = Vec::new();
    let mut meta: Vec<usize> = Vec::new();
    for (si, steps) in slot.traj.iter().enumerate().take(num_agents) {
        if (all_perspectives || slot.ep.agent_active(game, si)) && !steps.is_empty() {
            obs_flat.extend(slot.ep.observe(encoder, si));
            meta.push(si);
        }
    }
    if meta.is_empty() {
        return tails;
    }
    let q = evaluator.forward(&meta, obs_flat, meta.len());
    let stride = q.len() / meta.len();
    // Cancellation yields zero-width rows; empty tails degrade to the terminal path.
    if stride == 0 {
        return tails;
    }
    for (i, &si) in meta.iter().enumerate() {
        let row = &q[i * stride..(i + 1) * stride];
        let state = &slot.ep.state;
        // Sequential non-mover rows still bootstrap over the mover's available actions.
        let legal = match game.actor(state) {
            Actor::Agent(mover) => game.legal_actions(state, mover),
            Actor::Simultaneous => game.legal_actions(state, si),
            Actor::Chance => unreachable!("chance actors are not searched"),
        };
        tails.insert(si, learner.tail_from_row(row, a, &legal, encoder, si));
    }
    tails
}

/// Emit one finished game's episode records and respawn it.
#[allow(clippy::too_many_arguments)]
fn flush_finished_game<G, P, L>(
    tails: &HashMap<usize, Vec<f64>>,
    game: &G,
    policy: &P,
    learner: &L,
    encoder: &dyn StateEncoder<State = G::State>,
    slot: &mut SlotCtx<'_, G, P>,
    start: &mut StartAccess<'_, '_, G::State>,
    out: &mut Vec<L::Record>,
    stats: &mut CollectStats,
) where
    G: Game + Sync,
    G::State: Send,
    P: Policy,
    L: Learner<P::Evaluation>,
{
    let num_agents = game.num_agents();
    let mut ep_reward = vec![0.0; num_agents];
    for (si, ep_slot) in ep_reward.iter_mut().enumerate() {
        let steps = std::mem::take(&mut slot.traj[si]);
        *ep_slot = std::mem::take(&mut slot.returns[si]);
        if steps.is_empty() {
            continue;
        }
        let tail = tails.get(&si).cloned().unwrap_or_default();
        out.extend(learner.episode_records(&steps, &tail, encoder, si, &mut slot.ep.rng));
    }
    stats.episodes.push(EpisodeSummary {
        reward: ep_reward,
        length: *slot.tick,
        seeded: *slot.seeded,
    });
    match start.choose() {
        Start::Restore(state) => {
            Episode::assert_decision_state(game, &state);
            slot.ep.state = state;
            *slot.seeded = true;
        }
        Start::Fresh => {
            slot.ep.reset(game);
            *slot.seeded = false;
        }
    }
    *slot.tick = 0;
    *slot.policy_state = policy.begin_episode(&mut slot.ep.rng);
}

/// Emit records and respawn every game in `finished` (grouped path).
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
    start: &mut StartAccess<'_, '_, G::State>,
    out: &mut Vec<L::Record>,
    stats: &mut CollectStats,
) where
    G: Game + Sync,
    G::State: Send,
    P: Policy,
    L: Learner<P::Evaluation>,
{
    for &(gi, _) in finished {
        let game_tails: HashMap<usize, Vec<f64>> = tails
            .iter()
            .filter(|((tg, _), _)| *tg == gi)
            .map(|((_, si), v)| (*si, v.clone()))
            .collect();
        let mut slot = SlotCtx {
            ep: &mut episodes[gi],
            traj: &mut traj[gi],
            tick: &mut ticks[gi],
            policy_state: &mut policy_states[gi],
            returns: &mut episode_returns[gi],
            seeded: &mut seeded[gi],
        };
        flush_finished_game(
            &game_tails,
            game,
            policy,
            learner,
            encoder,
            &mut slot,
            start,
            out,
            stats,
        );
    }
}

/// Truncation-tail bootstraps for every game in `finished`, gathered into ONE forward:
/// a fragment cut over a large pool must not fire per-game callbacks.
#[allow(clippy::too_many_arguments)]
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
        for (si, steps) in traj[gi].iter().enumerate() {
            if (all_perspectives || episodes[gi].agent_active(game, si)) && !steps.is_empty() {
                obs_flat.extend(episodes[gi].observe(encoder, si));
                meta.push((gi, si));
            }
        }
    }
    if meta.is_empty() {
        return tails;
    }
    let players: Vec<usize> = meta.iter().map(|&(_, si)| si).collect();
    let q = evaluator.forward(&players, obs_flat, meta.len());
    let stride = q.len() / meta.len();
    // Cancellation yields zero-width rows; empty tails degrade to the terminal path.
    if stride == 0 {
        return tails;
    }
    for (i, &(gi, si)) in meta.iter().enumerate() {
        let row = &q[i * stride..(i + 1) * stride];
        let state = &episodes[gi].state;
        // Sequential non-mover rows still bootstrap over the mover's available actions.
        let legal = match game.actor(state) {
            Actor::Agent(mover) => game.legal_actions(state, mover),
            Actor::Simultaneous => game.legal_actions(state, si),
            Actor::Chance => unreachable!("chance actors are not searched"),
        };
        tails.insert((gi, si), learner.tail_from_row(row, a, &legal, encoder, si));
    }
    tails
}

#[cfg(test)]
mod tests {
    use super::RequestQueue;

    #[test]
    fn the_queue_compacts_its_consumed_prefix() {
        let mut q = RequestQueue::default();
        for i in 0..8 {
            q.push_row(0, &[i as f32, 0.0], i, 0);
        }
        q.advance(3);
        q.advance(3);
        // 6 of 8 consumed: compaction must have dropped the prefix, keeping the tail.
        assert!(q.players.len() <= 2, "consumed rows were retained");
        assert_eq!(q.pending(), 2);
        assert_eq!(
            q.obs[q.head * q.dim],
            6.0,
            "pending rows survive compaction"
        );
        q.advance(2);
        assert_eq!(q.pending(), 0);
        assert!(q.players.is_empty(), "a drained queue resets its storage");
    }

    use super::*;
    use crate::policies::tree::alphazero::{AlphaZero, AlphaZeroConfig};
    use crate::policies::tree::mcts::{NoiseScope, SequentialBackup};
    use crate::rollout::infer_cache::{InferCache, ShardedInferCache};
    use crate::{Actor, AlphaZeroLearner, ChanceMode, Space, StateCodec, Transition};
    use std::sync::atomic::AtomicU64;
    use std::sync::Arc;

    #[derive(Clone)]
    struct St {
        tick: usize,
    }
    struct Count;
    impl Game for Count {
        type State = St;
        type Event = ();
        fn num_agents(&self) -> usize {
            1
        }
        fn action_count(&self) -> usize {
            2
        }
        fn actor(&self, _s: &St) -> Actor {
            Actor::Agent(0)
        }
        fn legal_actions(&self, _s: &St, _agent: usize) -> Vec<usize> {
            vec![0, 1]
        }
        fn step(&self, s: &St, _a: &[usize]) -> Transition<St, ()> {
            Transition {
                next_state: St { tick: s.tick + 1 },
                events: vec![None],
                terminal: s.tick + 1 >= 3,
            }
        }
        fn initial_state(&self) -> St {
            St { tick: 0 }
        }
    }
    struct Enc;
    impl crate::ActionView for Enc {}
    impl StateEncoder for Enc {
        type State = St;
        fn encode(&self, s: &St, _agent: usize) -> Vec<f32> {
            vec![s.tick as f32, 1.0]
        }
        fn obs_shape(&self) -> (usize, usize, usize) {
            (1, 1, 2)
        }
        fn observation_space(&self) -> Space {
            Space::unit_box(vec![1, 1, 2])
        }
    }
    struct Zero;
    impl Reward for Zero {
        type Event = ();
        fn step_reward(&self, _e: &(), _agent: usize) -> f64 {
            0.0
        }
    }
    struct Codec;
    impl StateCodec for Codec {
        type State = St;
        fn encode(&self, s: &St) -> Vec<u8> {
            (s.tick as u64).to_le_bytes().to_vec()
        }
        fn decode(&self, b: &[u8]) -> Result<St, String> {
            let arr: [u8; 8] = b.try_into().map_err(|_| "bad state".to_string())?;
            Ok(St {
                tick: u64::from_le_bytes(arr) as usize,
            })
        }
        fn validate_decoded_state(&self, s: &St, _done: bool) -> Result<(), String> {
            if s.tick <= 3 {
                Ok(())
            } else {
                Err("tick out of range".into())
            }
        }
    }

    fn engine(n_groups: usize) -> Engine<Count, AlphaZero, AlphaZeroLearner> {
        Engine::new(
            Count,
            Box::new(Enc),
            Box::new(Zero),
            AlphaZero::new(AlphaZeroConfig {
                num_simulations: 6,
                c_puct: 1.5,
                gamma: 1.0,
                max_depth: 64,
                noise_epsilon: 0.25,
                noise_alpha: 0.5,
                temperature: 1.0,
                temperature_drop: 4,
                chance: ChanceMode::AlwaysResample,
                noise_scope: NoiseScope::Requester,
                sequential_backup: SequentialBackup::Auto,
            }),
            AlphaZeroLearner::new(1.0),
            EngineParams {
                n_games: 4,
                seed: 5,
                n_groups,
                ..Default::default()
            },
        )
    }

    #[test]
    #[should_panic(expected = "sharded caches require n_groups=2")]
    fn sharded_caches_rejected_on_ungrouped_engine() {
        let generation = Arc::new(AtomicU64::new(0));
        let caches = vec![
            ShardedInferCache::new(64, 4, generation.clone()),
            ShardedInferCache::new(64, 4, generation),
        ];
        let _ = engine(1).with_sharded_infer_caches(caches);
    }

    #[test]
    #[should_panic(expected = "exclusive caches require n_groups=1")]
    fn exclusive_caches_rejected_on_grouped_engine() {
        let generation = Arc::new(AtomicU64::new(0));
        let caches = vec![
            InferCache::new(64, generation.clone()),
            InferCache::new(64, generation),
        ];
        let _ = engine(2).with_infer_caches(caches);
    }

    #[test]
    fn restore_clears_sharded_caches() {
        let generation = Arc::new(AtomicU64::new(0));
        let caches = vec![
            ShardedInferCache::new(64, 4, generation.clone()),
            ShardedInferCache::new(64, 4, generation),
        ];
        let mut eng = engine(2).with_sharded_infer_caches(caches);
        let snap = eng.snapshot_bytes(&Codec).unwrap();
        let slot = &eng.sharded_caches.as_ref().unwrap()[0];
        slot.insert(InferCache::key(&[1.0, 2.0]), &[0.5], 0);
        assert_eq!(slot.len(), 1);
        eng.restore_bytes(&Codec, &snap).unwrap();
        assert_eq!(eng.sharded_caches.as_ref().unwrap()[0].len(), 0);
    }
}
