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
    /// Pad drain batches with zero rows to `batch_size`, fixing every callback
    /// invocation at exactly `batch_size` rows; see the inference contract.
    pub pad: bool,
    /// Scheduler firing threshold in rows (None = max(1, n_games / 2)).
    pub batch_size: Option<usize>,
    /// CPU fan-out width for search rounds (None = available cores).
    pub n_threads: Option<usize>,
}

impl Default for EngineParams {
    fn default() -> Self {
        EngineParams {
            n_games: 1,
            seed: 0,
            pad: false,
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
    start_dist: Box<dyn StartDistribution<G::State>>,
    // Slot 0 is shared; slots 1..=N isolate per-player networks.
    infer_caches: Option<Vec<InferCache>>,
    learn_mask: Vec<bool>,
    // Returns cannot be derived from steps: an agent may be rewarded before ever acting.
    episode_returns: Vec<Vec<f64>>,
    sequential: bool,
    pad: bool,
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
        assert!(
            params.batch_size.is_none_or(|b| b >= 1),
            "batch_size must be >= 1"
        );
        let mut episodes: Vec<Episode<G>> = (0..params.n_games)
            .map(|i| {
                Episode::new(
                    &game,
                    SplitMix64::keyed(params.seed, crate::rng::stream::GAME, i as u64),
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
        let buffer_rng = SplitMix64::keyed(params.seed, crate::rng::stream::BUFFER, 0);
        let seeded = vec![false; params.n_games];
        let perms = crate::encoder::PermTable::build(&*encoder, game.action_count(), num_agents);
        Engine {
            game,
            encoder,
            reward,
            policy,
            learner,
            episodes,
            start_dist: Box::new(AlwaysInitialState),
            infer_caches: None,
            learn_mask: vec![true; num_agents],
            episode_returns: vec![vec![0.0; num_agents]; params.n_games],
            sequential,
            pad: params.pad,
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
                .num_threads(
                    params
                        .n_threads
                        .or_else(|| std::thread::available_parallelism().ok().map(|n| n.get()))
                        .unwrap_or(1)
                        .max(1),
                )
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
            !self.pad || matches!(mode, InferMode::Shared),
            "pad supports a single shared infer callback"
        );
        let mut evaluator = Evaluator::new(&mut infer, mode, cache_slice)
            .with_pad_to(self.pad.then_some(self.batch_size));

        let fragments = self.learner.bootstraps_fragments();
        if fragments {
            discard_fragments(&mut self.traj);
        }
        // Free-running scheduler: worker tasks run search rounds and feed the queue
        // through a channel; the callback fires on this thread the moment `batch_size`
        // rows are queued, overlapping inference with the remaining rounds.
        {
            let n_games = self.episodes.len();
            let batch_size = self.batch_size;
            let base_cursor = self.sweep_cursor % n_games.max(1);
            let game = &self.game;
            let encoder = &*self.encoder;
            let reward = &*self.reward;
            let policy = &self.policy;
            let perms = &self.perms;
            let eps: Vec<std::sync::Mutex<&mut Episode<G>>> = self
                .episodes
                .iter_mut()
                .map(std::sync::Mutex::new)
                .collect();
            let (tx, rx) = std::sync::mpsc::channel::<Msg<P::Search<G::State>>>();

            let task =
                |gi: usize,
                 work: Work<P::Search<G::State>>,
                 tx: std::sync::mpsc::Sender<Msg<P::Search<G::State>>>| {
                    let run = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        let mut guard = eps[gi].lock().expect("slot episode lock");
                        let ep: &mut Episode<G> = &mut guard;
                        let (mut search, perspectives) = match work {
                            Work::Begin => {
                                let perspectives: Vec<usize> = (0..game.num_agents())
                                    .filter(|&si| ep.agent_active(game, si))
                                    .collect();
                                if perspectives.is_empty() {
                                    return None;
                                }
                                let ctx = crate::policy::SearchCtx {
                                    game,
                                    enc: encoder,
                                    reward,
                                    rng: &mut ep.rng,
                                    perms,
                                    collect_interior,
                                };
                                (
                                    policy.begin_search(ctx, &ep.state, &perspectives),
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
                                        rng: &mut ep.rng,
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
                            rng: &mut ep.rng,
                            perms,
                            collect_interior,
                        };
                        let mut sink = crate::policy::RequestSink::default();
                        let status = policy.round(ctx, &mut search, &mut sink);
                        assert!(
                            (!sink.is_empty()) == (status == crate::policy::RoundStatus::Pending),
                            "round contract: Pending emits at least one request, Done emits none"
                        );
                        let out = if sink.is_empty() {
                            TaskOut::Done
                        } else {
                            let n = sink.len();
                            let (players, obs) = sink.into_parts();
                            TaskOut::Emitted { players, obs, n }
                        };
                        Some((search, perspectives, out))
                    }));
                    let msg = match run {
                        Ok(None) => Msg {
                            gi,
                            search: None,
                            out: TaskOut::Skip,
                        },
                        Ok(Some((search, perspectives, out))) => Msg {
                            gi,
                            search: Some((search, perspectives)),
                            out,
                        },
                        Err(payload) => Msg {
                            gi,
                            search: None,
                            out: TaskOut::Panicked(payload),
                        },
                    };
                    let _ = tx.send(msg);
                };
            let task = &task;

            let mut phases: Vec<SlotPhase<P::Search<G::State>>> =
                (0..n_games).map(|_| SlotPhase::Idle).collect();
            let mut queue = RequestQueue::default();
            let mut in_flight = 0usize;
            let mut cutting = if fragments {
                fragment_potential(out.len(), &self.traj, &self.learn_mask)
            } else {
                out.len()
            } >= n_records;
            let admitted = !cutting;
            let mut frag_stage = false;

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
                // Fire `take` rows and settle the slots they freed: rounds respawn onto
                // the pool, completed tail jobs flush. If the callback panics, pending
                // episode-boundary tails resolve empty first — AwaitingTail never
                // survives a collect.
                macro_rules! fire_settled {
                    ($take:expr) => {{
                        let fired =
                            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                fire_batch(&mut queue, $take, &mut phases, &mut evaluator, |ph| {
                                    match ph {
                                        SlotPhase::Blocked {
                                            outstanding,
                                            rows,
                                            stride,
                                            ..
                                        }
                                        | SlotPhase::AwaitingTail {
                                            outstanding,
                                            rows,
                                            stride,
                                            ..
                                        } => (outstanding, rows, stride),
                                        _ => unreachable!(
                                            "row routed to a slot with no outstanding round"
                                        ),
                                    }
                                })
                            }));
                        if let Err(payload) = fired {
                            for gi in 0..n_games {
                                if !matches!(
                                    phases[gi],
                                    SlotPhase::AwaitingTail {
                                        fragment: false,
                                        ..
                                    }
                                ) {
                                    continue;
                                }
                                phases[gi] = SlotPhase::Idle;
                                let mut guard = eps[gi].lock().expect("slot episode lock");
                                let ep: &mut Episode<G> = &mut guard;
                                let tails = HashMap::new();
                                let mut start = StartParts {
                                    dist: &mut *self.start_dist,
                                    rng: &mut self.buffer_rng,
                                };
                                flush_finished_game(
                                    gi,
                                    &tails,
                                    game,
                                    policy,
                                    &self.learner,
                                    encoder,
                                    ep,
                                    &mut self.traj,
                                    &mut self.ticks,
                                    &mut self.policy_states,
                                    &mut self.episode_returns,
                                    &mut self.seeded,
                                    &mut start,
                                    &mut out,
                                    &mut stats,
                                );
                            }
                            std::panic::resume_unwind(payload);
                        }
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
                            } else if matches!(
                                phases[gi],
                                SlotPhase::AwaitingTail { outstanding: 0, .. }
                            ) {
                                let taken = std::mem::replace(&mut phases[gi], SlotPhase::Idle);
                                let SlotPhase::AwaitingTail {
                                    fragment,
                                    meta,
                                    rows,
                                    stride,
                                    ..
                                } = taken
                                else {
                                    unreachable!()
                                };
                                {
                                    let mut guard = eps[gi].lock().expect("slot episode lock");
                                    let ep: &mut Episode<G> = &mut guard;
                                    let tails = tails_from_rows(
                                        &meta,
                                        &rows,
                                        stride,
                                        game,
                                        &self.learner,
                                        encoder,
                                        ep,
                                    );
                                    if fragment {
                                        flush_fragment_slot(
                                            gi,
                                            &tails,
                                            game,
                                            &self.learner,
                                            encoder,
                                            &self.learn_mask,
                                            ep,
                                            &mut self.traj,
                                            &mut out,
                                        );
                                    } else {
                                        let mut start = StartParts {
                                            dist: &mut *self.start_dist,
                                            rng: &mut self.buffer_rng,
                                        };
                                        flush_finished_game(
                                            gi,
                                            &tails,
                                            game,
                                            policy,
                                            &self.learner,
                                            encoder,
                                            ep,
                                            &mut self.traj,
                                            &mut self.ticks,
                                            &mut self.policy_states,
                                            &mut self.episode_returns,
                                            &mut self.seeded,
                                            &mut start,
                                            &mut out,
                                            &mut stats,
                                        );
                                    }
                                }
                                let collected = if fragments {
                                    fragment_potential(out.len(), &self.traj, &self.learn_mask)
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
                    }};
                }
                if !cutting {
                    for k in 0..n_games {
                        let gi = (base_cursor + k) % n_games;
                        spawn(gi, Work::Begin, &mut phases, &mut in_flight);
                    }
                }
                loop {
                    let collected = if fragments {
                        fragment_potential(out.len(), &self.traj, &self.learn_mask)
                    } else {
                        out.len()
                    };
                    if collected >= n_records {
                        cutting = true;
                    }

                    // Fire every full batch, then settle the slots each batch freed.
                    while queue.pending() >= batch_size {
                        fire_settled!(batch_size);
                    }

                    if in_flight == 0 {
                        if queue.pending() > 0 {
                            // Drain: no round can progress without these rows.
                            let n = queue.pending();
                            fire_settled!(n);
                            continue;
                        }
                        // Cut step 5: bootstrap live fragments through the same queue.
                        if fragments && !frag_stage {
                            frag_stage = true;
                            let mut queued_any = false;
                            for gi in 0..n_games {
                                if !self.traj[gi].iter().any(|steps| !steps.is_empty()) {
                                    continue;
                                }
                                let mut guard = eps[gi].lock().expect("slot episode lock");
                                let ep: &mut Episode<G> = &mut guard;
                                let reqs = tail_requests(
                                    gi,
                                    game,
                                    policy,
                                    &self.learner,
                                    encoder,
                                    self.sequential,
                                    ep,
                                    &self.traj,
                                );
                                if reqs.is_empty() {
                                    flush_fragment_slot(
                                        gi,
                                        &HashMap::new(),
                                        game,
                                        &self.learner,
                                        encoder,
                                        &self.learn_mask,
                                        ep,
                                        &mut self.traj,
                                        &mut out,
                                    );
                                    continue;
                                }
                                drop(guard);
                                let mut players = Vec::with_capacity(reqs.len());
                                let mut obs_flat: Vec<f32> = Vec::new();
                                for (si, obs) in reqs {
                                    players.push(si);
                                    obs_flat.extend(obs);
                                }
                                let n = players.len();
                                let meta = players.clone();
                                queue.push(players, obs_flat, gi, n);
                                phases[gi] = SlotPhase::AwaitingTail {
                                    fragment: true,
                                    meta,
                                    outstanding: n,
                                    rows: Vec::new(),
                                    stride: 0,
                                };
                                queued_any = true;
                            }
                            if queued_any {
                                continue;
                            }
                        }
                        break;
                    }

                    let msg = rx.recv().expect("scheduler channel closed");
                    in_flight -= 1;
                    match msg.out {
                        TaskOut::Panicked(payload) => std::panic::resume_unwind(payload),
                        TaskOut::Skip => phases[msg.gi] = SlotPhase::Parked,
                        TaskOut::Emitted { players, obs, n } => {
                            queue.push(players, obs, msg.gi, n);
                            let (search, perspectives) = msg.search.expect("emitting slot");
                            phases[msg.gi] = SlotPhase::Blocked {
                                search,
                                perspectives,
                                outstanding: n,
                                rows: Vec::new(),
                                stride: 0,
                            };
                        }
                        TaskOut::Done => {
                            let gi = msg.gi;
                            let (search, perspectives) = msg.search.expect("finished slot");
                            phases[gi] = SlotPhase::Idle;
                            let mut guard = eps[gi].lock().expect("slot episode lock");
                            let ep: &mut Episode<G> = &mut guard;
                            let ctx = crate::policy::SearchCtx {
                                game,
                                enc: encoder,
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
                            let finished = {
                                let mut start = StartParts {
                                    dist: &mut *self.start_dist,
                                    rng: &mut self.buffer_rng,
                                };
                                process_game_tick(
                                    game,
                                    encoder,
                                    reward,
                                    policy,
                                    &self.learner,
                                    &self.learn_mask,
                                    self.sequential,
                                    gi,
                                    results,
                                    &perspectives,
                                    ep,
                                    &mut self.traj,
                                    &mut self.ticks,
                                    &mut self.policy_states,
                                    &mut self.episode_returns,
                                    &mut start,
                                    &mut out,
                                    &mut stats,
                                )
                            };
                            if let Some(terminal) = finished {
                                // Tail bootstraps are queue jobs: the queue is the only
                                // inference pathway.
                                let reqs = if terminal {
                                    Vec::new()
                                } else {
                                    tail_requests(
                                        gi,
                                        game,
                                        policy,
                                        &self.learner,
                                        encoder,
                                        self.sequential,
                                        ep,
                                        &self.traj,
                                    )
                                };
                                if reqs.is_empty() {
                                    let tails = HashMap::new();
                                    let mut start = StartParts {
                                        dist: &mut *self.start_dist,
                                        rng: &mut self.buffer_rng,
                                    };
                                    flush_finished_game(
                                        gi,
                                        &tails,
                                        game,
                                        policy,
                                        &self.learner,
                                        encoder,
                                        ep,
                                        &mut self.traj,
                                        &mut self.ticks,
                                        &mut self.policy_states,
                                        &mut self.episode_returns,
                                        &mut self.seeded,
                                        &mut start,
                                        &mut out,
                                        &mut stats,
                                    );
                                } else {
                                    let mut players = Vec::with_capacity(reqs.len());
                                    let mut obs_flat: Vec<f32> = Vec::new();
                                    for (si, obs) in reqs {
                                        players.push(si);
                                        obs_flat.extend(obs);
                                    }
                                    let n = players.len();
                                    let meta = players.clone();
                                    queue.push(players, obs_flat, gi, n);
                                    phases[gi] = SlotPhase::AwaitingTail {
                                        fragment: false,
                                        meta,
                                        outstanding: n,
                                        rows: Vec::new(),
                                        stride: 0,
                                    };
                                }
                            }
                            drop(guard);
                            // Re-check the floor before readmitting: this completion's
                            // records may have crossed it.
                            let collected = if fragments {
                                fragment_potential(out.len(), &self.traj, &self.learn_mask)
                            } else {
                                out.len()
                            };
                            if collected >= n_records {
                                cutting = true;
                            }
                            if !cutting && matches!(phases[gi], SlotPhase::Idle) {
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
        (stats.infer_seconds, stats.infer_calls, stats.infer_rows) =
            (evaluator.seconds, evaluator.calls, evaluator.rows);
        stats.padded_rows = evaluator.padded_rows;
        (stats.cache_lookups, stats.cache_hits) =
            (evaluator.cache_lookups(), evaluator.cache_hits());
        self.infer_caches = caches;
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
        put_u64(&mut out, self.buffer_rng.state());
        // Result-bearing: restores must resume the rotation, not restart it.
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

/// One `(player, obs)` tail-bootstrap request per learning perspective; empty when the
/// learner takes no tail.
#[allow(clippy::too_many_arguments)]
fn tail_requests<G, P, L>(
    gi: usize,
    game: &G,
    policy: &P,
    learner: &L,
    encoder: &dyn StateEncoder<State = G::State>,
    sequential: bool,
    ep: &Episode<G>,
    traj: &[Vec<Vec<Step<P::Evaluation>>>],
) -> Vec<(usize, Vec<f32>)>
where
    G: Game + Sync,
    G::State: Send,
    P: Policy,
    L: Learner<P::Evaluation>,
{
    if !learner.uses_episode_tail() {
        return Vec::new();
    }
    let a = game.action_count();
    let num_agents = game.num_agents();
    let all_perspectives = (policy.evaluates_all_perspectives(sequential, num_agents)
        && learner.value_only_evaluation(a).is_some())
        || learner.tails_all_trajectories();
    let mut reqs = Vec::new();
    for (si, steps) in traj[gi].iter().enumerate() {
        if (all_perspectives || ep.agent_active(game, si)) && !steps.is_empty() {
            reqs.push((si, ep.observe(encoder, si)));
        }
    }
    reqs
}

/// Routed tail rows back to per-perspective tail values; zero-width (cancelled) rows
/// yield an empty map.
#[allow(clippy::too_many_arguments)]
fn tails_from_rows<G, L, E>(
    meta: &[usize],
    rows: &[f64],
    stride: usize,
    game: &G,
    learner: &L,
    encoder: &dyn StateEncoder<State = G::State>,
    ep: &Episode<G>,
) -> HashMap<usize, Vec<f64>>
where
    G: Game + Sync,
    G::State: Send,
    L: Learner<E>,
{
    let mut tails = HashMap::new();
    if stride == 0 {
        return tails;
    }
    let a = game.action_count();
    let state = &ep.state;
    for (i, &si) in meta.iter().enumerate() {
        let row = &rows[i * stride..(i + 1) * stride];
        // Sequential non-mover rows still bootstrap over the mover's available actions;
        // using `si` here would turn a valid sparse-action tail into an empty one.
        let legal = match game.actor(state) {
            Actor::Agent(mover) => game.legal_actions(state, mover),
            Actor::Simultaneous => game.legal_actions(state, si),
            Actor::Chance => unreachable!("chance actors are not searched"),
        };
        tails.insert(si, learner.tail_from_row(row, a, &legal, encoder, si));
    }
    tails
}

/// Cut one game's live trajectory: bootstrap each learning perspective from its tail and
/// emit its records; episode state, ticks, and telemetry persist into the next window.
#[allow(clippy::too_many_arguments)]
fn flush_fragment_slot<G, L, E>(
    gi: usize,
    tails: &HashMap<usize, Vec<f64>>,
    game: &G,
    learner: &L,
    encoder: &dyn StateEncoder<State = G::State>,
    learn_mask: &[bool],
    ep: &mut Episode<G>,
    traj: &mut [Vec<Vec<Step<E>>>],
    out: &mut Vec<L::Record>,
) where
    G: Game + Sync,
    G::State: Send,
    L: Learner<E>,
{
    for si in 0..game.num_agents() {
        if !learn_mask[si] {
            traj[gi][si].clear();
            continue;
        }
        let steps = std::mem::take(&mut traj[gi][si]);
        if steps.is_empty() {
            continue;
        }
        let tail = tails.get(&si).cloned().unwrap_or_default();
        out.extend(learner.episode_records(&steps, &tail, encoder, si, &mut ep.rng));
    }
}
/// A game slot's scheduler state.
enum SlotPhase<SE> {
    Idle,
    Running,
    Blocked {
        search: SE,
        perspectives: Vec<usize>,
        outstanding: usize,
        rows: Vec<f64>,
        stride: usize,
    },
    /// A finished (or fragment-cut) episode whose tail bootstraps ride the queue.
    AwaitingTail {
        fragment: bool,
        meta: Vec<usize>,
        outstanding: usize,
        rows: Vec<f64>,
        stride: usize,
    },
    Parked,
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

/// A slot task's result.
enum TaskOut {
    Skip,
    Emitted {
        players: Vec<usize>,
        obs: Vec<f32>,
        n: usize,
    },
    Done,
    Panicked(Box<dyn std::any::Any + Send>),
}

struct Msg<SE> {
    gi: usize,
    search: Option<(SE, Vec<usize>)>,
    out: TaskOut,
}

/// Forward the first `take` queued rows in one evaluator call and route the results to the
/// destination slots' row buffers, decrementing their outstanding counts.
/// FIFO request queue with an amortized-O(1) head: fires consume from `head`, appends
/// push to the back, and the storage resets once fully drained — no per-fire shifting.
#[derive(Default)]
struct RequestQueue {
    players: Vec<usize>,
    obs: Vec<f32>,
    dest: Vec<usize>,
    head: usize,
    dim: usize,
}

impl RequestQueue {
    fn pending(&self) -> usize {
        self.players.len() - self.head
    }

    fn push(&mut self, players: Vec<usize>, obs: Vec<f32>, gi: usize, n: usize) {
        debug_assert_eq!(players.len(), n);
        self.dim = obs.len().checked_div(n).unwrap_or(self.dim);
        self.players.extend(players);
        self.obs.extend(obs);
        self.dest.extend(std::iter::repeat_n(gi, n));
    }
}

fn fire_batch<SE, F>(
    queue: &mut RequestQueue,
    take: usize,
    phases: &mut [SE],
    evaluator: &mut Evaluator<'_, F>,
    mut route: impl FnMut(&mut SE) -> (&mut usize, &mut Vec<f64>, &mut usize),
) where
    F: FnMut(usize, Vec<f32>, usize) -> Vec<f64>,
{
    let (start, dim) = (queue.head, queue.dim);
    let players = &queue.players[start..start + take];
    let obs = queue.obs[start * dim..(start + take) * dim].to_vec();
    let rows = evaluator.forward(players, obs, take);
    let stride = rows.len() / take;
    for (i, &gi) in queue.dest[start..start + take].iter().enumerate() {
        let (outstanding, buf, st) = route(&mut phases[gi]);
        *st = stride;
        buf.extend_from_slice(&rows[i * stride..(i + 1) * stride]);
        *outstanding -= 1;
    }
    queue.head += take;
    if queue.head == queue.players.len() {
        queue.players.clear();
        queue.obs.clear();
        queue.dest.clear();
        queue.head = 0;
    }
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
    gi: usize,
    evals: Vec<(P::Evaluation, Vec<crate::learner::InteriorTarget>)>,
    perspectives: &[usize],
    ep: &mut Episode<G>,
    traj: &mut [Vec<Vec<Step<P::Evaluation>>>],
    ticks: &mut [usize],
    policy_states: &mut [P::PolicyState],
    episode_returns: &mut [Vec<f64>],
    start: &mut StartParts<'_, G::State>,
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
            let rel = policy.select(&eval, &mut policy_states[gi], &mut ep.rng);
            acted[si] = Some(rel);
            continue;
        }
        out.extend(learner.eval_records(&eval, targets, encoder, si, &mut ep.rng));
        let rel = policy.select(&eval, &mut policy_states[gi], &mut ep.rng);
        acted[si] = Some(rel);
        traj[gi][si].push(Step {
            obs: ep.observe(encoder, si),
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
                    traj[gi][si].push(Step {
                        obs: ep.observe(encoder, si),
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
    let (mut trace, terminal) = ep.advance(game, &joint);
    ticks[gi] += 1;
    let truncated = horizon.is_some_and(|h| ticks[gi] >= h) && !terminal;
    if truncated {
        game.mark_truncation(&ep.state, &mut trace);
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
        episode_returns[gi][si] += reward;
        if action.is_some() {
            let (next_obs, next_legal) = if needs_next_obs {
                (ep.observe(encoder, si), game.legal_actions(&ep.state, si))
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
        start.observe(&ep.state);
    }
    if terminal || truncated {
        Some(terminal)
    } else {
        None
    }
}

/// Emit one finished game's episode records and respawn it.
#[allow(clippy::too_many_arguments)]
fn flush_finished_game<G, P, L>(
    gi: usize,
    tails: &HashMap<usize, Vec<f64>>,
    game: &G,
    policy: &P,
    learner: &L,
    encoder: &dyn StateEncoder<State = G::State>,
    ep: &mut Episode<G>,
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
    let mut ep_reward = vec![0.0; num_agents];
    for (si, ep_slot) in ep_reward.iter_mut().enumerate() {
        let steps = std::mem::take(&mut traj[gi][si]);
        *ep_slot = std::mem::take(&mut episode_returns[gi][si]);
        if steps.is_empty() {
            continue;
        }
        let tail = tails.get(&si).cloned().unwrap_or_default();
        out.extend(learner.episode_records(&steps, &tail, encoder, si, &mut ep.rng));
    }
    stats.episodes.push(EpisodeSummary {
        reward: ep_reward,
        length: ticks[gi],
        seeded: seeded[gi],
    });
    match start.choose() {
        Start::Restore(state) => {
            Episode::assert_decision_state(game, &state);
            ep.state = state;
            seeded[gi] = true;
        }
        Start::Fresh => {
            ep.reset(game);
            seeded[gi] = false;
        }
    }
    ticks[gi] = 0;
    policy_states[gi] = policy.begin_episode(&mut ep.rng);
}
