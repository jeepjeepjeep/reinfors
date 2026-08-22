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

/// Start-distribution access: exclusive on the classic path, mutex-shared across group
/// workers.
pub(crate) struct StartParts<'a, S> {
    pub dist: &'a mut dyn StartDistribution<S>,
    pub rng: &'a mut SplitMix64,
}

pub(crate) enum StartAccess<'a, 'b, S> {
    Exclusive(StartParts<'a, S>),
    #[allow(dead_code)]
    Shared(&'b std::sync::Mutex<StartParts<'a, S>>),
}

impl<S> StartAccess<'_, '_, S> {
    fn observe(&mut self, state: &S) {
        match self {
            StartAccess::Exclusive(parts) => parts.dist.observe(state, &mut *parts.rng),
            StartAccess::Shared(shared) => {
                let mut parts = shared.lock().expect("start access poisoned");
                let StartParts { dist, rng } = &mut *parts;
                dist.observe(state, &mut **rng);
            }
        }
    }

    fn choose(&mut self) -> Start<S> {
        match self {
            StartAccess::Exclusive(parts) => parts.dist.choose(&mut *parts.rng),
            StartAccess::Shared(shared) => {
                let mut parts = shared.lock().expect("start access poisoned");
                let StartParts { dist, rng } = &mut *parts;
                dist.choose(&mut **rng)
            }
        }
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
    /// Worker threads running search rounds (None = 1).
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
            let mut start = StartAccess::Exclusive(StartParts {
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
                    // The completion computes effects only: records, stats, and the
                    // buffered-step delta ride the message; every shared-state effect
                    // (reservoir draws, respawns, floor) is applied by the scheduler in
                    // message order — the determinism contract at one worker.
                    let before = buffered_learn_steps(slot, learn_mask);
                    let mut records: Vec<L::Record> = Vec::new();
                    let mut tstats = CollectStats::default();
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
                        &mut records,
                        &mut tstats,
                    );
                    if finished == Some(true) {
                        flush_records_game(
                            &HashMap::new(),
                            game,
                            learner,
                            encoder,
                            slot,
                            &mut records,
                            &mut tstats,
                        );
                    }
                    let steps_delta =
                        buffered_learn_steps(slot, learn_mask) as isize - before as isize;
                    TaskOut::Completed {
                        records,
                        stats: tstats,
                        steps_delta,
                        finished,
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
            // The floor counts buffered learning steps through a scheduler-owned
            // counter fed by message deltas: recounting live trajectories would race
            // in-flight completions.
            let mut backlog: isize = slots
                .iter()
                .map(|m| buffered_learn_steps(&m.lock().expect("slot lock"), learn_mask) as isize)
                .sum();
            let mut cutting = if fragments {
                out.len().saturating_add_signed(backlog)
            } else {
                out.len()
            } >= n_records;
            let admitted = !cutting;
            let mut frag_stage = false;

            // FIFO spawning on the persistent pool: at one worker, tasks execute in
            // spawn order — the determinism contract's substrate (per-collect thread
            // creation measurably taxed small collects).
            self.thread_pool.in_place_scope_fifo(|s| {
                let spawn = |gi: usize,
                             work_item: Work<P::Search<G::State>>,
                             phases: &mut Vec<SlotPhase<P::Search<G::State>>>,
                             in_flight: &mut usize| {
                    phases[gi] = SlotPhase::Running;
                    *in_flight += 1;
                    let txc = tx.clone();
                    s.spawn_fifo(move |_| task(gi, work_item, txc));
                };
                // Fire `take` rows from one queue and hand freed slots back to the pool.
                macro_rules! fire_settled {
                    ($qi:expr, $take:expr) => {{
                        let fired = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            fire_batch(&mut queues[$qi], $take, &mut phases, &mut evaluator, |ph| {
                                match ph {
                                    SlotPhase::Blocked {
                                        outstanding,
                                        rows,
                                        stride,
                                        total,
                                        ..
                                    }
                                    | SlotPhase::AwaitingTail {
                                        outstanding,
                                        rows,
                                        stride,
                                        total,
                                        ..
                                    } => (outstanding, rows, stride, *total),
                                    _ => unreachable!(
                                        "row routed to a slot with no outstanding round"
                                    ),
                                }
                            })
                        }));
                        let freed = match fired {
                            Ok(freed) => freed,
                            Err(payload) => {
                                // Drain in-flight completions before surfacing: an episode
                                // that just finished must respawn, never strand over-horizon.
                                while in_flight > 0 {
                                    let msg = rx.recv().expect("scheduler channel closed");
                                    in_flight -= 1;
                                    if let TaskOut::Completed {
                                        finished: Some(terminal),
                                        ..
                                    } = msg.out
                                    {
                                        let mut guard = slots[msg.gi].lock().expect("slot lock");
                                        let slot: &mut SlotCtx<'_, G, P> = &mut guard;
                                        if !terminal {
                                            flush_records_game(
                                                &HashMap::new(),
                                                game,
                                                learner,
                                                encoder,
                                                slot,
                                                &mut out,
                                                &mut stats,
                                            );
                                        }
                                        respawn_game(game, policy, slot, &mut start);
                                    }
                                }
                                // Parked tails resolve empty too: AwaitingTail never
                                // survives a collect.
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
                                    let mut guard = slots[gi].lock().expect("slot lock");
                                    let slot: &mut SlotCtx<'_, G, P> = &mut guard;
                                    flush_records_game(
                                        &HashMap::new(),
                                        game,
                                        learner,
                                        encoder,
                                        slot,
                                        &mut out,
                                        &mut stats,
                                    );
                                    respawn_game(game, policy, slot, &mut start);
                                }
                                std::panic::resume_unwind(payload);
                            }
                        };
                        for gi in freed {
                            match std::mem::replace(&mut phases[gi], SlotPhase::Idle) {
                                SlotPhase::Blocked {
                                    search,
                                    perspectives,
                                    rows,
                                    stride,
                                    ..
                                } => spawn(
                                    gi,
                                    Work::Resume {
                                        search,
                                        perspectives,
                                        rows,
                                        stride,
                                    },
                                    &mut phases,
                                    &mut in_flight,
                                ),
                                SlotPhase::AwaitingTail {
                                    fragment,
                                    meta,
                                    rows,
                                    stride,
                                    ..
                                } => {
                                    let mut guard = slots[gi].lock().expect("slot lock");
                                    let slot: &mut SlotCtx<'_, G, P> = &mut guard;
                                    let tails = tails_from_rows(
                                        &meta, &rows, stride, game, learner, encoder, slot,
                                    );
                                    backlog -= buffered_learn_steps(slot, learn_mask) as isize;
                                    if fragment {
                                        flush_fragment_slot(
                                            &tails, game, learner, encoder, learn_mask, slot,
                                            &mut out,
                                        );
                                    } else {
                                        flush_records_game(
                                            &tails, game, learner, encoder, slot, &mut out,
                                            &mut stats,
                                        );
                                        respawn_game(game, policy, slot, &mut start);
                                    }
                                    drop(guard);
                                    let collected = if fragments {
                                        out.len().saturating_add_signed(backlog)
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
                                _ => unreachable!("a freed slot must hold outstanding rows"),
                            }
                        }
                    }};
                }
                // Queue one game's tail-bootstrap rows and park it in AwaitingTail.
                macro_rules! enqueue_tails {
                    ($gi:expr, $reqs:expr, $fragment:expr) => {{
                        let reqs = $reqs;
                        let mut meta = Vec::with_capacity(reqs.len());
                        match mode {
                            InferMode::Shared => {
                                let mut players = Vec::with_capacity(reqs.len());
                                let mut obs_flat: Vec<f32> = Vec::new();
                                for (si, obs) in reqs {
                                    players.push(si);
                                    obs_flat.extend(obs);
                                    meta.push(si);
                                }
                                let n = meta.len();
                                queues[0].push(players, obs_flat, $gi, n);
                            }
                            InferMode::PerPlayer => {
                                for (pos, (si, obs)) in reqs.into_iter().enumerate() {
                                    queues[si].push_row(si, &obs, $gi, pos);
                                    meta.push(si);
                                }
                            }
                        }
                        let n = meta.len();
                        phases[$gi] = SlotPhase::AwaitingTail {
                            fragment: $fragment,
                            meta,
                            outstanding: n,
                            total: n,
                            rows: Vec::new(),
                            stride: 0,
                        };
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
                        // Cut step 5: bootstrap live fragments through the same queue.
                        if fragments && !frag_stage {
                            frag_stage = true;
                            let mut queued_any = false;
                            for gi in 0..n_games {
                                let mut guard = slots[gi].lock().expect("slot lock");
                                let slot: &mut SlotCtx<'_, G, P> = &mut guard;
                                if !slot.traj.iter().any(|steps| !steps.is_empty()) {
                                    continue;
                                }
                                let reqs =
                                    tail_requests(game, policy, learner, encoder, sequential, slot);
                                if reqs.is_empty() {
                                    backlog -= buffered_learn_steps(slot, learn_mask) as isize;
                                    flush_fragment_slot(
                                        &HashMap::new(),
                                        game,
                                        learner,
                                        encoder,
                                        learn_mask,
                                        slot,
                                        &mut out,
                                    );
                                    continue;
                                }
                                drop(guard);
                                enqueue_tails!(gi, reqs, true);
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
                        TaskOut::Emitted {
                            search,
                            perspectives,
                            players,
                            obs,
                            n,
                        } => {
                            match mode {
                                InferMode::Shared => queues[0].push(players, obs, msg.gi, n),
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
                            steps_delta,
                            finished,
                        } => {
                            let gi = msg.gi;
                            phases[gi] = SlotPhase::Idle;
                            out.extend(records);
                            stats = fold_stats(std::mem::take(&mut stats), tstats);
                            backlog += steps_delta;
                            let mut guard = slots[gi].lock().expect("slot lock");
                            let slot: &mut SlotCtx<'_, G, P> = &mut guard;
                            if finished != Some(true) {
                                start.observe(&slot.ep.state);
                            }
                            let mut tail_reqs = None;
                            match finished {
                                None => {}
                                Some(true) => respawn_game(game, policy, slot, &mut start),
                                Some(false) => {
                                    // Tail bootstraps are queue jobs: the queue is the
                                    // only inference pathway.
                                    let reqs = tail_requests(
                                        game, policy, learner, encoder, sequential, slot,
                                    );
                                    if reqs.is_empty() {
                                        backlog -= buffered_learn_steps(slot, learn_mask) as isize;
                                        flush_records_game(
                                            &HashMap::new(),
                                            game,
                                            learner,
                                            encoder,
                                            slot,
                                            &mut out,
                                            &mut stats,
                                        );
                                        respawn_game(game, policy, slot, &mut start);
                                    } else {
                                        tail_reqs = Some(reqs);
                                    }
                                }
                            }
                            drop(guard);
                            if let Some(reqs) = tail_reqs {
                                enqueue_tails!(gi, reqs, false);
                            }
                            // The floor counts only scheduler-ordered effects: records
                            // still riding unprocessed messages surface as the documented
                            // admitted-at-cut overshoot (bounded by in-flight decisions).
                            // Counting them earlier would need racy reads and break the
                            // n_threads=1 determinism contract.
                            let collected = if fragments {
                                out.len().saturating_add_signed(backlog)
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
    /// A finished (or fragment-cut) episode whose tail bootstraps ride the queue.
    AwaitingTail {
        fragment: bool,
        meta: Vec<usize>,
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
        steps_delta: isize,
        finished: Option<bool>,
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

/// Returns the slots whose outstanding rows reached zero with this batch (each at most
/// once, in routing order) so the caller settles only those.
fn fire_batch<SE, F>(
    queue: &mut RequestQueue,
    take: usize,
    phases: &mut [SE],
    evaluator: &mut Evaluator<'_, F>,
    mut route: impl FnMut(&mut SE) -> (&mut usize, &mut Vec<f64>, &mut usize, usize),
) -> Vec<usize>
where
    F: FnMut(usize, Vec<f32>, usize) -> Vec<f64>,
{
    let (start, dim) = (queue.head, queue.dim);
    let players = &queue.players[start..start + take];
    let obs = queue.obs[start * dim..(start + take) * dim].to_vec();
    let rows = evaluator.forward(players, obs, take);
    let stride = rows.len() / take;
    let mut freed = Vec::new();
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
        if *outstanding == 0 {
            freed.push(gi);
        }
    }
    queue.advance(take);
    freed
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
    a.sum_requested_rows += b.sum_requested_rows;
    a.sum_extra_eval_rows += b.sum_extra_eval_rows;
    a.infer_rows += b.infer_rows;
    a.padded_rows += b.padded_rows;
    a.episodes.extend(b.episodes);
    a
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
    if terminal || truncated {
        Some(terminal)
    } else {
        None
    }
}

/// Emit one finished game's episode records and respawn it.
#[allow(clippy::too_many_arguments)]
/// The records half of an episode flush: emit every perspective's records and the
/// episode summary. Pure with respect to shared engine state — safe on a worker.
fn flush_records_game<G, P, L>(
    tails: &HashMap<usize, Vec<f64>>,
    game: &G,
    learner: &L,
    encoder: &dyn StateEncoder<State = G::State>,
    slot: &mut SlotCtx<'_, G, P>,
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
}

/// The respawn half: reservoir draw and reset. Scheduler-only — the start
/// distribution's stream is engine state and its draw order is result-bearing.
fn respawn_game<G, P>(
    game: &G,
    policy: &P,
    slot: &mut SlotCtx<'_, G, P>,
    start: &mut StartAccess<'_, '_, G::State>,
) where
    G: Game + Sync,
    G::State: Send,
    P: Policy,
{
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

/// Buffered learning-player steps in one slot — the fragment floor's unit.
fn buffered_learn_steps<G: Game, P: Policy>(
    slot: &SlotCtx<'_, G, P>,
    learn_mask: &[bool],
) -> usize {
    slot.traj
        .iter()
        .enumerate()
        .filter(|&(si, _)| learn_mask[si])
        .map(|(_, steps)| steps.len())
        .sum()
}

/// One `(player, obs)` tail-bootstrap request per learning perspective; empty when the
/// learner takes no tail.
fn tail_requests<G, P, L>(
    game: &G,
    policy: &P,
    learner: &L,
    encoder: &dyn StateEncoder<State = G::State>,
    sequential: bool,
    slot: &SlotCtx<'_, G, P>,
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
    for (si, steps) in slot.traj.iter().enumerate() {
        if (all_perspectives || slot.ep.agent_active(game, si)) && !steps.is_empty() {
            reqs.push((si, slot.ep.observe(encoder, si)));
        }
    }
    reqs
}

/// Routed tail rows back to per-perspective tail values; zero-width (cancelled) rows
/// yield an empty map.
fn tails_from_rows<G, P, L>(
    meta: &[usize],
    rows: &[f64],
    stride: usize,
    game: &G,
    learner: &L,
    encoder: &dyn StateEncoder<State = G::State>,
    slot: &SlotCtx<'_, G, P>,
) -> HashMap<usize, Vec<f64>>
where
    G: Game + Sync,
    G::State: Send,
    P: Policy,
    L: Learner<P::Evaluation>,
{
    let mut tails = HashMap::new();
    if stride == 0 {
        return tails;
    }
    let a = game.action_count();
    let state = &slot.ep.state;
    for (i, &si) in meta.iter().enumerate() {
        let row = &rows[i * stride..(i + 1) * stride];
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

/// Cut one game's live trajectory: bootstrap each learning perspective from its tail and
/// emit its records; episode state, ticks, and telemetry persist into the next window.
fn flush_fragment_slot<G, P, L>(
    tails: &HashMap<usize, Vec<f64>>,
    game: &G,
    learner: &L,
    encoder: &dyn StateEncoder<State = G::State>,
    learn_mask: &[bool],
    slot: &mut SlotCtx<'_, G, P>,
    out: &mut Vec<L::Record>,
) where
    G: Game + Sync,
    G::State: Send,
    P: Policy,
    L: Learner<P::Evaluation>,
{
    #[allow(clippy::needless_range_loop)]
    for si in 0..game.num_agents() {
        if !learn_mask[si] {
            slot.traj[si].clear();
            continue;
        }
        let steps = std::mem::take(&mut slot.traj[si]);
        if steps.is_empty() {
            continue;
        }
        let tail = tails.get(&si).cloned().unwrap_or_default();
        out.extend(learner.episode_records(&steps, &tail, encoder, si, &mut slot.ep.rng));
    }
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
}
