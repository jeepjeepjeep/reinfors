//! Batched rollout and training-record collection.

use std::collections::HashMap;

use crate::codec::StateCodec;
use crate::encoder::StateEncoder;
use crate::game::{Actor, Game};
use crate::learner::{Learner, Step};
use crate::policy::Policy;
use crate::reward::Reward;
use crate::rng::SplitMix64;
use crate::rollout::arena::{Arena, Reserve};
use crate::rollout::episode::Episode;
use crate::rollout::evaluator::{Evaluator, InferMode};
use crate::rollout::infer_cache::InferCache;
use crate::rollout::start::{AlwaysInitialState, Start, StartDistribution};

pub use crate::stats::EpisodeSummary;

pub use crate::stats::CollectStats;

/// The start distribution and its reservoir stream — scheduler-owned: draw order is
/// result-bearing.
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
    /// Worker threads running search rounds (None = available cores; always capped
    /// at `n_games` — more workers than game slots can never run).
    pub n_threads: Option<usize>,
    /// Experimental zero-copy collection: workers copy encoded rows into shared
    /// arenas and the scheduler routes metadata only. Excludes infer caches and
    /// `pad` until the arena path supports them.
    pub zero_copy: bool,
}

impl Default for EngineParams {
    fn default() -> Self {
        EngineParams {
            n_games: 1,
            seed: 0,
            pad: false,
            batch_size: None,
            n_threads: None,
            zero_copy: false,
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
    zero_copy: bool,
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
            zero_copy: params.zero_copy,
            thread_pool: rayon::ThreadPoolBuilder::new()
                // At most one task per game slot can run: workers beyond n_games idle.
                .num_threads(
                    params
                        .n_threads
                        .or_else(|| std::thread::available_parallelism().ok().map(|n| n.get()))
                        .unwrap_or(1)
                        .clamp(1, params.n_games.max(1)),
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
        if self.zero_copy {
            return self.collect_arena(n_records, mode, infer);
        }
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
            let mut start = StartParts {
                dist: &mut *self.start_dist,
                rng: &mut self.buffer_rng,
            };
            let (tx, rx) = std::sync::mpsc::channel::<Msg<P::Search<G::State>, L::Record>>();

            let task = |gi: usize,
                        work: Work<P::Search<G::State>>,
                        tx: MsgSender<P::Search<G::State>, L::Record>| {
                let run = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let mut guard = slots[gi].lock().expect("slot lock");
                    let slot: &mut SlotCtx<'_, G, P> = &mut guard;
                    let (mut search, perspectives, mut roots) = match work {
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
                            let roots = vec![RootMark::Vacant; perspectives.len()];
                            (
                                policy.begin_search(ctx, &slot.ep.state, &perspectives),
                                perspectives,
                                roots,
                            )
                        }
                        Work::Resume {
                            mut search,
                            perspectives,
                            roots,
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
                            (search, perspectives, roots)
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
                    let mut sink = crate::policy::RequestSink::capturing_roots();
                    let status = policy.round(ctx, &mut search, &mut sink);
                    assert!(
                        (!sink.is_empty()) == (status == crate::policy::RoundStatus::Pending),
                        "round contract: Pending emits at least one request, Done emits none"
                    );
                    if !sink.is_empty() {
                        let n = sink.len();
                        let (players, obs, marked) = sink.into_parts_with_roots();
                        for (persp, row) in marked {
                            let i = perspectives
                                .iter()
                                .position(|&si| si == persp)
                                .unwrap_or_else(|| {
                                    panic!(
                                        "push_root marked perspective {persp}, which is not \
                                         among this search's requested perspectives"
                                    )
                                });
                            assert!(
                                matches!(roots[i], RootMark::Vacant),
                                "push_root marked perspective {persp} twice in one search"
                            );
                            roots[i] = if learn_mask[persp] {
                                RootMark::Row(row)
                            } else {
                                RootMark::Dropped
                            };
                        }
                        return TaskOut::Emitted {
                            search,
                            perspectives,
                            roots,
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
                        roots,
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
            let mut spin_budget = SPIN_RECVS_MIN;
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
                                    roots,
                                    rows,
                                    stride,
                                    ..
                                } => spawn(
                                    gi,
                                    Work::Resume {
                                        search,
                                        perspectives,
                                        roots,
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
                        // Tail bootstraps are queued requests like any other; tail_rows
                        // lets consumers separate them from decision evaluations.
                        stats.sum_requested_rows += n;
                        stats.sum_tail_rows += n;
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

                    // Spin before parking: light rounds return in microseconds, so
                    // a spin dodges the park/unpark syscalls that dominate them. The
                    // budget adapts on park latency — short parks double it, long
                    // parks halve it (spin hits leave it unchanged) — so heavy rounds
                    // decay to parking immediately and give up the core.
                    let mut msg = None;
                    for _ in 0..spin_budget {
                        match rx.try_recv() {
                            Ok(m) => {
                                msg = Some(m);
                                break;
                            }
                            Err(_) => std::hint::spin_loop(),
                        }
                    }
                    let msg = match msg {
                        Some(m) => m,
                        None => {
                            let parked = std::time::Instant::now();
                            let m = rx.recv().expect("scheduler channel closed");
                            spin_budget = if parked.elapsed() < SPIN_WORTH {
                                (spin_budget * 2).min(SPIN_RECVS_MAX)
                            } else {
                                (spin_budget / 2).max(SPIN_RECVS_MIN)
                            };
                            m
                        }
                    };
                    in_flight -= 1;
                    match msg.out {
                        TaskOut::Panicked(payload) => std::panic::resume_unwind(payload),
                        TaskOut::Skip => phases[msg.gi] = SlotPhase::Parked,
                        TaskOut::Emitted {
                            search,
                            perspectives,
                            roots,
                            players,
                            obs,
                            n,
                        } => {
                            // requested_rows counts rows once, where they enter the
                            // queue — the only inference pathway.
                            stats.sum_requested_rows += n;
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
                                roots,
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

    /// The zero-copy collection scheduler (stage 1): workers copy encoded rows into
    /// per-route arenas and messages carry span metadata only; the calling thread
    /// routes rows, fires sealed arenas, and never touches observation bytes.
    fn collect_arena<F>(
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
        assert!(
            self.infer_caches.is_none(),
            "zero_copy does not support infer caches yet"
        );
        assert!(!self.pad, "zero_copy does not support pad yet");
        let mut out: Vec<L::Record> = Vec::new();
        let mut stats = CollectStats::default();
        let collect_interior = self.learner.needs_interior();
        let fragments = self.learner.bootstraps_fragments();
        if fragments {
            discard_fragments(&mut self.traj);
        }
        let (c, h, w) = self.encoder.obs_shape();
        let obs_dim = c * h * w;
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
            let mut start = StartParts {
                dist: &mut *self.start_dist,
                rng: &mut self.buffer_rng,
            };
            let (tx, rx) = std::sync::mpsc::channel::<AMsg<P::Search<G::State>, L::Record>>();
            let n_routes = match mode {
                InferMode::Shared => 1,
                InferMode::PerPlayer => self.game.num_agents(),
            };
            let routes: Vec<RouteShared> = (0..n_routes)
                .map(|_| RouteShared::new(batch_size, obs_dim))
                .collect();
            let routes = &routes;
            let route_of = move |si: usize| match mode {
                InferMode::Shared => 0,
                InferMode::PerPlayer => si,
            };

            let task = |gi: usize,
                        work: AWork<P::Search<G::State>>,
                        tx: AMsgSender<P::Search<G::State>, L::Record>| {
                let run = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let mut guard = slots[gi].lock().expect("slot lock");
                    let slot: &mut SlotCtx<'_, G, P> = &mut guard;
                    let (mut search, perspectives, mut roots) = match work {
                        AWork::Begin => {
                            let perspectives: Vec<usize> = (0..game.num_agents())
                                .filter(|&si| slot.ep.agent_active(game, si))
                                .collect();
                            if perspectives.is_empty() {
                                return ATaskOut::Skip;
                            }
                            let ctx = crate::policy::SearchCtx {
                                game,
                                enc: encoder,
                                reward,
                                rng: &mut slot.ep.rng,
                                perms,
                                collect_interior,
                            };
                            let roots = vec![RootMark::Vacant; perspectives.len()];
                            (
                                policy.begin_search(ctx, &slot.ep.state, &perspectives),
                                perspectives,
                                roots,
                            )
                        }
                        AWork::Resume {
                            mut search,
                            perspectives,
                            roots,
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
                            (search, perspectives, roots)
                        }
                        AWork::Tail { fragment } => {
                            let reqs =
                                tail_requests(game, policy, learner, encoder, sequential, slot);
                            let n = reqs.len();
                            let mut meta = Vec::with_capacity(n);
                            let mut spans = Vec::new();
                            for (pos, (si, obs)) in reqs.into_iter().enumerate() {
                                let route = route_of(si);
                                push_route_rows(
                                    route,
                                    &routes[route],
                                    &[(pos as u32, obs.as_slice())],
                                    &mut spans,
                                );
                                meta.push(si);
                            }
                            return ATaskOut::TailEmitted {
                                fragment,
                                meta,
                                n,
                                spans,
                            };
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
                    let mut sink = crate::policy::RequestSink::capturing_roots();
                    let status = policy.round(ctx, &mut search, &mut sink);
                    assert!(
                        (!sink.is_empty()) == (status == crate::policy::RoundStatus::Pending),
                        "round contract: Pending emits at least one request, Done emits none"
                    );
                    if !sink.is_empty() {
                        let n = sink.len();
                        let (players, obs, marked) = sink.into_parts_with_roots();
                        for (persp, row) in marked {
                            let i = perspectives
                                .iter()
                                .position(|&si| si == persp)
                                .unwrap_or_else(|| {
                                    panic!(
                                        "push_root marked perspective {persp}, which is not \
                                         among this search's requested perspectives"
                                    )
                                });
                            assert!(
                                matches!(roots[i], RootMark::Vacant),
                                "push_root marked perspective {persp} twice in one search"
                            );
                            roots[i] = if learn_mask[persp] {
                                RootMark::Row(row)
                            } else {
                                RootMark::Dropped
                            };
                        }
                        // Stage-1 copy: the surviving worker-side copy into the arena.
                        let mut spans = Vec::new();
                        match mode {
                            InferMode::Shared => {
                                let rows: Vec<(u32, &[f32])> = (0..n)
                                    .map(|i| (i as u32, &obs[i * obs_dim..(i + 1) * obs_dim]))
                                    .collect();
                                push_route_rows(0, &routes[0], &rows, &mut spans);
                            }
                            InferMode::PerPlayer =>
                            {
                                #[allow(clippy::needless_range_loop)]
                                for route in 0..n_routes {
                                    let rows: Vec<(u32, &[f32])> = players
                                        .iter()
                                        .enumerate()
                                        .filter(|&(_, &pl)| pl == route)
                                        .map(|(i, _)| {
                                            (i as u32, &obs[i * obs_dim..(i + 1) * obs_dim])
                                        })
                                        .collect();
                                    if !rows.is_empty() {
                                        push_route_rows(route, &routes[route], &rows, &mut spans);
                                    }
                                }
                            }
                        }
                        return ATaskOut::Emitted {
                            search,
                            perspectives,
                            roots,
                            n,
                            spans,
                        };
                    }
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
                        roots,
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
                    ATaskOut::Completed {
                        records,
                        stats: tstats,
                        steps_delta,
                        finished,
                    }
                }));
                let out = match run {
                    Ok(out) => out,
                    Err(payload) => ATaskOut::Panicked(payload),
                };
                let _ = tx.send(AMsg { gi, out });
            };
            let task = &task;

            let mut phases: Vec<SlotPhase<P::Search<G::State>>> =
                (0..n_games).map(|_| SlotPhase::Idle).collect();
            let mut pending: Vec<std::collections::HashMap<u64, PendingArena>> = (0..n_routes)
                .map(|_| std::collections::HashMap::new())
                .collect();
            let mut in_flight = 0usize;
            let mut spin_budget = SPIN_RECVS_MIN;
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

            self.thread_pool.in_place_scope_fifo(|s| {
                let spawn = |gi: usize,
                             work_item: AWork<P::Search<G::State>>,
                             phases: &mut Vec<SlotPhase<P::Search<G::State>>>,
                             in_flight: &mut usize| {
                    phases[gi] = SlotPhase::Running;
                    *in_flight += 1;
                    let txc = tx.clone();
                    s.spawn_fifo(move |_| task(gi, work_item, txc));
                };
                // Callback panic: drain in-flight completions before surfacing, so a
                // finished episode respawns instead of stranding over-horizon.
                macro_rules! drain_after_callback_panic {
                    ($payload:expr) => {{
                        while in_flight > 0 {
                            let msg = rx.recv().expect("scheduler channel closed");
                            in_flight -= 1;
                            if let ATaskOut::Completed {
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
                        std::panic::resume_unwind($payload);
                    }};
                }
                macro_rules! settle_freed {
                    ($freed:expr) => {{
                        for gi in $freed {
                            match std::mem::replace(&mut phases[gi], SlotPhase::Idle) {
                                SlotPhase::Blocked {
                                    search,
                                    perspectives,
                                    roots,
                                    rows,
                                    stride,
                                    ..
                                } => spawn(
                                    gi,
                                    AWork::Resume {
                                        search,
                                        perspectives,
                                        roots,
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
                                        spawn(gi, AWork::Begin, &mut phases, &mut in_flight);
                                    }
                                }
                                _ => unreachable!("a freed slot must hold outstanding rows"),
                            }
                        }
                    }};
                }
                // Fire one fully accounted, closed arena: seal, call, route, settle.
                macro_rules! try_fire {
                    ($route:expr, $id:expr) => {{
                        let ready = pending[$route].get(&$id).is_some_and(|pa| {
                            pa.arc
                                .arena
                                .close_info()
                                .is_some_and(|info| info.rows == pa.seen)
                        });
                        if ready {
                            let pa = pending[$route].remove(&$id).expect("readiness checked");
                            let take = pa.arc.arena.close_info().expect("closed").rows;
                            let obs = pa
                                .arc
                                .arena
                                .take_rows()
                                .expect("every span accounted and committed");
                            let t0 = std::time::Instant::now();
                            let call = std::panic::catch_unwind(std::panic::AssertUnwindSafe(
                                || infer($route, obs, take),
                            ));
                            let rows = match call {
                                Ok(rows) => rows,
                                Err(payload) => drain_after_callback_panic!(payload),
                            };
                            stats.infer_seconds += t0.elapsed().as_secs_f64();
                            stats.infer_calls += 1;
                            stats.infer_rows += take;
                            assert!(
                                rows.len().is_multiple_of(take),
                                "infer returned {} values for {} rows (not divisible)",
                                rows.len(),
                                take
                            );
                            let stride = rows.len() / take;
                            let mut freed = Vec::new();
                            for i in 0..take {
                                let (g, pos) = pa.dest[i];
                                let (gi, pos) = (g as usize, pos as usize);
                                let (outstanding, buf, st, total) = match &mut phases[gi] {
                                    SlotPhase::Blocked {
                                        outstanding,
                                        rows: buf,
                                        stride: st,
                                        total,
                                        ..
                                    }
                                    | SlotPhase::AwaitingTail {
                                        outstanding,
                                        rows: buf,
                                        stride: st,
                                        total,
                                        ..
                                    } => (outstanding, buf, st, *total),
                                    _ => unreachable!(
                                        "row routed to a slot with no outstanding round"
                                    ),
                                };
                                if buf.is_empty() {
                                    buf.resize(total * stride, 0.0);
                                    *st = stride;
                                } else {
                                    assert_eq!(
                                        *st, stride,
                                        "infer returned a different row width for one round's requests"
                                    );
                                }
                                buf[pos * stride..(pos + 1) * stride]
                                    .copy_from_slice(&rows[i * stride..(i + 1) * stride]);
                                *outstanding -= 1;
                                if *outstanding == 0 {
                                    freed.push(gi);
                                }
                            }
                            settle_freed!(freed);
                        }
                    }};
                }
                macro_rules! account_spans {
                    ($gi:expr, $spans:expr) => {{
                        for span in $spans {
                            let route = span.route;
                            let id = span.arena.id;
                            let pa = pending[route].entry(id).or_insert_with(|| PendingArena {
                                arc: span.arena.clone(),
                                dest: vec![(u32::MAX, u32::MAX); batch_size],
                                seen: 0,
                            });
                            for (j, &pos) in span.positions.iter().enumerate() {
                                pa.dest[span.first_row + j] = ($gi as u32, pos);
                            }
                            pa.seen += span.positions.len();
                            try_fire!(route, id);
                        }
                    }};
                }
                if !cutting {
                    for k in 0..n_games {
                        let gi = (base_cursor + k) % n_games;
                        spawn(gi, AWork::Begin, &mut phases, &mut in_flight);
                    }
                }
                loop {
                    if in_flight == 0 {
                        if pending.iter().any(|m| !m.is_empty()) {
                            // Drain: close open arenas and fire partials — no round can
                            // progress without these rows.
                            #[allow(clippy::needless_range_loop)]
                            for route in 0..n_routes {
                                let mut ids: Vec<u64> = pending[route].keys().copied().collect();
                                ids.sort_unstable();
                                for id in ids {
                                    if let Some(pa) = pending[route].get(&id) {
                                        if pa.arc.arena.close_info().is_none() {
                                            pa.arc.arena.close();
                                        }
                                    }
                                    try_fire!(route, id);
                                }
                            }
                            continue;
                        }
                        // Cut step 5: bootstrap live fragments through worker tail tasks.
                        if fragments && !frag_stage {
                            frag_stage = true;
                            let mut spawned_any = false;
                            #[allow(clippy::needless_range_loop)]
                            for gi in 0..n_games {
                                let has_steps = slots[gi]
                                    .lock()
                                    .expect("slot lock")
                                    .traj
                                    .iter()
                                    .any(|steps| !steps.is_empty());
                                if !has_steps {
                                    continue;
                                }
                                spawn(
                                    gi,
                                    AWork::Tail { fragment: true },
                                    &mut phases,
                                    &mut in_flight,
                                );
                                spawned_any = true;
                            }
                            if spawned_any {
                                continue;
                            }
                        }
                        break;
                    }

                    let mut msg = None;
                    for _ in 0..spin_budget {
                        match rx.try_recv() {
                            Ok(m) => {
                                msg = Some(m);
                                break;
                            }
                            Err(_) => std::hint::spin_loop(),
                        }
                    }
                    let msg = match msg {
                        Some(m) => m,
                        None => {
                            let parked = std::time::Instant::now();
                            let m = rx.recv().expect("scheduler channel closed");
                            spin_budget = if parked.elapsed() < SPIN_WORTH {
                                (spin_budget * 2).min(SPIN_RECVS_MAX)
                            } else {
                                (spin_budget / 2).max(SPIN_RECVS_MIN)
                            };
                            m
                        }
                    };
                    in_flight -= 1;
                    match msg.out {
                        ATaskOut::Panicked(payload) => std::panic::resume_unwind(payload),
                        ATaskOut::Skip => phases[msg.gi] = SlotPhase::Parked,
                        ATaskOut::Emitted {
                            search,
                            perspectives,
                            roots,
                            n,
                            spans,
                        } => {
                            let gi = msg.gi;
                            stats.sum_requested_rows += n;
                            // The round mailbox precedes any accounting that could fire.
                            phases[gi] = SlotPhase::Blocked {
                                search,
                                perspectives,
                                roots,
                                outstanding: n,
                                total: n,
                                rows: Vec::new(),
                                stride: 0,
                            };
                            account_spans!(gi, spans);
                        }
                        ATaskOut::TailEmitted {
                            fragment,
                            meta,
                            n,
                            spans,
                        } => {
                            let gi = msg.gi;
                            if n == 0 {
                                phases[gi] = SlotPhase::Idle;
                                let mut guard = slots[gi].lock().expect("slot lock");
                                let slot: &mut SlotCtx<'_, G, P> = &mut guard;
                                backlog -= buffered_learn_steps(slot, learn_mask) as isize;
                                if fragment {
                                    flush_fragment_slot(
                                        &HashMap::new(),
                                        game,
                                        learner,
                                        encoder,
                                        learn_mask,
                                        slot,
                                        &mut out,
                                    );
                                } else {
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
                                drop(guard);
                                if !fragment {
                                    let collected = if fragments {
                                        out.len().saturating_add_signed(backlog)
                                    } else {
                                        out.len()
                                    };
                                    if collected >= n_records {
                                        cutting = true;
                                    }
                                    if !cutting && matches!(phases[gi], SlotPhase::Idle) {
                                        spawn(gi, AWork::Begin, &mut phases, &mut in_flight);
                                    }
                                }
                            } else {
                                stats.sum_requested_rows += n;
                                stats.sum_tail_rows += n;
                                phases[gi] = SlotPhase::AwaitingTail {
                                    fragment,
                                    meta,
                                    outstanding: n,
                                    total: n,
                                    rows: Vec::new(),
                                    stride: 0,
                                };
                                account_spans!(gi, spans);
                            }
                        }
                        ATaskOut::Completed {
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
                            let mut needs_tail = false;
                            match finished {
                                None => {}
                                Some(true) => respawn_game(game, policy, slot, &mut start),
                                Some(false) => {
                                    if learner.uses_episode_tail() {
                                        // Tail encoding is worker work on this path.
                                        needs_tail = true;
                                    } else {
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
                                    }
                                }
                            }
                            drop(guard);
                            if needs_tail {
                                spawn(
                                    gi,
                                    AWork::Tail { fragment: false },
                                    &mut phases,
                                    &mut in_flight,
                                );
                            }
                            let collected = if fragments {
                                out.len().saturating_add_signed(backlog)
                            } else {
                                out.len()
                            };
                            if collected >= n_records {
                                cutting = true;
                            }
                            if !cutting && matches!(phases[gi], SlotPhase::Idle) {
                                spawn(gi, AWork::Begin, &mut phases, &mut in_flight);
                            }
                        }
                    }
                }
            });
            if admitted {
                self.sweep_cursor = (base_cursor + 1) % n_games.max(1);
            }
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
        /// `push_root` rows, aligned with `perspectives`, carried across rounds.
        roots: Vec<RootMark>,
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
        roots: Vec<RootMark>,
        rows: Vec<f64>,
        stride: usize,
    },
}

/// One perspective's `push_root` slot: `Dropped` keeps duplicate detection for
/// non-learning perspectives without retaining their rows.
#[derive(Clone)]
enum RootMark {
    Vacant,
    Dropped,
    Row(Vec<f32>),
}

impl RootMark {
    fn take_row(&mut self) -> Option<Vec<f32>> {
        match std::mem::replace(self, RootMark::Dropped) {
            RootMark::Row(r) => Some(r),
            other => {
                *self = other;
                None
            }
        }
    }
}

/// A slot task's result. `Completed` carries the completion's effects — records, stats,
/// and the buffered-step delta — for the scheduler to apply in message order; `finished`
/// tells it whether to respawn or resolve a truncation tail.
enum TaskOut<SE, R> {
    Skip,
    Emitted {
        search: SE,
        perspectives: Vec<usize>,
        roots: Vec<RootMark>,
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

/// One arena in a route's ownership-per-fire sequence; `id` identifies it in span
/// metadata across the channel.
struct RouteArena {
    id: u64,
    arena: Arena,
}

/// A route's open-arena slot, shared with workers. Replacement arenas are
/// allocated by workers (uninitialized capacity — no memset), never the scheduler.
struct RouteShared {
    open: std::sync::Mutex<std::sync::Arc<RouteArena>>,
    next_id: std::sync::atomic::AtomicU64,
    capacity: usize,
    dim: usize,
}

impl RouteShared {
    fn new(capacity: usize, dim: usize) -> Self {
        RouteShared {
            open: std::sync::Mutex::new(std::sync::Arc::new(RouteArena {
                id: 0,
                arena: Arena::new(capacity, dim, 0),
            })),
            next_id: std::sync::atomic::AtomicU64::new(1),
            capacity,
            dim,
        }
    }

    fn current(&self) -> std::sync::Arc<RouteArena> {
        self.open.lock().expect("route lock").clone()
    }

    /// Install a fresh arena if `old` is still the open one; racing installers
    /// collapse to one winner and the losers reuse it.
    fn replace_if(&self, old: &std::sync::Arc<RouteArena>) {
        let mut open = self.open.lock().expect("route lock");
        if std::sync::Arc::ptr_eq(&open, old) {
            let id = self
                .next_id
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            *open = std::sync::Arc::new(RouteArena {
                id,
                arena: Arena::new(self.capacity, self.dim, 0),
            });
        }
    }
}

/// Span metadata riding a message: which arena rows hold which round positions.
/// Bytes never ride the channel.
struct ASpan {
    route: usize,
    arena: std::sync::Arc<RouteArena>,
    first_row: usize,
    positions: Vec<u32>,
}

/// Copy `(round position, encoded row)` pairs into `route`'s open arenas,
/// splitting across buffers at capacity.
fn push_route_rows(
    route_idx: usize,
    route: &RouteShared,
    rows: &[(u32, &[f32])],
    spans: &mut Vec<ASpan>,
) {
    let mut next = 0;
    while next < rows.len() {
        let cur = route.current();
        let (mut span, closes) = match cur.arena.try_reserve(rows.len() - next) {
            Reserve::Full { span, closed } => (span, closed.is_some()),
            Reserve::Partial { span, .. } => (span, true),
            Reserve::Closed => {
                route.replace_if(&cur);
                continue;
            }
        };
        let take = span.rows();
        let first = span.row_range().start;
        let mut positions = Vec::with_capacity(take);
        for (pos, row) in &rows[next..next + take] {
            span.push_row(row);
            positions.push(*pos);
        }
        span.commit();
        spans.push(ASpan {
            route: route_idx,
            arena: cur.clone(),
            first_row: first,
            positions,
        });
        next += take;
        if closes {
            route.replace_if(&cur);
        }
    }
}

/// Zero-copy work: `Tail` moves truncation-tail encoding onto workers.
enum AWork<SE> {
    Begin,
    Resume {
        search: SE,
        perspectives: Vec<usize>,
        roots: Vec<RootMark>,
        rows: Vec<f64>,
        stride: usize,
    },
    Tail {
        fragment: bool,
    },
}

/// Zero-copy task result: identical to `TaskOut` except observations stay in the
/// arenas and only span metadata rides the message.
enum ATaskOut<SE, R> {
    Skip,
    Emitted {
        search: SE,
        perspectives: Vec<usize>,
        roots: Vec<RootMark>,
        n: usize,
        spans: Vec<ASpan>,
    },
    TailEmitted {
        fragment: bool,
        meta: Vec<usize>,
        n: usize,
        spans: Vec<ASpan>,
    },
    Completed {
        records: Vec<R>,
        stats: CollectStats,
        steps_delta: isize,
        finished: Option<bool>,
    },
    Panicked(Box<dyn std::any::Any + Send>),
}

struct AMsg<SE, R> {
    gi: usize,
    out: ATaskOut<SE, R>,
}

type AMsgSender<SE, R> = std::sync::mpsc::Sender<AMsg<SE, R>>;

/// Scheduler-side accounting for one in-flight arena. `seen` counts rows through
/// processed messages only — never the live cursor — so the fire decision is a
/// pure function of message order.
struct PendingArena {
    arc: std::sync::Arc<RouteArena>,
    dest: Vec<(u32, u32)>,
    seen: usize,
}

/// Bounds for the adaptive busy-wait before a parking recv; parks that resolve
/// under SPIN_WORTH grow the budget, slower ones shrink it.
const SPIN_RECVS_MIN: u32 = 8;
const SPIN_RECVS_MAX: u32 = 4096;
const SPIN_WORTH: std::time::Duration = std::time::Duration::from_micros(50);

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
    a.sum_tail_rows += b.sum_tail_rows;
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
    mut roots: Vec<RootMark>,
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
    assert!(
        roots.len() == perspectives.len(),
        "root rows must stay aligned with perspectives across rounds"
    );
    let mut acted: Vec<Option<usize>> = vec![None; num_agents];
    for ((pi, (eval, targets)), &si) in evals.into_iter().enumerate().zip(perspectives.iter()) {
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
        let obs = roots[pi]
            .take_row()
            .unwrap_or_else(|| slot.ep.observe(encoder, si));
        slot.traj[si].push(Step {
            obs,
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
            // Chained: the next decision's obs is the successor; tails backfill at flush.
            let chained =
                learner.chains_successor_obs() && num_agents == 1 && !terminal && !truncated;
            let (next_obs, next_legal) = if needs_next_obs && !chained {
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
        let mut steps = std::mem::take(&mut slot.traj[si]);
        *ep_slot = std::mem::take(&mut slot.returns[si]);
        if steps.is_empty() {
            continue;
        }
        backfill_chained_tail(game, learner, encoder, &mut steps, slot, si);
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
    start: &mut StartParts<'_, G::State>,
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

/// A chained trajectory's last step may lack its successor observation; the episode
/// state is still its post-transition state, so fill it at the flush boundary.
fn backfill_chained_tail<G, P, L>(
    game: &G,
    learner: &L,
    encoder: &dyn StateEncoder<State = G::State>,
    steps: &mut [Step<P::Evaluation>],
    slot: &mut SlotCtx<'_, G, P>,
    si: usize,
) where
    G: Game + Sync,
    G::State: Send,
    P: Policy,
    L: Learner<P::Evaluation>,
{
    if !learner.chains_successor_obs() || game.num_agents() != 1 {
        return;
    }
    if let Some(last) = steps.last_mut() {
        if !last.terminal && last.next_obs.is_empty() {
            last.next_obs = slot.ep.observe(encoder, si);
            last.next_legal = game.legal_actions(&slot.ep.state, si);
        }
    }
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
        let mut steps = std::mem::take(&mut slot.traj[si]);
        if steps.is_empty() {
            continue;
        }
        backfill_chained_tail(game, learner, encoder, &mut steps, slot, si);
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
