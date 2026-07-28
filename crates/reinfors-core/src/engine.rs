//! Parallel rollout collector — the generic data-generation substrate. It is algorithm-agnostic: the
//! acting algorithm lives behind the [`Policy`] seam (evaluate the options + select an action) and the
//! training-record production behind the [`Learner`] seam (immediate records + episode-end records);
//! the Engine owns only what is common to every synchronous rollout.
//!
//! An `Engine<G, P, L>` holds N independent games of an arbitrary [`Game`]. Each `collect` step gathers
//! one request per active agent across every game, has the `Policy` evaluate them all in one pooled
//! pass (one batched `infer` per round), then per decision: folds the policy's
//! telemetry, lets the `Learner` emit its immediate records, has the `Policy` `select` an action, and
//! buffers the step; finally it advances every game and flushes finished ones' trajectories through the
//! `Learner` for their episode-end records.
//!
//! Per-game diversity comes from each game's own per-episode policy state and
//! its own RNG environment chance: games start from the same deterministic
//! placement, so without this they would be identical. The framework realizes env chance
//! (`game::step_env`, one draw from the game's DECLARED distribution) from each game's own RNG —
//! the same `chance_outcomes` the searches consume from their own seeded streams, so env and
//! search share one chance model by construction. When a game hits its
//! `truncation_horizon`, the engine has it `mark_truncation` the tick's events (e.g. snake's survival
//! flag) so the `Reward` scores the bonus, which the `Learner`'s z-mix carries back to earlier steps.

use std::collections::HashMap;

use crate::encoder::StateEncoder;
use crate::episode::Episode;
use crate::game::Game;
use crate::learner::{Learner, Step};
use crate::policy::Policy;
use crate::reward::Reward;
use crate::rng::SplitMix64;
use crate::start::{AlwaysInitialState, Start, StartDistribution};

/// One finished episode's outcome, for logging: per-agent total realized reward (one entry per
/// agent), the episode length in ticks, and whether it was seeded from the start-state buffer (a
/// `StartDistribution::Restore`) rather than a fresh `initial_state` — so off-`d₀` episodes can be kept
/// out of the true-start learning curves.
#[derive(Clone)]
pub struct EpisodeSummary {
    pub reward: Vec<f64>,
    pub length: usize,
    pub seeded: bool,
}

/// Telemetry for one `collect` call: finished-episode summaries and aggregated search diagnostics.
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
    // Global Evaluator throughput for this collect — every forward from every consumer (search
    // leaves, episode-tail bootstraps, plain policy forwards), measured at the single choke point.
    // Timing the `collect()` call and subtracting `infer_seconds` gives the search (game sim + tree
    // expansion + assembly) cost; `infer_rows / infer_calls` is the mean batch size. With the infer
    // cache enabled, `infer_rows` counts only MISS rows, so rows-per-state falls as the hit rate
    // rises. `cache_lookups`/`cache_hits` are likewise global (0 when the cache is disabled).
    pub infer_seconds: f64,
    pub infer_calls: usize,
    pub infer_rows: usize,
    pub cache_lookups: usize,
    pub cache_hits: usize,
    // Tree-search sim fates summed over the collect (0 for non-tree policies), counted by the
    // trees themselves — see `SearchStats`. These alone assemble the exact per-collect identity
    // `decisions × num_simulations =
    //   sum_fresh_rows + sum_hit_rows + sum_shared_rows + sum_terminal_sims + sum_depthcap_sims`;
    // no global counter appears in it, so non-search forwards cannot unbalance it.
    pub sum_terminal_sims: usize,
    pub sum_depthcap_sims: usize,
    pub sum_shared_rows: usize,
    pub sum_fresh_rows: usize,
    pub sum_hit_rows: usize,
    pub sum_extra_eval_rows: usize,
}

/// Engine-level rollout knobs. The truncation horizon is the game's (`truncation_horizon`), not an
/// engine knob — the engine only counts ticks and enforces it.
pub struct EngineParams {
    pub n_games: usize,
    pub seed: u64,
}

pub struct Engine<G: Game + Sync, P: Policy, L: Learner<P::Evaluation>> {
    game: G,
    encoder: Box<dyn StateEncoder<State = G::State>>,
    reward: Box<dyn Reward<Event = G::Event>>,
    policy: P,
    learner: L,
    episodes: Vec<Episode<G>>,
    search_rng: SplitMix64,
    // Where each fresh episode starts (default: the game's `initial_state`). Its RNG is a stream
    // disjoint from `search_rng` and the per-game env RNGs, so an enabled buffer never perturbs the
    // env-chance draw order — and `AlwaysInitialState` never draws it, so it stays bit-identical.
    start_dist: Box<dyn StartDistribution<G::State>>,
    // Optional net-evaluation cache (see `infer_cache`), applied inside the per-collect
    // `Evaluator` — the single path every consumer's forwards take. Lifetime spans collects;
    // cleared when the shared weights generation is bumped (`weights_updated` at the binding).
    infer_cache: Option<crate::infer_cache::InferCache>,
    // The game's decision dynamics (probed once from the initial state): sequential games with
    // N>2 agents buffer value-only steps for non-mover perspectives (see `collect`).
    sequential: bool,
    buffer_rng: SplitMix64,
    seeded: Vec<bool>, // per game: was the current episode seeded from the start buffer?
    policy_states: Vec<P::PolicyState>, // per-game acting state for the current episode (Thompson head)
    ticks: Vec<usize>,
    traj: Vec<Vec<Vec<Step<P::Evaluation>>>>, // [game][agent] decisions awaiting episode-end records
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
        // Real asserts, not debug ones: an unsupported composition doesn't fail loudly later, it
        // silently computes wrong values (or overflows the joint space). The binding pre-checks
        // and errors; these are the backstops for direct core callers. Dynamics are probed from
        // the initial state (games are uniformly one dynamics; the searches assert mixing).
        let sequential = episodes
            .first()
            .is_some_and(|ep| matches!(game.actor(&ep.state), crate::game::Actor::Agent(_)));
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
        let seeded = vec![false; params.n_games]; // the initial episodes are fresh
        Engine {
            game,
            encoder,
            reward,
            policy,
            learner,
            episodes,
            search_rng,
            start_dist: Box::new(AlwaysInitialState),
            infer_cache: None,
            sequential,
            buffer_rng,
            seeded,
            policy_states,
            ticks,
            traj,
        }
    }

    /// Override the start-state distribution (default [`AlwaysInitialState`]). A
    /// [`ReachedStateBuffer`](crate::ReachedStateBuffer) seeds some episodes from previously-reached
    /// states to flatten start-state coverage. Consuming builder, so the common (default) `new` path
    /// stays untouched.
    pub fn with_start_distribution(
        mut self,
        start_dist: Box<dyn StartDistribution<G::State>>,
    ) -> Self {
        self.start_dist = start_dist;
        self
    }

    /// Enable the net-evaluation cache (consuming builder, like `with_start_distribution`).
    pub fn with_infer_cache(mut self, cache: crate::infer_cache::InferCache) -> Self {
        self.infer_cache = Some(cache);
        self
    }

    /// Roll the games forward until at least `n_records` training records have been collected,
    /// returning each record's observation (a flat `[C*H*W]` buffer), per-head target `[K][A]`,
    /// and per-head bootstrap mask `[K]`. Executed decisions are z-mixed at episode end; interior MAX
    /// nodes (when enabled) are emitted immediately. `infer` is the value-network forward.
    pub fn collect<F>(&mut self, n_records: usize, mut infer: F) -> (Vec<L::Record>, CollectStats)
    where
        F: FnMut(Vec<f32>, usize) -> Vec<f64>,
    {
        let mut out: Vec<L::Record> = Vec::new();
        let mut stats = CollectStats::default();
        let num_agents = self.game.num_agents();
        let collect_interior = self.learner.needs_interior();
        // Take the cache out of `self` for the collect so the Evaluator (which lives across the
        // whole loop) can hold it without pinning a long-lived borrow of `self`; restored at the end.
        let mut cache = self.infer_cache.take();
        if let Some(c) = cache.as_mut() {
            c.begin_collect();
        }
        // The single evaluation service for this collect: every consumer's forwards (search
        // leaves, episode-tail bootstraps, plain policy forwards) route through it, picking up
        // caching, within-batch dedup, and throughput telemetry uniformly.
        let mut evaluator = crate::evaluator::Evaluator::new(&mut infer, cache.as_mut());

        while out.len() < n_records {
            // 1. Gather one search request per active agent across all games.
            let mut requests: Vec<(G::State, usize)> = Vec::new();
            let mut meta: Vec<(usize, usize)> = Vec::new(); // (game index, agent index)
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

            // 2. The policy evaluates them all in one pooled pass
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

            // 3. Per decision: fold its telemetry, emit the learner's immediate records,
            //    choose the action, and buffer the step. `acted[gi][si]` records the
            //    chosen action index for an agent that decided this tick.
            let mut acted: Vec<Vec<Option<usize>>> =
                vec![vec![None; num_agents]; self.episodes.len()];
            for (mut eval, &(gi, si)) in evals.into_iter().zip(meta.iter()) {
                stats.decisions += 1;
                self.policy.fold_telemetry(&eval, &mut stats);
                out.extend(self.learner.eval_records(
                    &mut eval,
                    &*self.encoder,
                    si,
                    &mut self.episodes[gi].rng,
                ));
                let rel = self.policy.select(
                    &eval,
                    &mut self.policy_states[gi],
                    &mut self.episodes[gi].rng,
                );
                acted[gi][si] = Some(rel);
                self.traj[gi][si].push(Step {
                    obs: self.episodes[gi].observe(&*self.encoder, si),
                    evaluation: eval,
                    action: rel,
                    reward: 0.0, // filled in from this tick's transition after advancing
                    next_obs: Vec::new(), // filled below when the learner needs it
                    next_legal: Vec::new(),
                    terminal: false,
                });
            }

            // 3b. Sequential N>2 general-sum: every non-mover perspective of this real self-play
            //    state gets a VALUE-ONLY step in its own trajectory (its observation now; its own
            //    reward stream stamps it below, exactly per tick) — so the per-perspective leaf
            //    values Max^N consumes are supervised. The learner opts in by supplying the
            //    placeholder evaluation; its records carry policy weight 0. Perspectives are
            //    emitted only where the search consumes them: 2p-sequential (negamax reads the
            //    mover row only) and simultaneous games (all active agents hold real steps)
            //    buffer nothing extra.
            if self
                .policy
                .evaluates_all_perspectives(self.sequential, num_agents)
            {
                let action_count = self.game.action_count();
                for (gi, agents) in acted.iter().enumerate() {
                    if agents.iter().all(|s| s.is_none()) {
                        continue; // no decision this tick
                    }
                    for (si, slot) in agents.iter().enumerate() {
                        if slot.is_none() {
                            if let Some(evaluation) =
                                self.learner.value_only_evaluation(action_count)
                            {
                                self.traj[gi][si].push(Step {
                                    obs: self.episodes[gi].observe(&*self.encoder, si),
                                    evaluation,
                                    action: 0,
                                    reward: 0.0, // stamped from this tick's transition below
                                    next_obs: Vec::new(),
                                    next_legal: Vec::new(),
                                    terminal: false,
                                });
                            }
                        }
                    }
                }
            }

            // 4. Advance every game via the env transition (sampling its chance from the per-game RNG);
            //    record the executed decisions' rewards; flush finished games' trajectories with
            //    z-mixing and reset them. On a truncation tick the game stamps the truncation outcome
            //    onto the events (`mark_truncation`), so the survival reward flows through `step_reward`
            //    like any other outcome — no separate truncation-reward path.
            let horizon = self.game.truncation_horizon();
            let mut finished: Vec<(usize, bool)> = Vec::new(); // (game index, terminal?)
            for (gi, agents) in acted.into_iter().enumerate() {
                let joint: Vec<usize> = agents.iter().map(|a| a.unwrap_or(0)).collect();
                let (mut events, terminal) = self.episodes[gi].advance(&self.game, &joint);
                self.ticks[gi] += 1;
                let truncated = horizon.is_some_and(|h| self.ticks[gi] >= h) && !terminal;
                if truncated {
                    self.game
                        .mark_truncation(&self.episodes[gi].state, &mut events);
                }
                let needs_next_obs = self.learner.needs_next_obs();
                for (si, action) in agents.iter().enumerate() {
                    let reward = self.reward.step_reward(&events[si], si);
                    if action.is_some() {
                        let (next_obs, next_legal) = if needs_next_obs {
                            (
                                self.episodes[gi].observe(&*self.encoder, si),
                                self.game.legal_actions(&self.episodes[gi].state, si),
                            )
                        } else {
                            (Vec::new(), Vec::new())
                        };
                        if let Some(step) = self.traj[gi][si].last_mut() {
                            step.reward = reward;
                            step.next_obs = next_obs;
                            step.next_legal = next_legal;
                            step.terminal = terminal;
                        }
                    } else if let Some(step) = self.traj[gi][si].last_mut() {
                        // An agent that did not act this tick can still be scored by it — in a
                        // sequential game the loser's terminal event fires on the winner's move.
                        // Fold the reward into their last buffered step (their previous decision;
                        // or, when value-only steps are on, THIS tick's value step — per-tick
                        // exact) and mark it terminal so the outcome reaches their trajectory.
                        step.reward += reward;
                        step.terminal |= terminal;
                    }
                }
                // Buffer the reached state for start-state coverage — non-terminal states only (you
                // can't restart from a terminal one). A no-op under `AlwaysInitialState` (default).
                if !terminal {
                    self.start_dist
                        .observe(&self.episodes[gi].state, &mut self.buffer_rng);
                }
                if terminal || truncated {
                    finished.push((gi, terminal));
                }
            }

            self.flush_finished(&finished, &mut out, &mut stats, &mut evaluator);
        }
        (stats.infer_seconds, stats.infer_calls, stats.infer_rows) =
            (evaluator.seconds, evaluator.calls, evaluator.rows);
        (stats.cache_lookups, stats.cache_hits) =
            (evaluator.cache_lookups(), evaluator.cache_hits());
        self.infer_cache = cache; // the evaluator's borrow ends above; put the cache back
        (out, stats)
    }

    /// Flush each finished game's buffered trajectories: z-mix the executed-action targets with the
    /// realized return (tail-seeded by the net's value of the final state on truncation), emit the
    /// records, then reset the game and resample its head + initial chance.
    fn flush_finished<F>(
        &mut self,
        finished: &[(usize, bool)],
        out: &mut Vec<L::Record>,
        stats: &mut CollectStats,
        evaluator: &mut crate::evaluator::Evaluator<'_, F>,
    ) where
        F: FnMut(Vec<f32>, usize) -> Vec<f64>,
    {
        let tails = self.tail_values(finished, evaluator);
        let num_agents = self.game.num_agents();

        for &(gi, _) in finished {
            let mut ep_reward = vec![0.0; num_agents];
            for (si, ep_slot) in ep_reward.iter_mut().enumerate() {
                let steps = std::mem::take(&mut self.traj[gi][si]);
                if steps.is_empty() {
                    continue;
                }
                *ep_slot = steps.iter().map(|s| s.reward).sum();
                let tail = tails.get(&(gi, si)).cloned().unwrap_or_default();
                out.extend(self.learner.episode_records(
                    &steps,
                    &tail,
                    &*self.encoder,
                    si,
                    &mut self.episodes[gi].rng,
                ));
            }
            // Tag the summary with the finishing episode's seeded flag (set at its start) BEFORE the
            // reset below overwrites it for the next episode.
            stats.episodes.push(EpisodeSummary {
                reward: ep_reward,
                length: self.ticks[gi],
                seeded: self.seeded[gi],
            });
            // Start the next episode: the buffer either restores a reached state or falls back to a
            // fresh `initial_state`. `AlwaysInitialState` always chooses `Fresh` and draws no buffer
            // RNG, so this is the current reset path unchanged.
            match self.start_dist.choose(&mut self.buffer_rng) {
                Start::Restore(state) => {
                    self.episodes[gi].state = state;
                    self.seeded[gi] = true;
                }
                Start::Fresh => {
                    self.episodes[gi].reset(&self.game);
                    self.seeded[gi] = false;
                }
            }
            self.ticks[gi] = 0;
            self.policy_states[gi] = self.policy.begin_episode(&mut self.episodes[gi].rng);
        }
    }

    /// Per-(game, agent) z-tail: the net's per-head value `max_a Q(final_obs)` for agents still active
    /// at a truncation (terminal episodes and inactive agents seed with 0, so they are absent here).
    /// Empty when `outcome_weight == 0`, where the tail never enters the (no-op) blend.
    /// An ordinary [`Evaluator`](crate::evaluator::Evaluator) consumer: the final state was almost
    /// always just evaluated by the last search, so with the cache on this is typically hit-served.
    fn tail_values<F>(
        &mut self,
        finished: &[(usize, bool)],
        evaluator: &mut crate::evaluator::Evaluator<'_, F>,
    ) -> HashMap<(usize, usize), Vec<f64>>
    where
        F: FnMut(Vec<f32>, usize) -> Vec<f64>,
    {
        let mut tails: HashMap<(usize, usize), Vec<f64>> = HashMap::new();
        if !self.learner.uses_episode_tail() {
            return tails;
        }
        let a = self.game.action_count();
        let num_agents = self.game.num_agents();
        // Under the value-only regime every perspective holds a per-tick trajectory and the
        // search consumes every perspective's value at the final state — so every non-empty
        // trajectory gets its own tail V_i(final_state), active or not (a sequential non-mover
        // has no legal actions there, but the AZ tail reads the value slot, not the legal set).
        // Off the regime, the active-only condition is unchanged.
        let all_perspectives =
            self.sequential && num_agents > 2 && self.learner.value_only_evaluation(a).is_some();
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
            let q = evaluator.forward(obs_flat, meta.len()); // one flat row per state; layout = the family's contract
            let stride = q.len() / meta.len();
            for (i, &(gi, si)) in meta.iter().enumerate() {
                let row = &q[i * stride..(i + 1) * stride];
                // The learner knows its family's row layout (default: [K][A] Q-rows, per-head max
                // over the state's LEGAL actions — the mover-convention set, so a truncation tail
                // on a sparse-action game cannot bootstrap a phantom illegal Q).
                let state = &self.episodes[gi].state;
                let legal = match self.game.actor(state) {
                    crate::game::Actor::Agent(mover) => self.game.legal_actions(state, mover),
                    crate::game::Actor::Simultaneous => self.game.legal_actions(state, si),
                    crate::game::Actor::Chance => unreachable!("chance actors are not searched"),
                };
                // The row was encoded for `si`, so the gather maps game ids through si's frame.
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

/// Exact-resume snapshot/restore of the engine's MUTABLE collection state. Immutable composition
/// (game/reward/policy/learner/encoder config) is NOT here — it is reconstructed from the resolved
/// config, and the binding gates restore on the config fingerprint. The infer cache is deliberately
/// excluded: cache hits return bit-identical rows to the forwards they replace, so collected
/// RECORDS after restore are byte-identical with a cold cache — the guarantee is record-exact,
/// not inference-call-pattern-exact.
impl<G: Game + Sync, P: Policy, L: Learner<P::Evaluation>> Engine<G, P, L>
where
    G::State: Send,
{
    pub fn snapshot_bytes(
        &self,
        codec: &dyn crate::StateCodec<State = G::State>,
    ) -> Result<Vec<u8>, String> {
        use crate::codec::bytes::*;
        let mut out = vec![1u8]; // engine payload layout version
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
        codec: &dyn crate::StateCodec<State = G::State>,
        bytes: &[u8],
    ) -> Result<(), String> {
        use crate::codec::bytes::*;
        let mut r = Reader::new(bytes);
        if r.u8()? != 1 {
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
        // Decode EVERYTHING before mutating anything: a malformed snapshot must leave the engine
        // untouched.
        struct GameSlice<S, E, PS> {
            state: S,
            rng: u64,
            tick: usize,
            seeded: bool,
            policy_state: PS,
            traj: Vec<Vec<Step<E>>>,
        }
        let mut slices: Vec<GameSlice<G::State, P::Evaluation, P::PolicyState>> =
            Vec::with_capacity(n_games);
        for _ in 0..n_games {
            let state = codec.decode(r.blob()?)?;
            // Engine episodes are always live (terminal episodes flush and reset immediately).
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
                policy_state,
                traj,
            });
        }
        let start_blob = r.blob()?.to_vec();
        r.done()?;
        // The first mutation on a still-fallible path: the trait contract requires implementors
        // to restore transactionally (decode fully, then swap), so an Err here leaves the
        // distribution — and everything below, not yet touched — unchanged.
        self.start_dist.restore_bytes(&start_blob, &|b| {
            let s = codec.decode(b)?;
            codec.validate_decoded_state(&s, false)?; // buffered start states are mid-episode: live
            Ok(s)
        })?;
        self.search_rng = SplitMix64::from_state(search_rng);
        self.buffer_rng = SplitMix64::from_state(buffer_rng);
        for (gi, slice) in slices.into_iter().enumerate() {
            self.episodes[gi].state = slice.state;
            self.episodes[gi].rng = SplitMix64::from_state(slice.rng);
            self.ticks[gi] = slice.tick;
            self.seeded[gi] = slice.seeded;
            self.policy_states[gi] = slice.policy_state;
            self.traj[gi] = slice.traj;
        }
        // The cache may hold rows from OTHER weights at a numerically equal generation — stale
        // rows would silently break the record-exact restore contract. Cold cache = same records.
        if let Some(cache) = self.infer_cache.as_mut() {
            cache.force_clear();
        }
        Ok(())
    }
}
