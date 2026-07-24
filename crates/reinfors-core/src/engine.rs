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
        debug_assert!((1..=2).contains(&game.num_agents()));
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
                out.extend(
                    self.learner
                        .eval_records(&mut eval, &mut self.episodes[gi].rng),
                );
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
                    terminal: false,
                });
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
                        let next_obs = if needs_next_obs {
                            self.episodes[gi].observe(&*self.encoder, si)
                        } else {
                            Vec::new()
                        };
                        if let Some(step) = self.traj[gi][si].last_mut() {
                            step.reward = reward;
                            step.next_obs = next_obs;
                            step.terminal = terminal;
                        }
                    } else if let Some(step) = self.traj[gi][si].last_mut() {
                        // An agent that did not act this tick can still be scored by it — in a
                        // sequential game the loser's terminal event fires on the winner's move.
                        // Fold the reward into their last decision (0 for the common Ongoing/dead
                        // events, so this only ever carries real outcomes) and mark it terminal so
                        // the episode outcome reaches their trajectory.
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
                out.extend(
                    self.learner
                        .episode_records(&steps, &tail, &mut self.episodes[gi].rng),
                );
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
        let mut obs_flat: Vec<f32> = Vec::new();
        let mut meta: Vec<(usize, usize)> = Vec::new();
        for &(gi, terminal) in finished {
            if terminal {
                continue;
            }
            for si in 0..num_agents {
                if self.episodes[gi].agent_active(&self.game, si) && !self.traj[gi][si].is_empty() {
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
                tails.insert((gi, si), self.learner.tail_from_row(row, a, &legal));
            }
        }
        tails
    }
}
