//! Parallel rollout collector — the generic data-generation substrate. It is algorithm-agnostic: the
//! acting algorithm lives behind the [`Policy`] seam (evaluate the options + select an action) and the
//! training-record production behind the [`Learner`] seam (immediate records + episode-end records);
//! the Engine owns only what is common to every synchronous rollout.
//!
//! An `Engine<G, P, L>` holds N independent games of an arbitrary [`Game`]. Each `collect` step gathers
//! one request per active agent across every game, has the `Policy` evaluate them all in one pooled
//! pass (one batched `infer` per round — the throughput win), then per decision: folds the policy's
//! telemetry, lets the `Learner` emit its immediate records, has the `Policy` `select` an action, and
//! buffers the step; finally it advances every game and flushes finished ones' trajectories through the
//! `Learner` for their episode-end records.
//!
//! Per-game diversity comes from each game's own per-episode policy state (e.g. a Thompson head) and
//! its own RNG environment chance (snake's apple spawns): games start from the same deterministic
//! placement, so without this they would be identical. `step_env`/`initial_state` draw the true env
//! chance from each game's own RNG — the same `sample_chance` the search Monte-Carlos in-tree (from its
//! own seeded stream), so env and search share one chance model; see `search`. A truncation tick
//! reached alive pays the game's `truncation_bonus` (snake's `survival`), which the `Learner`'s z-mix
//! carries back to earlier steps.

use std::collections::HashMap;

use crate::encoder::StateEncoder;
use crate::episode::Episode;
use crate::game::Game;
use crate::learner::{Learner, Step};
use crate::policy::Policy;
use crate::rng::SplitMix64;

/// One finished episode's outcome, for logging: per-agent total realized reward (one entry per
/// agent) and the episode length in ticks.
#[derive(Clone)]
pub struct EpisodeSummary {
    pub reward: Vec<f64>,
    pub length: usize,
}

/// Telemetry for one `collect` call: finished-episode summaries and aggregated search diagnostics.
/// Search fields are sums over the call's `decisions`; the caller divides to get means (mirroring the
/// per-step scalars `snake_RL`'s `EnsembleTreeStrapRunner` logs).
#[derive(Default, Clone)]
pub struct CollectStats {
    pub episodes: Vec<EpisodeSummary>,
    pub decisions: usize,
    pub max_depth: i32,
    pub sum_leaves: f64,
    pub sum_rounds: f64,
    pub sum_expansions: f64,
    pub sum_sigma: f64, // sum over decisions of the search's mean leaf sigma
    pub sum_disagreement: f64, // sum over decisions of the root head-disagreement
}

/// Engine-level rollout knobs (everything that is not a game or *algorithm* parameter — search config,
/// interior-target flag, ensemble heads, and epsilon live on the `Policy`; z-mix `outcome_weight` and
/// bootstrap masking live on the `Learner`).
pub struct EngineParams {
    pub n_games: usize,
    pub max_ticks: usize,
    pub seed: u64,
}

pub struct Engine<G: Game + Sync, P: Policy, L: Learner<P::Evaluation>> {
    game: G,
    encoder: Box<dyn StateEncoder<State = G::State>>,
    policy: P,
    learner: L,
    params: EngineParams,
    episodes: Vec<Episode<G>>, // per-game live state + env-chance RNG (shared mechanics with `Env`)
    // The search's chance-sampling stream — independent of the per-game env RNGs so adding search
    // draws never perturbs the env's draw order (deterministic games stay bit-reproducible). A fresh
    // per-decision seed is drawn from it so each search samples with fresh randomness, like the env.
    search_rng: SplitMix64,
    policy_states: Vec<P::PolicyState>, // per-game acting state for the current episode (Thompson head)
    ticks: Vec<usize>,
    traj: Vec<Vec<Vec<Step<P::Evaluation>>>>, // [game][agent] decisions awaiting episode-end records
}

impl<G: Game + Sync, P: Policy, L: Learner<P::Evaluation>> Engine<G, P, L>
where
    G::State: Send,
{
    /// Build an engine over `game` driven by `policy` (evaluation + acting) and `learner` (training
    /// records), with the rollout knobs from `params`. The game owns its reward and rules.
    pub fn new(
        game: G,
        encoder: Box<dyn StateEncoder<State = G::State>>,
        policy: P,
        learner: L,
        params: EngineParams,
    ) -> Self {
        debug_assert!((1..=2).contains(&game.num_agents()));
        // Each game gets its own env-chance stream, seeded deterministically from the game index. The
        // episode draws its initial state from that stream, then the policy draws its episode state —
        // same per-game RNG order as before, so the rollout stays bit-reproducible.
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
        Engine {
            game,
            encoder,
            policy,
            learner,
            params,
            episodes,
            search_rng,
            policy_states,
            ticks,
            traj,
        }
    }

    /// Roll the games forward until at least `n_records` training records have been collected,
    /// returning each record's observation (a flat `[C*H*W]` buffer), per-head target `[K][A]`,
    /// and per-head bootstrap mask `[K]`. Executed decisions are z-mixed at episode end; interior MAX
    /// nodes (when enabled) are emitted immediately. `infer` is the value-network forward.
    pub fn collect<F>(&mut self, n_records: usize, mut infer: F) -> (Vec<L::Record>, CollectStats)
    where
        F: FnMut(Vec<f32>, usize) -> Vec<f64>,
    {
        // (The infer head-count check lives in the Python binding, where it can distinguish a real
        // wrong-K output from the error fallback and surface a clean error; here we only consume the
        // values, with the per-game clamp handling the all-terminal single-head case.)
        let mut out: Vec<L::Record> = Vec::new();
        let mut stats = CollectStats::default();
        let num_agents = self.game.num_agents();
        // Whether the policy should collect interior targets is the learner's call (it consumes them),
        // so the two can't silently disagree — mirrors `needs_next_obs`.
        let collect_interior = self.learner.needs_interior();

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
                break; // every game dead this instant (resets below normally keep at least one alive)
            }

            // 2. The policy evaluates them all in one pooled pass (one batched forward per round,
            //    shared across games — the throughput win); for selective expectimax this is the search.
            let search_seed = self.search_rng.next_u64();
            let evals = self.policy.evaluate(
                &self.game,
                &*self.encoder,
                requests,
                search_seed,
                collect_interior,
                &mut infer,
            );

            // 3. Per decision: fold its telemetry, emit the learner's immediate records (TreeStrap
            //    interior nodes), choose the action, and buffer the step. `acted[gi][si]` records the
            //    chosen action index for an agent that decided this tick.
            let mut acted: Vec<Vec<Option<usize>>> =
                vec![vec![None; num_agents]; self.episodes.len()];
            for (mut eval, &(gi, si)) in evals.into_iter().zip(meta.iter()) {
                stats.decisions += 1;
                self.policy.fold_telemetry(&eval, &mut stats);
                // The learner drains its immediate records out of the evaluation so interior nodes are
                // never buffered for the whole episode.
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
            //    record the executed decisions' rewards (plus the truncation bonus on a truncation
            //    tick); flush finished games' trajectories with z-mixing and reset them.
            let mut finished: Vec<(usize, bool)> = Vec::new(); // (game index, terminal?)
            for (gi, agents) in acted.into_iter().enumerate() {
                let joint: Vec<usize> = agents.iter().map(|a| a.unwrap_or(0)).collect();
                let (rewards, terminal) = self.episodes[gi].advance(&self.game, &joint);
                self.ticks[gi] += 1;
                // A truncation tick — max_ticks reached while the game is still playing — pays the
                // game's truncation bonus to agents that acted (snake's survival bonus to the living),
                // matching snake_RL's runner setting `survived_to_max_ticks` so it propagates through
                // z-mixing to earlier steps.
                let truncated = self.ticks[gi] >= self.params.max_ticks && !terminal;
                let needs_next_obs = self.learner.needs_next_obs();
                for (si, action) in agents.iter().enumerate() {
                    if action.is_some() {
                        let mut reward = rewards[si];
                        if truncated {
                            reward += self.game.truncation_bonus(&self.episodes[gi].state, si);
                        }
                        // `s'` for a transition learner (DQN): the post-transition observation. Skipped
                        // (left empty) for return-based learners so they pay no per-step obs cost.
                        let next_obs = if needs_next_obs {
                            self.episodes[gi].observe(&*self.encoder, si)
                        } else {
                            Vec::new()
                        };
                        // this agent acted this tick — attach the realized transition to its last decision
                        if let Some(step) = self.traj[gi][si].last_mut() {
                            step.reward = reward;
                            step.next_obs = next_obs;
                            step.terminal = terminal;
                        }
                    }
                }
                if terminal || self.ticks[gi] >= self.params.max_ticks {
                    finished.push((gi, terminal));
                }
            }

            self.flush_finished(&finished, &mut out, &mut stats, &mut infer);
        }
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
        infer: &mut F,
    ) where
        F: FnMut(Vec<f32>, usize) -> Vec<f64>,
    {
        // On truncation (alive, not terminal) z is seeded by the net's per-head state value, so early
        // decisions of long episodes are not systematically undervalued. Batch those into one forward.
        let tails = self.tail_values(finished, infer);
        let num_agents = self.game.num_agents();

        for &(gi, _) in finished {
            let mut ep_reward = vec![0.0; num_agents];
            for (si, ep_slot) in ep_reward.iter_mut().enumerate() {
                let steps = std::mem::take(&mut self.traj[gi][si]);
                if steps.is_empty() {
                    continue;
                }
                *ep_slot = steps.iter().map(|s| s.reward).sum();
                // The tail is the truncation bootstrap (one per active agent), or empty for a terminal
                // episode — the learner seeds a zero tail of the right head count when it is empty (it
                // knows the head count from its concrete evaluation; the generic engine does not).
                let tail = tails.get(&(gi, si)).cloned().unwrap_or_default();
                // The learner turns the buffered trajectory into records (TreeStrap z-mixing + masks).
                out.extend(
                    self.learner
                        .episode_records(&steps, &tail, &mut self.episodes[gi].rng),
                );
            }
            stats.episodes.push(EpisodeSummary {
                reward: ep_reward,
                length: self.ticks[gi],
            });
            self.episodes[gi].reset(&self.game);
            self.ticks[gi] = 0;
            self.policy_states[gi] = self.policy.begin_episode(&mut self.episodes[gi].rng);
        }
    }

    /// Per-(game, agent) z-tail: the net's per-head value `max_a Q(final_obs)` for agents still active
    /// at a truncation (terminal episodes and inactive agents seed with 0, so they are absent here).
    /// Empty when `outcome_weight == 0`, where the tail never enters the (no-op) blend.
    fn tail_values<F>(
        &mut self,
        finished: &[(usize, bool)],
        infer: &mut F,
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
            // A terminal episode seeds the tail with 0; only a truncation (an episode that ended
            // because the horizon was reached, with an agent still active) gets a net-value tail.
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
            let q = infer(obs_flat, meta.len()); // flat [n, k, A], row-major
            let k = q.len() / (meta.len() * a);
            for (i, &key) in meta.iter().enumerate() {
                let row = &q[i * k * a..(i + 1) * k * a]; // [k, A], head-major
                let per_head = (0..k)
                    .map(|h| {
                        row[h * a..(h + 1) * a]
                            .iter()
                            .copied()
                            .fold(f64::NEG_INFINITY, f64::max)
                    })
                    .collect();
                tails.insert(key, per_head);
            }
        }
        tails
    }
}
