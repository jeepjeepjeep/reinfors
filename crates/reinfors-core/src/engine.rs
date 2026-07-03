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
//! placement, so without this they would be identical. `step_env`/`initial_state` draw the true env
//! chance from each game's own RNG — the same `sample_chance` the search Monte-Carlos in-tree (from its
//! own seeded stream), so env and search share one chance model; see `search`. A truncation tick
//! reached alive pays the game's `truncation_bonus`, which the `Learner`'s z-mix
//! carries back to earlier steps.

use std::collections::HashMap;

use crate::encoder::StateEncoder;
use crate::episode::Episode;
use crate::game::Game;
use crate::learner::{Learner, Step};
use crate::policy::Policy;
use crate::reward::Reward;
use crate::rng::SplitMix64;

/// One finished episode's outcome, for logging: per-agent total realized reward (one entry per
/// agent) and the episode length in ticks.
#[derive(Clone)]
pub struct EpisodeSummary {
    pub reward: Vec<f64>,
    pub length: usize,
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
        Engine {
            game,
            encoder,
            reward,
            policy,
            learner,
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
        let mut out: Vec<L::Record> = Vec::new();
        let mut stats = CollectStats::default();
        let num_agents = self.game.num_agents();
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
                &mut infer,
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
                    if action.is_some() {
                        let reward = self.reward.step_reward(&events[si], si);
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
                    }
                }
                if terminal || truncated {
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
                let tail = tails.get(&(gi, si)).cloned().unwrap_or_default();
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
