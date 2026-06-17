//! Parallel rollout collector — the data generator that turns the search core into training data.
//!
//! An `Engine<G>` holds N independent games of an arbitrary [`Game`]. Each `collect` step runs the
//! pooled selective search for every active agent across every game in lockstep (one batched `infer`
//! per round, shared across all games — the throughput win), records each decision's
//! `(observation, searched per-head values)` as a TreeStrap target, picks an action by Thompson
//! sampling (one head per game per episode) with an epsilon-greedy override, advances every game, and
//! resets finished ones.
//!
//! Per-game diversity comes from each game drawing its own Thompson head and epsilon noise, plus its
//! own RNG environment chance (e.g. snake's apple spawns): games start from the same deterministic
//! placement, so without this they would be identical. The game's `step_env`/`initial_state` draw the
//! true environment chance from each game's own RNG — the same `sample_chance` the search Monte-Carlos
//! in-tree (from its own seeded stream), so env and search share one chance model; see `search`.)
//!
//! Records carry the full TreeStrap training semantics of `snake_RL`'s `EnsembleTreeStrapRunner`:
//! - **z-mixing** — each executed decision is held back until its episode ends, then the executed
//!   action's entry of every head is blended with the realized discounted return (`blend_outcome_targets`),
//!   so deaths the search failed to foresee still reach the training signal. A truncation tick reached
//!   alive pays the game's `truncation_bonus` (snake's `survival`), which z-mixing carries to earlier steps.
//! - **interior targets** — with `interior_targets`, every expanded interior MAX node of the search
//!   tree is emitted as an extra `(obs, values)` record (true TreeStrap). These are counterfactual
//!   states with no realized outcome, so they are emitted immediately and never z-blended.
//! - **bootstrap masks** — every record carries a per-head `(K,)` mask (`rng < bootstrap_p`) so the
//!   ensemble heads train on different subsets and stay diverse.

use std::collections::HashMap;

use crate::algo::{Learner, SearchEvaluation, Step};
use crate::game::{Game, Rng};
use crate::planner::Planner;
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

/// Engine-level rollout knobs (everything that is not a game or *algorithm* parameter — the search
/// config + interior-target flag live on the `Planner`; the z-mix `outcome_weight` and bootstrap
/// masking live on the `Learner`).
pub struct EngineParams {
    pub n_games: usize,
    pub max_ticks: usize,
    pub epsilon: f64,
    pub n_heads: usize,
    pub seed: u64,
}

pub struct Engine<G: Game + Sync, P: Planner, L: Learner<SearchEvaluation>> {
    game: G,
    planner: P,
    learner: L,
    params: EngineParams,
    states: Vec<G::State>,
    rngs: Vec<SplitMix64>,
    // The search's chance-sampling stream — independent of the per-game env `rngs` so adding search
    // draws never perturbs the env's draw order (deterministic games stay bit-reproducible). A fresh
    // per-decision seed is drawn from it so each search samples with fresh randomness, like the env.
    search_rng: SplitMix64,
    heads: Vec<usize>, // per-game Thompson head for the current episode
    ticks: Vec<usize>,
    traj: Vec<Vec<Vec<Step<SearchEvaluation>>>>, // [game][agent] decisions awaiting episode-end records
}

impl<G: Game + Sync, P: Planner, L: Learner<SearchEvaluation>> Engine<G, P, L>
where
    G::State: Send,
{
    /// Build an engine over `game` driven by `planner` (search/evaluation) and `learner` (training
    /// records), with the rollout knobs from `params`. The game owns its reward and rules.
    pub fn new(game: G, planner: P, learner: L, params: EngineParams) -> Self {
        debug_assert!((1..=2).contains(&game.num_agents()));
        let n_heads = params.n_heads.max(1);
        let mut rngs: Vec<SplitMix64> = (0..params.n_games)
            .map(|i| {
                SplitMix64::new(
                    params
                        .seed
                        .wrapping_add((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)),
                )
            })
            .collect();
        let mut states: Vec<G::State> = Vec::with_capacity(params.n_games);
        let mut heads: Vec<usize> = Vec::with_capacity(params.n_games);
        for rng in rngs.iter_mut() {
            states.push(game.initial_state(rng));
            heads.push(rng.below(n_heads));
        }
        let ticks = vec![0; params.n_games];
        let num_agents = game.num_agents();
        let traj = (0..params.n_games)
            .map(|_| (0..num_agents).map(|_| Vec::new()).collect())
            .collect();
        let search_rng = SplitMix64::new(params.seed ^ 0xD1B5_4A32_D192_ED03);
        Engine {
            game,
            planner,
            learner,
            params,
            states,
            rngs,
            search_rng,
            heads,
            ticks,
            traj,
        }
    }

    fn agent_active(&self, state: &G::State, agent: usize) -> bool {
        !self.game.legal_actions(state, agent).is_empty()
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
        let action_count = self.game.action_count();
        let num_agents = self.game.num_agents();

        while out.len() < n_records {
            // 1. Gather one search request per active agent across all games.
            let mut requests: Vec<(G::State, usize)> = Vec::new();
            let mut meta: Vec<(usize, usize)> = Vec::new(); // (game index, agent index)
            for (gi, state) in self.states.iter().enumerate() {
                for si in 0..num_agents {
                    if self.agent_active(state, si) {
                        requests.push((state.clone(), si));
                        meta.push((gi, si));
                    }
                }
            }
            if requests.is_empty() {
                break; // every game dead this instant (resets below normally keep at least one alive)
            }

            // 2. The planner evaluates them all in one pooled pass (one batched forward per round,
            //    shared across games — the throughput win); for TreeStrap this is the selective search.
            let search_seed = self.search_rng.next_u64();
            let results = self
                .planner
                .evaluate(&self.game, requests, search_seed, &mut infer);

            // 3. Emit interior targets immediately, buffer each executed decision, choose its action.
            //    `acted[gi][si]` records the relative action index for an agent that decided this tick.
            let mut acted: Vec<Vec<Option<usize>>> =
                vec![vec![None; num_agents]; self.states.len()];
            for ((values, interior, search_stats), &(gi, si)) in
                results.into_iter().zip(meta.iter())
            {
                stats.decisions += 1;
                stats.max_depth = stats.max_depth.max(search_stats.max_depth);
                stats.sum_leaves += search_stats.leaves as f64;
                stats.sum_rounds += search_stats.rounds as f64;
                stats.sum_expansions += search_stats.expansions as f64;
                if search_stats.leaves > 0 {
                    stats.sum_sigma += search_stats.sigma_sum / search_stats.leaves as f64;
                }
                stats.sum_disagreement += root_disagreement(&values);
                // A search whose root children are all terminal evaluates no leaves, so the generic
                // search cannot infer the head count and returns a single (head-agnostic, terminal-
                // reward) row. Broadcast it to the configured `n_heads` so every emitted record's
                // target is `[n_heads][A]`. Searches that evaluated leaves already return `[n_heads][A]`,
                // so this is a no-op for them (e.g. snake).
                let nh = self.params.n_heads.max(1);
                let values: Vec<Vec<f64>> = if values.len() < nh {
                    vec![values[0].clone(); nh]
                } else {
                    values
                };
                // The learner emits its immediate records (TreeStrap interior nodes) now, draining them
                // out of the evaluation so they are never buffered for the whole episode.
                let mut eval = SearchEvaluation {
                    values,
                    interior,
                    stats: search_stats,
                };
                out.extend(self.learner.eval_records(&mut eval, &mut self.rngs[gi]));

                let k = eval.values.len();
                let head = self.heads[gi].min(k - 1);
                let mut rel = argmax(&eval.values[head]);
                if self.params.epsilon > 0.0 && self.rngs[gi].unit() < self.params.epsilon {
                    rel = self.rngs[gi].below(action_count);
                }
                acted[gi][si] = Some(rel);
                self.traj[gi][si].push(Step {
                    obs: self.game.observe(&self.states[gi], si),
                    evaluation: eval,
                    action: rel,
                    reward: 0.0, // filled in from this tick's transition after advancing
                });
            }

            // 4. Advance every game via the env transition (sampling its chance from the per-game RNG);
            //    record the executed decisions' rewards (plus the truncation bonus on a truncation
            //    tick); flush finished games' trajectories with z-mixing and reset them.
            let mut finished: Vec<(usize, bool)> = Vec::new(); // (game index, terminal?)
            for (gi, agents) in acted.into_iter().enumerate() {
                let joint: Vec<usize> = agents.iter().map(|a| a.unwrap_or(0)).collect();
                let transition = self
                    .game
                    .step_env(&self.states[gi], &joint, &mut self.rngs[gi]);
                let terminal = transition.terminal;
                self.states[gi] = transition.next_state;
                self.ticks[gi] += 1;
                // A truncation tick — max_ticks reached while the game is still playing — pays the
                // game's truncation bonus to agents that acted (snake's survival bonus to the living),
                // matching snake_RL's runner setting `survived_to_max_ticks` so it propagates through
                // z-mixing to earlier steps.
                let truncated = self.ticks[gi] >= self.params.max_ticks && !terminal;
                for (si, action) in agents.iter().enumerate() {
                    if action.is_some() {
                        let mut reward = transition.rewards[si];
                        if truncated {
                            reward += self.game.truncation_bonus(&self.states[gi], si);
                        }
                        // this agent acted this tick — attach the realized reward to its last decision
                        if let Some(step) = self.traj[gi][si].last_mut() {
                            step.reward = reward;
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
                let k = steps[0].evaluation.values.len();
                let tail = tails
                    .get(&(gi, si))
                    .cloned()
                    .unwrap_or_else(|| vec![0.0; k]);
                // The learner turns the buffered trajectory into records (TreeStrap z-mixing + masks).
                out.extend(
                    self.learner
                        .episode_records(&steps, &tail, &mut self.rngs[gi]),
                );
            }
            stats.episodes.push(EpisodeSummary {
                reward: ep_reward,
                length: self.ticks[gi],
            });
            self.states[gi] = self.game.initial_state(&mut self.rngs[gi]);
            self.ticks[gi] = 0;
            self.heads[gi] = self.rngs[gi].below(self.params.n_heads.max(1));
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
                if self.agent_active(&self.states[gi], si) && !self.traj[gi][si].is_empty() {
                    obs_flat.extend(self.game.observe(&self.states[gi], si));
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

/// Root head-disagreement: the per-action population std across heads of the root values `[K][A]`,
/// averaged over actions (`values.std(axis=0).mean()` in snake_RL). 0 with fewer than two heads.
fn root_disagreement(values: &[Vec<f64>]) -> f64 {
    let k = values.len();
    if k < 2 || values[0].is_empty() {
        return 0.0;
    }
    let a = values[0].len();
    let inv_k = 1.0 / k as f64;
    let total: f64 = (0..a)
        .map(|ai| {
            let mean = values.iter().map(|h| h[ai]).sum::<f64>() * inv_k;
            let var = values.iter().map(|h| (h[ai] - mean).powi(2)).sum::<f64>() * inv_k;
            var.sqrt()
        })
        .sum();
    total / a as f64
}

fn argmax(values: &[f64]) -> usize {
    let mut best = 0;
    for (i, &v) in values.iter().enumerate() {
        if v > values[best] {
            best = i;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn root_disagreement_matches_population_std_definition() {
        // Single action so the per-action std is the whole metric: heads [0, 2] -> mean 1, std 1.
        assert!((root_disagreement(&[vec![0.0], vec![2.0]]) - 1.0).abs() < 1e-12);
        // Identical heads disagree by zero; a single head has no spread.
        assert_eq!(root_disagreement(&[vec![5.0, 5.0], vec![5.0, 5.0]]), 0.0);
        assert_eq!(root_disagreement(&[vec![1.0, 2.0, 3.0]]), 0.0);
    }
}
