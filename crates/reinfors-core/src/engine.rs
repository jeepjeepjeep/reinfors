//! Parallel rollout collector — the data generator that turns the search core into training data.
//!
//! An `Engine` holds N independent games. Each `collect` step runs the pooled selective search for
//! every alive snake across every game in lockstep (one batched `infer` per round, shared across all
//! games — the throughput win), records each decision's `(observation, searched per-head values)` as
//! a TreeStrap target, picks an action by Thompson sampling (one head per game per episode) with an
//! epsilon-greedy override, advances every game, and resets finished ones.
//!
//! Per-game diversity comes from each game drawing its own Thompson head and epsilon noise, plus its
//! own RNG apple spawns (games start from the same deterministic placement, so without this they
//! would be identical). Each game's apples spawn uniformly over empty cells from its own RNG — the
//! true env model. (The search's in-tree spawn belief is the deterministic first-empty rule; see
//! `search`.)
//!
//! Records carry the full TreeStrap training semantics of `snake_RL`'s `EnsembleTreeStrapRunner`:
//! - **z-mixing** — each executed decision is held back until its episode ends, then the executed
//!   action's entry of every head is blended with the realized discounted return (`blend_outcome_targets`),
//!   so deaths the search failed to foresee still reach the training signal. A truncation tick reached
//!   alive pays the `survival` reward (`survived_to_max_ticks`), which z-mixing carries to earlier steps.
//! - **interior targets** — with `interior_targets`, every expanded interior MAX node of the search
//!   tree is emitted as an extra `(obs, values)` record (true TreeStrap). These are counterfactual
//!   states with no realized outcome, so they are emitted immediately and never z-blended.
//! - **bootstrap masks** — every record carries a per-head `(K,)` mask (`rng < bootstrap_p`) so the
//!   ensemble heads train on different subsets and stay diverse.

use std::collections::{HashMap, HashSet};

use crate::action::{relative_to_absolute, RELATIVE_ACTIONS};
use crate::obs::egocentric;
use crate::search::{selective_search_many, SearchParams};
use crate::snake::{empty_cells, Cell, Snake, SnakeEnv};

/// One buffered decision, held until its episode ends so the realized return is known for z-mixing.
struct TrajStep {
    obs: Vec<f32>,
    values: Vec<Vec<f64>>, // per-head searched action values [K][A]
    action: usize,         // executed relative-action index (after any epsilon override)
    reward: f64,
}

/// A collected training record: observation, per-head target `[K][A]`, and per-head bootstrap mask.
type Record = (Vec<f32>, Vec<Vec<f64>>, Vec<f32>);

/// One finished episode's outcome, for logging: per-snake total realized reward and the episode
/// length in ticks.
#[derive(Clone, Copy)]
pub struct EpisodeSummary {
    pub reward: [f64; 2],
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

/// One pooled-search request: a game state (snakes, food) and the agent searching it.
type Request = ([Snake; 2], HashSet<Cell>, usize);

/// Tiny deterministic PRNG (splitmix64) for Thompson-head and epsilon draws — keeps rollouts
/// reproducible from a seed without pulling in an RNG dependency.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        SplitMix64 { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

pub struct EngineConfig {
    pub n_games: usize,
    pub grid_size: i32,
    pub initial_length: usize,
    pub play_to_last: bool,
    pub win_food_lead: Option<usize>,
    pub initial_food_count: usize,
    pub max_ticks: usize,
    pub epsilon: f64,
    pub n_heads: usize,
    pub outcome_weight: f64,
    pub interior_targets: bool,
    pub bootstrap_p: f64,
    pub seed: u64,
    pub search: SearchParams,
}

pub struct Engine {
    cfg: EngineConfig,
    games: Vec<SnakeEnv>,
    rngs: Vec<SplitMix64>,
    heads: Vec<usize>, // per-game Thompson head for the current episode
    ticks: Vec<usize>,
    traj: Vec<[Vec<TrajStep>; 2]>, // per-game, per-snake decisions awaiting episode-end z-mixing
}

impl Engine {
    pub fn new(cfg: EngineConfig) -> Self {
        let n_heads = cfg.n_heads.max(1);
        let mut rngs: Vec<SplitMix64> = (0..cfg.n_games)
            .map(|i| {
                SplitMix64::new(
                    cfg.seed
                        .wrapping_add((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)),
                )
            })
            .collect();
        let mut games: Vec<SnakeEnv> = Vec::with_capacity(cfg.n_games);
        let mut heads: Vec<usize> = Vec::with_capacity(cfg.n_games);
        for rng in rngs.iter_mut() {
            let mut game = SnakeEnv::new(
                cfg.grid_size,
                cfg.initial_length,
                cfg.play_to_last,
                cfg.win_food_lead,
            );
            spawn_food(&mut game, rng, cfg.initial_food_count);
            heads.push(rng.below(n_heads));
            games.push(game);
        }
        let ticks = vec![0; cfg.n_games];
        let traj = (0..cfg.n_games).map(|_| [Vec::new(), Vec::new()]).collect();
        Engine {
            cfg,
            games,
            rngs,
            heads,
            ticks,
            traj,
        }
    }

    /// Roll the games forward until at least `n_records` training records have been collected,
    /// returning each record's observation (a flat `[5 * g * g]` buffer), per-head target `[K][A]`,
    /// and per-head bootstrap mask `[K]`. Executed decisions are z-mixed at episode end; interior MAX
    /// nodes (when enabled) are emitted immediately. `infer` is the value-network forward.
    pub fn collect<F>(&mut self, n_records: usize, mut infer: F) -> (Vec<Record>, CollectStats)
    where
        F: FnMut(Vec<f32>, usize) -> Vec<f64>,
    {
        // (The infer head-count check lives in the Python binding, where it can distinguish a real
        // wrong-K output from the error fallback and surface a clean error; here we only consume the
        // values, with the per-game clamp handling the all-terminal single-head case.)
        let mut out: Vec<Record> = Vec::new();
        let mut stats = CollectStats::default();

        while out.len() < n_records {
            // 1. Gather one search request per alive snake across all games.
            let mut requests: Vec<Request> = Vec::new();
            let mut meta: Vec<(usize, usize)> = Vec::new(); // (game index, snake index)
            for (gi, game) in self.games.iter().enumerate() {
                for si in 0..2 {
                    if game.snakes[si].alive {
                        requests.push((game.snakes.clone(), game.food.clone(), si));
                        meta.push((gi, si));
                    }
                }
            }
            if requests.is_empty() {
                break; // every game dead this instant (resets below normally keep at least one alive)
            }

            // 2. One pooled search for all of them (one batched forward per round, shared across games).
            let results = selective_search_many(
                &self.cfg.search,
                requests,
                self.cfg.interior_targets,
                &mut infer,
            );

            // 3. Emit interior targets immediately, buffer each executed decision, choose its action.
            let mut actions = vec![[None, None]; self.games.len()];
            for ((values, interior, search_stats), &(gi, si)) in results.iter().zip(meta.iter()) {
                stats.decisions += 1;
                stats.max_depth = stats.max_depth.max(search_stats.max_depth);
                stats.sum_leaves += search_stats.leaves as f64;
                stats.sum_rounds += search_stats.rounds as f64;
                stats.sum_expansions += search_stats.expansions as f64;
                if search_stats.leaves > 0 {
                    stats.sum_sigma += search_stats.sigma_sum / search_stats.leaves as f64;
                }
                stats.sum_disagreement += root_disagreement(values);
                let k = values.len();
                for (iobs, ivalues) in interior {
                    let mask = sample_mask(&mut self.rngs[gi], k, self.cfg.bootstrap_p);
                    out.push((iobs.clone(), ivalues.clone(), mask));
                }

                let head = self.heads[gi].min(k - 1);
                let mut rel = argmax(&values[head]);
                if self.cfg.epsilon > 0.0 && self.rngs[gi].unit() < self.cfg.epsilon {
                    rel = self.rngs[gi].below(RELATIVE_ACTIONS.len());
                }
                let direction = self.games[gi].snakes[si].direction;
                actions[gi][si] = Some(relative_to_absolute(direction, RELATIVE_ACTIONS[rel]));
                self.traj[gi][si].push(TrajStep {
                    obs: egocentric(&self.games[gi], si),
                    values: values.clone(),
                    action: rel,
                    reward: 0.0, // filled in from this tick's event after advancing
                });
            }

            // 4. Advance every game (spawning a replacement apple per eaten one); record the executed
            //    decisions' rewards; flush finished games' trajectories with z-mixing and reset them.
            let mut finished: Vec<usize> = Vec::new();
            for (gi, act) in actions.into_iter().enumerate() {
                let mut events = self.games[gi].advance(act, || None);
                for ev in events.iter() {
                    if ev.ate_food {
                        spawn_food(&mut self.games[gi], &mut self.rngs[gi], 1);
                    }
                }
                self.ticks[gi] += 1;
                // A truncation tick — max_ticks reached while the game is still playing — pays the
                // survival bonus to snakes that did not die this tick, matching snake_RL's runner
                // setting `survived_to_max_ticks` (so it propagates through z-mixing to earlier steps).
                let truncated = self.ticks[gi] >= self.cfg.max_ticks && !self.games[gi].done;
                for (si, ev) in events.iter_mut().enumerate() {
                    if act[si].is_some() {
                        if truncated && !ev.died {
                            ev.survived_to_max_ticks = true;
                        }
                        // this snake acted this tick — attach the realized reward to its last decision
                        if let Some(step) = self.traj[gi][si].last_mut() {
                            step.reward = self.cfg.search.reward.eval(ev);
                        }
                    }
                }
                if self.games[gi].done || self.ticks[gi] >= self.cfg.max_ticks {
                    finished.push(gi);
                }
            }

            self.flush_finished(&finished, &mut out, &mut stats, &mut infer);
        }
        (out, stats)
    }

    /// Flush each finished game's buffered trajectories: z-mix the executed-action targets with the
    /// realized return (tail-seeded by the net's value of the final state on truncation), emit the
    /// records, then reset the game and resample its head + initial food.
    fn flush_finished<F>(
        &mut self,
        finished: &[usize],
        out: &mut Vec<Record>,
        stats: &mut CollectStats,
        infer: &mut F,
    ) where
        F: FnMut(Vec<f32>, usize) -> Vec<f64>,
    {
        // On truncation (alive, not terminal) z is seeded by the net's per-head state value, so early
        // decisions of long episodes are not systematically undervalued. Batch those into one forward.
        let tails = self.tail_values(finished, infer);

        for &gi in finished {
            let mut ep_reward = [0.0; 2];
            for (si, ep_slot) in ep_reward.iter_mut().enumerate() {
                let steps = std::mem::take(&mut self.traj[gi][si]);
                if steps.is_empty() {
                    continue;
                }
                *ep_slot = steps.iter().map(|s| s.reward).sum();
                let k = steps[0].values.len();
                let tail = tails
                    .get(&(gi, si))
                    .cloned()
                    .unwrap_or_else(|| vec![0.0; k]);
                let traj: Vec<(Vec<Vec<f64>>, usize, f64)> = steps
                    .iter()
                    .map(|s| (s.values.clone(), s.action, s.reward))
                    .collect();
                let blended = blend_outcome_targets(
                    &traj,
                    self.cfg.search.gamma,
                    self.cfg.outcome_weight,
                    &tail,
                );
                for (step, target) in steps.iter().zip(blended) {
                    let mask = sample_mask(&mut self.rngs[gi], k, self.cfg.bootstrap_p);
                    out.push((step.obs.clone(), target, mask));
                }
            }
            stats.episodes.push(EpisodeSummary {
                reward: ep_reward,
                length: self.ticks[gi],
            });
            let mut game = SnakeEnv::new(
                self.cfg.grid_size,
                self.cfg.initial_length,
                self.cfg.play_to_last,
                self.cfg.win_food_lead,
            );
            spawn_food(&mut game, &mut self.rngs[gi], self.cfg.initial_food_count);
            self.games[gi] = game;
            self.ticks[gi] = 0;
            self.heads[gi] = self.rngs[gi].below(self.cfg.n_heads.max(1));
        }
    }

    /// Per-(game, snake) z-tail: the net's per-head value `max_a Q(final_obs)` for snakes still alive
    /// at a truncation (terminal episodes and dead snakes seed with 0, so they are absent here). Empty
    /// when `outcome_weight == 0`, where the tail never enters the (no-op) blend.
    fn tail_values<F>(
        &mut self,
        finished: &[usize],
        infer: &mut F,
    ) -> HashMap<(usize, usize), Vec<f64>>
    where
        F: FnMut(Vec<f32>, usize) -> Vec<f64>,
    {
        let mut tails: HashMap<(usize, usize), Vec<f64>> = HashMap::new();
        if self.cfg.outcome_weight <= 0.0 {
            return tails;
        }
        let mut obs_flat: Vec<f32> = Vec::new();
        let mut meta: Vec<(usize, usize)> = Vec::new();
        for &gi in finished {
            if self.games[gi].done {
                continue; // terminal: tail stays 0
            }
            for si in 0..2 {
                if self.games[gi].snakes[si].alive && !self.traj[gi][si].is_empty() {
                    obs_flat.extend(egocentric(&self.games[gi], si));
                    meta.push((gi, si));
                }
            }
        }
        if !meta.is_empty() {
            let a = RELATIVE_ACTIONS.len();
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

/// AlphaGo-style z-mixing: blend each step's realized discounted return-to-go into the executed
/// action's entry of every head, `(1 - w) * V + w * z`. `trajectory` is time-ordered
/// `(searched values [K][A], executed action, reward)`; `tail` (len K) seeds z past the last step
/// (0 at a terminal, the net's per-head state value at a truncation). Unexecuted entries keep their
/// pure searched value. Returns the per-step blended `[K][A]` targets in time order.
pub fn blend_outcome_targets(
    trajectory: &[(Vec<Vec<f64>>, usize, f64)],
    gamma: f64,
    outcome_weight: f64,
    tail: &[f64],
) -> Vec<Vec<Vec<f64>>> {
    let mut z: Vec<f64> = tail.to_vec();
    let mut out: Vec<Vec<Vec<f64>>> = Vec::with_capacity(trajectory.len());
    for (values, action, reward) in trajectory.iter().rev() {
        for zi in z.iter_mut() {
            *zi = reward + gamma * *zi;
        }
        let mut blended = values.clone();
        if outcome_weight > 0.0 {
            for (h, row) in blended.iter_mut().enumerate() {
                row[*action] = (1.0 - outcome_weight) * row[*action] + outcome_weight * z[h];
            }
        }
        out.push(blended);
    }
    out.reverse();
    out
}

/// A per-head bootstrap mask `[K]`: head `h` trains on this record iff `rng.unit() < p` (Osband et
/// al. 2016). `p = 1` includes every head (the masks are all-ones).
fn sample_mask(rng: &mut SplitMix64, n_heads: usize, p: f64) -> Vec<f32> {
    (0..n_heads)
        .map(|_| if rng.unit() < p { 1.0 } else { 0.0 })
        .collect()
}

/// Spawn `n` apples into `game`, each uniformly over the currently empty cells drawn from `rng` (the
/// env's true spawn model). Stops early if the grid fills.
fn spawn_food(game: &mut SnakeEnv, rng: &mut SplitMix64, n: usize) {
    for _ in 0..n {
        let empty = empty_cells(&game.snakes, &game.food, game.grid_size);
        if empty.is_empty() {
            break;
        }
        game.food.insert(empty[rng.below(empty.len())]);
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
    use crate::reward::Reward;
    use crate::search::Opponent;

    fn config(n_games: usize, n_heads: usize, seed: u64) -> EngineConfig {
        EngineConfig {
            n_games,
            grid_size: 12,
            initial_length: 3,
            play_to_last: false,
            win_food_lead: None,
            initial_food_count: 3,
            max_ticks: 50,
            epsilon: 0.1,
            n_heads,
            outcome_weight: 0.5,
            interior_targets: true,
            bootstrap_p: 0.8,
            seed,
            search: SearchParams {
                grid_size: 12,
                initial_length: 3,
                play_to_last: false,
                win_food_lead: None,
                gamma: 0.99,
                beta: 1.0,
                expansion_budget: 24,
                top_k: 4,
                max_depth: 6,
                food_samples: 1,
                reward: Reward {
                    step: 0.0,
                    food: 0.0,
                    loss: -10.0,
                    draw: -6.0,
                    kill: 20.0,
                    win: 20.0,
                    survival: 0.0,
                },
                opponent: Opponent::Uniform,
            },
        }
    }

    // Two disagreeing heads, sum-dependent — flat `(obs[n*dim], n) -> values[n*2*3]` (head-major).
    fn infer(obs: Vec<f32>, n: usize) -> Vec<f64> {
        let dim = obs.len() / n;
        let mut out = Vec::with_capacity(n * 2 * 3);
        for i in 0..n {
            let s = obs[i * dim..(i + 1) * dim].iter().sum::<f32>() as f64;
            out.extend_from_slice(&[
                s.sin(),
                s.cos(),
                (s * 0.5).sin(),
                (s + 1.0).sin(),
                (s * 0.3).cos(),
                (s * 0.2).sin(),
            ]);
        }
        out
    }

    #[test]
    fn collect_returns_well_formed_records() {
        let mut e = Engine::new(config(4, 2, 0));
        let (records, _stats) = e.collect(50, infer);
        assert!(records.len() >= 50);
        for (obs, tgt, mask) in &records {
            assert_eq!(obs.len(), 5 * 12 * 12); // flat observation
            assert_eq!(tgt.len(), 2); // K heads
            assert!(tgt.iter().all(|row| row.len() == 3)); // A actions
            assert_eq!(mask.len(), 2); // per-head bootstrap mask
            assert!(mask.iter().all(|&m| m == 0.0 || m == 1.0));
        }
    }

    #[test]
    fn collect_is_deterministic_for_a_seed() {
        let r1 = Engine::new(config(4, 2, 7)).collect(60, infer).0;
        let r2 = Engine::new(config(4, 2, 7)).collect(60, infer).0;
        assert_eq!(r1, r2);
    }

    #[test]
    fn distinct_seeds_diverge() {
        let r1 = Engine::new(config(4, 2, 1)).collect(80, infer).0;
        let r2 = Engine::new(config(4, 2, 2)).collect(80, infer).0;
        assert_ne!(r1, r2, "different seeds should produce different rollouts");
    }

    #[test]
    fn games_carry_food_so_snakes_can_eat() {
        // Over a long rollout some snake should grow past its initial length (it ate), exercising the
        // in-tree spawn + env respawn path. Interior off so the record floor tracks decisions (with it
        // on, the floor is reached in far fewer ticks). The apple count is invariant: eating discards
        // one and respawns one, so every game always holds initial_food_count.
        let mut cfg = config(8, 2, 3);
        cfg.interior_targets = false;
        let mut e = Engine::new(cfg);
        let mut grew = false;
        for _ in 0..4 {
            e.collect(300, infer);
            grew |= e.games.iter().any(|g| g.snakes.iter().any(|s| s.len() > 3));
        }
        assert!(grew, "no snake ever ate across the rollout");
        assert!(e.games.iter().all(|g| g.food.len() == 3));
    }

    #[test]
    fn bootstrap_p_extremes_set_all_or_no_heads() {
        let mut all = config(4, 2, 5); // n_heads matches `infer`'s 2 heads
        all.bootstrap_p = 1.0;
        for (_, _, mask) in Engine::new(all).collect(40, infer).0 {
            assert!(
                mask.iter().all(|&m| m == 1.0),
                "p=1 must include every head"
            );
        }
        let mut none = config(4, 2, 5);
        none.bootstrap_p = 0.0;
        for (_, _, mask) in Engine::new(none).collect(40, infer).0 {
            assert!(mask.iter().all(|&m| m == 0.0), "p=0 must include no head");
        }
    }

    #[test]
    fn zero_outcome_weight_leaves_targets_unblended() {
        // With outcome_weight = 0 the z-mix is a no-op, so a record's target equals its raw searched
        // values; with weight > 0 some executed-action entry must differ. We can't read the search
        // values here, but determinism lets us assert the two configs diverge.
        let mut w0 = config(4, 2, 9);
        w0.outcome_weight = 0.0;
        w0.interior_targets = false;
        let mut w1 = config(4, 2, 9);
        w1.outcome_weight = 0.9;
        w1.interior_targets = false;
        let r0 = Engine::new(w0).collect(60, infer).0;
        let r1 = Engine::new(w1).collect(60, infer).0;
        let targets_differ = r0.iter().zip(&r1).any(|((_, t0, _), (_, t1, _))| t0 != t1);
        assert!(
            targets_differ,
            "outcome_weight should change executed-action targets"
        );
    }

    #[test]
    fn blend_outcome_targets_mixes_only_the_executed_action() {
        // Two heads, three actions, action 1 executed; one step, terminal tail (z = reward).
        let values = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let traj = vec![(values.clone(), 1usize, 10.0)];
        let blended = blend_outcome_targets(&traj, 0.9, 0.25, &[0.0, 0.0]);
        // z = 10 + 0.9*0 = 10; executed entry -> 0.75*v + 0.25*10.
        assert!((blended[0][0][1] - (0.75 * 2.0 + 0.25 * 10.0)).abs() < 1e-12);
        assert!((blended[0][1][1] - (0.75 * 5.0 + 0.25 * 10.0)).abs() < 1e-12);
        // unexecuted entries unchanged.
        assert_eq!(blended[0][0][0], 1.0);
        assert_eq!(blended[0][1][2], 6.0);
    }

    #[test]
    fn survival_bonus_propagates_through_z_mixing_on_truncation() {
        // max_ticks = 1: every episode truncates after one (surviving) decision. With outcome_weight
        // = 1 the executed action's target equals the realized return, which on a truncation includes
        // the survival bonus. Two engines identical but for `survival` must differ in their targets by
        // exactly the bonus, and only in the executed action's entry — survival touches neither the
        // search values, the chosen action, nor the z-tail.
        let bonus = 0.25;
        let mk = |survival: f64| {
            let mut c = config(4, 2, 0);
            c.max_ticks = 1;
            c.outcome_weight = 1.0;
            c.interior_targets = false;
            c.initial_food_count = 0;
            c.search.reward.survival = survival;
            c
        };
        let base = Engine::new(mk(0.0)).collect(4, infer).0;
        let surv = Engine::new(mk(bonus)).collect(4, infer).0;
        assert_eq!(base.len(), surv.len());
        assert!(!base.is_empty());
        for ((_, tb, _), (_, ts, _)) in base.iter().zip(surv.iter()) {
            for (rb, rs) in tb.iter().zip(ts.iter()) {
                let changed: Vec<usize> = (0..rb.len())
                    .filter(|&a| (rs[a] - rb[a]).abs() > 1e-9)
                    .collect();
                assert_eq!(
                    changed.len(),
                    1,
                    "only the executed action's target should move"
                );
                assert!((rs[changed[0]] - rb[changed[0]] - bonus).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn collect_reports_episode_and_search_telemetry() {
        // A long enough rollout finishes several episodes and runs many searches; the telemetry must
        // be populated and internally consistent (means finite, lengths bounded by max_ticks).
        // Interior off so the record floor tracks decisions (with it on, the floor is reached via
        // interior targets before any episode completes).
        let mut cfg = config(4, 2, 11);
        cfg.interior_targets = false;
        let max_ticks = cfg.max_ticks;
        let mut e = Engine::new(cfg);
        let mut episodes = 0usize;
        let (mut decisions, mut max_depth, mut leaves, mut sigma, mut disagree) =
            (0usize, 0i32, 0.0, 0.0, 0.0);
        for _ in 0..4 {
            let (_records, stats) = e.collect(400, infer);
            for ep in &stats.episodes {
                assert!(
                    ep.length >= 1 && ep.length <= max_ticks,
                    "length {}",
                    ep.length
                );
                assert!(ep.reward.iter().all(|r| r.is_finite()));
            }
            episodes += stats.episodes.len();
            decisions += stats.decisions;
            max_depth = max_depth.max(stats.max_depth);
            leaves += stats.sum_leaves;
            sigma += stats.sum_sigma;
            disagree += stats.sum_disagreement;
        }
        assert!(decisions > 0, "no searches counted");
        assert!(max_depth > 0, "search reached no depth");
        assert!(leaves > 0.0, "no leaves expanded");
        assert!(episodes > 0, "no episodes finished");
        let mean_sigma = sigma / decisions as f64;
        let mean_disagreement = disagree / decisions as f64;
        assert!(mean_sigma.is_finite() && mean_sigma >= 0.0);
        assert!(mean_disagreement.is_finite() && mean_disagreement >= 0.0);
    }

    #[test]
    fn telemetry_is_deterministic_for_a_seed() {
        let stats1 = Engine::new(config(4, 2, 13)).collect(200, infer).1;
        let stats2 = Engine::new(config(4, 2, 13)).collect(200, infer).1;
        assert_eq!(stats1.decisions, stats2.decisions);
        assert_eq!(stats1.episodes.len(), stats2.episodes.len());
        for (a, b) in stats1.episodes.iter().zip(stats2.episodes.iter()) {
            assert_eq!(a.reward, b.reward);
            assert_eq!(a.length, b.length);
        }
    }

    #[test]
    fn root_disagreement_matches_population_std_definition() {
        // Single action so the per-action std is the whole metric: heads [0, 2] -> mean 1, std 1.
        assert!((root_disagreement(&[vec![0.0], vec![2.0]]) - 1.0).abs() < 1e-12);
        // Identical heads disagree by zero; a single head has no spread.
        assert_eq!(root_disagreement(&[vec![5.0, 5.0], vec![5.0, 5.0]]), 0.0);
        assert_eq!(root_disagreement(&[vec![1.0, 2.0, 3.0]]), 0.0);
    }
}
