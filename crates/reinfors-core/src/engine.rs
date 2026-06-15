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

use std::collections::HashSet;

use crate::action::{relative_to_absolute, RELATIVE_ACTIONS};
use crate::obs::egocentric;
use crate::search::{selective_search_many, SearchParams};
use crate::snake::{empty_cells, Cell, Snake, SnakeEnv};

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
    pub seed: u64,
    pub search: SearchParams,
}

pub struct Engine {
    cfg: EngineConfig,
    games: Vec<SnakeEnv>,
    rngs: Vec<SplitMix64>,
    heads: Vec<usize>, // per-game Thompson head for the current episode
    ticks: Vec<usize>,
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
        Engine {
            cfg,
            games,
            rngs,
            heads,
            ticks,
        }
    }

    /// Roll the games forward until at least `n_records` decisions have been collected, returning the
    /// per-decision observations (each a flat `[5 * g * g]` buffer) and per-head searched targets
    /// (each `[K][A]`). `infer` is the value network forward, called once per pooled round.
    pub fn collect<F>(
        &mut self,
        n_records: usize,
        mut infer: F,
    ) -> (Vec<Vec<f32>>, Vec<Vec<Vec<f64>>>)
    where
        F: FnMut(&[Vec<f32>]) -> Vec<Vec<Vec<f64>>>,
    {
        let mut obs_out: Vec<Vec<f32>> = Vec::new();
        let mut tgt_out: Vec<Vec<Vec<f64>>> = Vec::new();

        while obs_out.len() < n_records {
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
            let results = selective_search_many(&self.cfg.search, requests, &mut infer);

            // 3. Record each decision and choose its action (Thompson head argmax + epsilon).
            let mut actions = vec![[None, None]; self.games.len()];
            for ((values, _stats), &(gi, si)) in results.iter().zip(meta.iter()) {
                obs_out.push(egocentric(&self.games[gi], si));
                tgt_out.push(values.clone());

                let head = self.heads[gi].min(values.len() - 1);
                let mut rel = argmax(&values[head]);
                if self.cfg.epsilon > 0.0 && self.rngs[gi].unit() < self.cfg.epsilon {
                    rel = self.rngs[gi].below(RELATIVE_ACTIONS.len());
                }
                let direction = self.games[gi].snakes[si].direction;
                actions[gi][si] = Some(relative_to_absolute(direction, RELATIVE_ACTIONS[rel]));
            }

            // 4. Advance every game (spawning a replacement apple per eaten one from the game's own
            //    RNG); reset finished ones and resample their Thompson head + initial food.
            for (gi, act) in actions.into_iter().enumerate() {
                let events = self.games[gi].advance(act, || None);
                for ev in events.iter() {
                    if ev.ate_food {
                        spawn_food(&mut self.games[gi], &mut self.rngs[gi], 1);
                    }
                }
                self.ticks[gi] += 1;
                if self.games[gi].done || self.ticks[gi] >= self.cfg.max_ticks {
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
        }
        (obs_out, tgt_out)
    }
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

    // Two disagreeing heads, sum-dependent.
    fn infer(obs: &[Vec<f32>]) -> Vec<Vec<Vec<f64>>> {
        obs.iter()
            .map(|o| {
                let s = o.iter().sum::<f32>() as f64;
                vec![
                    vec![s.sin(), s.cos(), (s * 0.5).sin()],
                    vec![(s + 1.0).sin(), (s * 0.3).cos(), (s * 0.2).sin()],
                ]
            })
            .collect()
    }

    #[test]
    fn collect_returns_well_formed_records() {
        let mut e = Engine::new(config(4, 2, 0));
        let (obs, tgt) = e.collect(50, infer);
        assert!(obs.len() >= 50 && obs.len() == tgt.len());
        for (o, t) in obs.iter().zip(tgt.iter()) {
            assert_eq!(o.len(), 5 * 12 * 12); // flat observation
            assert_eq!(t.len(), 2); // K heads
            assert!(t.iter().all(|row| row.len() == 3)); // A actions
        }
    }

    #[test]
    fn collect_is_deterministic_for_a_seed() {
        let (o1, t1) = Engine::new(config(4, 2, 7)).collect(60, infer);
        let (o2, t2) = Engine::new(config(4, 2, 7)).collect(60, infer);
        assert_eq!(o1, o2);
        assert_eq!(t1, t2);
    }

    #[test]
    fn distinct_seeds_diverge() {
        let (o1, _) = Engine::new(config(4, 2, 1)).collect(80, infer);
        let (o2, _) = Engine::new(config(4, 2, 2)).collect(80, infer);
        assert_ne!(o1, o2, "different seeds should produce different rollouts");
    }

    #[test]
    fn games_carry_food_so_snakes_can_eat() {
        // With initial_food_count > 0 the games start with apples; over a rollout some snake should
        // grow past its initial length (it ate), exercising the in-tree spawn + env respawn path.
        let mut e = Engine::new(config(6, 2, 3));
        e.collect(200, infer);
        let grew = e.games.iter().any(|g| g.snakes.iter().any(|s| s.len() > 3));
        assert!(grew, "no snake ever ate across the rollout");
        assert!(e.games.iter().all(|g| !g.food.is_empty()));
    }
}
