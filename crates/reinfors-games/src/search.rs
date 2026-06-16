//! Snake wrappers over the generic [`reinfors_core::search::search_many`]: `SearchParams` bundles the
//! snake config + the game-agnostic search knobs, and `selective_search`/`selective_search_many` build
//! a `SnakeGame` + `SnakeState` from it and run the generic engine. The public API is unchanged.

use std::collections::HashSet;

use reinfors_core::search::{search_many, Opponent, SearchConfig, SearchResult};

use crate::reward::Reward;
use crate::snake::{Cell, Snake};
use crate::snake_game::{SnakeGame, SnakeState};

pub struct SearchParams {
    pub grid_size: i32,
    pub initial_length: usize,
    pub play_to_last: bool,
    pub win_food_lead: Option<usize>,
    pub gamma: f64,
    pub beta: f64,
    pub expansion_budget: usize,
    pub top_k: usize,
    pub max_depth: i32,
    /// Monte-Carlo apple-spawn samples per eaten-apple branch (>= 1). With the deterministic
    /// first-empty spawn belief the samples are identical, so this is the fan-out structure a
    /// stochastic spawn would populate; 1 disables it.
    pub food_samples: usize,
    pub reward: Reward,
    pub opponent: Opponent,
}

/// Build a `SnakeGame` + `SearchConfig` from `SearchParams`. The snake-specific config splits onto
/// the game; the search keeps the game-agnostic knobs.
fn snake_game_and_config(p: &SearchParams) -> (SnakeGame, SearchConfig) {
    let game = SnakeGame {
        grid_size: p.grid_size,
        initial_length: p.initial_length,
        play_to_last: p.play_to_last,
        win_food_lead: p.win_food_lead,
        initial_food_count: 0, // unused on the search path (chance_outcomes derives eaten from the food drop)
        reward: p.reward,
    };
    let cfg = SearchConfig {
        gamma: p.gamma,
        beta: p.beta,
        expansion_budget: p.expansion_budget,
        top_k: p.top_k,
        max_depth: p.max_depth,
        food_samples: p.food_samples,
        opponent: p.opponent,
    };
    (game, cfg)
}

/// Snake wrapper over the generic [`search_many`]: maps each `([Snake;2], HashSet<Cell>)` request to a
/// `SnakeState` and runs the generic engine. The public API is unchanged.
pub fn selective_search_many<F>(
    p: &SearchParams,
    requests: Vec<([Snake; 2], HashSet<Cell>, usize)>,
    collect_interior: bool,
    infer: F,
) -> Vec<SearchResult>
where
    F: FnMut(Vec<f32>, usize) -> Vec<f64>,
{
    let (game, cfg) = snake_game_and_config(p);
    let requests: Vec<(SnakeState, usize)> = requests
        .into_iter()
        .map(|(snakes, food, agent)| (SnakeState { snakes, food }, agent))
        .collect();
    search_many(&game, &cfg, requests, collect_interior, infer)
}

/// Single-request convenience wrapper over [`selective_search_many`].
pub fn selective_search<F>(
    p: &SearchParams,
    snakes: [Snake; 2],
    food: HashSet<Cell>,
    agent: usize,
    collect_interior: bool,
    infer: F,
) -> SearchResult
where
    F: FnMut(Vec<f32>, usize) -> Vec<f64>,
{
    selective_search_many(p, vec![(snakes, food, agent)], collect_interior, infer)
        .pop()
        .unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::action::Action;
    use crate::snake::Snake;
    use reinfors_core::search::search_many;

    fn snake(cells: &[Cell], dir: Action) -> Snake {
        Snake {
            body: cells.iter().copied().collect(),
            direction: dir,
            alive: true,
        }
    }

    fn params() -> SearchParams {
        SearchParams {
            grid_size: 12,
            initial_length: 3,
            play_to_last: true,
            win_food_lead: None,
            gamma: 0.99,
            beta: 1.0,
            expansion_budget: 30,
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
        }
    }

    #[test]
    fn fatal_actions_score_the_loss_and_survivable_turns_score_higher() {
        // A in the top-left corner heading Left: Forward (Left) and Right (Up) both run off-grid;
        // only Left (Down) survives. With a zero value function the only signal is the death penalty.
        let snakes = [
            snake(&[(0, 0), (0, 1), (0, 2)], Action::Left),
            snake(&[(6, 6), (6, 7), (6, 8)], Action::Left), // opponent, far away
        ];
        let p = params();
        let (values, _interior, stats) =
            selective_search(&p, snakes, HashSet::new(), 0, false, |_obs, n| {
                vec![0.0; n * 3]
            });
        let v = &values[0]; // single head
        assert!(
            (v[0] - (-10.0)).abs() < 1e-9,
            "fatal Forward should score the loss: {v:?}"
        );
        assert!(
            (v[2] - (-10.0)).abs() < 1e-9,
            "fatal Right should score the loss: {v:?}"
        );
        assert!(
            v[1] > v[0] && v[1] > v[2],
            "survivable Left should win: {v:?}"
        );
        assert!(stats.expansions > 0 && stats.rounds > 0);
    }

    #[test]
    fn ensemble_bootstrap_sigma_drives_voi_priority() {
        // Two heads that disagree -> nonzero sigma -> beta=1 priority can steer on it (smoke check
        // that multi-head search runs and produces per-head root values).
        let snakes = [
            snake(&[(6, 5), (6, 4), (6, 3)], Action::Right),
            snake(&[(2, 8), (2, 9), (1, 9)], Action::Left),
        ];
        let mut p = params();
        p.expansion_budget = 24;
        let (values, _interior, stats) =
            selective_search(&p, snakes, HashSet::new(), 0, false, two_head_infer);
        assert_eq!(values.len(), 2); // two heads
        assert_eq!(values[0].len(), 3);
        assert!(stats.expansions > 0);
    }

    // Two disagreeing heads, sum-dependent — exercises sigma + the VOI priority under pooling. Flat
    // `(obs[n*dim], n) -> values[n*2*3]` (head-major rows), matching the new infer interface.
    fn two_head_infer(obs: Vec<f32>, n: usize) -> Vec<f64> {
        let dim = obs.len() / n;
        let mut out = Vec::with_capacity(n * 2 * 3);
        for i in 0..n {
            let s = obs[i * dim..(i + 1) * dim].iter().sum::<f32>() as f64;
            out.extend_from_slice(&[
                s.sin(),
                s.cos(),
                (s * 0.5).sin(), // head 0
                (s + 1.0).sin(),
                (s * 0.3).cos(),
                (s * 0.2).sin(), // head 1
            ]);
        }
        out
    }

    type Request = ([Snake; 2], HashSet<Cell>, usize);

    fn two_requests() -> (Request, Request) {
        let a = (
            [
                snake(&[(6, 5), (6, 4), (6, 3)], Action::Right),
                snake(&[(2, 8), (2, 9), (1, 9)], Action::Left),
            ],
            HashSet::new(),
            0usize,
        );
        let b = (
            [
                snake(&[(3, 3), (3, 2), (3, 1)], Action::Right),
                snake(&[(8, 8), (8, 9), (9, 9)], Action::Left),
            ],
            HashSet::new(),
            1usize,
        );
        (a, b)
    }

    #[test]
    fn pooling_matches_solo_searches_bit_for_bit() {
        let (a, b) = two_requests();
        let mut p = params();
        p.expansion_budget = 24;
        let many = selective_search_many(&p, vec![a.clone(), b.clone()], false, two_head_infer);
        let solo_a = selective_search(&p, a.0.clone(), a.1.clone(), a.2, false, two_head_infer);
        let solo_b = selective_search(&p, b.0.clone(), b.1.clone(), b.2, false, two_head_infer);
        assert_eq!(
            many[0].0, solo_a.0,
            "pooled values must equal the solo search"
        );
        assert_eq!(many[1].0, solo_b.0);
        assert_eq!(many[0].2.expansions, solo_a.2.expansions);
        assert_eq!(many[1].2.expansions, solo_b.2.expansions);
        assert_eq!(many[0].2.rounds, solo_a.2.rounds);
    }

    #[test]
    fn parallel_search_is_thread_count_independent() {
        // The rayon-parallel per-search work is value-neutral: running the same pooled search inside a
        // 1-thread pool and a 4-thread pool must give bit-identical values, interior, and stats.
        let (a, b) = two_requests();
        let mut p = params();
        p.expansion_budget = 24;
        let run = |threads: usize| {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            pool.install(|| {
                selective_search_many(&p, vec![a.clone(), b.clone()], true, two_head_infer)
            })
        };
        let one = run(1);
        let many = run(4);
        for i in 0..2 {
            assert_eq!(
                one[i].0, many[i].0,
                "values must not depend on thread count"
            );
            assert_eq!(
                one[i].1, many[i].1,
                "interior targets must not depend on thread count"
            );
            assert_eq!(
                (
                    one[i].2.max_depth,
                    one[i].2.expansions,
                    one[i].2.leaves,
                    one[i].2.rounds
                ),
                (
                    many[i].2.max_depth,
                    many[i].2.expansions,
                    many[i].2.leaves,
                    many[i].2.rounds
                ),
            );
        }
    }

    #[test]
    fn pooling_issues_fewer_forwards_than_solo() {
        use std::cell::Cell as Counter;
        let (a, b) = two_requests();
        let mut p = params();
        p.expansion_budget = 24;

        let pooled = Counter::new(0usize);
        selective_search_many(&p, vec![a.clone(), b.clone()], false, |o, n| {
            pooled.set(pooled.get() + 1);
            two_head_infer(o, n)
        });
        let solo = Counter::new(0usize);
        selective_search(&p, a.0.clone(), a.1.clone(), a.2, false, |o, n| {
            solo.set(solo.get() + 1);
            two_head_infer(o, n)
        });
        selective_search(&p, b.0.clone(), b.1.clone(), b.2, false, |o, n| {
            solo.set(solo.get() + 1);
            two_head_infer(o, n)
        });
        assert!(
            pooled.get() < solo.get(),
            "pooled forwards {} should be fewer than solo {}",
            pooled.get(),
            solo.get()
        );
    }

    #[test]
    fn all_terminal_root_returns_single_head_without_calling_infer() {
        // Agent boxed in heading Left at the top-left: Forward (Left) and Right (Up) hit the wall, and
        // Left (Down) moves onto its own neck (self-collision). Every root child is terminal, so the
        // round produces no observations -> infer is never called and n_heads falls back to 1.
        let snakes = [
            snake(&[(0, 0), (1, 0), (2, 0)], Action::Left),
            snake(&[(5, 5), (5, 6), (5, 7)], Action::Left),
        ];
        let p = params(); // uniform opponent, loss = -10
        let mut calls = 0usize;
        let results =
            selective_search_many(&p, vec![(snakes, HashSet::new(), 0)], false, |_obs, n| {
                calls += 1;
                vec![0.0; n * 3]
            });
        let (values, _interior, stats) = &results[0];
        assert_eq!(
            calls, 0,
            "no observations this round -> infer must not be called"
        );
        assert_eq!(
            values.len(),
            1,
            "n_heads falls back to 1 when nothing was evaluated"
        );
        for v in &values[0] {
            assert!(
                (v - (-10.0)).abs() < 1e-9,
                "every action is fatal -> the loss: {values:?}"
            );
        }
        assert_eq!(stats.leaves, 0);
        assert_eq!(stats.expansions, 1);
    }

    #[test]
    fn food_samples_fans_out_only_eating_branches() {
        // Agent mid-grid heading Right with an apple directly ahead; opponent dead (one opp branch per
        // edge). In a single root expansion only Forward eats, so food_samples=3 turns its one child
        // into three while Left/Right keep one each: 3 leaves at k=1 -> 5 at k=3.
        let snakes = [
            snake(&[(6, 5), (6, 4), (6, 3)], Action::Right),
            Snake {
                body: [(0, 0), (1, 0)].into_iter().collect(),
                direction: Action::Down,
                alive: false,
            },
        ];
        let food: HashSet<Cell> = [(6, 6)].into_iter().collect();
        let infer = |_o: Vec<f32>, n: usize| vec![0.0; n * 3]; // single head, zero values
        let leaves = |samples: usize| {
            let mut p = params();
            p.expansion_budget = 1; // one expansion: just the root
            p.food_samples = samples;
            selective_search(&p, snakes.clone(), food.clone(), 0, false, infer)
                .2
                .leaves
        };
        assert_eq!(leaves(1), 3);
        assert_eq!(
            leaves(3),
            5,
            "the eating (Forward) branch fans 1 -> 3; the others are unchanged"
        );
    }

    #[test]
    fn generic_search_many_matches_snake_wrapper() {
        // The generic path (SnakeGame + SnakeState fed straight into search_many) must produce
        // bit-identical results to the public snake wrapper on the same state.
        let snakes = [
            snake(&[(6, 5), (6, 4), (6, 3)], Action::Right),
            snake(&[(2, 8), (2, 9), (1, 9)], Action::Left),
        ];
        let food: HashSet<Cell> = [(4, 4)].into_iter().collect();
        let mut p = params();
        p.expansion_budget = 24;

        let (game, cfg) = snake_game_and_config(&p);
        let state = SnakeState {
            snakes: snakes.clone(),
            food: food.clone(),
        };
        let generic = search_many(&game, &cfg, vec![(state, 0usize)], true, two_head_infer)
            .pop()
            .unwrap();
        let wrapped = selective_search(&p, snakes, food, 0, true, two_head_infer);

        assert_eq!(generic.0, wrapped.0, "root values must match");
        assert_eq!(generic.1, wrapped.1, "interior targets must match");
        assert_eq!(
            (
                generic.2.max_depth,
                generic.2.expansions,
                generic.2.leaves,
                generic.2.rounds
            ),
            (
                wrapped.2.max_depth,
                wrapped.2.expansions,
                wrapped.2.leaves,
                wrapped.2.rounds
            ),
        );
    }
}
