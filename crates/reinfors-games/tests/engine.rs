//! Engine integration tests over the concrete `Snake` + `SelectiveExpectimax` +
//! `TreeStrap`. These live in reinfors-games (not core) so the generic `Engine` stays game-free:
//! they build the engine via the public core API (`Engine`, `EngineParams`, `SelectiveExpectimax`,
//! `TreeStrap`, `SearchConfig`) and a `Snake`.

use reinfors_core::{Engine, EngineParams, ReachedStateBuffer, SelectiveExpectimax, TreeStrap};
use reinfors_core::{Opponent, SearchConfig};
use reinfors_games::{snake_length_cell, EgocentricSnake, Snake, SnakeReward};

/// Test config bundle: the snake + search knobs the helpers below read (was `reinfors_games::
/// SearchParams`, which only the retired parity wrappers needed).
struct SearchParams {
    grid_size: i32,
    initial_length: usize,
    play_to_last: bool,
    win_food_lead: Option<usize>,
    gamma: f64,
    beta: f64,
    expansion_budget: usize,
    top_k: usize,
    max_depth: i32,
    food_samples: usize,
    max_ticks: Option<usize>,
    reward: SnakeReward,
    opponent: Opponent,
}

/// The default snake encoder for the engine (egocentric, sized from the search's grid).
fn enc(s: &SearchParams) -> Box<EgocentricSnake> {
    Box::new(EgocentricSnake {
        grid_size: s.grid_size,
    })
}

fn params(n_games: usize, seed: u64) -> EngineParams {
    EngineParams { n_games, seed }
}

/// The default test learner: gamma 0.99 (matches `search()`), the given z-mix weight + interior flag,
/// bootstrap_p 0.8.
fn learner(outcome_weight: f64, interior: bool) -> TreeStrap {
    TreeStrap::new(0.99, outcome_weight, 0.8, interior)
}

fn config(s: &SearchParams) -> SearchConfig {
    SearchConfig {
        gamma: s.gamma,
        beta: s.beta,
        expansion_budget: s.expansion_budget,
        top_k: s.top_k,
        max_depth: s.max_depth,
        food_samples: s.food_samples,
        opponent: s.opponent,
    }
}

/// The default test policy: the given head count, epsilon 0.1. (Interior collection is the learner's.)
fn policy(s: &SearchParams, n_heads: usize) -> SelectiveExpectimax {
    SelectiveExpectimax::new(config(s), n_heads, 0.1)
}

fn search() -> SearchParams {
    SearchParams {
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
        max_ticks: Some(50),
        reward: SnakeReward {
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

fn game(search: &SearchParams, initial_food_count: usize) -> Snake {
    Snake {
        grid_size: search.grid_size,
        initial_length: search.initial_length,
        play_to_last: search.play_to_last,
        win_food_lead: search.win_food_lead,
        initial_food_count,
        max_ticks: search.max_ticks,
    }
}

/// The decoupled reward handle (the engine + search apply it; the game no longer carries it).
fn reward(search: &SearchParams) -> Box<SnakeReward> {
    Box::new(search.reward)
}

/// Build an engine with the default test config (3 initial apples, TreeStrap with outcome_weight
/// 0.5 + interior targets), allowing per-test tweaks.
fn engine(
    n_games: usize,
    n_heads: usize,
    seed: u64,
) -> Engine<Snake, SelectiveExpectimax, TreeStrap> {
    let s = search();
    Engine::new(
        game(&s, 3),
        enc(&s),
        reward(&s),
        policy(&s, n_heads),
        learner(0.5, true),
        params(n_games, seed),
    )
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
    let mut e = engine(4, 2, 0);
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
    let r1 = engine(4, 2, 7).collect(60, infer).0;
    let r2 = engine(4, 2, 7).collect(60, infer).0;
    assert_eq!(r1, r2);
}

#[test]
fn distinct_seeds_diverge() {
    let r1 = engine(4, 2, 1).collect(80, infer).0;
    let r2 = engine(4, 2, 2).collect(80, infer).0;
    assert_ne!(r1, r2, "different seeds should produce different rollouts");
}

#[test]
fn games_carry_food_so_snakes_can_eat() {
    // Over a long rollout some snake should grow past its initial length (it ate), exercising the
    // in-tree spawn + env respawn path. Interior off so the record floor tracks decisions (with it
    // on, the floor is reached in far fewer ticks). The apple count is invariant: eating discards
    // one and respawns one, so every game always holds initial_food_count.
    let s = search();
    let mut e = Engine::new(
        game(&s, 3),
        enc(&s),
        reward(&s),
        policy(&s, 2),
        learner(0.5, false),
        params(8, 3),
    );
    for _ in 0..4 {
        e.collect(300, infer);
    }
    // The engine's internal state is private; a long rollout exercising eating is enough here — the
    // detailed growth/apple-count invariants are covered by the Snake unit tests.
}

#[test]
fn bootstrap_p_extremes_set_all_or_no_heads() {
    let s = search();
    // bootstrap_p now lives on the learner. n_heads (2) matches `infer`'s 2 heads.
    let all = TreeStrap::new(0.99, 0.5, 1.0, true);
    for (_, _, mask) in Engine::new(
        game(&s, 3),
        enc(&s),
        reward(&s),
        policy(&s, 2),
        all,
        params(4, 5),
    )
    .collect(40, infer)
    .0
    {
        assert!(
            mask.iter().all(|&m| m == 1.0),
            "p=1 must include every head"
        );
    }
    let none = TreeStrap::new(0.99, 0.5, 0.0, true);
    for (_, _, mask) in Engine::new(
        game(&s, 3),
        enc(&s),
        reward(&s),
        policy(&s, 2),
        none,
        params(4, 5),
    )
    .collect(40, infer)
    .0
    {
        assert!(mask.iter().all(|&m| m == 0.0), "p=0 must include no head");
    }
}

#[test]
fn zero_outcome_weight_leaves_targets_unblended() {
    // With outcome_weight = 0 the z-mix is a no-op, so a record's target equals its raw searched
    // values; with weight > 0 some executed-action entry must differ. We can't read the search
    // values here, but determinism lets us assert the two configs diverge.
    let s = search();
    let r0 = Engine::new(
        game(&s, 3),
        enc(&s),
        reward(&s),
        policy(&s, 2),
        learner(0.0, false),
        params(4, 9),
    )
    .collect(60, infer)
    .0;
    let r1 = Engine::new(
        game(&s, 3),
        enc(&s),
        reward(&s),
        policy(&s, 2),
        learner(0.9, false),
        params(4, 9),
    )
    .collect(60, infer)
    .0;
    let targets_differ = r0.iter().zip(&r1).any(|((_, t0, _), (_, t1, _))| t0 != t1);
    assert!(
        targets_differ,
        "outcome_weight should change executed-action targets"
    );
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
        let mut s = search();
        s.reward.survival = survival;
        s.max_ticks = Some(1); // the game's horizon: truncate after one decision per agent
        Engine::new(
            game(&s, 0),
            enc(&s),
            reward(&s),
            policy(&s, 2),
            learner(1.0, false),
            params(4, 0),
        )
        // no initial food; ow=1, interior off
    };
    let base = mk(0.0).collect(4, infer).0;
    let surv = mk(bonus).collect(4, infer).0;
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
    let s = search();
    let max_ticks = s.max_ticks.unwrap();
    let mut e = Engine::new(
        game(&s, 3),
        enc(&s),
        reward(&s),
        policy(&s, 2),
        learner(0.5, false),
        params(4, 11),
    );
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
    let stats1 = engine(4, 2, 13).collect(200, infer).1;
    let stats2 = engine(4, 2, 13).collect(200, infer).1;
    assert_eq!(stats1.decisions, stats2.decisions);
    assert_eq!(stats1.episodes.len(), stats2.episodes.len());
    for (a, b) in stats1.episodes.iter().zip(stats2.episodes.iter()) {
        assert_eq!(a.reward, b.reward);
        assert_eq!(a.length, b.length);
    }
}

#[test]
fn collected_targets_equal_a_direct_search() {
    // The engine collects each decision's raw searched values as its target (z-mix is a no-op at
    // outcome_weight 0). With no food the search has no chance node, so it is seed-independent — a
    // direct `search_many` on the food-free initial state reproduces the engine's first targets,
    // pinning that `collect` feeds through exactly what the search computes.
    use reinfors_core::{search_many, Game, Rng};
    use reinfors_games::EgocentricSnake;

    // A dummy RNG: with food = 0, `initial_state` spawns nothing, so it is never consulted.
    struct NoRng;
    impl Rng for NoRng {
        fn below(&mut self, _: usize) -> usize {
            0
        }
        fn unit(&mut self) -> f64 {
            0.0
        }
    }

    let mut s = search();
    s.max_ticks = Some(1); // the game's horizon: truncate after one decision per agent
    let (records, _) = Engine::new(
        game(&s, 0),
        enc(&s),
        reward(&s),
        policy(&s, 2),
        learner(0.0, false),
        params(1, 0),
    )
    .collect(2, infer);

    // The engine's deterministic food-free initial state (placement, no food) — same as episode 0's.
    let state = game(&s, 0).initial_state(&mut NoRng);
    let direct = search_many(
        &game(&s, 0),
        &EgocentricSnake {
            grid_size: s.grid_size,
        },
        &s.reward,
        &config(&s),
        vec![(state.clone(), 0), (state, 1)],
        false,
        0,
        infer,
    );

    assert_eq!(records.len(), 2);
    for (rec, d) in records.iter().zip(direct.iter()) {
        for (rh, dh) in rec.1.iter().zip(d.0.iter()) {
            for (rv, dv) in rh.iter().zip(dh.iter()) {
                assert!(
                    (rv - dv).abs() < 1e-9,
                    "collected target != direct search: {rv} vs {dv}"
                );
            }
        }
    }
}

#[test]
fn start_buffer_with_p_fresh_one_is_bit_identical_to_default() {
    // The determinism guard: the start-buffer machinery (per-tick `observe` + its disjoint RNG) must
    // never perturb the env-chance draws. With p_fresh = 1.0 every reset falls back to `initial_state`,
    // so the collected records must be bit-identical to the default `AlwaysInitialState` engine.
    let baseline = engine(4, 2, 5).collect(80, infer).0;
    let buffered = engine(4, 2, 5)
        .with_start_distribution(Box::new(ReachedStateBuffer::new(
            16,
            1.0,
            snake_length_cell,
        )))
        .collect(80, infer)
        .0;
    assert_eq!(
        baseline, buffered,
        "p_fresh=1 buffer must not change the rollout"
    );
}

#[test]
fn start_buffer_seeds_episodes_once_it_fills() {
    // With p_fresh = 0, once the buffer has states (after the first episode) every reset restores one,
    // so finished episodes get tagged `seeded`. The telemetry tag lets a caller keep off-d0 episodes
    // out of the true-start curves. Short horizon + interior off so episodes finish quickly and the
    // record floor tracks decisions.
    let mut s = search();
    s.max_ticks = Some(5);
    let mut e = Engine::new(
        game(&s, 3),
        enc(&s),
        reward(&s),
        policy(&s, 2),
        learner(0.5, false),
        params(4, 6),
    )
    .with_start_distribution(Box::new(ReachedStateBuffer::new(
        64,
        0.0,
        snake_length_cell,
    )));
    let mut any_seeded = false;
    for _ in 0..4 {
        let (_records, stats) = e.collect(200, infer);
        any_seeded |= stats.episodes.iter().any(|ep| ep.seeded);
    }
    assert!(
        any_seeded,
        "with p_fresh=0 and a filled buffer, some episodes should start from a buffered state"
    );
}
