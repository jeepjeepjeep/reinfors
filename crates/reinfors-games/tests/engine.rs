use reinfors_core::{ChanceMode, Opponent, SearchConfig};
use reinfors_core::{Engine, EngineParams, ReachedStateBuffer, SelectiveExpectimax, TreeStrap};
use reinfors_games::{snake_length_cell, EgocentricSnake, Snake, SnakeReward};

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
    chance: ChanceMode,
    max_ticks: Option<usize>,
    reward: SnakeReward,
    opponent: Opponent,
}

fn enc(s: &SearchParams) -> Box<EgocentricSnake> {
    Box::new(EgocentricSnake {
        grid_size: s.grid_size,
    })
}

fn params(n_games: usize, seed: u64) -> EngineParams {
    EngineParams {
        n_games,
        seed,
        n_groups: 1,
        ..Default::default()
    }
}

fn learner(outcome_weight: f64, interior: bool) -> TreeStrap {
    // Keep gamma coupled to search(), so direct-search and engine targets agree.
    TreeStrap::new(0.99, outcome_weight, 0.8, interior)
}

fn config(s: &SearchParams) -> SearchConfig {
    SearchConfig {
        gamma: s.gamma,
        beta: s.beta,
        expansion_budget: s.expansion_budget,
        top_k: s.top_k,
        max_depth: s.max_depth,
        chance: s.chance,
        opponent: s.opponent,
    }
}

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
        chance: ChanceMode::Committed { samples: 1 },
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
        num_snakes: 2,
        grid_size: search.grid_size,
        initial_length: search.initial_length,
        play_to_last: search.play_to_last,
        win_food_lead: search.win_food_lead,
        initial_food_count,
        max_ticks: search.max_ticks,
    }
}

fn reward(search: &SearchParams) -> Box<SnakeReward> {
    Box::new(search.reward)
}

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

// The two heads deliberately disagree, so tests exercise per-head targets rather than duplicates.
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
    for (obs, tgt, mask, _player) in &records {
        assert_eq!(obs.len(), 5 * 12 * 12);
        assert_eq!(tgt.len(), 2);
        assert!(tgt.iter().all(|row| row.len() == 3));
        assert_eq!(mask.len(), 2);
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
    // Long collection is the assertion: interior=false keeps its floor tied to
    // decisions while Snake's unit tests own the detailed food/growth invariants.
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
}

#[test]
fn bootstrap_p_extremes_set_all_or_no_heads() {
    let s = search();
    let all = TreeStrap::new(0.99, 0.5, 1.0, true);
    for (_, _, mask, _player) in Engine::new(
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
    for (_, _, mask, _player) in Engine::new(
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
    // This only proves outcome_weight changes targets; direct equality to raw search values is
    // covered by collected_targets_equal_a_direct_search below.
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
    let targets_differ = r0
        .iter()
        .zip(&r1)
        .any(|((_, t0, _, _), (_, t1, _, _))| t0 != t1);
    assert!(
        targets_differ,
        "outcome_weight should change executed-action targets"
    );
}

#[test]
fn survival_bonus_propagates_through_z_mixing_on_truncation() {
    // One-tick truncation and no food isolate the survival bonus. Interior records are never
    // z-mixed, so disabling them ensures every compared target is eligible for the bonus.
    let bonus = 0.25;
    let mk = |survival: f64| {
        let mut s = search();
        s.reward.survival = survival;
        s.max_ticks = Some(1);
        Engine::new(
            game(&s, 0),
            enc(&s),
            reward(&s),
            policy(&s, 2),
            learner(1.0, false),
            params(4, 0),
        )
    };
    let base = mk(0.0).collect(4, infer).0;
    let surv = mk(bonus).collect(4, infer).0;
    assert_eq!(base.len(), surv.len());
    assert!(!base.is_empty());
    for ((_, tb, _, _), (_, ts, _, _)) in base.iter().zip(surv.iter()) {
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
    // interior=false prevents interior records satisfying the floor before episodes finish.
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
    use reinfors_core::{search_many, Game};
    use reinfors_games::EgocentricSnake;

    let mut s = search();
    s.max_ticks = Some(1);
    let (records, _) = Engine::new(
        // No food removes chance, making the direct search seed-independent.
        game(&s, 0),
        enc(&s),
        reward(&s),
        policy(&s, 2),
        learner(0.0, false),
        params(1, 0),
    )
    .collect(2, infer);

    let state = game(&s, 0).initial_state();
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
        |_players: &[usize], o, n| infer(o, n),
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
    // p_fresh=1 must leave rollout chance untouched; the buffer uses a disjoint RNG stream.
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
    // A short horizon fills the buffer quickly; p_fresh=0 then forces restored starts.
    let mut s = search();
    s.max_ticks = Some(5);
    let mut e = Engine::new(
        game(&s, 3),
        enc(&s),
        reward(&s),
        policy(&s, 2),
        // Interior records could satisfy the floor before enough resets exercise the buffer.
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
