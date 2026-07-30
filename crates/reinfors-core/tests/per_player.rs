//! Per-player inference routing: `collect_routed(InferMode::PerPlayer, ...)` serves each
//! row's PLAYER from its own network, records carry their player (the primary mechanism —
//! each player's transitions are off-policy data of the network that generated them),
//! `learn_players` skips frozen players' records at source, and per-player caches never
//! cross-contaminate. The shared wrapper stays byte-identical to the historical path.

use reinfors_core::{
    Actor, Dqn, Engine, EngineParams, EpsilonGreedyQ, Game, InferCache, InferMode, Transition,
};

#[derive(Clone)]
struct St {
    tick: usize,
}

/// Two players alternating for six plies; actions are free (both always legal); terminal pays
/// nothing — the tests read ACTIONS and record tags, not returns.
struct Alt;

impl Game for Alt {
    type State = St;
    type Event = ();
    fn num_agents(&self) -> usize {
        2
    }
    fn action_count(&self) -> usize {
        2
    }
    fn actor(&self, s: &St) -> Actor {
        Actor::Agent(s.tick % 2)
    }
    fn legal_actions(&self, s: &St, agent: usize) -> Vec<usize> {
        if agent == s.tick % 2 && s.tick < 6 {
            vec![0, 1]
        } else {
            Vec::new()
        }
    }
    fn step(&self, s: &St, _actions: &[usize]) -> Transition<St, ()> {
        Transition {
            next_state: St { tick: s.tick + 1 },
            events: vec![None; 2],
            terminal: s.tick + 1 >= 6,
        }
    }
    fn initial_state(&self) -> St {
        St { tick: 0 }
    }
}

struct Enc;
impl reinfors_core::ActionView for Enc {}
impl reinfors_core::StateEncoder for Enc {
    type State = St;
    fn encode(&self, s: &St, _agent: usize) -> Vec<f32> {
        vec![s.tick as f32]
    }
    fn obs_shape(&self) -> (usize, usize, usize) {
        (1, 1, 1)
    }
    fn observation_space(&self) -> reinfors_core::Space {
        reinfors_core::Space::unit_box(vec![1, 1, 1])
    }
}

struct Zero;
impl reinfors_core::Reward for Zero {
    type Event = ();
    fn step_reward(&self, _e: &(), _agent: usize) -> f64 {
        0.0
    }
}

fn engine() -> Engine<Alt, EpsilonGreedyQ, Dqn> {
    Engine::new(
        Alt,
        Box::new(Enc),
        Box::new(Zero),
        EpsilonGreedyQ::new(1, 0.0), // greedy: actions read the nets directly
        Dqn::new(1, 1.0),
        EngineParams {
            n_games: 2,
            seed: 3,
        },
    )
}

/// Player 0's net prefers action 0; player 1's prefers action 1.
fn opposed_nets(player: usize, _obs: Vec<f32>, n: usize) -> Vec<f64> {
    let row = if player == 0 { [1.0, 0.0] } else { [0.0, 1.0] };
    (0..n).flat_map(|_| row).collect()
}

#[test]
fn per_player_routing_serves_each_players_network() {
    let mut e = engine();
    let (records, _) = e.collect_routed(24, InferMode::PerPlayer, opposed_nets);
    assert!(records.len() >= 24);
    for r in &records {
        assert_eq!(
            r.action, r.player,
            "greedy actions read each player's own net (0 prefers 0, 1 prefers 1)"
        );
    }
    let players: std::collections::HashSet<usize> = records.iter().map(|r| r.player).collect();
    assert_eq!(players.len(), 2, "records carry both players");
}

#[test]
fn the_shared_wrapper_is_the_historical_path() {
    let digest = |records: Vec<reinfors_core::DqnRecord>| -> Vec<(usize, usize, Vec<u32>)> {
        records
            .into_iter()
            .map(|r| {
                (
                    r.player,
                    r.action,
                    r.obs.iter().map(|f| f.to_bits()).collect(),
                )
            })
            .collect()
    };
    let mut a = engine();
    let (ra, _) = a.collect(24, |_obs, n| vec![0.5; n * 2]);
    let mut b = engine();
    let (rb, _) = b.collect_routed(24, InferMode::Shared, |_p, _obs, n| vec![0.5; n * 2]);
    assert_eq!(digest(ra), digest(rb));
}

#[test]
fn learn_players_freezes_records_at_source() {
    let mut e = engine().with_learn_players(&[1]);
    let (records, stats) = e.collect_routed(12, InferMode::PerPlayer, opposed_nets);
    assert!(records.iter().all(|r| r.player == 1), "player 0 is frozen");
    // The frozen player still ACTS (decisions counted for both) — it just leaves no records.
    assert!(stats.decisions >= records.len() * 2 - 2);
}

#[test]
fn per_player_caches_never_cross_contaminate() {
    // Identical observations, different nets: with per-player caches, each player's rows must
    // come from its OWN network even though the obs bytes are identical.
    let caches = (0..3)
        .map(|_| {
            InferCache::new(
                1 << 10,
                std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0)),
            )
        })
        .collect();
    let mut e = engine().with_infer_caches(caches);
    let (records, _) = e.collect_routed(24, InferMode::PerPlayer, opposed_nets);
    for r in &records {
        assert_eq!(
            r.action, r.player,
            "a shared cache row would flip one player's actions"
        );
    }
}

/// Both players decide every tick (three ticks), so one pooled forward carries BOTH players'
/// rows — the cross-player row-width check has two groups to compare.
struct Simul;

impl Game for Simul {
    type State = St;
    type Event = ();
    fn num_agents(&self) -> usize {
        2
    }
    fn action_count(&self) -> usize {
        2
    }
    fn actor(&self, _s: &St) -> Actor {
        Actor::Simultaneous
    }
    fn legal_actions(&self, s: &St, _agent: usize) -> Vec<usize> {
        if s.tick < 3 {
            vec![0, 1]
        } else {
            Vec::new()
        }
    }
    fn step(&self, s: &St, _actions: &[usize]) -> Transition<St, ()> {
        Transition {
            next_state: St { tick: s.tick + 1 },
            events: vec![None; 2],
            terminal: s.tick + 1 >= 3,
        }
    }
    fn initial_state(&self) -> St {
        St { tick: 0 }
    }
}

#[test]
#[should_panic(expected = "not divisible")]
fn per_player_rows_must_divide_evenly() {
    let mut e = engine();
    let _ = e.collect_routed(4, InferMode::PerPlayer, |_p, _obs, n| vec![0.0; n * 2 + 1]);
}

#[test]
#[should_panic(expected = "row width")]
fn per_player_row_widths_must_agree_across_players() {
    let mut e = Engine::new(
        Simul,
        Box::new(Enc),
        Box::new(Zero),
        EpsilonGreedyQ::new(1, 0.0),
        Dqn::new(1, 1.0),
        EngineParams {
            n_games: 1,
            seed: 3,
        },
    );
    let _ = e.collect_routed(4, InferMode::PerPlayer, |p, _obs, n| vec![0.0; n * (2 + p)]);
}

#[test]
#[should_panic(expected = "out of range")]
fn learn_players_validates_indices() {
    let _ = engine().with_learn_players(&[2]);
}

#[test]
#[should_panic(expected = "at least one player")]
fn learn_players_rejects_empty() {
    let _ = engine().with_learn_players(&[]);
}
