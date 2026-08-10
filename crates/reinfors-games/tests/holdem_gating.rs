use reinfors_core::{
    ChanceMode, Dqn, Engine, EngineParams, EpsilonGreedyQ, Opponent, SearchConfig,
    SelectiveExpectimax, Space, StateEncoder, TreeStrap,
};
use reinfors_games::holdem::HOLDEM_ACTIONS;
use reinfors_games::{HoldemReward, HoldemState, TexasHoldem};

fn game() -> TexasHoldem {
    TexasHoldem {
        num_players: 3,
        stack: 100,
        small_blind: 5,
        big_blind: 10,
    }
}

struct FlatEnc;
impl reinfors_core::ActionView for FlatEnc {}
impl StateEncoder for FlatEnc {
    type State = HoldemState;
    fn encode(&self, s: &HoldemState, agent: usize) -> Vec<f32> {
        vec![
            s.hole[agent][0] as f32 / 51.0,
            s.hole[agent][1] as f32 / 51.0,
            s.board.len() as f32,
            s.stacks[agent] as f32,
        ]
    }
    fn obs_shape(&self) -> (usize, usize, usize) {
        (1, 1, 4)
    }
    fn observation_space(&self) -> Space {
        Space::unit_box(vec![1, 1, 4])
    }
}

#[test]
#[should_panic(expected = "clairvoyant")]
fn search_policies_reject_hidden_information_at_construction() {
    let policy = SelectiveExpectimax::new(
        SearchConfig {
            gamma: 1.0,
            beta: 1.0,
            expansion_budget: 4,
            top_k: 2,
            max_depth: 2,
            chance: ChanceMode::Committed { samples: 1 },
            opponent: Opponent::Uniform,
        },
        1,
        0.0,
    );
    let _ = Engine::new(
        game(),
        Box::new(FlatEnc),
        Box::new(HoldemReward { scale: 1.0 }),
        policy,
        TreeStrap::new(1.0, 0.3, 1.0, false),
        EngineParams {
            n_games: 1,
            seed: 0,
        },
    );
}

#[test]
fn dqn_family_collects_poker_hands() {
    let mut engine = Engine::new(
        game(),
        Box::new(FlatEnc),
        Box::new(HoldemReward { scale: 0.1 }),
        EpsilonGreedyQ::new(2, 0.2),
        Dqn::new(2, 1.0),
        EngineParams {
            n_games: 4,
            seed: 3,
        },
    );
    let (records, stats) = engine.collect(60, |_obs: Vec<f32>, n: usize| {
        vec![0.0; n * 2 * HOLDEM_ACTIONS]
    });
    assert!(records.len() >= 60);
    assert!(stats.decisions > 0 && !stats.episodes.is_empty());
    for ep in &stats.episodes {
        let sum: f64 = ep.reward.iter().sum();
        assert!(sum.abs() < 1e-9, "zero-sum hand, got {:?}", ep.reward);
    }
}
