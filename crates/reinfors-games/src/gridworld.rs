//! Single-agent GridWorld with a uniformly sampled start cell.

use reinfors_core::{ActionView, Actor, Game, Reward, Space, StateEncoder, Transition};

type Pos = (i32, i32);

const N_CHANNELS: usize = 2;
const DELTAS: [Pos; 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct GridState {
    pub pos: Pos,
    #[serde(skip)]
    pub done: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct GridEvent {
    pub reached_goal: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct GridWorldReward {
    pub step: f64,
    pub goal: f64,
}

impl Reward for GridWorldReward {
    type Event = GridEvent;

    fn step_reward(&self, event: &GridEvent, _agent: usize) -> f64 {
        if event.reached_goal {
            self.goal
        } else {
            self.step
        }
    }
}

#[derive(Clone)]
pub struct GridWorld {
    pub size: i32,
    pub goal: Pos,
    pub max_ticks: Option<usize>,
}

impl GridWorld {
    pub fn validate(&self) -> Result<(), String> {
        if self.size < 2 {
            return Err(format!("size must be >= 2, got {}", self.size));
        }
        let cells = self.size as u128 * self.size as u128;
        if N_CHANNELS as u128 * cells > i32::MAX as u128 {
            return Err(format!(
                "size {} makes the observation tensor exceed 2^31 elements",
                self.size
            ));
        }
        let (r, c) = self.goal;
        if !(0 <= r && r < self.size && 0 <= c && c < self.size) {
            return Err(format!(
                "goal ({r}, {c}) is outside the {size}x{size} grid",
                size = self.size
            ));
        }
        Ok(())
    }

    fn moved(&self, (r, c): Pos, action: usize) -> Pos {
        let (dr, dc) = DELTAS[action];
        let (nr, nc) = (r + dr, c + dc);
        if 0 <= nr && nr < self.size && 0 <= nc && nc < self.size {
            (nr, nc)
        } else {
            (r, c)
        }
    }
}

impl Game for GridWorld {
    type State = GridState;
    type Event = GridEvent;

    fn num_agents(&self) -> usize {
        1
    }

    fn action_count(&self) -> usize {
        DELTAS.len()
    }

    fn actor(&self, state: &GridState) -> Actor {
        if state.pos == (-1, -1) {
            Actor::Chance
        } else {
            Actor::Agent(0)
        }
    }

    fn legal_actions(&self, state: &GridState, agent: usize) -> Vec<usize> {
        if agent == 0 && !state.done && state.pos != (-1, -1) {
            (0..DELTAS.len()).collect()
        } else {
            Vec::new()
        }
    }

    fn step(&self, state: &GridState, actions: &[usize]) -> Transition<GridState, GridEvent> {
        let pos = self.moved(state.pos, actions[0]);
        let done = pos == self.goal;
        Transition {
            next_state: GridState { pos, done },
            events: vec![Some(GridEvent { reached_goal: done })],
            terminal: done,
        }
    }

    fn initial_state(&self) -> GridState {
        // The sentinel is resolved by the root chance node before observation.
        GridState {
            pos: (-1, -1),
            done: false,
        }
    }

    fn chance_node(&self, state: &GridState) -> reinfors_core::ChanceDist {
        debug_assert_eq!(state.pos, (-1, -1), "chance only at the unborn root");
        reinfors_core::ChanceDist::Uniform((self.size * self.size - 1) as usize)
    }

    fn apply_chance_node(
        &self,
        _state: &GridState,
        outcome: usize,
    ) -> Transition<GridState, GridEvent> {
        let goal_idx = (self.goal.0 * self.size + self.goal.1) as usize;
        // Skip exactly the goal in row-major outcome order.
        let i = (outcome + usize::from(outcome >= goal_idx)) as i32;
        Transition::silent(
            GridState {
                pos: (i / self.size, i % self.size),
                done: false,
            },
            1,
        )
    }

    fn truncation_horizon(&self) -> Option<usize> {
        self.max_ticks
    }
}

pub struct GridWorldPlanes {
    pub size: i32,
    pub goal: Pos,
}

impl ActionView for GridWorldPlanes {}

impl StateEncoder for GridWorldPlanes {
    type State = GridState;

    fn encode(&self, state: &GridState, _agent: usize) -> Vec<f32> {
        let g = self.size as usize;
        let mut obs = vec![0.0f32; N_CHANNELS * g * g];
        let at = |r: i32, c: i32| (r as usize) * g + (c as usize);
        obs[at(state.pos.0, state.pos.1)] = 1.0;
        obs[g * g + at(self.goal.0, self.goal.1)] = 1.0;
        obs
    }

    fn obs_shape(&self) -> (usize, usize, usize) {
        (N_CHANNELS, self.size as usize, self.size as usize)
    }

    fn observation_space(&self) -> Space {
        let (c, h, w) = self.obs_shape();
        Space::unit_box(vec![c, h, w])
    }
}

impl reinfors_core::StateCodec for GridWorld {
    type State = GridState;

    fn encode(&self, s: &GridState) -> Vec<u8> {
        crate::codec_util::serde_encode(2, s)
    }

    fn decode(&self, bytes: &[u8]) -> Result<GridState, String> {
        let mut s: GridState = crate::codec_util::serde_decode(2, bytes)?;
        s.done = s.pos == self.goal;
        Ok(s)
    }

    fn validate_decoded_state(&self, state: &GridState, done: bool) -> Result<(), String> {
        let pos = state.pos;
        if !(0 <= pos.0 && pos.0 < self.size && 0 <= pos.1 && pos.1 < self.size) {
            return Err(format!(
                "position {pos:?} outside the {0}x{0} grid",
                self.size
            ));
        }
        if state.done != done {
            return Err(format!(
                "state done flag {} disagrees with envelope done {done}",
                state.done
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use reinfors_core::{
        search_many, ChanceMode, Engine, EngineParams, Opponent, SearchConfig, SelectiveExpectimax,
        TreeStrap,
    };

    fn world() -> GridWorld {
        capped(None)
    }

    fn capped(max_ticks: Option<usize>) -> GridWorld {
        GridWorld {
            size: 5,
            goal: (0, 1),
            max_ticks,
        }
    }

    fn reward() -> GridWorldReward {
        GridWorldReward {
            step: 0.0,
            goal: 1.0,
        }
    }

    fn enc() -> GridWorldPlanes {
        let w = world();
        GridWorldPlanes {
            size: w.size,
            goal: w.goal,
        }
    }

    fn cfg() -> SearchConfig {
        SearchConfig {
            gamma: 0.99,
            beta: 1.0,
            expansion_budget: 32,
            top_k: 4,
            max_depth: 8,
            chance: ChanceMode::Committed { samples: 1 },
            opponent: Opponent::Uniform,
        }
    }

    fn zero_infer(_players: &[usize], _obs: Vec<f32>, n: usize) -> Vec<f64> {
        vec![0.0; n * 2 * 4]
    }

    #[test]
    fn step_moves_clamps_and_reaches_goal() {
        let w = world();
        let t = w.step(
            &GridState {
                pos: (0, 0),
                done: false,
            },
            &[3],
        );
        assert_eq!(
            t.next_state,
            GridState {
                pos: (0, 1),
                done: true
            }
        );
        assert!(t.terminal && t.events[0].unwrap().reached_goal);
        let t = w.step(
            &GridState {
                pos: (0, 0),
                done: false,
            },
            &[0],
        );
        assert_eq!(t.next_state.pos, (0, 0));
        assert!(!t.terminal && !t.events[0].unwrap().reached_goal);
    }

    #[test]
    fn single_agent_metadata_and_legality() {
        let w = world();
        assert_eq!(w.num_agents(), 1);
        assert_eq!(w.action_count(), 4);
        assert_eq!(
            w.actor(&GridState {
                pos: (2, 2),
                done: false
            }),
            Actor::Agent(0)
        );
        assert_eq!(
            w.legal_actions(
                &GridState {
                    pos: (2, 2),
                    done: false
                },
                0
            ),
            vec![0, 1, 2, 3]
        );
        assert!(w
            .legal_actions(
                &GridState {
                    pos: (2, 2),
                    done: false
                },
                1
            )
            .is_empty());
        assert!(w
            .legal_actions(
                &GridState {
                    pos: (0, 1),
                    done: true
                },
                0
            )
            .is_empty());
    }

    #[test]
    fn search_finds_the_step_onto_the_goal() {
        let w = world();
        let start = GridState {
            pos: (0, 0),
            done: false,
        };
        let results = search_many(
            &w,
            &enc(),
            &reward(),
            &cfg(),
            vec![(start, 0)],
            false,
            0,
            zero_infer,
        );
        let values = &results[0].0;
        for head in values {
            let best = (0..4)
                .max_by(|&a, &b| head[a].partial_cmp(&head[b]).unwrap())
                .unwrap();
            assert_eq!(
                best, 3,
                "right (onto goal) should be the best action: {head:?}"
            );
            assert!(
                (head[3] - 1.0).abs() < 1e-9,
                "its value is the undiscounted goal reward"
            );
        }
    }

    #[test]
    fn engine_rolls_out_a_single_agent_game() {
        let policy = SelectiveExpectimax::new(cfg(), 2, 0.0);
        let learner = TreeStrap::new(0.99, 0.3, 1.0, false);
        let params = EngineParams {
            n_games: 3,
            seed: 0,
            n_groups: 1,
            ..Default::default()
        };
        let mut engine = Engine::new(
            capped(Some(30)),
            Box::new(enc()),
            Box::new(reward()),
            policy,
            learner,
            params,
        );
        let (records, stats) = engine.collect(50, |o, n| zero_infer(&[], o, n));
        assert!(records.len() >= 50);
        for (obs, tgt, mask, _player) in &records {
            assert_eq!(obs.len(), N_CHANNELS * 25);
            assert_eq!(tgt.len(), 2);
            assert!(tgt.iter().all(|row| row.len() == 4));
            assert_eq!(mask.len(), 2);
        }
        assert!(stats.decisions > 0);
    }

    #[test]
    fn dqn_engine_collects_well_formed_transitions() {
        use reinfors_core::{Dqn, EpsilonGreedyQ};

        let policy = EpsilonGreedyQ::new(2, 0.0);
        let learner = Dqn::new(2, 1.0, 1, 0.99);
        let params = EngineParams {
            n_games: 3,
            seed: 0,
            n_groups: 1,
            ..Default::default()
        };
        let mut engine = Engine::new(
            capped(Some(10)),
            Box::new(enc()),
            Box::new(reward()),
            policy,
            learner,
            params,
        );
        let dim = N_CHANNELS * 25;
        let (records, stats) = engine.collect(120, |o, n| zero_infer(&[], o, n));
        assert!(records.len() >= 120);
        for t in &records {
            assert_eq!(t.obs.len(), dim);
            assert_eq!(
                t.next_obs.len(),
                dim,
                "s' is filled for a transition learner"
            );
            assert_eq!(t.mask, vec![1.0, 1.0]);
            assert!(t.action < 4);
        }
        assert!(
            !stats.episodes.is_empty(),
            "episodes should finish (truncate at max_ticks)"
        );
    }

    #[test]
    fn validate_accepts_in_grid_goals_and_rejects_bad_configs() {
        let ok = |size, goal| GridWorld {
            size,
            goal,
            max_ticks: None,
        };
        assert!(ok(4, (3, 3)).validate().is_ok());
        assert!(ok(2, (0, 1)).validate().is_ok());
        for (size, goal) in [
            (4, (4, 4)),
            (4, (-1, 0)),
            (1, (0, 0)),
            (0, (0, 0)),
            (-3, (0, 0)),
            (40_000, (0, 0)),
            (i32::MAX, (0, 0)),
        ] {
            assert!(
                ok(size, goal).validate().is_err(),
                "size {size} goal {goal:?}"
            );
        }
    }
}
