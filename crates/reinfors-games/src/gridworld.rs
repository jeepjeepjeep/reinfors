//! GridWorld — a minimal single-agent navigation game, the first non-snake `Game`. It exercises the
//! framework's single-agent path (`num_agents == 1`, `Actor::Agent(0)` at every node — pure MAX +
//! lookahead, no opponent) end to end through the generic search and rollout engine. Deterministic, so
//! chance is the default (none declared).

use reinfors_core::{Actor, Game, Reward, Space, StateEncoder, Transition};

type Pos = (i32, i32);

const N_CHANNELS: usize = 2; // 0 = agent, 1 = goal
const DELTAS: [Pos; 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)]; // up, down, left, right

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GridState {
    pub pos: Pos,
    pub done: bool, // reached the goal
}

/// The agent's outcome on one tick: whether it reached the goal (terminal) this tick.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct GridEvent {
    pub reached_goal: bool,
}

/// GridWorld's reward weights: `goal` on reaching the goal (terminal), `step` on every other tick.
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

/// A `size x size` grid: one agent navigates to `goal` (terminal on arrival). The four moves are
/// always legal; a move into a wall keeps the agent in place. Rules only; the reward is the decoupled
/// [`GridWorldReward`].
pub struct GridWorld {
    pub size: i32,
    pub goal: Pos,
    /// Episode-length cap (the agent can wander indefinitely); the rollout truncates here. `None` =
    /// never truncate. GridWorld has no survival bonus, so truncation just ends the episode.
    pub max_ticks: Option<usize>,
}

impl GridWorld {
    fn moved(&self, (r, c): Pos, action: usize) -> Pos {
        let (dr, dc) = DELTAS[action];
        let (nr, nc) = (r + dr, c + dc);
        if 0 <= nr && nr < self.size && 0 <= nc && nc < self.size {
            (nr, nc)
        } else {
            (r, c) // wall: stay
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

    fn actor(&self, _state: &GridState) -> Actor {
        Actor::Agent(0)
    }

    fn legal_actions(&self, state: &GridState, agent: usize) -> Vec<usize> {
        if agent == 0 && !state.done {
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
            events: vec![GridEvent { reached_goal: done }],
            terminal: done,
        }
    }

    fn initial_state(&self, rng: &mut dyn reinfors_core::Rng) -> GridState {
        // A uniform-random start cell that is not already the goal.
        let cells = (self.size * self.size) as usize;
        loop {
            let i = rng.below(cells) as i32;
            let pos = (i / self.size, i % self.size);
            if pos != self.goal {
                return GridState { pos, done: false };
            }
        }
    }

    fn truncation_horizon(&self) -> Option<usize> {
        self.max_ticks
    }

    // Deterministic: no `chance_outcomes` declaration needed (the trait default suffices).
}

/// The default GridWorld observation: an agent-position plane and a goal-position plane. Carries
/// `size`/`goal` (the goal lives on `GridWorld`, not in `GridState`).
pub struct GridWorldPlanes {
    pub size: i32,
    pub goal: Pos,
}

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
        Space::unit_box(vec![c, h, w]) // agent + goal one-hot planes: values in [0, 1]
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
            opponent: Opponent::Uniform, // irrelevant: single-agent, no opponent nodes
        }
    }

    // K=2 heads, A=4, leaf value 0 — so the only signal is the terminal goal reward.
    fn zero_infer(_obs: Vec<f32>, n: usize) -> Vec<f64> {
        vec![0.0; n * 2 * 4]
    }

    #[test]
    fn step_moves_clamps_and_reaches_goal() {
        let w = world();
        // Right from (0,0) lands on the goal (0,1): terminal, goal reward.
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
        assert!(t.terminal && t.events[0].reached_goal);
        // Up from the top row is a wall: stay, non-terminal, no goal reached.
        let t = w.step(
            &GridState {
                pos: (0, 0),
                done: false,
            },
            &[0],
        );
        assert_eq!(t.next_state.pos, (0, 0));
        assert!(!t.terminal && !t.events[0].reached_goal);
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
            .is_empty()); // no agent 1
        assert!(w
            .legal_actions(
                &GridState {
                    pos: (0, 1),
                    done: true
                },
                0
            )
            .is_empty()); // terminal
    }

    #[test]
    fn search_finds_the_step_onto_the_goal() {
        // Agent one cell left of the goal; with zero leaf values the only signal is the terminal goal
        // reward, so the move that lands on the goal (right = 3) must have the highest root value. This
        // drives the single-agent `Actor::Agent(0)` MAX path through the generic search.
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
        let values = &results[0].0; // [K][A]
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
        // The rollout engine drives a 1-agent game end to end: records have the right shape ([K][A=4])
        // and episodes finish (reach the goal or truncate). Exercises num_agents == 1 in the engine.
        let policy = SelectiveExpectimax::new(cfg(), 2, 0.0); // n_heads, epsilon
        let learner = TreeStrap::new(0.99, 0.3, 1.0, false); // gamma, outcome_weight, bootstrap_p, interior
        let params = EngineParams {
            n_games: 3,
            seed: 0,
        };
        let mut engine = Engine::new(
            capped(Some(30)),
            Box::new(enc()),
            Box::new(reward()),
            policy,
            learner,
            params,
        );
        let (records, stats) = engine.collect(50, zero_infer);
        assert!(records.len() >= 50);
        for (obs, tgt, mask) in &records {
            assert_eq!(obs.len(), N_CHANNELS * 25);
            assert_eq!(tgt.len(), 2); // K heads
            assert!(tgt.iter().all(|row| row.len() == 4)); // A actions
            assert_eq!(mask.len(), 2);
        }
        assert!(stats.decisions > 0);
    }

    #[test]
    fn dqn_engine_collects_well_formed_transitions() {
        // The model-free DQN algorithm (no search) driven through the same generic Engine + GridWorld:
        // it emits off-policy transitions (obs, action, reward, next_obs, terminal, mask) instead of
        // TreeStrap targets — exercising the seam's non-search evaluation + transition-record path.
        use reinfors_core::{Dqn, EpsilonGreedyQ};

        let policy = EpsilonGreedyQ::new(2, 0.0); // 2 heads, no epsilon -> greedy argmax of the head
        let learner = Dqn::new(2, 1.0); // 2 heads, bootstrap_p = 1 -> all-ones masks
        let params = EngineParams {
            n_games: 3,
            seed: 0,
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
        let (records, stats) = engine.collect(120, zero_infer);
        assert!(records.len() >= 120);
        for t in &records {
            assert_eq!(t.obs.len(), dim);
            assert_eq!(
                t.next_obs.len(),
                dim,
                "s' is filled for a transition learner"
            );
            assert_eq!(t.mask, vec![1.0, 1.0]); // bootstrap_p = 1
            assert!(t.action < 4);
        }
        assert!(
            !stats.episodes.is_empty(),
            "episodes should finish (truncate at max_ticks)"
        );
    }
}
