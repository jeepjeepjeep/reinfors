//! Single-agent delivery grid: fetch a parcel, carry it to the dropoff, on slippery ground.
//!
//! Two chance nodes, both enumerable. A root draw places the agent on a free cell, and — when
//! `p_slip` is non-zero — every action is resolved through a slip roll that may deflect the move
//! ninety degrees. The parcel and dropoff are fixed by configuration, so the only stochasticity an
//! agent must plan around is its own footing.
//!
//! One tick under slip: `step` parks the chosen action in `pending` and yields to chance;
//! `apply_chance_node` realizes the roll, moves the agent and emits the tick's single event. With
//! `p_slip == 0` the roll is skipped and `step` resolves the move directly, so the tree carries no
//! degenerate one-outcome fans.

use reinfors_core::{
    ActionView, Actor, ChanceDist, Game, Reward, Space, StateCodec, StateEncoder, Transition,
};

type Pos = (i32, i32);

const N_CHANNELS: usize = 3;

/// Row/column deltas for `up, down, left, right`.
const DELTAS: [Pos; 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];

/// The two actions perpendicular to each action — the deflections a slip can produce, indexed by
/// slip outcome minus one.
const PERP: [[usize; 2]; 4] = [[2, 3], [2, 3], [0, 1], [0, 1]];

/// Pre-birth sentinel: resolved by the root chance node before anything observes the state.
const UNBORN: Pos = (-1, -1);

/// Snapshot layout version for `StateCodec`.
const CODEC_VERSION: u8 = 1;

/// Full game state. `pos == UNBORN` and `pending.is_some()` are the two transient chance sentinels;
/// neither is ever observed, acted in, or accepted back from a snapshot.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct DeliveryState {
    pub pos: Pos,
    pub carrying: bool,
    /// Action awaiting its slip roll. `None` at every state an agent may act in.
    pub pending: Option<u8>,
    /// Derived: the parcel has been delivered. Rebuilt at decode, never on the wire.
    #[serde(skip)]
    pub done: bool,
}

/// What one tick decided. Components are independent: a single edge can slip and still land on
/// the parcel, and the truncating tick keeps whatever else it decided.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct DeliveryEvent {
    pub picked_up: bool,
    pub delivered: bool,
    pub slipped: bool,
    /// Set by `mark_truncation` on the final tick of an episode the engine cuts at `max_ticks`.
    pub timed_out: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct DeliveryReward {
    pub step: f64,
    pub pickup: f64,
    pub deliver: f64,
    pub slip: f64,
    pub timeout: f64,
}

impl Reward for DeliveryReward {
    type Event = DeliveryEvent;

    fn step_reward(&self, event: &DeliveryEvent, _agent: usize) -> f64 {
        let mut r = self.step;
        if event.picked_up {
            r += self.pickup;
        }
        if event.delivered {
            r += self.deliver;
        }
        if event.slipped {
            r += self.slip;
        }
        if event.timed_out {
            r += self.timeout;
        }
        r
    }
}

#[derive(Clone)]
pub struct DeliveryGrid {
    /// Side length of the square grid; at least 2 so a start cell exists beside the objectives.
    pub size: i32,
    pub parcel: Pos,
    pub dropoff: Pos,
    /// Probability that a move is deflected perpendicular, split evenly between the two sides.
    pub p_slip: f64,
    pub max_ticks: Option<usize>,
}

impl DeliveryGrid {
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
        for (label, p) in [("parcel", self.parcel), ("dropoff", self.dropoff)] {
            if !self.in_grid(p) {
                return Err(format!(
                    "{label} {p:?} is outside the {0}x{0} grid",
                    self.size
                ));
            }
        }
        if self.parcel == self.dropoff {
            return Err(format!(
                "parcel and dropoff must differ, both are {:?}",
                self.parcel
            ));
        }
        if !self.p_slip.is_finite() || !(0.0..=1.0).contains(&self.p_slip) {
            return Err(format!(
                "p_slip must be finite and within [0, 1], got {}",
                self.p_slip
            ));
        }
        Ok(())
    }

    /// Whether moves pass through the slip roll at all.
    fn slippery(&self) -> bool {
        self.p_slip / 2.0 > 0.0
    }

    fn in_grid(&self, (r, c): Pos) -> bool {
        0 <= r && r < self.size && 0 <= c && c < self.size
    }

    /// The single terminal condition: standing on the dropoff with the parcel in hand.
    fn delivered(&self, state: &DeliveryState) -> bool {
        state.carrying && state.pos == self.dropoff
    }

    fn moved(&self, (r, c): Pos, action: usize) -> Pos {
        let (dr, dc) = DELTAS[action];
        let next = (r + dr, c + dc);
        // Walls block rather than wrap; a blocked move still consumes the tick.
        if self.in_grid(next) {
            next
        } else {
            (r, c)
        }
    }

    fn cell_index(&self, (r, c): Pos) -> usize {
        (r * self.size + c) as usize
    }

    /// Start cells exclude the parcel and dropoff, so an episode never begins mid-objective.
    fn free_start_count(&self) -> usize {
        (self.size * self.size) as usize - 2
    }

    /// Row-major cell for the `outcome`th start cell, skipping the two objective cells.
    fn nth_free_cell(&self, outcome: usize) -> Pos {
        debug_assert!(
            outcome < self.free_start_count(),
            "birth outcome out of range"
        );
        let (p, d) = (self.cell_index(self.parcel), self.cell_index(self.dropoff));
        let (a, b) = (p.min(d), p.max(d));
        // Shift past each excluded index in ascending order.
        let mut i = outcome;
        if i >= a {
            i += 1;
        }
        if i >= b {
            i += 1;
        }
        let i = i as i32;
        (i / self.size, i % self.size)
    }

    /// Apply a realized move and emit what that edge decided.
    fn resolve(
        &self,
        state: &DeliveryState,
        action: usize,
        slipped: bool,
    ) -> Transition<DeliveryState, DeliveryEvent> {
        let pos = self.moved(state.pos, action);
        let mut carrying = state.carrying;
        let mut event = DeliveryEvent {
            slipped,
            ..Default::default()
        };
        if !carrying && pos == self.parcel {
            carrying = true;
            event.picked_up = true;
        } else if carrying && pos == self.dropoff {
            event.delivered = true;
        }
        Transition {
            next_state: DeliveryState {
                pos,
                carrying,
                pending: None,
                done: event.delivered,
            },
            events: vec![Some(event)],
            terminal: event.delivered,
        }
    }
}

impl Game for DeliveryGrid {
    type State = DeliveryState;
    type Event = DeliveryEvent;

    fn num_agents(&self) -> usize {
        1
    }

    fn action_count(&self) -> usize {
        DELTAS.len()
    }

    fn actor(&self, state: &DeliveryState) -> Actor {
        if state.pos == UNBORN || state.pending.is_some() {
            Actor::Chance
        } else {
            Actor::Agent(0)
        }
    }

    fn legal_actions(&self, state: &DeliveryState, agent: usize) -> Vec<usize> {
        if agent == 0 && !state.done && state.pos != UNBORN && state.pending.is_none() {
            (0..DELTAS.len()).collect()
        } else {
            Vec::new()
        }
    }

    fn step(
        &self,
        state: &DeliveryState,
        actions: &[usize],
    ) -> Transition<DeliveryState, DeliveryEvent> {
        debug_assert!(
            state.pos != UNBORN && state.pending.is_none(),
            "step only at a realized decision state"
        );
        let action = actions[0];
        if !self.slippery() {
            return self.resolve(state, action, false);
        }
        Transition {
            next_state: DeliveryState {
                pending: Some(action as u8),
                ..state.clone()
            },
            events: vec![None],
            terminal: false,
        }
    }

    fn initial_state(&self) -> DeliveryState {
        DeliveryState {
            pos: UNBORN,
            carrying: false,
            pending: None,
            done: false,
        }
    }

    fn chance_node(&self, state: &DeliveryState) -> ChanceDist {
        if state.pos == UNBORN {
            return ChanceDist::Uniform(self.free_start_count());
        }
        debug_assert!(
            state.pending.is_some() && self.slippery(),
            "chance only at birth or a slip roll"
        );
        if self.p_slip == 1.0 {
            return ChanceDist::Uniform(2);
        }
        // Outcome 0 keeps the intended heading; 1 and 2 are the two perpendicular deflections.
        let half = self.p_slip / 2.0;
        let weights = vec![1.0 - self.p_slip, half, half];
        debug_assert!(
            weights.iter().all(|&w| w > 0.0),
            "zero-weight slip outcome at p_slip={}",
            self.p_slip
        );
        ChanceDist::Weighted(weights)
    }

    fn apply_chance_node(
        &self,
        state: &DeliveryState,
        outcome: usize,
    ) -> Transition<DeliveryState, DeliveryEvent> {
        if state.pos == UNBORN {
            // The birth edge emits no events — there is no tick to deliver them into.
            return Transition::silent(
                DeliveryState {
                    pos: self.nth_free_cell(outcome),
                    carrying: false,
                    pending: None,
                    done: false,
                },
                1,
            );
        }
        let intended = usize::from(state.pending.expect("slip roll requires a pending action"));
        let outcome = if self.p_slip == 1.0 {
            outcome + 1
        } else {
            outcome
        };
        let (action, slipped) = if outcome == 0 {
            (intended, false)
        } else {
            (PERP[intended][outcome - 1], true)
        };
        self.resolve(state, action, slipped)
    }

    fn truncation_horizon(&self) -> Option<usize> {
        self.max_ticks
    }

    fn mark_truncation(&self, _state: &DeliveryState, trace: &mut Vec<(usize, DeliveryEvent)>) {
        for (_agent, event) in trace.iter_mut() {
            event.timed_out = true;
        }
    }
}

/// Three one-hot planes: agent, parcel, dropoff. The parcel plane empties exactly when the parcel
/// is held, so `carrying` stays observable without a fourth plane and the observation is Markov.
pub struct DeliveryPlanes {
    pub size: i32,
    pub parcel: Pos,
    pub dropoff: Pos,
}

impl ActionView for DeliveryPlanes {}

impl StateEncoder for DeliveryPlanes {
    type State = DeliveryState;

    fn encode(&self, state: &DeliveryState, _agent: usize) -> Vec<f32> {
        debug_assert!(
            state.pos != UNBORN && state.pending.is_none(),
            "transient chance states are never observed"
        );
        let g = self.size as usize;
        let mut obs = vec![0.0f32; N_CHANNELS * g * g];
        let at = |(r, c): Pos| (r as usize) * g + (c as usize);
        obs[at(state.pos)] = 1.0;
        if !state.carrying {
            obs[g * g + at(self.parcel)] = 1.0;
        }
        obs[2 * g * g + at(self.dropoff)] = 1.0;
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

impl StateCodec for DeliveryGrid {
    type State = DeliveryState;

    fn encode(&self, state: &DeliveryState) -> Vec<u8> {
        crate::codec_util::serde_encode(CODEC_VERSION, state)
    }

    fn decode(&self, bytes: &[u8]) -> Result<DeliveryState, String> {
        let mut s: DeliveryState = crate::codec_util::serde_decode(CODEC_VERSION, bytes)?;
        // `done` is derived, not stored: delivery is the only terminal.
        s.done = self.delivered(&s);
        Ok(s)
    }

    fn validate_decoded_state(&self, state: &DeliveryState, done: bool) -> Result<(), String> {
        if !self.in_grid(state.pos) {
            return Err(format!(
                "position {:?} outside the {1}x{1} grid",
                state.pos, self.size
            ));
        }
        if let Some(action) = state.pending {
            return Err(format!(
                "restored state retains a pending slip roll for action {action}"
            ));
        }
        let delivered = self.delivered(state);
        if state.done != delivered || done != delivered {
            return Err(format!(
                "done flags (state {}, envelope {done}) disagree with the state: carrying={} at \
                 {:?}, dropoff {:?}",
                state.done, state.carrying, state.pos, self.dropoff
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use reinfors_core::{
        check_action_view, search_many, ChanceMode, Dqn, Engine, EngineParams, EpsilonGreedyQ,
        Opponent, SearchConfig, SelectiveExpectimax, TreeStrap,
    };

    const UP: usize = 0;
    const DOWN: usize = 1;
    const LEFT: usize = 2;
    const RIGHT: usize = 3;

    fn grid(p_slip: f64) -> DeliveryGrid {
        DeliveryGrid {
            size: 4,
            parcel: (0, 3),
            dropoff: (3, 0),
            p_slip,
            max_ticks: Some(40),
        }
    }

    fn enc() -> DeliveryPlanes {
        let g = grid(0.0);
        DeliveryPlanes {
            size: g.size,
            parcel: g.parcel,
            dropoff: g.dropoff,
        }
    }

    fn reward() -> DeliveryReward {
        DeliveryReward {
            step: -0.01,
            pickup: 0.3,
            deliver: 1.0,
            slip: 0.0,
            timeout: 0.0,
        }
    }

    fn at(pos: Pos, carrying: bool) -> DeliveryState {
        DeliveryState {
            pos,
            carrying,
            pending: None,
            done: false,
        }
    }

    fn pending(pos: Pos, carrying: bool, action: usize) -> DeliveryState {
        DeliveryState {
            pending: Some(action as u8),
            ..at(pos, carrying)
        }
    }

    fn cfg(chance: ChanceMode) -> SearchConfig {
        SearchConfig {
            gamma: 0.99,
            beta: 1.0,
            expansion_budget: 64,
            top_k: 4,
            max_depth: 8,
            chance,
            opponent: Opponent::Uniform,
        }
    }

    fn zero_infer(_players: &[usize], _obs: Vec<f32>, n: usize) -> Vec<f64> {
        vec![0.0; n * 2 * 4]
    }

    #[test]
    fn move_tables_are_consistent() {
        for (a, &(dr, dc)) in DELTAS.iter().enumerate() {
            let [p, q] = PERP[a];
            assert_ne!(p, q);
            for perp in [p, q] {
                let (pr, pc) = DELTAS[perp];
                assert_eq!(
                    dr * pr + dc * pc,
                    0,
                    "PERP[{a}] must be orthogonal to DELTAS[{a}]"
                );
            }
        }
    }

    #[test]
    fn pickup_then_delivery_are_the_only_scoring_edges() {
        let g = grid(0.0);
        // Step right onto the parcel.
        let t = g.step(&at((0, 2), false), &[RIGHT]);
        let e = t.events[0].unwrap();
        assert!(e.picked_up && !e.delivered && !t.terminal);
        assert!(t.next_state.carrying);

        // Carrying the parcel onto the dropoff ends the episode.
        let t = g.step(&at((2, 0), true), &[DOWN]);
        let e = t.events[0].unwrap();
        assert!(e.delivered && t.terminal && t.next_state.done);
        assert!(g.legal_actions(&t.next_state, 0).is_empty());

        // Reaching the dropoff empty-handed, or revisiting the parcel while carrying, decides nothing.
        let t = g.step(&at((2, 0), false), &[DOWN]);
        assert!(!t.events[0].unwrap().delivered && !t.terminal);
        let t = g.step(&at((0, 2), true), &[RIGHT]);
        let e = t.events[0].unwrap();
        assert!(!e.picked_up && !e.delivered && !t.terminal && t.next_state.carrying);
    }

    #[test]
    fn walls_block_and_still_consume_the_tick() {
        let g = grid(0.0);
        let t = g.step(&at((0, 0), false), &[UP]);
        assert_eq!(t.next_state.pos, (0, 0));
        assert!(t.events[0].is_some(), "a blocked move still emits its edge");
    }

    #[test]
    fn slip_defers_to_a_chance_node_and_deflects_perpendicular() {
        let g = grid(0.2);
        let t = g.step(&at((1, 1), false), &[RIGHT]);
        assert_eq!(
            t.next_state,
            pending((1, 1), false, RIGHT),
            "the move waits on the roll"
        );
        assert!(
            t.events[0].is_none(),
            "the deferred edge decides nothing yet"
        );
        assert_eq!(g.actor(&t.next_state), Actor::Chance);
        assert!(g.legal_actions(&t.next_state, 0).is_empty());

        let p = t.next_state;
        // Outcome 0 is the intended move; 1 and 2 are the perpendicular deflections.
        let intended = g.apply_chance_node(&p, 0);
        assert_eq!(intended.next_state, at((1, 2), false));
        assert!(!intended.events[0].unwrap().slipped);
        for (outcome, expected) in [(1, (0, 1)), (2, (2, 1))] {
            let slipped = g.apply_chance_node(&p, outcome);
            assert_eq!(slipped.next_state, at(expected, false));
            assert!(slipped.events[0].unwrap().slipped);
        }

        // A deflection into a wall stays put, like any blocked move.
        let t = g.apply_chance_node(&pending((0, 1), false, RIGHT), 1);
        assert_eq!(t.next_state.pos, (0, 1));
        assert!(t.events[0].unwrap().slipped);

        // A slip can still land on an objective: the edge records both.
        let t = g.apply_chance_node(&pending((1, 3), false, LEFT), 1);
        let e = t.events[0].unwrap();
        assert_eq!(t.next_state.pos, g.parcel);
        assert!(e.slipped && e.picked_up && t.next_state.carrying);
    }

    #[test]
    fn zero_slip_skips_the_chance_node_entirely() {
        let t = grid(0.0).step(&at((1, 1), false), &[RIGHT]);
        assert_eq!(t.next_state, at((1, 2), false));
    }

    #[test]
    fn slip_probabilities_are_normalized_and_split_evenly() {
        let probs = |p_slip: f64| -> Vec<f64> {
            grid(p_slip)
                .chance_node(&pending((1, 1), false, RIGHT))
                .iter_probs()
                .unwrap()
                .collect()
        };
        for (p, expected) in [
            (0.3, [0.7, 0.15, 0.15]),
            (f64::MIN_POSITIVE, [1.0, 0.0, 0.0]),
        ] {
            let got = probs(p);
            assert_eq!(got.len(), 3);
            assert!(got.iter().all(|&g| g > 0.0), "p_slip={p}: {got:?}");
            for (g, e) in got.iter().zip(expected) {
                assert!((g - e).abs() < 1e-12, "p_slip={p}: {got:?}");
            }
        }
    }

    #[test]
    fn timeout_rewards_reach_engine_records_and_delivery_at_the_horizon_skips_them() {
        let money = DeliveryReward {
            step: 0.5,
            pickup: 2.0,
            deliver: 8.0,
            slip: 0.0,
            timeout: 32.0,
        };
        let run = |max_ticks: usize, infer: fn(&[f32]) -> Vec<f64>| -> Vec<f64> {
            let mut engine = Engine::new(
                DeliveryGrid {
                    size: 2,
                    parcel: (0, 1),
                    dropoff: (1, 1),
                    p_slip: 0.0,
                    max_ticks: Some(max_ticks),
                },
                Box::new(DeliveryPlanes {
                    size: 2,
                    parcel: (0, 1),
                    dropoff: (1, 1),
                }),
                Box::new(money),
                EpsilonGreedyQ::new(1, 0.0),
                Dqn::new(1, 1.0, 1, 0.95),
                EngineParams {
                    n_games: 4,
                    seed: 0,
                    ..Default::default()
                },
            );
            let (records, _) = engine.collect(40, move |o: Vec<f32>, n: usize| {
                let dim = o.len() / n;
                (0..n)
                    .flat_map(|i| infer(&o[i * dim..(i + 1) * dim]))
                    .collect()
            });
            records.iter().map(|t| t.reward).collect()
        };
        fn go_right(_obs: &[f32]) -> Vec<f64> {
            vec![0.0, 0.0, 0.0, 1.0]
        }
        let rewards = run(1, go_right);
        assert!(
            rewards.contains(&34.5),
            "pickup on the truncating tick stacks with timeout: {rewards:?}"
        );
        assert!(rewards.contains(&32.5), "{rewards:?}");
        assert!(
            rewards.iter().all(|&r| r == 34.5 || r == 32.5),
            "{rewards:?}"
        );
        fn right_then_down(obs: &[f32]) -> Vec<f64> {
            if obs[4..8].iter().sum::<f32>() == 0.0 {
                vec![0.0, 1.0, 0.0, 0.0]
            } else {
                vec![0.0, 0.0, 0.0, 1.0]
            }
        }
        let rewards = run(2, right_then_down);
        assert!(
            rewards.contains(&8.5),
            "delivery exactly at the horizon pays deliver, not timeout: {rewards:?}"
        );
        assert!(rewards.contains(&2.5), "{rewards:?}");
        assert!(rewards.contains(&32.5), "{rewards:?}");
        assert!(!rewards.contains(&40.5), "{rewards:?}");
    }

    #[test]
    fn shallow_chance_values_are_exact_at_both_slip_endpoints() {
        let flat = DeliveryReward {
            step: 0.0,
            pickup: 0.0,
            deliver: 1.0,
            slip: 0.0,
            timeout: 0.0,
        };
        let shallow = SearchConfig {
            max_depth: 1,
            ..cfg(ChanceMode::ExpandAll)
        };
        let heads = |p_slip: f64| -> Vec<Vec<f64>> {
            let g = grid(p_slip);
            let results = search_many(
                &g,
                &enc(),
                &flat,
                &shallow,
                vec![(at((2, 0), true), 0)],
                false,
                0,
                zero_infer,
            );
            results[0].0.clone()
        };
        for (p_slip, expected) in [
            (1.0, [0.0, 0.0, 0.5, 0.5]),
            (0.25, [0.0, 0.75, 0.125, 0.125]),
        ] {
            for head in heads(p_slip) {
                for (action, e) in expected.iter().enumerate() {
                    assert!((head[action] - e).abs() < 1e-9, "p_slip={p_slip}: {head:?}");
                }
            }
        }
    }

    #[test]
    fn certain_slip_enumerates_only_the_two_deflections() {
        let g = grid(1.0);
        let s = pending((1, 1), false, RIGHT);
        match g.chance_node(&s) {
            ChanceDist::Uniform(n) => assert_eq!(n, 2),
            other => panic!("certain slip must be uniform over the deflections, got {other:?}"),
        }
        let landed: Vec<Pos> = (0..2)
            .map(|o| g.apply_chance_node(&s, o).next_state.pos)
            .collect();
        assert_eq!(
            landed,
            vec![(0, 1), (2, 1)],
            "outcomes must map to UP/DOWN, never RIGHT"
        );
    }

    #[test]
    fn unrepresentably_small_p_slip_resolves_moves_directly() {
        let g = grid(5e-324);
        let t = g.step(&at((1, 1), false), &[RIGHT]);
        assert_eq!(t.next_state, at((1, 2), false));
    }

    #[test]
    fn birth_covers_every_cell_except_the_objectives_exactly_once() {
        let g = grid(0.1);
        let root = g.initial_state();
        assert_eq!(g.actor(&root), Actor::Chance);
        assert!(g.legal_actions(&root, 0).is_empty());
        let n = match g.chance_node(&root) {
            ChanceDist::Uniform(n) => n,
            other => panic!("birth should be a uniform draw, got {other:?}"),
        };
        assert_eq!(n, 14, "16 cells less the parcel and dropoff");

        let mut seen: Vec<Pos> = (0..n)
            .map(|o| {
                let t = g.apply_chance_node(&root, o);
                assert!(t.events[0].is_none(), "birth edges emit no events");
                assert!(!t.terminal);
                assert_eq!(g.actor(&t.next_state), Actor::Agent(0));
                assert_eq!(g.legal_actions(&t.next_state, 0).len(), 4);
                t.next_state.pos
            })
            .collect();
        seen.sort();
        seen.dedup();
        assert_eq!(seen.len(), n, "each outcome is a distinct cell");
        assert!(!seen.contains(&g.parcel) && !seen.contains(&g.dropoff));
        assert!(seen.iter().all(|&p| g.in_grid(p)));
    }

    #[test]
    fn birth_skips_objectives_wherever_they_sit() {
        // Objectives at the first two and the last two row-major indices stress the index shift.
        for (parcel, dropoff) in [((0, 0), (0, 1)), ((3, 3), (3, 2)), ((0, 1), (0, 0))] {
            let g = DeliveryGrid {
                parcel,
                dropoff,
                ..grid(0.0)
            };
            let root = g.initial_state();
            let cells: Vec<Pos> = (0..g.free_start_count())
                .map(|o| g.apply_chance_node(&root, o).next_state.pos)
                .collect();
            assert_eq!(cells.len(), 14);
            assert!(
                cells.windows(2).all(|w| w[0] < w[1]),
                "row-major and distinct"
            );
            assert!(!cells.contains(&parcel) && !cells.contains(&dropoff));
        }
    }

    #[test]
    fn truncation_marks_the_final_tick() {
        let g = grid(0.0);
        let mut trace = vec![(0, DeliveryEvent::default())];
        g.mark_truncation(&at((1, 1), true), &mut trace);
        assert!(trace[0].1.timed_out);
        assert!((reward().step_reward(&trace[0].1, 0) - -0.01).abs() < 1e-12);
    }

    #[test]
    fn reward_components_accumulate_on_one_edge() {
        let r = DeliveryReward {
            step: -0.01,
            pickup: 0.3,
            deliver: 1.0,
            slip: -0.05,
            timeout: -0.5,
        };
        let both = DeliveryEvent {
            picked_up: true,
            slipped: true,
            ..Default::default()
        };
        assert!((r.step_reward(&both, 0) - 0.24).abs() < 1e-12);
        let delivered = DeliveryEvent {
            delivered: true,
            ..Default::default()
        };
        assert!((r.step_reward(&delivered, 0) - 0.99).abs() < 1e-12);
        let timed_out = DeliveryEvent {
            timed_out: true,
            ..Default::default()
        };
        assert!((r.step_reward(&timed_out, 0) - -0.51).abs() < 1e-12);
    }

    #[test]
    fn parcel_plane_encodes_carrying_and_the_view_is_identity() {
        let (e, g) = (enc(), grid(0.0));
        let cells = (g.size * g.size) as usize;
        let empty = e.encode(&at((1, 1), false), 0);
        assert_eq!(empty.len(), N_CHANNELS * cells);
        assert_eq!(empty.iter().sum::<f32>(), 3.0, "one mark per plane");
        assert_eq!(
            empty[g.cell_index((1, 1))],
            1.0,
            "agent plane marks the position"
        );
        assert_eq!(empty[cells + g.cell_index(g.parcel)], 1.0);
        assert_eq!(empty[2 * cells + g.cell_index(g.dropoff)], 1.0);

        let held = e.encode(&at((1, 1), true), 0);
        assert!(
            held[cells..2 * cells].iter().all(|&v| v == 0.0),
            "the parcel plane empties once the parcel is held"
        );
        assert_eq!(held[..cells], empty[..cells]);
        assert_eq!(held[2 * cells..], empty[2 * cells..]);
        assert_eq!(e.obs_shape(), (N_CHANNELS, 4, 4));
        assert_eq!(
            e.observation_space(),
            Space::unit_box(vec![N_CHANNELS, 4, 4])
        );
        check_action_view(&e, g.action_count(), g.num_agents());
    }

    #[test]
    fn codec_round_trips_and_rejects_unsafe_states() {
        let g = grid(0.0);
        for state in [at((1, 2), false), at((0, 0), true)] {
            let bytes = g.encode(&state);
            let back = g.decode(&bytes).unwrap();
            assert_eq!(back, state);
            g.validate_decoded_state(&back, false).unwrap();
            assert!(g.validate_decoded_state(&back, true).is_err());
        }

        // Delivery is the only terminal, and `done` is rebuilt from the state.
        let delivered = at(g.dropoff, true);
        let back = g.decode(&g.encode(&delivered)).unwrap();
        assert!(back.done);
        g.validate_decoded_state(&back, true).unwrap();
        assert!(g.validate_decoded_state(&back, false).is_err());
        // An empty-handed visit to the dropoff is not a delivery.
        let back = g.decode(&g.encode(&at(g.dropoff, false))).unwrap();
        assert!(!back.done);
        g.validate_decoded_state(&back, false).unwrap();

        // A transient slip sentinel must never survive a restore.
        let restored = g.decode(&g.encode(&pending((1, 1), false, LEFT))).unwrap();
        assert!(g.validate_decoded_state(&restored, false).is_err());

        // A hand-built state whose done flag contradicts its contents is rejected too.
        let forged = DeliveryState {
            done: true,
            ..at((1, 1), true)
        };
        assert!(g.validate_decoded_state(&forged, true).is_err());

        assert!(g.validate_decoded_state(&at((9, 9), false), false).is_err());
        assert!(g.validate_decoded_state(&at(UNBORN, false), false).is_err());
        assert!(g.decode(&[]).is_err());
        assert!(g.decode(&[CODEC_VERSION + 1, 0, 0]).is_err());
    }

    #[test]
    fn validate_accepts_sane_configs_and_rejects_bad_ones() {
        assert!(grid(0.0).validate().is_ok());
        assert!(grid(1.0).validate().is_ok());
        let smallest = DeliveryGrid {
            size: 2,
            parcel: (0, 0),
            dropoff: (1, 1),
            p_slip: 0.5,
            max_ticks: None,
        };
        assert!(smallest.validate().is_ok());
        assert_eq!(smallest.free_start_count(), 2);
        let bad = |size, parcel, dropoff, p_slip| DeliveryGrid {
            size,
            parcel,
            dropoff,
            p_slip,
            max_ticks: None,
        };
        for (label, g) in [
            ("size 1", bad(1, (0, 0), (0, 1), 0.0)),
            ("negative size", bad(-4, (0, 0), (0, 1), 0.0)),
            ("parcel outside", bad(4, (4, 0), (0, 1), 0.0)),
            ("dropoff outside", bad(4, (0, 0), (-1, 1), 0.0)),
            ("coincident", bad(4, (2, 2), (2, 2), 0.0)),
            ("slip > 1", bad(4, (0, 0), (0, 1), 1.5)),
            ("slip negative", bad(4, (0, 0), (0, 1), -0.1)),
            ("slip nan", bad(4, (0, 0), (0, 1), f64::NAN)),
            ("slip inf", bad(4, (0, 0), (0, 1), f64::INFINITY)),
            ("obs overflow", bad(30_000, (0, 0), (0, 1), 0.0)),
            ("obs overflow past i64", bad(i32::MAX, (0, 0), (0, 1), 0.0)),
        ] {
            assert!(g.validate().is_err(), "{label} should be rejected");
        }
    }

    #[test]
    fn search_values_the_step_onto_the_dropoff_by_its_slip_odds() {
        // Expectimax enumerates the slip roll: with step reward 0 and deliver 1, stepping down onto
        // the dropoff from one cell above is worth at least 1 - p_slip (the deflected branches only
        // add discounted value on top). Perfect footing makes it exactly 1.
        let flat = DeliveryReward {
            step: 0.0,
            pickup: 0.0,
            deliver: 1.0,
            slip: 0.0,
            timeout: 0.0,
        };
        for p_slip in [0.0, 0.25] {
            let g = grid(p_slip);
            let results = search_many(
                &g,
                &enc(),
                &flat,
                &cfg(ChanceMode::ExpandAll),
                vec![(at((2, 0), true), 0)],
                false,
                0,
                zero_infer,
            );
            for head in &results[0].0 {
                let best = (0..4)
                    .max_by(|&a, &b| head[a].partial_cmp(&head[b]).unwrap())
                    .unwrap();
                assert_eq!(best, DOWN, "p_slip={p_slip}: {head:?}");
                let value = head[DOWN];
                assert!(value >= 1.0 - p_slip - 1e-9, "p_slip={p_slip}: {head:?}");
                if p_slip == 0.0 {
                    assert!((value - 1.0).abs() < 1e-9, "{head:?}");
                } else {
                    assert!(value < 1.0, "slipping must cost something: {head:?}");
                }
            }
        }
    }

    #[test]
    fn engine_collects_well_formed_transitions_under_slip() {
        let g = grid(0.25);
        let dim = N_CHANNELS * (g.size * g.size) as usize;
        let params = EngineParams {
            n_games: 4,
            seed: 0,
            ..Default::default()
        };
        let mut engine = Engine::new(
            g,
            Box::new(enc()),
            Box::new(reward()),
            EpsilonGreedyQ::new(1, 0.2),
            Dqn::new(1, 1.0, 1, 0.95),
            params,
        );
        let (records, stats) = engine.collect(200, |_o, n| vec![0.0; n * 4]);
        assert!(records.len() >= 200);
        for t in &records {
            assert_eq!(t.obs.len(), dim);
            assert_eq!(t.next_obs.len(), dim);
            assert!(t.action < 4);
        }
        assert!(stats.decisions > 0);
        assert!(
            !stats.episodes.is_empty(),
            "episodes should finish (deliver or truncate at max_ticks)"
        );
    }

    #[test]
    fn search_engine_enumerates_the_slip_roll_mid_episode() {
        let params = EngineParams {
            n_games: 3,
            seed: 0,
            ..Default::default()
        };
        let mut engine = Engine::new(
            grid(0.25),
            Box::new(enc()),
            Box::new(reward()),
            SelectiveExpectimax::new(cfg(ChanceMode::ExpandAll), 2, 0.0),
            TreeStrap::new(1.0, 0.3, 1.0, false),
            params,
        );
        let (records, stats) = engine.collect(50, |o, n| zero_infer(&[], o, n));
        assert!(records.len() >= 50);
        for (obs, tgt, mask, _player) in &records {
            assert_eq!(obs.len(), N_CHANNELS * 16);
            assert_eq!(tgt.len(), 2);
            assert!(tgt.iter().all(|row| row.len() == 4));
            assert_eq!(mask.len(), 2);
        }
        assert!(stats.decisions > 0);
    }
}
