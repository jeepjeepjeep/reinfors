//! N-player simultaneous Snake with egocentric observations and declared food chance.

use std::collections::{HashMap, HashSet, VecDeque};

use reinfors_core::game::{Actor, Game, Transition};
#[cfg(test)]
use reinfors_core::Rng;
use reinfors_core::{ActionView, Reward, Space, StateEncoder};

pub type Cell = (i32, i32);

pub const A: usize = 0;
pub const B: usize = 1;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DeathCause {
    Wall,
    SelfBody,
    OppBody,
    HeadOn,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SnakeBody {
    pub body: VecDeque<Cell>,
    pub direction: Action,
    pub alive: bool,
}

impl SnakeBody {
    pub fn head(&self) -> Cell {
        self.body[0]
    }

    pub fn len(&self) -> usize {
        self.body.len()
    }

    pub fn is_empty(&self) -> bool {
        self.body.is_empty()
    }
}

#[derive(Clone, Default, Debug, PartialEq, Eq)]
pub struct StepEvent {
    pub ate_food: bool,
    pub died: bool,
    pub death_cause: Option<DeathCause>,
    pub killed_opponent: bool,
    pub won: bool,
    pub lost: bool,
    pub drew: bool,
    pub survived_to_max_ticks: bool,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash, serde::Serialize, serde::Deserialize)]
pub enum Action {
    Up,
    Down,
    Left,
    Right,
}

impl Action {
    pub fn delta(self) -> (i32, i32) {
        match self {
            Action::Up => (-1, 0),
            Action::Down => (1, 0),
            Action::Left => (0, -1),
            Action::Right => (0, 1),
        }
    }

    pub fn opposite(self) -> Action {
        match self {
            Action::Up => Action::Down,
            Action::Down => Action::Up,
            Action::Left => Action::Right,
            Action::Right => Action::Left,
        }
    }

    pub fn ego_rot_k(self) -> u8 {
        match self {
            Action::Up => 0,
            Action::Right => 1,
            Action::Down => 2,
            Action::Left => 3,
        }
    }

    fn cw(self) -> Action {
        match self {
            Action::Up => Action::Right,
            Action::Right => Action::Down,
            Action::Down => Action::Left,
            Action::Left => Action::Up,
        }
    }

    fn ccw(self) -> Action {
        match self {
            Action::Up => Action::Left,
            Action::Left => Action::Down,
            Action::Down => Action::Right,
            Action::Right => Action::Up,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RelativeAction {
    Forward,
    Left,
    Right,
}

pub const RELATIVE_ACTIONS: [RelativeAction; 3] = [
    RelativeAction::Forward,
    RelativeAction::Left,
    RelativeAction::Right,
];

pub fn relative_to_absolute(heading: Action, rel: RelativeAction) -> Action {
    match rel {
        RelativeAction::Forward => heading,
        RelativeAction::Left => heading.ccw(),
        RelativeAction::Right => heading.cw(),
    }
}

pub const N_CHANNELS: usize = 5;
const CH_OWN_HEAD: usize = 0;
const CH_OWN_BODY: usize = 1;
const CH_OPP_HEAD: usize = 2;
const CH_OPP_BODY: usize = 3;
const CH_FOOD: usize = 4;

pub fn egocentric_parts(
    snakes: &[SnakeBody],
    food: &HashSet<Cell>,
    grid_size: i32,
    agent: usize,
) -> Vec<f32> {
    let g = grid_size;
    let edge = g - 1;
    let k = snakes[agent].direction.ego_rot_k();
    let plane = (g * g) as usize;
    let mut obs = vec![0.0f32; N_CHANNELS * plane];

    let rot = |r: i32, c: i32| -> (i32, i32) {
        match k {
            1 => (edge - c, r),
            2 => (edge - r, edge - c),
            3 => (c, edge - r),
            _ => (r, c),
        }
    };
    let mut set = |ch: usize, r: i32, c: i32| {
        let (rr, cc) = rot(r, c);
        obs[ch * plane + (rr as usize) * (g as usize) + (cc as usize)] = 1.0;
    };

    for (i, snake) in snakes.iter().enumerate() {
        if snake.is_empty() {
            continue;
        }
        let (head_ch, body_ch) = if i == agent {
            (CH_OWN_HEAD, CH_OWN_BODY)
        } else {
            (CH_OPP_HEAD, CH_OPP_BODY)
        };
        let mut ch = head_ch;
        for &(r, c) in &snake.body {
            set(ch, r, c);
            ch = body_ch;
        }
    }
    for &(r, c) in food {
        set(CH_FOOD, r, c);
    }
    obs
}

#[derive(Clone, Copy, Debug)]
pub struct SnakeReward {
    pub step: f64,
    pub food: f64,
    pub loss: f64,
    pub draw: f64,
    pub kill: f64,
    pub win: f64,
    pub survival: f64,
}

impl Reward for SnakeReward {
    type Event = StepEvent;

    fn step_reward(&self, e: &StepEvent, _agent: usize) -> f64 {
        let mut reward = self.step;
        if e.died {
            if e.lost {
                reward += self.loss;
            }
            if e.drew {
                reward += self.draw;
            }
            return reward;
        }
        if e.ate_food {
            reward += self.food;
        }
        if e.killed_opponent {
            reward += self.kill;
        }
        if e.won {
            reward += self.win;
        }
        if e.lost {
            reward += self.loss;
        }
        if e.survived_to_max_ticks {
            reward += self.survival;
        }
        reward
    }
}

mod food_serde {
    use super::Cell;
    use serde::de::Error as _;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::collections::HashSet;

    pub fn serialize<S: Serializer>(food: &HashSet<Cell>, ser: S) -> Result<S::Ok, S::Error> {
        // HashSet order must not make equal snapshots encode differently.
        let mut cells: Vec<Cell> = food.iter().copied().collect();
        cells.sort_unstable();
        cells.serialize(ser)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(de: D) -> Result<HashSet<Cell>, D::Error> {
        let cells = Vec::<Cell>::deserialize(de)?;
        let mut food = HashSet::with_capacity(cells.len());
        for cell in cells {
            if !food.insert(cell) {
                return Err(D::Error::custom(format!("duplicate food cell {cell:?}")));
            }
        }
        Ok(food)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SnakeState {
    pub snakes: Vec<SnakeBody>,
    #[serde(with = "food_serde")]
    pub food: HashSet<Cell>,
    #[serde(default)]
    // Nonzero means nature must restore eaten or initial food.
    pub pending_food: u32,
    #[serde(default)]
    // Birth places one apple per chance edge; respawns combine all apples in one edge.
    pub birth: bool,
}

const CELL_BUCKET_SIZE: usize = 3;
const CELL_N_BUCKETS: u64 = 10;

pub fn snake_length_cell(state: &SnakeState) -> Option<u64> {
    if state.snakes.iter().any(|s| !s.alive) {
        return None;
    }
    let bucket = |len: usize| ((len / CELL_BUCKET_SIZE) as u64).min(CELL_N_BUCKETS - 1);
    let mut buckets: Vec<u64> = state.snakes.iter().map(|s| bucket(s.len())).collect();
    buckets.sort_unstable();
    // Preserve legacy two-player keys: changing this shift re-buckets saved buffers.
    let shift = if buckets.len() <= 4 { 16 } else { 8 };
    Some(buckets.into_iter().fold(0u64, |acc, b| (acc << shift) | b))
}

pub struct EgocentricSnake {
    pub grid_size: i32,
}

impl ActionView for EgocentricSnake {}

impl StateEncoder for EgocentricSnake {
    type State = SnakeState;

    fn encode(&self, state: &SnakeState, agent: usize) -> Vec<f32> {
        egocentric_parts(&state.snakes, &state.food, self.grid_size, agent)
    }

    fn obs_shape(&self) -> (usize, usize, usize) {
        (N_CHANNELS, self.grid_size as usize, self.grid_size as usize)
    }

    fn observation_space(&self) -> Space {
        let (c, h, w) = self.obs_shape();
        Space::unit_box(vec![c, h, w])
    }
}

#[derive(Clone)]
pub struct Snake {
    pub num_snakes: usize,
    pub grid_size: i32,
    pub initial_length: usize,
    pub play_to_last: bool,
    pub win_food_lead: Option<usize>,
    pub initial_food_count: usize,
    pub max_ticks: Option<usize>,
}

impl Snake {
    fn in_bounds(&self, (r, c): Cell) -> bool {
        0 <= r && r < self.grid_size && 0 <= c && c < self.grid_size
    }

    pub fn validate(&self) -> Result<(), String> {
        if !(2..=8).contains(&self.num_snakes) {
            return Err("num_snakes must be in 2..=8".to_string());
        }
        if self.initial_length < 1 {
            return Err("initial_length must be >= 1".to_string());
        }
        if self.win_food_lead == Some(0) {
            return Err(
                "win_food_lead must be >= 1 (a lead of 0 ends the game on tick one)".to_string(),
            );
        }
        if self.win_food_lead.is_some() && self.num_snakes != 2 {
            return Err(
                "win_food_lead is a two-snake rule (a lead over WHOM is undefined past two); \
                 unset it for more snakes"
                    .to_string(),
            );
        }
        let g = self.grid_size;
        let cells = g as i64 * g as i64;
        if g >= 1 && N_CHANNELS as u128 * cells as u128 > i32::MAX as u128 {
            return Err(format!(
                "grid_size {g} makes the observation tensor exceed 2^31 elements"
            ));
        }
        let cell_count = self.grid_size.max(0) as u128 * self.grid_size.max(0) as u128;
        if self.initial_food_count as u128 > cell_count {
            return Err(format!(
                "initial_food_count {} exceeds the {cell_count}-cell grid",
                self.initial_food_count
            ));
        }
        if self.initial_food_count > reinfors_core::game::CHANCE_CHAIN_LIMIT {
            return Err(format!(
                "initial_food_count {} exceeds the {}-edge chance-chain limit",
                self.initial_food_count,
                reinfors_core::game::CHANCE_CHAIN_LIMIT
            ));
        }
        let k_max = self.num_snakes.min(self.initial_food_count) as u128;
        // Combined chance outcomes pass through f64; above 2^53 distinct indices alias.
        let worst: u128 = (0..k_max)
            .map(|i| cell_count.saturating_sub(i))
            .try_fold(1u128, |acc, f| acc.checked_mul(f))
            .unwrap_or(u128::MAX);
        if worst > (1u128 << 53).min(usize::MAX as u128) {
            return Err(format!(
                "worst-case respawn index space ({k_max} apples eatable at once on a \
                 {cell_count}-cell grid) exceeds min(2^53, usize::MAX); reduce food or grid size"
            ));
        }
        let snakes = self.initial_snakes_checked()?;
        let mut seen = HashSet::new();
        for snake in &snakes {
            for &(r, c) in &snake.body {
                if !(0 <= r && r < g && 0 <= c && c < g) {
                    return Err(format!(
                        "grid_size {g} is too small: initial snake placement leaves cell ({r}, {c}) outside the grid"
                    ));
                }
                if !seen.insert((r, c)) {
                    return Err(format!(
                        "grid_size {g} is too small for initial_length {}: initial snakes overlap at ({r}, {c})",
                        self.initial_length
                    ));
                }
            }
        }
        let free = cells - seen.len() as i64;
        if self.initial_food_count as u128 > free as u128 {
            return Err(format!(
                "food {} exceeds the {free} free cells left by the initial snakes",
                self.initial_food_count
            ));
        }
        Ok(())
    }

    fn initial_snakes(&self) -> Vec<SnakeBody> {
        self.initial_snakes_checked()
            .expect("snake config validated at construction")
    }

    fn initial_snakes_checked(&self) -> Result<Vec<SnakeBody>, String> {
        let g = self.grid_size;
        let mid = g / 2;
        let rows: Vec<i32> = if self.num_snakes == 2 {
            vec![mid, mid]
        } else {
            (0..self.num_snakes)
                .map(|i| (i as i32 + 1) * g / (self.num_snakes as i32 + 1))
                .collect()
        };
        let mut snakes = Vec::with_capacity(self.num_snakes);
        for (i, &row) in rows.iter().enumerate() {
            let snake = if i % 2 == 0 {
                SnakeBody {
                    body: Self::trace_body(
                        g,
                        (row, g / 3),
                        &[Action::Left, Action::Down, Action::Right, Action::Up],
                        self.initial_length,
                    )?,
                    direction: Action::Right,
                    alive: true,
                }
            } else {
                SnakeBody {
                    body: Self::trace_body(
                        g,
                        (row, g - g / 3),
                        &[Action::Right, Action::Up, Action::Left, Action::Down],
                        self.initial_length,
                    )?,
                    direction: Action::Left,
                    alive: true,
                }
            };
            snakes.push(snake);
        }
        Ok(snakes)
    }

    fn trace_body(
        grid_size: i32,
        head: Cell,
        dirs: &[Action],
        length: usize,
    ) -> Result<VecDeque<Cell>, String> {
        let g = grid_size;
        let mut cells = VecDeque::from([head]);
        let (mut r, mut c) = head;
        let mut d = 0usize;
        while cells.len() < length {
            let (dr, dc) = dirs[d].delta();
            let (nr, nc) = (r + dr, c + dc);
            if !(0 <= nr && nr < g && 0 <= nc && nc < g) {
                d += 1;
                if d >= dirs.len() {
                    return Err(format!(
                        "initial_length {length} does not fit grid_size {g}"
                    ));
                }
                continue;
            }
            r = nr;
            c = nc;
            cells.push_back((r, c));
        }
        Ok(cells)
    }

    fn advance(
        &self,
        snakes: &mut [SnakeBody],
        food: &mut HashSet<Cell>,
        actions: &[Option<Action>],
        mut next_food: impl FnMut() -> Option<Cell>,
    ) -> (Vec<StepEvent>, bool) {
        let n = snakes.len();
        let mut events = vec![StepEvent::default(); n];

        let mut ate_intent = vec![false; n];
        let mut moved = vec![false; n];
        for i in 0..n {
            if !snakes[i].alive {
                continue;
            }
            moved[i] = true;
            let mut action = actions[i].unwrap_or(snakes[i].direction);
            if action.opposite() == snakes[i].direction {
                action = snakes[i].direction;
            }
            snakes[i].direction = action;
            let (dr, dc) = action.delta();
            let (hr, hc) = snakes[i].head();
            let new_head = (hr + dr, hc + dc);
            ate_intent[i] = food.contains(&new_head);
            snakes[i].body.push_front(new_head);
            if !ate_intent[i] {
                snakes[i].body.pop_back();
            }
        }

        // Resolve every collision against the same post-move world before removing corpses.
        let causes = self.resolve_collisions(snakes, &moved);
        let mut fatal_heads = vec![None; n];
        for i in 0..n {
            if causes[i].is_some() {
                fatal_heads[i] = Some(snakes[i].head());
            }
        }
        for i in 0..n {
            if let Some(cause) = causes[i] {
                snakes[i].alive = false;
                snakes[i].body.pop_front();
                events[i].died = true;
                events[i].death_cause = Some(cause);
            }
        }

        for i in 0..n {
            if ate_intent[i] && causes[i].is_none() {
                let head = snakes[i].head();
                food.remove(&head);
                events[i].ate_food = true;
                if let Some(cell) = next_food() {
                    food.insert(cell);
                }
            }
        }

        for i in 0..n {
            if causes[i] == Some(DeathCause::OppBody) {
                if let Some(killer) = self.find_killer(snakes, fatal_heads[i].unwrap(), i) {
                    events[killer].killed_opponent = true;
                }
            }
        }

        let alive_ids: Vec<usize> = (0..n).filter(|&i| snakes[i].alive).collect();
        let n_causes = causes.iter().filter(|c| c.is_some()).count();
        if alive_ids.len() == 1 && n_causes > 0 {
            events[alive_ids[0]].won = true;
        }
        for i in 0..n {
            if causes[i].is_some() {
                if !alive_ids.is_empty() {
                    events[i].lost = true;
                } else if n_causes >= 2 {
                    events[i].drew = true;
                }
            }
        }
        let done = self.is_terminal(snakes);
        if done && alive_ids.len() >= 2 {
            let (i0, i1) = (alive_ids[0], alive_ids[1]);
            let (leader, runner) = if snakes[i0].len() >= snakes[i1].len() {
                (i0, i1)
            } else {
                (i1, i0)
            };
            events[leader].won = true;
            events[runner].lost = true;
        }

        (events, done)
    }

    fn is_terminal(&self, snakes: &[SnakeBody]) -> bool {
        let alive: Vec<usize> = (0..snakes.len()).filter(|&i| snakes[i].alive).collect();
        if alive.len() <= if self.play_to_last { 0 } else { 1 } {
            return true;
        }
        if let (Some(lead), &[i0, i1]) = (self.win_food_lead, alive.as_slice()) {
            return snakes[i0].len().abs_diff(snakes[i1].len()) >= lead;
        }
        false
    }

    fn resolve_collisions(&self, snakes: &[SnakeBody], moved: &[bool]) -> Vec<Option<DeathCause>> {
        let n = snakes.len();
        let mut causes = vec![None; n];

        let mut by_head: HashMap<Cell, Vec<usize>> = HashMap::new();
        for (i, &m) in moved.iter().enumerate() {
            if m {
                by_head.entry(snakes[i].head()).or_default().push(i);
            }
        }
        for ids in by_head.values() {
            if ids.len() >= 2 {
                for &i in ids {
                    causes[i] = Some(DeathCause::HeadOn);
                }
            }
        }

        for i in 0..n {
            if !moved[i] || causes[i].is_some() {
                continue;
            }
            let head = snakes[i].head();
            if !self.in_bounds(head) {
                causes[i] = Some(DeathCause::Wall);
            } else if snakes[i].body.iter().skip(1).any(|&c| c == head) {
                causes[i] = Some(DeathCause::SelfBody);
            } else if self.find_killer(snakes, head, i).is_some() {
                causes[i] = Some(DeathCause::OppBody);
            }
        }
        causes
    }

    fn find_killer(&self, snakes: &[SnakeBody], fatal_head: Cell, victim: usize) -> Option<usize> {
        (0..snakes.len())
            .find(|&j| j != victim && snakes[j].alive && snakes[j].body.contains(&fatal_head))
    }

    fn occupied_of(&self, snakes: &[SnakeBody], food: &HashSet<Cell>) -> Vec<usize> {
        let g = self.grid_size;
        let mut occupied: Vec<usize> = food.iter().map(|&(r, c)| (r * g + c) as usize).collect();
        for s in snakes {
            occupied.extend(s.body.iter().map(|&(r, c)| (r * g + c) as usize));
        }
        occupied.sort_unstable();
        occupied.dedup();
        occupied
    }

    fn nth_free_of(occupied: &[usize], grid: i32, i: usize) -> Cell {
        // Translate an index in the compact free-cell list into a row-major cell id.
        let mut idx = i;
        for &o in occupied {
            if o <= idx {
                idx += 1;
            } else {
                break;
            }
        }
        (idx as i32 / grid, idx as i32 % grid)
    }

    fn occupied_cells(&self, state: &SnakeState) -> Vec<usize> {
        self.occupied_of(&state.snakes, &state.food)
    }

    fn free_cell_count(&self, state: &SnakeState) -> usize {
        (self.grid_size * self.grid_size) as usize - self.occupied_cells(state).len()
    }

    fn nth_free_cell(&self, state: &SnakeState, i: usize) -> Cell {
        Self::nth_free_of(&self.occupied_cells(state), self.grid_size, i)
    }
}

impl Game for Snake {
    type State = SnakeState;
    type Event = StepEvent;

    fn num_agents(&self) -> usize {
        self.num_snakes
    }

    fn action_count(&self) -> usize {
        RELATIVE_ACTIONS.len()
    }

    fn actor(&self, state: &SnakeState) -> Actor {
        if state.pending_food > 0 {
            Actor::Chance
        } else {
            Actor::Simultaneous
        }
    }

    fn legal_actions(&self, state: &SnakeState, agent: usize) -> Vec<usize> {
        if state.pending_food == 0 && state.snakes[agent].alive {
            (0..RELATIVE_ACTIONS.len()).collect()
        } else {
            Vec::new()
        }
    }

    fn step(&self, state: &SnakeState, actions: &[usize]) -> Transition<SnakeState, StepEvent> {
        let mut moves: Vec<Option<Action>> = vec![None; state.snakes.len()];
        for (i, (slot, snake)) in moves.iter_mut().zip(state.snakes.iter()).enumerate() {
            if snake.alive {
                *slot = Some(relative_to_absolute(
                    snake.direction,
                    RELATIVE_ACTIONS[actions[i]],
                ));
            }
        }
        let mut snakes = state.snakes.clone();
        let mut food = state.food.clone();
        let (events, done) = self.advance(&mut snakes, &mut food, &moves, || None);
        let eaten = state.food.len().saturating_sub(food.len());
        let pending_food = if done { 0 } else { eaten as u32 };
        Transition {
            next_state: SnakeState {
                snakes,
                food,
                pending_food,
                birth: false,
            },
            events: events.into_iter().map(Some).collect(),
            terminal: done,
        }
    }

    fn chance_node(&self, state: &SnakeState) -> reinfors_core::ChanceDist {
        debug_assert!(
            state.pending_food > 0,
            "chance only at AwaitingRespawn states"
        );
        let n = self.free_cell_count(state);
        // Chained birth draws avoid a huge initial combinatorial node. In-game
        // respawns stay combined so Committed{s} retains s complete worlds,
        // rather than branching into s^k partial worlds.
        if state.birth {
            return reinfors_core::ChanceDist::Uniform(n.max(1));
        }
        let placeable = (state.pending_food as usize).min(n);
        let outcomes = (0..placeable)
            .try_fold(1usize, |acc, i| acc.checked_mul(n - i))
            .expect("respawn index space overflows usize (bounded at construction and decode)")
            .max(1);
        reinfors_core::ChanceDist::Uniform(outcomes)
    }

    fn apply_chance_node(
        &self,
        state: &SnakeState,
        outcome: usize,
    ) -> Transition<SnakeState, StepEvent> {
        let n = self.free_cell_count(state);
        if state.birth {
            let drained = n == 0 || state.pending_food == 1;
            let mut out = SnakeState {
                snakes: state.snakes.clone(),
                food: state.food.clone(),
                pending_food: if n == 0 { 0 } else { state.pending_food - 1 },
                birth: !drained,
            };
            if n > 0 {
                let cell = self.nth_free_cell(&out, outcome);
                out.food.insert(cell);
            }
            return Transition::silent(out, self.num_snakes);
        }
        let placeable = (state.pending_food as usize).min(n);
        let mut out = SnakeState {
            snakes: state.snakes.clone(),
            food: state.food.clone(),
            pending_food: 0,
            birth: false,
        };
        let mut digits = vec![0usize; placeable];
        let mut rem = outcome;
        // Mixed-radix digits index ordered draws without replacement.
        for i in (0..placeable).rev() {
            let base = n - i;
            digits[i] = rem % base;
            rem /= base;
        }
        for &d in &digits {
            let cell = self.nth_free_cell(&out, d);
            out.food.insert(cell);
        }
        Transition::silent(out, self.num_snakes)
    }

    fn initial_state(&self) -> SnakeState {
        SnakeState {
            snakes: self.initial_snakes(),
            food: HashSet::new(),
            pending_food: self.initial_food_count as u32,
            birth: self.initial_food_count > 0,
        }
    }

    fn truncation_horizon(&self) -> Option<usize> {
        self.max_ticks
    }

    fn mark_truncation(&self, state: &SnakeState, trace: &mut Vec<(usize, StepEvent)>) {
        for (agent, event) in trace.iter_mut() {
            event.survived_to_max_ticks = state.snakes[*agent].alive;
        }
    }
}

#[cfg(test)]
mod game_tests {
    use super::*;

    const G: i32 = 8;

    fn reward() -> SnakeReward {
        SnakeReward {
            step: 0.0,
            food: 1.0,
            loss: -10.0,
            draw: -5.0,
            kill: 5.0,
            win: 10.0,
            survival: 0.0,
        }
    }

    fn game() -> Snake {
        Snake {
            num_snakes: 2,
            grid_size: G,
            initial_length: 3,
            play_to_last: false,
            win_food_lead: None,
            initial_food_count: 1,
            max_ticks: None,
        }
    }

    #[test]
    fn validate_screens_placement_food_and_bounds() {
        let cfg = |grid_size, initial_length, food| Snake {
            num_snakes: 2,
            grid_size,
            initial_length,
            play_to_last: false,
            win_food_lead: None,
            initial_food_count: food,
            max_ticks: None,
        };
        assert!(cfg(8, 3, 3).validate().is_ok());
        assert!(cfg(3, 3, 3).validate().is_ok());
        assert!(cfg(2, 3, 1).validate().is_err());
        assert!(cfg(0, 1, 0).validate().is_err());
        assert!(cfg(-4, 3, 1).validate().is_err());
        assert!(cfg(8, 0, 1).validate().is_err());
        assert!(cfg(8, 100, 1).validate().is_err());
        assert!(cfg(3, 3, 4).validate().is_err());
        assert!(cfg(8, 3, 1usize << 63).validate().is_err());
        assert!(cfg(46_000, 3, 1).validate().is_err());
        assert!(cfg(1_500_000_000, 3, 1).validate().is_err());
        assert!(cfg(i32::MAX, 3, 1).validate().is_err());

        let overlap = cfg(8, 16, 1).validate();
        assert!(overlap.is_err() && overlap.unwrap_err().contains("overlap"));
        assert!(Snake {
            win_food_lead: Some(0),
            ..cfg(8, 3, 1)
        }
        .validate()
        .is_err());
    }

    struct TestRng(u64);
    impl Rng for TestRng {
        fn below(&mut self, n: usize) -> usize {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 33) as usize) % n.max(1)
        }
        fn unit(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    fn initial_state(food: &[Cell]) -> SnakeState {
        SnakeState {
            snakes: game().initial_snakes(),
            food: food.iter().copied().collect(),
            pending_food: 0,
            birth: false,
        }
    }

    fn empty_cells(snakes: &[SnakeBody], food: &HashSet<Cell>, grid_size: i32) -> Vec<Cell> {
        let mut occupied: HashSet<Cell> = food.clone();
        for s in snakes {
            occupied.extend(s.body.iter().copied());
        }
        (0..grid_size)
            .flat_map(|r| (0..grid_size).map(move |c| (r, c)))
            .filter(|cell| !occupied.contains(cell))
            .collect()
    }

    #[test]
    fn step_env_realizes_the_declared_chance() {
        let g = game();
        let st = initial_state(&[(4, 3)]);
        let actions = [0usize, 0];
        let t = g.step(&st, &actions);
        assert!(
            matches!(g.actor(&t.next_state), reinfors_core::Actor::Chance),
            "an eaten apple lands on the AwaitingRespawn chance state"
        );
        let dist = g.chance_node(&t.next_state);
        let realized = reinfors_core::game::step_env(&g, &st, &actions, &mut TestRng(42));
        let trace_events: Vec<Option<StepEvent>> = realized
            .trace
            .iter()
            .map(|(_, e)| Some(e.clone()))
            .collect();
        assert_eq!(trace_events, t.events);
        assert_eq!(realized.terminal, t.terminal);
        assert!(
            (0..dist.enumerable_count().unwrap())
                .any(|d| realized.next_state == g.apply_chance_node(&t.next_state, d).next_state),
            "the realized state must be one of the declared outcomes"
        );
        let again = reinfors_core::game::step_env(&g, &st, &actions, &mut TestRng(42));
        assert_eq!(
            realized.next_state, again.next_state,
            "same seed, same realization"
        );
        assert!(
            (reward().step_reward(&realized.trace[0].1, 0) - 1.0).abs() < 1e-12,
            "A ate one apple"
        );
        assert_eq!(
            realized.next_state.food.len(),
            1,
            "respawn restored the count"
        );
    }

    #[test]
    fn no_eat_is_deterministic() {
        let g = game();
        let st = initial_state(&[(0, 0)]);
        for actions in [[0usize, 0], [1, 2], [2, 1], [0, 2]] {
            let t = g.step(&st, &actions);
            assert_eq!(t.next_state.pending_food, 0, "actions {actions:?}");
            let realized = reinfors_core::game::step_env(&g, &st, &actions, &mut TestRng(1));
            assert_eq!(realized.next_state, t.next_state, "actions {actions:?}");
            let trace_events: Vec<Option<StepEvent>> = realized
                .trace
                .iter()
                .map(|(_, e)| Some(e.clone()))
                .collect();
            assert_eq!(trace_events, t.events);
            assert_eq!(realized.terminal, t.terminal);
        }
    }

    #[test]
    fn realized_respawns_vary_and_land_on_empty_cells() {
        let g = game();
        let st = initial_state(&[(4, 3)]);
        let mut rng = TestRng(7);
        let samples: Vec<SnakeState> = (0..20)
            .map(|_| reinfors_core::game::step_env(&g, &st, &[0, 0], &mut rng).next_state)
            .collect();
        let occupied: std::collections::HashSet<Cell> = samples[0]
            .snakes
            .iter()
            .flat_map(|s| s.body.iter().copied())
            .collect();
        for s in &samples {
            assert_eq!(s.food.len(), 1, "respawn restored the apple count");
            let cell = *s.food.iter().next().unwrap();
            assert!(!occupied.contains(&cell), "apple spawns on an empty cell");
        }
        let distinct: std::collections::HashSet<Cell> = samples
            .iter()
            .map(|s| *s.food.iter().next().unwrap())
            .collect();
        assert!(
            distinct.len() > 1,
            "uniform realization should vary across draws"
        );
    }

    #[test]
    fn declared_chance_is_uniform_over_the_empty_cell_oracle() {
        let g = game();
        let st = initial_state(&[(4, 3)]);
        let t = g.step(&st, &[0, 0]);
        let dist = g.chance_node(&t.next_state);
        let empties = empty_cells(&t.next_state.snakes, &t.next_state.food, G);
        assert_eq!(
            dist,
            reinfors_core::ChanceDist::Uniform(empties.len()),
            "one uniform outcome per empty cell"
        );
        for (i, &cell) in empties.iter().enumerate() {
            let realized = g.apply_chance_node(&t.next_state, i);
            assert_eq!(realized.next_state.pending_food, 0);
            assert!(realized.events.iter().all(Option::is_none));
            let placed: Vec<Cell> = realized
                .next_state
                .food
                .difference(&t.next_state.food)
                .copied()
                .collect();
            assert_eq!(
                placed,
                vec![cell],
                "outcome {i} is the oracle's cell {cell:?}"
            );
        }
    }

    #[test]
    fn encoder_matches_egocentric() {
        let enc = EgocentricSnake { grid_size: G };
        let st = initial_state(&[(4, 3)]);
        for agent in 0..2 {
            assert_eq!(
                enc.encode(&st, agent),
                egocentric_parts(&st.snakes, &st.food, G, agent)
            );
        }
    }

    #[test]
    fn legal_actions_and_metadata() {
        let g = game();
        let st = initial_state(&[(4, 3)]);
        assert_eq!(g.num_agents(), 2);
        assert_eq!(g.action_count(), 3);
        assert_eq!(
            EgocentricSnake { grid_size: G }.obs_shape(),
            (N_CHANNELS, G as usize, G as usize)
        );
        assert_eq!(g.actor(&st), Actor::Simultaneous);
        assert_eq!(g.legal_actions(&st, 0), vec![0, 1, 2]);
        let mut dead = st.clone();
        dead.snakes[1].alive = false;
        assert!(g.legal_actions(&dead, 1).is_empty());
    }

    #[test]
    fn births_are_a_declared_root_respawn_phase() {
        let mut g = game();
        g.initial_food_count = 3;
        let root = g.initial_state();
        assert_eq!(root.pending_food, 3);
        assert!(root.food.is_empty());
        assert!(matches!(g.actor(&root), reinfors_core::Actor::Chance));
        let a = reinfors_core::realize_initial_state(&g, &mut TestRng(9));
        assert_eq!(a.pending_food, 0);
        assert_eq!(a.food.len(), 3);
        let mut wide = game();
        wide.grid_size = 8;
        wide.initial_food_count = 10;
        wide.validate().unwrap();
        let b = reinfors_core::realize_initial_state(&wide, &mut TestRng(11));
        assert_eq!(b.food.len(), 10);
        assert_eq!(b.pending_food, 0);
        assert_eq!(a.snakes, game().initial_snakes());
        let occupied: std::collections::HashSet<Cell> = a
            .snakes
            .iter()
            .flat_map(|s| s.body.iter().copied())
            .collect();
        assert!(a.food.iter().all(|c| !occupied.contains(c)));
    }

    #[test]
    fn step_env_realizes_the_move_plus_rng_respawn() {
        let g = game();
        let r = reward();
        let st = initial_state(&[(4, 3)]);
        let t = reinfors_core::game::step_env(&g, &st, &[0, 0], &mut TestRng(1));
        assert!(
            (r.step_reward(&t.trace[0].1, 0) - 1.0).abs() < 1e-12,
            "A ate -> food reward"
        );
        assert!(!t.terminal);
        assert_eq!(
            t.next_state.food.len(),
            1,
            "respawn restored the apple count"
        );
        let empty = initial_state(&[]);
        let t2 = reinfors_core::game::step_env(&g, &empty, &[0, 0], &mut TestRng(1));
        assert_eq!(
            [
                r.step_reward(&t2.trace[0].1, 0),
                r.step_reward(&t2.trace[1].1, 1)
            ],
            [0.0, 0.0]
        );
    }

    #[test]
    fn mark_truncation_pays_survival_to_the_living_only() {
        let r = SnakeReward {
            survival: 0.25,
            ..reward()
        };
        let mut st = initial_state(&[]);
        st.snakes[1].alive = false;
        let mut trace = vec![(0, StepEvent::default()), (1, StepEvent::default())];
        game().mark_truncation(&st, &mut trace);
        assert!(trace[0].1.survived_to_max_ticks && !trace[1].1.survived_to_max_ticks);
        assert!((r.step_reward(&trace[0].1, 0) - 0.25).abs() < 1e-12);
        assert_eq!(r.step_reward(&trace[1].1, 1), 0.0);
    }

    #[test]
    fn snake_length_cell_is_symmetric_off_diagonal_and_skips_dead() {
        let mk = |la: usize, lb: usize, b_alive: bool| {
            let body = |len: usize| (0..len as i32).map(|c| (0, c)).collect::<VecDeque<Cell>>();
            SnakeState {
                snakes: vec![
                    SnakeBody {
                        body: body(la),
                        direction: Action::Right,
                        alive: true,
                    },
                    SnakeBody {
                        body: body(lb),
                        direction: Action::Left,
                        alive: b_alive,
                    },
                ],
                food: HashSet::new(),
                pending_food: 0,
                birth: false,
            }
        };
        let lopsided = snake_length_cell(&mk(21, 6, true)).unwrap();
        assert_eq!(Some(lopsided), snake_length_cell(&mk(6, 21, true)));
        assert_ne!(
            lopsided >> 16,
            lopsided & 0xFFFF,
            "lopsided -> off-diagonal cell"
        );
        let sym = snake_length_cell(&mk(9, 9, true)).unwrap();
        assert_eq!(sym >> 16, sym & 0xFFFF, "equal lengths -> diagonal cell");
        assert_eq!(snake_length_cell(&mk(9, 9, false)), None);
    }
}

#[cfg(test)]
mod env_tests {
    use super::*;

    fn game(grid_size: i32, win_food_lead: Option<usize>) -> Snake {
        Snake {
            num_snakes: 2,
            grid_size,
            initial_length: 3,
            play_to_last: false,
            win_food_lead,
            initial_food_count: 0,
            max_ticks: None,
        }
    }

    #[test]
    fn initial_placement_matches_oracle() {
        let snakes = game(20, None).initial_snakes();
        assert_eq!(
            Vec::from(snakes[A].body.clone()),
            vec![(10, 6), (10, 5), (10, 4)]
        );
        assert_eq!(snakes[A].direction, Action::Right);
        assert_eq!(
            Vec::from(snakes[B].body.clone()),
            vec![(10, 14), (10, 15), (10, 16)]
        );
        assert_eq!(snakes[B].direction, Action::Left);
    }

    #[test]
    fn coast_step_moves_head_and_pops_tail() {
        let g = game(20, None);
        let mut snakes = g.initial_snakes();
        let mut food = HashSet::new();
        let (events, done) = g.advance(
            &mut snakes,
            &mut food,
            &[Some(Action::Right), Some(Action::Left)],
            || None,
        );
        assert_eq!(
            Vec::from(snakes[A].body.clone()),
            vec![(10, 7), (10, 6), (10, 5)]
        );
        assert_eq!(
            Vec::from(snakes[B].body.clone()),
            vec![(10, 13), (10, 14), (10, 15)]
        );
        assert!(!events[A].died && !events[B].died && !done);
    }

    #[test]
    fn reverse_action_coasts() {
        let g = game(20, None);
        let mut snakes = g.initial_snakes();
        let mut food = HashSet::new();
        g.advance(&mut snakes, &mut food, &[Some(Action::Left), None], || None);
        assert_eq!(snakes[A].head(), (10, 7));
        assert_eq!(snakes[A].direction, Action::Right);
    }

    #[test]
    fn head_on_collision_is_a_draw() {
        let g = game(20, None);
        let mut snakes = g.initial_snakes();
        let mut food = HashSet::new();
        let (mut events, mut done) = (Default::default(), false);
        for _ in 0..4 {
            (events, done) = g.advance(
                &mut snakes,
                &mut food,
                &[Some(Action::Right), Some(Action::Left)],
                || None,
            );
        }
        assert_eq!(events[A].death_cause, Some(DeathCause::HeadOn));
        assert_eq!(events[B].death_cause, Some(DeathCause::HeadOn));
        assert!(events[A].drew && events[B].drew && done);
    }

    #[test]
    fn eating_grows_snake_and_spawns_replacement() {
        let g = game(20, None);
        let mut snakes = g.initial_snakes();
        let mut food = HashSet::from([(10, 7)]);
        let mut replacement = vec![(0, 0)];
        let (events, _) = g.advance(
            &mut snakes,
            &mut food,
            &[Some(Action::Right), Some(Action::Left)],
            || replacement.pop(),
        );
        assert!(events[A].ate_food);
        assert_eq!(snakes[A].len(), 4);
        assert!(food.contains(&(0, 0)) && !food.contains(&(10, 7)));
    }

    #[test]
    fn wall_death_when_running_off_grid() {
        let g = game(20, None);
        let mut snakes = g.initial_snakes();
        let mut food = HashSet::new();
        snakes[A].body = VecDeque::from([(0, 19), (0, 18), (0, 17)]);
        snakes[A].direction = Action::Right;
        let (events, _) = g.advance(&mut snakes, &mut food, &[Some(Action::Right), None], || {
            None
        });
        assert_eq!(events[A].death_cause, Some(DeathCause::Wall));
        assert!(!snakes[A].alive);
    }

    #[test]
    fn self_body_collision_is_a_death() {
        let g = game(20, None);
        let mut snakes = g.initial_snakes();
        let mut food = HashSet::new();
        snakes[A].body = VecDeque::from([(5, 5), (5, 4), (4, 4), (4, 5), (4, 6)]);
        snakes[A].direction = Action::Right;
        let (events, _) = g.advance(&mut snakes, &mut food, &[Some(Action::Up), None], || None);
        assert_eq!(events[A].death_cause, Some(DeathCause::SelfBody));
        assert!(!snakes[A].alive);
    }

    #[test]
    fn opponent_body_collision_is_a_kill_and_win() {
        let g = game(20, None);
        let mut snakes = g.initial_snakes();
        let mut food = HashSet::new();
        snakes[A].body = VecDeque::from([(10, 5), (10, 4), (10, 3)]);
        snakes[A].direction = Action::Right;
        snakes[B].body = VecDeque::from([(9, 5), (8, 5), (7, 5)]);
        snakes[B].direction = Action::Down;
        let (events, done) = g.advance(
            &mut snakes,
            &mut food,
            &[Some(Action::Right), Some(Action::Down)],
            || None,
        );
        assert_eq!(events[B].death_cause, Some(DeathCause::OppBody));
        assert!(events[A].killed_opponent && events[A].won && events[B].lost && done);
    }

    #[test]
    fn food_lead_wins_outright() {
        let g = game(20, Some(2));
        let mut snakes = g.initial_snakes();
        let mut food = HashSet::new();
        snakes[A].body = VecDeque::from([(2, 5), (2, 4), (2, 3), (2, 2), (2, 1)]);
        snakes[A].direction = Action::Right;
        let (events, done) =
            g.advance(&mut snakes, &mut food, &[Some(Action::Right), None], || {
                None
            });
        assert!(events[A].won && events[B].lost && done);
    }
}

#[cfg(test)]
mod obs_tests {
    use super::*;

    fn at(obs: &[f32], g: i32, ch: usize, r: i32, c: i32) -> f32 {
        obs[ch * (g * g) as usize + (r as usize) * (g as usize) + (c as usize)]
    }

    fn snakes() -> Vec<SnakeBody> {
        Snake {
            num_snakes: 2,
            grid_size: 20,
            initial_length: 3,
            play_to_last: false,
            win_food_lead: None,
            initial_food_count: 0,
            max_ticks: None,
        }
        .initial_snakes()
    }

    #[test]
    fn egocentric_rotates_by_heading() {
        let obs = egocentric_parts(&snakes(), &HashSet::new(), 20, A);
        assert_eq!(at(&obs, 20, CH_OWN_HEAD, 13, 10), 1.0);
        assert_eq!(at(&obs, 20, CH_OWN_BODY, 14, 10), 1.0);
        assert_eq!(at(&obs, 20, CH_OWN_BODY, 15, 10), 1.0);
        assert_eq!(at(&obs, 20, CH_OWN_BODY, 13, 10), 0.0);
        assert_eq!(at(&obs, 20, CH_OPP_HEAD, 5, 10), 1.0);
    }

    #[test]
    fn food_lands_in_food_channel_rotated() {
        let food = HashSet::from([(10, 6)]);
        let obs = egocentric_parts(&snakes(), &food, 20, A);
        assert_eq!(at(&obs, 20, CH_FOOD, 13, 10), 1.0);
    }

    #[test]
    fn buffer_has_expected_length() {
        assert_eq!(
            egocentric_parts(&snakes(), &HashSet::new(), 20, A).len(),
            N_CHANNELS * 20 * 20
        );
    }
}

#[cfg(test)]
mod reward_tests {
    use super::*;

    fn reward() -> SnakeReward {
        SnakeReward {
            step: 0.0,
            food: 0.1,
            loss: -0.5,
            draw: -0.25,
            kill: 0.5,
            win: 0.0,
            survival: 0.25,
        }
    }

    #[test]
    fn step_reward_scores_events_not_survival() {
        let r = reward();
        assert_eq!(r.step_reward(&StepEvent::default(), 0), 0.0);
        let ate = StepEvent {
            ate_food: true,
            ..Default::default()
        };
        assert!((r.step_reward(&ate, 0) - 0.1).abs() < 1e-12);
    }

    #[test]
    fn a_dead_snake_scores_only_its_terminal_outcome() {
        let r = reward();
        let dead = StepEvent {
            died: true,
            lost: true,
            ..Default::default()
        };
        assert!((r.step_reward(&dead, 0) - (-0.5)).abs() < 1e-12);
    }
}

impl reinfors_core::StateCodec for Snake {
    type State = SnakeState;

    fn encode(&self, s: &SnakeState) -> Vec<u8> {
        // Layout 5 widened pending_food to u32 (4 introduced it as u8).
        crate::codec_util::serde_encode(5, s)
    }

    fn decode(&self, bytes: &[u8]) -> Result<SnakeState, String> {
        crate::codec_util::serde_decode(5, bytes)
    }

    fn validate_decoded_state(&self, state: &SnakeState, done: bool) -> Result<(), String> {
        if state.pending_food != 0 || state.birth {
            return Err("a live position cannot await a respawn".to_string());
        }
        if state.snakes.len() != self.num_snakes {
            return Err(format!(
                "state has {} snakes; this game has {}",
                state.snakes.len(),
                self.num_snakes
            ));
        }
        let g = self.grid_size;
        let in_grid = |cell: Cell| 0 <= cell.0 && cell.0 < g && 0 <= cell.1 && cell.1 < g;
        for (i, snake) in state.snakes.iter().enumerate() {
            if snake.alive && snake.body.is_empty() {
                return Err(format!("snake {i} is alive with an empty body"));
            }
            for &cell in &snake.body {
                if !in_grid(cell) {
                    return Err(format!("snake {i} cell {cell:?} outside the grid"));
                }
            }
        }
        for &cell in &state.food {
            if !in_grid(cell) {
                return Err(format!("food cell {cell:?} outside the grid"));
            }
        }
        let k = self.num_snakes.min(state.food.len()) as u128;
        let cells = self.grid_size.max(0) as u128 * self.grid_size.max(0) as u128;
        let worst: u128 = (0..k)
            .map(|i| cells.saturating_sub(i))
            .try_fold(1u128, |acc, f| acc.checked_mul(f))
            .unwrap_or(u128::MAX);
        // Outcome indices cross the f64 seam; beyond 2^53 distinct integers would alias.
        if worst > (1u128 << 53).min(usize::MAX as u128) {
            return Err(format!(
                "{} food cells with {} snakes imply a respawn index space past \
                 min(2^53, usize::MAX)",
                state.food.len(),
                self.num_snakes
            ));
        }
        let terminal = self.is_terminal(&state.snakes);
        if terminal != done {
            return Err(format!(
                "snakes imply terminal={terminal}, but envelope done is {done}"
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod n_player_tests {
    use super::*;

    fn game(num_snakes: usize, grid: i32, food: usize) -> Snake {
        Snake {
            num_snakes,
            grid_size: grid,
            initial_length: 2,
            play_to_last: false,
            win_food_lead: None,
            initial_food_count: food,
            max_ticks: None,
        }
    }

    #[test]
    fn validation_bounds_the_config() {
        assert!(game(3, 8, 2).validate().is_ok());
        assert!(game(8, 12, 2).validate().is_ok());
        assert!(game(1, 8, 2).validate().is_err());
        assert!(game(9, 20, 2).validate().is_err());
        let mut lead = game(3, 8, 2);
        lead.win_food_lead = Some(2);
        assert!(lead.validate().is_err(), "the lead rule is two-snake only");
        assert!(game(3, 20, 3).validate().is_ok());
        assert!(game(2, 8, 10).validate().is_ok());
        assert!(game(2, 4, 17).validate().is_err(), "more food than cells");
        assert!(
            game(3, 1000, 3).validate().is_err(),
            "a 1000-grid triple-eat's combined index (~1e18) is past 2^53"
        );
        assert!(
            game(2, 200, 10_001).validate().is_err(),
            "a birth chain past CHANCE_CHAIN_LIMIT edges could never realize"
        );
    }

    #[test]
    fn three_snakes_place_disjoint_and_in_grid() {
        for g in [6, 8, 12] {
            for n in [3, 4, 5] {
                let snakes = game(n, g, 1).initial_snakes();
                assert_eq!(snakes.len(), n);
                let mut seen = HashSet::new();
                for s in &snakes {
                    assert_eq!(s.len(), 2);
                    for &(r, c) in &s.body {
                        assert!(0 <= r && r < g && 0 <= c && c < g);
                        assert!(seen.insert((r, c)), "snakes overlap at ({r}, {c})");
                    }
                }
            }
        }
    }

    #[test]
    fn two_snake_placement_is_the_legacy_layout() {
        let snakes = game(2, 8, 1).initial_snakes();
        assert_eq!(snakes[0].head(), (4, 2));
        assert_eq!(snakes[0].direction, Action::Right);
        assert_eq!(snakes[1].head(), (4, 6));
        assert_eq!(snakes[1].direction, Action::Left);
    }

    fn body_at(cells: &[Cell], dir: Action) -> SnakeBody {
        SnakeBody {
            body: VecDeque::from(cells.to_vec()),
            direction: dir,
            alive: true,
        }
    }

    #[test]
    fn three_way_head_on_draws_everyone() {
        let g = game(3, 8, 0);
        let mut snakes = vec![
            body_at(&[(4, 3)], Action::Right),
            body_at(&[(4, 5)], Action::Left),
            body_at(&[(3, 4)], Action::Down),
        ];
        let mut food = HashSet::new();
        let (events, done) = g.advance(
            &mut snakes,
            &mut food,
            &[Some(Action::Right), Some(Action::Left), Some(Action::Down)],
            || None,
        );
        assert!(done);
        for e in &events {
            assert!(e.died && e.drew && !e.lost && !e.won);
            assert_eq!(e.death_cause, Some(DeathCause::HeadOn));
        }
    }

    #[test]
    fn deaths_with_survivors_lose_and_the_last_one_standing_wins() {
        let g = game(3, 8, 0);
        let mut snakes = vec![
            body_at(&[(0, 0)], Action::Up),
            body_at(&[(4, 4)], Action::Right),
            body_at(&[(6, 6)], Action::Right),
        ];
        let mut food = HashSet::new();
        let (events, done) = g.advance(
            &mut snakes,
            &mut food,
            &[Some(Action::Up), Some(Action::Right), Some(Action::Right)],
            || None,
        );
        assert!(!done, "two snakes still alive");
        assert!(events[0].died && events[0].lost);
        assert!(!events[1].won && !events[2].won);
        let orbit = [Action::Down, Action::Left, Action::Up, Action::Right];
        let mut events;
        let mut done;
        let mut step = 0;
        loop {
            let out = g.advance(
                &mut snakes,
                &mut food,
                &[None, Some(Action::Up), Some(orbit[step % 4])],
                || None,
            );
            events = out.0;
            done = out.1;
            step += 1;
            if done || step > 10 {
                break;
            }
        }
        assert!(done);
        assert!(events[2].won, "lone survivor of the deadly tick wins");
        assert!(events[1].died && events[1].lost);
    }

    #[test]
    fn k_eater_respawn_matches_the_ordered_tuple_layout() {
        let g = game(3, 6, 3);
        let mut state = g.initial_state();
        state.pending_food = 0;
        state.food.clear();
        let mut expected_heads = Vec::new();
        for s in &state.snakes {
            let (dr, dc) = s.direction.delta();
            let (hr, hc) = s.head();
            state.food.insert((hr + dr, hc + dc));
            expected_heads.push((hr + dr, hc + dc));
        }
        let t = g.step(&state, &[0, 0, 0]);
        assert!(state.food.len() == 3 && t.next_state.food.is_empty());
        assert_eq!(t.next_state.pending_food, 3);
        assert!(
            !t.next_state.birth,
            "in-game respawns are combined, not chained"
        );
        let dist = g.chance_node(&t.next_state);
        let n = (6 * 6)
            - t.next_state
                .snakes
                .iter()
                .map(|s| s.body.len())
                .sum::<usize>() as i32;
        let n = n as usize;
        assert_eq!(
            dist.enumerable_count().unwrap(),
            n * (n - 1) * (n - 2),
            "ONE combined draw: a Committed search keeps s complete worlds, not s^k branches"
        );
        let s0 = g.apply_chance_node(&t.next_state, 0).next_state;
        assert_eq!(s0.food.len(), 3);
        assert_eq!(s0.pending_food, 0);
        let last = g
            .apply_chance_node(&t.next_state, dist.enumerable_count().unwrap() - 1)
            .next_state;
        assert_eq!(last.food.len(), 3);
        assert_ne!(s0.food, last.food);
    }

    #[test]
    fn a_pending_respawn_state_is_not_restorable() {
        use reinfors_core::StateCodec;
        let g = game(2, 6, 1);
        let mut st = SnakeState {
            snakes: vec![
                body_at(&[(0, 0)], Action::Right),
                body_at(&[(3, 3)], Action::Right),
            ],
            food: std::iter::once((5, 5)).collect(),
            pending_food: 0,
            birth: false,
        };
        g.validate_decoded_state(&st, false).unwrap();
        st.pending_food = 1;
        let err = g.validate_decoded_state(&st, false).unwrap_err();
        assert!(err.contains("await"), "{err}");
        let decoded = g.decode(&g.encode(&st)).unwrap();
        assert_eq!(decoded.pending_food, 1);
        assert!(g.validate_decoded_state(&decoded, false).is_err());
    }

    #[test]
    fn decoded_food_counts_past_the_index_space_reject() {
        use reinfors_core::StateCodec;
        let g = game(8, 20, 1);
        assert!(g.validate().is_ok());
        let snakes: Vec<SnakeBody> = (0..8)
            .map(|i| body_at(&[(2 * i, 0)], Action::Right))
            .collect();
        let ok_state = SnakeState {
            snakes: snakes.clone(),
            food: (0..3).map(|c| (1, c)).collect(),
            pending_food: 0,
            birth: false,
        };
        g.validate_decoded_state(&ok_state, false).unwrap();
        let over = SnakeState {
            snakes,
            food: (0..8).map(|c| (1, c)).collect(),
            pending_food: 0,
            birth: false,
        };
        assert!(g
            .validate_decoded_state(&over, false)
            .unwrap_err()
            .contains("index space"));
    }

    #[test]
    fn length_cell_keys_are_legacy_at_two_and_sorted_multisets_past() {
        let two = SnakeState {
            snakes: vec![
                body_at(&[(1, 1)], Action::Up),
                body_at(&[(2, 2), (2, 3), (2, 4)], Action::Up),
            ],
            food: HashSet::new(),
            pending_food: 0,
            birth: false,
        };
        assert_eq!(snake_length_cell(&two), Some(1));
        let three = SnakeState {
            snakes: vec![
                body_at(&[(1, 1), (1, 2), (1, 3)], Action::Up),
                body_at(&[(3, 3)], Action::Up),
                body_at(&[(5, 5)], Action::Up),
            ],
            food: HashSet::new(),
            pending_food: 0,
            birth: false,
        };
        assert_eq!(snake_length_cell(&three), Some(1));
        let mut dead = three.clone();
        dead.snakes[1].alive = false;
        assert_eq!(snake_length_cell(&dead), None);
    }
}
