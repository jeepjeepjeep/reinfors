//! The two-player snake game, in one self-contained module (mirroring `connect4.rs`): grid actions,
//! the egocentric observation encoder, reward shaping, and the `Snake` adapter implementing
//! `reinfors_core::Game` (its deterministic dynamics live in private methods, alongside the
//! `EgocentricSnake` state encoder).
//!
//! The dynamics are deterministic integer arithmetic with food placement injected, so a given set of
//! actions + spawns yields bit-identical trajectories — the basis for the native regression tests.

use std::collections::{HashMap, HashSet, VecDeque};

use reinfors_core::game::{Actor, Game, Rng, Transition};
use reinfors_core::{Reward, Space, StateEncoder};

pub type Cell = (i32, i32);

/// Player A is index 0, player B is index 1 (matching the Python insertion order).
pub const A: usize = 0;
pub const B: usize = 1;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DeathCause {
    Wall,
    SelfBody,
    OppBody,
    HeadOn,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SnakeBody {
    pub body: VecDeque<Cell>, // body[0] is the head, body[len-1] the tail
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

/// Per-snake outcome of one tick, mirroring `snake_RL`'s `StepEvent`.
#[derive(Clone, Default, Debug, PartialEq, Eq)]
pub struct StepEvent {
    pub ate_food: bool,
    pub died: bool,
    pub death_cause: Option<DeathCause>,
    pub killed_opponent: bool,
    pub won: bool,
    pub lost: bool,
    pub drew: bool,
    /// Set by the rollout (via `Game::mark_truncation`) on a truncation tick the snake reached alive,
    /// so its reward pays the `survival` bonus there. Never set by `advance` or the search.
    pub survived_to_max_ticks: bool,
}

// `SnakeEnv` is gone: snake's dynamics are methods on `Snake` (which holds the config), operating on
// a working `(snakes, food)` state, mirroring how `Connect4` keeps its dynamics in private methods.
// These live in the `impl Snake` block alongside the `Game` adapter, further down the file.

// ============================ Grid actions ============================

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum Action {
    Up,
    Down,
    Left,
    Right,
}

impl Action {
    /// (row, col) step. Row grows downward, matching the Python grid convention.
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

    /// CCW quarter-turns that bring this heading to "up", for the egocentric observation
    /// (matches `_EGO_ROT_K`: Up=0, Right=1, Down=2, Left=3).
    pub fn ego_rot_k(self) -> u8 {
        match self {
            Action::Up => 0,
            Action::Right => 1,
            Action::Down => 2,
            Action::Left => 3,
        }
    }

    /// Clockwise quarter-turn of this heading (matches `_CW`).
    fn cw(self) -> Action {
        match self {
            Action::Up => Action::Right,
            Action::Right => Action::Down,
            Action::Down => Action::Left,
            Action::Left => Action::Up,
        }
    }

    /// Counter-clockwise quarter-turn (matches `_CCW`).
    fn ccw(self) -> Action {
        match self {
            Action::Up => Action::Left,
            Action::Left => Action::Down,
            Action::Down => Action::Right,
            Action::Right => Action::Up,
        }
    }
}

/// Heading-relative action (matches `snake_RL`'s `RelativeAction`); never produces the reverse.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RelativeAction {
    Forward,
    Left,
    Right,
}

/// Same order as `snake_RL`'s `RELATIVE_ACTIONS` (Forward, Left, Right).
pub const RELATIVE_ACTIONS: [RelativeAction; 3] = [
    RelativeAction::Forward,
    RelativeAction::Left,
    RelativeAction::Right,
];

/// Resolve a relative action to an absolute heading (matches `relative_to_absolute`).
pub fn relative_to_absolute(heading: Action, rel: RelativeAction) -> Action {
    match rel {
        RelativeAction::Forward => heading,
        RelativeAction::Left => heading.ccw(),
        RelativeAction::Right => heading.cw(),
    }
}

// ===================== Egocentric observation =====================
pub const N_CHANNELS: usize = 5;
const CH_OWN_HEAD: usize = 0;
const CH_OWN_BODY: usize = 1;
const CH_OPP_HEAD: usize = 2;
const CH_OPP_BODY: usize = 3;
const CH_FOOD: usize = 4;

/// Build the egocentric observation for `agent` (0 = A, 1 = B) as a flat `[5 * g * g]` f32 buffer,
/// from a `(snakes, food)` state. Coordinates are pre-rotated so the queried snake faces "up".
pub fn egocentric_parts(
    snakes: &[SnakeBody; 2],
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
        let mut ch = head_ch; // head is body[0]; everything after lands in the body channel
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

// ========================= Reward shaping =========================
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
            reward += self.loss; // lost while alive: out-eaten under win_food_lead
        }
        if e.survived_to_max_ticks {
            reward += self.survival; // set by `Snake::mark_truncation` on a truncation tick, if alive
        }
        reward
    }
}

// ========= The `Snake` Game adapter + `EgocentricSnake` encoder =========

/// Snake's dynamic state: the two snakes and the food. Static config (grid size, rules) lives on
/// `Snake` and the reward on the decoupled `SnakeReward`, so the search/engine carry just this per node.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SnakeState {
    pub snakes: [SnakeBody; 2],
    pub food: HashSet<Cell>,
}

// Coarsening for the reached-state buffer: bucket a snake length, and how many buckets. Difficulty
// tracks length, so a fixed linear bucketing spreads start coverage from early to late game; very-late
// lengths saturate the last bucket. Fixed in v1 (a configurable coarsening is a future refinement).
const CELL_BUCKET_SIZE: usize = 3;
const CELL_N_BUCKETS: u64 = 10;

/// A start-state-buffer coverage cell for a snake state: the **unordered** pair of the two snakes'
/// length buckets, packed into a `u64`. The pair is unordered because self-play is symmetric — one
/// lopsided rollout already yields both perspectives, so `(20, 4)` and `(4, 20)` are the same cell
/// (an off-diagonal one). Returns `None` for a state with a dead snake (the episode is effectively
/// over, so it is not a valid restart point). This is snake's `cell_key` for
/// [`ReachedStateBuffer`](reinfors_core::ReachedStateBuffer); another game supplies its own.
pub fn snake_length_cell(state: &SnakeState) -> Option<u64> {
    if !state.snakes[0].alive || !state.snakes[1].alive {
        return None;
    }
    let bucket = |len: usize| ((len / CELL_BUCKET_SIZE) as u64).min(CELL_N_BUCKETS - 1);
    let (a, b) = (bucket(state.snakes[0].len()), bucket(state.snakes[1].len()));
    let (lo, hi) = (a.min(b), a.max(b));
    Some((lo << 16) | hi)
}

/// The default snake observation: an egocentric 5-channel grid, the searching snake always facing up
/// (see [`egocentric_parts`]). Carries `grid_size` (which lives on `Snake`, not in `SnakeState`).
pub struct EgocentricSnake {
    pub grid_size: i32,
}

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
        Space::unit_box(vec![c, h, w]) // all planes are one-hot occupancy: values in [0, 1]
    }
}

/// Two-player simultaneous-move snake with environment chance (apple respawn). Its deterministic
/// dynamics live in the private `impl Snake` methods below, behind the `Game` trait.
pub struct Snake {
    pub grid_size: i32,
    pub initial_length: usize,
    pub play_to_last: bool,
    pub win_food_lead: Option<usize>,
    pub initial_food_count: usize,
    /// Episode-length cap (snake can run forever); the rollout truncates here, paying the survival
    /// reward to still-alive snakes. `None` = never truncate.
    pub max_ticks: Option<usize>,
}

impl Snake {
    fn in_bounds(&self, (r, c): Cell) -> bool {
        0 <= r && r < self.grid_size && 0 <= c && c < self.grid_size
    }

    /// Initial placement: heads at one-third / two-thirds along the middle row, bodies trailing to the
    /// nearer wall and wrapping (matches snake_RL's `_initial_snakes` / `_trace_body`).
    fn initial_snakes(&self) -> [SnakeBody; 2] {
        let g = self.grid_size;
        let mid = g / 2;
        let a_body = Self::trace_body(
            g,
            (mid, g / 3),
            &[Action::Left, Action::Down, Action::Right, Action::Up],
            self.initial_length,
        );
        let b_body = Self::trace_body(
            g,
            (mid, g - g / 3),
            &[Action::Right, Action::Up, Action::Left, Action::Down],
            self.initial_length,
        );
        [
            SnakeBody {
                body: a_body,
                direction: Action::Right,
                alive: true,
            },
            SnakeBody {
                body: b_body,
                direction: Action::Left,
                alive: true,
            },
        ]
    }

    fn trace_body(grid_size: i32, head: Cell, dirs: &[Action], length: usize) -> VecDeque<Cell> {
        let g = grid_size;
        let mut cells = VecDeque::from([head]);
        let (mut r, mut c) = head;
        let mut d = 0usize;
        while cells.len() < length {
            let (dr, dc) = dirs[d].delta();
            let (nr, nc) = (r + dr, c + dc);
            if !(0 <= nr && nr < g && 0 <= nc && nc < g) {
                d += 1;
                assert!(
                    d < dirs.len(),
                    "initial_length {length} too long for grid_size {g}"
                );
                continue;
            }
            r = nr;
            c = nc;
            cells.push_back((r, c));
        }
        cells
    }

    /// Advance one tick over `(snakes, food)` in place. `actions[i]` is an absolute move for snake `i`
    /// (None = coast in its current heading; a reverse move coasts, as in snake_RL). `next_food` is
    /// called once per eaten apple to supply its replacement cell. Returns the per-snake events and
    /// whether the episode is now over.
    fn advance(
        &self,
        snakes: &mut [SnakeBody; 2],
        food: &mut HashSet<Cell>,
        actions: [Option<Action>; 2],
        mut next_food: impl FnMut() -> Option<Cell>,
    ) -> ([StepEvent; 2], bool) {
        let mut events = [StepEvent::default(), StepEvent::default()];

        // Stage 1: move every living snake. Eating intent keeps the tail; survival is settled below.
        let mut ate_intent = [false, false];
        let mut moved = [false, false];
        for i in 0..2 {
            if !snakes[i].alive {
                continue;
            }
            moved[i] = true;
            let mut action = actions[i].unwrap_or(snakes[i].direction);
            if action.opposite() == snakes[i].direction {
                action = snakes[i].direction; // reverse is a no-op; coast straight
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

        // Stage 2: resolve collisions against the post-move world.
        let causes = self.resolve_collisions(snakes, &moved);
        let mut fatal_heads = [None, None];
        for i in 0..2 {
            if causes[i].is_some() {
                fatal_heads[i] = Some(snakes[i].head());
            }
        }
        for i in 0..2 {
            if let Some(cause) = causes[i] {
                snakes[i].alive = false;
                snakes[i].body.pop_front(); // corpse vacates the collision cell
                events[i].died = true;
                events[i].death_cause = Some(cause);
            }
        }

        // Eating: survivors whose new head landed on food eat it and trigger a replacement spawn.
        for i in 0..2 {
            if ate_intent[i] && causes[i].is_none() {
                let head = snakes[i].head();
                food.remove(&head);
                events[i].ate_food = true;
                if let Some(cell) = next_food() {
                    food.insert(cell);
                }
            }
        }

        // Kill credit: a snake whose body occupies the cell an opponent fatally moved into.
        for i in 0..2 {
            if causes[i] == Some(DeathCause::OppBody) {
                if let Some(killer) = self.find_killer(snakes, fatal_heads[i].unwrap(), i) {
                    events[killer].killed_opponent = true;
                }
            }
        }

        // Outcomes.
        let alive_ids: Vec<usize> = (0..2).filter(|&i| snakes[i].alive).collect();
        let n_causes = causes.iter().filter(|c| c.is_some()).count();
        if alive_ids.len() == 1 && n_causes > 0 {
            events[alive_ids[0]].won = true;
        }
        for i in 0..2 {
            if causes[i].is_some() {
                if !alive_ids.is_empty() {
                    events[i].lost = true;
                } else if n_causes >= 2 {
                    events[i].drew = true;
                }
            }
        }
        let mut done = alive_ids.len() <= if self.play_to_last { 0 } else { 1 };

        // Food-lead win: both alive, leader `win_food_lead` apples (length) ahead wins outright.
        if let Some(lead) = self.win_food_lead {
            if !done && alive_ids.len() >= 2 {
                let (i0, i1) = (alive_ids[0], alive_ids[1]);
                let (leader, runner) = if snakes[i0].len() >= snakes[i1].len() {
                    (i0, i1)
                } else {
                    (i1, i0)
                };
                if snakes[leader].len() - snakes[runner].len() >= lead {
                    events[leader].won = true;
                    events[runner].lost = true;
                    done = true;
                }
            }
        }

        (events, done)
    }

    fn resolve_collisions(
        &self,
        snakes: &[SnakeBody; 2],
        moved: &[bool; 2],
    ) -> [Option<DeathCause>; 2] {
        let mut causes = [None, None];

        // Two heads on one cell die together, whatever else is on it.
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

        for i in 0..2 {
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

    fn find_killer(
        &self,
        snakes: &[SnakeBody; 2],
        fatal_head: Cell,
        victim: usize,
    ) -> Option<usize> {
        (0..2).find(|&j| j != victim && snakes[j].alive && snakes[j].body.contains(&fatal_head))
    }

    /// Spawn one apple at a uniform-random empty cell (the env's true spawn), or nothing if the grid is
    /// full. A single `rng.below(n)` picks `k`, the index of the chosen cell among the empties in
    /// row-major order; the cell itself is found analytically from the sorted occupied indices (each
    /// occupied cell at or below the running target shifts it one further on), so there is no O(g²)
    /// scan over the grid. This selects the same cell a linear walk would for the same `k`.
    fn spawn_one(&self, snakes: &[SnakeBody; 2], food: &mut HashSet<Cell>, rng: &mut dyn Rng) {
        let g = self.grid_size;
        let mut occupied: Vec<usize> = food.iter().map(|&(r, c)| (r * g + c) as usize).collect();
        for s in snakes {
            occupied.extend(s.body.iter().map(|&(r, c)| (r * g + c) as usize));
        }
        occupied.sort_unstable();
        occupied.dedup();
        let n = (g * g) as usize - occupied.len();
        if n == 0 {
            return;
        }
        let mut idx = rng.below(n);
        for &o in &occupied {
            if o <= idx {
                idx += 1;
            } else {
                break;
            }
        }
        food.insert((idx as i32 / g, idx as i32 % g));
    }

    /// Occupied cell ids (food + both bodies) of a state, sorted + deduped — the same set
    /// `spawn_one` skips, so free-cell indices here match its draw exactly.
    fn occupied_cells(&self, state: &SnakeState) -> Vec<usize> {
        let g = self.grid_size;
        let mut occupied: Vec<usize> = state
            .food
            .iter()
            .map(|&(r, c)| (r * g + c) as usize)
            .collect();
        for s in &state.snakes {
            occupied.extend(s.body.iter().map(|&(r, c)| (r * g + c) as usize));
        }
        occupied.sort_unstable();
        occupied.dedup();
        occupied
    }

    fn free_cell_count(&self, state: &SnakeState) -> usize {
        (self.grid_size * self.grid_size) as usize - self.occupied_cells(state).len()
    }

    /// The `i`-th free cell in row-major order — `spawn_one`'s indexing, declared.
    fn nth_free_cell(&self, state: &SnakeState, i: usize) -> Cell {
        let g = self.grid_size;
        let mut idx = i;
        for &o in &self.occupied_cells(state) {
            if o <= idx {
                idx += 1;
            } else {
                break;
            }
        }
        (idx as i32 / g, idx as i32 % g)
    }
}

impl Game for Snake {
    type State = SnakeState;
    type Event = StepEvent;

    fn num_agents(&self) -> usize {
        2
    }

    fn action_count(&self) -> usize {
        RELATIVE_ACTIONS.len()
    }

    fn actor(&self, _state: &SnakeState) -> Actor {
        Actor::Simultaneous
    }

    fn legal_actions(&self, state: &SnakeState, agent: usize) -> Vec<usize> {
        if state.snakes[agent].alive {
            (0..RELATIVE_ACTIONS.len()).collect()
        } else {
            Vec::new()
        }
    }

    fn step(&self, state: &SnakeState, actions: &[usize]) -> Transition<SnakeState, StepEvent> {
        // Relative action index -> absolute heading per (living) snake; `advance` coasts dead ones.
        let mut moves: [Option<Action>; 2] = [None, None];
        for (i, (slot, snake)) in moves.iter_mut().zip(state.snakes.iter()).enumerate() {
            if snake.alive {
                *slot = Some(relative_to_absolute(
                    snake.direction,
                    RELATIVE_ACTIONS[actions[i]],
                ));
            }
        }
        // `|| None` = no in-advance respawn; the respawn is the chance step (`sample_chance`), so the
        // deterministic part and the sampled spawn stay separable and are shared by the rollout and search.
        let mut snakes = state.snakes.clone();
        let mut food = state.food.clone();
        let (events, done) = self.advance(&mut snakes, &mut food, moves, || None);
        Transition {
            next_state: SnakeState { snakes, food },
            events: events.into(),
            terminal: done,
        }
    }

    fn sample_chance(
        &self,
        state: &SnakeState,
        transition: &Transition<SnakeState, StepEvent>,
        rng: &mut dyn Rng,
    ) -> Option<SnakeState> {
        // An eaten apple is the only stochastic event: `step` removed it without respawning, so the
        // count drop = apples eaten. None eaten -> deterministic (`None`). Otherwise draw one
        // realization, respawning one uniform-random apple per eaten apple via `spawn_one` — the same
        // spawn the env rollout uses, so search and env share one chance model.
        let next = &transition.next_state;
        let eaten = state.food.len().saturating_sub(next.food.len());
        if eaten == 0 {
            return None;
        }
        let mut food = next.food.clone();
        for _ in 0..eaten {
            self.spawn_one(&next.snakes, &mut food, rng);
        }
        Some(SnakeState {
            snakes: next.snakes.clone(),
            food,
        })
    }

    /// The declared form of the respawn chance (the tree searches' seam), exactly mirroring
    /// `sample_chance`'s sequential `spawn_one` draws. One respawn: uniform over the free cells of
    /// the post-step state, indexed in `spawn_one`'s row-major order. Two respawns (both agents eat
    /// on one tick — the maximum, each head eats at most one apple): uniform over ORDERED pairs
    /// (first placement from the n free cells, second from the remaining n−1), matching the two
    /// sequential draws bijectively — n·(n−1) outcomes, so wide boards pay a transiently large
    /// probs vector on the rare double-eat edges. A full board degenerates to one no-op outcome
    /// (`spawn_one` can't place), keeping the sampler/declaration agreement exact.
    fn chance_outcomes(
        &self,
        state: &SnakeState,
        transition: &Transition<SnakeState, StepEvent>,
    ) -> Option<Vec<f64>> {
        let next = &transition.next_state;
        let eaten = state.food.len().saturating_sub(next.food.len());
        if eaten == 0 || transition.terminal {
            return None;
        }
        assert!(
            eaten <= 2,
            "at most two apples (one per head) can be eaten per tick"
        );
        let n = self.free_cell_count(next);
        let outcomes = match (eaten, n) {
            (_, 0) | (2, 1) => 1, // no (or only one) placeable cell: the tail draws no-op
            (1, _) => n,
            (2, _) => n * (n - 1),
            _ => unreachable!(),
        };
        Some(vec![1.0 / outcomes as f64; outcomes])
    }

    fn apply_chance(
        &self,
        state: &SnakeState,
        transition: &Transition<SnakeState, StepEvent>,
        outcome: usize,
    ) -> SnakeState {
        let next = &transition.next_state;
        let eaten = state.food.len().saturating_sub(next.food.len());
        let n = self.free_cell_count(next);
        let mut out = SnakeState {
            snakes: next.snakes.clone(),
            food: next.food.clone(),
        };
        match (eaten, n) {
            (_, 0) => {}
            (1, _) | (2, 1) => {
                let cell = self.nth_free_cell(&out, outcome % n.max(1));
                out.food.insert(cell);
            }
            (2, _) => {
                let (i, j) = (outcome / (n - 1), outcome % (n - 1));
                let first = self.nth_free_cell(&out, i);
                out.food.insert(first);
                let second = self.nth_free_cell(&out, j);
                out.food.insert(second);
            }
            _ => unreachable!(),
        }
        out
    }

    fn initial_state(&self, rng: &mut dyn Rng) -> SnakeState {
        let snakes = self.initial_snakes();
        let mut food = HashSet::new();
        for _ in 0..self.initial_food_count {
            self.spawn_one(&snakes, &mut food, rng);
        }
        SnakeState { snakes, food }
    }

    fn truncation_horizon(&self) -> Option<usize> {
        self.max_ticks
    }

    /// Flag each still-alive snake as having survived to the truncation, so its reward pays `survival`.
    fn mark_truncation(&self, state: &SnakeState, events: &mut [StepEvent]) {
        for (event, snake) in events.iter_mut().zip(state.snakes.iter()) {
            event.survived_to_max_ticks = snake.alive;
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
            grid_size: G,
            initial_length: 3,
            play_to_last: false,
            win_food_lead: None,
            initial_food_count: 1,
            max_ticks: None,
        }
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
        }
    }

    /// The unoccupied cells in row-major order — the oracle for what `spawn_one` may pick.
    fn empty_cells(snakes: &[SnakeBody; 2], food: &HashSet<Cell>, grid_size: i32) -> Vec<Cell> {
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
    fn step_env_equals_step_then_sample_chance() {
        // The unification invariant: the realized env step and the search's chance sampler are the
        // SAME draw. `step_env` must equal `step` then `sample_chance` under the same RNG seed, so the
        // rollout and the search can never use different chance dynamics.
        // Food directly in front of A (faces Right, head (4,2)): Forward eats it, triggering a respawn.
        let g = game();
        let st = initial_state(&[(4, 3)]);
        let actions = [0usize, 0];
        let realized = g.step_env(&st, &actions, &mut TestRng(42));
        let t = g.step(&st, &actions);
        let sampled = g.sample_chance(&st, &t, &mut TestRng(42));
        assert!(sampled.is_some(), "an eaten apple is a chance node");
        assert_eq!(realized.next_state, sampled.unwrap());
        assert_eq!(realized.events, t.events);
        assert_eq!(realized.terminal, t.terminal);
        assert!(
            (reward().step_reward(&realized.events[0], 0) - 1.0).abs() < 1e-12,
            "A ate one apple"
        );
        assert_eq!(
            realized.next_state.food.len(),
            1,
            "respawn restored the count"
        );
    }

    /// A scripted rng: hands out preset `below` draws in order (each must be in range).
    struct Forced(std::vec::IntoIter<usize>);
    impl Rng for Forced {
        fn below(&mut self, n: usize) -> usize {
            let v = self.0.next().expect("Forced rng exhausted");
            assert!(v < n.max(1), "forced draw {v} out of range {n}");
            v
        }
        fn unit(&mut self) -> f64 {
            0.0
        }
    }

    #[test]
    fn declared_chance_agrees_with_the_sampler_single_eat() {
        // The seam's foundational contract: `chance_outcomes`/`apply_chance` (the searches'
        // declared form) and `sample_chance` (the env's realized draw) are THE SAME distribution.
        // Exhaustive, not statistical: forcing the sampler's `below(n)` draw to d must equal
        // `apply_chance(d)` for EVERY d — index-for-index — so a refactor of `spawn_one`'s
        // indexing that misses `nth_free_cell` (or vice versa) fails loudly here instead of
        // silently training the search on a different chance model than the game produces.
        let g = game();
        let st = initial_state(&[(4, 3)]); // in front of A: Forward eats
        let t = g.step(&st, &[0, 0]);
        let probs = g
            .chance_outcomes(&st, &t)
            .expect("an eaten apple declares chance");
        let n = empty_cells(&t.next_state.snakes, &t.next_state.food, G).len();
        assert_eq!(probs.len(), n);
        assert!((probs.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!(probs.iter().all(|&p| (p - 1.0 / n as f64).abs() < 1e-15));
        for d in 0..n {
            let sampled = g
                .sample_chance(&st, &t, &mut Forced(vec![d].into_iter()))
                .unwrap();
            assert_eq!(sampled, g.apply_chance(&st, &t, d), "outcome {d} diverged");
        }
    }

    #[test]
    fn declared_chance_agrees_with_the_sampler_double_eat() {
        // The intricate path: both heads eat on one tick -> ordered placement pairs, second draw
        // over the board reduced by the first. `outcome = i*(n-1) + j` must equal forcing the
        // sampler's two sequential draws to (i, j), for EVERY pair — the full bijection.
        let g = game();
        let st = initial_state(&[(4, 3), (4, 5)]); // in front of A (faces Right) and B (faces Left)
        let t = g.step(&st, &[0, 0]);
        assert_eq!(
            st.food.len() - t.next_state.food.len(),
            2,
            "both heads must eat this tick"
        );
        let probs = g.chance_outcomes(&st, &t).expect("eats declare chance");
        let n = empty_cells(&t.next_state.snakes, &t.next_state.food, G).len();
        assert_eq!(
            probs.len(),
            n * (n - 1),
            "ordered pairs over the free cells"
        );
        assert!((probs.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        for i in 0..n {
            for j in 0..(n - 1) {
                let outcome = i * (n - 1) + j;
                let sampled = g
                    .sample_chance(&st, &t, &mut Forced(vec![i, j].into_iter()))
                    .unwrap();
                assert_eq!(
                    sampled,
                    g.apply_chance(&st, &t, outcome),
                    "pair ({i},{j}) diverged"
                );
            }
        }
    }

    #[test]
    fn no_eat_is_deterministic() {
        let g = game();
        let st = initial_state(&[(0, 0)]); // far corner, untouched
        for actions in [[0usize, 0], [1, 2], [2, 1], [0, 2]] {
            let t = g.step(&st, &actions);
            // Nothing eaten -> no chance node (`None`).
            assert!(
                g.sample_chance(&st, &t, &mut TestRng(1)).is_none(),
                "actions {actions:?}"
            );
            // ...so the realized env step is exactly the deterministic step.
            let realized = g.step_env(&st, &actions, &mut TestRng(1));
            assert_eq!(realized.next_state, t.next_state, "actions {actions:?}");
            assert_eq!(realized.events, t.events);
            assert_eq!(realized.terminal, t.terminal);
        }
    }

    #[test]
    fn sample_chance_draws_independent_valid_respawns() {
        // Repeated `sample_chance` draws (the caller's fan-out) are independent uniform-random apples on
        // a previously empty cell — not a single deterministic belief. One rng stream across the draws.
        let g = game();
        let st = initial_state(&[(4, 3)]);
        let t = g.step(&st, &[0, 0]);
        let mut rng = TestRng(7);
        let samples: Vec<SnakeState> = (0..20)
            .map(|_| g.sample_chance(&st, &t, &mut rng).unwrap())
            .collect();
        assert_eq!(samples.len(), 20);
        let occupied: std::collections::HashSet<Cell> = t
            .next_state
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
            "uniform sampling should vary across draws"
        );
    }

    #[test]
    fn sample_chance_is_uniform_over_empty_cells() {
        // The in-tree respawn must be uniform over the empty cells — the same draw the env makes, not a
        // bias toward any cell (e.g. the old first-empty belief). Over many single-apple respawns,
        // assert full coverage of the empty cells and a balanced hit frequency.
        let g = game();
        let st = initial_state(&[(4, 3)]);
        let t = g.step(&st, &[0, 0]); // A eats the only apple -> a respawn chance node
        let n = 20_000;
        let mut rng = TestRng(12345);
        let samples: Vec<SnakeState> = (0..n)
            .map(|_| g.sample_chance(&st, &t, &mut rng).unwrap())
            .collect();
        let empties = empty_cells(&t.next_state.snakes, &t.next_state.food, G);
        let mut counts: std::collections::HashMap<Cell, usize> = std::collections::HashMap::new();
        for s in &samples {
            let new: Vec<Cell> = s.food.difference(&t.next_state.food).copied().collect();
            assert_eq!(new.len(), 1, "exactly one apple respawns");
            *counts.entry(new[0]).or_default() += 1;
        }
        assert_eq!(
            counts.len(),
            empties.len(),
            "every empty cell must be reachable (full coverage)"
        );
        let min = *counts.values().min().unwrap();
        let max = *counts.values().max().unwrap();
        assert!(
            max <= 2 * min,
            "hit frequency should be balanced for a uniform draw: min={min} max={max}"
        );
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
        // A dead snake has no legal actions (the planner reads activeness from this).
        let mut dead = st.clone();
        dead.snakes[1].alive = false;
        assert!(g.legal_actions(&dead, 1).is_empty());
    }

    #[test]
    fn initial_state_spawns_the_configured_food_count_deterministically() {
        let mut g = game();
        g.initial_food_count = 3;
        let a = g.initial_state(&mut TestRng(7));
        let b = g.initial_state(&mut TestRng(7));
        assert_eq!(a, b, "same seed -> same initial state");
        assert_eq!(a.food.len(), 3);
        // Snakes match the initial placement; food sits on empty cells.
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
        // Realized transition: same move as `step`, but the eaten apple respawns at an RNG empty cell
        // (not the first-empty belief). Reward carries the food bonus; count is restored.
        let g = game();
        let r = reward();
        let st = initial_state(&[(4, 3)]);
        let t = g.step_env(&st, &[0, 0], &mut TestRng(1));
        assert!(
            (r.step_reward(&t.events[0], 0) - 1.0).abs() < 1e-12,
            "A ate -> food reward"
        );
        assert!(!t.terminal);
        assert_eq!(
            t.next_state.food.len(),
            1,
            "respawn restored the apple count"
        );
        // A coast with no food/death scores the bare step reward, never the survival bonus.
        let empty = initial_state(&[]);
        let t2 = g.step_env(&empty, &[0, 0], &mut TestRng(1));
        assert_eq!(
            [
                r.step_reward(&t2.events[0], 0),
                r.step_reward(&t2.events[1], 1)
            ],
            [0.0, 0.0]
        );
    }

    #[test]
    fn mark_truncation_pays_survival_to_the_living_only() {
        // `mark_truncation` flags each still-alive snake, and `step_reward` turns that flag into the
        // survival bonus — so only the living collect it.
        let r = SnakeReward {
            survival: 0.25,
            ..reward()
        };
        let mut st = initial_state(&[]);
        st.snakes[1].alive = false; // A alive, B dead
        let mut events = [StepEvent::default(), StepEvent::default()];
        game().mark_truncation(&st, &mut events);
        assert!(events[0].survived_to_max_ticks && !events[1].survived_to_max_ticks);
        assert!((r.step_reward(&events[0], 0) - 0.25).abs() < 1e-12); // A: survival
        assert_eq!(r.step_reward(&events[1], 1), 0.0); // B (dead): none
    }

    #[test]
    fn snake_length_cell_is_symmetric_off_diagonal_and_skips_dead() {
        // Only lengths + aliveness matter to the cell key, so filler bodies of the right length suffice.
        let mk = |la: usize, lb: usize, b_alive: bool| {
            let body = |len: usize| (0..len as i32).map(|c| (0, c)).collect::<VecDeque<Cell>>();
            SnakeState {
                snakes: [
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
            }
        };
        // Self-play symmetry: mirror states are the same (off-diagonal) cell.
        let lopsided = snake_length_cell(&mk(21, 6, true)).unwrap();
        assert_eq!(Some(lopsided), snake_length_cell(&mk(6, 21, true)));
        assert_ne!(
            lopsided >> 16,
            lopsided & 0xFFFF,
            "lopsided -> off-diagonal cell"
        );
        // Equal lengths -> a diagonal cell.
        let sym = snake_length_cell(&mk(9, 9, true)).unwrap();
        assert_eq!(sym >> 16, sym & 0xFFFF, "equal lengths -> diagonal cell");
        // A dead snake is not a valid restart point.
        assert_eq!(snake_length_cell(&mk(9, 9, false)), None);
    }
}

#[cfg(test)]
mod env_tests {
    use super::*;

    // A Snake config for driving the dynamics directly (these assert events, not rewards).
    // `advance` takes a working `(snakes, food)` and mutates it.
    fn game(grid_size: i32, win_food_lead: Option<usize>) -> Snake {
        Snake {
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
            [Some(Action::Right), Some(Action::Left)],
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
        // A heads Right; commanding Left (reverse) must coast Right, not reverse into itself.
        g.advance(&mut snakes, &mut food, [Some(Action::Left), None], || None);
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
                [Some(Action::Right), Some(Action::Left)],
                || None,
            );
        }
        // A and B meet at (10,10) on the 4th tick.
        assert_eq!(events[A].death_cause, Some(DeathCause::HeadOn));
        assert_eq!(events[B].death_cause, Some(DeathCause::HeadOn));
        assert!(events[A].drew && events[B].drew && done);
    }

    #[test]
    fn eating_grows_snake_and_spawns_replacement() {
        let g = game(20, None);
        let mut snakes = g.initial_snakes();
        let mut food = HashSet::from([(10, 7)]); // directly ahead of A
        let mut replacement = vec![(0, 0)];
        let (events, _) = g.advance(
            &mut snakes,
            &mut food,
            [Some(Action::Right), Some(Action::Left)],
            || replacement.pop(),
        );
        assert!(events[A].ate_food);
        assert_eq!(snakes[A].len(), 4); // tail kept
        assert!(food.contains(&(0, 0)) && !food.contains(&(10, 7)));
    }

    #[test]
    fn wall_death_when_running_off_grid() {
        let g = game(20, None);
        let mut snakes = g.initial_snakes();
        let mut food = HashSet::new();
        // Place A against the right wall (row 0, clear of B at row 10) so one Right step runs off-grid.
        snakes[A].body = VecDeque::from([(0, 19), (0, 18), (0, 17)]);
        snakes[A].direction = Action::Right;
        let (events, _) = g.advance(&mut snakes, &mut food, [Some(Action::Right), None], || None);
        assert_eq!(events[A].death_cause, Some(DeathCause::Wall));
        assert!(!snakes[A].alive);
    }

    #[test]
    fn self_body_collision_is_a_death() {
        let g = game(20, None);
        let mut snakes = g.initial_snakes();
        let mut food = HashSet::new();
        // A folds back on itself: head (5,5) facing Right; turning Up steps onto its own body at (4,5).
        snakes[A].body = VecDeque::from([(5, 5), (5, 4), (4, 4), (4, 5), (4, 6)]);
        snakes[A].direction = Action::Right;
        let (events, _) = g.advance(&mut snakes, &mut food, [Some(Action::Up), None], || None);
        assert_eq!(events[A].death_cause, Some(DeathCause::SelfBody));
        assert!(!snakes[A].alive);
    }

    #[test]
    fn opponent_body_collision_is_a_kill_and_win() {
        let g = game(20, None);
        let mut snakes = g.initial_snakes();
        let mut food = HashSet::new();
        // A lies along row 10; B drives its head down into A's body. B dies (OppBody), A is credited
        // the kill and — as the sole survivor — wins; B loses and the game ends (play_to_last = false).
        snakes[A].body = VecDeque::from([(10, 5), (10, 4), (10, 3)]);
        snakes[A].direction = Action::Right;
        snakes[B].body = VecDeque::from([(9, 5), (8, 5), (7, 5)]);
        snakes[B].direction = Action::Down;
        let (events, done) = g.advance(
            &mut snakes,
            &mut food,
            [Some(Action::Right), Some(Action::Down)],
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
        // A is two apples (length) ahead of B, both alive; the lead triggers an outright win this tick.
        snakes[A].body = VecDeque::from([(2, 5), (2, 4), (2, 3), (2, 2), (2, 1)]); // length 5 vs B's 3
        snakes[A].direction = Action::Right;
        let (events, done) =
            g.advance(&mut snakes, &mut food, [Some(Action::Right), None], || None);
        assert!(events[A].won && events[B].lost && done);
    }
}

#[cfg(test)]
mod obs_tests {
    use super::*;

    fn at(obs: &[f32], g: i32, ch: usize, r: i32, c: i32) -> f32 {
        obs[ch * (g * g) as usize + (r as usize) * (g as usize) + (c as usize)]
    }

    // The default initial placement on a 20-grid (heads at (10,6) / (10,14)).
    fn snakes() -> [SnakeBody; 2] {
        Snake {
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
        // A faces Right -> k=1 -> (r,c) maps to (edge-c, r). Head (10,6) -> (19-6, 10) = (13,10).
        let obs = egocentric_parts(&snakes(), &HashSet::new(), 20, A);
        assert_eq!(at(&obs, 20, CH_OWN_HEAD, 13, 10), 1.0);
        // Body cells (10,5),(10,4) -> (14,10),(15,10) in the own-body channel.
        assert_eq!(at(&obs, 20, CH_OWN_BODY, 14, 10), 1.0);
        assert_eq!(at(&obs, 20, CH_OWN_BODY, 15, 10), 1.0);
        // The head cell must not also be flagged as body.
        assert_eq!(at(&obs, 20, CH_OWN_BODY, 13, 10), 0.0);
        // Opponent B head (10,14) -> (19-14,10) = (5,10) in the opp-head channel.
        assert_eq!(at(&obs, 20, CH_OPP_HEAD, 5, 10), 1.0);
    }

    #[test]
    fn food_lands_in_food_channel_rotated() {
        let food = HashSet::from([(10, 6)]); // same transform as A's head -> (13,10)
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
        // The survival bonus rides on the `survived_to_max_ticks` flag (set only by `mark_truncation`),
        // never on an ordinary event: a nothing-happened tick scores 0, and eating scores the food bonus.
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
        // The died branch returns before the alive-only terms, so a lost death scores just `loss`.
        let r = reward();
        let dead = StepEvent {
            died: true,
            lost: true,
            ..Default::default()
        };
        assert!((r.step_reward(&dead, 0) - (-0.5)).abs() < 1e-12); // loss only
    }
}
