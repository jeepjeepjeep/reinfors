//! The N-player snake game (2-8 snakes, default 2), in one self-contained module (mirroring
//! `connect4.rs`): grid actions,
//! the egocentric observation encoder, reward shaping, and the `Snake` adapter implementing
//! `reinfors_core::Game` (its deterministic dynamics live in private methods, alongside the
//! `EgocentricSnake` state encoder).
//!
//! The dynamics are deterministic integer arithmetic with food placement injected, so a given set of
//! actions + spawns yields bit-identical trajectories — the basis for the native regression tests.

use std::collections::{HashMap, HashSet, VecDeque};

use reinfors_core::game::{Actor, Game, Rng, Transition};
use reinfors_core::{ActionView, Reward, Space, StateEncoder};

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

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
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

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash, serde::Serialize, serde::Deserialize)]
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

/// Build the egocentric observation for `agent` as a flat `[5 * g * g]` f32 buffer, from a
/// `(snakes, food)` state. Coordinates are pre-rotated so the queried snake faces "up". EVERY
/// other snake lands in the shared opponent head/body channels, so the observation shape is
/// independent of the snake count (a per-opponent-channel encoder is a possible future variant
/// behind the encoder seam).
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
/// Canonical food serialization: a HashSet iterates nondeterministically, which would make
/// equal states encode to different bytes (breaking snapshot byte-identity) — so food
/// serializes as a SORTED list, and deserialization rejects duplicates rather than silently
/// deduplicating them.
mod food_serde {
    use super::Cell;
    use serde::de::Error as _;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::collections::HashSet;

    pub fn serialize<S: Serializer>(food: &HashSet<Cell>, ser: S) -> Result<S::Ok, S::Error> {
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
    pub snakes: Vec<SnakeBody>, // [num_snakes]
    #[serde(with = "food_serde")]
    pub food: HashSet<Cell>,
}

// Coarsening for the reached-state buffer: bucket a snake length, and how many buckets. Difficulty
// tracks length, so a fixed linear bucketing spreads start coverage from early to late game; very-late
// lengths saturate the last bucket. Fixed in v1 (a configurable coarsening is a future refinement).
const CELL_BUCKET_SIZE: usize = 3;
const CELL_N_BUCKETS: u64 = 10;

/// A start-state-buffer coverage cell for a snake state: the **sorted multiset** of the snakes'
/// length buckets, packed into a `u64`. Unordered because self-play is symmetric — one lopsided
/// rollout already yields every perspective, so `(20, 4)` and `(4, 20)` are the same cell.
/// Returns `None` for a state with any dead snake (v1 keeps coverage on full games; a
/// mid-elimination restart seam is a future refinement). Packing: 16-bit fields up to 4 snakes
/// (2 snakes reproduce the legacy `(lo << 16) | hi` keys exactly, so existing buffers bucket
/// identically), 8-bit fields for 5-8 (buckets < 10 fit either way).
/// This is snake's `cell_key` for [`ReachedStateBuffer`](reinfors_core::ReachedStateBuffer).
pub fn snake_length_cell(state: &SnakeState) -> Option<u64> {
    if state.snakes.iter().any(|s| !s.alive) {
        return None;
    }
    let bucket = |len: usize| ((len / CELL_BUCKET_SIZE) as u64).min(CELL_N_BUCKETS - 1);
    let mut buckets: Vec<u64> = state.snakes.iter().map(|s| bucket(s.len())).collect();
    buckets.sort_unstable();
    let shift = if buckets.len() <= 4 { 16 } else { 8 };
    Some(buckets.into_iter().fold(0u64, |acc, b| (acc << shift) | b))
}

/// The default snake observation: an egocentric 5-channel grid, the searching snake always facing up
/// (see [`egocentric_parts`]). Carries `grid_size` (which lives on `Snake`, not in `SnakeState`).
pub struct EgocentricSnake {
    pub grid_size: i32,
}

impl ActionView for EgocentricSnake {} // absolute: identity action view

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

/// N-player simultaneous-move snake with environment chance (apple respawn). Its deterministic
/// dynamics live in the private `impl Snake` methods below, behind the `Game` trait.
pub struct Snake {
    /// Number of snakes (2-8; validated). 2 keeps the legacy placement and rules exactly; more
    /// snakes spread across the grid, with every rule generalizing per-agent (the lone survivor
    /// wins, deaths with survivors lose, simultaneous last deaths draw).
    pub num_snakes: usize,
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
    /// Config invariants, checked once at the construction boundary so no input reaches a panic
    /// later. Placement is checked by *constructing* it — the deterministic `initial_snakes`
    /// layout must fit the grid, stay disjoint (long bodies can wrap the perimeter into each
    /// other), and leave enough free cells for the food — rather than by re-deriving bounds that
    /// could drift from the layout code.
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
        // u128: `N_CHANNELS as i64 * cells` overflows i64 for g above ~1.36e9 — a panic in debug
        // builds and, worse, a wrap in release that bypasses this very ceiling
        if g >= 1 && N_CHANNELS as u128 * cells as u128 > i32::MAX as u128 {
            return Err(format!(
                "grid_size {g} makes the observation tensor exceed 2^31 elements"
            ));
        }
        // The respawn chance indexes ordered k-tuples of free cells for k apples eaten on one
        // tick (k <= min(snakes, food)). The declaration is O(1) at any size (`Uniform(count)`),
        // but the index must survive an f64 mantissa (the uniform draw) and a usize decode —
        // bound the worst case at 2^53. Unreachable for realistic configs (a 20-grid triple-eat
        // is ~6e7); negative grids clamp to zero cells here and are rejected by the placement
        // construction below.
        let k_max = self.num_snakes.min(self.initial_food_count) as u128;
        let cell_count = self.grid_size.max(0) as u128 * self.grid_size.max(0) as u128;
        let worst: u128 = (0..k_max)
            .map(|i| cell_count.saturating_sub(i))
            .try_fold(1u128, |acc, f| acc.checked_mul(f))
            .unwrap_or(u128::MAX);
        if worst > (1u128 << 53).min(usize::MAX as u128) {
            return Err(format!(
                "worst-case respawn index space ({k_max} apples eatable at once on a \
                 {cell_count} - cell grid) exceeds min(2^53, usize::MAX); reduce food or grid \
                 size"
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
        // u128: `initial_food_count as i64` would wrap negative for huge usize values and slip past
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
        // Two snakes keep the legacy placement byte-for-byte (heads at thirds of the middle row);
        // more spread across evenly-spaced rows, alternating the left/right pattern.
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

    /// Advance one tick over `(snakes, food)` in place. `actions[i]` is an absolute move for snake `i`
    /// (None = coast in its current heading; a reverse move coasts, as in snake_RL). `next_food` is
    /// called once per eaten apple to supply its replacement cell. Returns the per-snake events and
    /// whether the episode is now over.
    fn advance(
        &self,
        snakes: &mut [SnakeBody],
        food: &mut HashSet<Cell>,
        actions: &[Option<Action>],
        mut next_food: impl FnMut() -> Option<Cell>,
    ) -> (Vec<StepEvent>, bool) {
        let n = snakes.len();
        let mut events = vec![StepEvent::default(); n];

        // Stage 1: move every living snake. Eating intent keeps the tail; survival is settled below.
        let mut ate_intent = vec![false; n];
        let mut moved = vec![false; n];
        for i in 0..n {
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
        let mut fatal_heads = vec![None; n];
        for i in 0..n {
            if causes[i].is_some() {
                fatal_heads[i] = Some(snakes[i].head());
            }
        }
        for i in 0..n {
            if let Some(cause) = causes[i] {
                snakes[i].alive = false;
                snakes[i].body.pop_front(); // corpse vacates the collision cell
                events[i].died = true;
                events[i].death_cause = Some(cause);
            }
        }

        // Eating: survivors whose new head landed on food eat it and trigger a replacement spawn.
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

        // Kill credit: a snake whose body occupies the cell an opponent fatally moved into.
        for i in 0..n {
            if causes[i] == Some(DeathCause::OppBody) {
                if let Some(killer) = self.find_killer(snakes, fatal_heads[i].unwrap(), i) {
                    events[killer].killed_opponent = true;
                }
            }
        }

        // Outcomes (per-agent, any N): the lone survivor of a deadly tick wins outright; a death
        // with survivors remaining loses; simultaneous deaths that leave nobody draw.
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
        // A terminal tick with both still alive can only be the food-lead rule firing: attribute
        // the outright win/loss (`is_terminal` owns the rule itself).
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

    /// Whether this configuration of snakes ends the episode: the death rule (everyone dead, or a
    /// lone survivor unless `play_to_last`) or the food-lead rule (both alive, body-length
    /// difference at or past the configured lead). The single source of terminality — `advance`
    /// ends the tick on it, and the codec's lifecycle check compares the envelope's `done`
    /// against it rather than re-implementing the rules.
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

        // Heads sharing one cell die together, whatever else is on it.
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

    /// The lowest-indexed living snake whose body holds the fatal cell (deterministic when
    /// several bodies share it).
    fn find_killer(&self, snakes: &[SnakeBody], fatal_head: Cell, victim: usize) -> Option<usize> {
        (0..snakes.len())
            .find(|&j| j != victim && snakes[j].alive && snakes[j].body.contains(&fatal_head))
    }

    /// Spawn one apple at a uniform-random empty cell (the env's true spawn), or nothing if the grid is
    /// full. A single `rng.below(n)` picks `k`, the index of the chosen cell among the empties in
    /// row-major order; the cell itself is found analytically from the sorted occupied indices (each
    /// occupied cell at or below the running target shifts it one further on), so there is no O(g²)
    /// scan over the grid. This selects the same cell a linear walk would for the same `k`.
    /// Place one apple on the `i`-th free cell drawn by `rng` — initial-state placement, routed
    /// through the SAME free-cell indexing (`occupied_of` + `nth_free_of`) as `apply_chance`, so
    /// exactly one implementation of "the i-th free cell" exists.
    fn spawn_one(&self, snakes: &[SnakeBody], food: &mut HashSet<Cell>, rng: &mut dyn Rng) {
        let occupied = self.occupied_of(snakes, food);
        let n = (self.grid_size * self.grid_size) as usize - occupied.len();
        if n == 0 {
            return;
        }
        let i = rng.below(n);
        food.insert(Self::nth_free_of(&occupied, self.grid_size, i));
    }

    /// Occupied cell ids (food + both bodies), sorted + deduped — the complement enumerates the
    /// free cells in row-major order.
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

    /// The `i`-th free cell in row-major order given the sorted occupied set — THE free-cell
    /// indexing (initial placement and chance materialization both resolve through it).
    fn nth_free_of(occupied: &[usize], grid: i32, i: usize) -> Cell {
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

    /// The `i`-th free cell in row-major order of `state`.
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
        let mut moves: Vec<Option<Action>> = vec![None; state.snakes.len()];
        for (i, (slot, snake)) in moves.iter_mut().zip(state.snakes.iter()).enumerate() {
            if snake.alive {
                *slot = Some(relative_to_absolute(
                    snake.direction,
                    RELATIVE_ACTIONS[actions[i]],
                ));
            }
        }
        // `|| None` = no in-advance respawn; the respawn is the declared chance step
        // (`chance_outcomes`/`apply_chance`, realized by the framework), so the deterministic part
        // and the chance element stay separable and are shared by the rollout and search.
        let mut snakes = state.snakes.clone();
        let mut food = state.food.clone();
        let (events, done) = self.advance(&mut snakes, &mut food, &moves, || None);
        Transition {
            next_state: SnakeState { snakes, food },
            events,
            terminal: done,
        }
    }

    /// The respawn chance, declared (the game's only chance seam). `k` apples eaten on one tick
    /// (each living head eats at most one, so `k <= min(snakes, food)`) respawn as `k` sequential
    /// uniform draws over the shrinking free set — indexed as ordered tuples, mixed-radix with
    /// bases `n, n-1, …, n-k+1` over the free-cell count `n` (the k = 2 case is exactly the old
    /// `n·(n-1)` ordered-pair layout). Placements cap at the free cells available; a full board
    /// degenerates to one no-op outcome, keeping the sampler/declaration agreement exact.
    /// Declared as `Uniform(count)`: O(1) at any size — sampling consumers draw one index and
    /// `apply_chance` decodes it procedurally; only `ExpandAll` enumerates (bounded at the
    /// consumer). The index space is validated to fit 2^53 at construction.
    fn chance_outcomes(
        &self,
        state: &SnakeState,
        transition: &Transition<SnakeState, StepEvent>,
    ) -> Option<reinfors_core::ChanceDist> {
        let next = &transition.next_state;
        let eaten = state.food.len().saturating_sub(next.food.len());
        if eaten == 0 || transition.terminal {
            return None;
        }
        let n = self.free_cell_count(next);
        let placeable = eaten.min(n);
        // Checked as a defensive backstop: construction AND decoded-state validation bound the
        // index space, but a silent wrap here would mean silently wrong probabilities.
        let outcomes = (0..placeable)
            .try_fold(1usize, |acc, i| acc.checked_mul(n - i))
            .expect("respawn index space overflows usize (bounded at construction and decode)")
            .max(1);
        Some(reinfors_core::ChanceDist::Uniform(outcomes))
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
        let placeable = eaten.min(n);
        let mut out = SnakeState {
            snakes: next.snakes.clone(),
            food: next.food.clone(),
        };
        // Mixed-radix digits, least-significant (base n-k+1, the LAST draw) first; each placement
        // indexes the free cells REMAINING after the ones before it, matching the sequential
        // draws bijectively.
        let mut digits = vec![0usize; placeable];
        let mut rem = outcome;
        for i in (0..placeable).rev() {
            let base = n - i;
            digits[i] = rem % base;
            rem /= base;
        }
        for &d in &digits {
            let cell = self.nth_free_cell(&out, d);
            out.food.insert(cell);
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
        assert!(cfg(3, 3, 3).validate().is_ok()); // smallest grid for the default length
        assert!(cfg(2, 3, 1).validate().is_err()); // head placed outside the grid
        assert!(cfg(0, 1, 0).validate().is_err());
        assert!(cfg(-4, 3, 1).validate().is_err());
        assert!(cfg(8, 0, 1).validate().is_err()); // empty body
        assert!(cfg(8, 100, 1).validate().is_err()); // trace runs out of directions
        assert!(cfg(3, 3, 4).validate().is_err()); // food exceeds the 3 free cells
        assert!(cfg(8, 3, 1usize << 63).validate().is_err()); // would wrap an i64 food cast
        assert!(cfg(46_000, 3, 1).validate().is_err()); // obs tensor over 2^31 elements
        assert!(cfg(1_500_000_000, 3, 1).validate().is_err()); // past the i64 5*g^2 overflow point
        assert!(cfg(i32::MAX, 3, 1).validate().is_err());

        // Long bodies wrap the perimeter into each other without exhausting the trace — the
        // disjointness of the *constructed* placement is what catches it.
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
        }
    }

    /// The unoccupied cells in row-major order — the oracle for what `spawn_one` may pick.
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
        // Realization is the framework's (`reinfors_core::game::step_env` — one draw from the
        // game's declared distribution). The realized state must be `apply_chance` of SOME
        // declared outcome, with the deterministic parts (events, terminal) untouched — and the
        // same seed must realize the same outcome.
        let g = game();
        let st = initial_state(&[(4, 3)]); // in front of A: Forward eats
        let actions = [0usize, 0];
        let t = g.step(&st, &actions);
        let dist = g
            .chance_outcomes(&st, &t)
            .expect("an eaten apple declares chance");
        let realized = reinfors_core::game::step_env(&g, &st, &actions, &mut TestRng(42));
        assert_eq!(realized.events, t.events);
        assert_eq!(realized.terminal, t.terminal);
        assert!(
            (0..dist.count()).any(|d| realized.next_state == g.apply_chance(&st, &t, d)),
            "the realized state must be one of the declared outcomes"
        );
        let again = reinfors_core::game::step_env(&g, &st, &actions, &mut TestRng(42));
        assert_eq!(
            realized.next_state, again.next_state,
            "same seed, same realization"
        );
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

    #[test]
    fn no_eat_is_deterministic() {
        let g = game();
        let st = initial_state(&[(0, 0)]); // far corner, untouched
        for actions in [[0usize, 0], [1, 2], [2, 1], [0, 2]] {
            let t = g.step(&st, &actions);
            // Nothing eaten -> no declared chance (`None`).
            assert!(g.chance_outcomes(&st, &t).is_none(), "actions {actions:?}");
            // ...so the realized env step is exactly the deterministic step.
            let realized = reinfors_core::game::step_env(&g, &st, &actions, &mut TestRng(1));
            assert_eq!(realized.next_state, t.next_state, "actions {actions:?}");
            assert_eq!(realized.events, t.events);
            assert_eq!(realized.terminal, t.terminal);
        }
    }

    #[test]
    fn realized_respawns_vary_and_land_on_empty_cells() {
        // Repeated realizations through the framework's `step_env` are independent uniform apples
        // on previously empty cells — one rng stream across the draws.
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
        // Stronger than the old sampled-frequency check, and deterministic: the declared
        // distribution is uniform with exactly one outcome per empty cell, and `apply_chance(i)`
        // places the apple on precisely the i-th cell of the independent `empty_cells` oracle.
        let g = game();
        let st = initial_state(&[(4, 3)]);
        let t = g.step(&st, &[0, 0]); // A eats the only apple -> a respawn chance node
        let dist = g.chance_outcomes(&st, &t).expect("eat declares chance");
        let empties = empty_cells(&t.next_state.snakes, &t.next_state.food, G);
        assert_eq!(
            dist,
            reinfors_core::ChanceDist::Uniform(empties.len()),
            "one uniform outcome per empty cell"
        );
        for (i, &cell) in empties.iter().enumerate() {
            let placed: Vec<Cell> = g
                .apply_chance(&st, &t, i)
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
        let t = reinfors_core::game::step_env(&g, &st, &[0, 0], &mut TestRng(1));
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
        let t2 = reinfors_core::game::step_env(&g, &empty, &[0, 0], &mut TestRng(1));
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
        // A heads Right; commanding Left (reverse) must coast Right, not reverse into itself.
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
            &[Some(Action::Right), Some(Action::Left)],
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
        // A folds back on itself: head (5,5) facing Right; turning Up steps onto its own body at (4,5).
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
        // A lies along row 10; B drives its head down into A's body. B dies (OppBody), A is credited
        // the kill and — as the sole survivor — wins; B loses and the game ends (play_to_last = false).
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
        // A is two apples (length) ahead of B, both alive; the lead triggers an outright win this tick.
        snakes[A].body = VecDeque::from([(2, 5), (2, 4), (2, 3), (2, 2), (2, 1)]); // length 5 vs B's 3
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

    // The default initial placement on a 20-grid (heads at (10,6) / (10,14)).
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

impl reinfors_core::StateCodec for Snake {
    type State = SnakeState;

    fn encode(&self, s: &SnakeState) -> Vec<u8> {
        // Layout 3: `snakes` became a length-prefixed Vec (the fixed two-element array era was 2).
        crate::codec_util::serde_encode(3, s)
    }

    fn decode(&self, bytes: &[u8]) -> Result<SnakeState, String> {
        crate::codec_util::serde_decode(3, bytes)
    }

    // Safety per the narrowed codec contract: bounds that game methods index by, plus lifecycle
    // coherence — `done` controls whether the Env may continue, so it must agree with the game's
    // own terminal rule (`is_terminal`, the same function `advance` ends ticks on — a shared
    // source, not a re-implementation). Occupancy rules are NOT re-proved here; unreachable-but-
    // safe states are accepted.
    fn validate_decoded_state(&self, state: &SnakeState, done: bool) -> Result<(), String> {
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
        // A decoded state may carry MORE food than `initial_food_count` (validation is safety,
        // not reachability) — but the respawn index space it implies must still be indexable,
        // or a later multi-eat would overflow the chance arithmetic the construction-time bound
        // was computed to prevent.
        let k = self.num_snakes.min(state.food.len()) as u128;
        let cells = self.grid_size.max(0) as u128 * self.grid_size.max(0) as u128;
        let worst: u128 = (0..k)
            .map(|i| cells.saturating_sub(i))
            .try_fold(1u128, |acc, f| acc.checked_mul(f))
            .unwrap_or(u128::MAX);
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
        assert!(game(1, 8, 2).validate().is_err()); // solo snake is a different mode
        assert!(game(9, 20, 2).validate().is_err());
        let mut lead = game(3, 8, 2);
        lead.win_food_lead = Some(2);
        assert!(lead.validate().is_err(), "the lead rule is two-snake only");
        // The respawn index space is O(1) to declare (`Uniform`), so the bound is only the
        // 2^53 index-representability guard: every realistic config passes (the 20-grid
        // triple-eat's ~6e7 index space included); a 1000-grid triple-eat (~1e18) rejects.
        assert!(game(3, 20, 3).validate().is_ok());
        assert!(game(3, 200, 2).validate().is_ok());
        assert!(game(3, 200, 3).validate().is_ok()); // ~6.4e13: indexable
        assert!(game(3, 1000, 3).validate().is_err());
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
        // N=2 must stay byte-identical: heads at thirds of the middle row, facing inward.
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
        // Three heads converge on (4, 4).
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
        // Snake 0 runs into the wall; 1 and 2 survive -> 0 lost, nobody won, game continues.
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
        // Drive snake 1 up and out of bounds while snake 2 orbits a safe 2x2 box (it already
        // moved Right to (6,7) in the tick above, so the orbit starts at Down); when snake 1
        // hits the wall, snake 2 is the lone survivor of a deadly tick -> won, terminal.
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
        // 3 snakes all eating on one tick: outcomes = n·(n-1)·(n-2) ordered triples of free
        // cells, each applying as three sequential placements over the shrinking free set.
        let g = game(3, 6, 3);
        let mut rng = TestRng(3);
        let mut state = g.initial_state(&mut rng);
        // Put an apple directly ahead of each head so every snake eats simultaneously.
        state.food.clear();
        let mut expected_heads = Vec::new();
        for s in &state.snakes {
            let (dr, dc) = s.direction.delta();
            let (hr, hc) = s.head();
            state.food.insert((hr + dr, hc + dc));
            expected_heads.push((hr + dr, hc + dc));
        }
        let t = g.step(&state, &[0, 0, 0]); // Forward for everyone
        assert!(state.food.len() == 3 && t.next_state.food.is_empty());
        let dist = g
            .chance_outcomes(&state, &t)
            .expect("3 eats declare chance");
        let n = (6 * 6)
            - t.next_state
                .snakes
                .iter()
                .map(|s| s.body.len())
                .sum::<usize>() as i32;
        let n = n as usize;
        assert_eq!(dist.count(), n * (n - 1) * (n - 2));
        // Every outcome yields exactly 3 fresh apples on free cells; outcome 0 places the three
        // lowest free cells in row-major order (sequential draws at index 0 each time).
        let s0 = g.apply_chance(&state, &t, 0);
        assert_eq!(s0.food.len(), 3);
        let last = g.apply_chance(&state, &t, dist.count() - 1);
        assert_eq!(last.food.len(), 3);
        assert_ne!(s0.food, last.food);
    }

    #[test]
    fn decoded_food_counts_past_the_index_space_reject() {
        use reinfors_core::StateCodec;
        // The game constructs with 1 food (worst-case k = 1, trivially indexable) — but a decoded
        // state can carry any in-grid food set, and 8 snakes x 8 food implies P(400, 8) ≈ 4e20
        // ordered placements: past the index guard, so validation must reject it before the
        // chance arithmetic ever sees it.
        let g = game(8, 20, 1);
        assert!(g.validate().is_ok());
        let snakes: Vec<SnakeBody> = (0..8)
            .map(|i| body_at(&[(2 * i, 0)], Action::Right))
            .collect();
        let ok_state = SnakeState {
            snakes: snakes.clone(),
            food: (0..3).map(|c| (1, c)).collect(),
        };
        g.validate_decoded_state(&ok_state, false).unwrap();
        let over = SnakeState {
            snakes,
            food: (0..8).map(|c| (1, c)).collect(),
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
        };
        // buckets: len 1 -> 0, len 3 -> 1; legacy packing (lo << 16) | hi.
        assert_eq!(snake_length_cell(&two), Some(1));
        let three = SnakeState {
            snakes: vec![
                body_at(&[(1, 1), (1, 2), (1, 3)], Action::Up),
                body_at(&[(3, 3)], Action::Up),
                body_at(&[(5, 5)], Action::Up),
            ],
            food: HashSet::new(),
        };
        // sorted buckets [0, 0, 1] -> (((0)<<16 | 0) << 16) | 1
        assert_eq!(snake_length_cell(&three), Some(1));
        let mut dead = three.clone();
        dead.snakes[1].alive = false;
        assert_eq!(snake_length_cell(&dead), None);
    }

    struct TestRng(u64);
    impl Rng for TestRng {
        fn below(&mut self, n: usize) -> usize {
            self.0 = self
                .0
                .wrapping_mul(2862933555777941757)
                .wrapping_add(3037000493);
            (self.0 >> 33) as usize % n.max(1)
        }
        fn unit(&mut self) -> f64 {
            self.below(1 << 20) as f64 / (1 << 20) as f64
        }
    }
}
