//! The two-player snake game, in one self-contained module (mirroring `connect4.rs`): grid actions,
//! the egocentric observation encoder, reward shaping, the deterministic `SnakeEnv` dynamics, and the
//! `Snake` adapter implementing `reinfors_core::Game` (+ the `EgocentricSnake` state encoder).
//!
//! The dynamics are deterministic integer arithmetic with food placement injected, so a given set of
//! actions + spawns yields bit-identical trajectories — the basis for the native regression tests.

use std::collections::{HashMap, HashSet, VecDeque};

use reinfors_core::game::{Actor, Game, Rng, Transition};
use reinfors_core::StateEncoder;

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
    /// Set by the rollout engine (never by `advance` or the search) when this is a truncation tick
    /// the snake reached alive, so the `survival` reward fires only there — matching the runner.
    pub survived_to_max_ticks: bool,
}

pub struct SnakeEnv {
    pub grid_size: i32,
    pub initial_length: usize,
    pub play_to_last: bool,
    pub win_food_lead: Option<usize>,
    pub snakes: [SnakeBody; 2],
    pub food: HashSet<Cell>,
    pub done: bool,
}

impl SnakeEnv {
    pub fn new(
        grid_size: i32,
        initial_length: usize,
        play_to_last: bool,
        win_food_lead: Option<usize>,
    ) -> Self {
        let snakes = Self::initial_snakes(grid_size, initial_length);
        SnakeEnv {
            grid_size,
            initial_length,
            play_to_last,
            win_food_lead,
            snakes,
            food: HashSet::new(),
            done: false,
        }
    }

    /// Build an env from an explicit (snakes, food) state, bypassing initial placement — used by the
    /// search to simulate a tick from an arbitrary node.
    pub fn from_parts(
        grid_size: i32,
        initial_length: usize,
        play_to_last: bool,
        win_food_lead: Option<usize>,
        snakes: [SnakeBody; 2],
        food: HashSet<Cell>,
    ) -> Self {
        SnakeEnv {
            grid_size,
            initial_length,
            play_to_last,
            win_food_lead,
            snakes,
            food,
            done: false,
        }
    }

    /// Heads at one-third / two-thirds along the middle row, bodies trailing to the nearer wall and
    /// wrapping (matches `_initial_snakes` / `_trace_body`).
    fn initial_snakes(grid_size: i32, length: usize) -> [SnakeBody; 2] {
        let g = grid_size;
        let mid = g / 2;
        let a_body = Self::trace_body(
            g,
            (mid, g / 3),
            &[Action::Left, Action::Down, Action::Right, Action::Up],
            length,
        );
        let b_body = Self::trace_body(
            g,
            (mid, g - g / 3),
            &[Action::Right, Action::Up, Action::Left, Action::Down],
            length,
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

    fn in_bounds(&self, (r, c): Cell) -> bool {
        0 <= r && r < self.grid_size && 0 <= c && c < self.grid_size
    }

    /// Advance one tick in place. `actions[i]` is an absolute move for snake `i` (None = coast in its
    /// current heading; a reverse move is treated as coast, as in the Python env). `next_food` is
    /// called once per eaten apple to supply its replacement cell (None = no room / no replacement).
    /// Returns the per-snake events; `self.done` is also updated.
    pub fn advance(
        &mut self,
        actions: [Option<Action>; 2],
        mut next_food: impl FnMut() -> Option<Cell>,
    ) -> [StepEvent; 2] {
        let mut events = [StepEvent::default(), StepEvent::default()];

        // Stage 1: move every living snake. Eating intent keeps the tail; survival is settled below.
        let mut ate_intent = [false, false];
        let mut moved = [false, false];
        for i in 0..2 {
            if !self.snakes[i].alive {
                continue;
            }
            moved[i] = true;
            let mut action = actions[i].unwrap_or(self.snakes[i].direction);
            if action.opposite() == self.snakes[i].direction {
                action = self.snakes[i].direction; // reverse is a no-op; coast straight
            }
            self.snakes[i].direction = action;
            let (dr, dc) = action.delta();
            let (hr, hc) = self.snakes[i].head();
            let new_head = (hr + dr, hc + dc);
            ate_intent[i] = self.food.contains(&new_head);
            self.snakes[i].body.push_front(new_head);
            if !ate_intent[i] {
                self.snakes[i].body.pop_back();
            }
        }

        // Stage 2: resolve collisions against the post-move world.
        let causes = self.resolve_collisions(&moved);
        let mut fatal_heads = [None, None];
        for i in 0..2 {
            if causes[i].is_some() {
                fatal_heads[i] = Some(self.snakes[i].head());
            }
        }
        for i in 0..2 {
            if let Some(cause) = causes[i] {
                self.snakes[i].alive = false;
                self.snakes[i].body.pop_front(); // corpse vacates the collision cell
                events[i].died = true;
                events[i].death_cause = Some(cause);
            }
        }

        // Eating: survivors whose new head landed on food eat it and trigger a replacement spawn.
        for i in 0..2 {
            if ate_intent[i] && causes[i].is_none() {
                let head = self.snakes[i].head();
                self.food.remove(&head);
                events[i].ate_food = true;
                if let Some(cell) = next_food() {
                    self.food.insert(cell);
                }
            }
        }

        // Kill credit: a snake whose body occupies the cell an opponent fatally moved into.
        for i in 0..2 {
            if causes[i] == Some(DeathCause::OppBody) {
                if let Some(killer) = self.find_killer(fatal_heads[i].unwrap(), i) {
                    events[killer].killed_opponent = true;
                }
            }
        }

        // Outcomes.
        let alive_ids: Vec<usize> = (0..2).filter(|&i| self.snakes[i].alive).collect();
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
                let (leader, runner) = if self.snakes[i0].len() >= self.snakes[i1].len() {
                    (i0, i1)
                } else {
                    (i1, i0)
                };
                if self.snakes[leader].len() - self.snakes[runner].len() >= lead {
                    events[leader].won = true;
                    events[runner].lost = true;
                    done = true;
                }
            }
        }

        self.done = done;
        events
    }

    fn resolve_collisions(&self, moved: &[bool; 2]) -> [Option<DeathCause>; 2] {
        let mut causes = [None, None];

        // Two heads on one cell die together, whatever else is on it.
        let mut by_head: HashMap<Cell, Vec<usize>> = HashMap::new();
        for (i, &m) in moved.iter().enumerate() {
            if m {
                by_head.entry(self.snakes[i].head()).or_default().push(i);
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
            let head = self.snakes[i].head();
            if !self.in_bounds(head) {
                causes[i] = Some(DeathCause::Wall);
            } else if self.snakes[i].body.iter().skip(1).any(|&c| c == head) {
                causes[i] = Some(DeathCause::SelfBody);
            } else {
                for j in 0..2 {
                    if j == i || !self.snakes[j].alive {
                        continue;
                    }
                    if self.snakes[j].body.contains(&head) {
                        causes[i] = Some(DeathCause::OppBody);
                        break;
                    }
                }
            }
        }
        causes
    }

    fn find_killer(&self, fatal_head: Cell, victim: usize) -> Option<usize> {
        (0..2).find(|&j| {
            j != victim && self.snakes[j].alive && self.snakes[j].body.contains(&fatal_head)
        })
    }
}

/// Unoccupied cells in row-major order — the apple-spawn candidates (occupied = both snake bodies and
/// existing food, matching the oracle's `_spawn_cells` / `_sample_spawn`).
pub fn empty_cells(snakes: &[SnakeBody; 2], food: &HashSet<Cell>, grid_size: i32) -> Vec<Cell> {
    let mut occupied: HashSet<Cell> = food.clone();
    for s in snakes {
        occupied.extend(s.body.iter().copied());
    }
    let mut out = Vec::new();
    for r in 0..grid_size {
        for c in 0..grid_size {
            if !occupied.contains(&(r, c)) {
                out.push((r, c));
            }
        }
    }
    out
}

#[cfg(test)]
mod env_tests {
    use super::*;

    #[test]
    fn initial_placement_matches_oracle() {
        let env = SnakeEnv::new(20, 3, false, None);
        assert_eq!(
            Vec::from(env.snakes[A].body.clone()),
            vec![(10, 6), (10, 5), (10, 4)]
        );
        assert_eq!(env.snakes[A].direction, Action::Right);
        assert_eq!(
            Vec::from(env.snakes[B].body.clone()),
            vec![(10, 14), (10, 15), (10, 16)]
        );
        assert_eq!(env.snakes[B].direction, Action::Left);
    }

    #[test]
    fn coast_step_moves_head_and_pops_tail() {
        let mut env = SnakeEnv::new(20, 3, false, None);
        let events = env.advance([Some(Action::Right), Some(Action::Left)], || None);
        assert_eq!(
            Vec::from(env.snakes[A].body.clone()),
            vec![(10, 7), (10, 6), (10, 5)]
        );
        assert_eq!(
            Vec::from(env.snakes[B].body.clone()),
            vec![(10, 13), (10, 14), (10, 15)]
        );
        assert!(!events[A].died && !events[B].died && !env.done);
    }

    #[test]
    fn reverse_action_coasts() {
        let mut env = SnakeEnv::new(20, 3, false, None);
        // A heads Right; commanding Left (reverse) must coast Right, not reverse into itself.
        env.advance([Some(Action::Left), None], || None);
        assert_eq!(env.snakes[A].head(), (10, 7));
        assert_eq!(env.snakes[A].direction, Action::Right);
    }

    #[test]
    fn head_on_collision_is_a_draw() {
        let mut env = SnakeEnv::new(20, 3, false, None);
        let mut events = [StepEvent::default(), StepEvent::default()];
        for _ in 0..4 {
            events = env.advance([Some(Action::Right), Some(Action::Left)], || None);
        }
        // A and B meet at (10,10) on the 4th tick.
        assert_eq!(events[A].death_cause, Some(DeathCause::HeadOn));
        assert_eq!(events[B].death_cause, Some(DeathCause::HeadOn));
        assert!(events[A].drew && events[B].drew);
        assert!(env.done);
    }

    #[test]
    fn eating_grows_snake_and_spawns_replacement() {
        let mut env = SnakeEnv::new(20, 3, false, None);
        env.food.insert((10, 7)); // directly ahead of A
        let mut replacement = vec![(0, 0)];
        let events = env.advance([Some(Action::Right), Some(Action::Left)], || {
            replacement.pop()
        });
        assert!(events[A].ate_food);
        assert_eq!(env.snakes[A].len(), 4); // tail kept
        assert!(env.food.contains(&(0, 0)) && !env.food.contains(&(10, 7)));
    }

    #[test]
    fn wall_death_when_running_off_grid() {
        let mut env = SnakeEnv::new(20, 3, false, None);
        // Place A against the right wall (row 0, clear of B at row 10) so one Right step runs off-grid.
        env.snakes[A].body = VecDeque::from([(0, 19), (0, 18), (0, 17)]);
        env.snakes[A].direction = Action::Right;
        let events = env.advance([Some(Action::Right), None], || None);
        assert_eq!(events[A].death_cause, Some(DeathCause::Wall));
        assert!(!env.snakes[A].alive);
    }

    #[test]
    fn self_body_collision_is_a_death() {
        let mut env = SnakeEnv::new(20, 3, false, None);
        // A folds back on itself: head (5,5) facing Right; turning Up steps onto its own body at (4,5).
        env.snakes[A].body = VecDeque::from([(5, 5), (5, 4), (4, 4), (4, 5), (4, 6)]);
        env.snakes[A].direction = Action::Right;
        let events = env.advance([Some(Action::Up), None], || None);
        assert_eq!(events[A].death_cause, Some(DeathCause::SelfBody));
        assert!(!env.snakes[A].alive);
    }

    #[test]
    fn opponent_body_collision_is_a_kill_and_win() {
        let mut env = SnakeEnv::new(20, 3, false, None);
        // A lies along row 10; B drives its head down into A's body. B dies (OppBody), A is credited
        // the kill and — as the sole survivor — wins; B loses and the game ends (play_to_last = false).
        env.snakes[A].body = VecDeque::from([(10, 5), (10, 4), (10, 3)]);
        env.snakes[A].direction = Action::Right;
        env.snakes[B].body = VecDeque::from([(9, 5), (8, 5), (7, 5)]);
        env.snakes[B].direction = Action::Down;
        let events = env.advance([Some(Action::Right), Some(Action::Down)], || None);
        assert_eq!(events[B].death_cause, Some(DeathCause::OppBody));
        assert!(events[A].killed_opponent && events[A].won && events[B].lost);
        assert!(env.done);
    }

    #[test]
    fn food_lead_wins_outright() {
        let mut env = SnakeEnv::new(20, 3, false, Some(2));
        // A is two apples (length) ahead of B, both alive; the lead triggers an outright win this tick.
        env.snakes[A].body = VecDeque::from([(2, 5), (2, 4), (2, 3), (2, 2), (2, 1)]); // length 5 vs B's 3
        env.snakes[A].direction = Action::Right;
        let events = env.advance([Some(Action::Right), None], || None);
        assert!(events[A].won && events[B].lost);
        assert!(env.done);
    }
}

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

/// Build the egocentric observation for `agent` (0 = A, 1 = B) as a flat `[5 * g * g]` f32 buffer.
pub fn egocentric(env: &SnakeEnv, agent: usize) -> Vec<f32> {
    egocentric_parts(&env.snakes, &env.food, env.grid_size, agent)
}

/// Same as [`egocentric`], operating directly on a (snakes, food) state — used by the search, which
/// builds observations for simulated child states without constructing a full `SnakeEnv`.
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

#[cfg(test)]
mod obs_tests {
    use super::*;

    fn at(obs: &[f32], g: i32, ch: usize, r: i32, c: i32) -> f32 {
        obs[ch * (g * g) as usize + (r as usize) * (g as usize) + (c as usize)]
    }

    #[test]
    fn egocentric_rotates_by_heading() {
        // A faces Right -> k=1 -> (r,c) maps to (edge-c, r). Head (10,6) -> (19-6, 10) = (13,10).
        let env = SnakeEnv::new(20, 3, false, None);
        let obs = egocentric(&env, A);
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
        let mut env = SnakeEnv::new(20, 3, false, None);
        env.food.insert((10, 6)); // same transform as A's head -> (13,10)
        let obs = egocentric(&env, A);
        assert_eq!(at(&obs, 20, CH_FOOD, 13, 10), 1.0);
    }

    #[test]
    fn buffer_has_expected_length() {
        let env = SnakeEnv::new(20, 3, false, None);
        assert_eq!(egocentric(&env, A).len(), N_CHANNELS * 20 * 20);
    }
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

impl SnakeReward {
    pub fn eval(&self, e: &StepEvent) -> f64 {
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
            reward += self.survival;
        }
        reward
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
    fn survival_fires_only_when_survived_to_max_ticks() {
        let r = reward();
        assert_eq!(r.eval(&StepEvent::default()), 0.0); // alive, nothing happened
        let survived = StepEvent {
            survived_to_max_ticks: true,
            ..Default::default()
        };
        assert!((r.eval(&survived) - 0.25).abs() < 1e-12);
    }

    #[test]
    fn a_dead_snake_never_collects_survival() {
        // The died branch returns before the survival term, mirroring MinimalReward's early return.
        let r = reward();
        let dead = StepEvent {
            died: true,
            lost: true,
            survived_to_max_ticks: true,
            ..Default::default()
        };
        assert!((r.eval(&dead) - (-0.5)).abs() < 1e-12); // loss only
    }
}

// ========= The `Snake` Game adapter + `EgocentricSnake` encoder =========

/// Snake's dynamic state: the two snakes and the food. Static config (grid size, rules, reward) lives
/// on `Snake`, so the search/engine can carry just this around per node.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SnakeState {
    pub snakes: [SnakeBody; 2],
    pub food: HashSet<Cell>,
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
}

/// Two-player simultaneous-move snake with environment chance (apple respawn) — the concrete `SnakeEnv`
/// dynamics behind the `Game` trait.
pub struct Snake {
    pub grid_size: i32,
    pub initial_length: usize,
    pub play_to_last: bool,
    pub win_food_lead: Option<usize>,
    pub initial_food_count: usize,
    pub reward: SnakeReward,
}

impl Snake {
    fn env(&self, state: &SnakeState) -> SnakeEnv {
        SnakeEnv::from_parts(
            self.grid_size,
            self.initial_length,
            self.play_to_last,
            self.win_food_lead,
            state.snakes.clone(),
            state.food.clone(),
        )
    }

    /// Spawn one apple at a uniform-random empty cell (the env's true spawn), or nothing if the grid is
    /// full. Build the occupancy set once (food + both bodies, deduped), so the empty count is
    /// `g² − occupied.len()` (no count pass) and lookups are O(1); then walk to the k-th empty cell in
    /// row-major order. A single `rng.below(n)` indexing the row-major empties is identical to
    /// materializing the empties `Vec` and indexing it — same cell, same RNG — but without that `Vec`.
    fn spawn_one(&self, snakes: &[SnakeBody; 2], food: &mut HashSet<Cell>, rng: &mut dyn Rng) {
        let g = self.grid_size;
        let mut occupied: HashSet<Cell> = food.clone();
        for s in snakes {
            occupied.extend(s.body.iter().copied());
        }
        let n = (g * g) as usize - occupied.len();
        if n == 0 {
            return;
        }
        let mut k = rng.below(n);
        for r in 0..g {
            for c in 0..g {
                let cell = (r, c);
                if occupied.contains(&cell) {
                    continue;
                }
                if k == 0 {
                    food.insert(cell);
                    return;
                }
                k -= 1;
            }
        }
    }
}

impl Game for Snake {
    type State = SnakeState;

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

    fn step(&self, state: &SnakeState, actions: &[usize]) -> Transition<SnakeState> {
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
        // deterministic part and the sampled spawn stay separable and are shared by the env and search.
        let mut env = self.env(state);
        let events = env.advance(moves, || None);
        let rewards = vec![self.reward.eval(&events[0]), self.reward.eval(&events[1])];
        Transition {
            next_state: SnakeState {
                snakes: env.snakes,
                food: env.food,
            },
            rewards,
            terminal: env.done,
        }
    }

    fn sample_chance(
        &self,
        state: &SnakeState,
        transition: &Transition<SnakeState>,
        rng: &mut dyn Rng,
        n: usize,
    ) -> Vec<SnakeState> {
        // An eaten apple is the only stochastic event: `step` removed it without respawning, so the
        // count drop = apples eaten. None eaten -> deterministic (empty). Otherwise draw `n` independent
        // realizations, each respawning one uniform-random apple per eaten apple via `spawn_one` — the
        // same spawn the env rollout uses, so search and env share one chance model.
        let next = &transition.next_state;
        let eaten = state.food.len().saturating_sub(next.food.len());
        if eaten == 0 {
            return Vec::new();
        }
        (0..n)
            .map(|_| {
                let mut food = next.food.clone();
                for _ in 0..eaten {
                    self.spawn_one(&next.snakes, &mut food, rng);
                }
                SnakeState {
                    snakes: next.snakes.clone(),
                    food,
                }
            })
            .collect()
    }

    fn initial_state(&self, rng: &mut dyn Rng) -> SnakeState {
        let env = SnakeEnv::new(
            self.grid_size,
            self.initial_length,
            self.play_to_last,
            self.win_food_lead,
        );
        let mut food = HashSet::new();
        for _ in 0..self.initial_food_count {
            self.spawn_one(&env.snakes, &mut food, rng);
        }
        SnakeState {
            snakes: env.snakes,
            food,
        }
    }

    fn truncation_bonus(&self, state: &SnakeState, agent: usize) -> f64 {
        if state.snakes[agent].alive {
            self.reward.survival
        } else {
            0.0
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
            reward: reward(),
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
        let env = SnakeEnv::new(G, 3, false, None);
        SnakeState {
            snakes: env.snakes,
            food: food.iter().copied().collect(),
        }
    }

    #[test]
    fn step_env_equals_step_then_sample_chance() {
        // The unification invariant: the realized env step and the search's chance sampler are the
        // SAME draw. `step_env` must equal `step` then `sample_chance(.., 1)` under the same RNG seed,
        // so the rollout and the search can never use different chance dynamics.
        // Food directly in front of A (faces Right, head (4,2)): Forward eats it, triggering a respawn.
        let g = game();
        let st = initial_state(&[(4, 3)]);
        let actions = [0usize, 0];
        let realized = g.step_env(&st, &actions, &mut TestRng(42));
        let t = g.step(&st, &actions);
        let mut sampled = g.sample_chance(&st, &t, &mut TestRng(42), 1);
        assert_eq!(sampled.len(), 1, "an eaten apple is a chance node");
        assert_eq!(realized.next_state, sampled.swap_remove(0));
        assert_eq!(realized.rewards, t.rewards);
        assert_eq!(realized.terminal, t.terminal);
        assert!((realized.rewards[0] - 1.0).abs() < 1e-12, "A ate one apple");
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
            // Nothing eaten -> no chance node, regardless of how many samples are requested.
            assert!(
                g.sample_chance(&st, &t, &mut TestRng(1), 4).is_empty(),
                "actions {actions:?}"
            );
            // ...so the realized env step is exactly the deterministic step.
            let realized = g.step_env(&st, &actions, &mut TestRng(1));
            assert_eq!(realized.next_state, t.next_state, "actions {actions:?}");
            assert_eq!(realized.rewards, t.rewards);
            assert_eq!(realized.terminal, t.terminal);
        }
    }

    #[test]
    fn sample_chance_draws_independent_valid_respawns() {
        // food_samples > 1 fans the chance node into that many independent draws, each a uniform-random
        // apple on a previously empty cell — not a single deterministic belief.
        let g = game();
        let st = initial_state(&[(4, 3)]);
        let t = g.step(&st, &[0, 0]);
        let samples = g.sample_chance(&st, &t, &mut TestRng(7), 20);
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
        let samples = g.sample_chance(&st, &t, &mut TestRng(12345), n);
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
        // Snakes match the env's initial placement; food sits on empty cells.
        let env = SnakeEnv::new(G, 3, false, None);
        assert_eq!(a.snakes, env.snakes);
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
        let st = initial_state(&[(4, 3)]);
        let t = g.step_env(&st, &[0, 0], &mut TestRng(1));
        assert!((t.rewards[0] - 1.0).abs() < 1e-12, "A ate -> food reward");
        assert!(!t.terminal);
        assert_eq!(
            t.next_state.food.len(),
            1,
            "respawn restored the apple count"
        );
        // A coast with no food/death scores the bare step reward, never the survival bonus.
        let empty = initial_state(&[]);
        let t2 = g.step_env(&empty, &[0, 0], &mut TestRng(1));
        assert_eq!(t2.rewards, vec![0.0, 0.0]);
    }

    #[test]
    fn truncation_bonus_is_survival_for_the_living_only() {
        let mut g = game();
        g.reward.survival = 0.25;
        let st = initial_state(&[]);
        assert!((g.truncation_bonus(&st, 0) - 0.25).abs() < 1e-12);
        let mut dead = st.clone();
        dead.snakes[0].alive = false;
        assert_eq!(g.truncation_bonus(&dead, 0), 0.0);
    }
}
