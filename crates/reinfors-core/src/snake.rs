//! Two-player snake dynamics, ported to match `snake_RL`'s `CleanSnakeEnv` exactly (the oracle).
//!
//! All logic here is deterministic integer arithmetic, so given identical actions and food-spawn
//! cells it produces bit-identical trajectories to the Python env. Food placement is the only
//! nondeterminism in the Python env; here it is an *injected* input (`next_food`), so the
//! differential test can replay a captured Python rollout's spawns (the agreed Option B).

use std::collections::{HashMap, HashSet, VecDeque};

use crate::action::Action;

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

#[derive(Clone, Debug)]
pub struct Snake {
    pub body: VecDeque<Cell>, // body[0] is the head, body[len-1] the tail
    pub direction: Action,
    pub alive: bool,
}

impl Snake {
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
}

pub struct SnakeEnv {
    pub grid_size: i32,
    pub initial_length: usize,
    pub play_to_last: bool,
    pub win_food_lead: Option<usize>,
    pub snakes: [Snake; 2],
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
        snakes: [Snake; 2],
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
    fn initial_snakes(grid_size: i32, length: usize) -> [Snake; 2] {
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
            Snake {
                body: a_body,
                direction: Action::Right,
                alive: true,
            },
            Snake {
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

#[cfg(test)]
mod tests {
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
}
