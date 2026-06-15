//! Absolute grid actions, mirroring `snake_RL`'s `Action` enum and its delta/opposite tables.

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
}
