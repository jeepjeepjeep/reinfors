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
