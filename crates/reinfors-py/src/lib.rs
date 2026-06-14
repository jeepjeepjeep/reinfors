//! PyO3 bindings for reinfors. The compiled module is `reinfors._reinfors`.
//!
//! Phase 1 exposes just enough of the snake core to differential-test it against `CleanSnakeEnv`:
//! construct an env, inject the initial food, step it with explicit actions + spawn cells (the
//! agreed Option B — food placement is an injected input, not reproduced from numpy's RNG), and
//! read back state, per-snake events, and the egocentric observation.

use numpy::{IntoPyArray, PyArray1};
use pyo3::prelude::*;

use reinfors_core::snake::{Cell, DeathCause, SnakeEnv as CoreEnv};
use reinfors_core::Action;

fn action_from_u8(v: u8) -> PyResult<Action> {
    Ok(match v {
        0 => Action::Up,
        1 => Action::Down,
        2 => Action::Left,
        3 => Action::Right,
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "invalid action {v}"
            )))
        }
    })
}

fn action_to_u8(a: Action) -> u8 {
    match a {
        Action::Up => 0,
        Action::Down => 1,
        Action::Left => 2,
        Action::Right => 3,
    }
}

fn cause_str(c: DeathCause) -> &'static str {
    match c {
        DeathCause::Wall => "wall",
        DeathCause::SelfBody => "self_body",
        DeathCause::OppBody => "opp_body",
        DeathCause::HeadOn => "head_on",
    }
}

/// Per-snake tick outcome, mirroring `snake_RL`'s `StepEvent` fields.
type EventTuple = (bool, bool, Option<String>, bool, bool, bool, bool);

#[pyclass]
struct SnakeEnv {
    inner: CoreEnv,
}

#[pymethods]
impl SnakeEnv {
    #[new]
    fn new(
        grid_size: i32,
        initial_length: usize,
        play_to_last: bool,
        win_food_lead: Option<usize>,
    ) -> Self {
        SnakeEnv {
            inner: CoreEnv::new(grid_size, initial_length, play_to_last, win_food_lead),
        }
    }

    /// Replace the food set (used to inject the oracle's initial food before stepping).
    fn set_food(&mut self, cells: Vec<Cell>) {
        self.inner.food = cells.into_iter().collect();
    }

    /// Advance one tick. `actions` is (A, B), each 0..=3 (Up/Down/Left/Right) or None to coast.
    /// `spawns` are the replacement food cells to use, in order, as apples are eaten this tick.
    /// Returns a per-snake [A, B] list of (ate_food, died, death_cause|None, killed, won, lost, drew).
    fn step(
        &mut self,
        actions: (Option<u8>, Option<u8>),
        spawns: Vec<Cell>,
    ) -> PyResult<Vec<EventTuple>> {
        let a0 = actions.0.map(action_from_u8).transpose()?;
        let a1 = actions.1.map(action_from_u8).transpose()?;
        let mut q = spawns.into_iter();
        let events = self.inner.advance([a0, a1], || q.next());
        Ok(events
            .iter()
            .map(|e| {
                (
                    e.ate_food,
                    e.died,
                    e.death_cause.map(|c| cause_str(c).to_string()),
                    e.killed_opponent,
                    e.won,
                    e.lost,
                    e.drew,
                )
            })
            .collect())
    }

    /// (A body, B body), each head-first (matches `list(snake.body)`).
    fn bodies(&self) -> (Vec<Cell>, Vec<Cell>) {
        (
            self.inner.snakes[0].body.iter().copied().collect(),
            self.inner.snakes[1].body.iter().copied().collect(),
        )
    }

    fn directions(&self) -> (u8, u8) {
        (
            action_to_u8(self.inner.snakes[0].direction),
            action_to_u8(self.inner.snakes[1].direction),
        )
    }

    fn alive(&self) -> (bool, bool) {
        (self.inner.snakes[0].alive, self.inner.snakes[1].alive)
    }

    fn food(&self) -> Vec<Cell> {
        self.inner.food.iter().copied().collect()
    }

    fn is_done(&self) -> bool {
        self.inner.done
    }

    /// Egocentric observation for `agent` (0 = A, 1 = B) as a flat [5 * g * g] f32 array.
    fn obs<'py>(&self, py: Python<'py>, agent: usize) -> Bound<'py, PyArray1<f32>> {
        reinfors_core::egocentric(&self.inner, agent).into_pyarray(py)
    }
}

#[pyfunction]
fn core_version() -> &'static str {
    reinfors_core::version()
}

#[pymodule]
fn _reinfors(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(core_version, m)?)?;
    m.add_class::<SnakeEnv>()?;
    Ok(())
}
