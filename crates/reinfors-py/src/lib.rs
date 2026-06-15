//! PyO3 bindings for reinfors. The compiled module is `reinfors._reinfors`.
//!
//! Phase 1 exposes just enough of the snake core to differential-test it against `CleanSnakeEnv`:
//! construct an env, inject the initial food, step it with explicit actions + spawn cells (the
//! agreed Option B — food placement is an injected input, not reproduced from numpy's RNG), and
//! read back state, per-snake events, and the egocentric observation.

use std::collections::VecDeque;

use numpy::ndarray::{Array2, Array3};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray3};
use pyo3::prelude::*;

use reinfors_core::snake::{Cell, DeathCause, Snake, SnakeEnv as CoreEnv};
use reinfors_core::{Action, Engine as CoreEngine, EngineConfig, Opponent, Reward, SearchParams};

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
/// (max_depth, expansions, leaves, rounds).
type StatsTuple = (i32, usize, usize, usize);
/// (root action values as [K][A], search stats).
type SearchOutput = (Vec<Vec<f64>>, StatsTuple);

/// Reject search hyperparameters the core would mishandle (it does not validate them itself).
fn validate_search_params(
    expansion_budget: usize,
    top_k: usize,
    max_depth: i32,
    beta: f64,
) -> PyResult<()> {
    use pyo3::exceptions::PyValueError;
    if expansion_budget < 1 {
        return Err(PyValueError::new_err("expansion_budget must be >= 1"));
    }
    if top_k < 1 {
        return Err(PyValueError::new_err("top_k must be >= 1"));
    }
    if max_depth < 1 {
        return Err(PyValueError::new_err("max_depth must be >= 1"));
    }
    if !(0.0..=1.0).contains(&beta) {
        return Err(PyValueError::new_err("beta must be in [0, 1]"));
    }
    Ok(())
}

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

    /// Overwrite both snakes' full state (body head-first, direction 0..=3, alive) — lets the parity
    /// test mirror an arbitrary oracle WorldState into reinfors before searching.
    fn set_snakes(
        &mut self,
        a_body: Vec<Cell>,
        a_dir: u8,
        a_alive: bool,
        b_body: Vec<Cell>,
        b_dir: u8,
        b_alive: bool,
    ) -> PyResult<()> {
        self.inner.snakes[0] = Snake {
            body: VecDeque::from(a_body),
            direction: action_from_u8(a_dir)?,
            alive: a_alive,
        };
        self.inner.snakes[1] = Snake {
            body: VecDeque::from(b_body),
            direction: action_from_u8(b_dir)?,
            alive: b_alive,
        };
        Ok(())
    }

    /// Run a best-first selective-expectimax search from the current state for `agent` (0 or 1).
    /// `reward` is (step, food, loss, draw, kill, win, survival). `opponent` is "uniform" or
    /// "distributional" (the latter using `opp_temperature`/`opp_floor`). `infer` is a callable
    /// mapping an (N, 5*g*g) float32 batch to an (N, K, 3) float64 array of per-head action values.
    /// Returns (action_values[K][3], (max_depth, expansions, leaves, rounds)).
    #[allow(clippy::too_many_arguments)]
    fn selective_search(
        &self,
        py: Python<'_>,
        agent: usize,
        gamma: f64,
        beta: f64,
        expansion_budget: usize,
        top_k: usize,
        max_depth: i32,
        reward: (f64, f64, f64, f64, f64, f64, f64),
        opponent: &str,
        opp_temperature: f64,
        opp_floor: f64,
        infer: Bound<'_, PyAny>,
    ) -> PyResult<SearchOutput> {
        validate_search_params(expansion_budget, top_k, max_depth, beta)?;
        let g = self.inner.grid_size;
        let dim = 5 * (g as usize) * (g as usize);
        let opp_model = match opponent {
            "uniform" => Opponent::Uniform,
            "distributional" => Opponent::Distributional {
                temperature: opp_temperature,
                floor: opp_floor,
            },
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "unknown opponent {other}"
                )))
            }
        };
        let params = SearchParams {
            grid_size: g,
            initial_length: self.inner.initial_length,
            play_to_last: self.inner.play_to_last,
            win_food_lead: self.inner.win_food_lead,
            gamma,
            beta,
            expansion_budget,
            top_k,
            max_depth,
            reward: Reward {
                step: reward.0,
                food: reward.1,
                loss: reward.2,
                draw: reward.3,
                kill: reward.4,
                win: reward.5,
                survival: reward.6,
            },
            opponent: opp_model,
        };
        let snakes = self.inner.snakes.clone();
        let food = self.inner.food.clone();

        let mut callback_err: Option<PyErr> = None;
        let mut infer_fn = |obs_batch: &[Vec<f32>]| -> Vec<Vec<Vec<f64>>> {
            let n = obs_batch.len();
            if callback_err.is_some() {
                return vec![vec![vec![0.0; 3]]; n];
            }
            let flat: Vec<f32> = obs_batch.iter().flatten().copied().collect();
            let arr = Array2::from_shape_vec((n, dim), flat)
                .expect("obs batch shape")
                .into_pyarray(py);
            match infer
                .call1((arr,))
                .and_then(|r| r.extract::<PyReadonlyArray3<f64>>())
            {
                Ok(out) => out
                    .as_array()
                    .outer_iter()
                    .map(|head_mat| head_mat.outer_iter().map(|row| row.to_vec()).collect())
                    .collect(),
                Err(e) => {
                    callback_err = Some(e);
                    vec![vec![vec![0.0; 3]]; n]
                }
            }
        };
        let (values, stats) =
            reinfors_core::selective_search(&params, snakes, food, agent, &mut infer_fn);
        if let Some(e) = callback_err {
            return Err(e);
        }
        Ok((
            values,
            (
                stats.max_depth,
                stats.expansions,
                stats.leaves,
                stats.rounds,
            ),
        ))
    }
}

/// Pooled cross-game selective search: run a search for each `(env, agent)` in lockstep, batching
/// every round's observations across all of them into a single `infer` call (the throughput win).
/// Env config (grid/play_to_last/win_food_lead) is taken from the first env. Returns a per-request
/// list of (action_values[K][3], (max_depth, expansions, leaves, rounds)), in input order.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn selective_search_many(
    py: Python<'_>,
    envs: Vec<Py<SnakeEnv>>,
    agents: Vec<usize>,
    gamma: f64,
    beta: f64,
    expansion_budget: usize,
    top_k: usize,
    max_depth: i32,
    reward: (f64, f64, f64, f64, f64, f64, f64),
    opponent: &str,
    opp_temperature: f64,
    opp_floor: f64,
    infer: Bound<'_, PyAny>,
) -> PyResult<Vec<SearchOutput>> {
    if envs.len() != agents.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "envs and agents must have equal length",
        ));
    }
    if envs.is_empty() {
        return Ok(Vec::new());
    }
    validate_search_params(expansion_budget, top_k, max_depth, beta)?;
    let opp_model = match opponent {
        "uniform" => Opponent::Uniform,
        "distributional" => Opponent::Distributional {
            temperature: opp_temperature,
            floor: opp_floor,
        },
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "unknown opponent {other}"
            )))
        }
    };
    let (g, init_len, play_to_last, win_food_lead) = {
        let e0 = envs[0].borrow(py);
        (
            e0.inner.grid_size,
            e0.inner.initial_length,
            e0.inner.play_to_last,
            e0.inner.win_food_lead,
        )
    };
    // The pooled search applies one shared config (taken from envs[0]) to all requests; a differing
    // grid_size in particular would feed wrong-dimension observations into the search. Require them equal.
    for e in &envs[1..] {
        let r = e.borrow(py);
        if (
            r.inner.grid_size,
            r.inner.initial_length,
            r.inner.play_to_last,
            r.inner.win_food_lead,
        ) != (g, init_len, play_to_last, win_food_lead)
        {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "all envs must share grid_size/initial_length/play_to_last/win_food_lead for a pooled search",
            ));
        }
    }
    let dim = 5 * (g as usize) * (g as usize);
    let params = SearchParams {
        grid_size: g,
        initial_length: init_len,
        play_to_last,
        win_food_lead,
        gamma,
        beta,
        expansion_budget,
        top_k,
        max_depth,
        reward: Reward {
            step: reward.0,
            food: reward.1,
            loss: reward.2,
            draw: reward.3,
            kill: reward.4,
            win: reward.5,
            survival: reward.6,
        },
        opponent: opp_model,
    };
    let mut requests = Vec::with_capacity(envs.len());
    for (e, &a) in envs.iter().zip(agents.iter()) {
        let r = e.borrow(py);
        requests.push((r.inner.snakes.clone(), r.inner.food.clone(), a));
    }

    let mut callback_err: Option<PyErr> = None;
    let mut infer_fn = |obs_batch: &[Vec<f32>]| -> Vec<Vec<Vec<f64>>> {
        let n = obs_batch.len();
        if callback_err.is_some() {
            return vec![vec![vec![0.0; 3]]; n];
        }
        let flat: Vec<f32> = obs_batch.iter().flatten().copied().collect();
        let arr = Array2::from_shape_vec((n, dim), flat)
            .expect("obs batch shape")
            .into_pyarray(py);
        match infer
            .call1((arr,))
            .and_then(|r| r.extract::<PyReadonlyArray3<f64>>())
        {
            Ok(out) => out
                .as_array()
                .outer_iter()
                .map(|head_mat| head_mat.outer_iter().map(|row| row.to_vec()).collect())
                .collect(),
            Err(e) => {
                callback_err = Some(e);
                vec![vec![vec![0.0; 3]]; n]
            }
        }
    };
    let results = reinfors_core::selective_search_many(&params, requests, &mut infer_fn);
    if let Some(e) = callback_err {
        return Err(e);
    }
    Ok(results
        .into_iter()
        .map(|(v, s)| (v, (s.max_depth, s.expansions, s.leaves, s.rounds)))
        .collect())
}

/// (observations [M, 5*g*g] f32, per-head searched targets [M, K, A] f64).
type CollectOutput<'py> = (Bound<'py, PyArray2<f32>>, Bound<'py, PyArray3<f64>>);

/// Parallel rollout collector: drives N games via the pooled selective search and yields TreeStrap
/// records. Per-game Thompson-head + epsilon give the games diversity. Food-free this milestone.
#[pyclass]
struct Engine {
    inner: CoreEngine,
    dim: usize,
}

#[pymethods]
impl Engine {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        n_games: usize,
        grid_size: i32,
        initial_length: usize,
        play_to_last: bool,
        win_food_lead: Option<usize>,
        gamma: f64,
        beta: f64,
        expansion_budget: usize,
        top_k: usize,
        max_depth: i32,
        reward: (f64, f64, f64, f64, f64, f64, f64),
        opponent: &str,
        opp_temperature: f64,
        opp_floor: f64,
        epsilon: f64,
        max_ticks: usize,
        n_heads: usize,
        seed: u64,
    ) -> PyResult<Self> {
        validate_search_params(expansion_budget, top_k, max_depth, beta)?;
        let opponent = match opponent {
            "uniform" => Opponent::Uniform,
            "distributional" => Opponent::Distributional {
                temperature: opp_temperature,
                floor: opp_floor,
            },
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "unknown opponent {other}"
                )))
            }
        };
        let search = SearchParams {
            grid_size,
            initial_length,
            play_to_last,
            win_food_lead,
            gamma,
            beta,
            expansion_budget,
            top_k,
            max_depth,
            reward: Reward {
                step: reward.0,
                food: reward.1,
                loss: reward.2,
                draw: reward.3,
                kill: reward.4,
                win: reward.5,
                survival: reward.6,
            },
            opponent,
        };
        let cfg = EngineConfig {
            n_games,
            grid_size,
            initial_length,
            play_to_last,
            win_food_lead,
            max_ticks,
            epsilon,
            n_heads,
            seed,
            search,
        };
        let dim = 5 * (grid_size as usize) * (grid_size as usize);
        Ok(Engine {
            inner: CoreEngine::new(cfg),
            dim,
        })
    }

    /// Roll forward until at least `n_records` decisions are gathered. `infer` maps an (N, 5*g*g)
    /// float32 batch to (N, K, 3) float64 per-head action values. Returns the records as a flat
    /// (M, 5*g*g) observation array and an (M, K, 3) per-head target array.
    fn collect<'py>(
        &mut self,
        py: Python<'py>,
        n_records: usize,
        infer: Bound<'_, PyAny>,
    ) -> PyResult<CollectOutput<'py>> {
        let dim = self.dim;
        let mut callback_err: Option<PyErr> = None;
        let mut infer_fn = |obs_batch: &[Vec<f32>]| -> Vec<Vec<Vec<f64>>> {
            let n = obs_batch.len();
            if callback_err.is_some() {
                return vec![vec![vec![0.0; 3]]; n];
            }
            let flat: Vec<f32> = obs_batch.iter().flatten().copied().collect();
            let arr = Array2::from_shape_vec((n, dim), flat)
                .expect("obs batch shape")
                .into_pyarray(py);
            match infer
                .call1((arr,))
                .and_then(|r| r.extract::<PyReadonlyArray3<f64>>())
            {
                Ok(out) => out
                    .as_array()
                    .outer_iter()
                    .map(|head_mat| head_mat.outer_iter().map(|row| row.to_vec()).collect())
                    .collect(),
                Err(e) => {
                    callback_err = Some(e);
                    vec![vec![vec![0.0; 3]]; n]
                }
            }
        };
        let (obs, tgt) = self.inner.collect(n_records, &mut infer_fn);
        if let Some(e) = callback_err {
            return Err(e);
        }
        let m = obs.len();
        let (k, a) = if m > 0 {
            (tgt[0].len(), tgt[0][0].len())
        } else {
            (0, 0)
        };
        let obs_flat: Vec<f32> = obs.into_iter().flatten().collect();
        let tgt_flat: Vec<f64> = tgt.into_iter().flatten().flatten().collect();
        let obs_arr = Array2::from_shape_vec((m, dim), obs_flat)
            .expect("obs shape")
            .into_pyarray(py);
        let tgt_arr = Array3::from_shape_vec((m, k, a), tgt_flat)
            .expect("target shape")
            .into_pyarray(py);
        Ok((obs_arr, tgt_arr))
    }
}

#[pyfunction]
fn core_version() -> &'static str {
    reinfors_core::version()
}

#[pymodule]
fn _reinfors(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(core_version, m)?)?;
    m.add_function(wrap_pyfunction!(selective_search_many, m)?)?;
    m.add_class::<SnakeEnv>()?;
    m.add_class::<Engine>()?;
    Ok(())
}
