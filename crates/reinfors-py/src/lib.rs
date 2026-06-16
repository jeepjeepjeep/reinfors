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
use pyo3::types::PyDict;

use reinfors_core::snake::{Cell, DeathCause, Snake, SnakeEnv as CoreEnv};
use reinfors_core::{
    Action, Engine as CoreEngine, EngineParams, Opponent, Reward, SearchParams, SnakeGame,
};

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
/// One interior TreeStrap target returned to Python: (observation [5*g*g], per-head values [K][A]).
type InteriorOut = (Vec<f32>, Vec<Vec<f64>>);
/// (root action values [K][A], interior TreeStrap targets, search stats).
type SearchOutput = (Vec<Vec<f64>>, Vec<InteriorOut>, StatsTuple);

/// Reject search hyperparameters the core would mishandle (it does not validate them itself).
fn validate_search_params(
    expansion_budget: usize,
    top_k: usize,
    max_depth: i32,
    beta: f64,
    food_samples: usize,
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
    if food_samples < 1 {
        return Err(PyValueError::new_err("food_samples must be >= 1"));
    }
    if !(0.0..=1.0).contains(&beta) {
        return Err(PyValueError::new_err("beta must be in [0, 1]"));
    }
    Ok(())
}

/// Reject degenerate `Engine` rollout parameters (the search block is checked separately).
fn validate_engine_params(
    n_games: usize,
    max_ticks: usize,
    n_heads: usize,
    epsilon: f64,
    outcome_weight: f64,
    bootstrap_p: f64,
) -> PyResult<()> {
    use pyo3::exceptions::PyValueError;
    if n_games < 1 {
        return Err(PyValueError::new_err("n_games must be >= 1"));
    }
    if max_ticks < 1 {
        return Err(PyValueError::new_err("max_ticks must be >= 1"));
    }
    if n_heads < 1 {
        return Err(PyValueError::new_err("n_heads must be >= 1"));
    }
    for (name, v) in [
        ("epsilon", epsilon),
        ("outcome_weight", outcome_weight),
        ("bootstrap_p", bootstrap_p),
    ] {
        if !(0.0..=1.0).contains(&v) {
            return Err(PyValueError::new_err(format!("{name} must be in [0, 1]")));
        }
    }
    Ok(())
}

/// The core's `infer` callback, wrapping the Python network forward. Obs arrive as one flat
/// row-major `[n, dim]` buffer (moved straight into a numpy array — no copy), and per-head values
/// `[n, K, 3]` come back as one flat row-major buffer. The first failure — a Python error, or (when
/// `expected_heads` is set, i.e. the `Engine`) a returned head count that disagrees with the
/// configured `n_heads` — is latched into `callback_err` and zeros (K=1) are returned so the in-flight
/// search unwinds cheaply; the caller checks `callback_err` afterwards and propagates it. The check
/// is on the success path only, so a genuine network error always wins (it is never masked by the
/// head-count check), and it must be here rather than in the core, which cannot tell a real
/// wrong-K output from this very fallback.
fn infer_closure<'a, 'py>(
    py: Python<'py>,
    infer: &'a Bound<'py, PyAny>,
    dim: usize,
    expected_heads: Option<usize>,
    callback_err: &'a mut Option<PyErr>,
) -> impl FnMut(Vec<f32>, usize) -> Vec<f64> + 'a {
    move |obs_flat: Vec<f32>, n: usize| -> Vec<f64> {
        if callback_err.is_some() {
            return vec![0.0; n * 3]; // K=1 fallback (A = 3 relative actions)
        }
        let arr = Array2::from_shape_vec((n, dim), obs_flat)
            .expect("obs batch shape")
            .into_pyarray(py);
        match infer
            .call1((arr,))
            .and_then(|r| r.extract::<PyReadonlyArray3<f64>>())
        {
            Ok(out) => {
                let flat: Vec<f64> = out.as_array().iter().copied().collect(); // flat [n, K, 3]
                if let Some(k) = expected_heads {
                    if n > 0 && flat.len() != n * k * 3 {
                        callback_err.get_or_insert_with(|| {
                            pyo3::exceptions::PyValueError::new_err(format!(
                                "infer returned {} values for {n} rows; expected n_heads ({k}) x 3 \
                                 actions per row — the network's head count must equal n_heads",
                                flat.len()
                            ))
                        });
                        return vec![0.0; n * 3];
                    }
                }
                flat
            }
            Err(e) => {
                *callback_err = Some(e);
                vec![0.0; n * 3]
            }
        }
    }
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
    #[pyo3(signature = (agent, gamma, beta, expansion_budget, top_k, max_depth, reward, opponent, opp_temperature, opp_floor, infer, collect_interior=false, food_samples=1))]
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
        collect_interior: bool,
        food_samples: usize,
    ) -> PyResult<SearchOutput> {
        validate_search_params(expansion_budget, top_k, max_depth, beta, food_samples)?;
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
            food_samples,
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
        let (values, interior, stats) = {
            let mut infer_fn = infer_closure(py, &infer, dim, None, &mut callback_err);
            reinfors_core::selective_search(
                &params,
                snakes,
                food,
                agent,
                collect_interior,
                &mut infer_fn,
            )
        };
        if let Some(e) = callback_err {
            return Err(e);
        }
        Ok((
            values,
            interior,
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
#[pyo3(signature = (envs, agents, gamma, beta, expansion_budget, top_k, max_depth, reward, opponent, opp_temperature, opp_floor, infer, collect_interior=false, food_samples=1))]
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
    collect_interior: bool,
    food_samples: usize,
) -> PyResult<Vec<SearchOutput>> {
    if envs.len() != agents.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "envs and agents must have equal length",
        ));
    }
    if envs.is_empty() {
        return Ok(Vec::new());
    }
    validate_search_params(expansion_budget, top_k, max_depth, beta, food_samples)?;
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
        food_samples,
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
    let results = {
        let mut infer_fn = infer_closure(py, &infer, dim, None, &mut callback_err);
        reinfors_core::selective_search_many(&params, requests, collect_interior, &mut infer_fn)
    };
    if let Some(e) = callback_err {
        return Err(e);
    }
    Ok(results
        .into_iter()
        .map(|(v, interior, s)| (v, interior, (s.max_depth, s.expansions, s.leaves, s.rounds)))
        .collect())
}

/// (observations [M, 5*g*g] f32, per-head targets [M, K, A] f64, per-head bootstrap masks [M, K] f32,
/// telemetry dict). The dict holds `episodes` (a list of `(reward_a, reward_b, length)` for each
/// episode that finished during the call) plus the call's `decisions`, `max_depth`, and per-decision
/// means `mean_leaves`/`mean_rounds`/`mean_expansions`/`mean_sigma`/`mean_disagreement`.
type CollectOutput<'py> = (
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray3<f64>>,
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyDict>,
);

/// Parallel rollout collector: drives N games via the pooled selective search and yields TreeStrap
/// records (z-mixed roots + interior targets), each with a per-head bootstrap mask. Per-game
/// Thompson-head, epsilon, and RNG apple spawns give the games diversity.
#[pyclass]
struct Engine {
    inner: CoreEngine<SnakeGame>,
    dim: usize,
    n_heads: usize,
}

#[pymethods]
impl Engine {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (n_games, grid_size, initial_length, play_to_last, win_food_lead, initial_food_count, gamma, beta, expansion_budget, top_k, max_depth, reward, opponent, opp_temperature, opp_floor, epsilon, max_ticks, n_heads, outcome_weight, interior_targets, bootstrap_p, seed, food_samples=1))]
    fn new(
        n_games: usize,
        grid_size: i32,
        initial_length: usize,
        play_to_last: bool,
        win_food_lead: Option<usize>,
        initial_food_count: usize,
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
        outcome_weight: f64,
        interior_targets: bool,
        bootstrap_p: f64,
        seed: u64,
        food_samples: usize,
    ) -> PyResult<Self> {
        validate_search_params(expansion_budget, top_k, max_depth, beta, food_samples)?;
        validate_engine_params(
            n_games,
            max_ticks,
            n_heads,
            epsilon,
            outcome_weight,
            bootstrap_p,
        )?;
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
        let reward = Reward {
            step: reward.0,
            food: reward.1,
            loss: reward.2,
            draw: reward.3,
            kill: reward.4,
            win: reward.5,
            survival: reward.6,
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
            food_samples,
            reward,
            opponent,
        };
        let game = SnakeGame {
            grid_size,
            initial_length,
            play_to_last,
            win_food_lead,
            initial_food_count,
            reward,
        };
        let engine_params = EngineParams {
            n_games,
            max_ticks,
            epsilon,
            n_heads,
            outcome_weight,
            interior_targets,
            bootstrap_p,
            seed,
        };
        let dim = 5 * (grid_size as usize) * (grid_size as usize);
        Ok(Engine {
            inner: CoreEngine::new(game, &search, engine_params),
            dim,
            n_heads,
        })
    }

    /// Roll forward until at least `n_records` records are gathered. `infer` maps an (N, 5*g*g)
    /// float32 batch to (N, K, 3) float64 per-head action values. Returns the records as a flat
    /// (M, 5*g*g) observation array, an (M, K, 3) per-head target array, and an (M, K) per-head
    /// bootstrap-mask array.
    fn collect<'py>(
        &mut self,
        py: Python<'py>,
        n_records: usize,
        infer: Bound<'_, PyAny>,
    ) -> PyResult<CollectOutput<'py>> {
        let dim = self.dim;
        let mut callback_err: Option<PyErr> = None;
        let (records, stats) = {
            let mut infer_fn =
                infer_closure(py, &infer, dim, Some(self.n_heads), &mut callback_err);
            self.inner.collect(n_records, &mut infer_fn)
        };
        if let Some(e) = callback_err {
            return Err(e);
        }
        let m = records.len();
        let (k, a) = if m > 0 {
            (records[0].1.len(), records[0].1[0].len())
        } else {
            (0, 0)
        };
        let mut obs_flat: Vec<f32> = Vec::with_capacity(m * dim);
        let mut tgt_flat: Vec<f64> = Vec::with_capacity(m * k * a);
        let mut mask_flat: Vec<f32> = Vec::with_capacity(m * k);
        for (obs, tgt, mask) in records {
            obs_flat.extend(obs);
            tgt_flat.extend(tgt.into_iter().flatten());
            mask_flat.extend(mask);
        }
        let obs_arr = Array2::from_shape_vec((m, dim), obs_flat)
            .expect("obs shape")
            .into_pyarray(py);
        let tgt_arr = Array3::from_shape_vec((m, k, a), tgt_flat)
            .expect("target shape")
            .into_pyarray(py);
        let mask_arr = Array2::from_shape_vec((m, k), mask_flat)
            .expect("mask shape")
            .into_pyarray(py);
        let d = (stats.decisions.max(1)) as f64;
        let episodes: Vec<(f64, f64, usize)> = stats
            .episodes
            .iter()
            .map(|e| (e.reward[0], e.reward[1], e.length))
            .collect();
        let telemetry = PyDict::new(py);
        telemetry.set_item("episodes", episodes)?;
        telemetry.set_item("decisions", stats.decisions)?;
        telemetry.set_item("max_depth", stats.max_depth)?;
        telemetry.set_item("mean_leaves", stats.sum_leaves / d)?;
        telemetry.set_item("mean_rounds", stats.sum_rounds / d)?;
        telemetry.set_item("mean_expansions", stats.sum_expansions / d)?;
        telemetry.set_item("mean_sigma", stats.sum_sigma / d)?;
        telemetry.set_item("mean_disagreement", stats.sum_disagreement / d)?;
        Ok((obs_arr, tgt_arr, mask_arr, telemetry))
    }
}

#[pyfunction]
fn core_version() -> &'static str {
    reinfors_core::version()
}

/// AlphaGo-style z-mixing applied to a single trajectory: blend the realized discounted return into
/// each step's executed-action entry of every head. `search_values` is (T, K, A); `actions`/`rewards`
/// are length T; `tail` is (K,) — z's seed past the last step. Returns the blended (T, K, A) targets.
/// Exposed so the differential test can pin this against `EnsembleTreeStrapRunner._blend_outcome_targets`.
#[pyfunction]
fn blend_outcome_targets<'py>(
    py: Python<'py>,
    search_values: PyReadonlyArray3<f64>,
    actions: Vec<usize>,
    rewards: Vec<f64>,
    gamma: f64,
    outcome_weight: f64,
    tail: Vec<f64>,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    use pyo3::exceptions::PyValueError;
    let sv = search_values.as_array();
    let (t, k, a) = (sv.shape()[0], sv.shape()[1], sv.shape()[2]);
    if actions.len() != t || rewards.len() != t {
        return Err(PyValueError::new_err(
            "actions and rewards must have length T",
        ));
    }
    if tail.len() != k {
        return Err(PyValueError::new_err("tail must have length K"));
    }
    let trajectory: Vec<(Vec<Vec<f64>>, usize, f64)> = (0..t)
        .map(|i| {
            let values = (0..k)
                .map(|h| (0..a).map(|j| sv[[i, h, j]]).collect())
                .collect();
            (values, actions[i], rewards[i])
        })
        .collect();
    let blended = reinfors_core::blend_outcome_targets(&trajectory, gamma, outcome_weight, &tail);
    let flat: Vec<f64> = blended.into_iter().flatten().flatten().collect();
    Ok(Array3::from_shape_vec((t, k, a), flat)
        .expect("blend shape")
        .into_pyarray(py))
}

#[pymodule]
fn _reinfors(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(core_version, m)?)?;
    m.add_function(wrap_pyfunction!(selective_search_many, m)?)?;
    m.add_function(wrap_pyfunction!(blend_outcome_targets, m)?)?;
    m.add_class::<SnakeEnv>()?;
    m.add_class::<Engine>()?;
    Ok(())
}
