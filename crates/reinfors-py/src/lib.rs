//! PyO3 bindings for reinfors. The compiled module is `reinfors._reinfors`.
//!
//! Layout (each section is banner-commented below):
//!   * SHARED HELPERS (permanent) — `validate_search_params`, `infer_closure`, `parse_opponent`,
//!     `check_unit`; used by both the unified engine and the parity plumbing.
//!   * UNIFIED ENGINE (permanent) — the `Engine` pyclass + its type-erased dispatch (`ErasedEngine`,
//!     `RecordBatch`, `run_collect`, `EngineImpl`) and the `(GameSpec, PolicySpec, LearnerSpec)`
//!     two-axis factory. One class composes any game/policy/learner.
//!   * PER-GAME CONFIG (permanent) — the `rf.games`/`rf.policies`/`rf.learners` handles: a game's /
//!     algorithm's parameter surface. Adding one here + a factory arm is all a new composition needs.
//!   * SNAKE_RL PARITY PLUMBING (TEMPORARY) — `SnakeEnv`, `selective_search_many`,
//!     `blend_outcome_targets`, and their snake-specific conversions/types. These exist only to
//!     differential-test against the snake_RL oracle; they go when that suite is retired.

use std::collections::{HashMap, VecDeque};

use numpy::ndarray::{Array2, Array3};
use numpy::{IntoPyArray, PyArray1, PyArray3, PyReadonlyArray3};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use reinfors_core::{
    Dqn, DqnRecord, Engine, EngineParams, EpsilonGreedyQ, Game, Learner, Opponent, Policy,
    SearchConfig, SelectiveExpectimax, TreeStrap, TreeStrapRecord,
};
use reinfors_games::snake::{Cell, DeathCause, SnakeBody, SnakeEnv as CoreEnv};
use reinfors_games::{
    Action, Connect4, Connect4Reward, GridWorld, GridWorldReward, SearchParams, Snake, SnakeReward,
};

// ===========================================================================
// SNAKE_RL PARITY PLUMBING (TEMPORARY) — snake action/event conversions + the result types for the
// `SnakeEnv` / `selective_search_many` differential-parity surface below. Deleted with that surface.
// ===========================================================================

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

// ===========================================================================
// SHARED HELPERS (permanent) — used by both the unified engine and the parity plumbing.
// ===========================================================================

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

/// The core's `infer` callback, wrapping the Python network forward. Obs arrive as one flat
/// row-major `[n, dim]` buffer (moved straight into a numpy array — no copy), and per-head values
/// `[n, K, action_count]` come back as one flat row-major buffer. The first failure — a Python error, or (when
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
    action_count: usize,
    expected_heads: Option<usize>,
    callback_err: &'a mut Option<PyErr>,
) -> impl FnMut(Vec<f32>, usize) -> Vec<f64> + 'a {
    move |obs_flat: Vec<f32>, n: usize| -> Vec<f64> {
        if callback_err.is_some() {
            return vec![0.0; n * action_count]; // K=1 fallback
        }
        let arr = Array2::from_shape_vec((n, dim), obs_flat)
            .expect("obs batch shape")
            .into_pyarray(py);
        match infer
            .call1((arr,))
            .and_then(|r| r.extract::<PyReadonlyArray3<f64>>())
        {
            Ok(out) => {
                let flat: Vec<f64> = out.as_array().iter().copied().collect(); // flat [n, K, A]
                if let Some(k) = expected_heads {
                    if n > 0 && flat.len() != n * k * action_count {
                        callback_err.get_or_insert_with(|| {
                            pyo3::exceptions::PyValueError::new_err(format!(
                                "infer returned {} values for {n} rows; expected n_heads ({k}) x \
                                 {action_count} actions per row — the network's head count must \
                                 equal n_heads",
                                flat.len()
                            ))
                        });
                        return vec![0.0; n * action_count];
                    }
                }
                flat
            }
            Err(e) => {
                *callback_err = Some(e);
                vec![0.0; n * action_count]
            }
        }
    }
}

// ===========================================================================
// SNAKE_RL PARITY PLUMBING (TEMPORARY) — the snake env + pooled selective search exposed so the
// differential suite can pin reinfors against snake_RL's oracle. Not part of the engine composition;
// removed when that oracle suite is retired. (`blend_outcome_targets`, near the end, belongs here too.)
// ===========================================================================

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
        reinfors_games::egocentric(&self.inner, agent).into_pyarray(py)
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
        self.inner.snakes[0] = SnakeBody {
            body: VecDeque::from(a_body),
            direction: action_from_u8(a_dir)?,
            alive: a_alive,
        };
        self.inner.snakes[1] = SnakeBody {
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
    #[pyo3(signature = (agent, gamma, beta, expansion_budget, top_k, max_depth, reward, opponent, opp_temperature, opp_floor, infer, collect_interior=false, food_samples=1, seed=0))]
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
        seed: u64,
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
            reward: SnakeReward {
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
            let mut infer_fn = infer_closure(py, &infer, dim, 3, None, &mut callback_err);
            reinfors_games::selective_search(
                &params,
                snakes,
                food,
                agent,
                collect_interior,
                seed,
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
#[pyo3(signature = (envs, agents, gamma, beta, expansion_budget, top_k, max_depth, reward, opponent, opp_temperature, opp_floor, infer, collect_interior=false, food_samples=1, seed=0))]
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
    seed: u64,
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
        reward: SnakeReward {
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
        let mut infer_fn = infer_closure(py, &infer, dim, 3, None, &mut callback_err);
        reinfors_games::selective_search_many(
            &params,
            requests,
            collect_interior,
            seed,
            &mut infer_fn,
        )
    };
    if let Some(e) = callback_err {
        return Err(e);
    }
    Ok(results
        .into_iter()
        .map(|(v, interior, s)| (v, interior, (s.max_depth, s.expansions, s.leaves, s.rounds)))
        .collect())
}

/// Map an opponent name ("uniform"/"distributional") to its `Opponent`, rejecting anything else.
fn parse_opponent(opponent: &str, opp_temperature: f64, opp_floor: f64) -> PyResult<Opponent> {
    match opponent {
        "uniform" => Ok(Opponent::Uniform),
        "distributional" => Ok(Opponent::Distributional {
            temperature: opp_temperature,
            floor: opp_floor,
        }),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unknown opponent {other}"
        ))),
    }
}

/// The unified parallel rollout engine: composes a game + policy + learner handle and drives N games,
/// yielding the learner's records. Holds the composition type-erased, so one class serves every
/// `(game, policy, learner)`. Construct the handles via `rf.games.*` / `rf.policies.*` / `rf.learners.*`.
// ===========================================================================
// UNIFIED ENGINE (permanent) — the `Engine` pyclass + its type-erased dispatch and two-axis factory.
// One class composes any game/policy/learner; only a factory arm (+ a `RecordBatch` impl for a new
// record shape) is needed to extend it — never a per-game/per-family engine type.
// ===========================================================================

#[pyclass(name = "Engine")]
struct PyEngine {
    inner: Box<dyn ErasedEngine>,
}

#[pymethods]
impl PyEngine {
    #[new]
    #[pyo3(signature = (game, policy, learner, n_games, max_ticks, seed=0))]
    fn new(
        game: GameHandle,
        policy: PolicyHandle,
        learner: LearnerHandle,
        n_games: usize,
        max_ticks: usize,
        seed: u64,
    ) -> PyResult<Self> {
        if n_games < 1 || max_ticks < 1 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "n_games and max_ticks must be >= 1",
            ));
        }
        let engine_params = EngineParams {
            n_games,
            max_ticks,
            seed,
        };
        Ok(PyEngine {
            inner: build_engine(game.spec, policy.spec, learner.spec, engine_params)?,
        })
    }

    /// Roll forward until at least `n_records` records are gathered. `infer` maps an (N, C*H*W) float32
    /// batch to (N, K, A) float64 per-head action values. Returns the learner's record batch (today's
    /// tuple shape — a typed `Batch` lands in a later step).
    fn collect<'py>(
        &mut self,
        py: Python<'py>,
        n_records: usize,
        infer: Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.inner.collect(py, n_records, &infer)
    }
}

// ---------------------------------------------------------------------------
// Unified engine composition. `Engine<G, P, L>`'s generics live entirely below the `collect`
// boundary, so a non-generic `collect` is object-safe and one Python `Engine` can hold any
// composition behind `Box<dyn ErasedEngine>`. Construction is a two-axis factory: `build_for_game`
// enumerates (policy, learner) families once (generic over the game), `build_engine` enumerates games
// once — additive `#games + #families`, never the product.
// ---------------------------------------------------------------------------

/// A rollout engine with its concrete `(Game, Policy, Learner)` types erased, so one Python `Engine`
/// holds any composition. `collect` returns the learner's record batch as an opaque Python object.
trait ErasedEngine: Send + Sync {
    fn collect<'py>(
        &mut self,
        py: Python<'py>,
        n_records: usize,
        infer: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>>;
}

/// Marshal a learner's records (with the rollout telemetry) into the Python record batch. One impl per
/// record *shape* — a new learner adds an impl here, never an engine wrapper. The batch is the
/// learner's: TreeStrap → `(obs, targets, masks, telemetry)`; DQN → `(obs, actions, rewards, next_obs,
/// dones, masks, telemetry)`.
trait RecordBatch: Sized {
    fn into_py_batch<'py>(
        records: Vec<Self>,
        py: Python<'py>,
        dim: usize,
        n_heads: usize,
        telemetry: Bound<'py, PyDict>,
    ) -> PyResult<Bound<'py, PyAny>>;
}

impl RecordBatch for TreeStrapRecord {
    fn into_py_batch<'py>(
        records: Vec<Self>,
        py: Python<'py>,
        dim: usize,
        _n_heads: usize,
        telemetry: Bound<'py, PyDict>,
    ) -> PyResult<Bound<'py, PyAny>> {
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
        Ok((obs_arr, tgt_arr, mask_arr, telemetry)
            .into_pyobject(py)?
            .into_any())
    }
}

impl RecordBatch for DqnRecord {
    fn into_py_batch<'py>(
        records: Vec<Self>,
        py: Python<'py>,
        dim: usize,
        n_heads: usize,
        telemetry: Bound<'py, PyDict>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let m = records.len();
        let k = if m > 0 {
            records[0].mask.len()
        } else {
            n_heads
        };
        let mut obs_flat: Vec<f32> = Vec::with_capacity(m * dim);
        let mut next_flat: Vec<f32> = Vec::with_capacity(m * dim);
        let mut mask_flat: Vec<f32> = Vec::with_capacity(m * k);
        let mut actions: Vec<i64> = Vec::with_capacity(m);
        let mut rewards: Vec<f64> = Vec::with_capacity(m);
        let mut dones: Vec<bool> = Vec::with_capacity(m);
        for t in records {
            obs_flat.extend(t.obs);
            next_flat.extend(t.next_obs);
            mask_flat.extend(t.mask);
            actions.push(t.action as i64);
            rewards.push(t.reward);
            dones.push(t.terminal);
        }
        let obs_arr = Array2::from_shape_vec((m, dim), obs_flat)
            .expect("obs shape")
            .into_pyarray(py);
        let next_arr = Array2::from_shape_vec((m, dim), next_flat)
            .expect("next_obs shape")
            .into_pyarray(py);
        let mask_arr = Array2::from_shape_vec((m, k), mask_flat)
            .expect("mask shape")
            .into_pyarray(py);
        Ok((
            obs_arr,
            actions.into_pyarray(py),
            rewards.into_pyarray(py),
            next_arr,
            dones.into_pyarray(py),
            mask_arr,
            telemetry,
        )
            .into_pyobject(py)?
            .into_any())
    }
}

/// Shared rollout: drive any `Engine<G, P, L>` for `n_records`, returning the records and the (uniform)
/// telemetry dict. Search aggregates are zero for a search-less policy (its `fold_telemetry` is a no-op).
fn run_collect<'py, G, P, L>(
    inner: &mut Engine<G, P, L>,
    py: Python<'py>,
    n_records: usize,
    infer: &Bound<'_, PyAny>,
    dim: usize,
    action_count: usize,
    n_heads: usize,
) -> PyResult<(Vec<L::Record>, Bound<'py, PyDict>)>
where
    G: Game + Sync,
    G::State: Send,
    P: Policy,
    L: Learner<P::Evaluation>,
{
    let mut callback_err: Option<PyErr> = None;
    let (records, stats) = {
        let mut infer_fn = infer_closure(
            py,
            infer,
            dim,
            action_count,
            Some(n_heads),
            &mut callback_err,
        );
        inner.collect(n_records, &mut infer_fn)
    };
    if let Some(e) = callback_err {
        return Err(e);
    }
    let d = (stats.decisions.max(1)) as f64;
    let episodes: Vec<(Vec<f64>, usize)> = stats
        .episodes
        .iter()
        .map(|e| (e.reward.clone(), e.length))
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
    Ok((records, telemetry))
}

/// One generic wrapper: a composed engine + its sizing metadata, type-erased behind `ErasedEngine`.
/// A single blanket impl serves *every* `(G, P, L)` whose learner record can marshal to a batch — so a
/// new game, policy, or learner needs no new wrapper or impl, only a factory arm (and, for a genuinely
/// new record shape, a `RecordBatch` impl).
struct EngineImpl<G: Game + Sync, P: Policy, L: Learner<P::Evaluation>> {
    inner: Engine<G, P, L>,
    dim: usize,
    action_count: usize,
    n_heads: usize,
}

impl<G, P, L> ErasedEngine for EngineImpl<G, P, L>
where
    G: Game + Send + Sync + 'static,
    G::State: Send + Sync,
    P: Policy + Send + Sync + 'static,
    P::Evaluation: Send + Sync,
    P::PolicyState: Send + Sync,
    L: Learner<P::Evaluation> + Send + Sync + 'static,
    L::Record: RecordBatch,
{
    fn collect<'py>(
        &mut self,
        py: Python<'py>,
        n_records: usize,
        infer: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let (records, telemetry) = run_collect(
            &mut self.inner,
            py,
            n_records,
            infer,
            self.dim,
            self.action_count,
            self.n_heads,
        )?;
        L::Record::into_py_batch(records, py, self.dim, self.n_heads, telemetry)
    }
}

/// Game configuration, independent of the acting/learning algorithm.
#[derive(Clone)]
enum GameSpec {
    Snake {
        grid_size: i32,
        initial_length: usize,
        initial_food_count: usize,
        play_to_last: bool,
        win_food_lead: Option<usize>,
        reward: SnakeReward,
    },
    Connect4 {
        reward: Connect4Reward,
    },
    GridWorld {
        size: i32,
        goal: (i32, i32),
        reward: GridWorldReward,
    },
}

/// Acting-policy configuration. `n_heads` (ensemble size) lives here — the single source the learner
/// composition reads from.
#[derive(Clone)]
enum PolicySpec {
    SelectiveExpectimax {
        beta: f64,
        expansion_budget: usize,
        top_k: usize,
        max_depth: i32,
        food_samples: usize,
        opponent: Opponent,
        n_heads: usize,
        epsilon: f64,
    },
    EpsilonGreedyQ {
        n_heads: usize,
        epsilon: f64,
    },
}

/// Learning-algorithm configuration. TreeStrap's `gamma` is also threaded into the search config by
/// the factory, so the search and the z-mix share one discount.
#[derive(Clone)]
enum LearnerSpec {
    TreeStrap {
        gamma: f64,
        outcome_weight: f64,
        bootstrap_p: f64,
        interior_targets: bool,
    },
    Dqn {
        bootstrap_p: f64,
    },
}

/// Reject a probability/weight outside `[0, 1]`.
fn check_unit(name: &str, v: f64) -> PyResult<()> {
    if !(0.0..=1.0).contains(&v) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "{name} must be in [0, 1]"
        )));
    }
    Ok(())
}

/// Family axis: given a concrete game, build the engine for a valid (policy, learner) pair. Written
/// once, generic over `G`, so a new family applies to every game; invalid pairings error here. Also
/// where the composed params are validated (the handles store them unchecked).
fn build_for_game<G: Game + Send + Sync + 'static>(
    game: G,
    policy: PolicySpec,
    learner: LearnerSpec,
    engine_params: EngineParams,
) -> PyResult<Box<dyn ErasedEngine>>
where
    G::State: Send + Sync,
{
    let (c, h, w) = game.obs_shape();
    let dim = c * h * w;
    let action_count = game.action_count();
    match (policy, learner) {
        (
            PolicySpec::SelectiveExpectimax {
                beta,
                expansion_budget,
                top_k,
                max_depth,
                food_samples,
                opponent,
                n_heads,
                epsilon,
            },
            LearnerSpec::TreeStrap {
                gamma,
                outcome_weight,
                bootstrap_p,
                interior_targets,
            },
        ) => {
            validate_search_params(expansion_budget, top_k, max_depth, beta, food_samples)?;
            if n_heads < 1 {
                return Err(pyo3::exceptions::PyValueError::new_err("n_heads must be >= 1"));
            }
            check_unit("epsilon", epsilon)?;
            check_unit("outcome_weight", outcome_weight)?;
            check_unit("bootstrap_p", bootstrap_p)?;
            let cfg = SearchConfig {
                gamma,
                beta,
                expansion_budget,
                top_k,
                max_depth,
                food_samples,
                opponent,
            };
            let policy = SelectiveExpectimax::new(cfg, n_heads, epsilon);
            let learner = TreeStrap::new(gamma, outcome_weight, bootstrap_p, interior_targets);
            Ok(Box::new(EngineImpl {
                inner: Engine::new(game, policy, learner, engine_params),
                dim,
                action_count,
                n_heads,
            }))
        }
        (PolicySpec::EpsilonGreedyQ { n_heads, epsilon }, LearnerSpec::Dqn { bootstrap_p }) => {
            if n_heads < 1 {
                return Err(pyo3::exceptions::PyValueError::new_err("n_heads must be >= 1"));
            }
            check_unit("epsilon", epsilon)?;
            check_unit("bootstrap_p", bootstrap_p)?;
            let policy = EpsilonGreedyQ::new(n_heads, epsilon);
            let learner = Dqn::new(n_heads, bootstrap_p);
            Ok(Box::new(EngineImpl {
                inner: Engine::new(game, policy, learner, engine_params),
                dim,
                action_count,
                n_heads,
            }))
        }
        _ => Err(pyo3::exceptions::PyValueError::new_err(
            "incompatible policy/learner: TreeStrap pairs with SelectiveExpectimax, Dqn with EpsilonGreedyQ",
        )),
    }
}

/// Game axis: pick the concrete game from `GameSpec`, then dispatch to `build_for_game`. One arm per
/// game; each instantly works with every family.
fn build_engine(
    game: GameSpec,
    policy: PolicySpec,
    learner: LearnerSpec,
    engine_params: EngineParams,
) -> PyResult<Box<dyn ErasedEngine>> {
    match game {
        GameSpec::Snake {
            grid_size,
            initial_length,
            initial_food_count,
            play_to_last,
            win_food_lead,
            reward,
        } => build_for_game(
            Snake {
                grid_size,
                initial_length,
                play_to_last,
                win_food_lead,
                initial_food_count,
                reward,
            },
            policy,
            learner,
            engine_params,
        ),
        GameSpec::Connect4 { reward } => {
            build_for_game(Connect4 { reward }, policy, learner, engine_params)
        }
        GameSpec::GridWorld { size, goal, reward } => build_for_game(
            GridWorld { size, goal, reward },
            policy,
            learner,
            engine_params,
        ),
    }
}

// ===========================================================================
// PER-GAME CONFIG (permanent) — the `rf.games.*` / `rf.policies.*` / `rf.learners.*` handles: a
// game's / algorithm's parameter surface, which has to be expressed somewhere. One handle pyclass per
// axis, each carrying a spec, with a staticmethod per variant. The Python layer binds those
// staticmethods into namespaces (rf.games.Snake = GameHandle.Snake) and adds the name registry +
// make_* glue. The unified `Engine` extracts the specs and composes them via the factory above.
// ===========================================================================

/// `rf.Reward` — a generic named-weight reward: a `{component: weight}` map that a game interprets.
/// Each game declares the components it understands; an unrecognized key is an error (see
/// `resolve_reward`), so `rf.Reward(food=1.0)` for a non-snake game is rejected rather than ignored.
#[pyclass]
#[derive(Clone, Default)]
struct Reward {
    weights: HashMap<String, f64>,
}

#[pymethods]
impl Reward {
    #[new]
    #[pyo3(signature = (**weights))]
    fn new(weights: Option<HashMap<String, f64>>) -> Self {
        Reward {
            weights: weights.unwrap_or_default(),
        }
    }
}

/// Resolve a generic `Reward` for a game whose components + defaults are `schema`: every key the
/// caller passed must be one of the schema's components (any unknown key is an error listing the valid
/// set), and each component reads its weight, falling back to its schema default. Returns the resolved
/// weights in schema order.
fn resolve_reward(reward: Option<Reward>, schema: &[(&str, f64)]) -> PyResult<Vec<f64>> {
    let weights = reward.map(|r| r.weights).unwrap_or_default();
    if let Some(bad) = weights.keys().find(|k| !schema.iter().any(|(s, _)| s == k)) {
        let valid: Vec<&str> = schema.iter().map(|(s, _)| *s).collect();
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unknown reward key {bad:?}; valid keys for this game: {valid:?}"
        )));
    }
    Ok(schema
        .iter()
        .map(|(k, default)| weights.get(*k).copied().unwrap_or(*default))
        .collect())
}

/// Game handle (`rf.games.Snake` / `.Connect4` / `.GridWorld`).
#[pyclass]
#[derive(Clone)]
struct GameHandle {
    spec: GameSpec,
}

#[pymethods]
impl GameHandle {
    #[staticmethod]
    #[pyo3(signature = (grid_size=20, initial_length=3, food=3, play_to_last=true, win_food_lead=None, reward=None))]
    #[pyo3(name = "Snake")]
    fn snake(
        grid_size: i32,
        initial_length: usize,
        food: usize,
        play_to_last: bool,
        win_food_lead: Option<usize>,
        reward: Option<Reward>,
    ) -> PyResult<Self> {
        let r = resolve_reward(
            reward,
            &[
                ("step", 0.0),
                ("food", 0.0),
                ("loss", 0.0),
                ("draw", 0.0),
                ("kill", 0.0),
                ("win", 0.0),
                ("survival", 0.0),
            ],
        )?;
        Ok(GameHandle {
            spec: GameSpec::Snake {
                grid_size,
                initial_length,
                initial_food_count: food,
                play_to_last,
                win_food_lead,
                reward: SnakeReward {
                    step: r[0],
                    food: r[1],
                    loss: r[2],
                    draw: r[3],
                    kill: r[4],
                    win: r[5],
                    survival: r[6],
                },
            },
        })
    }

    #[staticmethod]
    #[pyo3(signature = (reward=None))]
    #[pyo3(name = "Connect4")]
    fn connect4(reward: Option<Reward>) -> PyResult<Self> {
        let r = resolve_reward(reward, &[("win", 1.0), ("loss", -1.0), ("draw", 0.0)])?;
        Ok(GameHandle {
            spec: GameSpec::Connect4 {
                reward: Connect4Reward {
                    win: r[0],
                    loss: r[1],
                    draw: r[2],
                },
            },
        })
    }

    #[staticmethod]
    #[pyo3(signature = (size=5, goal_row=4, goal_col=4, reward=None))]
    #[pyo3(name = "GridWorld")]
    fn gridworld(
        size: i32,
        goal_row: i32,
        goal_col: i32,
        reward: Option<Reward>,
    ) -> PyResult<Self> {
        let r = resolve_reward(reward, &[("step", 0.0), ("goal", 1.0)])?;
        Ok(GameHandle {
            spec: GameSpec::GridWorld {
                size,
                goal: (goal_row, goal_col),
                reward: GridWorldReward {
                    step: r[0],
                    goal: r[1],
                },
            },
        })
    }
}

/// Policy handle (`rf.policies.SelectiveExpectimax` / `.EpsilonGreedyQ`).
#[pyclass]
#[derive(Clone)]
struct PolicyHandle {
    spec: PolicySpec,
}

#[pymethods]
impl PolicyHandle {
    #[staticmethod]
    #[pyo3(signature = (expansion_budget=64, top_k=8, max_depth=12, beta=1.0, food_samples=1, n_heads=1, epsilon=0.0, opponent="uniform", opp_temperature=1.0, opp_floor=0.0))]
    #[pyo3(name = "SelectiveExpectimax")]
    #[allow(clippy::too_many_arguments)]
    fn selective_expectimax(
        expansion_budget: usize,
        top_k: usize,
        max_depth: i32,
        beta: f64,
        food_samples: usize,
        n_heads: usize,
        epsilon: f64,
        opponent: &str,
        opp_temperature: f64,
        opp_floor: f64,
    ) -> PyResult<Self> {
        Ok(PolicyHandle {
            spec: PolicySpec::SelectiveExpectimax {
                beta,
                expansion_budget,
                top_k,
                max_depth,
                food_samples,
                opponent: parse_opponent(opponent, opp_temperature, opp_floor)?,
                n_heads,
                epsilon,
            },
        })
    }

    #[staticmethod]
    #[pyo3(signature = (n_heads=1, epsilon=0.1))]
    #[pyo3(name = "EpsilonGreedyQ")]
    fn epsilon_greedy_q(n_heads: usize, epsilon: f64) -> Self {
        PolicyHandle {
            spec: PolicySpec::EpsilonGreedyQ { n_heads, epsilon },
        }
    }
}

/// Learner handle (`rf.learners.TreeStrap` / `.Dqn`).
#[pyclass]
#[derive(Clone)]
struct LearnerHandle {
    spec: LearnerSpec,
}

#[pymethods]
impl LearnerHandle {
    #[staticmethod]
    #[pyo3(signature = (gamma=0.99, outcome_weight=0.0, bootstrap_p=1.0, interior_targets=false))]
    #[pyo3(name = "TreeStrap")]
    fn treestrap(
        gamma: f64,
        outcome_weight: f64,
        bootstrap_p: f64,
        interior_targets: bool,
    ) -> Self {
        LearnerHandle {
            spec: LearnerSpec::TreeStrap {
                gamma,
                outcome_weight,
                bootstrap_p,
                interior_targets,
            },
        }
    }

    #[staticmethod]
    #[pyo3(signature = (bootstrap_p=1.0))]
    #[pyo3(name = "Dqn")]
    fn dqn(bootstrap_p: f64) -> Self {
        LearnerHandle {
            spec: LearnerSpec::Dqn { bootstrap_p },
        }
    }
}

#[pyfunction]
fn core_version() -> &'static str {
    reinfors_core::version()
}

// ===========================================================================
// SNAKE_RL PARITY PLUMBING (TEMPORARY) — z-mix helper exposed only for the differential test; removed
// with the rest of the parity surface.
// ===========================================================================

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
    let blended =
        reinfors_core::TreeStrap::blend_outcome_targets(&trajectory, gamma, outcome_weight, &tail);
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
    m.add_class::<PyEngine>()?;
    m.add_class::<GameHandle>()?;
    m.add_class::<PolicyHandle>()?;
    m.add_class::<LearnerHandle>()?;
    m.add_class::<Reward>()?;
    Ok(())
}
