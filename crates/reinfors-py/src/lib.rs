//! PyO3 bindings for reinfors. The compiled module is `reinfors._reinfors`.
//!
//! Layout (each section is banner-commented below):
//!   * SHARED HELPERS (permanent) — `validate_search_params`, `infer_closure`, `parse_opponent`,
//!     `check_unit`; used by both the unified engine and the parity plumbing.
//!   * UNIFIED ENGINE (permanent) — the `Engine` pyclass + its type-erased dispatch (`ErasedEngine`,
//!     `RecordBatch`, `run_collect`, `EngineImpl`) and the `(GameSpec, PolicySpec, LearnerSpec)`
//!     two-axis factory. One class composes any game/policy/learner.
//!   * UNIFIED ENV (permanent) — the `Env` pyclass: a caller-driven single-game instance for play /
//!     evaluation, mirroring the engine's type-erasure.
//!   * PER-GAME CONFIG (permanent) — the `rf.games`/`rf.policies`/`rf.learners` handles: a game's /
//!     algorithm's parameter surface. Adding one here + a factory arm is all a new composition needs.

use std::collections::HashMap;

use numpy::ndarray::{Array2, Array3, ArrayD, IxDyn};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3, PyArrayDyn, PyReadonlyArray3};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use reinfors_core::{
    Dqn, DqnRecord, Engine, EngineParams, Env, EpsilonGreedyQ, Game, Learner, Opponent, Policy,
    Reward, SearchConfig, SelectiveExpectimax, Space, StateEncoder, TreeStrap, TreeStrapRecord,
};
use reinfors_games::snake::{Cell, DeathCause};
use reinfors_games::{
    Action, Connect4, Connect4Event, Connect4Planes, Connect4Reward, Connect4State,
    EgocentricSnake, GridEvent, GridState, GridWorld, GridWorldPlanes, GridWorldReward, Snake,
    SnakeReward, SnakeState, StepEvent,
};

/// Absolute `Action` -> its `u8` code (Up/Down/Left/Right = 0/1/2/3), for native-state marshalling.
fn action_to_u8(a: Action) -> u8 {
    match a {
        Action::Up => 0,
        Action::Down => 1,
        Action::Left => 2,
        Action::Right => 3,
    }
}

// ===========================================================================
// SHARED HELPERS (permanent) — used by the unified engine + env.
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

/// Parse the opponent-model string a policy handle carries into the core `Opponent`.
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
    #[pyo3(signature = (game, reward, policy, learner, n_games, max_ticks, seed=0))]
    fn new(
        game: GameHandle,
        reward: Option<PyReward>,
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
            inner: build_engine(game.spec, reward, policy.spec, learner.spec, engine_params)?,
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

/// `engine.collect` result for the TreeStrap family: per-head searched targets + a bootstrap mask.
#[pyclass]
struct TreeStrapBatch {
    #[pyo3(get)]
    obs: Py<PyArray2<f32>>, // (M, C*H*W)
    #[pyo3(get)]
    targets: Py<PyArray3<f64>>, // (M, K, A)
    #[pyo3(get)]
    masks: Py<PyArray2<f32>>, // (M, K)
    #[pyo3(get)]
    telemetry: Py<PyDict>,
}

/// `engine.collect` result for the DQN family: off-policy transitions + a bootstrap mask.
#[pyclass]
struct DqnBatch {
    #[pyo3(get)]
    obs: Py<PyArray2<f32>>, // (M, dim)
    #[pyo3(get)]
    actions: Py<PyArray1<i64>>, // (M,)
    #[pyo3(get)]
    rewards: Py<PyArray1<f64>>, // (M,)
    #[pyo3(get)]
    next_obs: Py<PyArray2<f32>>, // (M, dim)
    #[pyo3(get)]
    dones: Py<PyArray1<bool>>, // (M,)
    #[pyo3(get)]
    masks: Py<PyArray2<f32>>, // (M, K)
    #[pyo3(get)]
    telemetry: Py<PyDict>,
}

#[pymethods]
impl TreeStrapBatch {
    fn __len__(&self) -> usize {
        4
    }
    /// Also unpacks positionally: `obs, targets, masks, telemetry = batch`.
    fn __getitem__<'py>(&self, py: Python<'py>, i: usize) -> PyResult<Bound<'py, PyAny>> {
        Ok(match i {
            0 => self.obs.bind(py).clone().into_any(),
            1 => self.targets.bind(py).clone().into_any(),
            2 => self.masks.bind(py).clone().into_any(),
            3 => self.telemetry.bind(py).clone().into_any(),
            _ => {
                return Err(pyo3::exceptions::PyIndexError::new_err(
                    "TreeStrapBatch index out of range",
                ))
            }
        })
    }
}

#[pymethods]
impl DqnBatch {
    fn __len__(&self) -> usize {
        7
    }
    /// Also unpacks positionally: `obs, actions, rewards, next_obs, dones, masks, telemetry = batch`.
    fn __getitem__<'py>(&self, py: Python<'py>, i: usize) -> PyResult<Bound<'py, PyAny>> {
        Ok(match i {
            0 => self.obs.bind(py).clone().into_any(),
            1 => self.actions.bind(py).clone().into_any(),
            2 => self.rewards.bind(py).clone().into_any(),
            3 => self.next_obs.bind(py).clone().into_any(),
            4 => self.dones.bind(py).clone().into_any(),
            5 => self.masks.bind(py).clone().into_any(),
            6 => self.telemetry.bind(py).clone().into_any(),
            _ => {
                return Err(pyo3::exceptions::PyIndexError::new_err(
                    "DqnBatch index out of range",
                ))
            }
        })
    }
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
        Ok(Bound::new(
            py,
            TreeStrapBatch {
                obs: obs_arr.unbind(),
                targets: tgt_arr.unbind(),
                masks: mask_arr.unbind(),
                telemetry: telemetry.unbind(),
            },
        )?
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
        Ok(Bound::new(
            py,
            DqnBatch {
                obs: obs_arr.unbind(),
                actions: actions.into_pyarray(py).unbind(),
                rewards: rewards.into_pyarray(py).unbind(),
                next_obs: next_arr.unbind(),
                dones: dones.into_pyarray(py).unbind(),
                masks: mask_arr.unbind(),
                telemetry: telemetry.unbind(),
            },
        )?
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

/// Game configuration (rules only — the reward is a separate handle, resolved per game at `Engine`
/// construction), independent of the acting/learning algorithm.
#[derive(Clone)]
enum GameSpec {
    Snake {
        grid_size: i32,
        initial_length: usize,
        initial_food_count: usize,
        play_to_last: bool,
        win_food_lead: Option<usize>,
    },
    Connect4,
    GridWorld {
        size: i32,
        goal: (i32, i32),
    },
}

impl GameSpec {
    /// The game's `(observation, action)` spaces — builds the concrete game and reads the trait
    /// defaults. Lets a caller size a network from a handle (`game.observation_space()`) without
    /// hard-coding any game's dimensions.
    fn spaces(&self) -> (Space, Space) {
        // Observation space comes from the game's default encoder (representation); the action space
        // is the game's (rules).
        fn of<G: Game>(game: G, enc: &dyn StateEncoder<State = G::State>) -> (Space, Space) {
            (enc.observation_space(), game.action_space())
        }
        match *self {
            GameSpec::Snake {
                grid_size,
                initial_length,
                initial_food_count,
                play_to_last,
                win_food_lead,
            } => of(
                Snake {
                    grid_size,
                    initial_length,
                    play_to_last,
                    win_food_lead,
                    initial_food_count,
                },
                &EgocentricSnake { grid_size },
            ),
            GameSpec::Connect4 => of(Connect4, &Connect4Planes),
            GameSpec::GridWorld { size, goal } => {
                of(GridWorld { size, goal }, &GridWorldPlanes { size, goal })
            }
        }
    }
}

/// Resolve the generic `rf.Reward` weights against the game's schema into the concrete reward struct,
/// boxed as the `dyn Reward` handle the engine threads. One arm per game (the reward keys + defaults
/// are the game's). Used at `Engine` construction, where both game and reward are known.
fn build_reward(game: &GameSpec, reward: Option<PyReward>) -> PyResult<RewardBox> {
    Ok(match game {
        GameSpec::Snake { .. } => {
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
            RewardBox::Snake(SnakeReward {
                step: r[0],
                food: r[1],
                loss: r[2],
                draw: r[3],
                kill: r[4],
                win: r[5],
                survival: r[6],
            })
        }
        GameSpec::Connect4 => {
            let r = resolve_reward(reward, &[("win", 1.0), ("loss", -1.0), ("draw", 0.0)])?;
            RewardBox::Connect4(Connect4Reward {
                win: r[0],
                loss: r[1],
                draw: r[2],
            })
        }
        GameSpec::GridWorld { .. } => {
            let r = resolve_reward(reward, &[("step", 0.0), ("goal", 1.0)])?;
            RewardBox::GridWorld(GridWorldReward {
                step: r[0],
                goal: r[1],
            })
        }
    })
}

/// The resolved per-game reward, kept concrete (not yet `Box<dyn Reward>`) so each `build_engine` arm
/// can pair it with the matching game type for `Engine::new`.
enum RewardBox {
    Snake(SnakeReward),
    Connect4(Connect4Reward),
    GridWorld(GridWorldReward),
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
    enc: Box<dyn StateEncoder<State = G::State>>,
    reward: Box<dyn Reward<Event = G::Event, State = G::State>>,
    policy: PolicySpec,
    learner: LearnerSpec,
    engine_params: EngineParams,
) -> PyResult<Box<dyn ErasedEngine>>
where
    G::State: Send + Sync,
{
    let (c, h, w) = enc.obs_shape();
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
                inner: Engine::new(game, enc, reward, policy, learner, engine_params),
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
                inner: Engine::new(game, enc, reward, policy, learner, engine_params),
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
    reward: Option<PyReward>,
    policy: PolicySpec,
    learner: LearnerSpec,
    engine_params: EngineParams,
) -> PyResult<Box<dyn ErasedEngine>> {
    let reward = build_reward(&game, reward)?;
    match (game, reward) {
        (
            GameSpec::Snake {
                grid_size,
                initial_length,
                initial_food_count,
                play_to_last,
                win_food_lead,
            },
            RewardBox::Snake(reward),
        ) => build_for_game(
            Snake {
                grid_size,
                initial_length,
                play_to_last,
                win_food_lead,
                initial_food_count,
            },
            Box::new(EgocentricSnake { grid_size }),
            Box::new(reward),
            policy,
            learner,
            engine_params,
        ),
        (GameSpec::Connect4, RewardBox::Connect4(reward)) => build_for_game(
            Connect4,
            Box::new(Connect4Planes),
            Box::new(reward),
            policy,
            learner,
            engine_params,
        ),
        (GameSpec::GridWorld { size, goal }, RewardBox::GridWorld(reward)) => build_for_game(
            GridWorld { size, goal },
            Box::new(GridWorldPlanes { size, goal }),
            Box::new(reward),
            policy,
            learner,
            engine_params,
        ),
        // `build_reward` returns the matching `RewardBox` arm for each game, so other pairings are unreachable.
        _ => unreachable!("build_reward returns the reward variant matching the game"),
    }
}

// ===========================================================================
// UNIFIED ENV (permanent) — `rf.Env`, the caller-driven single-game instance. Mirrors the engine's
// type-erasure: one `Env` pyclass holds any game behind `Box<dyn ErasedEnv>`, built via a per-game
// arm that pairs the game with its default encoder. Drives one game move-by-move (play / eval).
// ===========================================================================

#[pyclass(name = "Env")]
struct PyEnv {
    inner: Box<dyn ErasedEnv>,
}

#[pymethods]
impl PyEnv {
    #[new]
    #[pyo3(signature = (game, seed=0))]
    fn new(game: GameHandle, seed: u64) -> Self {
        PyEnv {
            inner: build_env(game.spec, seed),
        }
    }

    /// Start a new episode.
    fn reset(&mut self) {
        self.inner.reset();
    }

    /// Whether the current episode has ended.
    fn done(&self) -> bool {
        self.inner.done()
    }

    fn num_agents(&self) -> usize {
        self.inner.num_agents()
    }

    fn action_count(&self) -> usize {
        self.inner.action_count()
    }

    /// Agents that must supply an action this tick (one mover for a sequential game, all live agents
    /// for a simultaneous one); empty once the episode is over.
    fn active_agents(&self) -> Vec<usize> {
        self.inner.active_agents()
    }

    fn legal_actions(&self, agent: usize) -> Vec<usize> {
        self.inner.legal_actions(agent)
    }

    /// The encoded observation for `agent` as a `(C, H, W)` float32 array (the value-network view).
    fn observe<'py>(&self, py: Python<'py>, agent: usize) -> Bound<'py, PyArray3<f32>> {
        self.inner.observe(py, agent)
    }

    /// The observation `Space` — so a net can be sized/validated from the env alone.
    fn observation_space<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.inner.observation_space(py)
    }

    /// The native game state as an interpretable dict (game-specific: snake → bodies/food/directions/
    /// alive; connect4 → board/turn/done; gridworld → pos/done) — for rendering and human play.
    fn state<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.inner.state(py)
    }

    /// Advance one tick with `actions`, a `{agent: action}` map naming exactly the agents that act
    /// this tick (see `active_agents()`). Returns this tick's per-agent events (game-specific objects);
    /// a game-aware caller reads the outcome from them (`Env` holds no reward).
    fn step<'py>(
        &mut self,
        py: Python<'py>,
        actions: HashMap<usize, usize>,
    ) -> PyResult<Vec<Bound<'py, PyAny>>> {
        // `actions` must name exactly the active agents — reject anything else loudly, since a missing
        // active agent or a stray inactive one would let an unintended default move silently advance
        // (and corrupt) the episode.
        let active = self.inner.active_agents();
        if active.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "no agents to act — the episode is over; call reset()",
            ));
        }
        if let Some(&agent) = actions.keys().find(|a| !active.contains(a)) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "agent {agent} is not active this tick; active agents: {active:?}"
            )));
        }
        if let Some(&agent) = active.iter().find(|a| !actions.contains_key(a)) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "missing action for active agent {agent}; active agents: {active:?}"
            )));
        }
        let mut joint = vec![0usize; self.inner.num_agents()];
        for (agent, action) in actions {
            joint[agent] = action;
        }
        self.inner.step(py, joint)
    }
}

/// Marshal a game's native `State` into an interpretable Python object (for rendering / observers,
/// e.g. an interactive game). One impl per game state — the genuinely game-specific part of the `Env`
/// binding (the rest is generic). A binding-local trait over the foreign state types.
trait NativeState {
    fn to_py<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>>;
}

impl NativeState for SnakeState {
    fn to_py<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        let bodies: Vec<Vec<Cell>> = self
            .snakes
            .iter()
            .map(|s| s.body.iter().copied().collect())
            .collect();
        d.set_item("bodies", bodies)?;
        d.set_item("food", self.food.iter().copied().collect::<Vec<Cell>>())?;
        let directions: Vec<u8> = self
            .snakes
            .iter()
            .map(|s| action_to_u8(s.direction))
            .collect();
        d.set_item("directions", directions)?;
        let alive: Vec<bool> = self.snakes.iter().map(|s| s.alive).collect();
        d.set_item("alive", alive)?;
        Ok(d)
    }
}

impl NativeState for Connect4State {
    fn to_py<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("board", self.board())?; // [row][col] cell codes, row 0 = bottom
        d.set_item("turn", self.turn())?;
        d.set_item("done", self.is_done())?;
        Ok(d)
    }
}

impl NativeState for GridState {
    fn to_py<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("pos", self.pos)?;
        d.set_item("done", self.done)?;
        Ok(d)
    }
}

/// Marshal a game's per-agent `Event` (the outcome of a tick that `step` returns — what a reward maps
/// to a scalar) into an interpretable Python object, so a game-aware caller can read the outcome (e.g.
/// the win/loss/draw verdict in play/eval) without the `Env` holding a reward. One impl per event type.
trait NativeEvent {
    fn to_py<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>>;
}

impl NativeEvent for StepEvent {
    fn to_py<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let d = PyDict::new(py);
        d.set_item("ate_food", self.ate_food)?;
        d.set_item("died", self.died)?;
        d.set_item("killed_opponent", self.killed_opponent)?;
        d.set_item("won", self.won)?;
        d.set_item("lost", self.lost)?;
        d.set_item("drew", self.drew)?;
        let cause = self.death_cause.map(|c| match c {
            DeathCause::Wall => "wall",
            DeathCause::SelfBody => "self",
            DeathCause::OppBody => "opponent",
            DeathCause::HeadOn => "head_on",
        });
        d.set_item("death_cause", cause)?;
        Ok(d.into_any())
    }
}

impl NativeEvent for Connect4Event {
    fn to_py<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let s = match self {
            Connect4Event::Ongoing => "ongoing",
            Connect4Event::Win => "win",
            Connect4Event::Loss => "loss",
            Connect4Event::Draw => "draw",
        };
        Ok(s.into_pyobject(py)?.into_any())
    }
}

impl NativeEvent for GridEvent {
    fn to_py<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let d = PyDict::new(py);
        d.set_item("reached_goal", self.reached_goal)?;
        Ok(d.into_any())
    }
}

/// A single-game `Env` with its concrete `Game` erased, so one Python `Env` holds any game.
trait ErasedEnv: Send + Sync {
    fn reset(&mut self);
    fn done(&self) -> bool;
    fn num_agents(&self) -> usize;
    fn action_count(&self) -> usize;
    fn active_agents(&self) -> Vec<usize>;
    fn legal_actions(&self, agent: usize) -> Vec<usize>;
    fn observe<'py>(&self, py: Python<'py>, agent: usize) -> Bound<'py, PyArray3<f32>>;
    fn observation_space<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>>;
    fn state<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>>;
    /// Advance one tick; returns this tick's per-agent events as a Python list (game-specific objects).
    fn step<'py>(
        &mut self,
        py: Python<'py>,
        actions: Vec<usize>,
    ) -> PyResult<Vec<Bound<'py, PyAny>>>;
}

struct EnvImpl<G: Game> {
    inner: Env<G>,
    obs_shape: (usize, usize, usize),
}

impl<G> ErasedEnv for EnvImpl<G>
where
    G: Game + Send + Sync + 'static,
    G::State: Send + Sync + NativeState,
    G::Event: NativeEvent,
{
    fn reset(&mut self) {
        self.inner.reset();
    }
    fn done(&self) -> bool {
        self.inner.done()
    }
    fn num_agents(&self) -> usize {
        self.inner.num_agents()
    }
    fn action_count(&self) -> usize {
        self.inner.action_count()
    }
    fn active_agents(&self) -> Vec<usize> {
        self.inner.active_agents()
    }
    fn legal_actions(&self, agent: usize) -> Vec<usize> {
        self.inner.legal_actions(agent)
    }
    fn observe<'py>(&self, py: Python<'py>, agent: usize) -> Bound<'py, PyArray3<f32>> {
        let (c, h, w) = self.obs_shape;
        Array3::from_shape_vec((c, h, w), self.inner.observe(agent))
            .expect("obs shape")
            .into_pyarray(py)
    }
    fn observation_space<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        space_to_py(py, self.inner.observation_space())
    }
    fn state<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        Ok(self.inner.state().to_py(py)?.into_any())
    }
    fn step<'py>(
        &mut self,
        py: Python<'py>,
        actions: Vec<usize>,
    ) -> PyResult<Vec<Bound<'py, PyAny>>> {
        self.inner
            .step(&actions)
            .iter()
            .map(|e| e.to_py(py))
            .collect()
    }
}

/// Build a type-erased `Env` from a `GameSpec`, pairing the game with its default encoder. One arm per
/// game (mirrors `build_engine`'s game axis).
fn build_env(game: GameSpec, seed: u64) -> Box<dyn ErasedEnv> {
    match game {
        GameSpec::Snake {
            grid_size,
            initial_length,
            initial_food_count,
            play_to_last,
            win_food_lead,
        } => {
            let enc = EgocentricSnake { grid_size };
            let obs_shape = enc.obs_shape();
            Box::new(EnvImpl {
                inner: Env::new(
                    Snake {
                        grid_size,
                        initial_length,
                        play_to_last,
                        win_food_lead,
                        initial_food_count,
                    },
                    Box::new(enc),
                    seed,
                ),
                obs_shape,
            })
        }
        GameSpec::Connect4 => {
            let obs_shape = Connect4Planes.obs_shape();
            Box::new(EnvImpl {
                inner: Env::new(Connect4, Box::new(Connect4Planes), seed),
                obs_shape,
            })
        }
        GameSpec::GridWorld { size, goal } => {
            let enc = GridWorldPlanes { size, goal };
            let obs_shape = enc.obs_shape();
            Box::new(EnvImpl {
                inner: Env::new(GridWorld { size, goal }, Box::new(enc), seed),
                obs_shape,
            })
        }
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
/// Named `PyReward` (exposed to Python as `Reward`) so the Rust name doesn't clash with the core
/// `reinfors_core::Reward` trait — same `Py*` convention as `PyEngine` / `PyEnv` / `PyBox`.
#[pyclass(name = "Reward")]
#[derive(Clone, Default)]
struct PyReward {
    weights: HashMap<String, f64>,
}

#[pymethods]
impl PyReward {
    #[new]
    #[pyo3(signature = (**weights))]
    fn new(weights: Option<HashMap<String, f64>>) -> Self {
        PyReward {
            weights: weights.unwrap_or_default(),
        }
    }
}

/// Resolve a generic `Reward` for a game whose components + defaults are `schema`: every key the
/// caller passed must be one of the schema's components (any unknown key is an error listing the valid
/// set), and each component reads its weight, falling back to its schema default. Returns the resolved
/// weights in schema order.
fn resolve_reward(reward: Option<PyReward>, schema: &[(&str, f64)]) -> PyResult<Vec<f64>> {
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

/// `rf.spaces.Box` — a continuous N-d `f32` tensor space. `shape` is the contract; `low`/`high` are
/// numpy arrays broadcast to `shape` (the Rust side carries one scalar bound — see `Space` docs — but
/// the public type is per-element-shaped, mirroring Gymnasium, so tighter per-element bounds can land
/// later without a Python break).
#[pyclass(name = "Box", module = "reinfors.spaces")]
struct PyBox {
    shape: Vec<usize>,
    #[pyo3(get)]
    low: Py<PyArrayDyn<f32>>,
    #[pyo3(get)]
    high: Py<PyArrayDyn<f32>>,
}

#[pymethods]
impl PyBox {
    #[getter]
    fn shape<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, &self.shape)
    }

    fn __repr__(&self) -> String {
        format!("Box(shape={:?})", self.shape)
    }
}

/// `rf.spaces.Discrete` — a choice from `0..n`.
#[pyclass(name = "Discrete", module = "reinfors.spaces")]
struct PyDiscrete {
    #[pyo3(get)]
    n: usize,
}

#[pymethods]
impl PyDiscrete {
    fn __repr__(&self) -> String {
        format!("Discrete(n={})", self.n)
    }
}

/// Convert a core `Space` into its `rf.spaces.*` Python object.
fn space_to_py(py: Python<'_>, space: Space) -> PyResult<Bound<'_, PyAny>> {
    match space {
        Space::Box { shape, low, high } => {
            let lo = ArrayD::from_elem(IxDyn(&shape), low).into_pyarray(py);
            let hi = ArrayD::from_elem(IxDyn(&shape), high).into_pyarray(py);
            Ok(Bound::new(
                py,
                PyBox {
                    shape,
                    low: lo.unbind(),
                    high: hi.unbind(),
                },
            )?
            .into_any())
        }
        Space::Discrete { n } => Ok(Bound::new(py, PyDiscrete { n })?.into_any()),
    }
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
    #[pyo3(signature = (grid_size=20, initial_length=3, food=3, play_to_last=true, win_food_lead=None))]
    #[pyo3(name = "Snake")]
    fn snake(
        grid_size: i32,
        initial_length: usize,
        food: usize,
        play_to_last: bool,
        win_food_lead: Option<usize>,
    ) -> Self {
        GameHandle {
            spec: GameSpec::Snake {
                grid_size,
                initial_length,
                initial_food_count: food,
                play_to_last,
                win_food_lead,
            },
        }
    }

    #[staticmethod]
    #[pyo3(name = "Connect4")]
    fn connect4() -> Self {
        GameHandle {
            spec: GameSpec::Connect4,
        }
    }

    #[staticmethod]
    #[pyo3(signature = (size=5, goal_row=4, goal_col=4))]
    #[pyo3(name = "GridWorld")]
    fn gridworld(size: i32, goal_row: i32, goal_col: i32) -> Self {
        GameHandle {
            spec: GameSpec::GridWorld {
                size,
                goal: (goal_row, goal_col),
            },
        }
    }

    /// The game's observation `Space` (an `rf.spaces.Box`) — its `shape` sizes the value network's
    /// input, replacing a hard-coded obs shape.
    fn observation_space<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        space_to_py(py, self.spec.spaces().0)
    }

    /// The game's action `Space` (an `rf.spaces.Discrete`) — its `n` sizes the network's output head.
    fn action_space<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        space_to_py(py, self.spec.spaces().1)
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

#[pymodule]
fn _reinfors(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(core_version, m)?)?;
    m.add_class::<PyEngine>()?;
    m.add_class::<PyEnv>()?;
    m.add_class::<GameHandle>()?;
    m.add_class::<PolicyHandle>()?;
    m.add_class::<LearnerHandle>()?;
    m.add_class::<PyReward>()?;
    m.add_class::<TreeStrapBatch>()?;
    m.add_class::<DqnBatch>()?;
    m.add_class::<PyBox>()?;
    m.add_class::<PyDiscrete>()?;
    Ok(())
}
