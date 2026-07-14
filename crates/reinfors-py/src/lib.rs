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
#[cfg(feature = "nn")]
use numpy::{PyReadonlyArray2, PyReadonlyArrayDyn};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use reinfors_core::{
    ActBy, AlwaysInitialState, CollectStats, Dqn, DqnRecord, Engine, EngineParams, Env,
    EpsilonGreedyQ, Game, Learner, Mcts, MctsConfig, Opponent, Policy, ReachedStateBuffer, Reward,
    SearchConfig, SelectiveExpectimax, Space, StartDistribution, StateEncoder, TreeStrap,
    TreeStrapRecord,
};
use reinfors_games::snake::{Cell, DeathCause};
use reinfors_games::{
    snake_length_cell, Action, Connect4, Connect4Event, Connect4Planes, Connect4Reward,
    Connect4State, EgocentricSnake, GridEvent, GridState, GridWorld, GridWorldPlanes,
    GridWorldReward, Snake, SnakeReward, SnakeState, StepEvent,
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
    #[pyo3(signature = (game, reward, policy, learner, n_games, seed=0, start_buffer=false, start_buffer_capacity=1000, p_fresh=0.05))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        game: GameHandle,
        reward: Option<PyReward>,
        policy: PolicyHandle,
        learner: LearnerHandle,
        n_games: usize,
        seed: u64,
        start_buffer: bool,
        start_buffer_capacity: usize,
        p_fresh: f64,
    ) -> PyResult<Self> {
        if n_games < 1 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "n_games must be >= 1",
            ));
        }
        // Off by default. When on, seed a fraction of episodes from reached mid/late-game states to
        // flatten start-state coverage (snake only in v1). Validated here so the core stays permissive.
        let start_buffer = if start_buffer {
            if start_buffer_capacity < 1 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "start_buffer_capacity must be >= 1",
                ));
            }
            check_unit("p_fresh", p_fresh)?;
            Some(StartBufferConfig {
                capacity: start_buffer_capacity,
                p_fresh,
            })
        } else {
            None
        };
        let engine_params = EngineParams { n_games, seed };
        Ok(PyEngine {
            inner: build_engine(
                game.spec,
                reward,
                policy.spec,
                learner.spec,
                engine_params,
                start_buffer,
            )?,
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
        // A Rust-native net is just another `infer` source: dispatch it through the callback-free path.
        #[cfg(feature = "nn")]
        if let Ok(net) = infer.downcast::<PyNet>() {
            return self
                .inner
                .collect_native(py, n_records, net.borrow().0.as_ref());
        }
        self.inner.collect(py, n_records, &infer)
    }

    /// Train entirely in Rust: `steps` rounds of (collect `collect_size` records via the trainer's net's
    /// Rust forward, then one gradient step), with net + records never crossing into Python. Returns the
    /// per-step loss. Trains the net the `trainer` was built from (which it owns). TreeStrap only.
    #[cfg(feature = "nn")]
    fn train(
        &mut self,
        py: Python<'_>,
        mut trainer: PyRefMut<'_, PyTrainer>,
        steps: usize,
        collect_size: usize,
    ) -> PyResult<Vec<f64>> {
        let net = trainer.net.clone_ref(py); // the trainer's own net — drives both collect and the step
        let net = net.borrow(py);
        self.inner
            .train_native(py, net.0.as_ref(), &mut trainer.inner, steps, collect_size)
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

    /// Roll forward using a Rust-native net for `infer` (no per-round Python callback).
    #[cfg(feature = "nn")]
    fn collect_native<'py>(
        &mut self,
        py: Python<'py>,
        n_records: usize,
        net: &dyn reinfors_nn::ValueNet,
    ) -> PyResult<Bound<'py, PyAny>>;

    /// The fully-in-Rust training loop: `steps` rounds of (collect `collect_size` records with `net`,
    /// then one gradient step on them). Records never touch Python. Returns the per-step loss. Errors if
    /// the learner has no in-Rust training path (only TreeStrap does).
    #[cfg(feature = "nn")]
    fn train_native(
        &mut self,
        py: Python<'_>,
        net: &dyn reinfors_nn::ValueNet,
        trainer: &mut reinfors_nn::TreeStrapTrainer,
        steps: usize,
        collect_size: usize,
    ) -> PyResult<Vec<f64>>;
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

    /// Flatten a batch into the contiguous `f32` buffers a Rust-native trainer consumes, or `None` if
    /// the learner has no in-Rust training path (only TreeStrap does today). Keeps records in Rust for
    /// the fused `engine.train` — they never marshal to Python.
    #[cfg(feature = "nn")]
    fn to_train_buffers(_records: &[Self]) -> Option<TrainBuffers> {
        None
    }
}

/// Row-major training buffers for `reinfors_nn::TreeStrapTrainer::update`: `obs` `[n·dim]`,
/// `targets` `[n·K·A]`, `mask` `[n·K]`.
#[cfg(feature = "nn")]
struct TrainBuffers {
    obs: Vec<f32>,
    targets: Vec<f32>,
    mask: Vec<f32>,
    n: i64,
}

/// The net's head count must equal the learner's (`Mcts` → 1 head; `SelectiveExpectimax` → `n_heads`),
/// or the target tensor won't line up with the net's output. A clear error beats a candle reshape panic.
#[cfg(feature = "nn")]
fn check_head_match(net: &dyn reinfors_nn::ValueNet, target_len: i64, n: i64) -> PyResult<()> {
    let a = net.n_actions() as i64;
    let record_heads = if n > 0 {
        target_len / (n * a)
    } else {
        net.n_heads() as i64
    };
    if record_heads != net.n_heads() as i64 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "net has n_heads={} but the policy produced {}-head targets; build the net with n_heads={}",
            net.n_heads(),
            record_heads,
            record_heads
        )));
    }
    Ok(())
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

    #[cfg(feature = "nn")]
    fn to_train_buffers(records: &[Self]) -> Option<TrainBuffers> {
        if records.is_empty() {
            return None;
        }
        let (mut obs, mut targets, mut mask) = (Vec::new(), Vec::new(), Vec::new());
        for (o, t, m) in records {
            obs.extend_from_slice(o);
            targets.extend(t.iter().flatten().map(|&v| v as f32)); // f64 targets -> f32 for the net
            mask.extend_from_slice(m);
        }
        Some(TrainBuffers {
            obs,
            targets,
            mask,
            n: records.len() as i64,
        })
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
    Ok((records, build_telemetry(py, &stats)?))
}

/// Marshal the rollout's `CollectStats` into the uniform telemetry dict (shared by every collect path).
fn build_telemetry<'py>(py: Python<'py>, stats: &CollectStats) -> PyResult<Bound<'py, PyDict>> {
    let d = (stats.decisions.max(1)) as f64;
    // (per-agent reward, length, seeded-from-start-buffer) — the `seeded` tag lets a caller keep
    // off-d0 episodes out of the true-start learning curves.
    let episodes: Vec<(Vec<f64>, usize, bool)> = stats
        .episodes
        .iter()
        .map(|e| (e.reward.clone(), e.length, e.seeded))
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
    Ok(telemetry)
}

/// Native-net rollout: drive the engine with a Rust `ValueNet` as the `infer` source — the forward runs
/// in Rust with no per-round Python callback. Mirrors `run_collect` minus the callback error-latching.
#[cfg(feature = "nn")]
fn run_collect_native<'py, G, P, L>(
    inner: &mut Engine<G, P, L>,
    py: Python<'py>,
    n_records: usize,
    net: &dyn reinfors_nn::ValueNet,
) -> PyResult<(Vec<L::Record>, Bound<'py, PyDict>)>
where
    G: Game + Sync,
    G::State: Send,
    P: Policy,
    L: Learner<P::Evaluation>,
{
    let mut infer_fn = |obs: Vec<f32>, n: usize| net.forward(&obs, n);
    let (records, stats) = inner.collect(n_records, &mut infer_fn);
    Ok((records, build_telemetry(py, &stats)?))
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

    #[cfg(feature = "nn")]
    fn collect_native<'py>(
        &mut self,
        py: Python<'py>,
        n_records: usize,
        net: &dyn reinfors_nn::ValueNet,
    ) -> PyResult<Bound<'py, PyAny>> {
        let (records, telemetry) = run_collect_native(&mut self.inner, py, n_records, net)?;
        L::Record::into_py_batch(records, py, self.dim, self.n_heads, telemetry)
    }

    #[cfg(feature = "nn")]
    fn train_native(
        &mut self,
        py: Python<'_>,
        net: &dyn reinfors_nn::ValueNet,
        trainer: &mut reinfors_nn::TreeStrapTrainer,
        steps: usize,
        collect_size: usize,
    ) -> PyResult<Vec<f64>> {
        let mut losses = Vec::with_capacity(steps);
        for _ in 0..steps {
            // collect with the net's Rust forward (no callback), then one grad step — all in Rust.
            let (records, _stats) = self
                .inner
                .collect(collect_size, |obs, n| net.forward(&obs, n));
            // Check emptiness first, so `to_train_buffers`'s `None` unambiguously means "not TreeStrap".
            if records.is_empty() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "engine.train collected 0 records — increase collect_size",
                ));
            }
            match L::Record::to_train_buffers(&records) {
                Some(tb) => {
                    check_head_match(net, tb.targets.len() as i64, tb.n)?;
                    let loss = trainer.loss(net, &tb.obs, &tb.targets, &tb.mask, tb.n);
                    // Release the GIL for the (pure-Rust candle) backward — it touches no Python, so
                    // other Python threads can run while it computes.
                    let tr = &mut *trainer;
                    losses.push(py.allow_threads(move || tr.step(&loss)));
                }
                None => {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "engine.train requires the TreeStrap learner",
                    ))
                }
            }
        }
        Ok(losses)
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
        max_ticks: Option<usize>,
    },
    Connect4,
    GridWorld {
        size: i32,
        goal: (i32, i32),
        max_ticks: Option<usize>,
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
                max_ticks,
            } => of(
                Snake {
                    grid_size,
                    initial_length,
                    play_to_last,
                    win_food_lead,
                    initial_food_count,
                    max_ticks,
                },
                &EgocentricSnake { grid_size },
            ),
            GameSpec::Connect4 => of(Connect4, &Connect4Planes),
            GameSpec::GridWorld {
                size,
                goal,
                max_ticks,
            } => of(
                GridWorld {
                    size,
                    goal,
                    max_ticks,
                },
                &GridWorldPlanes { size, goal },
            ),
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
    Mcts {
        num_simulations: usize,
        uct_c: f64,
        max_depth: i32,
        act_by: ActBy,
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

/// Reject a zero truncation horizon (`max_ticks=0` would truncate before any decision). `None` (never
/// truncate) and any positive cap are fine.
fn check_max_ticks(max_ticks: Option<usize>) -> PyResult<()> {
    if max_ticks == Some(0) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "max_ticks must be >= 1 (or None to never truncate)",
        ));
    }
    Ok(())
}

/// Family axis: given a concrete game, build the engine for a valid (policy, learner) pair. Written
/// once, generic over `G`, so a new family applies to every game; invalid pairings error here. Also
/// where the composed params are validated (the handles store them unchecked).
#[allow(clippy::too_many_arguments)]
fn build_for_game<G: Game + Send + Sync + 'static>(
    game: G,
    enc: Box<dyn StateEncoder<State = G::State>>,
    reward: Box<dyn Reward<Event = G::Event>>,
    start_dist: Box<dyn StartDistribution<G::State>>,
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
                inner: Engine::new(game, enc, reward, policy, learner, engine_params)
                    .with_start_distribution(start_dist),
                dim,
                action_count,
                n_heads,
            }))
        }
        (
            PolicySpec::Mcts {
                num_simulations,
                uct_c,
                max_depth,
                act_by,
            },
            LearnerSpec::TreeStrap {
                gamma,
                outcome_weight,
                bootstrap_p,
                interior_targets: _, // MCTS emits no interior targets (only the root value)
            },
        ) => {
            if num_simulations < 1 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "num_simulations must be >= 1",
                ));
            }
            if max_depth < 1 {
                return Err(pyo3::exceptions::PyValueError::new_err("max_depth must be >= 1"));
            }
            if uct_c < 0.0 {
                return Err(pyo3::exceptions::PyValueError::new_err("uct_c must be >= 0"));
            }
            check_unit("outcome_weight", outcome_weight)?;
            check_unit("bootstrap_p", bootstrap_p)?;
            // MCTS is a single-head value search, so the net's head count is 1 for the batch shape.
            let policy = Mcts::new(
                MctsConfig {
                    num_simulations,
                    uct_c,
                    gamma,
                    max_depth,
                },
                act_by,
            );
            let learner = TreeStrap::new(gamma, outcome_weight, bootstrap_p, false);
            Ok(Box::new(EngineImpl {
                inner: Engine::new(game, enc, reward, policy, learner, engine_params)
                    .with_start_distribution(start_dist),
                dim,
                action_count,
                n_heads: 1,
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
                inner: Engine::new(game, enc, reward, policy, learner, engine_params)
                    .with_start_distribution(start_dist),
                dim,
                action_count,
                n_heads,
            }))
        }
        _ => Err(pyo3::exceptions::PyValueError::new_err(
            "incompatible policy/learner: TreeStrap pairs with SelectiveExpectimax or Mcts, Dqn with EpsilonGreedyQ",
        )),
    }
}

/// Reached-state start-buffer config (`rf.Engine(..., start_buffer=True, ...)`), validated at the
/// engine boundary.
struct StartBufferConfig {
    capacity: usize,
    p_fresh: f64,
}

/// Game axis: pick the concrete game from `GameSpec`, then dispatch to `build_for_game`. One arm per
/// game; each instantly works with every family. The start distribution is wired here too, since only
/// the snake arm has a cell key for the reached-state buffer (other games use `AlwaysInitialState`).
fn build_engine(
    game: GameSpec,
    reward: Option<PyReward>,
    policy: PolicySpec,
    learner: LearnerSpec,
    engine_params: EngineParams,
    start_buffer: Option<StartBufferConfig>,
) -> PyResult<Box<dyn ErasedEngine>> {
    let reward = build_reward(&game, reward)?;
    // The reached-state buffer needs a game-specific cell key; only snake supplies one in v1.
    if start_buffer.is_some() && !matches!(game, GameSpec::Snake { .. }) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "start_buffer is only supported for the snake game",
        ));
    }
    // MCTS assumes strictly sequential / single-agent play; snake is simultaneous (with chance), so
    // reject the pairing here rather than let it panic mid-rollout in the search.
    if matches!(policy, PolicySpec::Mcts { .. }) && matches!(game, GameSpec::Snake { .. }) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Mcts supports only sequential / single-agent games (connect4, gridworld); snake is \
             simultaneous — use SelectiveExpectimax for snake",
        ));
    }
    match (game, reward) {
        (
            GameSpec::Snake {
                grid_size,
                initial_length,
                initial_food_count,
                play_to_last,
                win_food_lead,
                max_ticks,
            },
            RewardBox::Snake(reward),
        ) => {
            let start_dist: Box<dyn StartDistribution<SnakeState>> = match start_buffer {
                Some(cfg) => Box::new(ReachedStateBuffer::new(
                    cfg.capacity,
                    cfg.p_fresh,
                    snake_length_cell,
                )),
                None => Box::new(AlwaysInitialState),
            };
            build_for_game(
                Snake {
                    grid_size,
                    initial_length,
                    play_to_last,
                    win_food_lead,
                    initial_food_count,
                    max_ticks,
                },
                Box::new(EgocentricSnake { grid_size }),
                Box::new(reward),
                start_dist,
                policy,
                learner,
                engine_params,
            )
        }
        (GameSpec::Connect4, RewardBox::Connect4(reward)) => build_for_game(
            Connect4,
            Box::new(Connect4Planes),
            Box::new(reward),
            Box::new(AlwaysInitialState),
            policy,
            learner,
            engine_params,
        ),
        (
            GameSpec::GridWorld {
                size,
                goal,
                max_ticks,
            },
            RewardBox::GridWorld(reward),
        ) => build_for_game(
            GridWorld {
                size,
                goal,
                max_ticks,
            },
            Box::new(GridWorldPlanes { size, goal }),
            Box::new(reward),
            Box::new(AlwaysInitialState),
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
    #[pyo3(signature = (game, reward=None, seed=0))]
    fn new(game: GameHandle, reward: Option<PyReward>, seed: u64) -> PyResult<Self> {
        Ok(PyEnv {
            inner: build_env(game.spec, reward, seed)?,
        })
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

    /// The per-agent scalar rewards for the most recent `step`, or `None` if this `Env` was built
    /// without a `reward` (the reward-free play/eval default) or before the first `step`. Lets a
    /// training-facing consumer (e.g. the Gymnasium/PettingZoo adapters) read scalars without
    /// duplicating the game's event→reward mapping, which stays in Rust.
    #[getter]
    fn rewards(&self) -> Option<Vec<f64>> {
        self.inner.last_rewards()
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
        // `survived_to_max_ticks` is intentionally omitted: it is a rollout-only flag (set by the
        // Engine's `mark_truncation`), never by `Env`, which has no truncation horizon — so it is always
        // false here.
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
    /// The most recent `step`'s per-agent scalar rewards, or `None` if built reward-free / pre-`step`.
    fn last_rewards(&self) -> Option<Vec<f64>>;
}

struct EnvImpl<G: Game> {
    inner: Env<G>,
    obs_shape: (usize, usize, usize),
    // Optional so play/eval stay reward-free (`Env` holds no reward); the training-facing adapters
    // supply one and read back scalars via `last_rewards`. The event→reward mapping stays in Rust.
    reward: Option<Box<dyn Reward<Event = G::Event>>>,
    last_rewards: Option<Vec<f64>>,
}

impl<G> ErasedEnv for EnvImpl<G>
where
    G: Game + Send + Sync + 'static,
    G::State: Send + Sync + NativeState,
    G::Event: NativeEvent,
{
    fn reset(&mut self) {
        self.inner.reset();
        self.last_rewards = None;
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
        let events = self.inner.step(&actions);
        self.last_rewards = self.reward.as_ref().map(|r| {
            events
                .iter()
                .enumerate()
                .map(|(agent, e)| r.step_reward(e, agent))
                .collect()
        });
        events.iter().map(|e| e.to_py(py)).collect()
    }
    fn last_rewards(&self) -> Option<Vec<f64>> {
        self.last_rewards.clone()
    }
}

/// Build a type-erased `Env` from a `GameSpec`, pairing the game with its default encoder. One arm per
/// game (mirrors `build_engine`'s game axis). An optional `reward` makes the `Env` report per-step
/// scalar rewards (for the training-facing adapters); `None` keeps it reward-free (play/eval).
fn build_env(game: GameSpec, reward: Option<PyReward>, seed: u64) -> PyResult<Box<dyn ErasedEnv>> {
    let reward = reward.map(|r| build_reward(&game, Some(r))).transpose()?;
    Ok(match game {
        GameSpec::Snake {
            grid_size,
            initial_length,
            initial_food_count,
            play_to_last,
            win_food_lead,
            max_ticks,
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
                        max_ticks,
                    },
                    Box::new(enc),
                    seed,
                ),
                obs_shape,
                reward: reward.map(|rb| match rb {
                    RewardBox::Snake(r) => Box::new(r) as Box<dyn Reward<Event = StepEvent>>,
                    _ => unreachable!("build_reward returns the reward variant matching the game"),
                }),
                last_rewards: None,
            })
        }
        GameSpec::Connect4 => {
            let obs_shape = Connect4Planes.obs_shape();
            Box::new(EnvImpl {
                inner: Env::new(Connect4, Box::new(Connect4Planes), seed),
                obs_shape,
                reward: reward.map(|rb| match rb {
                    RewardBox::Connect4(r) => Box::new(r) as Box<dyn Reward<Event = Connect4Event>>,
                    _ => unreachable!("build_reward returns the reward variant matching the game"),
                }),
                last_rewards: None,
            })
        }
        GameSpec::GridWorld {
            size,
            goal,
            max_ticks,
        } => {
            let enc = GridWorldPlanes { size, goal };
            let obs_shape = enc.obs_shape();
            Box::new(EnvImpl {
                inner: Env::new(
                    GridWorld {
                        size,
                        goal,
                        max_ticks,
                    },
                    Box::new(enc),
                    seed,
                ),
                obs_shape,
                reward: reward.map(|rb| match rb {
                    RewardBox::GridWorld(r) => Box::new(r) as Box<dyn Reward<Event = GridEvent>>,
                    _ => unreachable!("build_reward returns the reward variant matching the game"),
                }),
                last_rewards: None,
            })
        }
    })
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
    // Snake can loop forever (circling without eating/dying), so `max_ticks` defaults to a finite cap:
    // it keeps `Engine.collect` from spinning on a non-terminating episode. Pass `max_ticks=None` to
    // explicitly opt into never truncating.
    #[pyo3(signature = (grid_size=20, initial_length=3, food=3, play_to_last=true, win_food_lead=None, max_ticks=1000))]
    #[pyo3(name = "Snake")]
    fn snake(
        grid_size: i32,
        initial_length: usize,
        food: usize,
        play_to_last: bool,
        win_food_lead: Option<usize>,
        max_ticks: Option<usize>,
    ) -> PyResult<Self> {
        check_max_ticks(max_ticks)?;
        Ok(GameHandle {
            spec: GameSpec::Snake {
                grid_size,
                initial_length,
                initial_food_count: food,
                play_to_last,
                win_food_lead,
                max_ticks,
            },
        })
    }

    #[staticmethod]
    #[pyo3(name = "Connect4")]
    fn connect4() -> Self {
        GameHandle {
            spec: GameSpec::Connect4,
        }
    }

    #[staticmethod]
    // GridWorld can wander forever without reaching the goal, so `max_ticks` defaults to a finite cap
    // (pass `max_ticks=None` to opt into never truncating).
    #[pyo3(signature = (size=5, goal_row=4, goal_col=4, max_ticks=1000))]
    #[pyo3(name = "GridWorld")]
    fn gridworld(
        size: i32,
        goal_row: i32,
        goal_col: i32,
        max_ticks: Option<usize>,
    ) -> PyResult<Self> {
        check_max_ticks(max_ticks)?;
        Ok(GameHandle {
            spec: GameSpec::GridWorld {
                size,
                goal: (goal_row, goal_col),
                max_ticks,
            },
        })
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

    /// The episode-length cap after which the rollout truncates a still-running game, or `None` for a
    /// game that always ends on its own (Connect-4). Loop-prone games (snake, gridworld) default to a
    /// finite cap so `Engine.collect` can't spin on a non-terminating episode.
    fn truncation_horizon(&self) -> Option<usize> {
        match self.spec {
            GameSpec::Snake { max_ticks, .. } | GameSpec::GridWorld { max_ticks, .. } => max_ticks,
            GameSpec::Connect4 => None,
        }
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

    /// Monte-Carlo Tree Search (UCT). Pairs with `TreeStrap`. Sequential / single-agent games only
    /// (connect4, gridworld) — rejected for snake. `act_by` is `"value"` (argmax mean action value) or
    /// `"visits"` (argmax visit count). Acting is deterministic (no root noise / move sampling), so it
    /// targets evaluation and benchmarking; for training-data diversity it relies on the start buffer.
    #[staticmethod]
    #[pyo3(signature = (num_simulations=64, uct_c=2.0, max_depth=64, act_by="value"))]
    #[pyo3(name = "Mcts")]
    fn mcts(num_simulations: usize, uct_c: f64, max_depth: i32, act_by: &str) -> PyResult<Self> {
        let act_by = match act_by {
            "value" => ActBy::Value,
            "visits" => ActBy::Visits,
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "unknown act_by {other:?}; expected \"value\" or \"visits\""
                )))
            }
        };
        Ok(PolicyHandle {
            spec: PolicySpec::Mcts {
                num_simulations,
                uct_c,
                max_depth,
                act_by,
            },
        })
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

/// The Cargo profile the extension was compiled with: `"debug"` or `"release"`. A debug build runs the
/// Rust core roughly an order of magnitude slower, so any performance number taken against one is
/// meaningless — the benchmark harness checks this and warns loudly.
#[pyfunction]
fn core_build_profile() -> &'static str {
    if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    }
}

// ===========================================================================
// Rust-native nets (`rf.nn`, candle — pure Rust). A `Net` is just another `infer` source: pass it to
// `engine.collect` and the forward runs in Rust with no per-round Python callback. Weights round-trip as
// numpy in the net's fixed parameter order (the torch state_dict layout), so a torch-trained checkpoint
// syncs in via `set_weights` and `get_weights` exports out (e.g. for safetensors / a parity check).
// ===========================================================================

// `unsendable`: the net is only ever used from the GIL thread that drives `collect`/`train`, so it need
// not be thread-shareable — the simplest sound `#[pyclass]` for a `Box<dyn ValueNet>`.
#[cfg(feature = "nn")]
#[pyclass(name = "Net", unsendable)]
struct PyNet(Box<dyn reinfors_nn::ValueNet>);

#[cfg(feature = "nn")]
#[pymethods]
impl PyNet {
    /// Conv trunk + K heads, sized from a planar observation shape `(C, H, W)`.
    #[staticmethod]
    fn conv(obs_shape: (i64, i64, i64), n_actions: i64, n_heads: i64) -> Self {
        PyNet(Box::new(reinfors_nn::Conv::new(
            obs_shape, n_actions, n_heads,
        )))
    }

    /// Two-layer MLP + K heads, for a flattened observation vector of `in_dim`.
    #[staticmethod]
    fn mlp(in_dim: i64, hidden: i64, n_actions: i64, n_heads: i64) -> Self {
        PyNet(Box::new(reinfors_nn::Mlp::new(
            in_dim, hidden, n_actions, n_heads,
        )))
    }

    #[getter]
    fn n_heads(&self) -> usize {
        self.0.n_heads()
    }

    #[getter]
    fn n_actions(&self) -> usize {
        self.0.n_actions()
    }

    /// Forward an `(N, dim)` float32 batch to `(N, K, A)` float64 — exactly the values the search sees.
    fn forward<'py>(
        &self,
        py: Python<'py>,
        obs: PyReadonlyArray2<'py, f32>,
    ) -> Bound<'py, PyArray3<f64>> {
        let arr = obs.as_array();
        let n = arr.shape()[0];
        let flat: Vec<f32> = arr.iter().copied().collect();
        let values = self.0.forward(&flat, n);
        Array3::from_shape_vec((n, self.0.n_heads(), self.0.n_actions()), values)
            .expect("value buffer shape")
            .into_pyarray(py)
    }

    /// Parameters as a list of numpy arrays in the net's fixed order (the torch state_dict layout).
    fn get_weights<'py>(&self, py: Python<'py>) -> Vec<Bound<'py, PyArrayDyn<f32>>> {
        reinfors_nn::export_weights(self.0.as_ref())
            .into_iter()
            .map(|(shape, data)| {
                let dims: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
                ArrayD::from_shape_vec(IxDyn(&dims), data)
                    .expect("param shape")
                    .into_pyarray(py)
            })
            .collect()
    }

    /// Copy weights (same order/shapes as `get_weights`) into the live parameters — torch→Rust sync.
    fn set_weights(&mut self, weights: Vec<PyReadonlyArrayDyn<'_, f32>>) {
        let data: Vec<Vec<f32>> = weights
            .iter()
            .map(|w| w.as_array().iter().copied().collect())
            .collect();
        reinfors_nn::import_weights(self.0.as_ref(), &data);
    }
}

/// Adam + masked-Huber trainer over a `Net`'s parameters — the in-Rust learning half. Use it fused
/// (`engine.train(trainer, ...)`, the whole loop in Rust) or step-by-step from a Python loop
/// (`trainer.update(obs, targets, masks)` on a batch you collected). The trainer holds the `Net` it was
/// built from — the optimizer steps *those* parameters, so binding them here makes it impossible to
/// train the wrong net (a same-shape different instance would otherwise silently no-op).
#[cfg(feature = "nn")]
#[pyclass(name = "TreeStrapTrainer", unsendable)]
struct PyTrainer {
    inner: reinfors_nn::TreeStrapTrainer,
    net: Py<PyNet>,
}

#[cfg(feature = "nn")]
#[pymethods]
impl PyTrainer {
    #[new]
    #[pyo3(signature = (net, lr = 2.5e-4))]
    fn new(py: Python<'_>, net: Py<PyNet>, lr: f64) -> PyResult<Self> {
        let inner = reinfors_nn::TreeStrapTrainer::new(net.borrow(py).0.as_ref(), lr)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner, net })
    }

    /// One gradient step on a batch you collected: `obs` (N, dim) f32, `targets` (N, K, A) f64,
    /// `masks` (N, K) f32 — the shapes `engine.collect` returns. Trains the trainer's own `Net`.
    fn update(
        &mut self,
        py: Python<'_>,
        obs: PyReadonlyArray2<'_, f32>,
        targets: PyReadonlyArray3<'_, f64>,
        masks: PyReadonlyArray2<'_, f32>,
    ) -> PyResult<f64> {
        let n = obs.as_array().shape()[0] as i64;
        let o: Vec<f32> = obs.as_array().iter().copied().collect();
        let t: Vec<f32> = targets.as_array().iter().map(|&v| v as f32).collect();
        let m: Vec<f32> = masks.as_array().iter().copied().collect();
        let net = self.net.borrow(py); // the net this trainer owns — can't diverge from the optimizer
        check_head_match(net.0.as_ref(), t.len() as i64, n)?;
        let loss = self.inner.loss(net.0.as_ref(), &o, &t, &m, n);
        // GIL released for the backward (see train_native) — the step touches no Python.
        let tr = &mut self.inner;
        Ok(py.allow_threads(move || tr.step(&loss)))
    }
}

#[pymodule]
fn _reinfors(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(core_version, m)?)?;
    m.add_function(wrap_pyfunction!(core_build_profile, m)?)?;
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
    #[cfg(feature = "nn")]
    m.add_class::<PyNet>()?;
    #[cfg(feature = "nn")]
    m.add_class::<PyTrainer>()?;
    Ok(())
}
