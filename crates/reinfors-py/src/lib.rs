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

use serde_json::{json, Value};

use reinfors_core::{
    ActBy, AlphaZero, AlphaZeroConfig, AlphaZeroLearner, AlphaZeroRecord, AlwaysInitialState,
    ChanceMode, Dqn, DqnRecord, Engine, EngineParams, Env, EpsilonGreedyQ, Game, InferCache,
    Learner, Mcts, MctsConfig, NoiseScope, Opponent, Policy, ReachedStateBuffer, Reward,
    SearchConfig, SelectiveExpectimax, Space, StartDistribution, StateCodec, StateEncoder,
    TreeStrap, TreeStrapRecord,
};
use reinfors_games::snake::{Cell, DeathCause};
use reinfors_games::{
    snake_length_cell, Action, Backgammon, BackgammonEvent, BackgammonReward, BackgammonState,
    BackgammonTesauro, Chess, ChessEvent, ChessPlanesAz119, ChessPlanesMinimal,
    ChessPlanesOpenSpiel, ChessPlanesRelative, ChessReward, ChessState, Connect4, Connect4Event,
    Connect4Planes, Connect4Reward, Connect4State, EgocentricSnake, GridEvent, GridState,
    GridWorld, GridWorldPlanes, GridWorldReward, Snake, SnakeReward, SnakeState, StepEvent,
    CHESS_ACTIONS,
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

/// The core's `infer` callback, wrapping the Python network forward. Obs arrive as one flat
/// row-major `[n, dim]` buffer (moved straight into a numpy array — no copy), and per-head values
/// `[n, K, action_count]` come back as one flat row-major buffer. The first failure — a Python error, or (when
/// `expected_heads` is set, i.e. the `Engine`) a returned head count that disagrees with the
/// configured `n_heads` — is latched into `callback_err` and zero rows (at the configured head count,
/// so trajectories never mix K shapes) are returned so the in-flight search unwinds cheaply; the
/// caller checks `callback_err` afterwards and propagates it. The check
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
        // Fallback rows keep the configured head count: a mid-collect error/drain must not mix
        // K-shaped and K=1 evaluations in one trajectory (the episode blend indexes per head).
        let fallback = n * expected_heads.unwrap_or(1) * action_count;
        if callback_err.is_some() {
            return vec![0.0; fallback];
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
                        return vec![0.0; fallback];
                    }
                }
                flat
            }
            Err(e) => {
                *callback_err = Some(e);
                vec![0.0; fallback]
            }
        }
    }
}

/// The AlphaZero family's `infer` callback: the Python callable returns a `(policy_logits (N, A) f64,
/// values (N,) f64)` tuple — no dummy priors for value-only families, no packed heads here — which is
/// flattened to the core's `[N·(A+1)]` row layout (`A` logits then the value, per row). Same
/// error-latching contract as `infer_closure`: the first failure (Python error or wrong shapes) is
/// latched and neutral rows (uniform logits, value 0) unwind the in-flight search cheaply.
fn az_infer_closure<'a, 'py>(
    py: Python<'py>,
    infer: &'a Bound<'py, PyAny>,
    dim: usize,
    action_count: usize,
    callback_err: &'a mut Option<PyErr>,
) -> impl FnMut(Vec<f32>, usize) -> Vec<f64> + 'a {
    move |obs_flat: Vec<f32>, n: usize| -> Vec<f64> {
        let stride = action_count + 1;
        if callback_err.is_some() {
            return vec![0.0; n * stride];
        }
        let arr = Array2::from_shape_vec((n, dim), obs_flat)
            .expect("obs batch shape")
            .into_pyarray(py);
        let extracted = infer.call1((arr,)).and_then(|r| {
            let (logits, values) =
                r.extract::<(numpy::PyReadonlyArray2<f64>, numpy::PyReadonlyArray1<f64>)>()?;
            Ok((logits.as_array().to_owned(), values.as_array().to_owned()))
        });
        match extracted {
            Ok((logits, values)) => {
                if logits.shape() != [n, action_count] || values.len() != n {
                    callback_err.get_or_insert_with(|| {
                        pyo3::exceptions::PyValueError::new_err(format!(
                            "AlphaZero infer must return (policy_logits ({n}, {action_count}), \
                             values ({n},)); got logits {:?} and values ({},)",
                            logits.shape(),
                            values.len()
                        ))
                    });
                    return vec![0.0; n * stride];
                }
                let mut out = Vec::with_capacity(n * stride);
                for row in 0..n {
                    out.extend(logits.row(row).iter().copied());
                    out.push(values[row]);
                }
                out
            }
            Err(e) => {
                *callback_err = Some(e);
                vec![0.0; n * stride]
            }
        }
    }
}

/// GIL-per-call variants of the two infer closures, for the stream worker thread: the worker holds no
/// GIL between search rounds (so the trainer's Python runs freely) and acquires it only for the
/// callback itself. A raised `stop` flag short-circuits to neutral rows *without* touching Python, so
/// a stopping stream drains its in-flight collect GIL-free.
fn infer_closure_gil<'a>(
    infer: &'a Py<PyAny>,
    dim: usize,
    action_count: usize,
    expected_heads: usize,
    layout: InferLayout,
    stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
    callback_err: std::sync::Arc<std::sync::Mutex<Option<PyErr>>>,
) -> impl FnMut(Vec<f32>, usize) -> Vec<f64> + 'a {
    move |obs_flat: Vec<f32>, n: usize| -> Vec<f64> {
        let fallback_len = match layout {
            InferLayout::ValueHeads => n * expected_heads * action_count,
            InferLayout::PolicyValue => n * (action_count + 1),
        };
        if stop.load(std::sync::atomic::Ordering::Relaxed) || callback_err.lock().unwrap().is_some()
        {
            return vec![0.0; fallback_len];
        }
        Python::with_gil(|py| {
            let mut err = callback_err.lock().unwrap().take();
            let out = {
                let bound = infer.bind(py);
                match layout {
                    InferLayout::ValueHeads => {
                        let mut f = infer_closure(
                            py,
                            bound,
                            dim,
                            action_count,
                            Some(expected_heads),
                            &mut err,
                        );
                        f(obs_flat, n)
                    }
                    InferLayout::PolicyValue => {
                        let mut f = az_infer_closure(py, bound, dim, action_count, &mut err);
                        f(obs_flat, n)
                    }
                }
            };
            *callback_err.lock().unwrap() = err;
            out
        })
    }
}

/// A finished collect, shipped from the worker to the consumer thread with its numpy/dict marshaling
/// deferred (the worker has no GIL guarantees; `CollectStream.next` runs the thunk under the GIL).
type BatchThunk = Box<dyn FnOnce(Python<'_>) -> PyResult<PyObject> + Send>;

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

// ===========================================================================
// UNIFIED ENGINE (permanent) — the `Engine` pyclass + its type-erased dispatch and two-axis factory.
// One class composes any game/policy/learner; only a factory arm (+ a `RecordBatch` impl for a new
// record shape) is needed to extend it — never a per-game/per-family engine type.
// ===========================================================================

// ===========================================================================
// RESOLVED CONFIG — a fully resolved, JSON-compatible view of an engine's immutable composition
// (defaults included), rendered from the same specs the factory consumes and the same reward
// schema `build_reward` resolves against. `engine_from_config(engine.resolved_config())`
// round-trips (the property test pins it). Canonical bytes = `serde_json` serialization (keys
// sorted by its BTreeMap-backed Map, floats via ryu shortest round-trip); the fingerprint is
// SHA-256 over those bytes — standard and permanent.
// ===========================================================================

const CONFIG_SCHEMA_VERSION: i64 = 1;

fn value_to_py<'py>(py: Python<'py>, v: &Value) -> PyResult<Bound<'py, PyAny>> {
    use pyo3::IntoPyObjectExt;
    Ok(match v {
        Value::Null => py.None().into_bound(py),
        Value::Bool(b) => b.into_bound_py_any(py)?,
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                i.into_bound_py_any(py)?
            } else if let Some(u) = n.as_u64() {
                u.into_bound_py_any(py)? // u64 seeds above i64::MAX stay exact integers
            } else {
                n.as_f64()
                    .expect("config numbers are i64, u64, or f64")
                    .into_bound_py_any(py)?
            }
        }
        Value::String(x) => x.into_bound_py_any(py)?,
        Value::Array(items) => {
            let list = pyo3::types::PyList::empty(py);
            for item in items {
                list.append(value_to_py(py, item)?)?;
            }
            list.into_bound_py_any(py)?
        }
        Value::Object(map) => {
            let d = PyDict::new(py);
            for (k, val) in map {
                d.set_item(k, value_to_py(py, val)?)?;
            }
            d.into_bound_py_any(py)?
        }
    })
}

fn chance_cfg(mode: &ChanceMode) -> Value {
    match mode {
        ChanceMode::AlwaysResample => json!({"name": "always_resample"}),
        ChanceMode::Committed { samples } => json!({"name": "committed", "samples": samples}),
        ChanceMode::ExpandAll => json!({"name": "expand_all"}),
    }
}

fn game_cfg(spec: &GameSpec) -> Value {
    match spec {
        GameSpec::Snake {
            grid_size,
            initial_length,
            initial_food_count,
            play_to_last,
            win_food_lead,
            max_ticks,
        } => json!({
            "name": "snake",
            "grid_size": grid_size,
            "initial_length": initial_length,
            "food": initial_food_count,
            "play_to_last": play_to_last,
            "win_food_lead": win_food_lead,
            "max_ticks": max_ticks,
        }),
        GameSpec::Connect4 => json!({"name": "connect4"}),
        GameSpec::Chess { max_ticks, encoder } => {
            let enc = match encoder {
                ChessEncoderSpec::Minimal => json!({"name": "minimal_chess"}),
                ChessEncoderSpec::Relative => json!({"name": "relative_chess"}),
                ChessEncoderSpec::OpenSpiel => json!({"name": "openspiel_chess"}),
                ChessEncoderSpec::AlphaZero { history } => {
                    json!({"name": "alphazero_chess", "history_length": history})
                }
            };
            json!({"name": "chess", "max_ticks": max_ticks, "encoder": enc})
        }
        GameSpec::Backgammon { max_ticks } => {
            json!({"name": "backgammon", "max_ticks": max_ticks})
        }
        GameSpec::GridWorld {
            size,
            goal,
            max_ticks,
        } => json!({
            "name": "gridworld",
            "size": size,
            "goal_row": goal.0,
            "goal_col": goal.1,
            "max_ticks": max_ticks,
        }),
    }
}

fn policy_cfg(spec: &PolicySpec) -> Value {
    let drop_cfg = |d: u32| {
        if d == u32::MAX {
            Value::Null
        } else {
            json!(d)
        }
    };
    match spec {
        PolicySpec::SelectiveExpectimax {
            beta,
            expansion_budget,
            top_k,
            max_depth,
            chance,
            opponent,
            n_heads,
            epsilon,
        } => {
            let mut v = json!({
                "name": "selective_expectimax",
                "beta": beta,
                "expansion_budget": expansion_budget,
                "top_k": top_k,
                "max_depth": max_depth,
                "chance": chance_cfg(chance),
                "n_heads": n_heads,
                "epsilon": epsilon,
            });
            let m = v.as_object_mut().expect("built as an object");
            match opponent {
                Opponent::Uniform => {
                    m.insert("opponent".into(), json!("uniform"));
                }
                Opponent::Distributional { temperature, floor } => {
                    m.insert("opponent".into(), json!("distributional"));
                    m.insert("opp_temperature".into(), json!(temperature));
                    m.insert("opp_floor".into(), json!(floor));
                }
            }
            v
        }
        PolicySpec::EpsilonGreedyQ { n_heads, epsilon } => {
            json!({"name": "epsilon_greedy_q", "n_heads": n_heads, "epsilon": epsilon})
        }
        PolicySpec::Mcts {
            num_simulations,
            uct_c,
            max_depth,
            act_by,
            temperature,
            temperature_drop,
            chance,
        } => json!({
            "name": "mcts",
            "num_simulations": num_simulations,
            "uct_c": uct_c,
            "max_depth": max_depth,
            "act_by": match act_by { ActBy::Value => "value", ActBy::Visits => "visits" },
            "temperature": temperature,
            "temperature_drop": drop_cfg(*temperature_drop),
            "chance": chance_cfg(chance),
        }),
        PolicySpec::AlphaZero {
            num_simulations,
            c_puct,
            max_depth,
            noise_epsilon,
            noise_alpha,
            temperature,
            temperature_drop,
            chance,
            noise_scope,
        } => {
            // `noise=None` is stored as epsilon 0 (the noise-free path) — rendered back as null.
            let noise = if *noise_epsilon == 0.0 {
                Value::Null
            } else {
                json!({
                    "name": "dirichlet",
                    "epsilon": noise_epsilon,
                    "alpha": noise_alpha,
                    "scope": match noise_scope {
                        NoiseScope::Requester => "requester",
                        NoiseScope::Both => "both",
                    },
                })
            };
            json!({
                "name": "alphazero",
                "num_simulations": num_simulations,
                "c_puct": c_puct,
                "max_depth": max_depth,
                "temperature": temperature,
                "temperature_drop": drop_cfg(*temperature_drop),
                "chance": chance_cfg(chance),
                "noise": noise,
            })
        }
    }
}

fn learner_cfg(spec: &LearnerSpec) -> Value {
    match spec {
        LearnerSpec::TreeStrap {
            gamma,
            outcome_weight,
            bootstrap_p,
            interior_targets,
        } => json!({
            "name": "treestrap",
            "gamma": gamma,
            "outcome_weight": outcome_weight,
            "bootstrap_p": bootstrap_p,
            "interior_targets": interior_targets,
        }),
        LearnerSpec::Dqn { bootstrap_p } => json!({"name": "dqn", "bootstrap_p": bootstrap_p}),
        LearnerSpec::AlphaZero { gamma } => json!({"name": "alphazero", "gamma": gamma}),
    }
}

/// Canonical bytes of a config value: `serde_json` with its BTreeMap-backed object ordering
/// (sorted keys) and ryu float formatting — deterministic by construction.
fn canonical_config_bytes(v: &Value) -> Vec<u8> {
    serde_json::to_vec(v).expect("config values contain no non-serializable data")
}

/// SHA-256 hex over the canonical bytes.
fn fingerprint_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

const ENGINE_SNAPSHOT_SCHEMA: u8 = 1;
const ENGINE_SNAPSHOT_MAGIC: &[u8; 4] = b"RFGS";

/// A quiescent, restorable capture of an `Engine`'s mutable collection state: live episodes and
/// their RNGs, tick counts, seeded flags, per-game policy state, partially accumulated
/// trajectories (with their evaluations), the start-buffer reservoirs, and the weights
/// generation. The infer cache is deliberately excluded — cache hits return bit-identical rows
/// to the forwards they replace, so collected RECORDS after restore are byte-identical with a
/// cold cache: the guarantee is record-exact, not inference-call-pattern-exact.
#[pyclass(name = "EngineSnapshot")]
#[derive(Clone)]
struct PyEngineSnapshot {
    schema: u8,
    fingerprint: String,
    weights_generation: u64,
    policy_version: Option<String>,
    payload: Vec<u8>,
}

#[pymethods]
impl PyEngineSnapshot {
    /// Fingerprint of the engine composition (reinfors version excluded, so snapshots survive
    /// upgrades with unchanged schemas).
    #[getter]
    fn fingerprint(&self) -> &str {
        &self.fingerprint
    }

    #[getter]
    fn schema_version(&self) -> u8 {
        self.schema
    }

    /// The engine's weights-generation counter at capture, plus the user-supplied external net
    /// version (if any) — reinfors cannot know the user's net, so strict checking is opt-in.
    #[getter]
    fn weights_generation(&self) -> u64 {
        self.weights_generation
    }

    #[getter]
    fn policy_version(&self) -> Option<&str> {
        self.policy_version.as_deref()
    }

    fn to_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, pyo3::types::PyBytes> {
        let mut out = Vec::with_capacity(self.payload.len() + 96);
        out.extend_from_slice(ENGINE_SNAPSHOT_MAGIC);
        out.push(self.schema);
        out.extend_from_slice(&(self.fingerprint.len() as u32).to_le_bytes());
        out.extend_from_slice(self.fingerprint.as_bytes());
        out.extend_from_slice(&self.weights_generation.to_le_bytes());
        let pv = self.policy_version.as_deref().unwrap_or("");
        out.push(u8::from(self.policy_version.is_some()));
        out.extend_from_slice(&(pv.len() as u32).to_le_bytes());
        out.extend_from_slice(pv.as_bytes());
        out.extend_from_slice(&(self.payload.len() as u32).to_le_bytes());
        out.extend_from_slice(&self.payload);
        pyo3::types::PyBytes::new(py, &out)
    }

    #[staticmethod]
    fn from_bytes(data: &[u8]) -> PyResult<Self> {
        let err = |m: &str| {
            pyo3::exceptions::PyValueError::new_err(format!("invalid EngineSnapshot: {m}"))
        };
        let mut r = reinfors_core::codec::bytes::Reader::new(data);
        let mut chk = |n: usize| r.take(n).map_err(|_| err("truncated"));
        if chk(4)? != ENGINE_SNAPSHOT_MAGIC {
            return Err(err("bad magic"));
        }
        let schema = chk(1)?[0];
        if schema != ENGINE_SNAPSHOT_SCHEMA {
            return Err(err(&format!("unsupported schema version {schema}")));
        }
        let fp_len = u32::from_le_bytes(chk(4)?.try_into().unwrap()) as usize;
        let fingerprint =
            String::from_utf8(chk(fp_len)?.to_vec()).map_err(|_| err("fingerprint not utf-8"))?;
        let weights_generation = u64::from_le_bytes(chk(8)?.try_into().unwrap());
        let has_pv = match chk(1)?[0] {
            0 => false,
            1 => true,
            b => {
                return Err(err(&format!(
                    "policy_version presence byte {b} is not a bool"
                )))
            }
        };
        let pv_len = u32::from_le_bytes(chk(4)?.try_into().unwrap()) as usize;
        let pv = String::from_utf8(chk(pv_len)?.to_vec())
            .map_err(|_| err("policy_version not utf-8"))?;
        let payload_len = u32::from_le_bytes(chk(4)?.try_into().unwrap()) as usize;
        let payload = chk(payload_len)?.to_vec();
        r.done().map_err(|_| err("trailing bytes"))?;
        Ok(PyEngineSnapshot {
            schema,
            fingerprint,
            weights_generation,
            policy_version: has_pv.then_some(pv),
            payload,
        })
    }
}

/// The unified parallel rollout engine: composes a game + policy + learner handle and drives N games,
/// yielding the learner's records. Holds the composition type-erased, so one class serves every
/// `(game, policy, learner)`. Construct the handles via `rf.games.*` / `rf.policies.*` / `rf.learners.*`.
#[pyclass(name = "Engine")]
struct PyEngine {
    // `None` while a `CollectStream` holds the engine on its worker thread; `stop()` returns it.
    inner: Option<Box<dyn ErasedEngine>>,
    // The immutable composition, resolved (defaults included) at construction — see `resolved_config`.
    config: Value,
    // `config` minus the reinfors version, fingerprinted — what snapshots embed and check.
    snapshot_fp: String,
    // Weights-version counter shared with the infer cache (if enabled). Held here — OUTSIDE the
    // movable engine — so `weights_updated()` works from the consumer thread while a stream's
    // worker owns the engine.
    weights_generation: std::sync::Arc<std::sync::atomic::AtomicU64>,
}

#[pymethods]
impl PyEngine {
    #[new]
    #[pyo3(signature = (game, reward, policy, learner, n_games, seed=0, start_buffer=false, start_buffer_capacity=1000, p_fresh=0.05, infer_cache=0))]
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
        infer_cache: usize,
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
        // Capture the resolved composition before the specs are consumed by the factory.
        let resolved_reward = {
            let weights = reward.as_ref().map(|r| PyReward {
                weights: r.weights.clone(),
            });
            let vals = resolve_reward(weights, reward_schema(&game.spec))?;
            Value::Object(
                reward_schema(&game.spec)
                    .iter()
                    .zip(vals)
                    .map(|((k, _), v)| ((*k).to_string(), json!(v)))
                    .collect(),
            )
        };
        let config = json!({
            "schema_version": CONFIG_SCHEMA_VERSION,
            "reinfors_version": reinfors_core::version(),
            "game": game_cfg(&game.spec),
            "reward": resolved_reward,
            "policy": policy_cfg(&policy.spec),
            "learner": learner_cfg(&learner.spec),
            "engine": {
                "n_games": n_games,
                "seed": seed,
                // Effective composition only: a disabled start buffer renders null, so ignored
                // capacity/p_fresh arguments cannot split fingerprints of identical engines
                // (the AlphaZero-noise convention).
                "start_buffer": start_buffer.as_ref().map_or(Value::Null, |sb| json!({
                    "capacity": sb.capacity,
                    "p_fresh": sb.p_fresh,
                })),
                "infer_cache": infer_cache,
            },
        });
        let engine_params = EngineParams { n_games, seed };
        let weights_generation = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
        let cache =
            (infer_cache > 0).then(|| InferCache::new(infer_cache, weights_generation.clone()));
        let snapshot_fp = {
            let mut stripped = config.clone();
            stripped
                .as_object_mut()
                .expect("config is an object")
                .remove("reinfors_version");
            fingerprint_hex(&canonical_config_bytes(&stripped))
        };
        Ok(PyEngine {
            snapshot_fp,
            inner: Some(build_engine(
                game.spec,
                reward,
                policy.spec,
                learner.spec,
                engine_params,
                start_buffer,
                cache,
            )?),
            config,
            weights_generation,
        })
    }

    /// Tell the engine the net's weights changed (call after every weight sync — e.g. right after
    /// `load_state_dict` onto the collector net). Bumps a shared version counter that clears the
    /// infer cache at the next search-round boundary; safe to call from any thread, including while
    /// a `collect_stream` worker holds the engine. A no-op when the cache is disabled. Never calling
    /// it asserts the weights never changed — correct, not stale. Future weight-dependent features
    /// hang off the same signal.
    fn weights_updated(&self) {
        self.weights_generation
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    /// The engine's immutable composition, fully resolved (defaults included) as a
    /// JSON-compatible dict — `rf.engine_from_config(engine.resolved_config())` reconstructs an
    /// equivalent engine. Excludes what reinfors does not own: network, optimizer, replay buffer,
    /// inference callback.
    fn resolved_config<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        value_to_py(py, &self.config)
    }

    /// A 128-bit hex fingerprint of `resolved_config`, hashed over reinfors-produced canonical
    /// bytes (sorted keys, Rust float formatting). Compare fingerprints; never recompute from
    /// re-serialized JSON. The algorithm may change with the config `schema_version`.
    fn config_fingerprint(&self) -> String {
        fingerprint_hex(&canonical_config_bytes(&self.config))
    }

    /// A quiescent snapshot of the engine's mutable collection state (see `EngineSnapshot`).
    /// `policy_version`: an optional caller-owned tag for the external net's weights, checked on
    /// restore only when the restorer asks. Unavailable while a `collect_stream` worker owns the
    /// engine — stop (or, later, pause) the stream first.
    #[pyo3(signature = (policy_version=None))]
    fn snapshot(&self, policy_version: Option<String>) -> PyResult<PyEngineSnapshot> {
        let engine = self.inner.as_ref().ok_or_else(stream_active_err)?;
        Ok(PyEngineSnapshot {
            schema: ENGINE_SNAPSHOT_SCHEMA,
            fingerprint: self.snapshot_fp.clone(),
            weights_generation: self
                .weights_generation
                .load(std::sync::atomic::Ordering::Relaxed),
            policy_version,
            payload: engine.snapshot_payload()?,
        })
    }

    /// Install a snapshot. Rejects a different composition (fingerprint), an unsupported schema,
    /// a malformed payload (engine left untouched), or — when `expect_policy_version` is given —
    /// a snapshot whose recorded net version differs. Restores the weights generation, so a
    /// restored engine's cache-clear behavior matches the captured one.
    #[pyo3(signature = (snapshot, expect_policy_version=None))]
    fn restore(
        &mut self,
        snapshot: &PyEngineSnapshot,
        expect_policy_version: Option<String>,
    ) -> PyResult<()> {
        if snapshot.schema != ENGINE_SNAPSHOT_SCHEMA {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "unsupported snapshot schema {}",
                snapshot.schema
            )));
        }
        if snapshot.fingerprint != self.snapshot_fp {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "snapshot is from a different composition (fingerprint {} != {})",
                snapshot.fingerprint, self.snapshot_fp
            )));
        }
        if let Some(expect) = expect_policy_version {
            if snapshot.policy_version.as_deref() != Some(expect.as_str()) {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "policy_version mismatch: snapshot has {:?}, expected {expect:?}",
                    snapshot.policy_version
                )));
            }
        }
        let engine = self.inner.as_mut().ok_or_else(stream_active_err)?;
        engine.restore_payload(&snapshot.payload)?;
        self.weights_generation.store(
            snapshot.weights_generation,
            std::sync::atomic::Ordering::Relaxed,
        );
        Ok(())
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
        self.inner
            .as_mut()
            .ok_or_else(stream_active_err)?
            .collect(py, n_records, &infer)
    }

    /// Start continuous background collection: a Rust worker thread runs collect after collect,
    /// pushing finished batches into a bounded queue of `depth` (None = unbounded — the OpenSpiel
    /// continuous-actor topology; the worker never pauses, so an outpaced consumer grows the queue).
    /// With `depth=1` the worker pipelines exactly one batch ahead of the consumer (backpressure).
    /// The engine is held by the stream until `stop()` returns it; `collect` in the meantime errors.
    /// Weight staleness is the caller's to manage — the callback reads whatever net it closes over,
    /// so sync a collector-net copy at your chosen cadence (see the AlphaZero example).
    #[pyo3(signature = (collect_size, infer, depth=Some(1)))]
    fn collect_stream(
        slf: &Bound<'_, Self>,
        collect_size: usize,
        infer: Py<PyAny>,
        depth: Option<usize>,
    ) -> PyResult<CollectStream> {
        if collect_size < 1 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "collect_size must be >= 1",
            ));
        }
        if depth == Some(0) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "depth must be >= 1, or None for unbounded",
            ));
        }
        let mut engine = slf
            .borrow_mut()
            .inner
            .take()
            .ok_or_else(stream_active_err)?;

        let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let pause = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let queued = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let (tx, rx) = StreamChannel::new(depth);
        let worker = {
            let (stop, queued) = (stop.clone(), queued.clone());
            let pause = pause.clone();
            std::thread::spawn(move || {
                loop {
                    // `pause` is honored only at batch boundaries: the in-flight collect finishes
                    // with real inference, so the engine's state matches the delivered batches.
                    if stop.load(std::sync::atomic::Ordering::Relaxed)
                        || pause.load(std::sync::atomic::Ordering::Relaxed)
                    {
                        break;
                    }
                    let result = engine.collect_thunk(collect_size, &infer, stop.clone());
                    let fatal = result.is_err();
                    if tx.send(result).is_err() {
                        break; // consumer dropped the receiver (stop) — engine returns via join
                    }
                    queued.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if fatal {
                        break; // callback error delivered; the stream is dead
                    }
                }
                engine
            })
        };
        Ok(CollectStream {
            rx: Some(std::sync::Mutex::new(rx)),
            stop,
            pause,
            queued,
            handle: Some(worker),
            engine: slf.clone().unbind(),
        })
    }
}

/// The chess observation view, carried by an `rf.encoders.*` handle (game-handle kwarg).
#[derive(Clone, Copy)]
enum ChessEncoderSpec {
    Minimal,
    Relative,
    OpenSpiel,
    AlphaZero { history: usize },
}

/// The chess game + encoder pair for an encoder choice — the game's `history_len` is set to exactly
/// what the selected encoder reads, so the two cannot drift apart.
fn chess_parts(
    max_ticks: Option<usize>,
    encoder: ChessEncoderSpec,
) -> (Chess, Box<dyn StateEncoder<State = ChessState>>) {
    match encoder {
        ChessEncoderSpec::Minimal => (
            Chess {
                max_ticks,
                history_len: 0,
            },
            Box::new(ChessPlanesMinimal),
        ),
        ChessEncoderSpec::Relative => (
            Chess {
                max_ticks,
                history_len: 0,
            },
            Box::new(ChessPlanesRelative),
        ),
        ChessEncoderSpec::OpenSpiel => (
            Chess {
                max_ticks,
                history_len: 0,
            },
            Box::new(ChessPlanesOpenSpiel),
        ),
        ChessEncoderSpec::AlphaZero { history } => (
            Chess {
                max_ticks,
                history_len: history,
            },
            Box::new(ChessPlanesAz119 { history }),
        ),
    }
}

fn stream_active_err() -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(
        "this engine is held by a collect_stream (call its stop() to get the engine back) — or was \
         permanently forfeited by a stream that was dropped without stop(); in that case create a \
         new Engine",
    )
}

/// One channel type for both depths: bounded (`sync_channel`, sender blocks when full — the
/// backpressure that makes `depth` mean 'batches the worker may run ahead') or unbounded.
enum StreamChannel {
    Bounded(std::sync::mpsc::SyncSender<PyResult<BatchThunk>>),
    Unbounded(std::sync::mpsc::Sender<PyResult<BatchThunk>>),
}

impl StreamChannel {
    fn new(
        depth: Option<usize>,
    ) -> (
        StreamChannel,
        std::sync::mpsc::Receiver<PyResult<BatchThunk>>,
    ) {
        match depth {
            Some(d) => {
                let (tx, rx) = std::sync::mpsc::sync_channel(d);
                (StreamChannel::Bounded(tx), rx)
            }
            None => {
                let (tx, rx) = std::sync::mpsc::channel();
                (StreamChannel::Unbounded(tx), rx)
            }
        }
    }

    fn send(&self, msg: PyResult<BatchThunk>) -> Result<(), ()> {
        match self {
            StreamChannel::Bounded(tx) => tx.send(msg).map_err(|_| ()),
            StreamChannel::Unbounded(tx) => tx.send(msg).map_err(|_| ()),
        }
    }
}

/// A running background collection (see `Engine.collect_stream`). `next()` blocks (GIL released)
/// until the worker's next batch is ready and marshals it on this thread; iteration is equivalent.
/// `stop()` ends the worker and returns the engine to its `Engine` — also on `with`-exit.
///
/// **Single-consumer contract:** one consumer thread loops `next()` (or iterates) and owns `stop()`.
/// Other Python threads run freely while `next()` waits (the GIL is released) — but they must not
/// touch the stream object itself: `next()` holds the pyclass borrow across its wait, so a
/// concurrent `stop()` raises pyo3's `RuntimeError: Already borrowed` rather than interrupting it
/// (and by design `stop()` could not interrupt a blocked `next()` anyway — the wait only ends when
/// a batch arrives or the worker exits). Concurrent `next()` calls from two threads are not
/// prevented, but batch order between them is arbitrary and `pending()` becomes advisory-only —
/// don't build a multi-thread consumer on this.
///
/// A dropped, never-stopped stream detaches its worker (which drains GIL-free via the stop flag)
/// and permanently forfeits the engine — always stop streams (the `with` form makes this hard to
/// get wrong).
#[pyclass]
struct CollectStream {
    rx: Option<std::sync::Mutex<std::sync::mpsc::Receiver<PyResult<BatchThunk>>>>,
    stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
    pause: std::sync::Arc<std::sync::atomic::AtomicBool>,
    queued: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    handle: Option<std::thread::JoinHandle<Box<dyn ErasedEngine>>>,
    engine: Py<PyEngine>,
}

#[pymethods]
impl CollectStream {
    /// The next finished batch (learner-shaped, same as `collect`'s). Blocks with the GIL released
    /// while the worker finishes one. Raises the latched callback error if the worker died on one,
    /// or RuntimeError once the stream is stopped/dead.
    fn next(&self, py: Python<'_>) -> PyResult<PyObject> {
        let rx = self.rx.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("collect stream is stopped")
        })?;
        let msg = py.allow_threads(|| rx.lock().unwrap().recv());
        match msg {
            Ok(result) => {
                self.queued
                    .fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
                result?(py)
            }
            Err(_) => Err(pyo3::exceptions::PyRuntimeError::new_err(
                "collect stream ended (stopped, or a callback error was already raised)",
            )),
        }
    }

    /// Finished batches waiting in the queue (advisory; the worker may also have one in flight).
    fn pending(&self) -> usize {
        self.queued.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Stop the worker and return the engine to its `Engine` (idempotent). Any queued batches are
    /// discarded. The engine is reusable afterwards — `collect` or a fresh `collect_stream`.
    /// Lossless quiescence, the checkpoint barrier: stop BEGINNING new collects, let the
    /// in-flight collect finish with real inference, drain and return every completed batch, and
    /// hand the engine back — its state then corresponds exactly to the returned batches, so
    /// `engine.snapshot()` right after is record-exact. (`stop()` is the fast lossy abort:
    /// it short-circuits the in-flight collect and discards the queue.)
    fn pause<'py>(&mut self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyAny>>> {
        self.pause.store(true, std::sync::atomic::Ordering::Relaxed);
        let rx = self
            .rx
            .take()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("stream already stopped"))?;
        // Drain until the worker drops its sender (it breaks at the batch boundary). recv blocks
        // while the worker finishes the in-flight batch, whose infer callback needs the GIL —
        // so recv MUST run with the GIL released (the Mutex wrapper keeps the receiver Sync).
        let mut thunks = Vec::new();
        loop {
            let item = py.allow_threads(|| {
                rx.lock()
                    .map_err(|_| ())
                    .and_then(|guard| guard.recv().map_err(|_| ()))
            });
            match item {
                Ok(item) => thunks.push(item),
                Err(()) => break, // disconnected: worker is at the boundary
            }
        }
        if let Some(handle) = self.handle.take() {
            let engine = py
                .allow_threads(|| handle.join())
                .map_err(|_| pyo3::exceptions::PyRuntimeError::new_err("stream worker panicked"))?;
            self.engine.bind(py).borrow_mut().inner = Some(engine);
        }
        thunks
            .into_iter()
            .map(|item| item.and_then(|thunk| Ok(thunk(py)?.into_bound(py))))
            .collect()
    }

    fn stop(&mut self, py: Python<'_>) -> PyResult<()> {
        self.stop.store(true, std::sync::atomic::Ordering::Relaxed);
        self.rx.take(); // unblocks a worker waiting on a full bounded queue
        if let Some(handle) = self.handle.take() {
            let engine = py
                .allow_threads(|| handle.join())
                .map_err(|_| pyo3::exceptions::PyRuntimeError::new_err("stream worker panicked"))?;
            self.engine.bind(py).borrow_mut().inner = Some(engine);
        }
        Ok(())
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    /// Iterator form of `next()`: yields batches until the stream is stopped (then StopIteration).
    /// A callback error still raises.
    fn __next__(&self, py: Python<'_>) -> PyResult<Option<PyObject>> {
        match self.next(py) {
            Ok(batch) => Ok(Some(batch)),
            Err(e) if e.is_instance_of::<pyo3::exceptions::PyRuntimeError>(py) => Ok(None),
            Err(e) => Err(e),
        }
    }

    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    #[pyo3(signature = (*_args))]
    fn __exit__(&mut self, py: Python<'_>, _args: &Bound<'_, PyTuple>) -> PyResult<()> {
        self.stop(py)
    }
}

impl Drop for CollectStream {
    fn drop(&mut self) {
        // Signal and detach — never join here: drop can run under the GIL while the worker's next
        // infer call waits for it. The stop flag makes the worker drain GIL-free and exit.
        self.stop.store(true, std::sync::atomic::Ordering::Relaxed);
        self.rx.take();
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
    /// Snapshot the mutable collection state (record-exact contract; see core `snapshot_bytes`).
    fn snapshot_payload(&self) -> PyResult<Vec<u8>>;
    fn restore_payload(&mut self, bytes: &[u8]) -> PyResult<()>;
    fn collect<'py>(
        &mut self,
        py: Python<'py>,
        n_records: usize,
        infer: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>>;

    /// GIL-free collect for the stream worker: runs the rollout acquiring the GIL only inside the
    /// infer callback, and returns the finished batch as a deferred-marshaling thunk (run under the
    /// GIL by the consumer). `stop` short-circuits the callback so a stopping stream drains fast.
    fn collect_thunk(
        &mut self,
        n_records: usize,
        infer: &Py<PyAny>,
        stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
    ) -> PyResult<BatchThunk>;
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
    // Legality in CSR form — record i's legal ids are `ids[offsets[i]..offsets[i+1]]`. Sparse on
    // purpose: dense (M, A) f32 masks on wide action spaces dwarf the observations (chess: ~7.7x
    // the obs bytes; ~37 GB per million transitions), while CSR is ~35 ids/row. Densify per
    // MINIBATCH at train time, never per record at storage time.
    #[pyo3(get)]
    legal_ids: Py<PyArray1<i64>>, // legality of `obs` (diagnostics; the action is already legal)
    #[pyo3(get)]
    legal_offsets: Py<PyArray1<i64>>, // (M + 1,)
    // Legality of `next_obs` + THE bootstrap rule: bootstrap iff record i's slice is NON-EMPTY —
    // an empty slice (terminal, or an alternating-game truncation tail) means "target = r".
    // `dones` is an episode-boundary flag, NOT a target-math input ((1 - done) * max meets a
    // masked max's -inf as NaN). The complete safe target, densified per minibatch:
    //   counts = np.diff(next_legal_offsets); rows = np.repeat(np.arange(M), counts)
    //   mask = np.zeros((M, A), bool); mask[rows, next_legal_ids] = True
    //   q  = np.where(mask, q_next, -np.inf).max(-1)
    //   td = rewards + gamma * np.where(np.isfinite(q), q, 0.0)
    #[pyo3(get)]
    next_legal_ids: Py<PyArray1<i64>>,
    #[pyo3(get)]
    next_legal_offsets: Py<PyArray1<i64>>, // (M + 1,)
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

/// `engine.collect` result for the AlphaZero family: root visit distributions + realized returns.
#[pyclass]
struct AlphaZeroBatch {
    #[pyo3(get)]
    obs: Py<PyArray2<f32>>, // (M, C*H*W)
    #[pyo3(get)]
    policy_targets: Py<PyArray2<f64>>, // (M, A) — π, τ=1 normalized root visit counts
    #[pyo3(get)]
    value_targets: Py<PyArray1<f64>>, // (M,) — z, discounted realized return
    #[pyo3(get)]
    telemetry: Py<PyDict>,
}

#[pymethods]
impl AlphaZeroBatch {
    fn __len__(&self) -> usize {
        4
    }
    /// Also unpacks positionally: `obs, policy_targets, value_targets, telemetry = batch`.
    fn __getitem__<'py>(&self, py: Python<'py>, i: usize) -> PyResult<Bound<'py, PyAny>> {
        Ok(match i {
            0 => self.obs.bind(py).clone().into_any(),
            1 => self.policy_targets.bind(py).clone().into_any(),
            2 => self.value_targets.bind(py).clone().into_any(),
            3 => self.telemetry.bind(py).clone().into_any(),
            _ => {
                return Err(pyo3::exceptions::PyIndexError::new_err(
                    "AlphaZeroBatch index out of range",
                ))
            }
        })
    }
}

impl RecordBatch for AlphaZeroRecord {
    fn into_py_batch<'py>(
        records: Vec<Self>,
        py: Python<'py>,
        dim: usize,
        _n_heads: usize,
        telemetry: Bound<'py, PyDict>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let m = records.len();
        let a = if m > 0 { records[0].1.len() } else { 0 };
        let mut obs_flat: Vec<f32> = Vec::with_capacity(m * dim);
        let mut pi_flat: Vec<f64> = Vec::with_capacity(m * a);
        let mut z: Vec<f64> = Vec::with_capacity(m);
        for (obs, pi, zi) in records {
            obs_flat.extend(obs);
            pi_flat.extend(pi);
            z.push(zi);
        }
        let obs_arr = Array2::from_shape_vec((m, dim), obs_flat)
            .expect("obs shape")
            .into_pyarray(py);
        let pi_arr = Array2::from_shape_vec((m, a), pi_flat)
            .expect("policy target shape")
            .into_pyarray(py);
        Ok(Bound::new(
            py,
            AlphaZeroBatch {
                obs: obs_arr.unbind(),
                policy_targets: pi_arr.unbind(),
                value_targets: z.into_pyarray(py).unbind(),
                telemetry: telemetry.unbind(),
            },
        )?
        .into_any())
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
        let mut legal_ids: Vec<i64> = Vec::new();
        let mut legal_offsets: Vec<i64> = Vec::with_capacity(m + 1);
        let mut next_legal_ids: Vec<i64> = Vec::new();
        let mut next_legal_offsets: Vec<i64> = Vec::with_capacity(m + 1);
        legal_offsets.push(0);
        next_legal_offsets.push(0);
        for t in records {
            obs_flat.extend(t.obs);
            next_flat.extend(t.next_obs);
            mask_flat.extend(t.mask);
            legal_ids.extend(t.legal.iter().map(|&a| a as i64));
            legal_offsets.push(legal_ids.len() as i64);
            next_legal_ids.extend(t.next_legal.iter().map(|&a| a as i64));
            next_legal_offsets.push(next_legal_ids.len() as i64);
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
                legal_ids: legal_ids.into_pyarray(py).unbind(),
                legal_offsets: legal_offsets.into_pyarray(py).unbind(),
                next_legal_ids: next_legal_ids.into_pyarray(py).unbind(),
                next_legal_offsets: next_legal_offsets.into_pyarray(py).unbind(),
                telemetry: telemetry.unbind(),
            },
        )?
        .into_any())
    }
}

/// Which Python `infer` contract the composed policy family speaks — the factory sets it, the collect
/// path builds the matching marshaling closure. Value families return `(N, K, A)` per-head values;
/// the AlphaZero family returns a `(policy_logits (N, A), values (N,))` tuple.
#[derive(Clone, Copy)]
enum InferLayout {
    ValueHeads,
    PolicyValue,
}

/// Shared rollout: drive any `Engine<G, P, L>` for `n_records`, returning the records and the (uniform)
/// telemetry dict. Search aggregates are zero for a search-less policy (its `fold_telemetry` is a no-op).
#[allow(clippy::too_many_arguments)]
fn run_collect<'py, G, P, L>(
    inner: &mut Engine<G, P, L>,
    py: Python<'py>,
    n_records: usize,
    infer: &Bound<'_, PyAny>,
    dim: usize,
    action_count: usize,
    n_heads: usize,
    layout: InferLayout,
) -> PyResult<(Vec<L::Record>, Bound<'py, PyDict>)>
where
    G: Game + Sync,
    G::State: Send,
    P: Policy,
    L: Learner<P::Evaluation>,
{
    let mut callback_err: Option<PyErr> = None;
    let (records, stats) = match layout {
        InferLayout::ValueHeads => {
            let mut infer_fn = infer_closure(
                py,
                infer,
                dim,
                action_count,
                Some(n_heads),
                &mut callback_err,
            );
            inner.collect(n_records, &mut infer_fn)
        }
        InferLayout::PolicyValue => {
            let mut infer_fn = az_infer_closure(py, infer, dim, action_count, &mut callback_err);
            inner.collect(n_records, &mut infer_fn)
        }
    };
    if let Some(e) = callback_err {
        return Err(e);
    }
    let telemetry = build_telemetry(py, &stats)?;
    Ok((records, telemetry))
}

/// The uniform collect telemetry dict, built from the core stats. Shared by the synchronous collect
/// path and the stream thunks (which marshal on the consumer thread, after the collect ran).
fn build_telemetry<'py>(
    py: Python<'py>,
    stats: &reinfors_core::CollectStats,
) -> PyResult<Bound<'py, PyDict>> {
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
    // Net-forward timing: total time in `infer`, call count, and rows summed across calls. Lets a caller
    // split collect time into infer vs search, and see the real per-call batch size.
    telemetry.set_item("infer_seconds", stats.infer_seconds)?;
    telemetry.set_item("infer_calls", stats.infer_calls)?;
    telemetry.set_item("infer_rows", stats.infer_rows)?;
    // Global infer-cache telemetry, all consumers (zeros when disabled). `infer_rows` counts only
    // miss rows, so rows-per-state falls as the hit rate rises.
    telemetry.set_item("cache_lookups", stats.cache_lookups)?;
    telemetry.set_item("cache_hits", stats.cache_hits)?;
    // Tree-search sim fates, counted by the trees themselves (search-local, exact):
    //   decisions × num_simulations =
    //     fresh_rows + hit_rows + shared_rows + terminal_sims + depthcap_sims
    telemetry.set_item("terminal_sims", stats.sum_terminal_sims)?;
    telemetry.set_item("depthcap_sims", stats.sum_depthcap_sims)?;
    telemetry.set_item("shared_rows", stats.sum_shared_rows)?;
    telemetry.set_item("fresh_rows", stats.sum_fresh_rows)?;
    telemetry.set_item("hit_rows", stats.sum_hit_rows)?;
    // ExpandAll chance fans: rows beyond one per simulation (the identity subtracts this term).
    telemetry.set_item("extra_eval_rows", stats.sum_extra_eval_rows)?;
    Ok(telemetry)
}

/// One generic wrapper: a composed engine + its sizing metadata, type-erased behind `ErasedEngine`.
/// A single blanket impl serves *every* `(G, P, L)` whose learner record can marshal to a batch — so a
/// new game, policy, or learner needs no new wrapper or impl, only a factory arm (and, for a genuinely
/// new record shape, a `RecordBatch` impl).
struct EngineImpl<G: Game + Sync, P: Policy, L: Learner<P::Evaluation>> {
    inner: Engine<G, P, L>,
    // The game's persistence codec — `None` = engine snapshots unsupported for this game.
    codec: Option<Box<dyn StateCodec<State = G::State>>>,
    dim: usize,
    action_count: usize,
    n_heads: usize,
    layout: InferLayout,
}

impl<G, P, L> ErasedEngine for EngineImpl<G, P, L>
where
    G: Game + Send + Sync + 'static,
    G::State: Send + Sync,
    P: Policy + Send + Sync + 'static,
    P::Evaluation: Send + Sync,
    P::PolicyState: Send + Sync,
    L: Learner<P::Evaluation> + Send + Sync + 'static,
    L::Record: RecordBatch + Send + 'static,
{
    fn snapshot_payload(&self) -> PyResult<Vec<u8>> {
        let codec = self.codec.as_deref().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("this game does not support engine snapshots")
        })?;
        self.inner
            .snapshot_bytes(codec)
            .map_err(pyo3::exceptions::PyValueError::new_err)
    }

    fn restore_payload(&mut self, bytes: &[u8]) -> PyResult<()> {
        let codec = self.codec.as_deref().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("this game does not support engine snapshots")
        })?;
        self.inner.restore_bytes(codec, bytes).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("invalid engine snapshot: {e}"))
        })
    }

    fn collect_thunk(
        &mut self,
        n_records: usize,
        infer: &Py<PyAny>,
        stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
    ) -> PyResult<BatchThunk> {
        let callback_err = std::sync::Arc::new(std::sync::Mutex::new(None));
        let mut infer_fn = infer_closure_gil(
            infer,
            self.dim,
            self.action_count,
            self.n_heads,
            self.layout,
            stop,
            callback_err.clone(),
        );
        let (records, stats) = self.inner.collect(n_records, &mut infer_fn);
        if let Some(e) = callback_err.lock().unwrap().take() {
            return Err(e);
        }
        let (dim, n_heads) = (self.dim, self.n_heads);
        Ok(Box::new(move |py: Python<'_>| {
            let telemetry = build_telemetry(py, &stats)?;
            Ok(L::Record::into_py_batch(records, py, dim, n_heads, telemetry)?.unbind())
        }))
    }

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
            self.layout,
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
        max_ticks: Option<usize>,
    },
    Connect4,
    Chess {
        max_ticks: Option<usize>,
        encoder: ChessEncoderSpec,
    },
    Backgammon {
        max_ticks: Option<usize>,
    },
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
            GameSpec::Chess { max_ticks, encoder } => {
                let (game, enc) = chess_parts(max_ticks, encoder);
                of(game, &*enc)
            }
            GameSpec::Backgammon { max_ticks } => of(Backgammon { max_ticks }, &BackgammonTesauro),
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
/// The per-game reward schema (component names + defaults) — the single table `build_reward`
/// resolves against and `resolved_config` renders from, so the two cannot drift.
fn reward_schema(game: &GameSpec) -> &'static [(&'static str, f64)] {
    match game {
        GameSpec::Snake { .. } => &[
            ("step", 0.0),
            ("food", 0.0),
            ("loss", 0.0),
            ("draw", 0.0),
            ("kill", 0.0),
            ("win", 0.0),
            ("survival", 0.0),
        ],
        GameSpec::Connect4 => &[("win", 1.0), ("loss", -1.0), ("draw", 0.0)],
        GameSpec::Chess { .. } => &[("win", 1.0), ("loss", -1.0), ("draw", 0.0)],
        GameSpec::Backgammon { .. } => &[("win", 1.0), ("gammon", 2.0), ("backgammon", 3.0)],
        GameSpec::GridWorld { .. } => &[("step", 0.0), ("goal", 1.0)],
    }
}

fn build_reward(game: &GameSpec, reward: Option<PyReward>) -> PyResult<RewardBox> {
    Ok(match game {
        GameSpec::Snake { .. } => {
            let r = resolve_reward(reward, reward_schema(game))?;
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
            let r = resolve_reward(reward, reward_schema(game))?;
            RewardBox::Connect4(Connect4Reward {
                win: r[0],
                loss: r[1],
                draw: r[2],
            })
        }
        GameSpec::Chess { .. } => {
            let r = resolve_reward(reward, reward_schema(game))?;
            RewardBox::Chess(ChessReward {
                win: r[0],
                loss: r[1],
                draw: r[2],
            })
        }
        GameSpec::Backgammon { .. } => {
            // Margin-aware zero-sum payoffs (the loser scores the negative automatically).
            let r = resolve_reward(reward, reward_schema(game))?;
            RewardBox::Backgammon(BackgammonReward {
                win: r[0],
                gammon: r[1],
                backgammon: r[2],
            })
        }
        GameSpec::GridWorld { .. } => {
            let r = resolve_reward(reward, reward_schema(game))?;
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
    Chess(ChessReward),
    Backgammon(BackgammonReward),
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
        chance: ChanceMode,
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
        temperature: f64,
        temperature_drop: u32,
        chance: ChanceMode,
    },
    AlphaZero {
        num_simulations: usize,
        c_puct: f64,
        max_depth: i32,
        noise_epsilon: f64,
        noise_alpha: f64,
        temperature: f64,
        temperature_drop: u32,
        chance: ChanceMode,
        noise_scope: NoiseScope,
    },
}

/// Learning-algorithm configuration. TreeStrap's `gamma` is also threaded into the search config by
/// the factory, so the search and the z-mix share one discount (AlphaZero's likewise).
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
    AlphaZero {
        gamma: f64,
    },
}

/// Reject a probability/weight outside `[0, 1]`.
/// Reject a non-finite or non-positive value — for parameters that DIVIDE (a zero temperature
/// turns the distributional softmax into NaN, which panics downstream comparisons).
fn check_positive_finite(name: &str, v: f64) -> PyResult<()> {
    if !v.is_finite() || v <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "{name} must be finite and > 0"
        )));
    }
    Ok(())
}

/// Reject a non-finite or negative value (NaN-proof: `!(v >= 0.0)` is true for NaN).
fn check_nonneg_finite(name: &str, v: f64) -> PyResult<()> {
    if !v.is_finite() || v < 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "{name} must be finite and >= 0"
        )));
    }
    Ok(())
}

fn check_unit(name: &str, v: f64) -> PyResult<()> {
    if !(0.0..=1.0).contains(&v) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "{name} must be in [0, 1]"
        )));
    }
    Ok(())
}

/// The agent-count capability gate (no-panic contract): a policy that cannot plan for this many
/// agents is a config error here, before `Engine::new`'s assert backstop.
fn check_max_agents<P: Policy>(policy: &P, label: &str, num_agents: usize) -> PyResult<()> {
    if num_agents == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "num_agents must be > 0; not {num_agents}"
        )));
    }
    match policy.max_agents() {
        Some(cap) if num_agents > cap => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "the {label} policy supports at most {cap} agents; this game has {num_agents}"
        ))),
        _ => Ok(()),
    }
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
    mut codec: Option<Box<dyn StateCodec<State = G::State>>>,
    policy: PolicySpec,
    learner: LearnerSpec,
    engine_params: EngineParams,
    infer_cache: Option<InferCache>,
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
                chance,
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
            validate_search_params(expansion_budget, top_k, max_depth, beta)?;
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
                chance,
                opponent,
            };
            let policy = SelectiveExpectimax::new(cfg, n_heads, epsilon);
            check_max_agents(&policy, "SelectiveExpectimax", game.num_agents())?;
            let learner = TreeStrap::new(gamma, outcome_weight, bootstrap_p, interior_targets);
            Ok(Box::new(EngineImpl {
                codec: codec.take(),
                inner: {
                    let mut e = Engine::new(game, enc, reward, policy, learner, engine_params)
                        .with_start_distribution(start_dist);
                    if let Some(c) = infer_cache {
                        e = e.with_infer_cache(c);
                    }
                    e
                },
                dim,
                action_count,
                n_heads,
                layout: InferLayout::ValueHeads,
            }))
        }
        (
            PolicySpec::Mcts {
                num_simulations,
                uct_c,
                max_depth,
                act_by,
                temperature,
                temperature_drop,
                chance,
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
            if !(temperature >= 0.0 && temperature.is_finite()) {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "temperature must be finite and >= 0",
                ));
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
                    temperature,
                    temperature_drop,
                    chance,
                },
                act_by,
            );
            check_max_agents(&policy, "Mcts", game.num_agents())?;
            let learner = TreeStrap::new(gamma, outcome_weight, bootstrap_p, false);
            Ok(Box::new(EngineImpl {
                codec: codec.take(),
                inner: {
                    let mut e = Engine::new(game, enc, reward, policy, learner, engine_params)
                        .with_start_distribution(start_dist);
                    if let Some(c) = infer_cache {
                        e = e.with_infer_cache(c);
                    }
                    e
                },
                dim,
                action_count,
                n_heads: 1,
                layout: InferLayout::ValueHeads,
            }))
        }
        (
            PolicySpec::AlphaZero {
                num_simulations,
                c_puct,
                max_depth,
                noise_epsilon,
                noise_alpha,
                temperature,
                temperature_drop,
                chance,
                noise_scope,
            },
            LearnerSpec::AlphaZero { gamma },
        ) => {
            // >= 2: sim 1 evaluates the root itself, so visit-bearing π targets need at least one more.
            if num_simulations < 2 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "num_simulations must be >= 2",
                ));
            }
            if max_depth < 1 {
                return Err(pyo3::exceptions::PyValueError::new_err("max_depth must be >= 1"));
            }
            if c_puct < 0.0 {
                return Err(pyo3::exceptions::PyValueError::new_err("c_puct must be >= 0"));
            }
            check_unit("noise_epsilon", noise_epsilon)?;
            if !(noise_alpha > 0.0 && noise_alpha.is_finite()) {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "noise_alpha must be finite and > 0",
                ));
            }
            if !(temperature >= 0.0 && temperature.is_finite()) {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "temperature must be finite and >= 0",
                ));
            }
            let policy = AlphaZero::new(AlphaZeroConfig {
                num_simulations,
                c_puct,
                gamma,
                max_depth,
                noise_epsilon,
                noise_alpha,
                temperature,
                temperature_drop,
                chance,
                noise_scope,
            });
            check_max_agents(&policy, "AlphaZero", game.num_agents())?;
            let learner = AlphaZeroLearner::new(gamma);
            Ok(Box::new(EngineImpl {
                codec: codec.take(),
                inner: {
                    let mut e = Engine::new(game, enc, reward, policy, learner, engine_params)
                        .with_start_distribution(start_dist);
                    if let Some(c) = infer_cache {
                        e = e.with_infer_cache(c);
                    }
                    e
                },
                dim,
                action_count,
                n_heads: 1, // single value head; π targets are (M, A), no bootstrap masks
                layout: InferLayout::PolicyValue,
            }))
        }
        (PolicySpec::EpsilonGreedyQ { n_heads, epsilon }, LearnerSpec::Dqn { bootstrap_p }) => {
            if n_heads < 1 {
                return Err(pyo3::exceptions::PyValueError::new_err("n_heads must be >= 1"));
            }
            check_unit("epsilon", epsilon)?;
            check_unit("bootstrap_p", bootstrap_p)?;
            let policy = EpsilonGreedyQ::new(n_heads, epsilon);
            check_max_agents(&policy, "EpsilonGreedyQ", game.num_agents())?;
            let learner = Dqn::new(n_heads, bootstrap_p);
            Ok(Box::new(EngineImpl {
                codec: codec.take(),
                inner: {
                    let mut e = Engine::new(game, enc, reward, policy, learner, engine_params)
                        .with_start_distribution(start_dist);
                    if let Some(c) = infer_cache {
                        e = e.with_infer_cache(c);
                    }
                    e
                },
                dim,
                action_count,
                n_heads,
                layout: InferLayout::ValueHeads,
            }))
        }
        _ => Err(pyo3::exceptions::PyValueError::new_err(
            "incompatible policy/learner: TreeStrap pairs with SelectiveExpectimax or Mcts, Dqn with \
             EpsilonGreedyQ, and AlphaZero (policy) with AlphaZero (learner)",
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
    infer_cache: Option<InferCache>,
) -> PyResult<Box<dyn ErasedEngine>> {
    let reward = build_reward(&game, reward)?;
    // The reached-state buffer needs a game-specific cell key; only snake supplies one in v1.
    if start_buffer.is_some() && !matches!(game, GameSpec::Snake { .. }) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "start_buffer is only supported for the snake game",
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
                Some(Box::new(Snake {
                    grid_size,
                    initial_length,
                    play_to_last,
                    win_food_lead,
                    initial_food_count,
                    max_ticks,
                })),
                policy,
                learner,
                engine_params,
                infer_cache,
            )
        }
        (GameSpec::Connect4, RewardBox::Connect4(reward)) => build_for_game(
            Connect4,
            Box::new(Connect4Planes),
            Box::new(reward),
            Box::new(AlwaysInitialState),
            Some(Box::new(Connect4)),
            policy,
            learner,
            engine_params,
            infer_cache,
        ),
        (GameSpec::Chess { max_ticks, encoder }, RewardBox::Chess(reward)) => {
            let (game, enc) = chess_parts(max_ticks, encoder);
            build_for_game(
                game,
                enc,
                Box::new(reward),
                Box::new(AlwaysInitialState),
                Some(Box::new(chess_parts(max_ticks, encoder).0)),
                policy,
                learner,
                engine_params,
                infer_cache,
            )
        }
        (GameSpec::Backgammon { max_ticks }, RewardBox::Backgammon(reward)) => build_for_game(
            Backgammon { max_ticks },
            Box::new(BackgammonTesauro),
            Box::new(reward),
            Box::new(AlwaysInitialState),
            Some(Box::new(Backgammon { max_ticks })),
            policy,
            learner,
            engine_params,
            infer_cache,
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
            Some(Box::new(GridWorld {
                size,
                goal,
                max_ticks,
            })),
            policy,
            learner,
            engine_params,
            infer_cache,
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

const ENV_SNAPSHOT_SCHEMA: u8 = 1;
const ENV_SNAPSHOT_MAGIC: &[u8; 4] = b"RFES";

/// An opaque, restorable point-in-time capture of an `Env`: native game state (via the game's
/// `StateCodec`), env RNG state, terminal status, plus the env's config fingerprint and a schema
/// version. NOT the inspection format — `env.state()` stays the human-readable view; this one is
/// produced by reinfors and validated while decoding, so a malformed blob is a `ValueError`,
/// never a corrupted env.
#[pyclass(name = "EnvSnapshot")]
#[derive(Clone)]
struct PyEnvSnapshot {
    schema: u8,
    fingerprint: String,
    state: Vec<u8>,
    rng_state: u64,
    done: bool,
}

#[pymethods]
impl PyEnvSnapshot {
    /// The captured env's config fingerprint (game + reward; excludes the reinfors version, so a
    /// snapshot survives a library upgrade with an unchanged schema).
    #[getter]
    fn fingerprint(&self) -> &str {
        &self.fingerprint
    }

    #[getter]
    fn schema_version(&self) -> u8 {
        self.schema
    }

    /// Serialize to a self-describing byte blob (magic + schema + fingerprint + payload).
    fn to_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, pyo3::types::PyBytes> {
        let mut out = Vec::with_capacity(self.state.len() + self.fingerprint.len() + 32);
        out.extend_from_slice(ENV_SNAPSHOT_MAGIC);
        out.push(self.schema);
        out.extend_from_slice(&(self.fingerprint.len() as u32).to_le_bytes());
        out.extend_from_slice(self.fingerprint.as_bytes());
        out.extend_from_slice(&self.rng_state.to_le_bytes());
        out.push(u8::from(self.done));
        out.extend_from_slice(&(self.state.len() as u32).to_le_bytes());
        out.extend_from_slice(&self.state);
        pyo3::types::PyBytes::new(py, &out)
    }

    /// Parse a blob produced by `to_bytes`. Structure is validated here; the game state inside is
    /// validated by the game's codec at `restore` time.
    #[staticmethod]
    fn from_bytes(data: &[u8]) -> PyResult<Self> {
        let err =
            |m: &str| pyo3::exceptions::PyValueError::new_err(format!("invalid EnvSnapshot: {m}"));
        let take = |data: &[u8], pos: &mut usize, n: usize| -> PyResult<Vec<u8>> {
            let end = pos
                .checked_add(n)
                .filter(|&e| e <= data.len())
                .ok_or_else(|| err("truncated"))?;
            let out = data[*pos..end].to_vec();
            *pos = end;
            Ok(out)
        };
        let mut pos = 0usize;
        if take(data, &mut pos, 4)? != ENV_SNAPSHOT_MAGIC {
            return Err(err("bad magic"));
        }
        let schema = take(data, &mut pos, 1)?[0];
        if schema != ENV_SNAPSHOT_SCHEMA {
            return Err(err(&format!("unsupported schema version {schema}")));
        }
        let fp_len = u32::from_le_bytes(take(data, &mut pos, 4)?.try_into().unwrap()) as usize;
        let fingerprint = String::from_utf8(take(data, &mut pos, fp_len)?)
            .map_err(|_| err("fingerprint is not utf-8"))?;
        let rng_state = u64::from_le_bytes(take(data, &mut pos, 8)?.try_into().unwrap());
        let done = match take(data, &mut pos, 1)?[0] {
            0 => false,
            1 => true,
            b => return Err(err(&format!("done byte {b} is not a bool"))),
        };
        let state_len = u32::from_le_bytes(take(data, &mut pos, 4)?.try_into().unwrap()) as usize;
        let state = take(data, &mut pos, state_len)?;
        if pos != data.len() {
            return Err(err("trailing bytes"));
        }
        Ok(PyEnvSnapshot {
            schema,
            fingerprint,
            state,
            rng_state,
            done,
        })
    }
}

#[pyclass(name = "Env")]
struct PyEnv {
    inner: Box<dyn ErasedEnv>,
    // Retained for `fork` (rebuilds the composition) and `resolved_config`/snapshot fingerprints.
    game_spec: GameSpec,
    reward_weights: Option<HashMap<String, f64>>,
    config: Value,
    fingerprint: String,
}

impl PyEnv {
    /// Boundary check for the agent-indexed methods: out-of-range indices must be a `ValueError`,
    /// not a panic (snake indexes `snakes[agent]`) or — worse — a silently wrong answer (connect4
    /// encodes any unknown index from player 0's perspective).
    fn check_agent(&self, agent: usize) -> PyResult<()> {
        let n = self.inner.num_agents();
        if agent >= n {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "agent {agent} out of range for a {n}-agent game"
            )));
        }
        Ok(())
    }
}

#[pymethods]
impl PyEnv {
    #[new]
    #[pyo3(signature = (game, reward=None, seed=0))]
    fn new(game: GameHandle, reward: Option<PyReward>, seed: u64) -> PyResult<Self> {
        let game_spec = game.spec.clone();
        let reward_weights = reward.as_ref().map(|r| r.weights.clone());
        let reward_cfg = match &reward_weights {
            None => Value::Null, // reward-free (play/eval): rewards is always None
            Some(w) => {
                let vals = resolve_reward(
                    Some(PyReward { weights: w.clone() }),
                    reward_schema(&game_spec),
                )?;
                Value::Object(
                    reward_schema(&game_spec)
                        .iter()
                        .zip(vals)
                        .map(|((k, _), v)| ((*k).to_string(), json!(v)))
                        .collect(),
                )
            }
        };
        let config = json!({
            "schema_version": ENV_SNAPSHOT_SCHEMA,
            "game": game_cfg(&game_spec),
            "reward": reward_cfg,
        });
        let fingerprint = fingerprint_hex(&canonical_config_bytes(&config));
        Ok(PyEnv {
            inner: build_env(game.spec, reward, seed)?,
            game_spec,
            reward_weights,
            config,
            fingerprint,
        })
    }

    /// The env's immutable composition (game incl. encoder + resolved reward), JSON-compatible.
    fn resolved_config<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        value_to_py(py, &self.config)
    }

    /// Fingerprint of `resolved_config` — embedded in snapshots and checked on `restore`.
    /// Excludes the reinfors version: a snapshot survives an upgrade with an unchanged schema.
    fn config_fingerprint(&self) -> String {
        self.fingerprint.clone()
    }

    /// An opaque, restorable capture of the env right now (state + RNG + terminal status).
    fn snapshot(&self) -> PyResult<PyEnvSnapshot> {
        let (state, rng_state, done) = self.inner.snapshot_parts()?;
        Ok(PyEnvSnapshot {
            schema: ENV_SNAPSHOT_SCHEMA,
            fingerprint: self.fingerprint.clone(),
            state,
            rng_state,
            done,
        })
    }

    /// Install a snapshot. Rejects a snapshot from a different composition (fingerprint), an
    /// unsupported schema, or malformed state bytes (the game codec validates while decoding).
    /// Restore lands at a step boundary: `rewards` is None until the next step.
    fn restore(&mut self, snapshot: &PyEnvSnapshot) -> PyResult<()> {
        if snapshot.schema != ENV_SNAPSHOT_SCHEMA {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "unsupported snapshot schema {}",
                snapshot.schema
            )));
        }
        if snapshot.fingerprint != self.fingerprint {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "snapshot is from a different composition (fingerprint {} != {})",
                snapshot.fingerprint, self.fingerprint
            )));
        }
        self.inner
            .restore_parts(&snapshot.state, snapshot.rng_state, snapshot.done)
    }

    /// A new independent env at this env's exact current point. Clone-exact by default: identical
    /// state AND identical future chance stream (replay/analysis semantics — the fork and the
    /// original make the same draws). Pass `seed` to give the fork a divergent chance stream.
    #[pyo3(signature = (seed=None))]
    fn fork(&self, seed: Option<u64>) -> PyResult<PyEnv> {
        let reward = self
            .reward_weights
            .clone()
            .map(|weights| PyReward { weights });
        let mut forked = PyEnv {
            inner: build_env(self.game_spec.clone(), reward, 0)?,
            game_spec: self.game_spec.clone(),
            reward_weights: self.reward_weights.clone(),
            config: self.config.clone(),
            fingerprint: self.fingerprint.clone(),
        };
        let (state, rng_state, done) = self.inner.snapshot_parts()?;
        forked
            .inner
            .restore_parts(&state, seed.unwrap_or(rng_state), done)?;
        Ok(forked)
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

    fn legal_actions(&self, agent: usize) -> PyResult<Vec<usize>> {
        self.check_agent(agent)?;
        Ok(self.inner.legal_actions(agent))
    }

    /// The encoded observation for `agent` as a `(C, H, W)` float32 array (the value-network view).
    fn observe<'py>(&self, py: Python<'py>, agent: usize) -> PyResult<Bound<'py, PyArray3<f32>>> {
        self.check_agent(agent)?;
        Ok(self.inner.observe(py, agent))
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
        // Validate legality at the boundary: an illegal action id can corrupt game state (e.g.
        // backgammon decoding moves for checkers that are not there), so it never enters the core.
        for (&agent, &action) in &actions {
            let legal = self.inner.legal_actions(agent);
            if !legal.contains(&action) {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "action {action} is illegal for agent {agent} (legal: {legal:?})"
                )));
            }
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

impl NativeState for ChessState {
    fn to_py<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("fen", self.fen())?;
        d.set_item("turn", self.turn())?;
        d.set_item("done", self.is_done())?;
        Ok(d)
    }
}

impl NativeState for BackgammonState {
    fn to_py<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("board", [self.board[0].to_vec(), self.board[1].to_vec()])?;
        d.set_item("bar", self.bar.to_vec())?;
        d.set_item("scores", self.scores.to_vec())?;
        d.set_item("to_move", self.to_move)?;
        d.set_item("dice", self.dice.to_vec())?;
        d.set_item("double_turn", self.double_turn)?;
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

impl NativeEvent for BackgammonEvent {
    fn to_py<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let d = PyDict::new(py);
        let (result, margin) = match self {
            BackgammonEvent::Ongoing => ("ongoing", 0u8),
            BackgammonEvent::Win(m) => ("win", *m),
            BackgammonEvent::Loss(m) => ("loss", *m),
        };
        d.set_item("result", result)?;
        d.set_item("margin", margin)?; // 1 plain, 2 gammon, 3 backgammon
        Ok(d.into_any())
    }
}

impl NativeEvent for ChessEvent {
    fn to_py<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let s = match self {
            ChessEvent::Ongoing => "ongoing",
            ChessEvent::Win => "win",
            ChessEvent::Loss => "loss",
            ChessEvent::Draw => "draw",
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
    /// Snapshot parts: `(codec state bytes, rng state, done)`. Errors if the game has no codec.
    fn snapshot_parts(&self) -> PyResult<(Vec<u8>, u64, bool)>;
    /// Install parts (state bytes decoded + validated by the codec).
    fn restore_parts(&mut self, state: &[u8], rng_state: u64, done: bool) -> PyResult<()>;
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
    // The game's persistence codec (built-ins all supply one); `None` = snapshots unsupported.
    codec: Option<Box<dyn StateCodec<State = G::State>>>,
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
    fn snapshot_parts(&self) -> PyResult<(Vec<u8>, u64, bool)> {
        let codec = self.codec.as_deref().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("this game does not support snapshots")
        })?;
        let (state, rng_state, done) = self.inner.parts();
        Ok((codec.encode(&state), rng_state, done))
    }

    fn restore_parts(&mut self, state: &[u8], rng_state: u64, done: bool) -> PyResult<()> {
        let codec = self.codec.as_deref().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("this game does not support snapshots")
        })?;
        let decoded = codec.decode(state).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("invalid snapshot state: {e}"))
        })?;
        codec.validate_decoded_state(&decoded, done).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("invalid snapshot state: {e}"))
        })?;
        self.inner.set_parts(decoded, rng_state, done);
        self.last_rewards = None; // transient last-step output belongs to the step that produced it
        Ok(())
    }

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
                codec: Some(Box::new(Snake {
                    grid_size,
                    initial_length,
                    play_to_last,
                    win_food_lead,
                    initial_food_count,
                    max_ticks,
                })),
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
                codec: Some(Box::new(Connect4)),
                reward: reward.map(|rb| match rb {
                    RewardBox::Connect4(r) => Box::new(r) as Box<dyn Reward<Event = Connect4Event>>,
                    _ => unreachable!("build_reward returns the reward variant matching the game"),
                }),
                last_rewards: None,
            })
        }
        GameSpec::Chess { max_ticks, encoder } => {
            let (game, enc) = chess_parts(max_ticks, encoder);
            let obs_shape = enc.obs_shape();
            Box::new(EnvImpl {
                inner: Env::new(game, enc, seed),
                obs_shape,
                codec: Some(Box::new(chess_parts(max_ticks, encoder).0)),
                reward: reward.map(|rb| match rb {
                    RewardBox::Chess(r) => Box::new(r) as Box<dyn Reward<Event = ChessEvent>>,
                    _ => unreachable!("build_reward returns the reward variant matching the game"),
                }),
                last_rewards: None,
            })
        }
        GameSpec::Backgammon { max_ticks } => {
            let enc = BackgammonTesauro;
            let obs_shape = enc.obs_shape();
            Box::new(EnvImpl {
                inner: Env::new(Backgammon { max_ticks }, Box::new(enc), seed),
                obs_shape,
                codec: Some(Box::new(Backgammon { max_ticks })),
                reward: reward.map(|rb| match rb {
                    RewardBox::Backgammon(r) => {
                        Box::new(r) as Box<dyn Reward<Event = BackgammonEvent>>
                    }
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
                codec: Some(Box::new(GridWorld {
                    size,
                    goal,
                    max_ticks,
                })),
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
    fn new(weights: Option<HashMap<String, f64>>) -> PyResult<Self> {
        let weights = weights.unwrap_or_default();
        // NaN/inf don't panic — worse, they silently poison every value they touch downstream.
        if let Some((k, v)) = weights.iter().find(|(_, v)| !v.is_finite()) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "reward weight {k:?} must be finite, got {v}"
            )));
        }
        Ok(PyReward { weights })
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
        // Validate by constructing: the game's own invariants are the single source of truth.
        Snake {
            grid_size,
            initial_length,
            initial_food_count: food,
            play_to_last,
            win_food_lead,
            max_ticks,
        }
        .validate()
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
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
    // Weak-net self-play can shuffle indefinitely inside the fifty-move window, so `max_ticks`
    // defaults to a finite cap (pass `max_ticks=None` to opt into never truncating). `encoder` is an
    // `rf.encoders.*` handle picking the observation view (default: `MinimalChess`); the state's
    // history bookkeeping follows the selected encoder automatically.
    #[pyo3(signature = (max_ticks=512, encoder=None))]
    #[pyo3(name = "Chess")]
    fn chess(max_ticks: Option<usize>, encoder: Option<EncoderHandle>) -> PyResult<Self> {
        check_max_ticks(max_ticks)?;
        let encoder = encoder.map_or(ChessEncoderSpec::Minimal, |e| e.chess);
        Ok(GameHandle {
            spec: GameSpec::Chess { max_ticks, encoder },
        })
    }

    #[staticmethod]
    // Backgammon with the OpenSpiel-compatible 1352-action encoding and declared dice chance; no
    // doubling cube. Reward keys: win/gammon/backgammon (defaults 1/2/3, zero-sum). Weak nets can
    // shuffle checkers for a long time, so `max_ticks` defaults to a finite cap.
    #[pyo3(signature = (max_ticks=1000))]
    #[pyo3(name = "Backgammon")]
    fn backgammon(max_ticks: Option<usize>) -> PyResult<Self> {
        check_max_ticks(max_ticks)?;
        Ok(GameHandle {
            spec: GameSpec::Backgammon { max_ticks },
        })
    }

    #[staticmethod]
    // GridWorld can wander forever without reaching the goal, so `max_ticks` defaults to a finite cap
    // (pass `max_ticks=None` to opt into never truncating). The goal defaults to the far corner,
    // DERIVED from `size` — an absolute default would silently sit mid-grid (or out of it) for other
    // sizes.
    #[pyo3(signature = (size=5, goal_row=None, goal_col=None, max_ticks=1000))]
    #[pyo3(name = "GridWorld")]
    fn gridworld(
        size: i32,
        goal_row: Option<i32>,
        goal_col: Option<i32>,
        max_ticks: Option<usize>,
    ) -> PyResult<Self> {
        check_max_ticks(max_ticks)?;
        // saturating: a nonsense `size` must reach validate() as an error, not overflow here
        let corner = size.saturating_sub(1);
        let goal = (goal_row.unwrap_or(corner), goal_col.unwrap_or(corner));
        GridWorld {
            size,
            goal,
            max_ticks,
        }
        .validate()
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(GameHandle {
            spec: GameSpec::GridWorld {
                size,
                goal,
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
            GameSpec::Snake { max_ticks, .. }
            | GameSpec::Chess { max_ticks, .. }
            | GameSpec::Backgammon { max_ticks }
            | GameSpec::GridWorld { max_ticks, .. } => max_ticks,
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
    #[pyo3(signature = (expansion_budget=64, top_k=8, max_depth=12, beta=1.0, chance=None, n_heads=1, epsilon=0.0, opponent="uniform", opp_temperature=1.0, opp_floor=0.0))]
    #[pyo3(name = "SelectiveExpectimax")]
    #[allow(clippy::too_many_arguments)]
    fn selective_expectimax(
        expansion_budget: usize,
        top_k: usize,
        max_depth: i32,
        beta: f64,
        chance: Option<ChanceModeHandle>,
        n_heads: usize,
        epsilon: f64,
        opponent: &str,
        opp_temperature: f64,
        opp_floor: f64,
    ) -> PyResult<Self> {
        // Default: the historical `food_samples = 1` estimator, on the shared vocabulary.
        check_unit("beta", beta)?;
        check_unit("epsilon", epsilon)?;
        check_unit("opp_floor", opp_floor)?;
        check_positive_finite("opp_temperature", opp_temperature)?;
        let chance = chance.map_or(ChanceMode::Committed { samples: 1 }, |c| c.mode);
        if !SelectiveExpectimax::supports_chance_mode(chance) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "SelectiveExpectimax expands each node exactly once (best-first) and cannot \
                 express per-traversal chance modes; use Committed or ExpandAll",
            ));
        }
        Ok(PolicyHandle {
            spec: PolicySpec::SelectiveExpectimax {
                beta,
                expansion_budget,
                top_k,
                max_depth,
                chance,
                opponent: parse_opponent(opponent, opp_temperature, opp_floor)?,
                n_heads,
                epsilon,
            },
        })
    }

    #[staticmethod]
    #[pyo3(signature = (n_heads=1, epsilon=0.1))]
    #[pyo3(name = "EpsilonGreedyQ")]
    fn epsilon_greedy_q(n_heads: usize, epsilon: f64) -> PyResult<Self> {
        check_unit("epsilon", epsilon)?;
        Ok(PolicyHandle {
            spec: PolicySpec::EpsilonGreedyQ { n_heads, epsilon },
        })
    }

    /// Monte-Carlo Tree Search (UCT). Pairs with `TreeStrap`. Sequential, single-agent, and
    /// simultaneous games (decoupled/DUCT per-agent statistics — snake included). `act_by` is `"value"` (argmax mean action value) or
    /// `"visits"` (argmax visit count). Acting defaults to deterministic (temperature 0) — ideal for
    /// evaluation and benchmarking. For training self-play diversity set `temperature > 0`
    /// (AlphaZero-style): the first `temperature_drop` plies of each episode are sampled
    /// `∝ visits^(1/temperature)` from the seeded acting RNG (later plies act greedily);
    /// `temperature_drop=None` applies it to the whole episode. Same seed → same games.
    /// `chance_mode` (games with declared `chance_outcomes` only; inert otherwise) picks how the
    /// search consumes stochastic transitions: `"always_resample"` (fresh draw ∝ probability every
    /// descent — unbiased, the asymptotically correct default), `"committed"` (freeze
    /// `chance_samples` draws at edge expansion and plan deeply inside them — expectimax's
    /// `food_samples` treatment, for fans wide relative to the sim budget), or `"expand_all"`
    /// (evaluate every outcome at expansion — exact, for narrow fans).
    #[staticmethod]
    #[pyo3(signature = (num_simulations=64, uct_c=2.0, max_depth=64, act_by="value", temperature=0.0, temperature_drop=None, chance=None))]
    #[pyo3(name = "Mcts")]
    #[allow(clippy::too_many_arguments)]
    fn mcts(
        num_simulations: usize,
        uct_c: f64,
        max_depth: i32,
        act_by: &str,
        temperature: f64,
        temperature_drop: Option<u32>,
        chance: Option<ChanceModeHandle>,
    ) -> PyResult<Self> {
        check_nonneg_finite("uct_c", uct_c)?;
        check_nonneg_finite("temperature", temperature)?;
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
                temperature,
                temperature_drop: temperature_drop.unwrap_or(u32::MAX),
                chance: chance.map_or(ChanceMode::AlwaysResample, |c| c.mode),
            },
        })
    }

    /// AlphaZero (PUCT) planner; pairs with `rf.learners.AlphaZero`; sequential, single-agent,
    /// and simultaneous (decoupled/DUCT) games. The net callback returns a `(policy_logits (N, A), values (N,))` tuple — one forward,
    /// both heads. Root Dirichlet noise `(1-noise_epsilon)·P + noise_epsilon·Dir(noise_alpha)`
    /// supplies search-level exploration (drawn from the seeded stream — collects stay reproducible);
    /// the acting temperature (same semantics as `Mcts`) supplies move-level diversity. Acting is by
    /// visit count (classic AlphaZero).
    /// `chance`: an `rf.chance_modes.*` handle (default `AlwaysResample`). `noise`: an
    /// `rf.noise.Dirichlet` handle; `None` disables root noise; omitted = the classic self-play
    /// default `Dirichlet(0.25, 0.3, "requester")`.
    #[staticmethod]
    #[pyo3(signature = (num_simulations=64, c_puct=1.5, max_depth=64, temperature=1.0, temperature_drop=8, chance=None, noise=Some(NoiseHandle::default_dirichlet())))]
    #[pyo3(name = "AlphaZero")]
    fn alphazero(
        num_simulations: usize,
        c_puct: f64,
        max_depth: i32,
        temperature: f64,
        temperature_drop: Option<u32>,
        chance: Option<ChanceModeHandle>,
        noise: Option<NoiseHandle>,
    ) -> PyResult<Self> {
        // `noise=None` = off (epsilon 0 internally — the core's noise-free path); the alpha/scope
        // placeholders are inert at epsilon 0.
        check_nonneg_finite("c_puct", c_puct)?;
        check_nonneg_finite("temperature", temperature)?;
        let (noise_epsilon, noise_alpha, noise_scope) = match noise {
            Some(n) => (n.epsilon, n.alpha, n.scope),
            None => (0.0, 0.3, NoiseScope::Requester),
        };
        Ok(PolicyHandle {
            spec: PolicySpec::AlphaZero {
                num_simulations,
                c_puct,
                max_depth,
                noise_epsilon,
                noise_alpha,
                temperature,
                temperature_drop: temperature_drop.unwrap_or(u32::MAX),
                chance: chance.map_or(ChanceMode::AlwaysResample, |c| c.mode),
                noise_scope,
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
    ) -> PyResult<Self> {
        check_unit("gamma", gamma)?;
        check_unit("outcome_weight", outcome_weight)?;
        check_unit("bootstrap_p", bootstrap_p)?;
        Ok(LearnerHandle {
            spec: LearnerSpec::TreeStrap {
                gamma,
                outcome_weight,
                bootstrap_p,
                interior_targets,
            },
        })
    }

    #[staticmethod]
    #[pyo3(signature = (bootstrap_p=1.0))]
    #[pyo3(name = "Dqn")]
    fn dqn(bootstrap_p: f64) -> PyResult<Self> {
        check_unit("bootstrap_p", bootstrap_p)?;
        Ok(LearnerHandle {
            spec: LearnerSpec::Dqn { bootstrap_p },
        })
    }

    /// AlphaZero record production: each decision -> `(obs, π, z)` — π the root visit distribution
    /// (τ=1), z the discounted realized return (γ=1 with win/loss rewards = the paper's z). Pairs with
    /// `rf.policies.AlphaZero`.
    #[staticmethod]
    #[pyo3(signature = (gamma=1.0))]
    #[pyo3(name = "AlphaZero")]
    fn alphazero(gamma: f64) -> PyResult<Self> {
        check_unit("gamma", gamma)?;
        Ok(LearnerHandle {
            spec: LearnerSpec::AlphaZero { gamma },
        })
    }
}

/// Chance-mode handle (`rf.chance_modes.*`): how a search consumes a stochastic transition's
/// declared distribution, passed to a search policy's `chance=` kwarg. Parameterized variants
/// carry their parameters here (the `rf.encoders` pattern) — no orphan kwargs on the policy.
#[pyclass]
#[derive(Clone)]
struct ChanceModeHandle {
    mode: ChanceMode,
}

#[pymethods]
impl ChanceModeHandle {
    /// Fresh draw proportional to probability on every descent — unbiased, the asymptotically
    /// correct default for sampled-trajectory searches; thin on fans wide relative to the budget.
    #[staticmethod]
    #[pyo3(name = "AlwaysResample")]
    fn always_resample() -> Self {
        ChanceModeHandle {
            mode: ChanceMode::AlwaysResample,
        }
    }

    /// Freeze `samples` draws per chance edge at expansion and plan deeply inside them (the
    /// expectimax `food_samples` estimator) — the wide-fan/small-budget trade.
    #[staticmethod]
    #[pyo3(signature = (samples=1))]
    #[pyo3(name = "Committed")]
    fn committed(samples: usize) -> PyResult<Self> {
        if samples < 1 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "samples must be >= 1",
            ));
        }
        Ok(ChanceModeHandle {
            mode: ChanceMode::Committed { samples },
        })
    }

    /// Materialize and evaluate every outcome at expansion (exact) — for narrow fans.
    #[staticmethod]
    #[pyo3(name = "ExpandAll")]
    fn expand_all() -> Self {
        ChanceModeHandle {
            mode: ChanceMode::ExpandAll,
        }
    }
}

/// Root-exploration-noise handle (`rf.noise.*`), passed to `rf.policies.AlphaZero(noise=...)`.
/// `noise=None` disables noise honestly (no `epsilon=0` sentinel); omitting the kwarg keeps the
/// classic self-play default. `scope` ("requester" | "both") only exists inside the config it
/// modifies: which root prior(s) the noise perturbs in a *simultaneous* search tree.
#[pyclass]
#[derive(Clone)]
struct NoiseHandle {
    epsilon: f64,
    alpha: f64,
    scope: NoiseScope,
}

impl NoiseHandle {
    /// The classic AlphaZero self-play default, used when the `noise=` kwarg is omitted.
    fn default_dirichlet() -> Self {
        NoiseHandle {
            epsilon: 0.25,
            alpha: 0.3,
            scope: NoiseScope::Requester,
        }
    }
}

#[pymethods]
impl NoiseHandle {
    /// Dirichlet root noise: `(1-epsilon)·P + epsilon·Dir(alpha)` at each search root, drawn from
    /// the seeded stream (collects stay reproducible).
    #[staticmethod]
    #[pyo3(signature = (epsilon=0.25, alpha=0.3, scope="requester"))]
    #[pyo3(name = "Dirichlet")]
    fn dirichlet(epsilon: f64, alpha: f64, scope: &str) -> PyResult<Self> {
        check_unit("epsilon", epsilon)?;
        if !(alpha > 0.0 && alpha.is_finite()) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "alpha must be finite and > 0",
            ));
        }
        let scope = match scope {
            "requester" => NoiseScope::Requester,
            "both" => NoiseScope::Both,
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "unknown scope {other:?}; expected \"requester\" or \"both\""
                )))
            }
        };
        Ok(NoiseHandle {
            epsilon,
            alpha,
            scope,
        })
    }
}

/// Observation-encoder handle (`rf.encoders.*`): a configurable view of a game's state, passed to
/// the game handle (e.g. `rf.games.Chess(encoder=rf.encoders.AlphaZeroChess(history_length=8))`).
/// Encoders are game-specific; the game handle validates the pairing.
#[pyclass]
#[derive(Clone)]
struct EncoderHandle {
    chess: ChessEncoderSpec,
}

#[pymethods]
impl EncoderHandle {
    /// The default chess view: (19, 8, 8) piece/castling/ep/clock planes, no history.
    #[staticmethod]
    #[pyo3(name = "MinimalChess")]
    fn minimal_chess() -> Self {
        EncoderHandle {
            chess: ChessEncoderSpec::Minimal,
        }
    }

    /// Mover-relative chess view, (19, 8, 8): the position seen from the mover's side (board
    /// rank-reflected and colors role-swapped for Black), with the action head indexed under the
    /// SAME symmetry — role equivariance as an inductive bias (the AlphaZero paper's convention).
    /// Layout mirrors MinimalChess with my/opponent planes in place of White/Black.
    #[staticmethod]
    #[pyo3(name = "RelativeChess")]
    fn relative_chess() -> Self {
        EncoderHandle {
            chess: ChessEncoderSpec::Relative,
        }
    }

    /// OpenSpiel's chess observation replicated exactly, (20, 8, 8) — the interop/benchmark view
    /// (identical net inputs on both sides of a comparison, including their encoding's
    /// en-passant blindness). Absolute frame; parity-gated against pyspiel in reinfors-benchmarks.
    #[staticmethod]
    #[pyo3(name = "OpenSpielChess")]
    fn openspiel_chess() -> Self {
        EncoderHandle {
            chess: ChessEncoderSpec::OpenSpiel,
        }
    }

    /// The encoder's action map: the net-head index of game action `action` from `agent`'s
    /// perspective (identity for absolute encoders). For driving a trained net OUTSIDE the
    /// engine (play/eval scripts): read logits at `head_index(a, agent)` for each legal game
    /// action `a` from `env.legal_actions`.
    fn head_index(&self, action: usize, agent: usize) -> PyResult<usize> {
        if action >= CHESS_ACTIONS {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "action {action} out of range for the {CHESS_ACTIONS}-action chess encoding"
            )));
        }
        let (_, enc) = chess_parts(None, self.chess);
        Ok(enc.head_index(action, agent))
    }

    /// Inverse of `head_index`.
    fn game_action(&self, head: usize, agent: usize) -> PyResult<usize> {
        if head >= CHESS_ACTIONS {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "head {head} out of range for the {CHESS_ACTIONS}-action chess encoding"
            )));
        }
        let (_, enc) = chess_parts(None, self.chess);
        Ok(enc.game_action(head, agent))
    }

    /// AlphaZero's chess view: `14·history_length + 7` planes (12 piece + 2 repetition planes per
    /// history step, newest first, + 7 auxiliaries). `history_length=8` reproduces the paper's 119.
    #[staticmethod]
    #[pyo3(signature = (history_length=8))]
    #[pyo3(name = "AlphaZeroChess")]
    fn alphazero_chess(history_length: usize) -> PyResult<Self> {
        if history_length < 1 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "history_length must be >= 1",
            ));
        }
        if (history_length as u128 * 14 + 7) * 64 > i32::MAX as u128 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "history_length {history_length} makes the observation tensor exceed 2^31 elements"
            )));
        }
        Ok(EncoderHandle {
            chess: ChessEncoderSpec::AlphaZero {
                history: history_length,
            },
        })
    }
}

/// Chess interop: the action id of the legal move whose STANDARD-UCI string is `uci` in the
/// position `fen` (castling as "e1g1"/"e1c1"). ValueError on a bad FEN or a string matching no
/// legal move. Pure — for referees/tools translating between engines' move languages.
#[pyfunction]
fn chess_uci_action(uci: &str, fen: &str) -> PyResult<usize> {
    let board: reinfors_games::ChessBoard = fen
        .parse()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err(format!("invalid FEN: {fen:?}")))?;
    reinfors_games::chess_uci_to_action(uci, &board).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(format!("{uci:?} is not a legal move in {fen:?}"))
    })
}

/// Inverse of `chess_uci_action`: the standard-UCI string of `action` in `fen`'s position.
#[pyfunction]
fn chess_action_uci(action: usize, fen: &str) -> PyResult<String> {
    let board: reinfors_games::ChessBoard = fen
        .parse()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err(format!("invalid FEN: {fen:?}")))?;
    let mv = reinfors_games::chess_decode_move(action, &board).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "action {action} does not decode in {fen:?}"
        ))
    })?;
    if !board.is_legal(mv) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "action {action} decodes to {} which is not legal in {fen:?}",
            reinfors_games::chess_move_to_uci(mv, &board)
        )));
    }
    Ok(reinfors_games::chess_move_to_uci(mv, &board))
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

#[pymodule]
fn _reinfors(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(core_version, m)?)?;
    m.add_class::<PyEnvSnapshot>()?;
    m.add_class::<PyEngineSnapshot>()?;
    m.add_function(wrap_pyfunction!(chess_uci_action, m)?)?;
    m.add_function(wrap_pyfunction!(chess_action_uci, m)?)?;
    m.add_function(wrap_pyfunction!(core_build_profile, m)?)?;
    m.add_class::<PyEngine>()?;
    m.add_class::<PyEnv>()?;
    m.add_class::<GameHandle>()?;
    m.add_class::<PolicyHandle>()?;
    m.add_class::<LearnerHandle>()?;
    m.add_class::<PyReward>()?;
    m.add_class::<TreeStrapBatch>()?;
    m.add_class::<AlphaZeroBatch>()?;
    m.add_class::<CollectStream>()?;
    m.add_class::<EncoderHandle>()?;
    m.add_class::<ChanceModeHandle>()?;
    m.add_class::<NoiseHandle>()?;
    m.add_class::<DqnBatch>()?;
    m.add_class::<PyBox>()?;
    m.add_class::<PyDiscrete>()?;
    Ok(())
}
