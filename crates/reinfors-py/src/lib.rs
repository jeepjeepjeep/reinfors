//! PyO3 bindings for the `reinfors._reinfors` extension module.

use std::collections::HashMap;

use numpy::ndarray::{Array2, Array3, ArrayD, IxDyn};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3, PyArrayDyn};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use serde_json::{json, Value};

use reinfors_core::{
    ActBy, AlphaZero, AlphaZeroConfig, AlphaZeroLearner, AlphaZeroRecord, AlwaysInitialState,
    ChanceMode, Dqn, DqnRecord, Engine, EngineParams, Env, EpsilonGreedyQ, Evaluator, Game,
    InferCache, InferMode, Learner, Mcts, MctsConfig, Minimax, NoiseScope, Opponent, Policy,
    ReachedStateBuffer, Reward, SearchConfig, SelectiveExpectimax, Space, SplitMix64,
    StartDistribution, StateCodec, StateEncoder, TreeStrap, TreeStrapRecord,
};
use reinfors_games::snake::{Cell, DeathCause};
use reinfors_games::{
    snake_length_cell, Action, Backgammon, BackgammonEvent, BackgammonReward, BackgammonState,
    BackgammonTesauro, Chess, ChessEvent, ChessPlanesAz119, ChessPlanesMinimal,
    ChessPlanesOpenSpiel, ChessPlanesRelative, ChessReward, ChessState, Connect4, Connect4Event,
    Connect4Planes, Connect4Reward, Connect4State, EgocentricSnake, GridEvent, GridState,
    GridWorld, GridWorldPlanes, GridWorldReward, HoldemEgocentric, HoldemReward, KuhnEncoder,
    KuhnPoker, LeducEncoder, LeducPoker, Snake, SnakeReward, SnakeState, StepEvent, TexasHoldem,
    CHESS_ACTIONS,
};

fn action_to_u8(a: Action) -> u8 {
    match a {
        Action::Up => 0,
        Action::Down => 1,
        Action::Left => 2,
        Action::Right => 3,
    }
}

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

fn infer_array<'py, const N: usize>(
    r: &Bound<'py, PyAny>,
    what: &str,
) -> PyResult<InferArray<'py, N>> {
    use numpy::prelude::{PyArrayDescrMethods, PyUntypedArrayMethods};
    let Ok(u) = r.downcast::<numpy::PyUntypedArray>() else {
        return Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "{what} must be a float64 or float32 ndarray; got type {}",
            r.get_type()
                .name()
                .map(|n| n.to_string())
                .unwrap_or_default()
        )));
    };
    if u.ndim() != N {
        return Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "{what} must be a {N}-d ndarray; got {}-d",
            u.ndim()
        )));
    }
    let py = r.py();
    if u.dtype().is_equiv_to(&numpy::dtype::<f64>(py)) {
        Ok(InferArray::F64(r.extract()?))
    } else if u.dtype().is_equiv_to(&numpy::dtype::<f32>(py)) {
        Ok(InferArray::F32(r.extract()?))
    } else {
        Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "{what} must be a float64 or float32 ndarray; got dtype {}",
            u.dtype()
        )))
    }
}

enum InferArray<'py, const N: usize> {
    F64(numpy::PyReadonlyArrayDyn<'py, f64>),
    F32(numpy::PyReadonlyArrayDyn<'py, f32>),
}

impl<const N: usize> InferArray<'_, N> {
    fn shape(&self) -> [usize; N] {
        let s = match self {
            InferArray::F64(a) => a.as_array().shape().to_vec(),
            InferArray::F32(a) => a.as_array().shape().to_vec(),
        };
        let mut out = [0usize; N];
        out.copy_from_slice(&s);
        out
    }

    fn widen_into(&self, out: &mut Vec<f64>) {
        fn go<T: Copy>(v: &numpy::ndarray::ArrayViewD<'_, T>, out: &mut Vec<f64>)
        where
            f64: From<T>,
        {
            match v.as_slice() {
                Some(s) => out.extend(s.iter().map(|&x| f64::from(x))),
                // Sliced network outputs may be strided; preserve row-major logical iteration.
                None => out.extend(v.iter().map(|&x| f64::from(x))),
            }
        }
        match self {
            InferArray::F64(a) => go(&a.as_array(), out),
            InferArray::F32(a) => go(&a.as_array(), out),
        }
    }

    fn to_flat(&self) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.shape().iter().product());
        self.widen_into(&mut out);
        out
    }

    fn pack_policy_value(&self, values: &[f64], a: usize) -> Vec<f64> {
        // Padded policy heads are accepted; only the first `a` logits belong to the game.
        fn go<T: Copy>(v: &numpy::ndarray::ArrayViewD<'_, T>, values: &[f64], a: usize) -> Vec<f64>
        where
            f64: From<T>,
        {
            let width = v.shape()[1];
            let mut out = Vec::with_capacity(values.len() * (a + 1));
            match v.as_slice() {
                Some(s) => {
                    for (row, chunk) in s.chunks_exact(width).enumerate() {
                        out.extend(chunk[..a].iter().map(|&x| f64::from(x)));
                        out.push(values[row]);
                    }
                }
                None => {
                    for (row, r) in v.rows().into_iter().enumerate() {
                        out.extend(r.iter().take(a).map(|&x| f64::from(x)));
                        out.push(values[row]);
                    }
                }
            }
            out
        }
        match self {
            InferArray::F64(arr) => go(&arr.as_array(), values, a),
            InferArray::F32(arr) => go(&arr.as_array(), values, a),
        }
    }
}

fn infer_rows_1d(r: &Bound<'_, PyAny>, what: &str) -> PyResult<Vec<f64>> {
    Ok(infer_array::<1>(r, what)?.to_flat())
}

fn infer_rows_2d(r: &Bound<'_, PyAny>, what: &str) -> PyResult<([usize; 2], Vec<f64>)> {
    let a = infer_array::<2>(r, what)?;
    Ok((a.shape(), a.to_flat()))
}

fn infer_rows_3d(r: &Bound<'_, PyAny>, what: &str) -> PyResult<([usize; 3], Vec<f64>)> {
    let a = infer_array::<3>(r, what)?;
    Ok((a.shape(), a.to_flat()))
}

fn infer_closure<'a, 'py>(
    py: Python<'py>,
    callbacks: &'a [Py<PyAny>],
    dim: usize,
    action_count: usize,
    expected_heads: Option<usize>,
    callback_err: &'a mut Option<PyErr>,
) -> impl FnMut(usize, Vec<f32>, usize) -> Vec<f64> + 'a
where
    'py: 'a,
{
    move |player: usize, obs_flat: Vec<f32>, n: usize| -> Vec<f64> {
        // Preserve K while unwinding after an error; mixing head counts corrupts trajectories.
        let fallback = n * expected_heads.unwrap_or(1) * action_count;
        if callback_err.is_some() {
            return vec![0.0; fallback];
        }
        // Ownership moves into NumPy; the observation batch is not copied at the boundary.
        let arr = Array2::from_shape_vec((n, dim), obs_flat)
            .expect("obs batch shape")
            .into_pyarray(py);
        let callback = if callbacks.len() == 1 {
            &callbacks[0]
        } else {
            &callbacks[player]
        };
        let infer = callback.bind(py);
        match infer
            .call1((arr,))
            .and_then(|r| infer_rows_3d(&r, "infer output"))
        {
            Ok((shape, flat)) => {
                // Validate binding-only shape/head contracts only after extraction succeeds, so a
                // genuine Python callback error remains the error reported to the caller.
                // Element counts alone would admit transposed arrays and scramble evaluations.
                let bad = shape[0] != n
                    || shape[2] != action_count
                    || expected_heads.is_some_and(|k| shape[1] != k);
                if n > 0 && bad {
                    callback_err.get_or_insert_with(|| {
                        pyo3::exceptions::PyValueError::new_err(format!(
                            "infer returned shape {shape:?} for {n} rows; expected ({n}, \
                             n_heads{}, {action_count})",
                            expected_heads.map_or(String::new(), |k| format!(" = {k}"))
                        ))
                    });
                    return vec![0.0; fallback];
                }
                if flat.iter().any(|value| !value.is_finite()) {
                    callback_err.get_or_insert_with(|| {
                        pyo3::exceptions::PyValueError::new_err(
                            "infer outputs must contain only finite values",
                        )
                    });
                    return vec![0.0; fallback];
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

// `labels.label` names the composed policy in contract errors; the troubleshooting table
// matches these strings, so existing labels must stay byte-stable.
struct PvLabels {
    label: &'static str,
    logits: String,
    values: String,
}

impl PvLabels {
    fn new(label: &'static str) -> Self {
        PvLabels {
            label,
            logits: format!("{label} infer policy_logits"),
            values: format!("{label} infer values"),
        }
    }
}

fn policy_value_infer_closure<'a, 'py>(
    py: Python<'py>,
    callbacks: &'a [Py<PyAny>],
    dim: usize,
    action_count: usize,
    labels: &'a PvLabels,
    callback_err: &'a mut Option<PyErr>,
) -> impl FnMut(usize, Vec<f32>, usize) -> Vec<f64> + 'a
where
    'py: 'a,
{
    let label = labels.label;
    move |player: usize, obs_flat: Vec<f32>, n: usize| -> Vec<f64> {
        let stride = action_count + 1;
        if callback_err.is_some() {
            return vec![0.0; n * stride];
        }
        // Ownership moves into NumPy; the observation batch is not copied at the boundary.
        let arr = Array2::from_shape_vec((n, dim), obs_flat)
            .expect("obs batch shape")
            .into_pyarray(py);
        let callback = if callbacks.len() == 1 {
            &callbacks[0]
        } else {
            &callbacks[player]
        };
        let infer = callback.bind(py);
        let extracted = infer.call1((arr,)).and_then(|r| {
            let (logits, values) = r
                .extract::<(Bound<'_, PyAny>, Bound<'_, PyAny>)>()
                .map_err(|_| {
                    let got = match r.downcast::<pyo3::types::PyTuple>() {
                        Ok(t) => format!("a {}-element tuple", t.len()),
                        Err(_) => r
                            .get_type()
                            .name()
                            .map(|n| n.to_string())
                            .unwrap_or_else(|_| "an unknown type".to_string()),
                    };
                    pyo3::exceptions::PyTypeError::new_err(format!(
                        "{label} infer must return a (policy_logits, values) tuple; got {got}"
                    ))
                })?;
            let logits = infer_array::<2>(&logits, &labels.logits)?;
            let values = infer_rows_1d(&values, &labels.values)?;
            Ok((logits, values))
        });
        // Callback/extraction failures take precedence; binding-owned shape checks run only after a
        // successful return and therefore cannot mask the original Python error.
        match extracted {
            Ok((logits, values)) => {
                let lshape = logits.shape();
                if lshape[0] != n || lshape[1] < action_count || values.len() != n {
                    callback_err.get_or_insert_with(|| {
                        pyo3::exceptions::PyValueError::new_err(format!(
                            "{label} infer must return (policy_logits ({n}, >={action_count}), \
                             values ({n},)); got logits {lshape:?} and values ({},)",
                            values.len()
                        ))
                    });
                    return vec![0.0; n * stride];
                }
                let packed = logits.pack_policy_value(&values, action_count);
                if packed.iter().any(|value| !value.is_finite()) {
                    callback_err.get_or_insert_with(|| {
                        pyo3::exceptions::PyValueError::new_err(format!(
                            "{label} infer outputs must contain only finite values"
                        ))
                    });
                    return vec![0.0; n * stride];
                }
                packed
            }
            Err(e) => {
                *callback_err = Some(e);
                vec![0.0; n * stride]
            }
        }
    }
}

fn infer_closure_gil<C: AsRef<[Py<PyAny>]> + Send>(
    callbacks: C,
    dim: usize,
    action_count: usize,
    expected_heads: usize,
    layout: InferLayout,
    stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
    callback_err: std::sync::Arc<std::sync::Mutex<Option<PyErr>>>,
) -> impl FnMut(usize, Vec<f32>, usize) -> Vec<f64> + Send {
    let pv_labels = match layout {
        InferLayout::PolicyValue { label } => Some(PvLabels::new(label)),
        InferLayout::ValueHeads => None,
    };
    move |player: usize, obs_flat: Vec<f32>, n: usize| -> Vec<f64> {
        let fallback_len = match layout {
            InferLayout::ValueHeads => n * expected_heads * action_count,
            InferLayout::PolicyValue { .. } => n * (action_count + 1),
        };
        if stop.load(std::sync::atomic::Ordering::Relaxed) || callback_err.lock().unwrap().is_some()
        {
            return vec![0.0; fallback_len];
        }
        // A stopping worker must be able to unwind without reacquiring Python's GIL.
        Python::with_gil(|py| {
            let mut err = callback_err.lock().unwrap().take();
            let out = {
                match layout {
                    InferLayout::ValueHeads => {
                        let mut f = infer_closure(
                            py,
                            callbacks.as_ref(),
                            dim,
                            action_count,
                            Some(expected_heads),
                            &mut err,
                        );
                        f(player, obs_flat, n)
                    }
                    InferLayout::PolicyValue { .. } => {
                        let mut f = policy_value_infer_closure(
                            py,
                            callbacks.as_ref(),
                            dim,
                            action_count,
                            pv_labels.as_ref().expect("labels exist for PolicyValue"),
                            &mut err,
                        );
                        f(player, obs_flat, n)
                    }
                }
            };
            *callback_err.lock().unwrap() = err;
            out
        })
    }
}

// The worker cannot marshal Python objects safely; next() runs this under the GIL.
type BatchThunk = Box<dyn FnOnce(Python<'_>) -> PyResult<PyObject> + Send>;

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
                // Preserve seeds above i64::MAX as exact Python integers.
                u.into_bound_py_any(py)?
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

fn game_cfg(spec: &GameSpec, selected_encoder: EncoderSpec) -> Value {
    let encoder = selected_encoder.cfg();
    match spec {
        GameSpec::Snake {
            num_snakes,
            grid_size,
            initial_length,
            initial_food_count,
            play_to_last,
            win_food_lead,
            max_ticks,
        } => json!({
            "name": "snake",
            "num_snakes": num_snakes,
            "grid_size": grid_size,
            "initial_length": initial_length,
            "food": initial_food_count,
            "play_to_last": play_to_last,
            "win_food_lead": win_food_lead,
            "max_ticks": max_ticks,
            "encoder": encoder,
        }),
        GameSpec::Connect4 => json!({"name": "connect4", "encoder": encoder}),
        GameSpec::TexasHoldem {
            num_players,
            stack,
            small_blind,
            big_blind,
        } => json!({
            "name": "texas_holdem",
            "num_players": num_players,
            "stack": stack,
            "small_blind": small_blind,
            "big_blind": big_blind,
            "encoder": encoder,
        }),
        GameSpec::KuhnPoker { players } => {
            json!({"name": "kuhn_poker", "players": players, "encoder": encoder})
        }
        GameSpec::LeducPoker => json!({"name": "leduc_poker", "encoder": encoder}),
        GameSpec::Chess { max_ticks, .. } => {
            json!({"name": "chess", "max_ticks": max_ticks, "encoder": encoder})
        }
        GameSpec::Backgammon { max_ticks } => {
            json!({"name": "backgammon", "max_ticks": max_ticks, "encoder": encoder})
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
            "encoder": encoder,
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
                Opponent::Adversarial { .. } => {
                    unreachable!("SelectiveExpectimax never stores an adversarial opponent")
                }
            }
            v
        }
        PolicySpec::Minimax {
            depth,
            top_k,
            chance,
        } => json!({
            "name": "minimax",
            "depth": depth,
            // None renders null so an omitted beam cannot split fingerprints.
            "top_k": top_k,
            "chance": chance_cfg(chance),
        }),
        PolicySpec::EpsilonGreedyQ { n_heads, epsilon } => {
            json!({"name": "epsilon_greedy_q", "n_heads": n_heads, "epsilon": epsilon})
        }
        PolicySpec::Ppo => json!({"name": "ppo"}),
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
            sequential_backup,
        } => {
            // Epsilon zero and noise=None are behaviorally identical and canonicalize to null.
            let noise = if *noise_epsilon == 0.0 {
                Value::Null
            } else {
                json!({
                    "name": "dirichlet",
                    "epsilon": noise_epsilon,
                    "alpha": noise_alpha,
                    "scope": match noise_scope {
                        NoiseScope::Requester => "requester",
                        NoiseScope::All => "all",
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
                "sequential_backup": match sequential_backup {
                    reinfors_core::SequentialBackup::Auto => "auto",
                    reinfors_core::SequentialBackup::MaxN => "maxn",
                },
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
        LearnerSpec::Dqn {
            bootstrap_p,
            n_step,
            gamma,
        } => json!({"name": "dqn", "bootstrap_p": bootstrap_p, "n_step": n_step, "gamma": gamma}),
        LearnerSpec::Ppo { gamma, lam } => json!({"name": "ppo", "gamma": gamma, "lam": lam}),
        LearnerSpec::AlphaZero { gamma } => json!({"name": "alphazero", "gamma": gamma}),
    }
}

fn canonical_config_bytes(v: &Value) -> Vec<u8> {
    // Fingerprints depend on serde_json's default BTreeMap ordering and ryu
    // float formatting. Enabling its transitive `preserve_order` feature is a
    // persisted-format change and requires a schema migration.
    serde_json::to_vec(v).expect("config values contain no non-serializable data")
}

fn fingerprint_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

const ENGINE_SNAPSHOT_SCHEMA: u8 = 1;
const ENGINE_SNAPSHOT_MAGIC: &[u8; 4] = b"RFGS";

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
    #[getter]
    fn fingerprint(&self) -> &str {
        &self.fingerprint
    }

    #[getter]
    fn schema_version(&self) -> u8 {
        self.schema
    }

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

#[pyclass(name = "Engine")]
struct PyEngine {
    inner: Option<Box<dyn ErasedEngine>>,
    config: Value,
    snapshot_fp: String,
    // These stay outside the engine moved to the stream worker, making
    // weights_updated() safe from the consumer thread.
    weights_generations: Vec<std::sync::Arc<std::sync::atomic::AtomicU64>>,
    weights_version: std::sync::Arc<std::sync::atomic::AtomicU64>,
}

#[pymethods]
impl PyEngine {
    #[new]
    #[pyo3(signature = (game, reward, policy, learner, n_games, seed=0, start_buffer=false, start_buffer_capacity=1000, p_fresh=0.05, infer_cache=0, learn_players=None, pad=false, batch_size=0, n_threads=0))]
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
        learn_players: Option<Vec<usize>>,
        pad: bool,
        batch_size: usize,
        n_threads: usize,
    ) -> PyResult<Self> {
        if n_games < 1 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "n_games must be >= 1",
            ));
        }
        // eager per-game allocation: an absurd count must fail here, not OOM-kill the process
        if n_games > 1 << 16 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "n_games must be <= {} (got {n_games})",
                1 << 16
            )));
        }
        if n_threads > 512 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "n_threads must be <= 512 (got {n_threads})",
            )));
        }
        if batch_size > 1 << 20 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "batch_size must be <= {} (got {batch_size})",
                1 << 20
            )));
        }
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
            "game": game_cfg(&game.spec, game.encoder),
            "reward": resolved_reward,
            "policy": policy_cfg(&policy.spec),
            "learner": learner_cfg(&learner.spec),
            "engine": {
                "n_games": n_games,
                "seed": seed,
                // Disabled knobs canonicalize to null so ignored arguments do
                // not split fingerprints of behaviorally identical engines.
                "start_buffer": start_buffer.as_ref().map_or(Value::Null, |sb| json!({
                    "capacity": sb.capacity,
                    "p_fresh": sb.p_fresh,
                })),
                "infer_cache": infer_cache,
                "learn_players": learn_players,
            },
        });
        let engine_params = EngineParams {
            n_games,
            seed,
            pad,
            batch_size: (batch_size > 0).then_some(batch_size),
            n_threads: (n_threads > 0).then_some(n_threads),
        };
        let num_agents = game.spec.num_agents();
        // Slot 0 serves a shared callback; slots 1..=N serve per-player callbacks.
        let weights_generations: Vec<std::sync::Arc<std::sync::atomic::AtomicU64>> = (0
            ..=num_agents)
            .map(|_| std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0)))
            .collect();
        let caches = (infer_cache > 0).then(|| {
            CacheSet::Exclusive(
                weights_generations
                    .iter()
                    .map(|generation| InferCache::new(infer_cache, generation.clone()))
                    .collect(),
            )
        });
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
                caches,
                learn_players,
            )?),
            config,
            weights_generations,
            weights_version: std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0)),
        })
    }

    #[pyo3(signature = (player=None))]
    fn weights_updated(&self, player: Option<usize>) -> PyResult<()> {
        match player {
            None => {
                for generation in &self.weights_generations {
                    generation.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
            }
            Some(p) => {
                // Avoid `p + 1 >= len`: p may be usize::MAX.
                if p >= self.weights_generations.len() - 1 {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "player {p} out of range (this game has {} players)",
                        self.weights_generations.len() - 1
                    )));
                }
                self.weights_generations[p + 1].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        }
        self.weights_version
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    fn resolved_config<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        value_to_py(py, &self.config)
    }

    fn config_fingerprint(&self) -> String {
        fingerprint_hex(&canonical_config_bytes(&self.config))
    }

    #[pyo3(signature = (policy_version=None))]
    fn snapshot(&self, policy_version: Option<String>) -> PyResult<PyEngineSnapshot> {
        let engine = self.inner.as_ref().ok_or_else(stream_active_err)?;
        Ok(PyEngineSnapshot {
            schema: ENGINE_SNAPSHOT_SCHEMA,
            fingerprint: self.snapshot_fp.clone(),
            weights_generation: self
                .weights_version
                .load(std::sync::atomic::Ordering::Relaxed),
            policy_version,
            payload: engine.snapshot_payload()?,
        })
    }

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
        self.weights_version.store(
            snapshot.weights_generation,
            std::sync::atomic::Ordering::Relaxed,
        );
        // Restoring clears caches; slot generations only need to change, not match the snapshot.
        for generation in &self.weights_generations {
            generation.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        Ok(())
    }

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
        // Validate before moving the engine into the worker so bad input cannot forfeit it.
        let (infer, mode) = {
            let borrow = slf.borrow();
            let engine = borrow.inner.as_ref().ok_or_else(stream_active_err)?;
            let num_agents = engine.routing();
            let pair = Python::with_gil(|py| engine_callbacks(infer.bind(py), num_agents))?;
            reject_padded_per_player(engine.pad(), pair.1)?;
            pair
        };
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
                    // Pausing at batch boundaries keeps state aligned with delivered batches.
                    if stop.load(std::sync::atomic::Ordering::Relaxed)
                        || pause.load(std::sync::atomic::Ordering::Relaxed)
                    {
                        break;
                    }
                    let result = engine.collect_thunk(collect_size, &infer, mode, stop.clone());
                    let fatal = result.is_err();
                    if tx.send(result).is_err() {
                        break;
                    }
                    queued.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if fatal {
                        break;
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

#[derive(Clone, Copy)]
enum ChessEncoderSpec {
    Minimal,
    Relative,
    OpenSpiel,
    AlphaZero { history: usize },
}

#[derive(Clone, Copy)]
enum EncoderSpec {
    Snake,
    Connect4,
    Chess(ChessEncoderSpec),
    Backgammon,
    TexasHoldem,
    KuhnPoker,
    LeducPoker,
    GridWorld,
}

impl EncoderSpec {
    fn cfg(self) -> Value {
        match self {
            EncoderSpec::Snake => json!({"name": "snake"}),
            EncoderSpec::Connect4 => json!({"name": "connect4"}),
            EncoderSpec::Chess(ChessEncoderSpec::Minimal) => json!({"name": "minimal_chess"}),
            EncoderSpec::Chess(ChessEncoderSpec::Relative) => json!({"name": "relative_chess"}),
            EncoderSpec::Chess(ChessEncoderSpec::OpenSpiel) => {
                json!({"name": "openspiel_chess"})
            }
            EncoderSpec::Chess(ChessEncoderSpec::AlphaZero { history }) => {
                json!({"name": "alphazero_chess", "history_length": history})
            }
            EncoderSpec::Backgammon => json!({"name": "backgammon"}),
            EncoderSpec::TexasHoldem => json!({"name": "texas_holdem"}),
            EncoderSpec::KuhnPoker => json!({"name": "kuhn_poker"}),
            EncoderSpec::LeducPoker => json!({"name": "leduc_poker"}),
            EncoderSpec::GridWorld => json!({"name": "gridworld"}),
        }
    }

    fn name(self) -> &'static str {
        match self {
            EncoderSpec::Snake => "snake",
            EncoderSpec::Connect4 => "connect4",
            EncoderSpec::Chess(ChessEncoderSpec::Minimal) => "minimal_chess",
            EncoderSpec::Chess(ChessEncoderSpec::Relative) => "relative_chess",
            EncoderSpec::Chess(ChessEncoderSpec::OpenSpiel) => "openspiel_chess",
            EncoderSpec::Chess(ChessEncoderSpec::AlphaZero { .. }) => "alphazero_chess",
            EncoderSpec::Backgammon => "backgammon",
            EncoderSpec::TexasHoldem => "texas_holdem",
            EncoderSpec::KuhnPoker => "kuhn_poker",
            EncoderSpec::LeducPoker => "leduc_poker",
            EncoderSpec::GridWorld => "gridworld",
        }
    }

    fn action_count(self) -> usize {
        match self {
            EncoderSpec::Snake | EncoderSpec::TexasHoldem | EncoderSpec::LeducPoker => 3,
            EncoderSpec::Connect4 => 7,
            EncoderSpec::Chess(_) => CHESS_ACTIONS,
            EncoderSpec::Backgammon => 1352,
            EncoderSpec::KuhnPoker => 2,
            EncoderSpec::GridWorld => 4,
        }
    }
}

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

    fn pending(&self) -> usize {
        self.queued.load(std::sync::atomic::Ordering::Relaxed)
    }

    fn pause<'py>(&mut self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyAny>>> {
        self.pause.store(true, std::sync::atomic::Ordering::Relaxed);
        let rx = self
            .rx
            .take()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("stream already stopped"))?;
        let mut thunks = Vec::new();
        loop {
            // The worker may need the GIL to finish its in-flight callback while we drain.
            let item = py.allow_threads(|| {
                rx.lock()
                    .map_err(|_| ())
                    .and_then(|guard| guard.recv().map_err(|_| ()))
            });
            match item {
                Ok(item) => thunks.push(item),
                Err(()) => break,
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
        // Dropping the receiver releases a worker blocked on a full bounded queue.
        self.rx.take();
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
        // Never join under the GIL: the worker may be waiting to acquire it for inference.
        self.stop.store(true, std::sync::atomic::Ordering::Relaxed);
        self.rx.take();
    }
}

trait ErasedEngine: Send + Sync {
    fn snapshot_payload(&self) -> PyResult<Vec<u8>>;
    fn restore_payload(&mut self, bytes: &[u8]) -> PyResult<()>;
    fn collect<'py>(
        &mut self,
        py: Python<'py>,
        n_records: usize,
        infer: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>>;

    fn collect_thunk(
        &mut self,
        n_records: usize,
        infer: &[Py<PyAny>],
        mode: InferMode,
        stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
    ) -> PyResult<BatchThunk>;

    fn routing(&self) -> usize;

    fn pad(&self) -> bool;
}

trait RecordBatch: Sized {
    fn into_py_batch<'py>(
        records: Vec<Self>,
        py: Python<'py>,
        dim: usize,
        n_heads: usize,
        telemetry: Bound<'py, PyDict>,
    ) -> PyResult<Bound<'py, PyAny>>;
}

#[pyclass]
struct TreeStrapBatch {
    #[pyo3(get)]
    obs: Py<PyArray2<f32>>,
    #[pyo3(get)]
    players: Py<PyArray1<i64>>,
    #[pyo3(get)]
    targets: Py<PyArray3<f64>>,
    #[pyo3(get)]
    masks: Py<PyArray2<f32>>,
    #[pyo3(get)]
    telemetry: Py<PyDict>,
}

#[pyclass]
struct DqnBatch {
    #[pyo3(get)]
    obs: Py<PyArray2<f32>>,
    #[pyo3(get)]
    players: Py<PyArray1<i64>>,
    #[pyo3(get)]
    actions: Py<PyArray1<i64>>,
    #[pyo3(get)]
    rewards: Py<PyArray1<f64>>,
    #[pyo3(get)]
    next_obs: Py<PyArray2<f32>>,
    #[pyo3(get)]
    dones: Py<PyArray1<bool>>,
    #[pyo3(get)]
    can_bootstrap: Py<PyArray1<bool>>,
    // gamma^k for each record's own-decision window; 0 where it cannot bootstrap. The single
    // discount source: TD targets are `rewards + discounts * next_value`, no caller-side gamma.
    #[pyo3(get)]
    discounts: Py<PyArray1<f64>>,
    #[pyo3(get)]
    masks: Py<PyArray2<f32>>,
    // Legal actions use CSR; an empty next-state slice means the row cannot bootstrap.
    #[pyo3(get)]
    legal_ids: Py<PyArray1<i64>>,
    #[pyo3(get)]
    legal_offsets: Py<PyArray1<i64>>,
    #[pyo3(get)]
    next_legal_ids: Py<PyArray1<i64>>,
    #[pyo3(get)]
    next_legal_offsets: Py<PyArray1<i64>>,
    #[pyo3(get)]
    telemetry: Py<PyDict>,
}

#[pyclass]
struct PpoBatch {
    #[pyo3(get)]
    obs: Py<PyArray2<f32>>,
    #[pyo3(get)]
    players: Py<PyArray1<i64>>,
    #[pyo3(get)]
    actions: Py<PyArray1<i64>>,
    #[pyo3(get)]
    behavior_log_probs: Py<PyArray1<f64>>,
    #[pyo3(get)]
    advantages: Py<PyArray1<f64>>,
    #[pyo3(get)]
    returns: Py<PyArray1<f64>>,
    #[pyo3(get)]
    values: Py<PyArray1<f64>>,
    #[pyo3(get)]
    legal_ids: Py<PyArray1<i64>>,
    #[pyo3(get)]
    legal_offsets: Py<PyArray1<i64>>,
    #[pyo3(get)]
    telemetry: Py<PyDict>,
}

#[pymethods]
impl TreeStrapBatch {
    fn __len__(&self) -> usize {
        4
    }
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

#[pyclass]
struct AlphaZeroBatch {
    #[pyo3(get)]
    obs: Py<PyArray2<f32>>,
    #[pyo3(get)]
    players: Py<PyArray1<i64>>,
    #[pyo3(get)]
    policy_targets: Py<PyArray2<f64>>,
    #[pyo3(get)]
    value_targets: Py<PyArray1<f64>>,
    #[pyo3(get)]
    policy_weights: Py<PyArray1<f64>>,
    // Legal IDs are in head frame; value-only rows have empty slices.
    #[pyo3(get)]
    legal_ids: Py<PyArray1<i64>>,
    #[pyo3(get)]
    legal_offsets: Py<PyArray1<i64>>,
    #[pyo3(get)]
    telemetry: Py<PyDict>,
}

#[pymethods]
impl AlphaZeroBatch {
    fn __len__(&self) -> usize {
        5
    }
    fn __getitem__<'py>(&self, py: Python<'py>, i: usize) -> PyResult<Bound<'py, PyAny>> {
        Ok(match i {
            0 => self.obs.bind(py).clone().into_any(),
            1 => self.policy_targets.bind(py).clone().into_any(),
            2 => self.value_targets.bind(py).clone().into_any(),
            3 => self.policy_weights.bind(py).clone().into_any(),
            4 => self.telemetry.bind(py).clone().into_any(),
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
        let mut weights: Vec<f64> = Vec::with_capacity(m);
        let mut players: Vec<i64> = Vec::with_capacity(m);
        let mut legal_ids: Vec<i64> = Vec::new();
        let mut legal_offsets: Vec<i64> = Vec::with_capacity(m + 1);
        legal_offsets.push(0);
        for (obs, pi, zi, w, player, legal) in records {
            obs_flat.extend(obs);
            pi_flat.extend(pi);
            z.push(zi);
            weights.push(w);
            players.push(player as i64);
            legal_ids.extend(legal.iter().map(|&x| x as i64));
            legal_offsets.push(legal_ids.len() as i64);
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
                players: players.into_pyarray(py).unbind(),
                policy_targets: pi_arr.unbind(),
                value_targets: z.into_pyarray(py).unbind(),
                policy_weights: weights.into_pyarray(py).unbind(),
                legal_ids: legal_ids.into_pyarray(py).unbind(),
                legal_offsets: legal_offsets.into_pyarray(py).unbind(),
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
        let mut players: Vec<i64> = Vec::with_capacity(m);
        for (obs, tgt, mask, player) in records {
            obs_flat.extend(obs);
            tgt_flat.extend(tgt.into_iter().flatten());
            mask_flat.extend(mask);
            players.push(player as i64);
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
                players: players.into_pyarray(py).unbind(),
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
        let mut players: Vec<i64> = Vec::with_capacity(m);
        let mut rewards: Vec<f64> = Vec::with_capacity(m);
        let mut dones: Vec<bool> = Vec::with_capacity(m);
        let mut can_bootstrap: Vec<bool> = Vec::with_capacity(m);
        let mut discounts: Vec<f64> = Vec::with_capacity(m);
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
            can_bootstrap.push(!t.next_legal.is_empty());
            actions.push(t.action as i64);
            players.push(t.player as i64);
            rewards.push(t.reward);
            dones.push(t.terminal);
            discounts.push(t.discount);
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
                players: players.into_pyarray(py).unbind(),
                actions: actions.into_pyarray(py).unbind(),
                rewards: rewards.into_pyarray(py).unbind(),
                next_obs: next_arr.unbind(),
                dones: dones.into_pyarray(py).unbind(),
                can_bootstrap: can_bootstrap.into_pyarray(py).unbind(),
                discounts: discounts.into_pyarray(py).unbind(),
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

impl RecordBatch for reinfors_core::PpoRecord {
    fn into_py_batch<'py>(
        records: Vec<Self>,
        py: Python<'py>,
        dim: usize,
        _n_heads: usize,
        telemetry: Bound<'py, PyDict>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let m = records.len();
        let mut obs_flat: Vec<f32> = Vec::with_capacity(m * dim);
        let mut players: Vec<i64> = Vec::with_capacity(m);
        let mut actions: Vec<i64> = Vec::with_capacity(m);
        let mut behavior_log_probs: Vec<f64> = Vec::with_capacity(m);
        let mut advantages: Vec<f64> = Vec::with_capacity(m);
        let mut returns: Vec<f64> = Vec::with_capacity(m);
        let mut values: Vec<f64> = Vec::with_capacity(m);
        let mut legal_ids: Vec<i64> = Vec::new();
        let mut legal_offsets: Vec<i64> = Vec::with_capacity(m + 1);
        legal_offsets.push(0);
        for r in records {
            obs_flat.extend(r.obs);
            players.push(r.player as i64);
            actions.push(r.action as i64);
            behavior_log_probs.push(r.behavior_log_prob);
            advantages.push(r.advantage);
            returns.push(r.ret);
            values.push(r.value);
            legal_ids.extend(r.legal.iter().map(|&a| a as i64));
            legal_offsets.push(legal_ids.len() as i64);
        }
        let obs_arr = Array2::from_shape_vec((m, dim), obs_flat)
            .expect("obs shape")
            .into_pyarray(py);
        Ok(Bound::new(
            py,
            PpoBatch {
                obs: obs_arr.unbind(),
                players: players.into_pyarray(py).unbind(),
                actions: actions.into_pyarray(py).unbind(),
                behavior_log_probs: behavior_log_probs.into_pyarray(py).unbind(),
                advantages: advantages.into_pyarray(py).unbind(),
                returns: returns.into_pyarray(py).unbind(),
                values: values.into_pyarray(py).unbind(),
                legal_ids: legal_ids.into_pyarray(py).unbind(),
                legal_offsets: legal_offsets.into_pyarray(py).unbind(),
                telemetry: telemetry.unbind(),
            },
        )?
        .into_any())
    }
}

#[derive(Clone, Copy)]
enum InferLayout {
    ValueHeads,
    PolicyValue { label: &'static str },
}

#[allow(clippy::too_many_arguments)]
fn engine_callbacks(
    infer: &Bound<'_, PyAny>,
    num_agents: usize,
) -> PyResult<(Vec<Py<PyAny>>, InferMode)> {
    if infer.is_callable() {
        return Ok((vec![infer.clone().unbind()], InferMode::Shared));
    }
    let callbacks: Vec<Py<PyAny>> = infer.extract().map_err(|_| {
        pyo3::exceptions::PyTypeError::new_err(
            "infer must be a callable or a sequence of per-player callables",
        )
    })?;
    if callbacks.len() != num_agents {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "expected {num_agents} per-player infer callables (one per player), got {}",
            callbacks.len()
        )));
    }
    for (player, cb) in callbacks.iter().enumerate() {
        if !cb.bind(infer.py()).is_callable() {
            return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                "per-player infer element {player} is not callable"
            )));
        }
    }
    Ok((callbacks, InferMode::PerPlayer))
}

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
    num_agents: usize,
) -> PyResult<(Vec<L::Record>, Bound<'py, PyDict>)>
where
    G: Game + Sync,
    G::State: Send,
    P: Policy + Sync,
    L: Learner<P::Evaluation>,
{
    let (callbacks, mode) = engine_callbacks(infer, num_agents)?;
    let mut callback_err: Option<PyErr> = None;
    let (records, stats) = match layout {
        InferLayout::ValueHeads => {
            let mut infer_fn = infer_closure(
                py,
                &callbacks,
                dim,
                action_count,
                Some(n_heads),
                &mut callback_err,
            );
            inner.collect_routed(n_records, mode, &mut infer_fn)
        }
        InferLayout::PolicyValue { label } => {
            let labels = PvLabels::new(label);
            let mut infer_fn = policy_value_infer_closure(
                py,
                &callbacks,
                dim,
                action_count,
                &labels,
                &mut callback_err,
            );
            inner.collect_routed(n_records, mode, &mut infer_fn)
        }
    };
    if let Some(e) = callback_err {
        return Err(e);
    }
    let telemetry = build_telemetry(py, &stats)?;
    Ok((records, telemetry))
}

fn reject_padded_per_player(pad: bool, mode: InferMode) -> PyResult<()> {
    if pad && matches!(mode, InferMode::PerPlayer) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "pad supports a single shared infer callback",
        ));
    }
    Ok(())
}

fn build_telemetry<'py>(
    py: Python<'py>,
    stats: &reinfors_core::CollectStats,
) -> PyResult<Bound<'py, PyDict>> {
    let d = (stats.decisions.max(1)) as f64;
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
    telemetry.set_item("infer_seconds", stats.infer_seconds)?;
    telemetry.set_item("infer_calls", stats.infer_calls)?;
    telemetry.set_item("infer_rows", stats.infer_rows)?;
    telemetry.set_item("padded_rows", stats.padded_rows)?;
    telemetry.set_item("cache_lookups", stats.cache_lookups)?;
    telemetry.set_item("cache_hits", stats.cache_hits)?;
    // Exact Mcts/AlphaZero tree sim-fate identity: decisions*sims = fresh + hit + shared + terminal
    // + depthcap - extra_eval_rows; the subtraction removes auxiliary perspective/fan rows.
    telemetry.set_item("terminal_sims", stats.sum_terminal_sims)?;
    telemetry.set_item("depthcap_sims", stats.sum_depthcap_sims)?;
    telemetry.set_item("requested_rows", stats.sum_requested_rows)?;
    telemetry.set_item("extra_eval_rows", stats.sum_extra_eval_rows)?;
    Ok(telemetry)
}

struct EngineImpl<G: Game + Sync, P: Policy, L: Learner<P::Evaluation>> {
    inner: Engine<G, P, L>,
    codec: Option<Box<dyn StateCodec<State = G::State>>>,
    dim: usize,
    action_count: usize,
    n_heads: usize,
    layout: InferLayout,
    num_agents: usize,
    pad: bool,
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
        infer: &[Py<PyAny>],
        mode: InferMode,
        stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
    ) -> PyResult<BatchThunk> {
        reject_padded_per_player(self.pad, mode)?;
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
        let (records, stats) = self.inner.collect_routed(n_records, mode, &mut infer_fn);
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
        if self.pad {
            let (_callbacks, mode) = engine_callbacks(infer, self.num_agents)?;
            reject_padded_per_player(self.pad, mode)?;
        }
        let (records, telemetry) = run_collect(
            &mut self.inner,
            py,
            n_records,
            infer,
            self.dim,
            self.action_count,
            self.n_heads,
            self.layout,
            self.num_agents,
        )?;
        L::Record::into_py_batch(records, py, self.dim, self.n_heads, telemetry)
    }

    fn routing(&self) -> usize {
        self.num_agents
    }

    fn pad(&self) -> bool {
        self.pad
    }
}

enum CacheSet {
    Exclusive(Vec<InferCache>),
}

#[derive(Clone)]
enum GameSpec {
    Snake {
        num_snakes: usize,
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
    TexasHoldem {
        num_players: usize,
        stack: u32,
        small_blind: u32,
        big_blind: u32,
    },
    KuhnPoker {
        players: usize,
    },
    LeducPoker,
    GridWorld {
        size: i32,
        goal: (i32, i32),
        max_ticks: Option<usize>,
    },
}

impl GameSpec {
    fn name(&self) -> &'static str {
        match self {
            GameSpec::Snake { .. } => "snake",
            GameSpec::Connect4 => "connect4",
            GameSpec::Chess { .. } => "chess",
            GameSpec::Backgammon { .. } => "backgammon",
            GameSpec::TexasHoldem { .. } => "texas_holdem",
            GameSpec::KuhnPoker { .. } => "kuhn_poker",
            GameSpec::LeducPoker => "leduc_poker",
            GameSpec::GridWorld { .. } => "gridworld",
        }
    }

    fn default_encoder(&self) -> EncoderSpec {
        match *self {
            GameSpec::Snake { .. } => EncoderSpec::Snake,
            GameSpec::Connect4 => EncoderSpec::Connect4,
            GameSpec::Chess { encoder, .. } => EncoderSpec::Chess(encoder),
            GameSpec::Backgammon { .. } => EncoderSpec::Backgammon,
            GameSpec::TexasHoldem { .. } => EncoderSpec::TexasHoldem,
            GameSpec::KuhnPoker { .. } => EncoderSpec::KuhnPoker,
            GameSpec::LeducPoker => EncoderSpec::LeducPoker,
            GameSpec::GridWorld { .. } => EncoderSpec::GridWorld,
        }
    }

    fn accepts_encoder(&self, encoder: EncoderSpec) -> bool {
        matches!(
            (self, encoder),
            (GameSpec::Snake { .. }, EncoderSpec::Snake)
                | (GameSpec::Connect4, EncoderSpec::Connect4)
                | (GameSpec::Chess { .. }, EncoderSpec::Chess(_))
                | (GameSpec::Backgammon { .. }, EncoderSpec::Backgammon)
                | (GameSpec::TexasHoldem { .. }, EncoderSpec::TexasHoldem)
                | (GameSpec::KuhnPoker { .. }, EncoderSpec::KuhnPoker)
                | (GameSpec::LeducPoker, EncoderSpec::LeducPoker)
                | (GameSpec::GridWorld { .. }, EncoderSpec::GridWorld)
        )
    }

    fn num_agents(&self) -> usize {
        match *self {
            GameSpec::Snake { num_snakes, .. } => num_snakes,
            GameSpec::TexasHoldem { num_players, .. } => num_players,
            GameSpec::KuhnPoker { players } => players,
            GameSpec::Connect4
            | GameSpec::Chess { .. }
            | GameSpec::Backgammon { .. }
            | GameSpec::LeducPoker => 2,
            GameSpec::GridWorld { .. } => 1,
        }
    }

    fn spaces(&self) -> (Space, Space) {
        fn of<G: Game>(game: G, enc: &dyn StateEncoder<State = G::State>) -> (Space, Space) {
            (enc.observation_space(), game.action_space())
        }
        match *self {
            GameSpec::Snake {
                num_snakes,
                grid_size,
                initial_length,
                initial_food_count,
                play_to_last,
                win_food_lead,
                max_ticks,
            } => of(
                Snake {
                    num_snakes,
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
            GameSpec::TexasHoldem {
                num_players,
                stack,
                small_blind,
                big_blind,
            } => of(
                TexasHoldem {
                    num_players,
                    stack,
                    small_blind,
                    big_blind,
                },
                &HoldemEgocentric { num_players, stack },
            ),
            GameSpec::KuhnPoker { players } => of(KuhnPoker { players }, &KuhnEncoder { players }),
            GameSpec::LeducPoker => of(LeducPoker, &LeducEncoder),
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

// Single source for concrete rewards and resolved configs; duplicating defaults
// would let runtime behavior and fingerprints diverge.
fn reward_schema(game: &GameSpec) -> &'static [(&'static str, f64)] {
    match game {
        GameSpec::Snake { .. } => &[
            ("step", 0.0),
            ("food", 0.0),
            ("loss", -1.0),
            ("draw", 0.0),
            ("kill", 0.0),
            ("win", 1.0),
            ("survival", 0.0),
        ],
        GameSpec::Connect4 => &[("win", 1.0), ("loss", -1.0), ("draw", 0.0)],
        GameSpec::Chess { .. } => &[("win", 1.0), ("loss", -1.0), ("draw", 0.0)],
        GameSpec::Backgammon { .. } => &[("win", 1.0), ("gammon", 2.0), ("backgammon", 3.0)],
        GameSpec::GridWorld { .. } => &[("step", 0.0), ("goal", 1.0)],
        GameSpec::TexasHoldem { .. } => &[("scale", 1.0)],
        GameSpec::KuhnPoker { .. } | GameSpec::LeducPoker => &[("scale", 1.0)],
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
        GameSpec::TexasHoldem { .. } | GameSpec::KuhnPoker { .. } | GameSpec::LeducPoker => {
            let r = resolve_reward(reward, reward_schema(game))?;
            RewardBox::Holdem(HoldemReward { scale: r[0] })
        }
    })
}

enum RewardBox {
    Snake(SnakeReward),
    Holdem(HoldemReward),
    Connect4(Connect4Reward),
    Chess(ChessReward),
    Backgammon(BackgammonReward),
    GridWorld(GridWorldReward),
}

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
    Ppo,
    Minimax {
        depth: i32,
        top_k: Option<usize>,
        chance: ChanceMode,
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
        sequential_backup: reinfors_core::SequentialBackup,
    },
}

#[derive(Clone)]
// The factory threads each learner gamma into its paired search config, keeping tree backup and
// episode targets on the same discount convention.
enum LearnerSpec {
    TreeStrap {
        gamma: f64,
        outcome_weight: f64,
        bootstrap_p: f64,
        interior_targets: bool,
    },
    Dqn {
        bootstrap_p: f64,
        n_step: usize,
        gamma: f64,
    },
    Ppo {
        gamma: f64,
        lam: f64,
    },
    AlphaZero {
        gamma: f64,
    },
}

fn check_positive_finite(name: &str, v: f64) -> PyResult<()> {
    // Scale-like parameters such as opp_temperature appear in denominators and must stay positive.
    if !v.is_finite() || v <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "{name} must be finite and > 0"
        )));
    }
    Ok(())
}

fn check_nonneg_finite(name: &str, v: f64) -> PyResult<()> {
    // Exploration strengths and temperatures may be disabled with zero, but never be negative.
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

// Construction probes use an isolated RNG so they cannot perturb collection determinism.
struct ProbeRng(u64);
impl reinfors_core::Rng for ProbeRng {
    fn below(&mut self, n: usize) -> usize {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as usize % n.max(1)
    }
    fn unit(&mut self) -> f64 {
        self.below(1 << 20) as f64 / (1 << 20) as f64
    }
}

fn game_is_sequential<G: Game>(game: &G) -> bool {
    // A declared-deal raw root is Chance and says nothing about turn-taking; probe only after the
    // root chance chain has been realized or poker/backgammon are silently misclassified.
    matches!(
        game.actor(&reinfors_core::game::realize_initial_state(
            game,
            &mut ProbeRng(7)
        )),
        reinfors_core::Actor::Agent(_)
    )
}

#[allow(clippy::too_many_arguments)]
fn expectimax_from_spec(
    beta: f64,
    expansion_budget: usize,
    top_k: usize,
    max_depth: i32,
    chance: ChanceMode,
    opponent: Opponent,
    n_heads: usize,
    epsilon: f64,
    gamma: f64,
) -> PyResult<SelectiveExpectimax> {
    validate_search_params(expansion_budget, top_k, max_depth, beta)?;
    if n_heads < 1 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "n_heads must be >= 1",
        ));
    }
    check_unit("epsilon", epsilon)?;
    let cfg = SearchConfig {
        gamma,
        beta,
        expansion_budget,
        top_k,
        max_depth,
        chance,
        opponent,
    };
    Ok(SelectiveExpectimax::new(cfg, n_heads, epsilon))
}

fn qgreedy_from_spec(n_heads: usize, epsilon: f64) -> PyResult<EpsilonGreedyQ> {
    if n_heads < 1 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "n_heads must be >= 1",
        ));
    }
    check_unit("epsilon", epsilon)?;
    Ok(EpsilonGreedyQ::new(n_heads, epsilon))
}

#[allow(clippy::too_many_arguments)]
fn mcts_from_spec(
    num_simulations: usize,
    uct_c: f64,
    max_depth: i32,
    act_by: ActBy,
    temperature: f64,
    temperature_drop: u32,
    chance: ChanceMode,
    gamma: f64,
) -> PyResult<Mcts> {
    if num_simulations < 1 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "num_simulations must be >= 1",
        ));
    }
    if max_depth < 1 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "max_depth must be >= 1",
        ));
    }
    if uct_c < 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "uct_c must be >= 0",
        ));
    }
    if !(temperature >= 0.0 && temperature.is_finite()) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "temperature must be finite and >= 0",
        ));
    }
    Ok(Mcts::new(
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
    ))
}

#[allow(clippy::too_many_arguments)]
fn alphazero_from_spec(
    num_simulations: usize,
    c_puct: f64,
    max_depth: i32,
    noise_epsilon: f64,
    noise_alpha: f64,
    temperature: f64,
    temperature_drop: u32,
    chance: ChanceMode,
    noise_scope: NoiseScope,
    sequential_backup: reinfors_core::SequentialBackup,
    gamma: f64,
) -> PyResult<AlphaZero> {
    // Simulation one evaluates the root; a visit-policy target needs another.
    if num_simulations < 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "num_simulations must be >= 2",
        ));
    }
    if max_depth < 1 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "max_depth must be >= 1",
        ));
    }
    if c_puct < 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "c_puct must be >= 0",
        ));
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
    Ok(AlphaZero::new(AlphaZeroConfig {
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
        sequential_backup,
    }))
}

fn check_max_agents<P: Policy, G: Game>(policy: &P, label: &str, game: &G) -> PyResult<()> {
    let num_agents = game.num_agents();
    // Reject before probing initial_state: a malformed zero-agent game may panic while realizing it.
    if num_agents == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "num_agents must be > 0",
        ));
    }
    let sequential = game_is_sequential(game);
    if let Some(cap) = policy.max_agents(sequential) {
        if num_agents > cap {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "the {label} policy supports at most {cap} agents for this game's dynamics; \
                 this game has {num_agents}"
            )));
        }
    }
    Ok(())
}

fn check_information<G: Game>(label: &str, game: &G) -> PyResult<()> {
    // Keep this Python-facing capability error ahead of Engine::new's assertion backstop.
    if !game.perfect_information() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "the {label} policy searches the true state and would be clairvoyant on this \
             hidden-information game; see {}",
            reinfors_core::COMPATIBILITY_DOCS
        )));
    }
    Ok(())
}

// No static width bound on purpose: the action vocabulary wildly overestimates realized legal
// branching (chess: 4672 vs ~35), so the search bounds the realized tree instead.
fn check_minimax_composition<G: Game>(game: &G) -> PyResult<()> {
    if game.num_agents() != 2 || !game_is_sequential(game) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Minimax requires a two-player sequential game; see {}",
            reinfors_core::COMPATIBILITY_DOCS
        )));
    }
    Ok(())
}

// The adversarial negation is sound only for antisymmetric rewards. Backgammon's events carry
// the sign themselves (Loss(m) = -magnitude), so any weights stay zero-sum; chess and connect4
// weight win/loss/draw independently and must be constrained.
fn check_minimax_zero_sum(game: &GameSpec, weights: Option<&HashMap<String, f64>>) -> PyResult<()> {
    if !matches!(game, GameSpec::Chess { .. } | GameSpec::Connect4) {
        return Ok(());
    }
    let effective = |key: &str| -> f64 {
        weights
            .and_then(|m| m.get(key).copied())
            .unwrap_or_else(|| {
                reward_schema(game)
                    .iter()
                    .find(|(name, _)| *name == key)
                    .map(|(_, default)| *default)
                    .expect("schema names its outcome keys")
            })
    };
    if effective("win") + effective("loss") != 0.0 || effective("draw") != 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Minimax's zero-sum negation requires an antisymmetric reward: set loss = -win and \
             draw = 0",
        ));
    }
    Ok(())
}

fn minimax_from_spec(depth: i32, top_k: Option<usize>, chance: ChanceMode, gamma: f64) -> Minimax {
    Minimax::new(depth, top_k, chance, gamma)
}

fn check_joint_space<G: Game>(label: &str, game: &G, movers: usize) -> PyResult<()> {
    // `movers` is family-specific: all agents for dense MCTS/AZ tables, but
    // only co-movers for expectimax's per-MAX-edge product. A wrong exponent
    // turns a construction error into a possible mid-collect allocation blow-up.
    if !game_is_sequential(game) {
        let worst = (game.action_count() as u128).saturating_pow(movers as u32);
        if worst > reinfors_core::MAX_JOINT_SLOTS as u128 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "the {label} policy's simultaneous joint fan (action_count ^ {movers} = \
                 {worst}) exceeds the {} - branch bound",
                reinfors_core::MAX_JOINT_SLOTS
            )));
        }
    }
    Ok(())
}

fn check_max_ticks(max_ticks: Option<usize>) -> PyResult<()> {
    if max_ticks == Some(0) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "max_ticks must be >= 1 (or None to never truncate)",
        ));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn build_inner<G: Game + Sync, P: Policy, L: Learner<P::Evaluation>>(
    game: G,
    enc: Box<dyn StateEncoder<State = G::State>>,
    reward: Box<dyn Reward<Event = G::Event>>,
    policy: P,
    learner: L,
    engine_params: EngineParams,
    start_dist: Box<dyn reinfors_core::StartDistribution<G::State>>,
    infer_caches: Option<CacheSet>,
    learn_players: Option<Vec<usize>>,
) -> Engine<G, P, L>
where
    G::State: Send,
{
    let mut e = Engine::new(game, enc, reward, policy, learner, engine_params)
        .with_start_distribution(start_dist);
    match infer_caches {
        Some(CacheSet::Exclusive(c)) => e = e.with_infer_caches(c),
        None => {}
    }
    if let Some(lp) = learn_players {
        e = e.with_learn_players(&lp);
    }
    e
}

fn check_search_budgets(policy: &PolicySpec) -> PyResult<()> {
    const MAX: u64 = 1 << 20;
    // n_heads multiplies per-node search memory (heads x actions per node), so its
    // sane ceiling is far below the budget caps
    const MAX_HEADS: u64 = 4096;
    let mut items: Vec<(&'static str, u64)> = Vec::new();
    match policy {
        PolicySpec::SelectiveExpectimax { n_heads, .. }
        | PolicySpec::EpsilonGreedyQ { n_heads, .. }
            if *n_heads as u64 > MAX_HEADS =>
        {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "n_heads must be <= {MAX_HEADS} (got {n_heads})"
            )));
        }
        _ => {}
    }
    let chance_items = |items: &mut Vec<(&'static str, u64)>, chance: &ChanceMode| {
        if let ChanceMode::Committed { samples } = chance {
            items.push(("chance samples", *samples as u64));
        }
    };
    match policy {
        PolicySpec::Ppo => {}
        PolicySpec::SelectiveExpectimax {
            expansion_budget,
            top_k,
            max_depth,
            chance,
            ..
        } => {
            items.push(("expansion_budget", *expansion_budget as u64));
            items.push(("top_k", *top_k as u64));
            items.push(("max_depth", i64::from(*max_depth).max(0) as u64));
            chance_items(&mut items, chance);
        }
        PolicySpec::Minimax {
            depth,
            top_k,
            chance,
        } => {
            items.push(("depth", i64::from(*depth).max(0) as u64));
            if let Some(k) = top_k {
                items.push(("top_k", *k as u64));
            }
            chance_items(&mut items, chance);
        }
        PolicySpec::Mcts {
            num_simulations,
            max_depth,
            chance,
            ..
        }
        | PolicySpec::AlphaZero {
            num_simulations,
            max_depth,
            chance,
            ..
        } => {
            items.push(("num_simulations", *num_simulations as u64));
            items.push(("max_depth", i64::from(*max_depth).max(0) as u64));
            chance_items(&mut items, chance);
        }
        PolicySpec::EpsilonGreedyQ { .. } => {}
    }
    for (name, v) in items {
        if v > MAX {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "{name} must be <= {MAX} (got {v})"
            )));
        }
    }
    Ok(())
}

/// Per-call buffers are eagerly allocated (and zero-filled on the fallback path), so every
/// composition point must bound them by BYTES before any collect/choose can run: simultaneous
/// games stage one row per ACTIVE AGENT (`agents` is the conservative multiplier; 1 where the
/// knob already counts exact callback rows), observations stage rows x dim f32, callback
/// outputs rows x heads x (actions+1) f64 (stride upper bound across both infer layouts).
fn check_call_buffers(
    knob: &str,
    count: usize,
    agents: usize,
    dim: usize,
    n_heads: usize,
    action_count: usize,
) -> PyResult<()> {
    const MAX_BUFFER_BYTES: usize = 1 << 29;
    let rows = count.checked_mul(agents.max(1));
    let in_bytes = rows
        .and_then(|r| r.checked_mul(dim))
        .and_then(|t| t.checked_mul(4));
    let out_bytes = rows
        .and_then(|r| r.checked_mul(n_heads.max(1)))
        .and_then(|t| t.checked_mul(action_count + 1))
        .and_then(|t| t.checked_mul(8));
    match (in_bytes, out_bytes) {
        (Some(i), Some(o)) if i <= MAX_BUFFER_BYTES && o <= MAX_BUFFER_BYTES => Ok(()),
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "{knob} ({count}) is too large for this composition: with {agents} \
             simultaneous agents per game, observation input (rows x {dim} f32) and \
             callback output (rows x {n_heads} heads x {action_count} actions f64) \
             must each stay under {} bytes",
            MAX_BUFFER_BYTES
        ))),
    }
}

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
    infer_caches: Option<CacheSet>,
    learn_players: Option<Vec<usize>>,
) -> PyResult<Box<dyn ErasedEngine>>
where
    G::State: Send + Sync,
{
    // Handles intentionally store unchecked params; composition validates them here.
    // Capability gates below run before Engine::new so unsupported compositions raise ValueError
    // instead of surfacing the core's invariant assertions as panic exceptions.
    let (c, h, w) = enc.obs_shape();
    let dim = c * h * w;
    let action_count = game.action_count();
    let num_agents = game.num_agents();
    let n_heads = match &policy {
        PolicySpec::SelectiveExpectimax { n_heads, .. }
        | PolicySpec::EpsilonGreedyQ { n_heads, .. } => *n_heads,
        _ => 1,
    };
    check_search_budgets(&policy)?;
    check_call_buffers(
        "n_games",
        engine_params.n_games,
        num_agents,
        dim,
        n_heads,
        action_count,
    )?;
    if engine_params.pad {
        // pad fixes every call at exactly batch_size rows: no agent multiplier
        let rows = engine_params
            .batch_size
            .unwrap_or_else(|| (engine_params.n_games / 2).max(1));
        check_call_buffers("batch_size", rows, 1, dim, n_heads, action_count)?;
    }
    if let Some(lp) = &learn_players {
        if lp.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "learn_players must name at least one player",
            ));
        }
        if let Some(&bad) = lp.iter().find(|&&p| p >= num_agents) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "learn player {bad} out of range (this game has {num_agents} players)"
            )));
        }
    }
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
            check_unit("outcome_weight", outcome_weight)?;
            check_unit("bootstrap_p", bootstrap_p)?;
            let policy = expectimax_from_spec(
                beta,
                expansion_budget,
                top_k,
                max_depth,
                chance,
                opponent,
                n_heads,
                epsilon,
                gamma,
            )?;
            check_information("SelectiveExpectimax", &game)?;
            check_max_agents(&policy, "SelectiveExpectimax", &game)?;
            check_joint_space(
                "SelectiveExpectimax",
                &game,
                game.num_agents().saturating_sub(1),
            )?;
            let learner = TreeStrap::new(gamma, outcome_weight, bootstrap_p, interior_targets);
            Ok(Box::new(EngineImpl {
                codec: codec.take(),
                pad: engine_params.pad,
                inner: build_inner(
                    game,
                    enc,
                    reward,
                    policy,
                    learner,
                    engine_params,
                    start_dist,
                    infer_caches,
                    learn_players,
                ),
                dim,
                action_count,
                n_heads,
                layout: InferLayout::ValueHeads,
                num_agents,
            }))
        }
        (
            PolicySpec::Minimax {
                depth,
                top_k,
                chance,
            },
            LearnerSpec::TreeStrap {
                gamma,
                outcome_weight,
                bootstrap_p,
                interior_targets,
            },
        ) => {
            check_unit("outcome_weight", outcome_weight)?;
            check_unit("bootstrap_p", bootstrap_p)?;
            let policy = minimax_from_spec(depth, top_k, chance, gamma);
            check_information("Minimax", &game)?;
            check_minimax_composition(&game)?;
            check_max_agents(&policy, "Minimax", &game)?;
            let learner = TreeStrap::new(gamma, outcome_weight, bootstrap_p, interior_targets);
            Ok(Box::new(EngineImpl {
                codec: codec.take(),
                pad: engine_params.pad,
                inner: build_inner(
                    game,
                    enc,
                    reward,
                    policy,
                    learner,
                    engine_params,
                    start_dist,
                    infer_caches,
                    learn_players,
                ),
                dim,
                action_count,
                n_heads: 1,
                layout: InferLayout::ValueHeads,
                num_agents,
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
                interior_targets: _, // MCTS produces root records only.
            },
        ) => {
            check_unit("outcome_weight", outcome_weight)?;
            check_unit("bootstrap_p", bootstrap_p)?;
            let policy = mcts_from_spec(
                num_simulations,
                uct_c,
                max_depth,
                act_by,
                temperature,
                temperature_drop,
                chance,
                gamma,
            )?;
            check_information("Mcts", &game)?;
            check_max_agents(&policy, "Mcts", &game)?;
            check_joint_space("Mcts", &game, game.num_agents())?;
            let learner = TreeStrap::new(gamma, outcome_weight, bootstrap_p, false);
            Ok(Box::new(EngineImpl {
                codec: codec.take(),
                pad: engine_params.pad,
                inner: build_inner(
                    game,
                    enc,
                    reward,
                    policy,
                    learner,
                    engine_params,
                    start_dist,
                    infer_caches,
                    learn_players,
                ),
                dim,
                action_count,
                n_heads: 1,
                layout: InferLayout::ValueHeads,
                num_agents,
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
                sequential_backup,
            },
            LearnerSpec::AlphaZero { gamma },
        ) => {
            let policy = alphazero_from_spec(
                num_simulations,
                c_puct,
                max_depth,
                noise_epsilon,
                noise_alpha,
                temperature,
                temperature_drop,
                chance,
                noise_scope,
                sequential_backup,
                gamma,
            )?;
            check_information("AlphaZero", &game)?;
            check_max_agents(&policy, "AlphaZero", &game)?;
            check_joint_space("AlphaZero", &game, game.num_agents())?;
            let learner = AlphaZeroLearner::new(gamma);
            Ok(Box::new(EngineImpl {
                codec: codec.take(),
                pad: engine_params.pad,
                inner: build_inner(
                    game,
                    enc,
                    reward,
                    policy,
                    learner,
                    engine_params,
                    start_dist,
                    infer_caches,
                    learn_players,
                ),
                dim,
                action_count,
                n_heads: 1,
                layout: InferLayout::PolicyValue { label: "AlphaZero" },
                num_agents,
            }))
        }
        (PolicySpec::Ppo, LearnerSpec::Ppo { gamma, lam }) => {
            let policy = reinfors_core::PpoActor::new();
            check_max_agents(&policy, "Ppo", &game)?;
            let learner = reinfors_core::Ppo::new(gamma, lam);
            Ok(Box::new(EngineImpl {
                codec: codec.take(),
                pad: engine_params.pad,
                inner: build_inner(
                    game,
                    enc,
                    reward,
                    policy,
                    learner,
                    engine_params,
                    start_dist,
                    infer_caches,
                    learn_players,
                ),
                dim,
                action_count,
                n_heads: 1,
                layout: InferLayout::PolicyValue { label: "Ppo" },
                num_agents,
            }))
        }
        (
            PolicySpec::EpsilonGreedyQ { n_heads, epsilon },
            LearnerSpec::Dqn {
                bootstrap_p,
                n_step,
                gamma,
            },
        ) => {
            check_unit("bootstrap_p", bootstrap_p)?;
            let policy = qgreedy_from_spec(n_heads, epsilon)?;
            check_max_agents(&policy, "EpsilonGreedyQ", &game)?;
            let learner = Dqn::new(n_heads, bootstrap_p, n_step, gamma);
            Ok(Box::new(EngineImpl {
                codec: codec.take(),
                pad: engine_params.pad,
                inner: build_inner(
                    game,
                    enc,
                    reward,
                    policy,
                    learner,
                    engine_params,
                    start_dist,
                    infer_caches,
                    learn_players,
                ),
                dim,
                action_count,
                n_heads,
                layout: InferLayout::ValueHeads,
                num_agents,
            }))
        }
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "incompatible policy/learner composition; see {}",
            reinfors_core::COMPATIBILITY_DOCS
        ))),
    }
}

struct StartBufferConfig {
    capacity: usize,
    p_fresh: f64,
}

#[allow(clippy::too_many_arguments)]
fn build_engine(
    game: GameSpec,
    reward: Option<PyReward>,
    policy: PolicySpec,
    learner: LearnerSpec,
    engine_params: EngineParams,
    start_buffer: Option<StartBufferConfig>,
    infer_caches: Option<CacheSet>,
    learn_players: Option<Vec<usize>>,
) -> PyResult<Box<dyn ErasedEngine>> {
    if matches!(policy, PolicySpec::Minimax { .. }) {
        check_minimax_zero_sum(&game, reward.as_ref().map(|r| &r.weights))?;
    }
    let reward = build_reward(&game, reward)?;
    if start_buffer.is_some() && !matches!(game, GameSpec::Snake { .. }) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "this game does not support start_buffer; see {}",
            reinfors_core::COMPATIBILITY_DOCS
        )));
    }
    match (game, reward) {
        (
            GameSpec::Snake {
                num_snakes,
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
            let game = Snake {
                num_snakes,
                grid_size,
                initial_length,
                play_to_last,
                win_food_lead,
                initial_food_count,
                max_ticks,
            };
            build_for_game(
                game.clone(),
                Box::new(EgocentricSnake { grid_size }),
                Box::new(reward),
                start_dist,
                Some(Box::new(game)),
                policy,
                learner,
                engine_params,
                infer_caches,
                learn_players,
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
            infer_caches,
            learn_players,
        ),
        (GameSpec::Chess { max_ticks, encoder }, RewardBox::Chess(reward)) => {
            let (game, enc) = chess_parts(max_ticks, encoder);
            build_for_game(
                game.clone(),
                enc,
                Box::new(reward),
                Box::new(AlwaysInitialState),
                Some(Box::new(game)),
                policy,
                learner,
                engine_params,
                infer_caches,
                learn_players,
            )
        }
        (GameSpec::Backgammon { max_ticks }, RewardBox::Backgammon(reward)) => {
            let game = Backgammon { max_ticks };
            build_for_game(
                game.clone(),
                Box::new(BackgammonTesauro),
                Box::new(reward),
                Box::new(AlwaysInitialState),
                Some(Box::new(game)),
                policy,
                learner,
                engine_params,
                infer_caches,
                learn_players,
            )
        }
        (
            GameSpec::TexasHoldem {
                num_players,
                stack,
                small_blind,
                big_blind,
            },
            RewardBox::Holdem(reward),
        ) => {
            let game = TexasHoldem {
                num_players,
                stack,
                small_blind,
                big_blind,
            };
            build_for_game(
                game.clone(),
                Box::new(HoldemEgocentric { num_players, stack }),
                Box::new(reward),
                Box::new(AlwaysInitialState),
                Some(Box::new(game)),
                policy,
                learner,
                engine_params,
                infer_caches,
                learn_players,
            )
        }
        (GameSpec::KuhnPoker { players }, RewardBox::Holdem(reward)) => {
            let game = KuhnPoker { players };
            build_for_game(
                game.clone(),
                Box::new(KuhnEncoder { players }),
                Box::new(reward),
                Box::new(AlwaysInitialState),
                Some(Box::new(game)),
                policy,
                learner,
                engine_params,
                infer_caches,
                learn_players,
            )
        }
        (GameSpec::LeducPoker, RewardBox::Holdem(reward)) => build_for_game(
            LeducPoker,
            Box::new(LeducEncoder),
            Box::new(reward),
            Box::new(AlwaysInitialState),
            Some(Box::new(LeducPoker)),
            policy,
            learner,
            engine_params,
            infer_caches,
            learn_players,
        ),
        (
            GameSpec::GridWorld {
                size,
                goal,
                max_ticks,
            },
            RewardBox::GridWorld(reward),
        ) => {
            let game = GridWorld {
                size,
                goal,
                max_ticks,
            };
            build_for_game(
                game.clone(),
                Box::new(GridWorldPlanes { size, goal }),
                Box::new(reward),
                Box::new(AlwaysInitialState),
                Some(Box::new(game)),
                policy,
                learner,
                engine_params,
                infer_caches,
                learn_players,
            )
        }
        _ => unreachable!("build_reward returns the reward variant matching the game"),
    }
}

// v2 added the episode tick count (temperature-drop plies derive from it).
const ENV_SNAPSHOT_SCHEMA: u8 = 2;
const ENV_SNAPSHOT_MAGIC: &[u8; 4] = b"RFES";

#[pyclass(name = "EnvSnapshot")]
#[derive(Clone)]
struct PyEnvSnapshot {
    schema: u8,
    fingerprint: String,
    state: Vec<u8>,
    rng_state: u64,
    done: bool,
    ticks: u64,
}

#[pymethods]
impl PyEnvSnapshot {
    #[getter]
    fn fingerprint(&self) -> &str {
        &self.fingerprint
    }

    #[getter]
    fn schema_version(&self) -> u8 {
        self.schema
    }

    fn to_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, pyo3::types::PyBytes> {
        let mut out = Vec::with_capacity(self.state.len() + self.fingerprint.len() + 32);
        out.extend_from_slice(ENV_SNAPSHOT_MAGIC);
        out.push(self.schema);
        out.extend_from_slice(&(self.fingerprint.len() as u32).to_le_bytes());
        out.extend_from_slice(self.fingerprint.as_bytes());
        out.extend_from_slice(&self.rng_state.to_le_bytes());
        out.push(u8::from(self.done));
        out.extend_from_slice(&self.ticks.to_le_bytes());
        out.extend_from_slice(&(self.state.len() as u32).to_le_bytes());
        out.extend_from_slice(&self.state);
        pyo3::types::PyBytes::new(py, &out)
    }

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
        let ticks = u64::from_le_bytes(take(data, &mut pos, 8)?.try_into().unwrap());
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
            ticks,
        })
    }
}

#[pyclass(name = "Env")]
struct PyEnv {
    inner: Box<dyn ErasedEnv>,
    game_spec: GameSpec,
    encoder_spec: EncoderSpec,
    reward_weights: Option<HashMap<String, f64>>,
    config: Value,
    fingerprint: String,
}

impl PyEnv {
    // Out-of-range agents can panic index-based games and silently select player 0's perspective in
    // Connect4, so every agent-indexed Python method must pass this boundary check.
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
        let encoder_spec = game.encoder;
        let reward_weights = reward.as_ref().map(|r| r.weights.clone());
        let reward_cfg = match &reward_weights {
            None => Value::Null,
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
            "game": game_cfg(&game_spec, encoder_spec),
            "reward": reward_cfg,
        });
        let fingerprint = fingerprint_hex(&canonical_config_bytes(&config));
        Ok(PyEnv {
            inner: build_env(game.spec, reward, seed)?,
            game_spec,
            encoder_spec,
            reward_weights,
            config,
            fingerprint,
        })
    }

    fn resolved_config<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        value_to_py(py, &self.config)
    }

    fn config_fingerprint(&self) -> String {
        self.fingerprint.clone()
    }

    fn snapshot(&self) -> PyResult<PyEnvSnapshot> {
        let (state, rng_state, done, ticks) = self.inner.snapshot_parts()?;
        Ok(PyEnvSnapshot {
            schema: ENV_SNAPSHOT_SCHEMA,
            fingerprint: self.fingerprint.clone(),
            state,
            rng_state,
            done,
            ticks,
        })
    }

    /// Completed steps this episode (the current decision's ply in sequential games).
    #[getter]
    fn ticks(&self) -> u64 {
        self.inner.ticks()
    }

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
        self.inner.restore_parts(
            &snapshot.state,
            snapshot.rng_state,
            snapshot.done,
            snapshot.ticks,
        )
    }

    #[pyo3(signature = (seed=None))]
    fn fork(&self, seed: Option<u64>) -> PyResult<PyEnv> {
        let reward = self
            .reward_weights
            .clone()
            .map(|weights| PyReward { weights });
        let mut forked = PyEnv {
            inner: build_env(self.game_spec.clone(), reward, 0)?,
            game_spec: self.game_spec.clone(),
            encoder_spec: self.encoder_spec,
            reward_weights: self.reward_weights.clone(),
            config: self.config.clone(),
            fingerprint: self.fingerprint.clone(),
        };
        let (state, rng_state, done, ticks) = self.inner.snapshot_parts()?;
        forked
            .inner
            .restore_parts(&state, seed.unwrap_or(rng_state), done, ticks)?;
        Ok(forked)
    }

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

    fn legal_actions(&self, agent: usize) -> PyResult<Vec<usize>> {
        self.check_agent(agent)?;
        Ok(self.inner.legal_actions(agent))
    }

    fn observe<'py>(&self, py: Python<'py>, agent: usize) -> PyResult<Bound<'py, PyArray3<f32>>> {
        self.check_agent(agent)?;
        Ok(self.inner.observe(py, agent))
    }

    fn observation_space<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.inner.observation_space(py)
    }

    fn state<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.inner.state(py)
    }

    fn information_state_key<'py>(
        &self,
        py: Python<'py>,
        agent: usize,
    ) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
        Ok(pyo3::types::PyBytes::new(
            py,
            &self.inner.information_state_key(agent)?,
        ))
    }

    fn step<'py>(
        &mut self,
        py: Python<'py>,
        actions: HashMap<usize, usize>,
    ) -> PyResult<Vec<Bound<'py, PyAny>>> {
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
        // Reject before core dispatch: an illegal backgammon id can decode a move from an empty
        // source and corrupt or panic the native state transition.
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

    #[getter]
    fn rewards(&self) -> Option<Vec<f64>> {
        self.inner.last_rewards()
    }
}

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
        d.set_item("board", self.board())?;
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

impl NativeState for reinfors_games::HoldemState {
    fn to_py<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item(
            "hole",
            self.hole.iter().map(|h| h.to_vec()).collect::<Vec<_>>(),
        )?;
        d.set_item("board", self.board.clone())?;
        d.set_item(
            "street",
            match self.street {
                reinfors_games::Street::Preflop => "preflop",
                reinfors_games::Street::Flop => "flop",
                reinfors_games::Street::Turn => "turn",
                reinfors_games::Street::River => "river",
                reinfors_games::Street::Done => "done",
            },
        )?;
        d.set_item("button", self.button)?;
        d.set_item("to_act", self.to_act)?;
        d.set_item("stacks", self.stacks.clone())?;
        d.set_item("street_committed", self.street_committed.clone())?;
        d.set_item("total_committed", self.total_committed.clone())?;
        d.set_item("folded", self.folded.clone())?;
        d.set_item("needs_action", self.needs_action.clone())?;
        d.set_item("raises", self.raises)?;
        d.set_item("history", self.history.clone())?;
        d.set_item("done", self.is_done())?;
        Ok(d)
    }
}

impl NativeState for reinfors_games::KuhnState {
    fn to_py<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("cards", self.cards.clone())?;
        d.set_item("history", self.history.clone())?;
        d.set_item("done", self.is_terminal_pub())?;
        Ok(d)
    }
}

impl NativeState for reinfors_games::LeducState {
    fn to_py<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("cards", self.cards.clone())?;
        d.set_item("public", self.public)?;
        d.set_item("round", self.round_pub())?;
        d.set_item("history", self.history.to_vec())?;
        d.set_item("done", self.is_terminal_pub())?;
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
        // survived_to_max_ticks is engine-only truncation metadata; Env never sets it, so exposing
        // the field here would promise a permanently false event value.
        Ok(d.into_any())
    }
}

impl NativeEvent for f64 {
    fn to_py<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        Ok(pyo3::types::PyFloat::new(py, *self).into_any())
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
        d.set_item("margin", margin)?;
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

trait ErasedEnv: Send + Sync {
    fn snapshot_parts(&self) -> PyResult<(Vec<u8>, u64, bool, u64)>;
    fn restore_parts(
        &mut self,
        state: &[u8],
        rng_state: u64,
        done: bool,
        ticks: u64,
    ) -> PyResult<()>;
    fn ticks(&self) -> u64;
    fn reset(&mut self);
    fn done(&self) -> bool;
    fn num_agents(&self) -> usize;
    fn action_count(&self) -> usize;
    fn active_agents(&self) -> Vec<usize>;
    fn legal_actions(&self, agent: usize) -> Vec<usize>;
    fn observe<'py>(&self, py: Python<'py>, agent: usize) -> Bound<'py, PyArray3<f32>>;
    fn observation_space<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>>;
    fn state<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>>;
    fn information_state_key(&self, agent: usize) -> PyResult<Vec<u8>>;
    fn step<'py>(
        &mut self,
        py: Python<'py>,
        actions: Vec<usize>,
    ) -> PyResult<Vec<Bound<'py, PyAny>>>;
    fn last_rewards(&self) -> Option<Vec<f64>>;
    fn as_any(&self) -> &dyn std::any::Any;
    fn obs_dim(&self) -> usize;
    #[allow(clippy::too_many_arguments)]
    fn choose_batch(
        &self,
        peers: &[&dyn ErasedEnv],
        spec: &PolicySpec,
        gamma: f64,
        seed: u64,
        plies: &[u64],
        infer: &mut dyn FnMut(usize, Vec<f32>, usize) -> Vec<f64>,
    ) -> PyResult<Vec<usize>>;
}

// Per-request selection streams derive from (seed, batch index) via the engine's
// keying scheme: reproducible for the same ordered batch and seed, per the choose
// determinism contract.
use reinfors_core::rng::stream as rng_stream;

#[allow(clippy::too_many_arguments)]
fn run_choose<G, P>(
    policy: &P,
    game: &G,
    enc: &dyn StateEncoder<State = G::State>,
    reward: &dyn Reward<Event = G::Event>,
    requests: Vec<(G::State, usize)>,
    seed: u64,
    mut states: Vec<P::PolicyState>,
    infer: &mut dyn FnMut(usize, Vec<f32>, usize) -> Vec<f64>,
) -> PyResult<Vec<usize>>
where
    G: Game + Sync,
    G::State: Send,
    P: Policy + Sync,
{
    let mut infer_fn = |p: usize, o: Vec<f32>, n: usize| infer(p, o, n);
    // Within-batch dedup only: no store, nothing survives a round, stochastic
    // inference is never frozen across rounds.
    let mut evaluator = Evaluator::with_batch_dedup(&mut infer_fn, InferMode::Shared);
    let perms =
        reinfors_core::encoder::PermTable::build(enc, game.action_count(), game.num_agents());
    let decisions: Vec<(G::State, Vec<usize>)> =
        requests.into_iter().map(|(s, a)| (s, vec![a])).collect();
    let mut drive_rng = SplitMix64::keyed(seed, rng_stream::CHOOSE_DRIVE, 0);
    let results = reinfors_core::rollout::driver::drive_to_completion(
        policy,
        game,
        enc,
        reward,
        &perms,
        false,
        &decisions,
        &mut drive_rng,
        &mut evaluator,
    );
    let mut out = Vec::with_capacity(results.len());
    for (i, per) in results.iter().enumerate() {
        let (eval, _) = &per[0];
        let mut rng = SplitMix64::keyed(seed, rng_stream::CHOOSE_SELECT, i as u64);
        out.push(policy.select(eval, &mut states[i], &mut rng));
    }
    Ok(out)
}

struct EnvImpl<G: Game> {
    inner: Env<G>,
    obs_shape: (usize, usize, usize),
    codec: Option<Box<dyn StateCodec<State = G::State>>>,
    reward: Option<Box<dyn Reward<Event = G::Event>>>,
    last_rewards: Option<Vec<f64>>,
}

impl<G> ErasedEnv for EnvImpl<G>
where
    G: Game + Send + Sync + 'static,
    G::State: Send + Sync + NativeState,
    G::Event: NativeEvent,
{
    fn snapshot_parts(&self) -> PyResult<(Vec<u8>, u64, bool, u64)> {
        let codec = self.codec.as_deref().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("this game does not support snapshots")
        })?;
        let (state, rng_state, done, ticks) = self.inner.parts();
        Ok((codec.encode(&state), rng_state, done, ticks as u64))
    }

    fn ticks(&self) -> u64 {
        self.inner.ticks() as u64
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn obs_dim(&self) -> usize {
        let (c, h, w) = self.obs_shape;
        c * h * w
    }

    fn choose_batch(
        &self,
        peers: &[&dyn ErasedEnv],
        spec: &PolicySpec,
        gamma: f64,
        seed: u64,
        plies: &[u64],
        infer: &mut dyn FnMut(usize, Vec<f32>, usize) -> Vec<f64>,
    ) -> PyResult<Vec<usize>> {
        let mut impls: Vec<&EnvImpl<G>> = Vec::with_capacity(peers.len());
        for peer in peers {
            impls.push(peer.as_any().downcast_ref::<EnvImpl<G>>().ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(
                    "envs must share one composition (game, encoder, reward)",
                )
            })?);
        }
        let game = impls[0].inner.game();
        let enc = impls[0].inner.encoder();
        let reward = impls[0].reward.as_deref().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "choose needs terminal values: construct these Envs with a reward",
            )
        })?;
        let mut requests: Vec<(G::State, usize)> = Vec::with_capacity(impls.len());
        for e in &impls {
            let agents = e.inner.active_agents();
            if agents.len() != 1 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "choose supports sequential games: exactly one active agent per env",
                ));
            }
            requests.push((e.inner.state().clone(), agents[0]));
        }
        match spec.clone() {
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
                let policy = expectimax_from_spec(
                    beta,
                    expansion_budget,
                    top_k,
                    max_depth,
                    chance,
                    opponent,
                    n_heads,
                    epsilon,
                    gamma,
                )?;
                check_information("SelectiveExpectimax", game)?;
                check_max_agents(&policy, "SelectiveExpectimax", game)?;
                check_joint_space(
                    "SelectiveExpectimax",
                    game,
                    game.num_agents().saturating_sub(1),
                )?;
                let states: Vec<usize> = (0..impls.len())
                    .map(|i| {
                        policy.begin_episode(&mut SplitMix64::keyed(
                            seed,
                            rng_stream::CHOOSE_DRIVE,
                            i as u64,
                        ))
                    })
                    .collect();
                run_choose(&policy, game, enc, reward, requests, seed, states, infer)
            }
            PolicySpec::Ppo => {
                let policy = reinfors_core::PpoActor::new();
                check_max_agents(&policy, "Ppo", game)?;
                let states = vec![(); impls.len()];
                run_choose(&policy, game, enc, reward, requests, seed, states, infer)
            }
            PolicySpec::EpsilonGreedyQ { n_heads, epsilon } => {
                let policy = qgreedy_from_spec(n_heads, epsilon)?;
                check_max_agents(&policy, "EpsilonGreedyQ", game)?;
                let states: Vec<usize> = (0..impls.len())
                    .map(|i| {
                        policy.begin_episode(&mut SplitMix64::keyed(
                            seed,
                            rng_stream::CHOOSE_DRIVE,
                            i as u64,
                        ))
                    })
                    .collect();
                run_choose(&policy, game, enc, reward, requests, seed, states, infer)
            }
            PolicySpec::Minimax {
                depth,
                top_k,
                chance,
            } => {
                let policy = minimax_from_spec(depth, top_k, chance, gamma);
                check_information("Minimax", game)?;
                check_minimax_composition(game)?;
                check_max_agents(&policy, "Minimax", game)?;
                let states = vec![(); impls.len()];
                run_choose(&policy, game, enc, reward, requests, seed, states, infer)
            }
            PolicySpec::Mcts {
                num_simulations,
                uct_c,
                max_depth,
                act_by,
                temperature,
                temperature_drop,
                chance,
            } => {
                let policy = mcts_from_spec(
                    num_simulations,
                    uct_c,
                    max_depth,
                    act_by,
                    temperature,
                    temperature_drop,
                    chance,
                    gamma,
                )?;
                check_information("Mcts", game)?;
                check_max_agents(&policy, "Mcts", game)?;
                check_joint_space("Mcts", game, game.num_agents())?;
                let states = plies
                    .iter()
                    .map(|&p| {
                        policy
                            .policy_state_from_u64(p)
                            .map_err(pyo3::exceptions::PyValueError::new_err)
                    })
                    .collect::<PyResult<Vec<u32>>>()?;
                run_choose(&policy, game, enc, reward, requests, seed, states, infer)
            }
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
                sequential_backup,
            } => {
                let policy = alphazero_from_spec(
                    num_simulations,
                    c_puct,
                    max_depth,
                    noise_epsilon,
                    noise_alpha,
                    temperature,
                    temperature_drop,
                    chance,
                    noise_scope,
                    sequential_backup,
                    gamma,
                )?;
                check_information("AlphaZero", game)?;
                check_max_agents(&policy, "AlphaZero", game)?;
                check_joint_space("AlphaZero", game, game.num_agents())?;
                let states = plies
                    .iter()
                    .map(|&p| {
                        policy
                            .policy_state_from_u64(p)
                            .map_err(pyo3::exceptions::PyValueError::new_err)
                    })
                    .collect::<PyResult<Vec<u32>>>()?;
                run_choose(&policy, game, enc, reward, requests, seed, states, infer)
            }
        }
    }

    fn restore_parts(
        &mut self,
        state: &[u8],
        rng_state: u64,
        done: bool,
        ticks: u64,
    ) -> PyResult<()> {
        let codec = self.codec.as_deref().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("this game does not support snapshots")
        })?;
        let decoded = codec.decode(state).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("invalid snapshot state: {e}"))
        })?;
        codec.validate_decoded_state(&decoded, done).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("invalid snapshot state: {e}"))
        })?;
        self.inner
            .set_parts(decoded, rng_state, done, ticks as usize);
        self.last_rewards = None;
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
    fn information_state_key(&self, agent: usize) -> PyResult<Vec<u8>> {
        let game = self.inner.game();
        if !game.information_states() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "this game does not declare information states",
            ));
        }
        if agent >= game.num_agents() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "agent {agent} out of range"
            )));
        }
        Ok(game.information_state_key(self.inner.state(), agent))
    }
    fn step<'py>(
        &mut self,
        py: Python<'py>,
        actions: Vec<usize>,
    ) -> PyResult<Vec<Bound<'py, PyAny>>> {
        let trace = self.inner.step(&actions);
        self.last_rewards = self.reward.as_ref().map(|r| {
            let mut out = vec![0.0; self.inner.num_agents()];
            for (agent, e) in &trace {
                out[*agent] += r.step_reward(e, *agent);
            }
            out
        });
        trace
            .iter()
            .map(|(agent, e)| {
                let ev = e.to_py(py)?;
                Ok((*agent, ev).into_pyobject(py)?.into_any())
            })
            .collect()
    }
    fn last_rewards(&self) -> Option<Vec<f64>> {
        self.last_rewards.clone()
    }
}

fn build_env(game: GameSpec, reward: Option<PyReward>, seed: u64) -> PyResult<Box<dyn ErasedEnv>> {
    let reward = reward.map(|r| build_reward(&game, Some(r))).transpose()?;
    Ok(match game {
        GameSpec::Snake {
            num_snakes,
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
                        num_snakes,
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
                    num_snakes,
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
                codec: Some(Box::new(game.clone())),
                inner: Env::new(game, enc, seed),
                obs_shape,
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
        GameSpec::TexasHoldem {
            num_players,
            stack,
            small_blind,
            big_blind,
        } => {
            let enc = HoldemEgocentric { num_players, stack };
            let obs_shape = enc.obs_shape();
            Box::new(EnvImpl {
                inner: Env::new(
                    TexasHoldem {
                        num_players,
                        stack,
                        small_blind,
                        big_blind,
                    },
                    Box::new(enc),
                    seed,
                ),
                obs_shape,
                codec: Some(Box::new(TexasHoldem {
                    num_players,
                    stack,
                    small_blind,
                    big_blind,
                })),
                reward: reward.map(|rb| match rb {
                    RewardBox::Holdem(r) => Box::new(r) as Box<dyn Reward<Event = f64>>,
                    _ => unreachable!("build_reward returns the reward variant matching the game"),
                }),
                last_rewards: None,
            })
        }
        GameSpec::KuhnPoker { players } => {
            let obs_shape = KuhnEncoder { players }.obs_shape();
            Box::new(EnvImpl {
                inner: Env::new(
                    KuhnPoker { players },
                    Box::new(KuhnEncoder { players }),
                    seed,
                ),
                obs_shape,
                codec: Some(Box::new(KuhnPoker { players })),
                reward: reward.map(|rb| match rb {
                    RewardBox::Holdem(r) => Box::new(r) as Box<dyn Reward<Event = f64>>,
                    _ => unreachable!("build_reward returns the reward variant matching the game"),
                }),
                last_rewards: None,
            })
        }
        GameSpec::LeducPoker => {
            let obs_shape = LeducEncoder.obs_shape();
            Box::new(EnvImpl {
                inner: Env::new(LeducPoker, Box::new(LeducEncoder), seed),
                obs_shape,
                codec: Some(Box::new(LeducPoker)),
                reward: reward.map(|rb| match rb {
                    RewardBox::Holdem(r) => Box::new(r) as Box<dyn Reward<Event = f64>>,
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
        if let Some((k, v)) = weights.iter().find(|(_, v)| !v.is_finite()) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "reward weight {k:?} must be finite, got {v}"
            )));
        }
        Ok(PyReward { weights })
    }
}

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

#[pyclass(name = "Box", module = "reinfors.spaces")]
// Bounds are arrays shaped like the observation even though core currently supplies scalar bounds;
// this leaves room for per-element bounds without changing the Python type.
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

#[pyclass]
#[derive(Clone)]
struct GameHandle {
    spec: GameSpec,
    encoder: EncoderSpec,
}

const ENCODER_DOCS: &str =
    "https://jeepjeepjeep.github.io/reinfors/catalogue/games/#observation-encoders";

fn game_handle(mut spec: GameSpec, encoder: Option<EncoderHandle>) -> PyResult<GameHandle> {
    let selected = encoder.map_or_else(|| spec.default_encoder(), |handle| handle.spec);
    if !spec.accepts_encoder(selected) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "incompatible encoder {:?} for game {:?}; see {}",
            selected.name(),
            spec.name(),
            ENCODER_DOCS
        )));
    }
    if let (GameSpec::Chess { encoder, .. }, EncoderSpec::Chess(chess)) = (&mut spec, selected) {
        *encoder = chess;
    }
    Ok(GameHandle {
        spec,
        encoder: selected,
    })
}

#[pymethods]
impl GameHandle {
    #[staticmethod]
    #[pyo3(signature = (grid_size=20, initial_length=3, food=3, play_to_last=true, win_food_lead=None, max_ticks=1000, num_snakes=2, encoder=None))]
    #[pyo3(name = "Snake")]
    #[allow(clippy::too_many_arguments)]
    fn snake(
        grid_size: i32,
        initial_length: usize,
        food: usize,
        play_to_last: bool,
        win_food_lead: Option<usize>,
        max_ticks: Option<usize>,
        num_snakes: usize,
        encoder: Option<EncoderHandle>,
    ) -> PyResult<Self> {
        check_max_ticks(max_ticks)?;
        Snake {
            num_snakes,
            grid_size,
            initial_length,
            initial_food_count: food,
            play_to_last,
            win_food_lead,
            max_ticks,
        }
        .validate()
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
        game_handle(
            GameSpec::Snake {
                num_snakes,
                grid_size,
                initial_length,
                initial_food_count: food,
                play_to_last,
                win_food_lead,
                max_ticks,
            },
            encoder,
        )
    }

    #[staticmethod]
    #[pyo3(signature = (num_players=6, stack=200, small_blind=5, big_blind=10, encoder=None))]
    #[pyo3(name = "TexasHoldem")]
    fn texas_holdem(
        num_players: usize,
        stack: u32,
        small_blind: u32,
        big_blind: u32,
        encoder: Option<EncoderHandle>,
    ) -> PyResult<Self> {
        TexasHoldem {
            num_players,
            stack,
            small_blind,
            big_blind,
        }
        .validate()
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
        game_handle(
            GameSpec::TexasHoldem {
                num_players,
                stack,
                small_blind,
                big_blind,
            },
            encoder,
        )
    }

    #[staticmethod]
    #[pyo3(name = "KuhnPoker", signature = (players=2, encoder=None))]
    fn kuhn_poker(players: usize, encoder: Option<EncoderHandle>) -> PyResult<Self> {
        (KuhnPoker { players })
            .validate()
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        game_handle(GameSpec::KuhnPoker { players }, encoder)
    }

    #[staticmethod]
    #[pyo3(name = "LeducPoker", signature = (encoder=None))]
    fn leduc_poker(encoder: Option<EncoderHandle>) -> PyResult<Self> {
        game_handle(GameSpec::LeducPoker, encoder)
    }

    #[staticmethod]
    #[pyo3(name = "Connect4", signature = (encoder=None))]
    fn connect4(encoder: Option<EncoderHandle>) -> PyResult<Self> {
        game_handle(GameSpec::Connect4, encoder)
    }

    #[staticmethod]
    #[pyo3(signature = (max_ticks=512, encoder=None))]
    #[pyo3(name = "Chess")]
    fn chess(max_ticks: Option<usize>, encoder: Option<EncoderHandle>) -> PyResult<Self> {
        check_max_ticks(max_ticks)?;
        game_handle(
            GameSpec::Chess {
                max_ticks,
                encoder: ChessEncoderSpec::Minimal,
            },
            encoder,
        )
    }

    #[staticmethod]
    #[pyo3(signature = (max_ticks=1000, encoder=None))]
    #[pyo3(name = "Backgammon")]
    fn backgammon(max_ticks: Option<usize>, encoder: Option<EncoderHandle>) -> PyResult<Self> {
        check_max_ticks(max_ticks)?;
        game_handle(GameSpec::Backgammon { max_ticks }, encoder)
    }

    #[staticmethod]
    #[pyo3(signature = (size=5, goal_row=None, goal_col=None, max_ticks=1000, encoder=None))]
    #[pyo3(name = "GridWorld")]
    fn gridworld(
        size: i32,
        goal_row: Option<i32>,
        goal_col: Option<i32>,
        max_ticks: Option<usize>,
        encoder: Option<EncoderHandle>,
    ) -> PyResult<Self> {
        check_max_ticks(max_ticks)?;
        // Invalid negative sizes must reach validate() as errors, not overflow here.
        let corner = size.saturating_sub(1);
        let goal = (goal_row.unwrap_or(corner), goal_col.unwrap_or(corner));
        GridWorld {
            size,
            goal,
            max_ticks,
        }
        .validate()
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
        game_handle(
            GameSpec::GridWorld {
                size,
                goal,
                max_ticks,
            },
            encoder,
        )
    }

    fn observation_space<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        space_to_py(py, self.spec.spaces().0)
    }

    fn action_space<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        space_to_py(py, self.spec.spaces().1)
    }

    #[getter]
    fn encoder(&self) -> EncoderHandle {
        EncoderHandle { spec: self.encoder }
    }

    fn truncation_horizon(&self) -> Option<usize> {
        match self.spec {
            GameSpec::Snake { max_ticks, .. }
            | GameSpec::Chess { max_ticks, .. }
            | GameSpec::Backgammon { max_ticks }
            | GameSpec::GridWorld { max_ticks, .. } => max_ticks,
            GameSpec::Connect4
            | GameSpec::TexasHoldem { .. }
            | GameSpec::KuhnPoker { .. }
            | GameSpec::LeducPoker => None,
        }
    }
}

trait ErasedDeepCfr: Send + Sync {
    fn next_iteration(&mut self);
    fn iteration(&self) -> u64;
    #[allow(clippy::type_complexity)]
    fn collect(
        &mut self,
        player: usize,
        traversals: usize,
        infer: &mut dyn FnMut(usize, Vec<f32>, usize) -> Vec<f64>,
    ) -> (
        Vec<reinfors_core::AdvantageSample>,
        Vec<reinfors_core::StrategySample>,
        reinfors_core::DeepCfrStats,
    );
    #[allow(clippy::type_complexity)]
    fn infoset_features(
        &self,
    ) -> Result<Vec<(Vec<u8>, Vec<f32>, Vec<usize>)>, reinfors_core::EnumerationCapExceeded>;
    fn exploitability_of(
        &self,
        probs: &HashMap<Vec<u8>, Vec<f64>>,
    ) -> Result<f64, reinfors_core::EnumerationCapExceeded>;
    fn rollback_collect(&mut self);
}

impl<G: Game + Send + Sync> ErasedDeepCfr for reinfors_core::DeepCfrSolver<G> {
    fn next_iteration(&mut self) {
        reinfors_core::DeepCfrSolver::next_iteration(self)
    }
    fn iteration(&self) -> u64 {
        reinfors_core::DeepCfrSolver::iteration(self)
    }
    fn collect(
        &mut self,
        player: usize,
        traversals: usize,
        infer: &mut dyn FnMut(usize, Vec<f32>, usize) -> Vec<f64>,
    ) -> (
        Vec<reinfors_core::AdvantageSample>,
        Vec<reinfors_core::StrategySample>,
        reinfors_core::DeepCfrStats,
    ) {
        reinfors_core::DeepCfrSolver::collect(self, player, traversals, infer)
    }
    fn infoset_features(
        &self,
    ) -> Result<Vec<(Vec<u8>, Vec<f32>, Vec<usize>)>, reinfors_core::EnumerationCapExceeded> {
        reinfors_core::DeepCfrSolver::infoset_features(self)
    }
    fn exploitability_of(
        &self,
        probs: &HashMap<Vec<u8>, Vec<f64>>,
    ) -> Result<f64, reinfors_core::EnumerationCapExceeded> {
        reinfors_core::DeepCfrSolver::exploitability_of(self, probs)
    }
    fn rollback_collect(&mut self) {
        reinfors_core::DeepCfrSolver::rollback_collect(self)
    }
}

trait ErasedCfr: Send + Sync {
    fn iterate(&mut self, n: u64);
    fn iterations(&self) -> u64;
    fn num_infosets(&self) -> usize;
    fn exploitability(&self) -> Result<f64, reinfors_core::EnumerationCapExceeded>;
    fn nash_conv(&self) -> Result<f64, reinfors_core::EnumerationCapExceeded>;
    fn best_response_values(&self) -> Result<Vec<f64>, reinfors_core::EnumerationCapExceeded>;
    fn num_players(&self) -> usize;
    fn expected_value(&self, player: usize) -> f64;
    fn average_strategy(&self, key: &[u8]) -> Option<(Vec<usize>, Vec<f64>)>;
    fn save(&self) -> Vec<u8>;
    fn load(&mut self, bytes: &[u8]) -> Result<(), String>;
}

impl<G: Game + Send + Sync> ErasedCfr for reinfors_core::CfrSolver<G> {
    fn iterate(&mut self, n: u64) {
        reinfors_core::CfrSolver::iterate(self, n)
    }
    fn iterations(&self) -> u64 {
        reinfors_core::CfrSolver::iterations(self)
    }
    fn num_infosets(&self) -> usize {
        reinfors_core::CfrSolver::num_infosets(self)
    }
    fn exploitability(&self) -> Result<f64, reinfors_core::EnumerationCapExceeded> {
        reinfors_core::CfrSolver::exploitability(self)
    }
    fn nash_conv(&self) -> Result<f64, reinfors_core::EnumerationCapExceeded> {
        reinfors_core::CfrSolver::nash_conv(self)
    }
    fn best_response_values(&self) -> Result<Vec<f64>, reinfors_core::EnumerationCapExceeded> {
        reinfors_core::CfrSolver::best_response_values(self)
    }
    fn num_players(&self) -> usize {
        reinfors_core::CfrSolver::num_players(self)
    }
    fn expected_value(&self, player: usize) -> f64 {
        reinfors_core::CfrSolver::expected_value(self, player)
    }
    fn average_strategy(&self, key: &[u8]) -> Option<(Vec<usize>, Vec<f64>)> {
        reinfors_core::CfrSolver::average_strategy(self, key)
    }
    fn save(&self) -> Vec<u8> {
        reinfors_core::CfrSolver::save(self)
    }
    fn load(&mut self, bytes: &[u8]) -> Result<(), String> {
        reinfors_core::CfrSolver::load(self, bytes)
    }
}

fn cap_err(e: reinfors_core::EnumerationCapExceeded) -> pyo3::PyErr {
    pyo3::exceptions::PyValueError::new_err(e.to_string())
}

#[pyclass(name = "Cfr")]
struct PyCfr {
    inner: Box<dyn ErasedCfr>,
    // Hashes (game parameters, native-unit reward, variant); loading another tuple would silently
    // reinterpret its tables.
    fingerprint: String,
}

const CFR_SNAPSHOT_MAGIC: &[u8; 4] = b"RFCF";
const CFR_SNAPSHOT_SCHEMA: u8 = 1;

#[pymethods]
impl PyCfr {
    #[new]
    #[pyo3(signature = (game, variant="plus", seed=0))]
    fn new(game: &GameHandle, variant: &str, seed: u64) -> PyResult<Self> {
        use reinfors_core::{CfrSolver, CfrVariant};
        let variant_name = variant;
        let variant = match variant {
            "vanilla" => CfrVariant::Vanilla,
            "plus" => CfrVariant::Plus,
            "external_mccfr" => CfrVariant::ExternalMccfr,
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "unknown CFR variant {other:?}: expected \"vanilla\", \"plus\", or \"external_mccfr\""
                )))
            }
        };
        let reward = HoldemReward { scale: 1.0 };
        let inner: Box<dyn ErasedCfr> = match game.spec {
            GameSpec::KuhnPoker { players } => Box::new(CfrSolver::new(
                KuhnPoker { players },
                Box::new(reward),
                variant,
                seed,
            )),
            GameSpec::LeducPoker => {
                Box::new(CfrSolver::new(LeducPoker, Box::new(reward), variant, seed))
            }
            GameSpec::TexasHoldem {
                num_players,
                stack,
                small_blind,
                big_blind,
            } => {
                if variant != CfrVariant::ExternalMccfr {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "this game's chance fan is not supported by an exact CFR variant; \
                             use variant=\"external_mccfr\" or see {}",
                        reinfors_core::COMPATIBILITY_DOCS
                    )));
                }
                Box::new(CfrSolver::new(
                    TexasHoldem {
                        num_players,
                        stack,
                        small_blind,
                        big_blind,
                    },
                    Box::new(reward),
                    variant,
                    seed,
                ))
            }
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "this game is not compatible with CFR; see {}",
                    reinfors_core::COMPATIBILITY_DOCS
                )))
            }
        };
        let composition = json!({
            "solver": {"name": "cfr", "variant": variant_name},
            "game": game_cfg(&game.spec, game.encoder),
            "reward": {"scale": 1.0},
        });
        let fingerprint = fingerprint_hex(&canonical_config_bytes(&composition));
        Ok(PyCfr { inner, fingerprint })
    }

    fn iterate(&mut self, py: Python<'_>, n: u64) {
        py.allow_threads(|| self.inner.iterate(n));
    }

    #[getter]
    fn iterations(&self) -> u64 {
        self.inner.iterations()
    }

    #[getter]
    fn num_infosets(&self) -> usize {
        self.inner.num_infosets()
    }

    #[getter]
    fn num_players(&self) -> usize {
        self.inner.num_players()
    }

    fn exploitability(&self, py: Python<'_>) -> PyResult<f64> {
        py.allow_threads(|| self.inner.exploitability())
            .map_err(cap_err)
    }

    fn nash_conv(&self, py: Python<'_>) -> PyResult<f64> {
        py.allow_threads(|| self.inner.nash_conv()).map_err(cap_err)
    }

    fn best_response_values(&self, py: Python<'_>) -> PyResult<Vec<f64>> {
        py.allow_threads(|| self.inner.best_response_values())
            .map_err(cap_err)
    }

    fn expected_value(&self, py: Python<'_>, player: usize) -> PyResult<f64> {
        if player >= self.inner.num_players() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "player must be below {}",
                self.inner.num_players()
            )));
        }
        Ok(py.allow_threads(|| self.inner.expected_value(player)))
    }

    fn average_strategy(&self, key: &[u8]) -> Option<(Vec<usize>, Vec<f64>)> {
        self.inner.average_strategy(key)
    }

    fn save<'py>(&self, py: Python<'py>) -> Bound<'py, pyo3::types::PyBytes> {
        // Core payload includes tables, iteration, and sampling RNG, so an MCCFR resume is
        // bit-identical when this composition fingerprint matches.
        let mut out = Vec::new();
        out.extend_from_slice(CFR_SNAPSHOT_MAGIC);
        out.push(CFR_SNAPSHOT_SCHEMA);
        // Deliberately no length prefix: SHA-256 hex is fixed at 64 bytes, and
        // load's [5..69] offsets are part of this schema.
        out.extend_from_slice(self.fingerprint.as_bytes());
        out.extend_from_slice(&self.inner.save());
        pyo3::types::PyBytes::new(py, &out)
    }

    fn load(&mut self, bytes: &[u8]) -> PyResult<()> {
        let err = pyo3::exceptions::PyValueError::new_err;
        if bytes.len() < 4 + 1 + 64 || &bytes[..4] != CFR_SNAPSHOT_MAGIC {
            return Err(err("not a CFR snapshot payload"));
        }
        if bytes[4] != CFR_SNAPSHOT_SCHEMA {
            return Err(err("unknown CFR snapshot schema version"));
        }
        let fingerprint = &bytes[5..69];
        if fingerprint != self.fingerprint.as_bytes() {
            return Err(err(
                "this snapshot was saved from a different composition (game parameters, \
                 reward, or CFR variant)",
            ));
        }
        self.inner
            .load(&bytes[69..])
            .map_err(pyo3::exceptions::PyValueError::new_err)
    }
}

#[pyclass(name = "DeepCfrBatch")]
struct DeepCfrBatch {
    #[pyo3(get)]
    advantage_obs: Py<PyArray2<f32>>,
    #[pyo3(get)]
    advantage_iterations: Py<PyArray1<i64>>,
    #[pyo3(get)]
    advantage_legal_offsets: Py<PyArray1<i64>>,
    #[pyo3(get)]
    advantage_legal_ids: Py<PyArray1<i64>>,
    #[pyo3(get)]
    advantage_targets: Py<PyArray1<f64>>,
    #[pyo3(get)]
    strategy_obs: Py<PyArray2<f32>>,
    #[pyo3(get)]
    strategy_iterations: Py<PyArray1<i64>>,
    #[pyo3(get)]
    strategy_players: Py<PyArray1<i64>>,
    #[pyo3(get)]
    strategy_legal_offsets: Py<PyArray1<i64>>,
    #[pyo3(get)]
    strategy_legal_ids: Py<PyArray1<i64>>,
    #[pyo3(get)]
    strategy_probs: Py<PyArray1<f64>>,
    #[pyo3(get)]
    telemetry: Py<PyDict>,
}

#[pyclass(name = "DeepCfr")]
struct PyDeepCfr {
    inner: Box<dyn ErasedDeepCfr>,
    obs_dim: usize,
    action_count: usize,
    num_players: usize,
    config: Value,
}

fn deep_cfr_callbacks(infer: &Bound<'_, PyAny>, num_players: usize) -> PyResult<Vec<Py<PyAny>>> {
    if infer.is_callable() {
        return Ok(vec![infer.clone().unbind()]);
    }
    let callbacks: Vec<Py<PyAny>> = infer.extract().map_err(|_| {
        pyo3::exceptions::PyTypeError::new_err(
            "infer must be a callable or a sequence of per-player callables",
        )
    })?;
    if callbacks.len() != num_players {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "expected {num_players} per-player infer callables, got {}",
            callbacks.len()
        )));
    }
    for (player, cb) in callbacks.iter().enumerate() {
        if !cb.bind(infer.py()).is_callable() {
            return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                "per-player infer element {player} is not callable"
            )));
        }
    }
    Ok(callbacks)
}

#[pymethods]
impl PyDeepCfr {
    #[new]
    #[pyo3(signature = (game, seed=0))]
    fn new(game: &GameHandle, seed: u64) -> PyResult<Self> {
        use reinfors_core::DeepCfrSolver;
        let reward = HoldemReward { scale: 1.0 };
        let (inner, obs_dim, action_count): (Box<dyn ErasedDeepCfr>, usize, usize) = match game.spec
        {
            GameSpec::KuhnPoker { players } => (
                Box::new(DeepCfrSolver::new(
                    KuhnPoker { players },
                    Box::new(KuhnEncoder { players }),
                    Box::new(reward),
                    seed,
                )),
                3 * players,
                2,
            ),
            GameSpec::LeducPoker => (
                Box::new(DeepCfrSolver::new(
                    LeducPoker,
                    Box::new(LeducEncoder),
                    Box::new(reward),
                    seed,
                )),
                21,
                3,
            ),
            GameSpec::TexasHoldem {
                num_players,
                stack,
                small_blind,
                big_blind,
            } => {
                let enc = HoldemEgocentric { num_players, stack };
                let (c, h, w) = enc.obs_shape();
                (
                    Box::new(DeepCfrSolver::new(
                        TexasHoldem {
                            num_players,
                            stack,
                            small_blind,
                            big_blind,
                        },
                        Box::new(enc),
                        Box::new(reward),
                        seed,
                    )),
                    c * h * w,
                    3,
                )
            }
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "this game is not compatible with Deep CFR; see {}",
                    reinfors_core::COMPATIBILITY_DOCS
                )))
            }
        };
        let config = json!({
            "schema": CONFIG_SCHEMA_VERSION,
            "solver": {"name": "deep_cfr", "seed": seed},
            "game": game_cfg(&game.spec, game.encoder),
            "reward": {"scale": 1.0},
        });
        Ok(PyDeepCfr {
            inner,
            obs_dim,
            action_count,
            num_players: game.spec.num_agents(),
            config,
        })
    }

    fn next_iteration(&mut self) {
        self.inner.next_iteration();
    }

    #[getter]
    fn iteration(&self) -> u64 {
        self.inner.iteration()
    }

    fn resolved_config<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        value_to_py(py, &self.config)
    }

    #[pyo3(signature = (player, traversals, infer))]
    fn collect<'py>(
        &mut self,
        py: Python<'py>,
        player: usize,
        traversals: usize,
        infer: &Bound<'py, PyAny>,
    ) -> PyResult<DeepCfrBatch> {
        if player >= self.num_players {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "player must be below {}",
                self.num_players
            )));
        }
        if self.inner.iteration() == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "call next_iteration() before collecting (samples are weighted by the iteration)",
            ));
        }
        let callbacks = deep_cfr_callbacks(infer, self.num_players)?;
        let (dim, a) = (self.obs_dim, self.action_count);
        let mut callback_err: Option<PyErr> = None;
        let mut rust_infer = |who: usize, obs_flat: Vec<f32>, rows: usize| -> Vec<f64> {
            if callback_err.is_some() {
                return vec![0.0; rows * a];
            }
            let target = if callbacks.len() == 1 {
                &callbacks[0]
            } else {
                &callbacks[who]
            };
            let arr = Array2::from_shape_vec((rows, dim), obs_flat)
                .expect("obs batch shape")
                .into_pyarray(py);
            match target
                .bind(py)
                .call1((arr,))
                .and_then(|r| infer_rows_2d(&r, "infer output"))
            {
                Ok((shape, flat)) => {
                    // Element counts alone would admit transposed advantage arrays.
                    if shape != [rows, a] {
                        callback_err.get_or_insert_with(|| {
                            pyo3::exceptions::PyValueError::new_err(format!(
                                "infer returned shape {shape:?} for {rows} rows; expected \
                                 ({rows}, {a}) — one row of {a} advantages per query"
                            ))
                        });
                        return vec![0.0; rows * a];
                    }
                    if flat.iter().any(|value| !value.is_finite()) {
                        callback_err.get_or_insert_with(|| {
                            pyo3::exceptions::PyValueError::new_err(
                                "Deep CFR infer outputs must contain only finite values",
                            )
                        });
                        return vec![0.0; rows * a];
                    }
                    flat
                }
                Err(e) => {
                    callback_err = Some(e);
                    vec![0.0; rows * a]
                }
            }
        };
        let (advantage, strategy, stats) = self.inner.collect(player, traversals, &mut rust_infer);
        if let Some(e) = callback_err {
            // A failed collection is transactional: retry from the same sampling sequence.
            self.inner.rollback_collect();
            return Err(e);
        }

        fn csr(items: &[(Vec<usize>, Vec<f64>)]) -> (Vec<i64>, Vec<i64>, Vec<f64>) {
            let mut offsets = Vec::with_capacity(items.len() + 1);
            let mut ids = Vec::new();
            let mut values = Vec::new();
            offsets.push(0i64);
            for (legal, vals) in items {
                ids.extend(legal.iter().map(|&x| x as i64));
                values.extend_from_slice(vals);
                offsets.push(ids.len() as i64);
            }
            (offsets, ids, values)
        }
        let adv_rows: Vec<(Vec<usize>, Vec<f64>)> = advantage
            .iter()
            .map(|s| (s.legal.clone(), s.targets.clone()))
            .collect();
        let (a_off, a_ids, a_vals) = csr(&adv_rows);
        let strat_rows: Vec<(Vec<usize>, Vec<f64>)> = strategy
            .iter()
            .map(|s| (s.legal.clone(), s.probs.clone()))
            .collect();
        let (s_off, s_ids, s_vals) = csr(&strat_rows);

        let flat2 = |rows: usize, data: Vec<f32>| -> Py<PyArray2<f32>> {
            Array2::from_shape_vec((rows, dim), data)
                .expect("obs shape")
                .into_pyarray(py)
                .unbind()
        };
        let adv_obs: Vec<f32> = advantage
            .iter()
            .flat_map(|s| s.obs.iter().copied())
            .collect();
        let strat_obs: Vec<f32> = strategy
            .iter()
            .flat_map(|s| s.obs.iter().copied())
            .collect();

        let telemetry = PyDict::new(py);
        telemetry.set_item("player", player)?;
        telemetry.set_item("traversals", stats.traversals)?;
        telemetry.set_item("advantage_samples", stats.advantage_samples)?;
        telemetry.set_item("strategy_samples", stats.strategy_samples)?;
        telemetry.set_item("infer_calls", stats.infer_calls)?;
        telemetry.set_item("infer_rows", stats.infer_rows)?;
        telemetry.set_item("infer_seconds", stats.infer_seconds)?;
        telemetry.set_item("collect_seconds", stats.collect_seconds)?;
        telemetry.set_item("cache_lookups", stats.cache_lookups)?;
        telemetry.set_item("cache_hits", stats.cache_hits)?;

        Ok(DeepCfrBatch {
            advantage_obs: flat2(advantage.len(), adv_obs),
            advantage_iterations: PyArray1::from_iter(
                py,
                advantage.iter().map(|s| s.iteration as i64),
            )
            .unbind(),
            advantage_legal_offsets: PyArray1::from_vec(py, a_off).unbind(),
            advantage_legal_ids: PyArray1::from_vec(py, a_ids).unbind(),
            advantage_targets: PyArray1::from_vec(py, a_vals).unbind(),
            strategy_obs: flat2(strategy.len(), strat_obs),
            strategy_iterations: PyArray1::from_iter(
                py,
                strategy.iter().map(|s| s.iteration as i64),
            )
            .unbind(),
            strategy_players: PyArray1::from_iter(py, strategy.iter().map(|s| s.player as i64))
                .unbind(),
            strategy_legal_offsets: PyArray1::from_vec(py, s_off).unbind(),
            strategy_legal_ids: PyArray1::from_vec(py, s_ids).unbind(),
            strategy_probs: PyArray1::from_vec(py, s_vals).unbind(),
            telemetry: telemetry.unbind(),
        })
    }

    fn exploitability(&self, py: Python<'_>, policy_infer: &Bound<'_, PyAny>) -> PyResult<f64> {
        let features = self.inner.infoset_features().map_err(cap_err)?;
        let (dim, a) = (self.obs_dim, self.action_count);
        let mut obs_flat: Vec<f32> = Vec::with_capacity(features.len() * dim);
        for (_, obs, _) in &features {
            obs_flat.extend_from_slice(obs);
        }
        let arr = Array2::from_shape_vec((features.len(), dim), obs_flat)
            .expect("obs shape")
            .into_pyarray(py);
        let (oshape, flat) = infer_rows_2d(&policy_infer.call1((arr,))?, "policy_infer output")?;
        if oshape != [features.len(), a] {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "policy_infer returned shape {oshape:?} for {} infosets; expected ({}, {a})",
                features.len(),
                features.len()
            )));
        }
        let mut probs: HashMap<Vec<u8>, Vec<f64>> = HashMap::with_capacity(features.len());
        for (i, (key, _, legal)) in features.iter().enumerate() {
            let row = &flat[i * a..(i + 1) * a];
            let clamped: Vec<f64> = legal.iter().map(|&x| row[x].max(0.0)).collect();
            let total: f64 = clamped.iter().sum();
            let sigma = if total > 0.0 {
                clamped.iter().map(|c| c / total).collect()
            } else {
                vec![1.0 / legal.len() as f64; legal.len()]
            };
            probs.insert(key.clone(), sigma);
        }
        py.allow_threads(|| self.inner.exploitability_of(&probs))
            .map_err(cap_err)
    }
}

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
    #[pyo3(signature = (depth=4, top_k=None, chance=None))]
    #[pyo3(name = "Minimax")]
    fn minimax(
        depth: i32,
        top_k: Option<usize>,
        chance: Option<ChanceModeHandle>,
    ) -> PyResult<Self> {
        if depth < 1 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "minimax needs at least one ply of lookahead (depth >= 1)",
            ));
        }
        if top_k == Some(0) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "top_k must keep at least one move per node",
            ));
        }
        // ExpandAll by default: the documented expectiminimax semantics are an exact
        // expectation, not a seeded sample; Committed{k} is the opt-in for wide fans.
        let chance = chance.map_or(ChanceMode::ExpandAll, |c| c.mode);
        if !Minimax::supports_chance_mode(chance) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Minimax expands each node exactly once and cannot express per-traversal \
                 chance modes; use Committed or ExpandAll",
            ));
        }
        Ok(PolicyHandle {
            spec: PolicySpec::Minimax {
                depth,
                top_k,
                chance,
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

    #[staticmethod]
    #[pyo3(name = "Ppo")]
    fn ppo() -> PolicyHandle {
        PolicyHandle {
            spec: PolicySpec::Ppo,
        }
    }

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

    #[staticmethod]
    #[pyo3(signature = (num_simulations=64, c_puct=1.5, max_depth=64, temperature=1.0, temperature_drop=8, chance=None, noise=Some(NoiseHandle::default_dirichlet()), sequential_backup="auto"))]
    #[pyo3(name = "AlphaZero")]
    #[allow(clippy::too_many_arguments)]
    fn alphazero(
        num_simulations: usize,
        c_puct: f64,
        max_depth: i32,
        temperature: f64,
        temperature_drop: Option<u32>,
        chance: Option<ChanceModeHandle>,
        noise: Option<NoiseHandle>,
        sequential_backup: &str,
    ) -> PyResult<Self> {
        check_nonneg_finite("c_puct", c_puct)?;
        check_nonneg_finite("temperature", temperature)?;
        let (noise_epsilon, noise_alpha, noise_scope) = match noise {
            Some(n) => (n.epsilon, n.alpha, n.scope),
            None => (0.0, 0.3, NoiseScope::Requester),
        };
        let sequential_backup = match sequential_backup {
            "auto" => reinfors_core::SequentialBackup::Auto,
            "maxn" => reinfors_core::SequentialBackup::MaxN,
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "unknown sequential_backup {other:?}; expected \"auto\" or \"maxn\""
                )))
            }
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
                sequential_backup,
            },
        })
    }

    /// One batched decision per env: one pooled search across the envs, leaves batched
    /// into a single shared `infer` callback. STATELESS: nothing persists between calls —
    /// multi-head Thompson draws happen per call, not per episode (the engine is the
    /// reference for per-episode training semantics). Reproducible for the same ordered
    /// batch, seed and deterministic inference. Pure: envs are not mutated. `plies`
    /// defaults to each env's own tick count; `gamma` must be the discount the model was
    /// trained with (the engine takes it from the learner).
    #[pyo3(signature = (envs, infer, seed=0, plies=None, *, gamma))]
    fn choose(
        &self,
        py: Python<'_>,
        envs: Vec<Py<PyEnv>>,
        infer: Bound<'_, PyAny>,
        seed: u64,
        plies: Option<Vec<u64>>,
        gamma: f64,
    ) -> PyResult<Vec<usize>> {
        if envs.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "choose needs at least one env",
            ));
        }
        if !infer.is_callable() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "choose takes a single shared infer callable",
            ));
        }
        check_unit("gamma", gamma)?;
        let borrowed: Vec<PyRef<'_, PyEnv>> = envs.iter().map(|e| e.borrow(py)).collect();
        if matches!(self.spec, PolicySpec::Minimax { .. }) {
            check_minimax_zero_sum(&borrowed[0].game_spec, borrowed[0].reward_weights.as_ref())?;
        }
        let fingerprint = borrowed[0].fingerprint.clone();
        for env in &borrowed {
            if env.fingerprint != fingerprint {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "envs must share one composition (game, encoder, reward)",
                ));
            }
            if env.inner.done() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "choose on a finished env",
                ));
            }
            if env.reward_weights.is_none() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "choose needs terminal values: construct these Envs with a reward",
                ));
            }
        }
        let plies = match plies {
            Some(p) => {
                if p.len() != envs.len() {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "{} plies for {} envs",
                        p.len(),
                        envs.len()
                    )));
                }
                p
            }
            None => borrowed.iter().map(|b| b.inner.ticks()).collect(),
        };
        let (expected_heads, layout) = match &self.spec {
            PolicySpec::AlphaZero { .. } => {
                (1usize, InferLayout::PolicyValue { label: "AlphaZero" })
            }
            PolicySpec::Mcts { .. } => (1, InferLayout::ValueHeads),
            PolicySpec::SelectiveExpectimax { n_heads, .. } => (*n_heads, InferLayout::ValueHeads),
            PolicySpec::EpsilonGreedyQ { n_heads, .. } => (*n_heads, InferLayout::ValueHeads),
            PolicySpec::Minimax { .. } => (1, InferLayout::ValueHeads),
            PolicySpec::Ppo => (1, InferLayout::PolicyValue { label: "Ppo" }),
        };
        let dim = borrowed[0].inner.obs_dim();
        let action_count = borrowed[0].inner.action_count();
        check_search_budgets(&self.spec)?;
        check_call_buffers(
            "envs",
            envs.len(),
            borrowed[0].inner.num_agents(),
            dim,
            expected_heads,
            action_count,
        )?;
        let callbacks = vec![infer.clone().unbind()];
        let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let callback_err = std::sync::Arc::new(std::sync::Mutex::new(None));
        let mut infer_fn = infer_closure_gil(
            &callbacks,
            dim,
            action_count,
            expected_heads,
            layout,
            stop,
            callback_err.clone(),
        );
        let env_refs: Vec<&dyn ErasedEnv> = borrowed.iter().map(|b| &*b.inner).collect();
        let spec = &self.spec;
        let outcome = py.allow_threads(|| {
            env_refs[0].choose_batch(&env_refs, spec, gamma, seed, &plies, &mut infer_fn)
        });
        if let Some(e) = callback_err.lock().unwrap().take() {
            return Err(e);
        }
        let actions = outcome?;
        for (env, &action) in borrowed.iter().zip(&actions) {
            let agent = env.inner.active_agents()[0];
            if !env.inner.legal_actions(agent).contains(&action) {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "internal: policy chose illegal action {action}"
                )));
            }
        }
        Ok(actions)
    }
}

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
    #[pyo3(signature = (bootstrap_p=1.0, n_step=1, gamma=0.99))]
    #[pyo3(name = "Dqn")]
    fn dqn(bootstrap_p: f64, n_step: usize, gamma: f64) -> PyResult<Self> {
        check_unit("bootstrap_p", bootstrap_p)?;
        check_unit("gamma", gamma)?;
        if n_step < 1 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "n_step must be at least 1, got {n_step}"
            )));
        }
        Ok(LearnerHandle {
            spec: LearnerSpec::Dqn {
                bootstrap_p,
                n_step,
                gamma,
            },
        })
    }

    #[staticmethod]
    #[pyo3(signature = (gamma=0.99, lam=0.95))]
    #[pyo3(name = "Ppo")]
    fn ppo(gamma: f64, lam: f64) -> PyResult<Self> {
        check_unit("gamma", gamma)?;
        check_unit("lam", lam)?;
        Ok(LearnerHandle {
            spec: LearnerSpec::Ppo { gamma, lam },
        })
    }

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

#[pyclass]
#[derive(Clone)]
struct ChanceModeHandle {
    mode: ChanceMode,
}

#[pymethods]
impl ChanceModeHandle {
    #[staticmethod]
    #[pyo3(name = "AlwaysResample")]
    fn always_resample() -> Self {
        ChanceModeHandle {
            mode: ChanceMode::AlwaysResample,
        }
    }

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

    #[staticmethod]
    #[pyo3(name = "ExpandAll")]
    fn expand_all() -> Self {
        ChanceModeHandle {
            mode: ChanceMode::ExpandAll,
        }
    }
}

#[pyclass]
#[derive(Clone)]
struct NoiseHandle {
    epsilon: f64,
    alpha: f64,
    scope: NoiseScope,
}

impl NoiseHandle {
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
            "all" => NoiseScope::All,
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "unknown scope {other:?}; expected \"requester\" or \"all\""
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

#[pyclass]
#[derive(Clone)]
struct EncoderHandle {
    spec: EncoderSpec,
}

#[pymethods]
impl EncoderHandle {
    #[getter]
    fn name(&self) -> &'static str {
        self.spec.name()
    }

    #[staticmethod]
    #[pyo3(name = "Snake")]
    fn snake() -> Self {
        EncoderHandle {
            spec: EncoderSpec::Snake,
        }
    }

    #[staticmethod]
    #[pyo3(name = "Connect4")]
    fn connect4() -> Self {
        EncoderHandle {
            spec: EncoderSpec::Connect4,
        }
    }

    #[staticmethod]
    #[pyo3(name = "Backgammon")]
    fn backgammon() -> Self {
        EncoderHandle {
            spec: EncoderSpec::Backgammon,
        }
    }

    #[staticmethod]
    #[pyo3(name = "TexasHoldem")]
    fn texas_holdem() -> Self {
        EncoderHandle {
            spec: EncoderSpec::TexasHoldem,
        }
    }

    #[staticmethod]
    #[pyo3(name = "KuhnPoker")]
    fn kuhn_poker() -> Self {
        EncoderHandle {
            spec: EncoderSpec::KuhnPoker,
        }
    }

    #[staticmethod]
    #[pyo3(name = "LeducPoker")]
    fn leduc_poker() -> Self {
        EncoderHandle {
            spec: EncoderSpec::LeducPoker,
        }
    }

    #[staticmethod]
    #[pyo3(name = "GridWorld")]
    fn gridworld() -> Self {
        EncoderHandle {
            spec: EncoderSpec::GridWorld,
        }
    }

    #[staticmethod]
    #[pyo3(name = "MinimalChess")]
    fn minimal_chess() -> Self {
        EncoderHandle {
            spec: EncoderSpec::Chess(ChessEncoderSpec::Minimal),
        }
    }

    #[staticmethod]
    #[pyo3(name = "RelativeChess")]
    fn relative_chess() -> Self {
        EncoderHandle {
            spec: EncoderSpec::Chess(ChessEncoderSpec::Relative),
        }
    }

    #[staticmethod]
    #[pyo3(name = "OpenSpielChess")]
    fn openspiel_chess() -> Self {
        EncoderHandle {
            spec: EncoderSpec::Chess(ChessEncoderSpec::OpenSpiel),
        }
    }

    fn head_index(&self, action: usize, agent: usize) -> PyResult<usize> {
        let action_count = self.spec.action_count();
        if action >= action_count {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "action {action} out of range for the {action_count}-action {} encoding",
                self.spec.name()
            )));
        }
        match self.spec {
            EncoderSpec::Chess(chess) => {
                let (_, enc) = chess_parts(None, chess);
                Ok(enc.head_index(action, agent))
            }
            // Keep this exhaustive so every new encoder declares its Python action frame.
            EncoderSpec::Snake
            | EncoderSpec::Connect4
            | EncoderSpec::Backgammon
            | EncoderSpec::TexasHoldem
            | EncoderSpec::KuhnPoker
            | EncoderSpec::LeducPoker
            | EncoderSpec::GridWorld => Ok(action),
        }
    }

    fn game_action(&self, head: usize, agent: usize) -> PyResult<usize> {
        let action_count = self.spec.action_count();
        if head >= action_count {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "head {head} out of range for the {action_count}-action {} encoding",
                self.spec.name()
            )));
        }
        match self.spec {
            EncoderSpec::Chess(chess) => {
                let (_, enc) = chess_parts(None, chess);
                Ok(enc.game_action(head, agent))
            }
            EncoderSpec::Snake
            | EncoderSpec::Connect4
            | EncoderSpec::Backgammon
            | EncoderSpec::TexasHoldem
            | EncoderSpec::KuhnPoker
            | EncoderSpec::LeducPoker
            | EncoderSpec::GridWorld => Ok(head),
        }
    }

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
            spec: EncoderSpec::Chess(ChessEncoderSpec::AlphaZero {
                history: history_length,
            }),
        })
    }
}

#[pyfunction]
fn chess_uci_action(uci: &str, fen: &str) -> PyResult<usize> {
    let board: reinfors_games::ChessBoard = fen
        .parse()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err(format!("invalid FEN: {fen:?}")))?;
    reinfors_games::chess_uci_to_action(uci, &board).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(format!("{uci:?} is not a legal move in {fen:?}"))
    })
}

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

/// Source identity baked at compile time; "unknown" for builds without a git checkout.
#[pyfunction]
fn build_info(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("git_sha", env!("REINFORS_GIT_SHA"))?;
    match env!("REINFORS_GIT_DIRTY") {
        "true" => d.set_item("git_dirty", true)?,
        "false" => d.set_item("git_dirty", false)?,
        _ => d.set_item("git_dirty", "unknown")?,
    }
    let tag = env!("REINFORS_GIT_TAG");
    d.set_item("git_tag", if tag.is_empty() { None } else { Some(tag) })?;
    d.set_item(
        "profile",
        if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        },
    )?;
    d.set_item("version", reinfors_core::version())?;
    Ok(d)
}

#[pymodule]
fn _reinfors(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(core_version, m)?)?;
    m.add_class::<PyEnvSnapshot>()?;
    m.add_class::<PyEngineSnapshot>()?;
    m.add_function(wrap_pyfunction!(chess_uci_action, m)?)?;
    m.add_function(wrap_pyfunction!(chess_action_uci, m)?)?;
    m.add_function(wrap_pyfunction!(build_info, m)?)?;
    m.add_class::<PyEngine>()?;
    m.add_class::<PyEnv>()?;
    m.add_class::<PyCfr>()?;
    m.add_class::<PyDeepCfr>()?;
    m.add_class::<DeepCfrBatch>()?;
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
    m.add_class::<PpoBatch>()?;
    m.add_class::<PyBox>()?;
    m.add_class::<PyDiscrete>()?;
    Ok(())
}
