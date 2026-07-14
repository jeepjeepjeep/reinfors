//! Optional Rust-native "standard" value nets on candle (pure Rust — no libtorch).
//!
//! These satisfy the search's `infer` contract — `(obs [n×dim], n) -> values [n·K·A]` — entirely in
//! Rust, so `Engine::collect` runs the forward pass without the per-round Python callback, and the whole
//! `engine.train` loop runs with no external C++ library and no Python. The core stays model-agnostic: it
//! never depends on this crate; the arbitrary-Python-callback path is primary. Architectures mirror
//! `scripts/train_example.py`. Device is chosen per-net at runtime (`resolve_device`); GPU backends
//! (`metal`/`cuda`) must be compiled in (`--features …`), then selectable without a rebuild. candle's
//! Metal backend needs macOS 15+.
//!
//! Performance caveat: candle's CPU `conv2d` is ~8-10x slower than PyTorch (open candle issue
//! <https://github.com/huggingface/candle/issues/3119> — im2col copy overhead dominates; a BLAS feature
//! like `accelerate`/`mkl` speeds only the matmul/linear path, not conv). So `Mlp` (and linear-heavy
//! nets) are fast on CPU, but `Conv` is conv-bound — use a GPU device for conv nets, or drive the search
//! with a Python net (torch's CPU conv uses oneDNN and is fast).
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{
    conv2d, linear, Conv2d, Conv2dConfig, Linear, Module, Optimizer, VarBuilder, VarMap,
};

/// "auto" — a GPU when built with (and able to open) that backend, else CPU.
fn default_device() -> Device {
    #[cfg(feature = "cuda")]
    if let Ok(d) = Device::new_cuda(0) {
        return d;
    }
    #[cfg(feature = "metal")]
    if let Ok(d) = Device::new_metal(0) {
        return d;
    }
    Device::Cpu
}

/// Select a device by name at RUNTIME — `"cpu"`, `"metal"`, `"cuda"`, or `"auto"`. The backend must have
/// been compiled in (`--features nn-metal`/`nn-cuda`, as the release wheels are); otherwise a requested
/// GPU returns a clear error rather than silently falling back. This is the torch-style split: the wheel
/// carries the backend, the caller picks the device per run without rebuilding.
pub fn resolve_device(name: &str) -> std::result::Result<Device, String> {
    match name {
        "cpu" => Ok(Device::Cpu),
        "auto" => Ok(default_device()),
        "metal" => {
            #[cfg(feature = "metal")]
            {
                Device::new_metal(0).map_err(|e| e.to_string())
            }
            #[cfg(not(feature = "metal"))]
            {
                Err("built without Metal support — install/build reinfors with the nn-metal feature".into())
            }
        }
        "cuda" => {
            #[cfg(feature = "cuda")]
            {
                Device::new_cuda(0).map_err(|e| e.to_string())
            }
            #[cfg(not(feature = "cuda"))]
            {
                Err(
                    "built without CUDA support — install/build reinfors with the nn-cuda feature"
                        .into(),
                )
            }
        }
        other => Err(format!(
            "unknown device '{other}' (expected cpu / metal / cuda / auto)"
        )),
    }
}

/// A batched, per-head action-value network: a pooled `[n, C·H·W]` observation batch in, `[n·K·A]`
/// (K ensemble heads × A actions, row-major) out. The K heads supply the search's disagreement signal.
pub trait ValueNet {
    fn n_heads(&self) -> usize;
    fn n_actions(&self) -> usize;
    /// Inference forward: flat `[n·dim]` obs in, `[n·K·A]` values out.
    fn forward(&self, obs: &[f32], n: usize) -> Vec<f64>;
    /// Autograd forward: an `[n, dim]` float tensor in, an `[n, K, A]` tensor out (candle tracks the
    /// graph, so a loss on the result backpropagates into the parameters). The training core; `forward`
    /// wraps this. Object-safe (concrete tensor types).
    fn forward_t(&self, x: &Tensor) -> Result<Tensor>;
    /// The `VarMap` holding the trainable parameters — a trainer builds an optimizer over its vars, and
    /// `import_weights` sets them by name.
    fn varmap(&self) -> &VarMap;
    /// Parameter names in a fixed order (matching the torch `state_dict` layout: trunk before head,
    /// weight before bias) — the order `export_weights` / `import_weights` use.
    fn param_names(&self) -> Vec<String>;
    fn device(&self) -> &Device;
    /// The mutable seam closure `Engine::collect` expects — a Rust-native alternative to a Python
    /// callable. `Sized` so the trait stays object-safe (a `dyn ValueNet` uses `forward` directly).
    fn infer_fn(&self) -> impl FnMut(Vec<f32>, usize) -> Vec<f64> + '_
    where
        Self: Sized,
    {
        move |obs, n| self.forward(&obs, n)
    }
}

/// Each parameter as `(shape, row-major f32 data)`, in `param_names()` order — for exporting weights into
/// a torch module / safetensors, or checkpointing.
pub fn export_weights(net: &dyn ValueNet) -> Vec<(Vec<i64>, Vec<f32>)> {
    let vars = net.varmap().data().lock().unwrap();
    net.param_names()
        .iter()
        .map(|name| {
            let t = vars[name].as_tensor();
            let shape = t.dims().iter().map(|&d| d as i64).collect();
            let flat = t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            (shape, flat)
        })
        .collect()
}

/// Copy `data` (in `param_names()` order, shapes matching) into the net's live parameters — the
/// torch→Rust weight-sync path: push a `state_dict`'s tensors in before a collect.
pub fn import_weights(net: &dyn ValueNet, data: &[Vec<f32>]) {
    let names = net.param_names();
    assert_eq!(
        names.len(),
        data.len(),
        "weight count mismatch: {} vs {}",
        names.len(),
        data.len()
    );
    let vars = net.varmap().data().lock().unwrap();
    for (name, d) in names.iter().zip(data) {
        let var = &vars[name];
        let shape = var.as_tensor().dims().to_vec();
        let t = Tensor::from_slice(d, shape.as_slice(), net.device()).unwrap();
        var.set(&t).unwrap();
    }
}

fn to_values(logits: &Tensor) -> Vec<f64> {
    logits
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
        .into_iter()
        .map(|v| v as f64)
        .collect()
}

/// Conv trunk + K linear heads — mirrors `ExampleNet` (Conv2d(c,16,3,pad=1) · ReLU · Flatten · Linear).
/// Covers the planar-observation games (snake, connect4, gridworld).
pub struct Conv {
    varmap: VarMap,
    conv: Conv2d,
    head: Linear,
    device: Device,
    shape: (usize, usize, usize),
    n_actions: usize,
    n_heads: usize,
}

impl Conv {
    pub fn new(obs_shape: (i64, i64, i64), n_actions: i64, n_heads: i64, device: Device) -> Self {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let (c, h, w) = (
            obs_shape.0 as usize,
            obs_shape.1 as usize,
            obs_shape.2 as usize,
        );
        let (a, k) = (n_actions as usize, n_heads as usize);
        let cfg = Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        let conv = conv2d(c, 16, 3, cfg, vb.pp("trunk_conv")).unwrap();
        let head = linear(16 * h * w, k * a, vb.pp("head")).unwrap();
        Self {
            varmap,
            conv,
            head,
            device,
            shape: (c, h, w),
            n_actions: a,
            n_heads: k,
        }
    }
}

impl ValueNet for Conv {
    fn n_heads(&self) -> usize {
        self.n_heads
    }
    fn n_actions(&self) -> usize {
        self.n_actions
    }
    fn forward(&self, obs: &[f32], n: usize) -> Vec<f64> {
        let (c, h, w) = self.shape;
        let x = Tensor::from_slice(obs, (n, c * h * w), &self.device).unwrap();
        to_values(&self.forward_t(&x).unwrap())
    }
    fn forward_t(&self, x: &Tensor) -> Result<Tensor> {
        let (c, h, w) = self.shape;
        let n = x.dim(0)?;
        let x = self.conv.forward(&x.reshape((n, c, h, w))?)?.relu()?;
        let x = self.head.forward(&x.flatten_from(1)?)?;
        x.reshape((n, self.n_heads, self.n_actions))
    }
    fn varmap(&self) -> &VarMap {
        &self.varmap
    }
    fn param_names(&self) -> Vec<String> {
        [
            "trunk_conv.weight",
            "trunk_conv.bias",
            "head.weight",
            "head.bias",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect()
    }
    fn device(&self) -> &Device {
        &self.device
    }
}

/// Two-layer MLP + K heads, for callers that hand the net a flattened observation vector.
pub struct Mlp {
    varmap: VarMap,
    l1: Linear,
    head: Linear,
    device: Device,
    in_dim: usize,
    n_actions: usize,
    n_heads: usize,
}

impl Mlp {
    pub fn new(in_dim: i64, hidden: i64, n_actions: i64, n_heads: i64, device: Device) -> Self {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let (in_dim, a, k) = (in_dim as usize, n_actions as usize, n_heads as usize);
        let l1 = linear(in_dim, hidden as usize, vb.pp("l1")).unwrap();
        let head = linear(hidden as usize, k * a, vb.pp("head")).unwrap();
        Self {
            varmap,
            l1,
            head,
            device,
            in_dim,
            n_actions: a,
            n_heads: k,
        }
    }
}

impl ValueNet for Mlp {
    fn n_heads(&self) -> usize {
        self.n_heads
    }
    fn n_actions(&self) -> usize {
        self.n_actions
    }
    fn forward(&self, obs: &[f32], n: usize) -> Vec<f64> {
        let x = Tensor::from_slice(obs, (n, self.in_dim), &self.device).unwrap();
        to_values(&self.forward_t(&x).unwrap())
    }
    fn forward_t(&self, x: &Tensor) -> Result<Tensor> {
        let n = x.dim(0)?;
        let x = self.l1.forward(&x.reshape((n, self.in_dim))?)?.relu()?;
        self.head
            .forward(&x)?
            .reshape((n, self.n_heads, self.n_actions))
    }
    fn varmap(&self) -> &VarMap {
        &self.varmap
    }
    fn param_names(&self) -> Vec<String> {
        ["l1.weight", "l1.bias", "head.weight", "head.bias"]
            .iter()
            .map(|s| s.to_string())
            .collect()
    }
    fn device(&self) -> &Device {
        &self.device
    }
}

/// Regresses TreeStrap's per-head backed-up value targets (masked Huber + Adam) — the training
/// counterpart to `forward`. One `update` is a single gradient step on a collected batch; the outer
/// loop (collect with live weights, then update) belongs to the caller or the fused driver. Synchronous
/// by design: every collect searches with the just-updated weights (no replay / stale-weight actor).
/// The optimizer is candle's `AdamW` with `weight_decay = 0`, i.e. plain Adam (candle ships no `Adam`),
/// matching the reference trainer's `torch.optim.Adam`.
pub struct TreeStrapTrainer {
    opt: candle_nn::AdamW,
}

impl TreeStrapTrainer {
    /// Adam (AdamW with weight_decay=0) over the net's parameters. `net` must be the same net later
    /// passed to `update` — the optimizer captures its `VarMap`'s vars, and `update`'s forward reads
    /// those same tensors.
    pub fn new(net: &dyn ValueNet, lr: f64) -> Result<Self> {
        // weight_decay pinned to 0 explicitly: AdamW == Adam only at wd=0, and we don't want a future
        // change to candle's ParamsAdamW default to silently introduce decay.
        let params = candle_nn::ParamsAdamW {
            lr,
            weight_decay: 0.0,
            ..Default::default()
        };
        Ok(Self {
            opt: candle_nn::AdamW::new(net.varmap().all_vars(), params)?,
        })
    }

    /// The masked-Huber loss tensor (candle tracks the graph). Row-major buffers: `obs` `[n·dim]`,
    /// `targets` `[n·K·A]`, `mask` `[n·K]` (per-head bootstrap). Mirrors the reference trainer:
    /// `(mask.unsqueeze(-1) * smooth_l1(pred, target)).sum() / (mask.sum().clamp_min(1) * A)`.
    pub fn loss(
        &self,
        net: &dyn ValueNet,
        obs: &[f32],
        targets: &[f32],
        mask: &[f32],
        n: i64,
    ) -> Tensor {
        self.try_loss(net, obs, targets, mask, n).expect("loss")
    }

    fn try_loss(
        &self,
        net: &dyn ValueNet,
        obs: &[f32],
        targets: &[f32],
        mask: &[f32],
        n: i64,
    ) -> Result<Tensor> {
        let (n, k, a) = (n as usize, net.n_heads(), net.n_actions());
        let dim = obs.len() / n;
        let dev = net.device();
        let obs_t = Tensor::from_slice(obs, (n, dim), dev)?;
        let tgt = Tensor::from_slice(targets, (n, k, a), dev)?;
        let mask = Tensor::from_slice(mask, (n, k), dev)?;
        let pred = net.forward_t(&obs_t)?;
        // smooth_l1 (beta=1): 0.5·d² where |d|<1, else |d|−0.5.
        let d = (pred - tgt)?;
        let abs = d.abs()?;
        let huber = abs
            .lt(1f64)?
            .where_cond(&(d.sqr()? * 0.5)?, &(&abs - 0.5)?)?;
        let masked = mask.unsqueeze(2)?.broadcast_mul(&huber)?;
        let denom = mask
            .sum_all()?
            .clamp(1f64, f64::INFINITY)?
            .affine(a as f64, 0.0)?;
        masked.sum_all()?.broadcast_div(&denom)
    }

    /// Backward + one Adam step on a precomputed `loss`; returns its scalar value.
    pub fn step(&mut self, loss: &Tensor) -> f64 {
        self.opt.backward_step(loss).expect("optimizer step");
        loss.to_scalar::<f32>().expect("scalar loss") as f64
    }

    /// Convenience `loss` + `step`, for Rust callers.
    pub fn update(
        &mut self,
        net: &dyn ValueNet,
        obs: &[f32],
        targets: &[f32],
        mask: &[f32],
        n: i64,
    ) -> f64 {
        let loss = self.loss(net, obs, targets, mask, n);
        self.step(&loss)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conv_forward_has_pooled_khead_shape() {
        let net = Conv::new((2, 6, 7), 7, 8, Device::Cpu); // connect4-ish: (C,H,W), A=7, K=8
        let out = net.forward(&vec![0.0; 3 * 2 * 6 * 7], 3);
        assert_eq!(out.len(), 3 * 8 * 7);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn mlp_forward_has_pooled_khead_shape() {
        let net = Mlp::new(42, 32, 7, 4, Device::Cpu);
        let out = net.forward(&vec![0.0; 5 * 42], 5);
        assert_eq!(out.len(), 5 * 4 * 7);
    }

    #[test]
    fn infer_fn_matches_forward() {
        let net = Conv::new((2, 6, 7), 7, 8, Device::Cpu);
        let obs = vec![0.25f32; 2 * 6 * 7];
        let mut infer = net.infer_fn();
        assert_eq!(infer(obs.clone(), 1), net.forward(&obs, 1));
    }

    #[test]
    fn import_then_export_round_trips_and_changes_forward() {
        let net = Conv::new((2, 6, 7), 7, 8, Device::Cpu);
        let obs = vec![0.5f32; 2 * 6 * 7];
        let before = net.forward(&obs, 1);
        let ramp: Vec<Vec<f32>> = export_weights(&net)
            .iter()
            .map(|(_, d)| (0..d.len()).map(|i| (i as f32) * 0.01).collect())
            .collect();
        import_weights(&net, &ramp);
        for ((_, got), want) in export_weights(&net).iter().zip(&ramp) {
            assert_eq!(got, want);
        }
        assert_ne!(net.forward(&obs, 1), before); // the update reached the live params
    }

    #[test]
    fn trainer_reduces_loss_and_moves_forward() {
        let net = Mlp::new(8, 16, 3, 2, Device::Cpu); // dim=8, K=2, A=3
        let mut tr = TreeStrapTrainer::new(&net, 1e-2).unwrap();
        let n = 4i64;
        let obs: Vec<f32> = (0..(n as usize * 8))
            .map(|i| (i as f32 * 0.017).sin())
            .collect();
        let targets = vec![0.5f32; n as usize * 2 * 3]; // constant targets the net can fit
        let mask = vec![1.0f32; n as usize * 2];
        let before_pred = net.forward(&obs, n as usize);
        let first = tr.update(&net, &obs, &targets, &mask, n);
        for _ in 0..100 {
            tr.update(&net, &obs, &targets, &mask, n);
        }
        let last = tr.update(&net, &obs, &targets, &mask, n);
        assert!(
            last < first * 0.5,
            "loss should fall with training: {first} -> {last}"
        );
        assert_ne!(net.forward(&obs, n as usize), before_pred); // steps reached the live params
    }
}
