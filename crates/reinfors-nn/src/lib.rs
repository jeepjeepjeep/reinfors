//! Optional Rust-native "standard" value nets (tch / libtorch) for reinfors.
//!
//! These satisfy the search's `infer` contract — `(obs [n×dim], n) -> values [n·K·A]` — entirely in
//! Rust, so `Engine::collect` runs the forward pass without the per-round Python callback. The core
//! stays model-agnostic: it never depends on this crate; the arbitrary-Python-callback path is primary.
//! Architectures mirror `scripts/train_example.py` so a torch-trained checkpoint loads unchanged and a
//! Rust/Python parity test is exact.
use tch::nn::{self, ConvConfig, Module, OptimizerConfig};
use tch::{Device, Kind, Tensor};

/// A batched, per-head action-value network: a pooled `[n, C·H·W]` observation batch in, `[n·K·A]`
/// (K ensemble heads × A actions, row-major) out. The K heads supply the search's disagreement signal.
pub trait ValueNet {
    fn n_heads(&self) -> usize;
    fn n_actions(&self) -> usize;
    /// Inference forward: flat `[n·dim]` obs in, `[n·K·A]` values out (no autograd — see `forward_t`).
    fn forward(&self, obs: &[f32], n: usize) -> Vec<f64>;
    /// Autograd forward: an `[n, dim]` float tensor in, an `[n, K, A]` tensor out, with the graph
    /// retained so a loss backpropagates into the parameters. The training core; `forward` is just this
    /// under a no-grad guard. Object-safe (concrete tensor types).
    fn forward_t(&self, x: &Tensor) -> Tensor;
    /// The net's `VarStore`, so a trainer can build an optimizer over exactly these parameters.
    fn var_store(&self) -> &nn::VarStore;
    /// Trainable parameters in a fixed order, as shallow views sharing storage with the net — so
    /// `import_weights` copies straight into the live parameters and `forward` sees the update.
    fn params(&self) -> Vec<Tensor>;
    /// The mutable seam closure `Engine::collect` expects — a Rust-native alternative to a Python
    /// callable. `Sized` so the trait stays object-safe (a `dyn ValueNet` uses `forward` directly).
    fn infer_fn(&self) -> impl FnMut(Vec<f32>, usize) -> Vec<f64> + '_
    where
        Self: Sized,
    {
        move |obs, n| self.forward(&obs, n)
    }
}

/// Each parameter as `(shape, row-major f32 data)`, in `params()` order — for exporting Rust-initialised
/// weights into a torch module (parity test) or checkpointing.
pub fn export_weights(net: &dyn ValueNet) -> Vec<(Vec<i64>, Vec<f32>)> {
    net.params()
        .iter()
        .map(|p| {
            let flat = Vec::<f32>::try_from(p.to_kind(Kind::Float).flatten(0, -1))
                .expect("f32 param buffer");
            (p.size(), flat)
        })
        .collect()
}

/// Copy `data` (in `params()` order, shapes matching) into the net's live parameters. This is the
/// torch→Rust weight-sync path: push a `state_dict`'s tensors in before a collect.
pub fn import_weights(net: &dyn ValueNet, data: &[Vec<f32>]) {
    let _no_grad = tch::no_grad_guard();
    let params = net.params();
    assert_eq!(
        params.len(),
        data.len(),
        "weight count mismatch: {} vs {}",
        params.len(),
        data.len()
    );
    for (mut p, d) in params.into_iter().zip(data) {
        let src = Tensor::from_slice(d).to_kind(p.kind()).reshape(p.size());
        p.copy_(&src); // in place on the shared storage -> updates the live parameter
    }
}

fn to_values(logits: Tensor, n: usize, k: i64, a: i64) -> Vec<f64> {
    let out = logits
        .reshape([n as i64, k, a])
        .to_kind(Kind::Double)
        .flatten(0, -1);
    Vec::<f64>::try_from(out).expect("contiguous f64 value buffer")
}

/// Conv trunk + K linear heads — mirrors `ExampleNet` (Conv2d(c,16,3,pad=1) · ReLU · Flatten · Linear).
/// Covers the planar-observation games (snake, connect4, gridworld).
pub struct Conv {
    vs: nn::VarStore,
    conv: nn::Conv2D,
    head: nn::Linear,
    shape: (i64, i64, i64),
    n_actions: i64,
    n_heads: i64,
}

impl Conv {
    pub fn new(obs_shape: (i64, i64, i64), n_actions: i64, n_heads: i64) -> Self {
        let vs = nn::VarStore::new(Device::Cpu);
        let root = vs.root();
        let (c, h, w) = obs_shape;
        let conv = nn::conv2d(
            &root / "trunk_conv",
            c,
            16,
            3,
            ConvConfig {
                padding: 1,
                ..Default::default()
            },
        );
        let head = nn::linear(
            &root / "head",
            16 * h * w,
            n_heads * n_actions,
            Default::default(),
        );
        Self {
            vs,
            conv,
            head,
            shape: obs_shape,
            n_actions,
            n_heads,
        }
    }
}

impl ValueNet for Conv {
    fn n_heads(&self) -> usize {
        self.n_heads as usize
    }
    fn n_actions(&self) -> usize {
        self.n_actions as usize
    }
    fn forward(&self, obs: &[f32], n: usize) -> Vec<f64> {
        let _no_grad = tch::no_grad_guard();
        let (c, h, w) = self.shape;
        let x = Tensor::from_slice(obs)
            .to_kind(Kind::Float)
            .reshape([n as i64, c * h * w]);
        to_values(self.forward_t(&x), n, self.n_heads, self.n_actions)
    }
    fn forward_t(&self, x: &Tensor) -> Tensor {
        let (c, h, w) = self.shape;
        let x = self
            .conv
            .forward(&x.reshape([-1, c, h, w]))
            .relu()
            .flatten(1, -1);
        self.head
            .forward(&x)
            .reshape([-1, self.n_heads, self.n_actions])
    }
    fn var_store(&self) -> &nn::VarStore {
        &self.vs
    }
    // Order mirrors ExampleNet's state_dict: trunk conv weight/bias, then head weight/bias.
    fn params(&self) -> Vec<Tensor> {
        vec![
            self.conv.ws.shallow_clone(),
            self.conv.bs.as_ref().unwrap().shallow_clone(),
            self.head.ws.shallow_clone(),
            self.head.bs.as_ref().unwrap().shallow_clone(),
        ]
    }
}

/// Two-layer MLP + K heads, for callers that hand the net a flattened observation vector.
pub struct Mlp {
    vs: nn::VarStore,
    l1: nn::Linear,
    head: nn::Linear,
    in_dim: i64,
    n_actions: i64,
    n_heads: i64,
}

impl Mlp {
    pub fn new(in_dim: i64, hidden: i64, n_actions: i64, n_heads: i64) -> Self {
        let vs = nn::VarStore::new(Device::Cpu);
        let root = vs.root();
        let l1 = nn::linear(&root / "l1", in_dim, hidden, Default::default());
        let head = nn::linear(
            &root / "head",
            hidden,
            n_heads * n_actions,
            Default::default(),
        );
        Self {
            vs,
            l1,
            head,
            in_dim,
            n_actions,
            n_heads,
        }
    }
}

impl ValueNet for Mlp {
    fn n_heads(&self) -> usize {
        self.n_heads as usize
    }
    fn n_actions(&self) -> usize {
        self.n_actions as usize
    }
    fn forward(&self, obs: &[f32], n: usize) -> Vec<f64> {
        let _no_grad = tch::no_grad_guard();
        let x = Tensor::from_slice(obs)
            .to_kind(Kind::Float)
            .reshape([n as i64, self.in_dim]);
        to_values(self.forward_t(&x), n, self.n_heads, self.n_actions)
    }
    fn forward_t(&self, x: &Tensor) -> Tensor {
        let x = self.l1.forward(&x.reshape([-1, self.in_dim])).relu();
        self.head
            .forward(&x)
            .reshape([-1, self.n_heads, self.n_actions])
    }
    fn var_store(&self) -> &nn::VarStore {
        &self.vs
    }
    fn params(&self) -> Vec<Tensor> {
        vec![
            self.l1.ws.shallow_clone(),
            self.l1.bs.as_ref().unwrap().shallow_clone(),
            self.head.ws.shallow_clone(),
            self.head.bs.as_ref().unwrap().shallow_clone(),
        ]
    }
}

/// Regresses TreeStrap's per-head backed-up value targets (masked Huber + Adam) — the training
/// counterpart to `forward`. One `update` is a single gradient step on a collected batch; the outer
/// loop (collect with live weights, then update) belongs to the caller or the fused driver. Synchronous
/// by design: every collect searches with the just-updated weights (no replay / stale-weight actor).
pub struct TreeStrapTrainer {
    opt: nn::Optimizer,
}

impl TreeStrapTrainer {
    /// Adam over the net's parameters. `net` must be the same net later passed to `update` — the
    /// optimizer captures its `VarStore`'s variables, and `update`'s forward reads those same tensors.
    pub fn new(net: &dyn ValueNet, lr: f64) -> Result<Self, tch::TchError> {
        Ok(Self {
            opt: nn::Adam::default().build(net.var_store(), lr)?,
        })
    }

    /// The masked-Huber loss tensor (grad graph attached; no backward). Row-major buffers: `obs`
    /// `[n·dim]`, `targets` `[n·K·A]`, `mask` `[n·K]` (per-head bootstrap). Mirrors the reference trainer:
    /// `(mask.unsqueeze(-1) * smooth_l1(pred, target)).sum() / (mask.sum().clamp_min(1) * A)`. Split from
    /// `step` so a Python caller can release the GIL around the (autograd) backward — torch's Python
    /// autograd engine panics if `backward` runs with the GIL held.
    pub fn loss(
        &self,
        net: &dyn ValueNet,
        obs: &[f32],
        targets: &[f32],
        mask: &[f32],
        n: i64,
    ) -> Tensor {
        let (k, a) = (net.n_heads() as i64, net.n_actions() as i64);
        let dim = obs.len() as i64 / n;
        let obs_t = Tensor::from_slice(obs)
            .to_kind(Kind::Float)
            .reshape([n, dim]);
        let tgt_t = Tensor::from_slice(targets)
            .to_kind(Kind::Float)
            .reshape([n, k, a]);
        let mask_t = Tensor::from_slice(mask)
            .to_kind(Kind::Float)
            .reshape([n, k]);
        let huber = net
            .forward_t(&obs_t)
            .smooth_l1_loss(&tgt_t, tch::Reduction::None, 1.0);
        (mask_t.unsqueeze(-1) * huber).sum(Kind::Float)
            / (mask_t.sum(Kind::Float).clamp_min(1.0) * (a as f64))
    }

    /// Backward + one Adam step on a precomputed `loss`; returns its scalar value. Expensive (runs
    /// autograd) — a Python caller should release the GIL around this.
    pub fn step(&mut self, loss: &Tensor) -> f64 {
        self.opt.backward_step(loss);
        loss.double_value(&[])
    }

    /// Convenience `loss` + `step`, for Rust callers with no GIL to manage. The Python binding splits
    /// them to release the GIL around the backward.
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
        let net = Conv::new((2, 6, 7), 7, 8); // connect4-ish: (C,H,W), A=7, K=8
        let out = net.forward(&vec![0.0; 3 * 2 * 6 * 7], 3);
        assert_eq!(out.len(), 3 * 8 * 7);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn mlp_forward_has_pooled_khead_shape() {
        let net = Mlp::new(42, 32, 7, 4);
        let out = net.forward(&vec![0.0; 5 * 42], 5);
        assert_eq!(out.len(), 5 * 4 * 7);
    }

    #[test]
    fn infer_fn_matches_forward() {
        let net = Conv::new((2, 6, 7), 7, 8);
        let obs = vec![0.25f32; 2 * 6 * 7];
        let mut infer = net.infer_fn();
        assert_eq!(infer(obs.clone(), 1), net.forward(&obs, 1));
    }

    #[test]
    fn trainer_reduces_loss_and_moves_forward() {
        let net = Mlp::new(8, 16, 3, 2); // dim=8, K=2, A=3
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

    #[test]
    fn import_then_export_round_trips_and_changes_forward() {
        let net = Conv::new((2, 6, 7), 7, 8);
        let obs = vec![0.5f32; 2 * 6 * 7];
        let before = net.forward(&obs, 1);
        // Set every parameter to a fixed ramp, then read it back — export must equal what we imported.
        let ramp: Vec<Vec<f32>> = export_weights(&net)
            .iter()
            .map(|(_, d)| (0..d.len()).map(|i| (i as f32) * 0.01).collect())
            .collect();
        import_weights(&net, &ramp);
        for ((_, got), want) in export_weights(&net).iter().zip(&ramp) {
            assert_eq!(got, want);
        }
        assert_ne!(net.forward(&obs, 1), before); // the update actually reached the live params
    }
}
