//! Optional Rust-native "standard" value nets (tch / libtorch) for reinfors.
//!
//! These satisfy the search's `infer` contract — `(obs [n×dim], n) -> values [n·K·A]` — entirely in
//! Rust, so `Engine::collect` runs the forward pass without the per-round Python callback. The core
//! stays model-agnostic: it never depends on this crate; the arbitrary-Python-callback path is primary.
//! Architectures mirror `scripts/train_example.py` so a torch-trained checkpoint loads unchanged and a
//! Rust/Python parity test is exact.
use tch::nn::{self, ConvConfig, Module};
use tch::{Device, Kind, Tensor};

/// A batched, per-head action-value network: a pooled `[n, C·H·W]` observation batch in, `[n·K·A]`
/// (K ensemble heads × A actions, row-major) out. The K heads supply the search's disagreement signal.
pub trait ValueNet {
    fn n_heads(&self) -> usize;
    fn n_actions(&self) -> usize;
    fn forward(&self, obs: &[f32], n: usize) -> Vec<f64>;
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

    pub fn var_store(&mut self) -> &mut nn::VarStore {
        &mut self.vs
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
            .reshape([n as i64, c, h, w]);
        let x = self.conv.forward(&x).relu().flatten(1, -1);
        to_values(self.head.forward(&x), n, self.n_heads, self.n_actions)
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

    pub fn var_store(&mut self) -> &mut nn::VarStore {
        &mut self.vs
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
        let x = self.l1.forward(&x).relu();
        to_values(self.head.forward(&x), n, self.n_heads, self.n_actions)
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
