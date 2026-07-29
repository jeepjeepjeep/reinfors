//! The framework's PRNG. One splitmix64 implementation, shared by the rollout engine (per-game
//! environment chance, Thompson-head, epsilon draws) and the search (per-search chance sampling), so
//! every stochastic draw in the system comes from the same generator — keeping runs reproducible from
//! a seed without pulling in an RNG dependency.

use crate::game::Rng;

/// Tiny deterministic PRNG (splitmix64).
#[derive(Clone, PartialEq, Eq, Debug)]
pub(crate) struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub(crate) fn new(seed: u64) -> Self {
        SplitMix64 { state: seed }
    }

    /// The raw generator state — with `from_state`, the snapshot/restore seam (an exact resume
    /// point, unlike `new`, whose argument is a seed about to be advanced).
    pub(crate) fn state(&self) -> u64 {
        self.state
    }

    pub(crate) fn from_state(state: u64) -> Self {
        SplitMix64 { state }
    }

    pub(crate) fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// Draw an index proportional to `probs` (one `unit()` draw; numeric fallback lands on the last
/// positive-mass entry). Shared by every search's chance-outcome sampling.
pub(crate) fn weighted_index(rng: &mut dyn Rng, probs: &[f64]) -> usize {
    let total: f64 = probs.iter().sum();
    let mut r = rng.unit() * total;
    let mut last = 0;
    for (i, &p) in probs.iter().enumerate() {
        if p > 0.0 {
            last = i;
            r -= p;
            if r <= 0.0 {
                return i;
            }
        }
    }
    last
}

impl Rng for SplitMix64 {
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Standard normal via Box–Muller over `unit()` draws (the trait's only continuous primitive).
fn normal(rng: &mut dyn Rng) -> f64 {
    let u1 = rng.unit().max(f64::MIN_POSITIVE); // unit() ∈ [0,1); keep ln finite
    let u2 = rng.unit();
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

/// Gamma(alpha, 1) via Marsaglia–Tsang, with the alpha<1 boost `Gamma(a) = Gamma(a+1)·U^(1/a)`.
fn gamma_sample(rng: &mut dyn Rng, alpha: f64) -> f64 {
    if alpha < 1.0 {
        let u = rng.unit().max(f64::MIN_POSITIVE);
        return gamma_sample(rng, alpha + 1.0) * u.powf(1.0 / alpha);
    }
    let d = alpha - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        let x = normal(rng);
        let v = (1.0 + c * x).powi(3);
        if v <= 0.0 {
            continue;
        }
        let u = rng.unit().max(f64::MIN_POSITIVE);
        if u < 1.0 - 0.0331 * x.powi(4) || u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) {
            return d * v;
        }
    }
}

/// A symmetric Dirichlet(alpha) draw of dimension `k` — normalized Gamma(alpha) draws. AlphaZero's
/// root-noise distribution.
pub(crate) fn dirichlet(rng: &mut dyn Rng, alpha: f64, k: usize) -> Vec<f64> {
    let draws: Vec<f64> = (0..k).map(|_| gamma_sample(rng, alpha)).collect();
    let total: f64 = draws.iter().sum();
    if total <= 0.0 {
        return vec![1.0 / k as f64; k]; // all-zero draws (vanishingly rare): fall back to uniform
    }
    draws.into_iter().map(|d| d / total).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dirichlet_is_a_distribution_and_deterministic() {
        let mut rng = SplitMix64::new(42);
        let d = dirichlet(&mut rng, 0.3, 7);
        assert_eq!(d.len(), 7);
        assert!(d.iter().all(|&x| x >= 0.0));
        assert!((d.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        let mut rng2 = SplitMix64::new(42);
        assert_eq!(d, dirichlet(&mut rng2, 0.3, 7)); // same seed, same draw
    }

    #[test]
    fn dirichlet_mean_is_uniform_for_symmetric_alpha() {
        // E[Dir(α)_i] = 1/k regardless of α; check the empirical mean over many draws.
        let mut rng = SplitMix64::new(7);
        let k = 4;
        let mut mean = vec![0.0; k];
        let n = 2000;
        for _ in 0..n {
            for (m, x) in mean.iter_mut().zip(dirichlet(&mut rng, 0.3, k)) {
                *m += x / n as f64;
            }
        }
        for m in mean {
            assert!((m - 0.25).abs() < 0.02, "mean component {m} far from 1/k");
        }
    }

    #[test]
    fn small_alpha_concentrates_mass() {
        // α << 1 puts most mass on one coordinate per draw — the property root noise relies on.
        // Reference (numpy, α=0.05, k=7): P(max > 0.5) ≈ 0.96.
        let mut rng = SplitMix64::new(3);
        let mut peaked = 0;
        for _ in 0..200 {
            let d = dirichlet(&mut rng, 0.05, 7);
            if d.iter().cloned().fold(0.0, f64::max) > 0.5 {
                peaked += 1;
            }
        }
        assert!(peaked > 170, "only {peaked}/200 draws were peaked");
    }
}
