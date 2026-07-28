//! The `Game` trait — the rules an environment exposes so the framework (search, rollout, training)
//! can drive it without knowing the game. The framework consumes a game through this trait only;
//! nothing here is game-specific. Concrete games (e.g. snake) live in the `reinfors-games` crate.

use crate::space::Space;

/// Minimal random source the rollout passes to a game's *realized* (non-belief) transitions.
pub trait Rng {
    fn below(&mut self, n: usize) -> usize;
    fn unit(&mut self) -> f64;
}

/// Who chooses at a node: one agent (a sequential turn), all agents at once (a simultaneous move), or
/// nature (a chance node).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Actor {
    Agent(usize),
    Simultaneous,
    Chance,
}

/// One transition's deterministic outcome: the resulting state, the per-agent `Event` (what happened
/// to each agent — a [`Reward`](crate::Reward) maps these to scalars), and whether the game ended.
pub struct Transition<S, E> {
    pub next_state: S,
    pub events: Vec<E>,
    pub terminal: bool,
}

/// A transition's declared chance distribution over its outcome indices. `Weighted` is the
/// general form (backgammon's 21 rolls). `Uniform(count)` declares a uniform distribution over
/// `count` outcomes in O(1) at ANY size — the outcome space can be combinatorial (snake's
/// k-apple respawn enumerates `P(free, k)` ordered placements) because sampling consumers draw
/// one index and decode it procedurally through `apply_chance`; only enumeration (`ExpandAll`)
/// ever pays per-outcome cost, and it bounds itself. `count` is the outcome-INDEX type
/// (`usize`, matching `apply_chance`) and must fit `min(2^53, usize::MAX)` — indices must
/// survive both an f64 mantissa (the uniform draw) and the platform word (validated by the
/// declaring game at construction AND against decoded state).
#[derive(Clone, Debug, PartialEq)]
pub enum ChanceDist {
    Weighted(Vec<f64>),
    Uniform(usize),
}

impl ChanceDist {
    pub fn count(&self) -> usize {
        match self {
            ChanceDist::Weighted(p) => p.len(),
            ChanceDist::Uniform(n) => *n,
        }
    }

    /// The normalized probabilities in outcome order, for exact enumeration. The weight total is
    /// computed ONCE up front — a per-outcome normalization would make an N-outcome fan
    /// quadratic in N (there is deliberately no `prob(i)` accessor for that reason).
    pub fn iter_probs(&self) -> impl Iterator<Item = f64> + '_ {
        let (total, uniform) = match self {
            ChanceDist::Weighted(p) => (Self::checked_total(p), 0.0),
            ChanceDist::Uniform(n) => (1.0, 1.0 / *n as f64),
        };
        (0..self.count()).map(move |i| match self {
            ChanceDist::Weighted(p) => p[i] / total,
            ChanceDist::Uniform(_) => uniform,
        })
    }

    /// The weight total, asserted finite and positive — weights that sum to `inf` (e.g. two
    /// `f64::MAX` entries), `0`, or `NaN` would make enumeration yield zero probabilities and
    /// sampling degenerate to the last index, silently. A real assert: silently-wrong chance is
    /// worse than a panic, and the check is one pass over a fan-bounded vector.
    fn checked_total(p: &[f64]) -> f64 {
        let total: f64 = p.iter().sum();
        assert!(
            total.is_finite() && total > 0.0,
            "Weighted chance needs a finite positive weight total, got {total}"
        );
        total
    }

    /// Draw one outcome index. Both arms consume exactly one `rng.unit()` — the uniform arm is
    /// the closed form of the weighted scan on an equal-weight vector, so seeded streams stay
    /// aligned with the vector-era realization (indices can differ only at float-tie
    /// boundaries, ~1e-13 per draw).
    pub fn draw(&self, rng: &mut dyn Rng) -> usize {
        match self {
            ChanceDist::Weighted(p) => {
                Self::checked_total(p);
                crate::rng::weighted_index(rng, p)
            }
            ChanceDist::Uniform(n) => {
                let u = rng.unit();
                (((*n as f64) * u) as usize).min(n.saturating_sub(1))
            }
        }
    }
}

/// A finite-action, perfect-information game. Single-agent, sequential or simultaneous multi-agent,
/// and N-player general-sum are all expressible via `actor` + the per-agent `Event`. The game owns
/// only dynamics + outcomes; turning an `Event` into a scalar reward is the [`Reward`](crate::Reward)'s
/// job, decoupled like the encoder.
pub trait Game {
    type State: Clone;

    /// The per-agent outcome of a tick (e.g. snake's `StepEvent`), consumed by a `Reward`.
    type Event;

    fn num_agents(&self) -> usize;

    fn action_count(&self) -> usize;

    fn action_space(&self) -> Space {
        Space::Discrete {
            n: self.action_count(),
        }
    }

    fn actor(&self, state: &Self::State) -> Actor;

    fn legal_actions(&self, state: &Self::State, agent: usize) -> Vec<usize>;

    fn step(&self, state: &Self::State, actions: &[usize]) -> Transition<Self::State, Self::Event>;

    /// The transition's chance distribution, *declared* (see [`ChanceDist`]): over the outcome
    /// indices that [`apply_chance`](Self::apply_chance) accepts. `None` means the transition is
    /// deterministic. This is the game's ONLY chance seam — there is no game-side sampler to
    /// diverge from it. The framework realizes env transitions from it ([`step_env`], one draw),
    /// and tree searches consume it per their configured [`ChanceMode`](crate::ChanceMode).
    /// Contract: `Weighted` probabilities are positive; `Uniform` counts are in `1..=2^53`;
    /// terminal transitions return `None`; and outcomes only vary the chance element — they share
    /// the transition's `terminal` flag, next actor, **and events: rewards are edge-level and
    /// outcome-invariant**. Searches score the action edge once from the pre-chance transition and
    /// share that reward across every outcome (snake: the eat reward is the same wherever the food
    /// respawns). A game whose chance element changes the reward — a stochastic payout — does not
    /// fit this seam; that is chance-as-a-player (`Actor::Chance`) territory, currently out of
    /// scope. The default declares every transition deterministic.
    fn chance_outcomes(
        &self,
        state: &Self::State,
        transition: &Transition<Self::State, Self::Event>,
    ) -> Option<ChanceDist> {
        let _ = (state, transition);
        None
    }

    /// Materialize outcome `outcome` (an index into the `chance_outcomes` probabilities) of the
    /// transition's chance distribution. Only called with indices of a `Some` distribution; games
    /// that declare no chance never see it.
    fn apply_chance(
        &self,
        state: &Self::State,
        transition: &Transition<Self::State, Self::Event>,
        outcome: usize,
    ) -> Self::State {
        let _ = (state, transition, outcome);
        unreachable!("apply_chance called on a game that declares no chance_outcomes")
    }

    fn initial_state(&self, rng: &mut dyn Rng) -> Self::State;

    /// The episode-length cap after which the rollout truncates a still-running game, or `None` for a
    /// game that always ends on its own (e.g. Connect-4). This is a property the game *declares* — the
    /// `Engine` does the tick-counting and enforces it, so the horizon never enters `State` or the
    /// search. Truncation is thus wholly a game concern (when *and*, via `mark_truncation`, what).
    fn truncation_horizon(&self) -> Option<usize> {
        None
    }

    /// Stamp the truncation outcome onto `events` when the rollout cuts the episode off at the horizon
    /// (the `Engine` calls this on that tick, before the reward evaluates the events). A game encodes
    /// "survived to the cutoff" here — e.g. snake flags its still-alive agents so their `Reward` pays
    /// the survival bonus. Default: no truncation-specific outcome.
    fn mark_truncation(&self, state: &Self::State, events: &mut [Self::Event]) {
        let _ = (state, events);
    }
}

/// Realize one environment transition: the deterministic [`Game::step`], then — for a transition
/// with declared chance — ONE draw from `chance_outcomes` materialized via `apply_chance`. The
/// framework realizes; the game only declares. This is the sole realization path (rollout engine
/// and `Env` both route through it), so the chance model the searches plan against and the one the
/// training trajectories are made of are the same object by construction — divergence is not
/// expressible. (The cost: realization materializes the probs vector; a game with a very large
/// outcome space pays that per stochastic tick — acceptable today, revisit if measured.)
pub fn step_env<G: Game>(
    game: &G,
    state: &G::State,
    actions: &[usize],
    rng: &mut dyn Rng,
) -> Transition<G::State, G::Event> {
    let t = game.step(state, actions);
    match game.chance_outcomes(state, &t) {
        None => t,
        Some(dist) => {
            let outcome = dist.draw(rng);
            Transition {
                next_state: game.apply_chance(state, &t, outcome),
                events: t.events,
                terminal: t.terminal,
            }
        }
    }
}

#[cfg(test)]
mod step_env_tests {
    use super::*;

    /// One action; chance outcomes {0: +10, 1: +20} at p = [0.25, 0.75] on every step.
    struct Chancy;
    impl Game for Chancy {
        type State = i32;
        type Event = ();
        fn num_agents(&self) -> usize {
            1
        }
        fn action_count(&self) -> usize {
            1
        }
        fn actor(&self, _: &i32) -> Actor {
            Actor::Agent(0)
        }
        fn legal_actions(&self, _: &i32, _: usize) -> Vec<usize> {
            vec![0]
        }
        fn step(&self, s: &i32, _: &[usize]) -> Transition<i32, ()> {
            Transition {
                next_state: *s + 1,
                events: vec![()],
                terminal: false,
            }
        }
        fn chance_outcomes(&self, _: &i32, _: &Transition<i32, ()>) -> Option<ChanceDist> {
            Some(ChanceDist::Weighted(vec![0.25, 0.75]))
        }
        fn apply_chance(&self, _: &i32, t: &Transition<i32, ()>, outcome: usize) -> i32 {
            t.next_state + if outcome == 0 { 10 } else { 20 }
        }
        fn initial_state(&self, _: &mut dyn Rng) -> i32 {
            0
        }
    }

    struct Unit(f64);
    impl Rng for Unit {
        fn below(&mut self, _: usize) -> usize {
            0
        }
        fn unit(&mut self) -> f64 {
            self.0
        }
    }

    #[test]
    fn realizes_the_declared_distribution() {
        // The framework's ONE derivation, pinned once here (not per game — games no longer have a
        // sampler to agree with): a unit draw below 0.25 lands on outcome 0, above on outcome 1,
        // and the realized state is exactly `apply_chance` of that outcome.
        let g = Chancy;
        let low = step_env(&g, &0, &[0], &mut Unit(0.1));
        assert_eq!(low.next_state, 11); // step +1, outcome 0 (+10)
        let high = step_env(&g, &0, &[0], &mut Unit(0.9));
        assert_eq!(high.next_state, 21); // step +1, outcome 1 (+20)
        assert!(!low.terminal);
    }

    #[test]
    fn deterministic_transitions_pass_through() {
        struct Det;
        impl Game for Det {
            type State = i32;
            type Event = ();
            fn num_agents(&self) -> usize {
                1
            }
            fn action_count(&self) -> usize {
                1
            }
            fn actor(&self, _: &i32) -> Actor {
                Actor::Agent(0)
            }
            fn legal_actions(&self, _: &i32, _: usize) -> Vec<usize> {
                vec![0]
            }
            fn step(&self, s: &i32, _: &[usize]) -> Transition<i32, ()> {
                Transition {
                    next_state: *s + 1,
                    events: vec![()],
                    terminal: false,
                }
            }
            fn initial_state(&self, _: &mut dyn Rng) -> i32 {
                0
            }
        }
        let t = step_env(&Det, &4, &[0], &mut Unit(0.5));
        assert_eq!(t.next_state, 5);
    }
}

#[cfg(test)]
mod chance_dist_tests {
    use super::*;

    #[test]
    #[should_panic(expected = "finite positive weight total")]
    fn infinite_weight_totals_panic_instead_of_degenerating() {
        let d = ChanceDist::Weighted(vec![f64::MAX, f64::MAX]); // sums to inf
        let _ = d.iter_probs().count();
    }

    #[test]
    #[should_panic(expected = "finite positive weight total")]
    fn zero_weight_totals_panic_on_draw() {
        struct R;
        impl Rng for R {
            fn below(&mut self, _n: usize) -> usize {
                0
            }
            fn unit(&mut self) -> f64 {
                0.5
            }
        }
        let d = ChanceDist::Weighted(vec![0.0, 0.0]);
        let _ = d.draw(&mut R);
    }

    #[test]
    fn iter_probs_normalizes_once_and_matches_both_arms() {
        // Weighted: normalized by the (single) total; Uniform: 1/count per outcome.
        let w = ChanceDist::Weighted(vec![1.0, 3.0]);
        let probs: Vec<f64> = w.iter_probs().collect();
        assert_eq!(probs, vec![0.25, 0.75]);
        let u = ChanceDist::Uniform(4);
        let probs: Vec<f64> = u.iter_probs().collect();
        assert_eq!(probs, vec![0.25; 4]);
        assert_eq!(u.count(), 4);
    }
}
