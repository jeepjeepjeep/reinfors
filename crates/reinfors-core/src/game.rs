//! The `Game` trait — the rules an environment exposes so the framework (search, rollout, training)
//! can drive it without knowing the game. The framework consumes a game through this trait only;
//! nothing here is game-specific. Concrete games (e.g. snake) live in the `reinfors-games` crate.

use crate::rng::weighted_index;
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
                weighted_index(rng, p)
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
    /// fit this seam; that is a chance NODE (`Actor::Chance` + [`chance_node`](Self::chance_node)),
    /// realized by rollout consumers and rejected by tree search. The default declares every
    /// transition deterministic.
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

    /// Whether chance-node states — states whose `actor` is [`Actor::Chance`], where no agent
    /// decides and the framework must draw from [`chance_node`](Self::chance_node) to continue —
    /// occur AFTER episode birth. Unlike transition-attached `chance_outcomes`, a chance node's
    /// realization MAY determine events and terminal status (poker's all-in runout: the river
    /// decides the showdown). Rollout consumers (`Env`, `Engine`) realize chains of them
    /// automatically inside one tick; tree searches reject games where they occur post-birth —
    /// scoring an outcome-dependent payout needs an explicit chance ply the searches do not
    /// implement.
    ///
    /// The ROOT is separate: `initial_state` may itself return a chance node (a declared deal —
    /// see [`all_chance_declared`](Self::all_chance_declared)), realized at episode birth by the
    /// same chain machinery; birth chains must reach a NON-terminal decision state (asserted —
    /// an episode over before any decision stays inexpressible), and any events they emit are
    /// contractually neutral (there is no tick to deliver them into). A game whose ONLY chance
    /// nodes are at the root returns `false` here — searches receive post-birth states and
    /// never meet them. `true` obliges `chance_node`/`apply_chance_node` to answer at every
    /// `Actor::Chance` state.
    fn chance_nodes(&self) -> bool {
        false
    }

    /// Whether EVERY random element of the game is declared through the chance seams (root
    /// chance nodes for the deal, `chance_outcomes`/chance nodes thereafter) — i.e.
    /// `initial_state` draws NOTHING from its rng. Solvers that enumerate chance (CFR) require
    /// this and verify the root claim by calling `initial_state` with an rng that panics on any
    /// draw; a game that samples privately would otherwise be solved against the wrong tree.
    /// Deliberate claim, default false.
    fn all_chance_declared(&self) -> bool {
        false
    }

    /// Whether this game provides information-set keys (below). Deliberate claim, default
    /// false; `true` obliges `information_state_key` to answer at every state.
    fn information_states(&self) -> bool {
        false
    }

    /// A canonical byte key for agent `agent`'s INFORMATION SET at `state`: everything the
    /// agent knows — own private information, all public state, and the full action/reveal
    /// history (perfect recall) — and nothing it doesn't. Contract: keys are equal iff the
    /// agent cannot distinguish the states; the encoder's observation carries the same
    /// information content (pinned by test), but the key is exact compact bytes where the
    /// observation is a lossy-by-design float tensor. Solvers index their tables by this key —
    /// which is what forces learned strategies to be measurable with respect to the player's
    /// information.
    fn information_state_key(&self, state: &Self::State, agent: usize) -> Vec<u8> {
        let _ = (state, agent);
        unreachable!("information_state_key called on a game that declares no information states")
    }

    /// The distribution at a chance-node state (`actor` returned [`Actor::Chance`]): over the
    /// outcome indices [`apply_chance_node`](Self::apply_chance_node) accepts. Same declaration
    /// contract as [`chance_outcomes`](Self::chance_outcomes) (positive weights, `Uniform` counts
    /// in `1..=2^53`).
    fn chance_node(&self, state: &Self::State) -> ChanceDist {
        let _ = state;
        unreachable!("chance_node called on a game that declares no chance nodes")
    }

    /// Realize outcome `outcome` at a chance-node state, completing (part of) the transition:
    /// the result may carry events and end the game. Within one tick the framework chains these
    /// until a decision state or terminal; the FINAL transition of the chain owns the tick's
    /// events — a transition INTO a chance node must therefore carry no outcome of its own
    /// (neutral events), and intermediate chain steps likewise.
    fn apply_chance_node(
        &self,
        state: &Self::State,
        outcome: usize,
    ) -> Transition<Self::State, Self::Event> {
        let _ = (state, outcome);
        unreachable!("apply_chance_node called on a game that declares no chance nodes")
    }

    /// Whether every agent could reconstruct the full state from its own observations (perfect
    /// information). Games with HIDDEN state (poker's hole cards) return false: the tree
    /// searches branch on the true state, so their backed-up values are clairvoyant about
    /// information the nets never see — search policies reject such games at construction
    /// (sound imperfect-information search is a different algorithm family). Observation-only
    /// policies (the DQN family) are unaffected.
    fn perfect_information(&self) -> bool {
        true
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
    let mut t = match game.chance_outcomes(state, &t) {
        None => t,
        Some(dist) => {
            let outcome = dist.draw(rng);
            Transition {
                next_state: game.apply_chance(state, &t, outcome),
                events: t.events,
                terminal: t.terminal,
            }
        }
    };
    // Chance-node chain: while the state belongs to no agent, draw and realize (see
    // `Game::chance_nodes`). Runs to a decision state or terminal within this tick, so
    // chain-interior states are never observable outside realization; the chain's final
    // transition owns the tick's events.
    while !t.terminal && matches!(game.actor(&t.next_state), Actor::Chance) {
        let outcome = game.chance_node(&t.next_state).draw(rng);
        t = game.apply_chance_node(&t.next_state, outcome);
    }
    t
}

#[cfg(test)]
mod step_env_tests {
    use super::*;

    /// Action at tick 0, then a two-node chance chain (ticks 1, 2), terminal at 3 with the
    /// payout decided by the FINAL chain step — the outcome-dependent shape transition chance
    /// cannot express.
    struct Chainy;
    impl Game for Chainy {
        type State = i32;
        type Event = f64;
        fn num_agents(&self) -> usize {
            1
        }
        fn action_count(&self) -> usize {
            1
        }
        fn actor(&self, s: &i32) -> Actor {
            if (1..=2).contains(s) {
                Actor::Chance
            } else {
                Actor::Agent(0)
            }
        }
        fn legal_actions(&self, s: &i32, _agent: usize) -> Vec<usize> {
            if *s == 0 {
                vec![0]
            } else {
                Vec::new()
            }
        }
        fn step(&self, s: &i32, _actions: &[usize]) -> Transition<i32, f64> {
            assert_eq!(*s, 0, "only the root offers a decision");
            Transition {
                next_state: 1,
                events: vec![0.0],
                terminal: false,
            }
        }
        fn chance_nodes(&self) -> bool {
            true
        }
        fn chance_node(&self, _s: &i32) -> ChanceDist {
            ChanceDist::Uniform(2)
        }
        fn apply_chance_node(&self, s: &i32, outcome: usize) -> Transition<i32, f64> {
            let terminal = *s == 2;
            Transition {
                next_state: s + 1,
                // Only the final chain step carries the tick's outcome.
                events: vec![if terminal { 10.0 + outcome as f64 } else { 0.0 }],
                terminal,
            }
        }
        fn initial_state(&self, _rng: &mut dyn Rng) -> i32 {
            0
        }
    }

    #[test]
    fn chance_node_chains_realize_within_one_step() {
        struct Half(u32);
        impl Rng for Half {
            fn below(&mut self, _n: usize) -> usize {
                0
            }
            fn unit(&mut self) -> f64 {
                self.0 += 1;
                if self.0.is_multiple_of(2) {
                    0.75
                } else {
                    0.25
                }
            }
        }
        let mut rng = Half(0);
        let t = step_env(&Chainy, &0, &[0], &mut rng);
        assert!(t.terminal, "the chain runs to terminal inside the tick");
        assert_eq!(t.next_state, 3);
        // Two chained draws consumed (0.25 -> outcome 0, 0.75 -> outcome 1); the final
        // realization owns the events.
        assert_eq!(t.events, vec![11.0]);
    }

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
