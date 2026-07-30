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

/// One EDGE's deterministic outcome: the resulting state, what this edge causally determined for
/// each agent (`None` — the common case — means the edge settled nothing for that agent), and
/// whether the game ended.
///
/// Events are INCREMENTAL, never cumulative: across an action→chance chain each outcome is
/// emitted exactly once, on the edge that decides it — snake's eat/death on the action edge (the
/// respawn draw settles nothing), hold'em's payout on the final runout edge (the call settles
/// nothing). No single edge owns the tick: the framework accumulates a tick's emissions in edge
/// order into a trace ([`Tick`]), and a [`Reward`](crate::Reward) maps each emitted event to a
/// scalar, summed per tick.
pub struct Transition<S, E> {
    pub next_state: S,
    pub events: Vec<Option<E>>,
    pub terminal: bool,
}

impl<S, E> Transition<S, E> {
    /// An edge that settles nothing: neutral events for every agent, game continues — the shape
    /// of interior chance edges (a deal, a reveal, a roll) and quiet decision edges alike.
    pub fn silent(next_state: S, num_agents: usize) -> Self {
        Transition {
            next_state,
            events: (0..num_agents).map(|_| None).collect(),
            terminal: false,
        }
    }
}

/// A chance state's declared distribution over its outcome indices. `Weighted` is the
/// general form (backgammon's 21 rolls). `Uniform(count)` declares a uniform distribution over
/// `count` outcomes in O(1) at ANY size — the outcome space can be combinatorial (snake's
/// k-apple respawn enumerates `P(free, k)` ordered placements) because sampling consumers draw
/// one index and decode it procedurally through `apply_chance_node`; only enumeration (`ExpandAll`)
/// ever pays per-outcome cost, and it bounds itself. `count` is the outcome-INDEX type
/// (`usize`, matching `apply_chance_node`) and must fit `min(2^53, usize::MAX)` — indices must
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

    /// What an EDGE causally decided for one agent (e.g. snake's `StepEvent`), consumed by a
    /// `Reward`; a tick's outcome is the ordered trace of its edges' emissions.
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

    /// Whether EVERY random element of the game is declared through chance nodes (root
    /// deals and interior states alike) — i.e. `initial_state` draws NOTHING from its rng. Solvers that enumerate chance (CFR) require
    /// this and verify the root claim by calling `initial_state` with an rng that panics on any
    /// draw; a game that samples privately would otherwise be solved against the wrong tree.
    /// Deliberate claim, default false.
    fn all_chance_declared(&self) -> bool {
        false
    }

    /// Whether this game provides information-set keys (below). Deliberate claim, default
    /// false; `true` obliges `information_state_key` to answer at every REALIZED state.
    fn information_states(&self) -> bool {
        false
    }

    /// A canonical byte key for agent `agent`'s INFORMATION SET at `state`: everything the
    /// agent knows — own private information, all public state, and the full action/reveal
    /// history (perfect recall) — and nothing it doesn't. Defined at every REALIZED state
    /// (decision and terminal, any agent); never queried at chance-node states, which are
    /// transient inside realization — implementations may panic on a partial deal.
    ///
    /// Contract: keys are equal iff the agent cannot distinguish the states. Regret-table
    /// correctness rests on what that implies, so it is worth spelling out — equal keys must
    /// have: the same agent to act, the same ORDERED legal-action list for the keyed agent —
    /// regrets and probabilities align by vector index, so equal sets in different orders are
    /// insufficient (the CFR solver debug-asserts list equality on every table revisit) — and,
    /// because the key carries the full history, perfect-recall equivalence. The encoder's observation carries the same
    /// information content (pinned by test), but the key is exact compact bytes where the
    /// observation is a lossy-by-design float tensor. Solvers index their tables by this key —
    /// which is what forces learned strategies to be measurable with respect to the player's
    /// information.
    fn information_state_key(&self, state: &Self::State, agent: usize) -> Vec<u8> {
        let _ = (state, agent);
        unreachable!("information_state_key called on a game that declares no information states")
    }

    /// The distribution at a chance-node state (`actor` returned [`Actor::Chance`]): over the
    /// outcome indices [`apply_chance_node`](Self::apply_chance_node) accepts. Declaration
    /// contract: `Weighted` probabilities are positive; `Uniform` counts are in `1..=2^53`.
    /// This is the game's ONLY chance seam — there is no game-side sampler to diverge from it.
    ///
    /// A chance state is where no agent decides and the framework draws to continue. Its
    /// realization MAY determine events and terminal status (poker's all-in runout: the river
    /// decides the showdown). Rollout consumers (`Env`, `Engine`) realize chains of them inside
    /// one tick; tree searches traverse them as fixed-probability plies per their configured
    /// [`ChanceMode`](crate::ChanceMode) — transparent to depth, discount, and perspective, with
    /// each edge's emissions joining the tick's reward. The ROOT is ordinary chance:
    /// `initial_state` may return a chance node (a declared deal), realized at episode birth by
    /// the same chain machinery; birth chains must reach a NON-terminal decision state
    /// (asserted) and emit no events (there is no tick to deliver them into).
    ///
    /// AUTHORING PATTERN (distilled from the backgammon and snake migrations): mark the pending
    /// nature operation with a cheap sentinel on the state (`dice == [0, 0]`, `pending_food > 0`)
    /// rather than a wrapper enum — `actor` branches on it, `legal_actions` returns empty there,
    /// `apply_chance_node` clears it (usually a [`Transition::silent`] — the outcome's effects
    /// ride the edges that causally decide them). Keep the sentinel out of observations (the
    /// framework never encodes chance states), and REJECT it in `validate_decoded_state` on live
    /// states: the pending state is transient inside a tick, and a restored env stuck awaiting
    /// nature could never be stepped.
    fn chance_node(&self, state: &Self::State) -> ChanceDist {
        let _ = state;
        unreachable!("chance_node called on a game that declares no chance nodes")
    }

    /// Realize outcome `outcome` at a chance-node state, completing (part of) the transition:
    /// the result may carry events and end the game. Within one tick the framework chains these
    /// until a decision state or terminal. Events are per-edge and CAUSAL like everywhere else
    /// ([`Transition`]): each chain edge emits exactly what its outcome decides — hold'em's
    /// final runout edge emits the payout, an interior reveal emits nothing — and the tick's
    /// trace accumulates every emission in edge order.
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

    /// Amend the tick's trace when the rollout cuts the episode off at the horizon (the `Engine`
    /// calls this on that tick, before rewards fold). A game encodes "survived to the cutoff" by
    /// mutating its agents' emitted events or pushing new `(agent, event)` entries — e.g. snake
    /// flags its still-alive agents so their `Reward` pays the survival bonus. Default: no
    /// truncation-specific outcome.
    fn mark_truncation(&self, state: &Self::State, trace: &mut Vec<(usize, Self::Event)>) {
        let _ = (state, trace);
    }
}

/// One realized environment TICK: an action edge plus any chance chain it entered, run to the
/// next decision state or terminal. `trace` is the tick's ordered `(agent, event)` emissions
/// across all of its edges — the tick owns the trace; no single edge owns the tick. Discount,
/// time, and decision counts advance once per tick regardless of chain length.
pub struct Tick<S, E> {
    pub next_state: S,
    pub trace: Vec<(usize, E)>,
    pub terminal: bool,
}

/// Backstop against a buggy game cycling through chance states forever.
pub const CHANCE_CHAIN_LIMIT: usize = 10_000;

fn push_edge_events<E>(trace: &mut Vec<(usize, E)>, events: Vec<Option<E>>, num_agents: usize) {
    // Position IS the agent id: a short vector would silently drop agents from the trace, a
    // long one would mint nonexistent ids that panic later in reward indexing.
    assert!(
        events.len() == num_agents,
        "a transition must carry exactly one event slot per agent ({} for {num_agents} agents)",
        events.len()
    );
    for (agent, e) in events.into_iter().enumerate() {
        if let Some(e) = e {
            trace.push((agent, e));
        }
    }
}

/// Realize one environment TICK: the deterministic [`Game::step`], then the chance-node chain
/// (one draw per chance state) run to the next decision state or terminal. The framework
/// realizes; the game only declares. This is the sole realization path (rollout engine and
/// `Env` both route through it), so the chance model the searches plan against and the one the
/// training trajectories are made of are the same object by construction — divergence is not
/// expressible.
pub fn step_env<G: Game>(
    game: &G,
    state: &G::State,
    actions: &[usize],
    rng: &mut dyn Rng,
) -> Tick<G::State, G::Event> {
    let mut t = game.step(state, actions);
    let num_agents = game.num_agents();
    let mut trace: Vec<(usize, G::Event)> = Vec::new();
    push_edge_events(&mut trace, std::mem::take(&mut t.events), num_agents);
    // Chance-node chain: while the state belongs to no agent, draw and realize. Runs to a
    // decision state or terminal within this tick, so
    // chain-interior states are never observable outside realization; each edge's emissions
    // accumulate into the tick's trace in edge order.
    let mut edges = 0usize;
    while !t.terminal && matches!(game.actor(&t.next_state), Actor::Chance) {
        edges += 1;
        assert!(
            edges <= CHANCE_CHAIN_LIMIT,
            "chance-node chain exceeded {CHANCE_CHAIN_LIMIT} edges — the game cycles through chance states"
        );
        let outcome = game.chance_node(&t.next_state).draw(rng);
        t = game.apply_chance_node(&t.next_state, outcome);
        push_edge_events(&mut trace, std::mem::take(&mut t.events), num_agents);
    }
    Tick {
        next_state: t.next_state,
        trace,
        terminal: t.terminal,
    }
}

/// Realize an episode's birth: `initial_state` may return a chance node (a declared deal —
/// see [`Game::all_chance_declared`]), possibly chaining; draw until the first decision
/// state. The single realization path for episode starts — used by the rollout runtime
/// (`Episode::new`/`reset`) and by any consumer that must probe POST-birth properties (the
/// binding's decision-dynamics probe: `actor` on an unrealized root is `Actor::Chance`, which
/// says nothing about how the game's agents take turns). Birth chains may not end the episode
/// (asserted) and their events are contractually neutral.
pub fn realize_initial_state<G: Game>(game: &G, rng: &mut dyn Rng) -> G::State {
    let mut state = game.initial_state(rng);
    let mut edges = 0usize;
    while matches!(game.actor(&state), Actor::Chance) {
        edges += 1;
        assert!(
            edges <= CHANCE_CHAIN_LIMIT,
            "chance-node chain exceeded {CHANCE_CHAIN_LIMIT} edges — the game cycles through chance states"
        );
        let outcome = game.chance_node(&state).draw(rng);
        let t = game.apply_chance_node(&state, outcome);
        assert!(
            !t.terminal,
            "an episode cannot end during its birth chain — the deal may not decide the game"
        );
        // A REAL assert: a rollout would silently discard a birth emission while a solver
        // scoring the same edge would not — release builds must reject the divergence too.
        assert!(
            t.events.iter().all(Option::is_none),
            "birth-chain edges emit no events — there is no tick to deliver them into"
        );
        state = t.next_state;
    }
    state
}

#[cfg(test)]
mod step_env_tests {
    use super::*;

    /// Action at tick 0, then a two-node chance chain (ticks 1, 2), terminal at 3 with the
    /// payout decided by the FINAL chain step.
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
                events: vec![Some(1.0)], // the action edge settles its own cost immediately
                terminal: false,
            }
        }
        fn chance_node(&self, _s: &i32) -> ChanceDist {
            ChanceDist::Uniform(2)
        }
        fn apply_chance_node(&self, s: &i32, outcome: usize) -> Transition<i32, f64> {
            let terminal = *s == 2;
            Transition {
                next_state: s + 1,
                // Interior chain edges settle nothing; the final draw decides the payout.
                events: vec![if terminal {
                    Some(10.0 + outcome as f64)
                } else {
                    None
                }],
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
        // Two chained draws consumed (0.25 -> outcome 0, 0.75 -> outcome 1). The trace holds
        // every emission in edge order: the action edge's own event, then the final chance
        // edge's payout — interior edges emitted nothing.
        assert_eq!(t.trace, vec![(0, 1.0), (0, 11.0)]);
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

    /// `chain_len` chance edges after the action, then a decision state; `emit_at_birth`
    /// makes `initial_state` return the chain head instead (a birth chain).
    struct Chains {
        chain_len: usize,
        from_birth: bool,
        emit: bool,
    }
    impl Game for Chains {
        type State = i32;
        type Event = f64;
        fn num_agents(&self) -> usize {
            1
        }
        fn action_count(&self) -> usize {
            1
        }
        fn actor(&self, s: &i32) -> Actor {
            if (*s as usize) < self.chain_len {
                Actor::Chance
            } else {
                Actor::Agent(0)
            }
        }
        fn legal_actions(&self, s: &i32, _agent: usize) -> Vec<usize> {
            if (*s as usize) >= self.chain_len {
                vec![0]
            } else {
                Vec::new()
            }
        }
        fn step(&self, _s: &i32, _actions: &[usize]) -> Transition<i32, f64> {
            Transition {
                next_state: 0,
                events: vec![None],
                terminal: false,
            }
        }
        fn chance_node(&self, _s: &i32) -> ChanceDist {
            ChanceDist::Uniform(1)
        }
        fn apply_chance_node(&self, s: &i32, _outcome: usize) -> Transition<i32, f64> {
            Transition {
                next_state: s + 1,
                events: vec![if self.emit { Some(1.0) } else { None }],
                terminal: false,
            }
        }
        fn initial_state(&self, _rng: &mut dyn Rng) -> i32 {
            if self.from_birth {
                0
            } else {
                self.chain_len as i32
            }
        }
    }

    #[test]
    fn chain_limit_allows_exactly_the_cap() {
        let g = Chains {
            chain_len: CHANCE_CHAIN_LIMIT,
            from_birth: false,
            emit: false,
        };
        let t = step_env(&g, &(g.chain_len as i32), &[0], &mut Unit(0.5));
        assert_eq!(t.next_state as usize, CHANCE_CHAIN_LIMIT);
        let born = realize_initial_state(
            &Chains {
                chain_len: CHANCE_CHAIN_LIMIT,
                from_birth: true,
                emit: false,
            },
            &mut Unit(0.5),
        );
        assert_eq!(born as usize, CHANCE_CHAIN_LIMIT);
    }

    #[test]
    #[should_panic(expected = "chance-node chain exceeded")]
    fn chain_limit_rejects_one_past_the_cap() {
        let g = Chains {
            chain_len: CHANCE_CHAIN_LIMIT + 1,
            from_birth: false,
            emit: false,
        };
        let _ = step_env(&g, &(g.chain_len as i32), &[0], &mut Unit(0.5));
    }

    #[test]
    #[should_panic(expected = "chance-node chain exceeded")]
    fn a_cyclic_birth_chain_panics_instead_of_hanging() {
        let _ = realize_initial_state(
            &Chains {
                chain_len: usize::MAX,
                from_birth: true,
                emit: false,
            },
            &mut Unit(0.5),
        );
    }

    #[test]
    #[should_panic(expected = "birth-chain edges emit no events")]
    fn a_birth_emission_is_rejected_in_every_build() {
        let _ = realize_initial_state(
            &Chains {
                chain_len: 1,
                from_birth: true,
                emit: true,
            },
            &mut Unit(0.5),
        );
    }

    #[test]
    #[should_panic(expected = "exactly one event slot per agent")]
    fn a_short_event_vector_is_rejected() {
        struct Short;
        impl Game for Short {
            type State = i32;
            type Event = f64;
            fn num_agents(&self) -> usize {
                2
            }
            fn action_count(&self) -> usize {
                1
            }
            fn actor(&self, _: &i32) -> Actor {
                Actor::Simultaneous
            }
            fn legal_actions(&self, _: &i32, _: usize) -> Vec<usize> {
                vec![0]
            }
            fn step(&self, _: &i32, _: &[usize]) -> Transition<i32, f64> {
                Transition {
                    next_state: 1,
                    events: vec![Some(1.0)], // one slot for two agents
                    terminal: true,
                }
            }
            fn initial_state(&self, _: &mut dyn Rng) -> i32 {
                0
            }
        }
        let _ = step_env(&Short, &0, &[0, 0], &mut Unit(0.5));
    }

    #[test]
    fn realizes_the_declared_distribution() {
        // The framework's ONE derivation, pinned once here (not per game — games have no
        // sampler to agree with): a unit draw below 0.25 lands on outcome 0, above on outcome 1,
        // and the realized state is exactly `apply_chance_node` of that outcome.
        struct Fanny;
        impl Game for Fanny {
            type State = i32;
            type Event = ();
            fn num_agents(&self) -> usize {
                1
            }
            fn action_count(&self) -> usize {
                1
            }
            fn actor(&self, s: &i32) -> Actor {
                if *s == 1 {
                    Actor::Chance
                } else {
                    Actor::Agent(0)
                }
            }
            fn legal_actions(&self, s: &i32, _: usize) -> Vec<usize> {
                if *s == 1 {
                    Vec::new()
                } else {
                    vec![0]
                }
            }
            fn step(&self, _: &i32, _: &[usize]) -> Transition<i32, ()> {
                Transition {
                    next_state: 1,
                    events: vec![None],
                    terminal: false,
                }
            }
            fn chance_node(&self, _: &i32) -> ChanceDist {
                ChanceDist::Weighted(vec![0.25, 0.75])
            }
            fn apply_chance_node(&self, _: &i32, outcome: usize) -> Transition<i32, ()> {
                Transition::silent(10 + 10 * outcome as i32, 1)
            }
            fn initial_state(&self, _: &mut dyn Rng) -> i32 {
                0
            }
        }
        let low = step_env(&Fanny, &0, &[0], &mut Unit(0.1));
        assert_eq!(low.next_state, 10); // outcome 0
        let high = step_env(&Fanny, &0, &[0], &mut Unit(0.9));
        assert_eq!(high.next_state, 20); // outcome 1
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
                    events: vec![None],
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
