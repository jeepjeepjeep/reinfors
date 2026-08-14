//! Game dynamics and chance realization.

use crate::rng::weighted_index;
use crate::space::Space;

/// Random source used by realized transitions.
pub trait Rng {
    fn below(&mut self, n: usize) -> usize;
    fn unit(&mut self) -> f64;
}

/// Who chooses at a node.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Actor {
    Agent(usize),
    Simultaneous,
    Chance,
}

/// One edge's resulting state, incremental events, and terminal status.
pub struct Transition<S, E> {
    pub next_state: S,
    pub events: Vec<Option<E>>,
    pub terminal: bool,
}

impl<S, E> Transition<S, E> {
    /// A non-terminal edge with no emitted events.
    pub fn silent(next_state: S, num_agents: usize) -> Self {
        Transition {
            next_state,
            events: (0..num_agents).map(|_| None).collect(),
            terminal: false,
        }
    }
}

/// A chance state's distribution over outcome indices. Uniform counts must fit
/// `min(2^53, usize::MAX)` so sampling cannot silently lose index precision.
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

    /// Normalized probabilities in outcome order. Normalization is computed once for the whole
    /// iterator; a `prob(i)` accessor would make callers accidentally build O(N^2) fans.
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

    fn checked_total(p: &[f64]) -> f64 {
        let total: f64 = p.iter().sum();
        assert!(
            total.is_finite() && total > 0.0,
            "Weighted chance needs a finite positive weight total, got {total}"
        );
        total
    }

    /// Draw one outcome using exactly one uniform random value.
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

/// Finite-action game dynamics.
pub trait Game {
    type State: Clone;

    /// One edge's incremental outcome for an agent.
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

    /// Whether the game provides information-state keys.
    fn information_states(&self) -> bool {
        false
    }

    /// Canonical perfect-recall information-set key for `agent` at a realized state.
    /// Equal keys must have the same actor and ordered legal-action list.
    fn information_state_key(&self, state: &Self::State, agent: usize) -> Vec<u8> {
        let _ = (state, agent);
        unreachable!("information_state_key called on a game that declares no information states")
    }

    /// Distribution over the outcome indices accepted by `apply_chance_node`.
    ///
    /// Pending chance is normally a transient state sentinel: `actor` exposes `Chance`, agent
    /// legality is empty, and `apply_chance_node` clears it. Keep it out of observations, and reject
    /// it when validating a live restored state, which otherwise cannot be stepped by an agent.
    fn chance_node(&self, state: &Self::State) -> ChanceDist {
        let _ = state;
        unreachable!("chance_node called on a game that declares no chance nodes")
    }

    /// Realize one outcome at a chance-node state.
    fn apply_chance_node(
        &self,
        state: &Self::State,
        outcome: usize,
    ) -> Transition<Self::State, Self::Event> {
        let _ = (state, outcome);
        unreachable!("apply_chance_node called on a game that declares no chance nodes")
    }

    /// Whether each agent's observation contains the full game state.
    fn perfect_information(&self) -> bool {
        true
    }

    /// Deterministic root state; random openings begin with a chance node.
    fn initial_state(&self) -> Self::State;

    /// Episode-length cap, or `None` for no horizon.
    fn truncation_horizon(&self) -> Option<usize> {
        None
    }

    /// Amend the final tick's events when the episode is truncated.
    fn mark_truncation(&self, state: &Self::State, trace: &mut Vec<(usize, Self::Event)>) {
        let _ = (state, trace);
    }
}

/// One action and its chance chain, realized to the next decision or terminal state.
pub struct Tick<S, E> {
    pub next_state: S,
    pub trace: Vec<(usize, E)>,
    pub terminal: bool,
}

pub const CHANCE_CHAIN_LIMIT: usize = 10_000;

fn push_edge_events<E>(trace: &mut Vec<(usize, E)>, events: Vec<Option<E>>, num_agents: usize) {
    // Event position is the agent id, so arity must be exact.
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

/// Realize an action and its chance chain as one environment tick.
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

/// Realize root chance nodes until the first decision state.
pub fn realize_initial_state<G: Game>(game: &G, rng: &mut dyn Rng) -> G::State {
    let mut state = game.initial_state();
    let num_agents = game.num_agents();
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
        assert!(
            t.events.len() == num_agents,
            "a transition must carry exactly one event slot per agent ({} for {num_agents} agents)",
            t.events.len()
        );
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
                events: vec![Some(1.0)],
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
                events: vec![if terminal {
                    Some(10.0 + outcome as f64)
                } else {
                    None
                }],
                terminal,
            }
        }
        fn initial_state(&self) -> i32 {
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
        fn initial_state(&self) -> i32 {
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

    struct BirthArity {
        len: usize,
    }
    impl Game for BirthArity {
        type State = i32;
        type Event = f64;
        fn num_agents(&self) -> usize {
            2
        }
        fn action_count(&self) -> usize {
            1
        }
        fn actor(&self, s: &i32) -> Actor {
            if *s == 0 {
                Actor::Chance
            } else {
                Actor::Agent(0)
            }
        }
        fn legal_actions(&self, s: &i32, agent: usize) -> Vec<usize> {
            if *s == 1 && agent == 0 {
                vec![0]
            } else {
                Vec::new()
            }
        }
        fn step(&self, _s: &i32, _actions: &[usize]) -> Transition<i32, f64> {
            Transition {
                next_state: 2,
                events: vec![None, None],
                terminal: true,
            }
        }
        fn chance_node(&self, _s: &i32) -> ChanceDist {
            ChanceDist::Uniform(1)
        }
        fn apply_chance_node(&self, _s: &i32, _outcome: usize) -> Transition<i32, f64> {
            Transition {
                next_state: 1,
                events: vec![None; self.len],
                terminal: false,
            }
        }
        fn initial_state(&self) -> i32 {
            0
        }
    }

    #[test]
    #[should_panic(expected = "exactly one event slot per agent")]
    fn a_short_birth_event_vector_is_rejected() {
        let _ = realize_initial_state(&BirthArity { len: 0 }, &mut Unit(0.5));
    }

    #[test]
    #[should_panic(expected = "exactly one event slot per agent")]
    fn a_long_birth_event_vector_is_rejected() {
        let _ = realize_initial_state(&BirthArity { len: 3 }, &mut Unit(0.5));
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
                    events: vec![Some(1.0)],
                    terminal: true,
                }
            }
            fn initial_state(&self) -> i32 {
                0
            }
        }
        let _ = step_env(&Short, &0, &[0, 0], &mut Unit(0.5));
    }

    #[test]
    fn realizes_the_declared_distribution() {
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
            fn initial_state(&self) -> i32 {
                0
            }
        }
        let low = step_env(&Fanny, &0, &[0], &mut Unit(0.1));
        assert_eq!(low.next_state, 10);
        let high = step_env(&Fanny, &0, &[0], &mut Unit(0.9));
        assert_eq!(high.next_state, 20);
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
            fn initial_state(&self) -> i32 {
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
        let d = ChanceDist::Weighted(vec![f64::MAX, f64::MAX]);
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
        let w = ChanceDist::Weighted(vec![1.0, 3.0]);
        let probs: Vec<f64> = w.iter_probs().collect();
        assert_eq!(probs, vec![0.25, 0.75]);
        let u = ChanceDist::Uniform(4);
        let probs: Vec<f64> = u.iter_probs().collect();
        assert_eq!(probs, vec![0.25; 4]);
        assert_eq!(u.count(), 4);
    }
}
