//! The representation seam: a `StateEncoder` maps a game's native `State` into the flat observation
//! tensor a value network consumes, and (via its [`ActionView`] supertrait) fixes how the net's
//! action head is indexed. This is *representation*, deliberately split from `Game` (the *rules*):
//! a game can be trained or played under different encodings without touching its dynamics, and the
//! same native state can serve a net (encoded) and a human (raw) at once.
//!
//! Frames: rules speak game-frame action ids everywhere (trees, `Env`, trajectories); everything
//! net-facing — observations, policy targets, legality masks handed to training, reads of net
//! output rows — is in the encoder's frame. The `ActionView` bijection is applied at exactly those
//! crossings and nowhere else. For an absolute encoder the view is the identity and the frames
//! coincide (today's behavior, bit for bit); a mover-relative encoder overrides `encode` and the
//! view with the SAME symmetry transform, so observations and action indexing transform together.
//!
//! The rollout `Engine` and the search hold an encoder as `dyn StateEncoder` (it returns a concrete
//! tensor, so it is object-safe), making the representation swappable at run time without threading a
//! type parameter through the hot path. An encoder is keyed to one game's `State`.

use crate::space::Space;

/// The action side of a representation: a per-agent bijection between game-frame action ids and
/// the net's policy/Q head indices. State-free (unlike `StateEncoder`), so learners — which never
/// see the game's `State` type — can take it as `&dyn ActionView`.
///
/// Contract: for each `agent`, `head_index(·, agent)` is a bijection on `0..action_count` and
/// `game_action` is its exact inverse. The map must be a pure function of `(action, agent)` — a
/// state-dependent map could not be replayed consistently between target construction and
/// training. Defaults are the identity (an absolute encoder); every concrete encoder declares its
/// view explicitly (`impl ActionView for X {}` opts into the identity).
pub trait ActionView: Send + Sync {
    /// Net-head index of game action `action`, from `agent`'s perspective.
    fn head_index(&self, action: usize, agent: usize) -> usize {
        let _ = agent;
        action
    }

    /// Inverse of [`head_index`](Self::head_index). Tools and tests only — the hot paths iterate
    /// legal game ids and map inward, never enumerate head indices.
    fn game_action(&self, head: usize, agent: usize) -> usize {
        let _ = agent;
        head
    }
}

/// The identity action view — the absolute frame, where game ids and head indices coincide.
/// For call sites that need a view but have no encoder at hand (tests, tools).
pub struct IdentityView;
impl ActionView for IdentityView {}

/// Materialize `agent`'s permutation (`perm[game_id] = head_index`) plus an is-identity flag.
/// For DENSE per-action loops: one virtual call per action id here, then plain indexing (or a
/// straight copy on the identity fast path) instead of `K × A` dynamic dispatches in a hot loop.
/// Sparse legal-set gathers (a handful of ids) call `head_index` directly.
pub fn head_permutation(
    view: &dyn ActionView,
    action_count: usize,
    agent: usize,
) -> (Vec<usize>, bool) {
    let perm: Vec<usize> = (0..action_count)
        .map(|a| view.head_index(a, agent))
        .collect();
    let identity = perm.iter().enumerate().all(|(i, &p)| i == p);
    (perm, identity)
}

/// Assert the [`ActionView`] contract: `head_index(·, agent)` is a bijection on
/// `0..action_count` with `game_action` its inverse, for each agent. Call from an encoder's tests.
pub fn check_action_view(view: &dyn ActionView, action_count: usize, num_agents: usize) {
    for agent in 0..num_agents {
        let mut seen = vec![false; action_count];
        for a in 0..action_count {
            let h = view.head_index(a, agent);
            assert!(
                h < action_count,
                "head_index({a}, {agent}) = {h} out of range"
            );
            assert!(!seen[h], "head_index(·, {agent}) not injective at head {h}");
            seen[h] = true;
            assert_eq!(
                view.game_action(h, agent),
                a,
                "game_action is not the inverse of head_index at ({a}, {agent})"
            );
        }
    }
}

pub trait StateEncoder: ActionView {
    type State;

    /// The observation for `agent` as a flat `[C*H*W]` f32 buffer (row-major, channel-major).
    fn encode(&self, state: &Self::State, agent: usize) -> Vec<f32>;

    /// Observation tensor shape `(C, H, W)` — sizes the value network's input.
    fn obs_shape(&self) -> (usize, usize, usize);

    /// The observation `Space`. Defaults to an unbounded `Box` of `obs_shape`; an encoder may override
    /// to advertise tighter bounds (e.g. one-hot planes in `[0, 1]`).
    fn observation_space(&self) -> Space {
        let (c, h, w) = self.obs_shape();
        Space::Box {
            shape: vec![c, h, w],
            low: f32::NEG_INFINITY,
            high: f32::INFINITY,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{Engine, EngineParams};
    use crate::game::{Actor, Game, Rng, Transition};
    use crate::learners::dqn::Dqn;
    use crate::policies::epsilon_greedy_q::EpsilonGreedyQ;
    use crate::reward::Reward;

    /// Single-agent, 3 actions, always legal, terminal after one step. State is a tick counter so
    /// observations are well-defined; the game itself is inert scaffolding for the frame tests.
    struct OneShot;
    #[derive(Clone)]
    struct St(i32);
    impl Game for OneShot {
        type State = St;
        type Event = ();
        fn num_agents(&self) -> usize {
            1
        }
        fn action_count(&self) -> usize {
            3
        }
        fn actor(&self, _: &St) -> Actor {
            Actor::Agent(0)
        }
        fn legal_actions(&self, s: &St, _: usize) -> Vec<usize> {
            if s.0 > 0 {
                Vec::new()
            } else {
                vec![0, 1, 2]
            }
        }
        fn step(&self, s: &St, _: &[usize]) -> Transition<St, ()> {
            Transition {
                next_state: St(s.0 + 1),
                events: vec![()],
                terminal: true,
            }
        }
        fn initial_state(&self, _: &mut dyn Rng) -> St {
            St(0)
        }
    }
    struct NoReward;
    impl Reward for NoReward {
        type Event = ();
        fn step_reward(&self, _: &(), _: usize) -> f64 {
            0.0
        }
    }

    /// Observation is the raw state; the action view rotates: head = (game + 1) % 3.
    struct RotEnc;
    impl ActionView for RotEnc {
        fn head_index(&self, action: usize, _agent: usize) -> usize {
            (action + 1) % 3
        }
        fn game_action(&self, head: usize, _agent: usize) -> usize {
            (head + 2) % 3
        }
    }
    impl StateEncoder for RotEnc {
        type State = St;
        fn encode(&self, s: &St, _: usize) -> Vec<f32> {
            vec![s.0 as f32]
        }
        fn obs_shape(&self) -> (usize, usize, usize) {
            (1, 1, 1)
        }
    }

    struct IdEnc;
    impl ActionView for IdEnc {}
    impl StateEncoder for IdEnc {
        type State = St;
        fn encode(&self, s: &St, _: usize) -> Vec<f32> {
            vec![s.0 as f32]
        }
        fn obs_shape(&self) -> (usize, usize, usize) {
            (1, 1, 1)
        }
    }

    #[test]
    fn identity_and_rotation_satisfy_the_view_contract() {
        check_action_view(&IdentityView, 7, 2);
        check_action_view(&RotEnc, 3, 2);
    }

    #[test]
    #[should_panic(expected = "not the inverse")]
    fn check_action_view_catches_a_wrong_inverse() {
        struct Broken;
        impl ActionView for Broken {
            fn head_index(&self, action: usize, _: usize) -> usize {
                (action + 1) % 3
            }
            // game_action left at the identity default: not the inverse of the rotation.
        }
        check_action_view(&Broken, 3, 1);
    }

    /// The seam end to end: the same net row (HEAD frame [10, 0, 5]) drives one collect under the
    /// identity encoder and one under the rotated view. Selection must happen in the game frame —
    /// identity argmaxes game 0; under rotation game action 2 reads head slot 0's 10.0 — and the
    /// DQN record's net-facing fields (action, legal) must come out in the head frame.
    #[test]
    fn rotated_view_reroutes_selection_and_head_frame_records() {
        let run = |enc: Box<dyn StateEncoder<State = St>>| {
            let mut engine = Engine::new(
                OneShot,
                enc,
                Box::new(NoReward),
                EpsilonGreedyQ::new(1, 0.0), // greedy: pure argmax, no exploration
                Dqn::new(1, 1.0),
                EngineParams {
                    n_games: 1,
                    seed: 0,
                },
            );
            let (records, _) = engine.collect(1, |_obs, n| {
                (0..n).flat_map(|_| [10.0, 0.0, 5.0]).collect() // head-frame row per request
            });
            records
        };

        let id = run(Box::new(IdEnc));
        assert_eq!(id[0].action, 0); // game 0 = head 0 = 10.0, and the record stays at head 0
        assert_eq!(id[0].legal, vec![0, 1, 2]);

        let rot = run(Box::new(RotEnc));
        // Game-frame values under rotation: game a reads head (a+1)%3 -> [0.0, 5.0, 10.0].
        // Greedy select picks GAME action 2; its head index is (2+1)%3 = 0 — the slot holding 10.0,
        // so the net trains on exactly the slot whose value drove the choice.
        assert_eq!(rot[0].action, 0);
        assert_eq!(rot[0].legal, vec![1, 2, 0]); // game [0,1,2] mapped through the view
        assert!(rot[0].next_legal.is_empty()); // terminal: unchanged by the view
    }
}

#[cfg(test)]
mod dispatch_tests {
    //! Frame coverage the engine-level test above cannot see: sparse legal sets under a
    //! permutation, and the identity fast path's dispatch count (the latency-regression guard —
    //! wall-clock is flaky in unit tests, so we count the mechanism instead: virtual
    //! `head_index` calls must be O(A) table builds, never O(K x A x n) per-scalar gathers).
    use super::*;
    use crate::evaluator::Evaluator;
    use crate::game::{Actor, Game, Rng, Transition};
    use crate::policies::epsilon_greedy_q::EpsilonGreedyQ;
    use crate::policy::Policy;
    use crate::reward::Reward;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// A=3 with only {0, 2} legal.
    struct SparseShot;
    #[derive(Clone)]
    struct St;
    impl Game for SparseShot {
        type State = St;
        type Event = ();
        fn num_agents(&self) -> usize {
            1
        }
        fn action_count(&self) -> usize {
            3
        }
        fn actor(&self, _: &St) -> Actor {
            Actor::Agent(0)
        }
        fn legal_actions(&self, _: &St, _: usize) -> Vec<usize> {
            vec![0, 2]
        }
        fn step(&self, _: &St, _: &[usize]) -> Transition<St, ()> {
            Transition {
                next_state: St,
                events: vec![()],
                terminal: true,
            }
        }
        fn initial_state(&self, _: &mut dyn Rng) -> St {
            St
        }
    }
    struct NoReward;
    impl Reward for NoReward {
        type Event = ();
        fn step_reward(&self, _: &(), _: usize) -> f64 {
            0.0
        }
    }

    /// Identity-valued view that counts its virtual `head_index` calls.
    struct CountingEnc(AtomicUsize);
    impl ActionView for CountingEnc {
        fn head_index(&self, action: usize, _: usize) -> usize {
            self.0.fetch_add(1, Ordering::Relaxed);
            action
        }
    }
    impl StateEncoder for CountingEnc {
        type State = St;
        fn encode(&self, _: &St, _: usize) -> Vec<f32> {
            vec![0.0]
        }
        fn obs_shape(&self) -> (usize, usize, usize) {
            (1, 1, 1)
        }
    }

    struct RotEnc; // head = (game + 1) % 3
    impl ActionView for RotEnc {
        fn head_index(&self, action: usize, _: usize) -> usize {
            (action + 1) % 3
        }
        fn game_action(&self, head: usize, _: usize) -> usize {
            (head + 2) % 3
        }
    }
    impl StateEncoder for RotEnc {
        type State = St;
        fn encode(&self, _: &St, _: usize) -> Vec<f32> {
            vec![0.0]
        }
        fn obs_shape(&self) -> (usize, usize, usize) {
            (1, 1, 1)
        }
    }

    #[test]
    fn sparse_legal_sets_map_and_select_under_a_permutation() {
        use crate::engine::{Engine, EngineParams};
        use crate::learners::dqn::Dqn;
        let mut engine = Engine::new(
            SparseShot,
            Box::new(RotEnc),
            Box::new(NoReward),
            EpsilonGreedyQ::new(1, 0.0),
            Dqn::new(1, 1.0),
            EngineParams {
                n_games: 1,
                seed: 0,
            },
        );
        // Head-frame row [10, 0, 5] -> game frame [0, 5, 10]; legal = {0, 2}: argmax is game 2.
        let (records, _) =
            engine.collect(1, |_obs, n| (0..n).flat_map(|_| [10.0, 0.0, 5.0]).collect());
        assert_eq!(records[0].action, 0); // head_index(2) = 0 — the slot whose 10.0 won
        assert_eq!(records[0].legal, vec![1, 0]); // game {0, 2} through the view
    }

    #[test]
    fn identity_path_pays_one_table_build_not_per_scalar_dispatch() {
        let enc = CountingEnc(AtomicUsize::new(0));
        let policy = EpsilonGreedyQ::new(2, 0.0); // K = 2
        let mut infer = |_obs: Vec<f32>, n: usize| vec![0.0; n * 2 * 3];
        let mut eval = Evaluator::new(&mut infer, None);
        let requests = vec![(St, 0), (St, 0), (St, 0), (St, 0)]; // n = 4
        let evals = policy.evaluate(&SparseShot, &enc, &NoReward, requests, 0, false, &mut eval);
        assert_eq!(evals.len(), 4);
        // One permutation build (A = 3 probes), memoized across all requests and heads — a
        // regression to per-scalar gathers would read K x A x n = 24.
        assert_eq!(enc.0.load(Ordering::Relaxed), 3);
    }
}
