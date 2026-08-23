//! State and action representation.

use crate::space::Space;

/// Per-agent bijection between game action ids and network-head indices. Both maps must be pure
/// functions of `(action, agent)`; the bijection check cannot detect state-dependent mappings.
pub trait ActionView: Send + Sync {
    fn head_index(&self, action: usize, agent: usize) -> usize {
        let _ = agent;
        action
    }

    /// Inverse of [`head_index`](Self::head_index).
    fn game_action(&self, head: usize, agent: usize) -> usize {
        let _ = agent;
        head
    }
}

/// Identity mapping between game and network action frames.
pub struct IdentityView;
impl ActionView for IdentityView {}

/// Materialize `perm[game_id] = head_index` and identify the identity fast path.
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

/// Per-agent head permutations, built once at engine construction (an encoder-lifetime
/// constant; rebuilding per call was the pre-scheduler wart).
pub struct PermTable {
    perms: Vec<(Vec<usize>, bool)>,
}

impl PermTable {
    pub fn build(view: &dyn ActionView, action_count: usize, num_agents: usize) -> Self {
        PermTable {
            perms: (0..num_agents)
                .map(|agent| head_permutation(view, action_count, agent))
                .collect(),
        }
    }

    /// `(game->head permutation, is_identity)` for `agent`.
    pub fn get(&self, agent: usize) -> (&[usize], bool) {
        let (perm, identity) = &self.perms[agent];
        (perm, *identity)
    }
}

/// Assert that each agent's action mapping is bijective and invertible.
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

    /// Flat channel-major observation for `agent`. Must be a pure function of
    /// `(state, agent)`: the engine reuses policy-encoded rows for training records
    /// (`RequestSink::push_root`) on that contract.
    fn encode(&self, state: &Self::State, agent: usize) -> Vec<f32>;

    /// Observation shape `(channels, height, width)`.
    fn obs_shape(&self) -> (usize, usize, usize);

    /// Observation space, unbounded by default.
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
    use crate::game::{Actor, Game, Transition};
    use crate::learners::dqn::Dqn;
    use crate::policies::modelfree::epsilon_greedy_q::EpsilonGreedyQ;
    use crate::reward::Reward;
    use crate::rollout::engine::{Engine, EngineParams};

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
                events: vec![None],
                terminal: true,
            }
        }
        fn initial_state(&self) -> St {
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
        }
        check_action_view(&Broken, 3, 1);
    }

    #[test]
    fn rotated_view_reroutes_selection_and_head_frame_records() {
        let run = |enc: Box<dyn StateEncoder<State = St>>| {
            let mut engine = Engine::new(
                OneShot,
                enc,
                Box::new(NoReward),
                EpsilonGreedyQ::new(1, 0.0),
                Dqn::new(1, 1.0, 1, 0.99),
                EngineParams {
                    n_games: 1,
                    seed: 0,
                    ..Default::default()
                },
            );
            let (records, _) =
                engine.collect(1, |_obs, n| (0..n).flat_map(|_| [10.0, 0.0, 5.0]).collect());
            records
        };

        let id = run(Box::new(IdEnc));
        assert_eq!(id[0].action, 0);
        assert_eq!(id[0].legal, vec![0, 1, 2]);

        let rot = run(Box::new(RotEnc));
        assert_eq!(rot[0].action, 0);
        assert_eq!(rot[0].legal, vec![1, 2, 0]);
        assert!(rot[0].next_legal.is_empty());
    }
}

#[cfg(test)]
mod dispatch_tests {
    use super::*;
    use crate::game::{Actor, Game, Transition};
    use crate::policies::modelfree::epsilon_greedy_q::EpsilonGreedyQ;
    use crate::reward::Reward;
    use crate::rollout::evaluator::Evaluator;
    use std::sync::atomic::{AtomicUsize, Ordering};

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
                events: vec![None],
                terminal: true,
            }
        }
        fn initial_state(&self) -> St {
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

    struct RotEnc;
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
        use crate::learners::dqn::Dqn;
        use crate::rollout::engine::{Engine, EngineParams};
        let mut engine = Engine::new(
            SparseShot,
            Box::new(RotEnc),
            Box::new(NoReward),
            EpsilonGreedyQ::new(1, 0.0),
            Dqn::new(1, 1.0, 1, 0.99),
            EngineParams {
                n_games: 1,
                seed: 0,
                ..Default::default()
            },
        );
        let (records, _) =
            engine.collect(1, |_obs, n| (0..n).flat_map(|_| [10.0, 0.0, 5.0]).collect());
        assert_eq!(records[0].action, 0);
        assert_eq!(records[0].legal, vec![1, 0]);
    }

    #[test]
    fn identity_path_pays_one_table_build_not_per_scalar_dispatch() {
        let enc = CountingEnc(AtomicUsize::new(0));
        let policy = EpsilonGreedyQ::new(2, 0.0);
        let mut infer = |_p: usize, _obs: Vec<f32>, n: usize| vec![0.0; n * 2 * 3];
        let mut eval = Evaluator::new(
            &mut infer,
            crate::rollout::evaluator::InferMode::Shared,
            None,
        );
        // The table is built once for the engine's lifetime, not per decision or per scalar.
        let perms = PermTable::build(&enc, 3, 1);
        let built = enc.0.load(Ordering::Relaxed);
        let decisions = vec![(St, vec![0]), (St, vec![0]), (St, vec![0]), (St, vec![0])];
        let mut rng = crate::rng::SplitMix64::new(0);
        let evals = crate::rollout::driver::drive_to_completion(
            &policy,
            &SparseShot,
            &enc,
            &NoReward,
            &perms,
            false,
            &decisions,
            &mut rng,
            &mut eval,
        );
        assert_eq!(evals.len(), 4);
        assert_eq!(
            enc.0.load(Ordering::Relaxed),
            built,
            "no rebuilds after construction"
        );
    }
}
