//! Model-free bootstrapped DQN — records half. Emits off-policy transitions `(s, a, r, s', done)`,
//! bootstrapped by the (Python) learner against a target net rather than episode-end z-mixed. Consumes
//! `QEvaluation` from `crate::policies::modelfree::epsilon_greedy_q`. Together these exercise the seam's two hardest cases: a
//! non-search evaluation and a transition record (a different shape from TreeStrap's targets).

use crate::encoder::ActionView;
use crate::game::Rng;
use crate::learner::{sample_mask, Learner, Step};
use crate::policies::modelfree::epsilon_greedy_q::QEvaluation;

/// One off-policy transition: `(obs, action, reward, next_obs, terminal, mask[K])`. The TD target is
/// computed in the (Python) learner from `next_obs`/`terminal` against a target net, so the engine
/// emits raw transitions rather than precomputed targets. `terminal` is true only at a real terminal
/// (a horizon truncation keeps it false, so the learner bootstraps from `next_obs`).
pub struct DqnRecord {
    pub obs: Vec<f32>,
    pub action: usize,
    pub reward: f64,
    pub next_obs: Vec<f32>,
    pub terminal: bool,
    pub mask: Vec<f32>,
    /// The legal action ids at `obs` (SPARSE — dense masks over wide action spaces dwarf the
    /// observations themselves; chess: ~35 ids vs a 4672-wide f32 row). Diagnostics/regularizers;
    /// the executed `action` is guaranteed legal already.
    pub legal: Vec<usize>,
    /// The legal action ids at `next_obs`, and the AUTHORITATIVE bootstrap signal: bootstrap the
    /// TD target from `max_a Q(s', a)` over exactly these ids iff the list is NON-EMPTY; an empty
    /// list (a terminal, or a truncation tail on an alternating game whose post-move view is
    /// opponent-to-move) means `target = r`, full stop. Consumers branch on emptiness — never
    /// multiply by `(1 - done)`, which meets a masked max's `-inf` as `0 * -inf = NaN`.
    pub next_legal: Vec<usize>,
}

/// Bootstrapped-DQN record production: one per-head-masked transition per step, no episode-end mixing.
pub struct Dqn {
    n_heads: usize,
    bootstrap_p: f64,
}

impl Dqn {
    pub fn new(n_heads: usize, bootstrap_p: f64) -> Self {
        Dqn {
            n_heads: n_heads.max(1),
            bootstrap_p,
        }
    }
}

impl Learner<QEvaluation> for Dqn {
    type Record = DqnRecord;

    fn needs_next_obs(&self) -> bool {
        true
    }

    fn eval_records(
        &self,
        _eval: &mut QEvaluation,
        _view: &dyn ActionView,
        _agent: usize,
        _rng: &mut dyn Rng,
    ) -> Vec<DqnRecord> {
        Vec::new() // nothing at decision time; transitions need the post-step s', formed at episode end
    }

    fn episode_records(
        &self,
        trajectory: &[Step<QEvaluation>],
        _tail: &[f64],
        view: &dyn ActionView,
        agent: usize,
        rng: &mut dyn Rng,
    ) -> Vec<DqnRecord> {
        // The record's action/legal/next_legal index the net's Q output in training, so they cross
        // into the head frame here (trajectory `Step`s hold game-frame ids).
        let to_head = |ids: &[usize]| ids.iter().map(|&x| view.head_index(x, agent)).collect();
        trajectory
            .iter()
            .enumerate()
            .map(|(t, s)| {
                // s' = the agent's NEXT DECISION STATE. For interior steps that is the next step
                // of its own trajectory — on strictly-alternating games the engine's per-tick
                // next_obs is the OPPONENT-to-move position (the agent has no legal actions
                // there), so bootstrapping needs the own-turn successor; on simultaneous/
                // single-agent games the two coincide exactly. The final step keeps the engine's
                // post-move view: terminal (all-zero mask, no bootstrap) or truncation — where,
                // on alternating games, the empty next-legal mask again means "do not bootstrap".
                let (next_obs, next_legal) = match trajectory.get(t + 1) {
                    Some(succ) => (succ.obs.clone(), to_head(&succ.evaluation.legal)),
                    None => (s.next_obs.clone(), to_head(&s.next_legal)),
                };
                DqnRecord {
                    obs: s.obs.clone(),
                    action: view.head_index(s.action, agent),
                    reward: s.reward,
                    next_obs,
                    terminal: s.terminal,
                    mask: sample_mask(rng, self.n_heads, self.bootstrap_p),
                    legal: to_head(&s.evaluation.legal),
                    next_legal: if s.terminal {
                        Vec::new() // no bootstrap at a terminal
                    } else {
                        next_legal
                    },
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoder::IdentityView;
    use crate::game::{Actor, Game, Transition};
    use crate::policies::modelfree::epsilon_greedy_q::EpsilonGreedyQ;
    use crate::rng::SplitMix64;
    use crate::rollout::engine::Engine;

    fn step(
        obs: Vec<f32>,
        action: usize,
        reward: f64,
        next_obs: Vec<f32>,
        terminal: bool,
    ) -> Step<QEvaluation> {
        Step {
            obs,
            evaluation: QEvaluation {
                values: vec![vec![0.0; 2]],
                legal: vec![0, 1],
            },
            action,
            reward,
            next_obs,
            next_legal: Vec::new(),
            terminal,
        }
    }

    #[test]
    fn episode_records_emit_one_masked_transition_per_step() {
        let learner = Dqn::new(3, 1.0); // bootstrap_p = 1 -> all-ones masks
        let traj = vec![
            step(vec![1.0], 0, 0.5, vec![2.0], false),
            step(vec![2.0], 1, -1.0, vec![], true), // terminal: next_obs unused
        ];
        let recs = learner.episode_records(&traj, &[], &IdentityView, 0, &mut SplitMix64::new(1));
        assert_eq!(recs.len(), 2);
        assert_eq!(recs[0].obs, vec![1.0]);
        assert_eq!(recs[0].action, 0);
        assert_eq!(recs[0].reward, 0.5);
        assert_eq!(recs[0].next_obs, vec![2.0]);
        assert!(!recs[0].terminal);
        assert_eq!(recs[0].mask, vec![1.0, 1.0, 1.0]);
        assert!(recs[1].terminal);
    }

    #[test]
    fn learner_declares_transition_needs() {
        let learner = Dqn::new(2, 0.5);
        assert!(learner.needs_next_obs());
        assert!(!learner.uses_episode_tail());
        assert!(learner
            .eval_records(
                &mut QEvaluation {
                    values: vec![vec![0.0; 2]],
                    legal: vec![0, 1],
                },
                &IdentityView,
                0,
                &mut SplitMix64::new(0)
            )
            .is_empty());
    }

    // Compile-only: a DQN policy + learner share QEvaluation, so they satisfy the engine coupling.
    fn _assert_seam_composes() -> Option<Engine<DummyGame, EpsilonGreedyQ, Dqn>> {
        None
    }

    #[allow(dead_code)] // used only as a type parameter in the compile-only assertion above
    struct DummyGame;
    impl Game for DummyGame {
        type State = ();
        type Event = ();
        fn num_agents(&self) -> usize {
            1
        }
        fn action_count(&self) -> usize {
            2
        }
        fn actor(&self, _: &()) -> Actor {
            Actor::Agent(0)
        }
        fn legal_actions(&self, _: &(), _: usize) -> Vec<usize> {
            vec![0, 1]
        }
        fn step(&self, _: &(), _: &[usize]) -> Transition<(), ()> {
            unimplemented!()
        }
        fn initial_state(&self, _: &mut dyn Rng) {}
    }
}
