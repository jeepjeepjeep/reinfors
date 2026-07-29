//! `Episode` — one game's live, mutable slice: the current `State` and the env-chance RNG, plus the
//! single-game mechanics (step / observe / which agents act). Shared by the two rollout drivers —
//! the caller-driven [`Env`](crate::Env) (which holds one) and the autonomous [`Engine`](crate::Engine)
//! (which holds N). It runs against *borrowed* rules (`Game`) + encoder, so those can be shared across
//! many episodes (the `Engine` keeps one of each; an `Env` owns its own).

use crate::encoder::StateEncoder;
use crate::game::{realize_initial_state, step_env, Actor, Game};
use crate::rng::SplitMix64;

pub(crate) struct Episode<G: Game> {
    pub(crate) state: G::State,
    pub(crate) rng: SplitMix64,
}

impl<G: Game> Episode<G> {
    pub(crate) fn new(game: &G, seed: u64) -> Self {
        let mut rng = SplitMix64::new(seed);
        let state = realize_initial_state(game, &mut rng);
        Episode { state, rng }
    }

    pub(crate) fn reset(&mut self, game: &G) {
        self.state = realize_initial_state(game, &mut self.rng);
    }

    /// Restored start-distribution states must be realized DECISION states — a chance-node
    /// start from a custom distribution would leave every consumer actor-less (empty active
    /// set, a collect that gathers nothing) rather than fail loudly.
    pub(crate) fn assert_decision_state(game: &G, state: &G::State) {
        assert!(
            !matches!(game.actor(state), Actor::Chance),
            "an episode cannot start at a chance node; start distributions must restore \
             realized decision states"
        );
    }

    pub(crate) fn agent_active(&self, game: &G, agent: usize) -> bool {
        !game.legal_actions(&self.state, agent).is_empty()
    }

    pub(crate) fn active_agents(&self, game: &G) -> Vec<usize> {
        (0..game.num_agents())
            .filter(|&a| self.agent_active(game, a))
            .collect()
    }

    pub(crate) fn observe(
        &self,
        encoder: &dyn StateEncoder<State = G::State>,
        agent: usize,
    ) -> Vec<f32> {
        encoder.encode(&self.state, agent)
    }

    /// Advance through the env transition, updating `state`; returns this tick's `(per-agent events,
    /// terminal)`. A [`Reward`](crate::Reward) (held by the caller, not the `Episode`) maps the events
    /// to scalar rewards.
    pub(crate) fn advance(&mut self, game: &G, actions: &[usize]) -> (Vec<G::Event>, bool) {
        let t = step_env(game, &self.state, actions, &mut self.rng);
        self.state = t.next_state;
        (t.events, t.terminal)
    }
}
