//! `Episode` — one game's live, mutable slice: the current `State` and the env-chance RNG, plus the
//! single-game mechanics (step / observe / which agents act). Shared by the two rollout drivers —
//! the caller-driven [`Env`](crate::Env) (which holds one) and the autonomous [`Engine`](crate::Engine)
//! (which holds N). It runs against *borrowed* rules (`Game`) + encoder, so those can be shared across
//! many episodes (the `Engine` keeps one of each; an `Env` owns its own).

use crate::encoder::StateEncoder;
use crate::game::Game;
use crate::rng::SplitMix64;

pub(crate) struct Episode<G: Game> {
    pub(crate) state: G::State,
    pub(crate) rng: SplitMix64,
}

impl<G: Game> Episode<G> {
    /// A fresh episode, drawing initial chance from `seed`.
    pub(crate) fn new(game: &G, seed: u64) -> Self {
        let mut rng = SplitMix64::new(seed);
        let state = game.initial_state(&mut rng);
        Episode { state, rng }
    }

    /// Begin a new episode, drawing fresh initial chance from the continuing RNG stream.
    pub(crate) fn reset(&mut self, game: &G) {
        self.state = game.initial_state(&mut self.rng);
    }

    /// Whether `agent` has a move at the current state (the rollout reads activeness from this).
    pub(crate) fn agent_active(&self, game: &G, agent: usize) -> bool {
        !game.legal_actions(&self.state, agent).is_empty()
    }

    /// Agents that must act this tick: a single mover for a sequential game, all live agents for a
    /// simultaneous one. Empty once the episode is over.
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

    /// Advance through the env transition, updating `state`; returns this tick's `(per-agent rewards,
    /// terminal)`.
    pub(crate) fn advance(&mut self, game: &G, actions: &[usize]) -> (Vec<f64>, bool) {
        let t = game.step_env(&self.state, actions, &mut self.rng);
        self.state = t.next_state;
        (t.rewards, t.terminal)
    }
}
