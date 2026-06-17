//! The `Game` trait — the rules an environment exposes so the framework (search, rollout, training)
//! can drive it without knowing the game. The framework consumes a game through this trait only;
//! nothing here is game-specific. Concrete games (e.g. snake) live in the `reinfors-games` crate.

use crate::space::Space;

/// Minimal random source the rollout passes to a game's *realized* (non-belief) transitions, so the
/// game can sample environment chance (e.g. apple placement) from the engine's per-game PRNG.
pub trait Rng {
    fn below(&mut self, n: usize) -> usize;
    fn unit(&mut self) -> f64;
}

/// Who chooses at a node: one agent (a sequential turn), all agents at once (a simultaneous move), or
/// nature (a chance node). The game only declares the shape; the planner decides how to expand each.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Actor {
    Agent(usize),
    Simultaneous,
    Chance,
}

/// One transition's deterministic outcome: the resulting state, the per-agent reward vector, and
/// whether the game ended. Per-agent activeness is read from `legal_actions` being empty, not here.
pub struct Transition<S> {
    pub next_state: S,
    pub rewards: Vec<f64>,
    pub terminal: bool,
}

/// A finite-action, perfect-information game. Single-agent, sequential or simultaneous multi-agent,
/// and N-player general-sum are all expressible via `actor` + the per-agent reward vector.
pub trait Game {
    type State: Clone;

    fn num_agents(&self) -> usize;
    /// Net head width: the (homogeneous) per-agent action-space size.
    fn action_count(&self) -> usize;
    /// Observation tensor shape `(C, H, W)` the value network consumes.
    fn obs_shape(&self) -> (usize, usize, usize);

    /// The observation `Space` the value network consumes. Defaults to an unbounded `Box` of
    /// `obs_shape`; a game may override to advertise tighter bounds (e.g. one-hot planes in `[0, 1]`).
    fn observation_space(&self) -> Space {
        let (c, h, w) = self.obs_shape();
        Space::Box {
            shape: vec![c, h, w],
            low: f32::NEG_INFINITY,
            high: f32::INFINITY,
        }
    }
    /// The per-agent action `Space`. Defaults to `Discrete(action_count)`. Assumes homogeneous action
    /// spaces across agents (consistent with `action_count`, the single per-agent action count, and the
    /// net's uniform head width); heterogeneous per-agent actions would need a per-agent form here plus
    /// broader changes to `action_count` and the network heads.
    fn action_space(&self) -> Space {
        Space::Discrete {
            n: self.action_count(),
        }
    }

    /// Who acts at `state`.
    fn actor(&self, state: &Self::State) -> Actor;
    /// Action indices (into `0..action_count`) legal for `agent`; empty when the agent is out of play.
    fn legal_actions(&self, state: &Self::State, agent: usize) -> Vec<usize>;
    /// Apply a joint action (one index per agent) — the deterministic part of the transition, before
    /// any chance resolution. The entry for an agent with no legal moves is ignored.
    fn step(&self, state: &Self::State, actions: &[usize]) -> Transition<Self::State>;
    /// Sample `n` independent realizations of the transition's environment chance — the stochastic
    /// successors of `transition.next_state` (e.g. apple respawn), each drawn from `rng`. An **empty**
    /// result means the transition is deterministic (no chance node); callers then use
    /// `transition.next_state` directly. The default is deterministic.
    ///
    /// This is the single source of a game's chance dynamics: the rollout draws one realization (`n =
    /// 1`, via the default `step_env`) and the search draws `food_samples` to Monte-Carlo the chance
    /// node — both through this method, so the env and the search can never use different dynamics.
    /// Takes the source `state` so a game can derive what happened (e.g. how many apples were eaten) by
    /// comparing it to `transition.next_state`.
    fn sample_chance(
        &self,
        state: &Self::State,
        transition: &Transition<Self::State>,
        rng: &mut dyn Rng,
        n: usize,
    ) -> Vec<Self::State> {
        let _ = (state, transition, rng, n);
        Vec::new()
    }
    /// Egocentric observation for `agent`, a flat `[C*H*W]` f32 buffer.
    fn observe(&self, state: &Self::State, agent: usize) -> Vec<f32>;

    /// A fresh episode's initial state, drawing any initial chance (e.g. apple placement) from `rng`.
    fn initial_state(&self, rng: &mut dyn Rng) -> Self::State;
    /// The *realized* environment transition for the rollout: the deterministic `step` followed by one
    /// sampled chance realization (`sample_chance` with `n = 1`) — the true env step. The reward vector
    /// excludes any horizon-truncation bonus. Defined in terms of `step` + `sample_chance` so the
    /// rollout and the search share one chance model; games should not override it.
    fn step_env(
        &self,
        state: &Self::State,
        actions: &[usize],
        rng: &mut dyn Rng,
    ) -> Transition<Self::State> {
        let t = self.step(state, actions);
        let mut outcomes = self.sample_chance(state, &t, rng, 1);
        let next_state = if outcomes.is_empty() {
            t.next_state
        } else {
            outcomes.swap_remove(0)
        };
        Transition {
            next_state,
            rewards: t.rewards,
            terminal: t.terminal,
        }
    }
    /// Extra reward for `agent` on a horizon-truncation tick (reached alive at `max_ticks`), e.g.
    /// snake's survival bonus. Added by the rollout engine, which owns the horizon. Default: none.
    fn truncation_bonus(&self, state: &Self::State, agent: usize) -> f64 {
        let _ = (state, agent);
        0.0
    }
}
