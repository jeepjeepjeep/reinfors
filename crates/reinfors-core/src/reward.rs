//! Event-to-scalar reward mapping.

pub trait Reward: Send + Sync {
    type Event;

    /// Map one emitted event to a scalar reward.
    fn step_reward(&self, event: &Self::Event, agent: usize) -> f64;
}

/// Score one transition edge for an agent.
pub fn edge_reward<E>(reward: &dyn Reward<Event = E>, events: &[Option<E>], agent: usize) -> f64 {
    events[agent]
        .as_ref()
        .map_or(0.0, |e| reward.step_reward(e, agent))
}
