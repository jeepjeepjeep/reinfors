//! Reward shaping, ported to match `snake_RL`'s `MinimalReward`.
//!
//! `survived_to_max_ticks` is set only by the rollout engine on a truncation tick, never during
//! search/`advance`, so the `survival` term fires for truncated rollout episodes (and propagates
//! through z-mixing) but never inside the search — matching the planner, which calls `reward_fn` on
//! events straight out of `advance`.

use crate::snake::StepEvent;

#[derive(Clone, Copy, Debug)]
pub struct Reward {
    pub step: f64,
    pub food: f64,
    pub loss: f64,
    pub draw: f64,
    pub kill: f64,
    pub win: f64,
    pub survival: f64,
}

impl Reward {
    pub fn eval(&self, e: &StepEvent) -> f64 {
        let mut reward = self.step;
        if e.died {
            if e.lost {
                reward += self.loss;
            }
            if e.drew {
                reward += self.draw;
            }
            return reward;
        }
        if e.ate_food {
            reward += self.food;
        }
        if e.killed_opponent {
            reward += self.kill;
        }
        if e.won {
            reward += self.win;
        }
        if e.lost {
            reward += self.loss; // lost while alive: out-eaten under win_food_lead
        }
        if e.survived_to_max_ticks {
            reward += self.survival;
        }
        reward
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::snake::StepEvent;

    fn reward() -> Reward {
        Reward {
            step: 0.0,
            food: 0.1,
            loss: -0.5,
            draw: -0.25,
            kill: 0.5,
            win: 0.0,
            survival: 0.25,
        }
    }

    #[test]
    fn survival_fires_only_when_survived_to_max_ticks() {
        let r = reward();
        assert_eq!(r.eval(&StepEvent::default()), 0.0); // alive, nothing happened
        let survived = StepEvent {
            survived_to_max_ticks: true,
            ..Default::default()
        };
        assert!((r.eval(&survived) - 0.25).abs() < 1e-12);
    }

    #[test]
    fn a_dead_snake_never_collects_survival() {
        // The died branch returns before the survival term, mirroring MinimalReward's early return.
        let r = reward();
        let dead = StepEvent {
            died: true,
            lost: true,
            survived_to_max_ticks: true,
            ..Default::default()
        };
        assert!((r.eval(&dead) - (-0.5)).abs() < 1e-12); // loss only
    }
}
