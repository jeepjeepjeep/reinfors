//! Reward shaping, ported to match `snake_RL`'s `MinimalReward`.
//!
//! `survived_to_max_ticks` is set by the training runner, never during search/`advance`, so it has
//! no field here and the `survival` term never fires inside the search (matching the planner, which
//! calls `reward_fn` on events straight out of `advance`).

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
        reward
    }
}
