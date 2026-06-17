//! Egocentric 5-channel grid observation, ported to match `snake_RL`'s `EgocentricGridObservation`.
//!
//! Channels (C-order, shape `[5, g, g]` flattened): 0 own-head, 1 own-body, 2 opp-head, 3 opp-body,
//! 4 food. Coordinates are pre-rotated by `k` CCW quarter-turns (k from the agent's heading) so the
//! queried snake always faces "up" — identical to the Python `_fill` in-place rotation.

use std::collections::HashSet;

use crate::snake::{Cell, SnakeBody, SnakeEnv};

pub const N_CHANNELS: usize = 5;
const CH_OWN_HEAD: usize = 0;
const CH_OWN_BODY: usize = 1;
const CH_OPP_HEAD: usize = 2;
const CH_OPP_BODY: usize = 3;
const CH_FOOD: usize = 4;

/// Build the egocentric observation for `agent` (0 = A, 1 = B) as a flat `[5 * g * g]` f32 buffer.
pub fn egocentric(env: &SnakeEnv, agent: usize) -> Vec<f32> {
    egocentric_parts(&env.snakes, &env.food, env.grid_size, agent)
}

/// Same as [`egocentric`], operating directly on a (snakes, food) state — used by the search, which
/// builds observations for simulated child states without constructing a full `SnakeEnv`.
pub fn egocentric_parts(
    snakes: &[SnakeBody; 2],
    food: &HashSet<Cell>,
    grid_size: i32,
    agent: usize,
) -> Vec<f32> {
    let g = grid_size;
    let edge = g - 1;
    let k = snakes[agent].direction.ego_rot_k();
    let plane = (g * g) as usize;
    let mut obs = vec![0.0f32; N_CHANNELS * plane];

    let rot = |r: i32, c: i32| -> (i32, i32) {
        match k {
            1 => (edge - c, r),
            2 => (edge - r, edge - c),
            3 => (c, edge - r),
            _ => (r, c),
        }
    };
    let mut set = |ch: usize, r: i32, c: i32| {
        let (rr, cc) = rot(r, c);
        obs[ch * plane + (rr as usize) * (g as usize) + (cc as usize)] = 1.0;
    };

    for (i, snake) in snakes.iter().enumerate() {
        if snake.is_empty() {
            continue;
        }
        let (head_ch, body_ch) = if i == agent {
            (CH_OWN_HEAD, CH_OWN_BODY)
        } else {
            (CH_OPP_HEAD, CH_OPP_BODY)
        };
        let mut ch = head_ch; // head is body[0]; everything after lands in the body channel
        for &(r, c) in &snake.body {
            set(ch, r, c);
            ch = body_ch;
        }
    }
    for &(r, c) in food {
        set(CH_FOOD, r, c);
    }
    obs
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::snake::{SnakeEnv, A};

    fn at(obs: &[f32], g: i32, ch: usize, r: i32, c: i32) -> f32 {
        obs[ch * (g * g) as usize + (r as usize) * (g as usize) + (c as usize)]
    }

    #[test]
    fn egocentric_rotates_by_heading() {
        // A faces Right -> k=1 -> (r,c) maps to (edge-c, r). Head (10,6) -> (19-6, 10) = (13,10).
        let env = SnakeEnv::new(20, 3, false, None);
        let obs = egocentric(&env, A);
        assert_eq!(at(&obs, 20, CH_OWN_HEAD, 13, 10), 1.0);
        // Body cells (10,5),(10,4) -> (14,10),(15,10) in the own-body channel.
        assert_eq!(at(&obs, 20, CH_OWN_BODY, 14, 10), 1.0);
        assert_eq!(at(&obs, 20, CH_OWN_BODY, 15, 10), 1.0);
        // The head cell must not also be flagged as body.
        assert_eq!(at(&obs, 20, CH_OWN_BODY, 13, 10), 0.0);
        // Opponent B head (10,14) -> (19-14,10) = (5,10) in the opp-head channel.
        assert_eq!(at(&obs, 20, CH_OPP_HEAD, 5, 10), 1.0);
    }

    #[test]
    fn food_lands_in_food_channel_rotated() {
        let mut env = SnakeEnv::new(20, 3, false, None);
        env.food.insert((10, 6)); // same transform as A's head -> (13,10)
        let obs = egocentric(&env, A);
        assert_eq!(at(&obs, 20, CH_FOOD, 13, 10), 1.0);
    }

    #[test]
    fn buffer_has_expected_length() {
        let env = SnakeEnv::new(20, 3, false, None);
        assert_eq!(egocentric(&env, A).len(), N_CHANNELS * 20 * 20);
    }
}
