//! Snapshot codec, scheme (c) of the plan's decision ladder: the rapier world is
//! serialized whole (schemes (a)/(b) — seed + compact dynamic state — do not resume
//! bit-identically; solver warm-start state matters). The track is still re-derived
//! from the seed, and decoded worlds are topology- and finiteness-validated.

use super::{dynamics::CarWorld, CarRacing, CarRacingState, LiveState};
use crate::codec_util::{serde_decode, serde_encode};
use reinfors_core::StateCodec;
use std::sync::Arc;

pub const CODEC_VERSION: u8 = 1;
const MAX_TICK: u32 = 1 << 30;

#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::large_enum_variant)]
enum Snap {
    Pending,
    Live {
        seed: u32,
        tick: u32,
        done: bool,
        visited: Vec<u64>,
        car: CarWorld,
    },
}

pub struct CarRacingCodec {
    pub game: CarRacing,
}

fn check_finite(label: &str, values: &[f64]) -> Result<(), String> {
    if values.iter().any(|v| !v.is_finite()) {
        return Err(format!("{label} contains a non-finite value"));
    }
    Ok(())
}

fn validate_car(car: &CarWorld) -> Result<(), String> {
    if car.bodies.len() != 5 {
        return Err(format!("expected 5 bodies, got {}", car.bodies.len()));
    }
    if car.impulse_joints.len() != 4 {
        return Err(format!(
            "expected 4 impulse joints, got {}",
            car.impulse_joints.len()
        ));
    }
    if car.multibody_joints.iter().next().is_some() {
        return Err("unexpected multibody joints".to_string());
    }
    let handles = std::iter::once(car.hull).chain(car.wheels.iter().copied());
    for h in handles {
        let Some(body) = car.bodies.get(h) else {
            return Err("a stored body handle is not in the body set".to_string());
        };
        let t = body.translation();
        let v = body.linvel();
        check_finite(
            "a body",
            &[
                f64::from(t.x),
                f64::from(t.y),
                f64::from(body.rotation().angle()),
                f64::from(v.x),
                f64::from(v.y),
                f64::from(body.angvel()),
            ],
        )?;
    }
    for j in car.joints {
        if car.impulse_joints.get(j).is_none() {
            return Err("a stored joint handle is not in the joint set".to_string());
        }
    }
    for c in &car.ctl {
        check_finite(
            "a wheel control",
            &[c.gas, c.brake, c.steer, c.omega, c.phase],
        )?;
    }
    check_finite("fuel_spent", &[car.fuel_spent])
}

impl StateCodec for CarRacingCodec {
    type State = CarRacingState;

    fn encode(&self, state: &CarRacingState) -> Vec<u8> {
        let snap = match state {
            CarRacingState::Pending => Snap::Pending,
            CarRacingState::Live(l) => Snap::Live {
                seed: l.seed,
                tick: l.tick,
                done: l.done,
                visited: l.visited.clone(),
                car: l.car.clone(),
            },
        };
        serde_encode(CODEC_VERSION, &snap)
    }

    fn decode(&self, bytes: &[u8]) -> Result<CarRacingState, String> {
        let snap: Snap = serde_decode(CODEC_VERSION, bytes)?;
        let Snap::Live {
            seed,
            tick,
            done,
            visited,
            car,
        } = snap
        else {
            return Ok(CarRacingState::Pending);
        };
        if tick > MAX_TICK {
            return Err(format!("tick {tick} exceeds the tick bound"));
        }
        validate_car(&car)?;

        let track = Arc::new(super::track::Track::generate(seed));
        let n_tiles = track.tiles.len();
        if visited.len() != n_tiles.div_ceil(64) {
            return Err(format!(
                "visited bitset length {} does not match the {n_tiles}-tile track for seed {seed}",
                visited.len(),
            ));
        }
        for (w, word) in visited.iter().enumerate() {
            for b in 0..64 {
                let id = w * 64 + b;
                if word & (1u64 << b) != 0 && id >= n_tiles {
                    return Err(format!("visited bit {id} exceeds tile count {n_tiles}"));
                }
            }
        }

        let visited_count = visited.iter().map(|w| w.count_ones()).sum();
        let mut live = LiveState {
            seed,
            track,
            car,
            tick,
            visited,
            visited_count,
            wheel_tiles: Default::default(),
            done,
        };
        self.game.contact_pass_derived(&mut live);
        Ok(CarRacingState::Live(Box::new(live)))
    }

    fn validate_decoded_state(&self, state: &CarRacingState, done: bool) -> Result<(), String> {
        match state {
            CarRacingState::Pending => {
                Err("a live restored state cannot be a pending chance sentinel".to_string())
            }
            CarRacingState::Live(l) if l.done != done => Err(format!(
                "snapshot done flag {} disagrees with lifecycle done {done}",
                l.done
            )),
            CarRacingState::Live(_) => Ok(()),
        }
    }
}
