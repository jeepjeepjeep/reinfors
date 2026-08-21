//! Snapshot codec, scheme (c) of the plan's decision ladder: the rapier world is
//! serialized whole (schemes (a)/(b) — seed + compact dynamic state — do not resume
//! bit-identically; solver warm-start state matters). The track is still re-derived
//! from the seed; decoded worlds are graph- and bounds-validated, and solver
//! parameters are replaced with the canonical task constants.

use super::{dynamics::CarWorld, CarRacing, CarRacingState, LiveState};
use crate::codec_util::{serde_decode, serde_encode};
use reinfors_core::StateCodec;
use std::sync::Arc;

pub const CODEC_VERSION: u8 = 1;
const MAX_TICK: u32 = 1 << 30;
const MAX_SNAPSHOT_BYTES: usize = 64 * 1024;
const MAX_COORD: f64 = 1e6;
const MAX_PHASE: f64 = 1e12;

#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::large_enum_variant)]
enum Snap {
    Pending,
    Live {
        seed: u32,
        tick: u32,
        done: bool,
        new_lap: bool,
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

fn check_bounded(label: &str, values: &[f64], bound: f64) -> Result<(), String> {
    check_finite(label, values)?;
    if values.iter().any(|v| v.abs() > bound) {
        return Err(format!("{label} exceeds the magnitude bound {bound}"));
    }
    Ok(())
}

/// Enforce the exact five-body / eight-collider / four-revolute-joint graph and bound
/// every step-relevant numeric field. Solver parameters are not validated here because
/// `decode` replaces them with the canonical task constants.
fn validate_car(car: &CarWorld) -> Result<(), String> {
    if car.bodies.len() != 5 {
        return Err(format!("expected 5 bodies, got {}", car.bodies.len()));
    }
    if car.colliders.len() != 8 {
        return Err(format!("expected 8 colliders, got {}", car.colliders.len()));
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

    let handles: Vec<_> = std::iter::once(car.hull)
        .chain(car.wheels.iter().copied())
        .collect();
    for (i, a) in handles.iter().enumerate() {
        if handles[i + 1..].contains(a) {
            return Err("duplicate body handles in the stored car".to_string());
        }
    }
    for h in &handles {
        let Some(body) = car.bodies.get(*h) else {
            return Err("a stored body handle is not in the body set".to_string());
        };
        let t = body.translation();
        let v = body.linvel();
        check_bounded(
            "a body pose",
            &[
                f64::from(t.x),
                f64::from(t.y),
                f64::from(body.rotation().angle()),
            ],
            MAX_COORD,
        )?;
        check_bounded(
            "a body velocity",
            &[f64::from(v.x), f64::from(v.y), f64::from(body.angvel())],
            MAX_COORD,
        )?;
    }

    let mut hull_colliders = 0usize;
    let mut wheel_colliders = [0usize; 4];
    for (_, collider) in car.colliders.iter() {
        let Some(parent) = collider.parent() else {
            return Err("an orphan collider in the stored car".to_string());
        };
        if parent == car.hull {
            hull_colliders += 1;
        } else if let Some(w) = car.wheels.iter().position(|&h| h == parent) {
            wheel_colliders[w] += 1;
        } else {
            return Err("a collider parented outside the car".to_string());
        }
    }
    if hull_colliders != 4 || wheel_colliders != [1; 4] {
        return Err(format!(
            "collider graph mismatch: hull {hull_colliders}, wheels {wheel_colliders:?}"
        ));
    }

    for (i, jh) in car.joints.iter().enumerate() {
        if car.joints[i + 1..].contains(jh) {
            return Err("duplicate joint handles in the stored car".to_string());
        }
        let Some(joint) = car.impulse_joints.get(*jh) else {
            return Err("a stored joint handle is not in the joint set".to_string());
        };
        if joint.body1() != car.hull || joint.body2() != car.wheels[i] {
            return Err(format!("joint {i} does not connect the hull to wheel {i}"));
        }
        if joint.data.as_revolute().is_none() {
            return Err(format!("joint {i} is not a revolute joint"));
        }
        check_bounded(
            "a joint impulse",
            &[
                f64::from(joint.impulses.x),
                f64::from(joint.impulses.y),
                f64::from(joint.impulses.z),
            ],
            MAX_PHASE,
        )?;
    }

    for c in &car.ctl {
        check_finite(
            "a wheel control",
            &[c.gas, c.brake, c.steer, c.omega, c.phase],
        )?;
        if !(0.0..=1.0).contains(&c.gas) || !(0.0..=1.0).contains(&c.brake) {
            return Err("gas/brake outside [0, 1]".to_string());
        }
        if c.steer.abs() > 1.0 {
            return Err("steer outside [-1, 1]".to_string());
        }
        if c.omega.abs() > MAX_COORD || c.phase.abs() > MAX_PHASE {
            return Err("wheel omega/phase exceeds bounds".to_string());
        }
    }
    check_bounded("fuel_spent", &[car.fuel_spent], MAX_PHASE)?;
    if car.fuel_spent < 0.0 {
        return Err("fuel_spent is negative".to_string());
    }
    Ok(())
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
                new_lap: l.new_lap,
                visited: l.visited.clone(),
                car: l.car.clone(),
            },
        };
        serde_encode(CODEC_VERSION, &snap)
    }

    fn decode(&self, bytes: &[u8]) -> Result<CarRacingState, String> {
        if bytes.len() > MAX_SNAPSHOT_BYTES {
            return Err(format!(
                "snapshot of {} bytes exceeds the {MAX_SNAPSHOT_BYTES}-byte bound",
                bytes.len()
            ));
        }
        let snap: Snap = serde_decode(CODEC_VERSION, bytes)?;
        let Snap::Live {
            seed,
            tick,
            done,
            new_lap,
            visited,
            mut car,
        } = snap
        else {
            return Ok(CarRacingState::Pending);
        };
        if tick > MAX_TICK {
            return Err(format!("tick {tick} exceeds the tick bound"));
        }
        validate_car(&car)?;
        car.params = super::dynamics::canonical_params();

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
            new_lap,
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

#[cfg(test)]
mod tests {
    use super::*;
    use rapier2d::prelude::*;

    fn game() -> CarRacing {
        CarRacing::default()
    }

    fn live(seed: u32) -> LiveState {
        game().realize(seed)
    }

    fn codec() -> CarRacingCodec {
        CarRacingCodec { game: game() }
    }

    fn encode_live(l: &LiveState) -> Vec<u8> {
        codec().encode(&CarRacingState::Live(Box::new(l.clone())))
    }

    fn decode_err(bytes: &[u8]) -> String {
        match codec().decode(bytes) {
            Err(e) => e,
            Ok(_) => panic!("forged snapshot decoded successfully"),
        }
    }

    #[test]
    fn forged_duplicate_body_handle_is_rejected() {
        let mut l = live(5);
        l.car.wheels[0] = l.car.hull;
        assert!(decode_err(&encode_live(&l)).contains("duplicate body"));
    }

    #[test]
    fn forged_joint_endpoints_are_rejected() {
        let mut l = live(5);
        l.car.joints.swap(0, 1);
        assert!(decode_err(&encode_live(&l)).contains("does not connect"));
    }

    #[test]
    fn forged_extra_joint_is_rejected() {
        let mut l = live(5);
        let j = RevoluteJointBuilder::new().build();
        l.car
            .impulse_joints
            .insert(l.car.hull, l.car.wheels[0], j, true);
        assert!(decode_err(&encode_live(&l)).contains("expected 4 impulse joints"));
    }

    #[test]
    fn forged_out_of_range_controls_are_rejected() {
        let mut l = live(5);
        l.car.ctl[0].gas = 5.0;
        assert!(decode_err(&encode_live(&l)).contains("gas/brake"));
    }

    #[test]
    fn tampered_solver_params_are_normalized() {
        let mut l = live(5);
        l.car.params.dt = 17.0;
        let decoded = codec().decode(&encode_live(&l)).unwrap();
        let CarRacingState::Live(out) = decoded else {
            panic!("expected live state");
        };
        assert_eq!(
            out.car.params.dt,
            super::super::dynamics::canonical_params().dt
        );
    }

    #[test]
    fn oversized_payload_is_rejected() {
        let bytes = vec![CODEC_VERSION; MAX_SNAPSHOT_BYTES + 1];
        assert!(decode_err(&bytes).contains("byte bound"));
    }

    #[test]
    fn forged_extreme_pose_is_rejected() {
        let mut l = live(5);
        let hull = l.car.hull;
        l.car.bodies[hull].set_translation(Vector::new(1e8, 0.0), true);
        assert!(decode_err(&encode_live(&l)).contains("magnitude bound"));
    }
}
