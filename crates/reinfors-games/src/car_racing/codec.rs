//! Snapshot codec with a scoped resume guarantee. The bytes carry a compact DTO of
//! dynamic values only — body poses/velocities, wheel controls, joint warm-start
//! impulses, progress. No rapier structure ever crosses the byte boundary (deserialized
//! physics objects range from unsound BVH indexing to attacker-set solver work);
//! decode reconstructs the canonical car and copies the bounded values in.
//!
//! Guarantee: restore is STATE-exact (the serialized dynamic values round-trip
//! bit-identically) and RESTORE-deterministic (one snapshot always resumes one
//! trajectory), but the resumed trajectory departs from the never-snapshotted run at
//! f32 rounding order — solver bookkeeping is rebuilt, not restored. Games with
//! pure-data states keep full bit-transparency; this physics game's guarantee is
//! deliberately narrower.

use super::dynamics::{CarWorld, WheelCtl};
use super::{CarRacing, CarRacingState, LiveState};
use crate::codec_util::{serde_decode, serde_encode};
use rapier2d::prelude::{JointAxis, Pose, RigidBodyHandle, Rotation, Vector};
use reinfors_core::StateCodec;
use std::sync::Arc;

pub const CODEC_VERSION: u8 = 2;
const MAX_TICK: u32 = 1 << 30;
const MAX_SNAPSHOT_BYTES: usize = 4 * 1024;
const MAX_COORD: f64 = 1e6;
const MAX_PHASE: f64 = 1e12;
const ANG_X: usize = JointAxis::AngX as usize;

#[derive(serde::Serialize, serde::Deserialize)]
struct BodyDyn {
    pos: [f32; 2],
    /// (cos, sin) raw components, so restore is bit-exact; validated near-unit.
    rot: [f32; 2],
    linvel: [f32; 2],
    angvel: f32,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct JointDyn {
    impulses: [f32; 3],
    motor_impulse: f32,
    limit_impulse: f32,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::large_enum_variant)] // transient decode value, never stored
enum Snap {
    Pending,
    Live {
        seed: u32,
        tick: u32,
        done: bool,
        new_lap: bool,
        visited: Vec<u64>,
        bodies: [BodyDyn; 5],
        joints: [JointDyn; 4],
        ctl: [WheelCtl; 4],
        fuel_spent: f64,
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

fn body_handles(car: &CarWorld) -> [RigidBodyHandle; 5] {
    [
        car.hull,
        car.wheels[0],
        car.wheels[1],
        car.wheels[2],
        car.wheels[3],
    ]
}

fn body_dyn(car: &CarWorld, h: RigidBodyHandle) -> BodyDyn {
    let body = &car.bodies[h];
    let t = body.translation();
    let r = body.rotation();
    let v = body.linvel();
    BodyDyn {
        pos: [t.x, t.y],
        rot: [r.cos(), r.sin()],
        linvel: [v.x, v.y],
        angvel: body.angvel(),
    }
}

fn joint_dyn(car: &CarWorld, i: usize) -> JointDyn {
    let joint = car
        .impulse_joints
        .get(car.joints[i])
        .expect("canonical joint handles are always live");
    JointDyn {
        impulses: [joint.impulses.x, joint.impulses.y, joint.impulses.z],
        motor_impulse: joint.data.motors[ANG_X].impulse,
        limit_impulse: joint.data.limits[ANG_X].impulse,
    }
}

fn validate_body(b: &BodyDyn) -> Result<(), String> {
    check_bounded(
        "a body pose",
        &[f64::from(b.pos[0]), f64::from(b.pos[1])],
        MAX_COORD,
    )?;
    check_finite(
        "a body rotation",
        &[f64::from(b.rot[0]), f64::from(b.rot[1])],
    )?;
    let norm = f64::from(b.rot[0]).powi(2) + f64::from(b.rot[1]).powi(2);
    if (norm - 1.0).abs() > 1e-3 {
        return Err("a body rotation is not a unit rotation".to_string());
    }
    check_bounded(
        "a body velocity",
        &[
            f64::from(b.linvel[0]),
            f64::from(b.linvel[1]),
            f64::from(b.angvel),
        ],
        MAX_COORD,
    )
}

fn validate_ctl(ctl: &[WheelCtl; 4], fuel_spent: f64) -> Result<(), String> {
    for c in ctl {
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
    check_bounded("fuel_spent", &[fuel_spent], MAX_PHASE)?;
    if fuel_spent < 0.0 {
        return Err("fuel_spent is negative".to_string());
    }
    Ok(())
}

/// Build the canonical car and copy the validated dynamic values in. Every
/// configuration field (masses, shapes, filters, joint definitions, solver params)
/// comes from `CarWorld::new`; nothing in the snapshot can influence them.
fn reconstruct_car(bodies: &[BodyDyn; 5], joints: &[JointDyn; 4]) -> CarWorld {
    let mut car = CarWorld::new(0.0, 0.0, 0.0);
    for (h, b) in body_handles(&car).into_iter().zip(bodies) {
        let body = &mut car.bodies[h];
        body.set_position(
            Pose {
                translation: Vector::new(b.pos[0], b.pos[1]),
                rotation: Rotation::from_cos_sin_unchecked(b.rot[0], b.rot[1]),
            },
            true,
        );
        body.set_linvel(Vector::new(b.linvel[0], b.linvel[1]), true);
        body.set_angvel(b.angvel, true);
    }
    for (jh, j) in car.joints.into_iter().zip(joints) {
        let joint = car
            .impulse_joints
            .get_mut(jh, false)
            .expect("just constructed");
        joint.impulses.x = j.impulses[0];
        joint.impulses.y = j.impulses[1];
        joint.impulses.z = j.impulses[2];
        joint.data.motors[ANG_X].impulse = j.motor_impulse;
        joint.data.limits[ANG_X].impulse = j.limit_impulse;
    }
    car
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
                bodies: body_handles(&l.car).map(|h| body_dyn(&l.car, h)),
                joints: [0, 1, 2, 3].map(|i| joint_dyn(&l.car, i)),
                ctl: l.car.ctl.clone(),
                fuel_spent: l.car.fuel_spent,
            },
        };
        serde_encode(CODEC_VERSION, &snap)
    }

    fn decode(&self, bytes: &[u8]) -> Result<CarRacingState, String> {
        // Belt-and-braces: the DTO path deserializes only plain numbers, but keep any
        // unexpected panic contained as a decode error.
        let result =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.decode_inner(bytes)));
        match result {
            Ok(r) => r,
            Err(_) => Err("snapshot bytes rejected: decoding panicked".to_string()),
        }
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

impl CarRacingCodec {
    fn decode_inner(&self, bytes: &[u8]) -> Result<CarRacingState, String> {
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
            bodies,
            joints,
            ctl,
            fuel_spent,
        } = snap
        else {
            return Ok(CarRacingState::Pending);
        };
        if tick > MAX_TICK {
            return Err(format!("tick {tick} exceeds the tick bound"));
        }
        for b in &bodies {
            validate_body(b)?;
        }
        for j in &joints {
            let vals: Vec<f64> = j
                .impulses
                .iter()
                .chain([&j.motor_impulse, &j.limit_impulse])
                .map(|&v| f64::from(v))
                .collect();
            check_bounded("a joint impulse", &vals, MAX_PHASE)?;
        }
        validate_ctl(&ctl, fuel_spent)?;

        let mut car = reconstruct_car(&bodies, &joints);
        car.ctl = ctl;
        car.fuel_spent = fuel_spent;

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
}

#[cfg(test)]
mod tests {

    use super::*;

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
    fn forged_out_of_range_controls_are_rejected() {
        let mut l = live(5);
        l.car.ctl[0].gas = 5.0;
        assert!(decode_err(&encode_live(&l)).contains("gas/brake"));
    }

    #[test]
    fn tampered_solver_params_never_reach_the_bytes() {
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

    #[test]
    fn forged_extreme_velocity_is_rejected() {
        let mut l = live(5);
        let w = l.car.wheels[2];
        l.car.bodies[w].set_linvel(Vector::new(0.0, 1e8), true);
        assert!(decode_err(&encode_live(&l)).contains("magnitude bound"));
    }

    #[test]
    fn forged_non_unit_rotation_is_rejected() {
        let mut l = live(5);
        let hull = l.car.hull;
        l.car.bodies[hull].set_rotation(Rotation::from_cos_sin_unchecked(2.0, 0.0), true);
        assert!(decode_err(&encode_live(&l)).contains("unit rotation"));
    }

    #[test]
    fn forged_extreme_joint_impulse_is_rejected() {
        let mut l = live(5);
        let jh = l.car.joints[0];
        l.car.impulse_joints.get_mut(jh, false).unwrap().impulses.x = 1e30;
        assert!(decode_err(&encode_live(&l)).contains("joint impulse"));
    }

    #[test]
    fn forged_configuration_cannot_be_expressed() {
        // The DTO carries dynamics only; every configuration field comes from the
        // canonical construction. Tampering with rapier config on a live state must
        // leave the decoded car canonical rather than smuggle the change through.
        let mut l = live(5);
        let hull = l.car.hull;
        l.car.bodies[hull].set_gravity_scale(3.0, false);
        let decoded = codec().decode(&encode_live(&l)).unwrap();
        let CarRacingState::Live(out) = decoded else {
            panic!("expected live state");
        };
        assert_eq!(out.car.bodies[out.car.hull].gravity_scale(), 1.0);
    }
}
