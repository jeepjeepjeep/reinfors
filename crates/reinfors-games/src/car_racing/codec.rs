//! Snapshot codec with a scoped resume guarantee. The rapier body/collider/joint sets
//! are serialized; the contact machinery never crosses the byte boundary (deserialized
//! broadphase state is memory-unsafe on hostile bytes) and is rebuilt fresh at decode.
//!
//! Guarantee: restore is STATE-exact (every serialized field bit-identical) and
//! RESTORE-deterministic (one snapshot always resumes one trajectory), but the resumed
//! trajectory departs from the never-snapshotted run at f32 rounding order — the fresh
//! island manager solves constraints in a different order. Games with pure-data states
//! keep full bit-transparency; this physics game's guarantee is deliberately narrower.

use super::dynamics::{canonical_params, CarWorld};
use super::{CarRacing, CarRacingState, LiveState};
use crate::codec_util::{serde_decode, serde_encode};
use rapier2d::prelude::{JointAxis, RigidBodyHandle, Vector};
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

/// Compare every step-relevant configuration field of the stored car against a
/// freshly constructed one: body mass/damping properties, collider geometry and
/// filters, and full joint definitions (motor velocity normalized — it is set from
/// steer state each step). The contact machinery never crosses serde, shapes are
/// whitelisted, and the rapier version is pinned exactly, so this enumeration is
/// frozen with the dependency.
fn validate_config(car: &CarWorld) -> Result<(), String> {
    let reference = CarWorld::new(0.0, 0.0, 0.0);
    let handles = |c: &CarWorld| -> Vec<_> {
        std::iter::once(c.hull)
            .chain(c.wheels.iter().copied())
            .collect()
    };
    for (i, (&ha, &hb)) in handles(car).iter().zip(&handles(&reference)).enumerate() {
        let (a, b) = (&car.bodies[ha], &reference.bodies[hb]);
        let a_cfg = (
            a.body_type(),
            a.linear_damping(),
            a.angular_damping(),
            a.gravity_scale(),
            a.dominance_group(),
            a.is_ccd_enabled(),
            a.mass(),
            a.mass_properties().local_mprops.inv_principal_inertia,
            a.mass_properties().local_mprops.local_com,
        );
        let b_cfg = (
            b.body_type(),
            b.linear_damping(),
            b.angular_damping(),
            b.gravity_scale(),
            b.dominance_group(),
            b.is_ccd_enabled(),
            b.mass(),
            b.mass_properties().local_mprops.inv_principal_inertia,
            b.mass_properties().local_mprops.local_com,
        );
        if a_cfg != b_cfg {
            return Err(format!(
                "body {i} configuration differs from the canonical car"
            ));
        }
    }

    let collider_cfg = |c: &CarWorld, h: RigidBodyHandle| -> Vec<Vec<u8>> {
        let mut out: Vec<Vec<u8>> = c
            .colliders
            .iter()
            .filter(|(_, col)| col.parent() == Some(h))
            .map(|(_, col)| {
                let poly = col.shape().as_convex_polygon().expect("whitelisted above");
                postcard::to_stdvec(&(
                    poly.points(),
                    col.density(),
                    col.is_sensor(),
                    col.friction(),
                    col.restitution(),
                    col.collision_groups(),
                    col.solver_groups(),
                    col.position_wrt_parent()
                        .map(|p| (p.translation, p.rotation.angle())),
                ))
                .expect("collider config serializes")
            })
            .collect();
        out.sort_unstable();
        out
    };
    for (i, (&ha, &hb)) in handles(car).iter().zip(&handles(&reference)).enumerate() {
        if collider_cfg(car, ha) != collider_cfg(&reference, hb) {
            return Err(format!(
                "collider configuration on body {i} differs from the canonical car"
            ));
        }
    }

    for (i, (&ja, &jb)) in car.joints.iter().zip(&reference.joints).enumerate() {
        let normalized = |c: &CarWorld, jh| -> Vec<u8> {
            let mut data = c.impulse_joints.get(jh).expect("validated above").data;
            data.set_motor_velocity(JointAxis::AngX, 0.0, 0.0);
            // Motor and limit impulses are solver state, not configuration.
            for m in &mut data.motors {
                m.impulse = 0.0;
            }
            for l in &mut data.limits {
                l.impulse = 0.0;
            }
            postcard::to_stdvec(&data).expect("joint config serializes")
        };
        if normalized(car, ja) != normalized(&reference, jb) {
            return Err(format!(
                "joint {i} configuration differs from the canonical car"
            ));
        }
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
        // Forces are consumed and reset every step; a legitimate snapshot never
        // carries one, and the fingerprint zeroes them, so reject rather than trust.
        if body.user_force() != Vector::ZERO || body.user_torque() != 0.0 {
            return Err("a stored body carries pending user forces".to_string());
        }
    }

    let mut hull_colliders = 0usize;
    let mut wheel_colliders = [0usize; 4];
    for (_, collider) in car.colliders.iter() {
        // Whitelist the shape before anything (fingerprint included) steps the world:
        // hostile bytes can smuggle BVH-bearing shapes whose traversal is unsound.
        if collider.shape().as_convex_polygon().is_none() {
            return Err("a stored collider is not a convex polygon".to_string());
        }
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

/// Reassemble a deserialized world through rapier's insertion path: fresh contact
/// machinery only learns about bodies via insertion bookkeeping, so a world whose sets
/// were deserialized directly would never be simulated.
fn rebuild_world(des: &CarWorld) -> CarWorld {
    use rapier2d::prelude::*;
    let mut bodies = RigidBodySet::new();
    let mut colliders = ColliderSet::new();
    let mut impulse_joints = ImpulseJointSet::new();
    let mut map = std::collections::HashMap::new();
    for (old, body) in des.bodies.iter() {
        map.insert(old, bodies.insert(body.clone()));
    }
    for (_, collider) in des.colliders.iter() {
        let parent = collider.parent().expect("validated: no orphan colliders");
        colliders.insert_with_parent(collider.clone(), map[&parent], &mut bodies);
    }
    // insert_with_parent added collider masses onto the clone's already-complete
    // mass properties; recompute from scratch to match the construction path.
    for &new_handle in map.values() {
        bodies[new_handle].recompute_mass_properties_from_colliders(&colliders);
    }
    let joints = des.joints.map(|jh| {
        let j = des.impulse_joints.get(jh).expect("validated above");
        let new = impulse_joints.insert(map[&j.body1()], map[&j.body2()], j.data, true);
        impulse_joints
            .get_mut(new, false)
            .expect("just inserted")
            .impulses = j.impulses;
        new
    });
    CarWorld {
        bodies,
        colliders,
        impulse_joints,
        multibody_joints: MultibodyJointSet::new(),
        islands: IslandManager::default(),
        broad_phase: BroadPhaseBvh::new(),
        narrow_phase: NarrowPhase::default(),
        ccd: CCDSolver::default(),
        params: super::dynamics::canonical_params(),
        hull: map[&des.hull],
        wheels: des.wheels.map(|w| map[&w]),
        joints,
        ctl: des.ctl.clone(),
        fuel_spent: des.fuel_spent,
    }
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
        // Hostile bytes can trip asserts inside rapier's deserialized structures before
        // our validation sees them; contain any panic as a decode error.
        let result =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.decode_inner(bytes)));
        match result {
            Ok(r) => r,
            Err(_) => Err("snapshot bytes rejected: physics deserialization panicked".to_string()),
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
            car,
        } = snap
        else {
            return Ok(CarRacingState::Pending);
        };
        if tick > MAX_TICK {
            return Err(format!("tick {tick} exceeds the tick bound"));
        }
        validate_car(&car)?;
        validate_config(&car).map_err(|e| format!("{e} construction"))?;
        let mut car = rebuild_world(&car);
        car.params = canonical_params();

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
    fn forged_body_config_is_rejected() {
        let mut l = live(5);
        let hull = l.car.hull;
        l.car.bodies[hull].set_linear_damping(f32::NAN);
        assert!(decode_err(&encode_live(&l)).contains("canonical car construction"));
        let mut l = live(5);
        let hull = l.car.hull;
        l.car.bodies[hull].set_gravity_scale(3.0, false);
        assert!(decode_err(&encode_live(&l)).contains("canonical car construction"));
    }

    #[test]
    fn forged_collider_config_is_rejected() {
        let mut l = live(5);
        let handle = l.car.colliders.iter().next().map(|(h, _)| h).unwrap();
        l.car.colliders[handle].set_density(9.9);
        assert!(decode_err(&encode_live(&l)).contains("canonical car construction"));
    }

    #[test]
    fn forged_joint_config_is_rejected() {
        let mut l = live(5);
        let jh = l.car.joints[0];
        let j = l.car.impulse_joints.get_mut(jh, false).unwrap();
        j.data.set_local_anchor1(Vector::new(5.0, 5.0));
        assert!(decode_err(&encode_live(&l)).contains("canonical car construction"));
    }

    #[test]
    fn forged_pending_force_is_rejected() {
        let mut l = live(5);
        let hull = l.car.hull;
        l.car.bodies[hull].add_force(Vector::new(1.0, 0.0), false);
        assert!(decode_err(&encode_live(&l)).contains("pending user forces"));
    }

    #[test]
    fn forged_extreme_pose_is_rejected() {
        let mut l = live(5);
        let hull = l.car.hull;
        l.car.bodies[hull].set_translation(Vector::new(1e8, 0.0), true);
        assert!(decode_err(&encode_live(&l)).contains("magnitude bound"));
    }
}
