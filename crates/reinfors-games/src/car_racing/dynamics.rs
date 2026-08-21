//! Car physics: rapier bodies/joints plus the tire model ported from Gymnasium's
//! `car_dynamics.py`. The rapier world holds exactly five bodies and four motorized
//! revolute joints; nothing collides (tile contact is geometric, in `track`).

use rapier2d::prelude::*;

pub const SIZE: f64 = 0.02;
const ENGINE_POWER: f64 = 100_000_000.0 * SIZE * SIZE;
const WHEEL_MOMENT_OF_INERTIA: f64 = 4000.0 * SIZE * SIZE;
pub const FRICTION_LIMIT: f64 = 1_000_000.0 * SIZE * SIZE;
pub const WHEEL_R: f64 = 27.0;
pub const WHEEL_W: f64 = 14.0;
pub const WHEELPOS: [[f64; 2]; 4] = [[-55.0, 80.0], [55.0, 80.0], [-55.0, -82.0], [55.0, -82.0]];
pub(crate) const HULL_POLY1: [[f64; 2]; 4] =
    [[-60.0, 130.0], [60.0, 130.0], [60.0, 110.0], [-60.0, 110.0]];
pub(crate) const HULL_POLY2: [[f64; 2]; 4] =
    [[-15.0, 120.0], [15.0, 120.0], [20.0, 20.0], [-20.0, 20.0]];
pub(crate) const HULL_POLY3: [[f64; 2]; 8] = [
    [25.0, 20.0],
    [50.0, -10.0],
    [50.0, -40.0],
    [20.0, -90.0],
    [-20.0, -90.0],
    [-50.0, -40.0],
    [-50.0, -10.0],
    [-25.0, 20.0],
];
pub(crate) const HULL_POLY4: [[f64; 2]; 4] = [
    [-50.0, -120.0],
    [50.0, -120.0],
    [50.0, -90.0],
    [-50.0, -90.0],
];

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct WheelCtl {
    pub gas: f64,
    pub brake: f64,
    pub steer: f64,
    pub omega: f64,
    pub phase: f64,
    pub on_road: bool,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct CarWorld {
    pub bodies: RigidBodySet,
    pub colliders: ColliderSet,
    pub impulse_joints: ImpulseJointSet,
    pub multibody_joints: MultibodyJointSet,
    pub islands: IslandManager,
    pub broad_phase: BroadPhaseBvh,
    pub narrow_phase: NarrowPhase,
    pub ccd: CCDSolver,
    pub params: IntegrationParameters,
    pub hull: RigidBodyHandle,
    pub wheels: [RigidBodyHandle; 4],
    pub joints: [ImpulseJointHandle; 4],
    pub ctl: [WheelCtl; 4],
    pub fuel_spent: f64,
}

fn poly_collider(points: &[[f64; 2]]) -> Collider {
    let verts: Vec<Vector> = points
        .iter()
        .map(|p| Vector::new((p[0] * SIZE) as Real, (p[1] * SIZE) as Real))
        .collect();
    ColliderBuilder::convex_hull(&verts)
        .expect("hull polys are convex and non-degenerate")
        .density(1.0)
        .collision_groups(InteractionGroups::none())
        .build()
}

thread_local! {
    // Solver scratch only (rapier serde-skips it); reuse avoids per-step allocation.
    static PIPELINE: std::cell::RefCell<PhysicsPipeline> =
        std::cell::RefCell::new(PhysicsPipeline::new());
}

/// Task-constant solver configuration; snapshots never override it.
pub(crate) fn canonical_params() -> IntegrationParameters {
    IntegrationParameters {
        dt: (1.0 / super::track::FPS) as Real,
        ..IntegrationParameters::default()
    }
}

impl CarWorld {
    pub fn new(init_angle: f64, init_x: f64, init_y: f64) -> CarWorld {
        let mut bodies = RigidBodySet::new();
        let mut colliders = ColliderSet::new();
        let mut impulse_joints = ImpulseJointSet::new();

        let hull = bodies.insert(
            RigidBodyBuilder::dynamic()
                .translation(Vector::new(init_x as Real, init_y as Real))
                .rotation(init_angle as Real)
                .can_sleep(false)
                .build(),
        );
        for poly in [
            &HULL_POLY1[..],
            &HULL_POLY2[..],
            &HULL_POLY3[..],
            &HULL_POLY4[..],
        ] {
            colliders.insert_with_parent(poly_collider(poly), hull, &mut bodies);
        }

        let wheel_poly = [
            [-WHEEL_W, WHEEL_R],
            [WHEEL_W, WHEEL_R],
            [WHEEL_W, -WHEEL_R],
            [-WHEEL_W, -WHEEL_R],
        ];
        let mut wheels = Vec::with_capacity(4);
        let mut joints = Vec::with_capacity(4);
        let (ia_c, ia_s) = (libm::cos(init_angle), libm::sin(init_angle));
        for [wx, wy] in WHEELPOS {
            // World-space anchor: the hull is rotated, so the local offset must be too.
            let ox = ia_c * wx * SIZE - ia_s * wy * SIZE;
            let oy = ia_s * wx * SIZE + ia_c * wy * SIZE;
            let w = bodies.insert(
                RigidBodyBuilder::dynamic()
                    .translation(Vector::new((init_x + ox) as Real, (init_y + oy) as Real))
                    .rotation(init_angle as Real)
                    .can_sleep(false)
                    .build(),
            );
            let mut c = poly_collider(&wheel_poly);
            c.set_density(0.1);
            colliders.insert_with_parent(c, w, &mut bodies);
            let joint = RevoluteJointBuilder::new()
                .local_anchor1(Vector::new((wx * SIZE) as Real, (wy * SIZE) as Real))
                .local_anchor2(Vector::ZERO)
                .limits([-0.4, 0.4])
                .motor_max_force((180.0 * 900.0 * SIZE * SIZE) as Real)
                .motor_velocity(0.0, 0.0)
                .build();
            joints.push(impulse_joints.insert(hull, w, joint, true));
            wheels.push(w);
        }

        let params = canonical_params();
        CarWorld {
            bodies,
            colliders,
            impulse_joints,
            multibody_joints: MultibodyJointSet::new(),
            islands: IslandManager::new(),
            broad_phase: BroadPhaseBvh::new(),
            narrow_phase: NarrowPhase::new(),
            ccd: CCDSolver::new(),
            params,
            hull,
            wheels: [wheels[0], wheels[1], wheels[2], wheels[3]],
            joints: [joints[0], joints[1], joints[2], joints[3]],
            ctl: Default::default(),
            fuel_spent: 0.0,
        }
    }

    pub fn gas(&mut self, gas: f64) {
        let gas = gas.clamp(0.0, 1.0);
        for ctl in &mut self.ctl[2..4] {
            let diff = (gas - ctl.gas).min(0.1);
            ctl.gas += diff;
        }
    }

    pub fn brake(&mut self, b: f64) {
        for ctl in &mut self.ctl {
            ctl.brake = b;
        }
    }

    pub fn steer(&mut self, s: f64) {
        self.ctl[0].steer = s;
        self.ctl[1].steer = s;
    }

    /// One tire-model pass followed by one physics step.
    pub fn step(&mut self, dt: f64) {
        for i in 0..4 {
            let joint_angle = self.joint_angle(i) as f64;
            let ctl = &mut self.ctl[i];

            let diff = ctl.steer - joint_angle;
            let motor = diff.signum() * (50.0 * diff.abs()).min(3.0);
            if let Some(j) = self.impulse_joints.get_mut(self.joints[i], true) {
                j.data
                    .set_motor_velocity(JointAxis::AngX, motor as Real, 0.0);
            }

            let ctl = &mut self.ctl[i];
            let friction_limit = FRICTION_LIMIT * if ctl.on_road { 1.0 } else { 0.6 };

            let wheel = &self.bodies[self.wheels[i]];
            let rot = wheel.rotation();
            let forw = [-(rot.sin() as f64), rot.cos() as f64];
            let side = [rot.cos() as f64, rot.sin() as f64];
            let v = wheel.linvel();
            let (vx, vy) = (v.x as f64, v.y as f64);
            let vf = forw[0] * vx + forw[1] * vy;
            let vs = side[0] * vx + side[1] * vy;

            ctl.omega +=
                dt * ENGINE_POWER * ctl.gas / WHEEL_MOMENT_OF_INERTIA / (ctl.omega.abs() + 5.0);
            self.fuel_spent += dt * ENGINE_POWER * ctl.gas;

            let ctl = &mut self.ctl[i];
            if ctl.brake >= 0.9 {
                ctl.omega = 0.0;
            } else if ctl.brake > 0.0 {
                let dir = -ctl.omega.signum();
                let mut val = 15.0 * ctl.brake;
                if val > ctl.omega.abs() {
                    val = ctl.omega.abs();
                }
                ctl.omega += dir * val;
            }
            ctl.phase += ctl.omega * dt;

            let wheel_rad = WHEEL_R * SIZE;
            let vr = ctl.omega * wheel_rad;
            let mut f_force = -vf + vr;
            let mut p_force = -vs;
            f_force *= 205_000.0 * SIZE * SIZE;
            p_force *= 205_000.0 * SIZE * SIZE;
            let force = libm::sqrt(f_force * f_force + p_force * p_force);
            if force > friction_limit {
                f_force = f_force / force * friction_limit;
                p_force = p_force / force * friction_limit;
            }
            ctl.omega -= dt * f_force * wheel_rad / WHEEL_MOMENT_OF_INERTIA;

            let fx = (p_force * side[0] + f_force * forw[0]) as Real;
            let fy = (p_force * side[1] + f_force * forw[1]) as Real;
            self.bodies[self.wheels[i]].add_force(Vector::new(fx, fy), true);
        }

        PIPELINE.with_borrow_mut(|pipeline| {
            pipeline.step(
                Vector::ZERO,
                &self.params,
                &mut self.islands,
                &mut self.broad_phase,
                &mut self.narrow_phase,
                &mut self.bodies,
                &mut self.colliders,
                &mut self.impulse_joints,
                &mut self.multibody_joints,
                &mut self.ccd,
                &(),
                &(),
            );
        });
        for i in 0..4 {
            self.bodies[self.wheels[i]].reset_forces(true);
        }
    }

    /// Relative hull->wheel rotation, wrap-safe across the +-pi boundary.
    pub(crate) fn joint_angle(&self, i: usize) -> Real {
        let h = self.bodies[self.hull].rotation();
        let w = self.bodies[self.wheels[i]].rotation();
        let (hc, hs) = (f64::from(h.cos()), f64::from(h.sin()));
        let (wc, ws) = (f64::from(w.cos()), f64::from(w.sin()));
        libm::atan2(hc * ws - hs * wc, hc * wc + hs * ws) as Real
    }

    pub fn hull_pos(&self) -> (f64, f64) {
        let t = self.bodies[self.hull].translation();
        (t.x as f64, t.y as f64)
    }

    pub fn hull_angle(&self) -> f64 {
        self.bodies[self.hull].rotation().angle() as f64
    }

    pub fn hull_speed(&self) -> f64 {
        let v = self.bodies[self.hull].linvel();
        libm::sqrt((v.x as f64).powi(2) + (v.y as f64).powi(2))
    }

    /// Wheel body pose: translation and rotation (cos, sin).
    pub(crate) fn wheel_pose(&self, i: usize) -> ([f64; 2], [f64; 2]) {
        let body = &self.bodies[self.wheels[i]];
        let t = body.translation();
        let rot = body.rotation();
        (
            [f64::from(t.x), f64::from(t.y)],
            [f64::from(rot.cos()), f64::from(rot.sin())],
        )
    }

    /// World-space footprint quad of wheel `i`.
    pub fn wheel_quad(&self, i: usize) -> [[f64; 2]; 4] {
        let body = &self.bodies[self.wheels[i]];
        let t = body.translation();
        let rot = body.rotation();
        let (c, s) = (rot.cos() as f64, rot.sin() as f64);
        let local = [
            [-WHEEL_W * SIZE, WHEEL_R * SIZE],
            [WHEEL_W * SIZE, WHEEL_R * SIZE],
            [WHEEL_W * SIZE, -WHEEL_R * SIZE],
            [-WHEEL_W * SIZE, -WHEEL_R * SIZE],
        ];
        local.map(|[x, y]| [t.x as f64 + c * x - s * y, t.y as f64 + s * x + c * y])
    }
}
