//! CarRacing frame renderer: Gymnasium's draw list reproduced through the shared
//! rendering seam. World content follows the window transform (1000x800, camera-follow
//! with the first-second zoom animation, vertical flip); indicators and the score draw
//! unflipped, exactly as pygame composes them; the whole frame supersamples then
//! box-filters to 96x96, mirroring the smoothscale step.

use super::dynamics::{
    CarWorld, HULL_POLY1, HULL_POLY2, HULL_POLY3, HULL_POLY4, SIZE, WHEEL_R, WHEEL_W,
};
use super::track::{Track, PLAYFIELD, SCALE};
use super::{CarRacingState, LiveState};
use crate::render::{draw_text, hwc_to_chw_f32, FrameRenderer, Raster};

pub(crate) const FRAME: usize = 96;
const SUPERSAMPLE: u32 = 4;
const WINDOW_W: f64 = 1000.0;
const WINDOW_H: f64 = 800.0;
const ZOOM: f64 = 2.7;

const BG_COLOR: [u8; 3] = [102, 204, 102];
const GRASS_COLOR: [u8; 3] = [102, 230, 102];
const ROAD_COLOR: [u8; 3] = [102, 102, 102];
const HULL_COLOR: [u8; 3] = [204, 0, 0];
const WHEEL_COLOR: [u8; 3] = [0, 0, 0];
const WHEEL_WHITE: [u8; 3] = [77, 77, 77];

struct Camera {
    zoom: f64,
    angle: f64,
    trans: [f64; 2],
}

impl Camera {
    fn new(live: &LiveState) -> Camera {
        let t = f64::from(live.tick) / super::track::FPS;
        let zoom = 0.1 * SCALE * (1.0 - t).max(0.0) + ZOOM * SCALE * t.min(1.0);
        let angle = -live.car.hull_angle();
        let (px, py) = live.car.hull_pos();
        let (ca, sa) = (libm::cos(angle), libm::sin(angle));
        let (sx, sy) = (-px * zoom, -py * zoom);
        Camera {
            zoom,
            angle,
            trans: [
                ca * sx - sa * sy + WINDOW_W / 2.0,
                sa * sx + ca * sy + WINDOW_H / 4.0,
            ],
        }
    }

    /// World point to final-frame coordinates (flipped, like pygame's world pass).
    fn world(&self, [x, y]: [f64; 2]) -> [f32; 2] {
        let (ca, sa) = (libm::cos(self.angle), libm::sin(self.angle));
        let wx = (ca * x - sa * y) * self.zoom + self.trans[0];
        let wy = (sa * x + ca * y) * self.zoom + self.trans[1];
        [
            (wx * FRAME as f64 / WINDOW_W) as f32,
            (FRAME as f64 - wy * FRAME as f64 / WINDOW_H) as f32,
        ]
    }
}

/// Window coordinates (indicator layer, drawn after the flip) to final-frame units.
fn window([x, y]: [f64; 2]) -> [f32; 2] {
    [
        (x * FRAME as f64 / WINDOW_W) as f32,
        (y * FRAME as f64 / WINDOW_H) as f32,
    ]
}

/// Gym's f"{reward:04.0f}": zero decimals, zero-padded to width 4 (negatives "-012").
pub(crate) fn score_text(score: f64) -> String {
    let rounded = libm::round(score) as i64;
    if rounded < 0 {
        format!("-{:03}", rounded.unsigned_abs().min(999))
    } else {
        format!("{:04}", rounded.min(9999))
    }
}

pub(crate) struct CarRacingRenderer;

impl CarRacingRenderer {
    fn draw_world_poly(r: &mut Raster, cam: &Camera, pts: &[[f64; 2]], rgb: [u8; 3]) {
        let mapped: Vec<[f32; 2]> = pts.iter().map(|&p| cam.world(p)).collect();
        r.fill_poly(&mapped, rgb);
    }

    fn draw_road(r: &mut Raster, cam: &Camera, track: &Track) {
        let b = PLAYFIELD;
        Self::draw_world_poly(r, cam, &[[b, b], [b, -b], [-b, -b], [-b, b]], BG_COLOR);
        let g = PLAYFIELD / 20.0;
        for x in (-20i32..20).step_by(2) {
            for y in (-20i32..20).step_by(2) {
                let (fx, fy) = (f64::from(x) * g, f64::from(y) * g);
                Self::draw_world_poly(
                    r,
                    cam,
                    &[[fx + g, fy], [fx, fy], [fx, fy + g], [fx + g, fy + g]],
                    GRASS_COLOR,
                );
            }
        }
        for (i, tile) in track.tiles.iter().enumerate() {
            let shade = (0.01 * (i % 3) as f64 * 255.0) as u8;
            let rgb = ROAD_COLOR.map(|c| c.saturating_add(shade));
            Self::draw_world_poly(r, cam, &tile.quad, rgb);
        }
        for border in &track.borders {
            let rgb = if border.white {
                [255, 255, 255]
            } else {
                [255, 0, 0]
            };
            Self::draw_world_poly(r, cam, &border.quad, rgb);
        }
    }

    fn draw_car(r: &mut Raster, cam: &Camera, car: &CarWorld) {
        for i in 0..4 {
            let (pos, rot) = car.wheel_pose(i);
            let to_world = |lx: f64, ly: f64| {
                [
                    pos[0] + rot[0] * lx - rot[1] * ly,
                    pos[1] + rot[1] * lx + rot[0] * ly,
                ]
            };
            let (w, rad) = (WHEEL_W * SIZE, WHEEL_R * SIZE);
            Self::draw_world_poly(
                r,
                cam,
                &[
                    to_world(-w, rad),
                    to_world(w, rad),
                    to_world(w, -rad),
                    to_world(-w, -rad),
                ],
                WHEEL_COLOR,
            );
            // Rolling marker: the wheel-phase window from Car.draw.
            let phase = car.ctl[i].phase;
            let (a1, a2) = (phase, phase + 1.2);
            let (s1, s2) = (libm::sin(a1), libm::sin(a2));
            if !(s1 > 0.0 && s2 > 0.0) {
                let mut c1 = libm::cos(a1);
                let mut c2 = libm::cos(a2);
                if s1 > 0.0 {
                    c1 = c1.signum();
                }
                if s2 > 0.0 {
                    c2 = c2.signum();
                }
                Self::draw_world_poly(
                    r,
                    cam,
                    &[
                        to_world(-w, rad * c1),
                        to_world(w, rad * c1),
                        to_world(w, rad * c2),
                        to_world(-w, rad * c2),
                    ],
                    WHEEL_WHITE,
                );
            }
        }
        let (hx, hy) = car.hull_pos();
        let ha = car.hull_angle();
        let (ca, sa) = (libm::cos(ha), libm::sin(ha));
        for poly in [
            &HULL_POLY1[..],
            &HULL_POLY2[..],
            &HULL_POLY3[..],
            &HULL_POLY4[..],
        ] {
            let pts: Vec<[f64; 2]> = poly
                .iter()
                .map(|&[lx, ly]| {
                    let (sx, sy) = (lx * SIZE, ly * SIZE);
                    [hx + ca * sx - sa * sy, hy + sa * sx + ca * sy]
                })
                .collect();
            Self::draw_world_poly(r, cam, &pts, HULL_COLOR);
        }
    }

    fn draw_indicators(r: &mut Raster, live: &LiveState) {
        let (s, h) = (WINDOW_W / 40.0, WINDOW_H / 40.0);
        let bar = [
            window([WINDOW_W, WINDOW_H]),
            window([WINDOW_W, WINDOW_H - 5.0 * h]),
            window([0.0, WINDOW_H - 5.0 * h]),
            window([0.0, WINDOW_H]),
        ];
        r.fill_poly(&bar, [0, 0, 0]);

        let vertical = |place: f64, val: f64| {
            [
                window([place * s, WINDOW_H - (h + h * val)]),
                window([(place + 1.0) * s, WINDOW_H - (h + h * val)]),
                window([(place + 1.0) * s, WINDOW_H - h]),
                window([place * s, WINDOW_H - h]),
            ]
        };
        let horiz = |place: f64, val: f64| {
            [
                window([place * s, WINDOW_H - 4.0 * h]),
                window([(place + val) * s, WINDOW_H - 4.0 * h]),
                window([(place + val) * s, WINDOW_H - 2.0 * h]),
                window([place * s, WINDOW_H - 2.0 * h]),
            ]
        };
        let mut show = |value: f64, quad: [[f32; 2]; 4], rgb: [u8; 3]| {
            if value.abs() > 1e-4 {
                r.fill_poly(&quad, rgb);
            }
        };

        let speed = live.car.hull_speed();
        show(speed, vertical(5.0, 0.02 * speed), [255, 255, 255]);
        for (i, place) in [(0usize, 7.0), (1, 8.0), (2, 9.0), (3, 10.0)] {
            let omega = live.car.ctl[i].omega;
            let rgb = if i < 2 { [0, 0, 255] } else { [51, 0, 255] };
            show(omega, vertical(place, 0.01 * omega), rgb);
        }
        let steer = f64::from(live.car.joint_angle(0));
        show(steer, horiz(20.0, -10.0 * steer), [0, 255, 0]);
        let angvel = f64::from(live.car.bodies[live.car.hull].angvel());
        show(angvel, horiz(30.0, -0.8 * angvel), [255, 0, 0]);
    }

    fn draw_score(r: &mut Raster, live: &LiveState) {
        let score = 1000.0 * f64::from(live.visited_count) / live.track.tiles.len().max(1) as f64
            - 0.1 * f64::from(live.tick);
        let text = score_text(score);
        let scale = (42.0 * FRAME as f64 / WINDOW_H / 7.0) as f32;
        let width = (text.len() as f32 * 6.0 - 1.0) * scale;
        let [cx, cy] = window([60.0, WINDOW_H - WINDOW_H * 2.5 / 40.0]);
        draw_text(
            r,
            &text,
            cx - width / 2.0,
            cy - 3.5 * scale,
            scale,
            [255, 255, 255],
        );
    }
}

impl FrameRenderer<CarRacingState> for CarRacingRenderer {
    fn frame_shape(&self) -> (usize, usize, usize) {
        (FRAME, FRAME, 3)
    }

    fn render(&self, state: &CarRacingState, dst: &mut [u8]) {
        let CarRacingState::Live(live) = state else {
            unreachable!("render on a pending CarRacing state (kept out of observations)");
        };
        let mut r = Raster::new(FRAME as u32, FRAME as u32, SUPERSAMPLE);
        let cam = Camera::new(live);
        Self::draw_road(&mut r, &cam, &live.track);
        Self::draw_car(&mut r, &cam, &live.car);
        Self::draw_indicators(&mut r, live);
        Self::draw_score(&mut r, live);
        r.downsample_into(dst, FRAME, FRAME);
    }
}

/// Pixel observation encoder: CHW (3, 96, 96), raw 0-255 as f32.
pub struct CarRacingPixels;

impl reinfors_core::ActionView for CarRacingPixels {}

impl reinfors_core::StateEncoder for CarRacingPixels {
    type State = CarRacingState;

    fn encode(&self, state: &CarRacingState, _agent: usize) -> Vec<f32> {
        let (h, w, c) = CarRacingRenderer.frame_shape();
        let mut hwc = vec![0u8; h * w * c];
        CarRacingRenderer.render(state, &mut hwc);
        let mut out = Vec::new();
        hwc_to_chw_f32(&hwc, h, w, &mut out);
        out
    }

    fn obs_shape(&self) -> (usize, usize, usize) {
        let (h, w, c) = CarRacingRenderer.frame_shape();
        (c, h, w)
    }

    fn observation_space(&self) -> reinfors_core::Space {
        reinfors_core::Space::Box {
            shape: vec![3, FRAME, FRAME],
            low: 0.0,
            high: 255.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::{CarRacing, CarRacingState};
    use super::*;
    use reinfors_core::{Game, StateEncoder};

    fn live(seed: u32, ticks: usize) -> CarRacingState {
        let g = CarRacing::default();
        let mut s = CarRacingState::Live(Box::new(g.realize(seed)));
        for _ in 0..ticks {
            let t = g.step(&s, &[3]);
            s = t.next_state;
            if t.terminal {
                break;
            }
        }
        s
    }

    fn frame(state: &CarRacingState) -> Vec<u8> {
        let mut hwc = vec![0u8; FRAME * FRAME * 3];
        CarRacingRenderer.render(state, &mut hwc);
        hwc
    }

    #[test]
    fn structural_shape_and_space() {
        let enc = CarRacingPixels;
        assert_eq!(enc.obs_shape(), (3, FRAME, FRAME));
        let reinfors_core::Space::Box { shape, low, high } = enc.observation_space() else {
            panic!("pixel obs must be a Box space");
        };
        assert_eq!((shape, low, high), (vec![3, FRAME, FRAME], 0.0, 255.0));
        let row = enc.encode(&live(5, 0), 0);
        assert_eq!(row.len(), 3 * FRAME * FRAME);
        assert!(row.iter().all(|v| (0.0..=255.0).contains(v)));
    }

    #[test]
    fn render_is_deterministic_and_reward_independent() {
        let s = live(5, 10);
        assert_eq!(frame(&s), frame(&s));
        // The renderer reads only state; reward configuration cannot reach it. The HUD
        // score comes from canonical constants regardless of any Reward weights.
        let a = CarRacingPixels.encode(&s, 0);
        let b = CarRacingPixels.encode(&s, 0);
        assert_eq!(a, b);
    }

    #[test]
    fn chw_layout_indexes_channels_first() {
        let hwc = [10u8, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120];
        let mut out = Vec::new();
        crate::render::hwc_to_chw_f32(&hwc, 2, 2, &mut out);
        assert_eq!(
            out,
            vec![
                10.0, 40.0, 70.0, 100.0, // R plane
                20.0, 50.0, 80.0, 110.0, // G plane
                30.0, 60.0, 90.0, 120.0, // B plane
            ]
        );
    }

    #[test]
    fn semantic_probes_grass_road_car_and_hud() {
        let hwc = frame(&live(5, 60));
        let count = |pred: &dyn Fn(u8, u8, u8) -> bool| {
            hwc.as_chunks::<3>()
                .0
                .iter()
                .filter(|c| pred(c[0], c[1], c[2]))
                .count()
        };
        let grass = count(&|r, g, b| g > 180 && r > 80 && r < 130 && b > 80 && b < 130);
        let road = count(&|r, g, b| {
            (102..=112).contains(&r) && (102..=112).contains(&g) && (102..=112).contains(&b)
        });
        let hull = count(&|r, g, b| r > 150 && g < 60 && b < 60);
        assert!(grass > 500, "grass pixels: {grass}");
        assert!(road > 500, "road pixels: {road}");
        assert!(hull > 10, "hull pixels: {hull}");
        let bar_px = &hwc[((FRAME - 1) * FRAME + 2) * 3..][..3];
        assert_eq!(bar_px, [0, 0, 0], "indicator bar bottom-left");
    }

    #[test]
    fn zoom_animation_changes_early_frames() {
        assert_ne!(frame(&live(5, 0)), frame(&live(5, 30)));
    }

    #[test]
    fn score_text_matches_gym_formatting() {
        assert_eq!(score_text(0.0), "0000");
        assert_eq!(score_text(926.4), "0926");
        assert_eq!(score_text(-12.3), "-012");
        assert_eq!(score_text(7.9), "0008");
    }

    #[test]
    fn frame_hash_diagnostic() {
        // Non-gating: prints the canonical-state hash for the cross-platform comparison.
        let hwc = frame(&live(5, 0));
        let mut h: u64 = 0xcbf29ce484222325;
        for &b in &hwc {
            h = (h ^ u64::from(b)).wrapping_mul(0x100000001b3);
        }
        println!("car_racing frame hash (seed 5, tick 0): {h:016x}");
    }

    #[test]
    #[ignore = "writes a debug PNG; run with -- --ignored"]
    fn dump_frame_png() {
        let dir = std::env::var("CARRACING_DUMP_DIR").unwrap_or_else(|_| "/tmp".into());
        for (seed, ticks) in [(5u32, 0usize), (5, 60), (11, 200)] {
            let hwc = frame(&live(seed, ticks));
            let path =
                std::path::PathBuf::from(&dir).join(format!("car_racing_s{seed}_t{ticks}.png"));
            crate::render::write_png(&path, &hwc, FRAME as u32, FRAME as u32).unwrap();
            println!("wrote {}", path.display());
        }
    }

    #[test]
    #[ignore = "benchmark; run with -- --ignored --nocapture"]
    fn benchmark_render() {
        let s = live(5, 60);
        let t0 = std::time::Instant::now();
        let mut n = 0u64;
        while t0.elapsed().as_millis() < 1000 {
            std::hint::black_box(CarRacingPixels.encode(&s, 0));
            n += 1;
        }
        println!(
            "pixel encode: {:.0}us",
            t0.elapsed().as_micros() as f64 / n as f64
        );
    }
}

#[cfg(test)]
mod fixture_tests {
    use super::super::track::{Track, TrackPoint};
    use super::super::CarRacingState;
    use super::*;

    /// Renders the geometry exported by scripts/car_racing_fixture.py next to the gym
    /// frame for the side-by-side eyeball review.
    #[test]
    #[ignore = "needs CARRACING_FIXTURE_DIR from scripts/car_racing_fixture.py"]
    fn render_fixture_side_by_side() {
        let dir = std::path::PathBuf::from(
            std::env::var("CARRACING_FIXTURE_DIR").expect("set CARRACING_FIXTURE_DIR"),
        );
        let fixture: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join("fixture.json")).unwrap())
                .unwrap();
        let points: Vec<TrackPoint> = fixture["track"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| TrackPoint {
                beta: p[0].as_f64().unwrap(),
                x: p[1].as_f64().unwrap(),
                y: p[2].as_f64().unwrap(),
            })
            .collect();
        let track = std::sync::Arc::new(Track::from_points(points, false));
        let p0 = track.points[0];
        let mut live = super::super::LiveState {
            seed: 0,
            track,
            car: super::super::dynamics::CarWorld::new(p0.beta, p0.x, p0.y),
            tick: fixture["tick"].as_u64().unwrap() as u32,
            visited: Vec::new(),
            visited_count: 0,
            wheel_tiles: Default::default(),
            done: false,
        };
        live.visited = vec![0u64; live.track.tiles.len().div_ceil(64)];

        use rapier2d::prelude::*;
        let set_pose = |body: &mut RigidBody, pos: &serde_json::Value, angle: f64| {
            body.set_translation(
                Vector::new(
                    pos[0].as_f64().unwrap() as Real,
                    pos[1].as_f64().unwrap() as Real,
                ),
                true,
            );
            body.set_rotation(Rotation::from_angle(angle as Real), true);
        };
        let hull = live.car.hull;
        set_pose(
            &mut live.car.bodies[hull],
            &fixture["hull"]["pos"],
            fixture["hull"]["angle"].as_f64().unwrap(),
        );
        let hv = &fixture["hull"]["linvel"];
        live.car.bodies[hull].set_linvel(
            Vector::new(
                hv[0].as_f64().unwrap() as Real,
                hv[1].as_f64().unwrap() as Real,
            ),
            true,
        );
        live.car.bodies[hull].set_angvel(fixture["hull"]["angvel"].as_f64().unwrap() as Real, true);
        for (i, w) in fixture["wheels"].as_array().unwrap().iter().enumerate() {
            let handle = live.car.wheels[i];
            set_pose(
                &mut live.car.bodies[handle],
                &w["pos"],
                w["angle"].as_f64().unwrap(),
            );
            live.car.ctl[i].omega = w["omega"].as_f64().unwrap();
            live.car.ctl[i].phase = w["phase"].as_f64().unwrap();
        }

        let state = CarRacingState::Live(Box::new(live));
        let mut hwc = vec![0u8; FRAME * FRAME * 3];
        CarRacingRenderer.render(&state, &mut hwc);
        let out = dir.join("rust_frame.png");
        crate::render::write_png(&out, &hwc, FRAME as u32, FRAME as u32).unwrap();
        println!("wrote {} (compare with gym_frame.png)", out.display());
    }
}
