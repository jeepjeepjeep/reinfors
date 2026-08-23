//! CarRacing frame renderer: Gymnasium's draw list reproduced through the shared
//! rendering seam. World content follows the window transform (1000x800, camera-follow
//! with the first-second zoom animation, vertical flip); indicators and the score draw
//! unflipped, exactly as pygame composes them; the whole frame supersamples then
//! box-filters to 96x96, mirroring the smoothscale step.

use super::dynamics::{
    CarWorld, HULL_POLY1, HULL_POLY2, HULL_POLY3, HULL_POLY4, SIZE, WHEEL_R, WHEEL_W,
};
use super::score_font::SCORE_GLYPHS;
use super::track::{Track, PLAYFIELD, SCALE};
use super::{CarRacingState, LiveState};
use crate::render::{FrameRenderer, Raster};

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
    rot: [f64; 2],
    trans: [f64; 2],
}

impl Camera {
    fn new(live: &LiveState) -> Camera {
        // Gym's reset performs one implicit step before its first observation, so its
        // camera clock reads (tick + 1)/FPS relative to our tick counter.
        let t = f64::from(live.tick + 1) / super::track::FPS;
        let zoom = 0.1 * SCALE * (1.0 - t).max(0.0) + ZOOM * SCALE * t.min(1.0);
        let angle = -live.car.hull_angle();
        let (px, py) = live.car.hull_pos();
        let (ca, sa) = (libm::cos(angle), libm::sin(angle));
        let (sx, sy) = (-px * zoom, -py * zoom);
        Camera {
            zoom,
            rot: [ca, sa],
            trans: [
                ca * sx - sa * sy + WINDOW_W / 2.0,
                sa * sx + ca * sy + WINDOW_H / 4.0,
            ],
        }
    }

    /// World-space AABB of the visible frame (inverse-mapped corners, small margin),
    /// for culling and clipping static geometry before any per-point transform. The
    /// margin keeps clip edges strictly offscreen so visible pixels are unaffected.
    fn view_aabb(&self) -> [[f64; 2]; 2] {
        let [ca, sa] = self.rot;
        let (mut lo, mut hi) = ([f64::MAX; 2], [f64::MIN; 2]);
        for corner in [
            [0.0, 0.0],
            [WINDOW_W, 0.0],
            [0.0, WINDOW_H],
            [WINDOW_W, WINDOW_H],
        ] {
            let dx = corner[0] - self.trans[0];
            let dy = corner[1] - self.trans[1];
            let x = (ca * dx + sa * dy) / self.zoom;
            let y = (-sa * dx + ca * dy) / self.zoom;
            lo = [lo[0].min(x), lo[1].min(y)];
            hi = [hi[0].max(x), hi[1].max(y)];
        }
        let m = 8.0 / self.zoom;
        [[lo[0] - m, lo[1] - m], [hi[0] + m, hi[1] + m]]
    }

    /// World point to final-frame coordinates (flipped, like pygame's world pass).
    fn world(&self, [x, y]: [f64; 2]) -> [f32; 2] {
        let [ca, sa] = self.rot;
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

/// Gym's f"{reward:04.0f}": ties-to-even, minimum width 4 with sign-aware zero padding
/// ("-012", "-1000", "12345"), and Python's negative-zero quirk ("-000").
pub(crate) fn score_text(score: f64) -> String {
    let rounded = score.round_ties_even();
    if rounded == 0.0 && rounded.is_sign_negative() {
        return "-000".to_string();
    }
    format!("{:04}", rounded as i64)
}

pub(crate) struct CarRacingRenderer;

impl CarRacingRenderer {
    fn draw_world_poly(r: &mut Raster, cam: &Camera, pts: &[[f64; 2]], rgb: [u8; 3]) {
        let mapped: smallvec::SmallVec<[[f32; 2]; 8]> = pts.iter().map(|&p| cam.world(p)).collect();
        r.fill_poly(&mapped, rgb);
    }

    fn draw_road(r: &mut Raster, cam: &Camera, track: &Track) {
        let [vlo, vhi] = cam.view_aabb();
        let visible = |quad: &[[f64; 2]]| {
            let (mut lo, mut hi) = ([f64::MAX; 2], [f64::MIN; 2]);
            for p in quad {
                lo = [lo[0].min(p[0]), lo[1].min(p[1])];
                hi = [hi[0].max(p[0]), hi[1].max(p[1])];
            }
            hi[0] >= vlo[0] && hi[1] >= vlo[1] && lo[0] <= vhi[0] && lo[1] <= vhi[1]
        };
        let g = PLAYFIELD / 20.0;
        for x in (-20i32..20).step_by(2) {
            for y in (-20i32..20).step_by(2) {
                let (fx, fy) = (f64::from(x) * g, f64::from(y) * g);
                if fx + g < vlo[0] || fy + g < vlo[1] || fx > vhi[0] || fy > vhi[1] {
                    continue;
                }
                // Clip to the view in world space: an unclipped square's screen bbox
                // can span the whole pixmap, and fill cost scales with bbox.
                let (x0, x1) = (fx.max(vlo[0]), (fx + g).min(vhi[0]));
                let (y0, y1) = (fy.max(vlo[1]), (fy + g).min(vhi[1]));
                Self::draw_world_poly(
                    r,
                    cam,
                    &[[x1, y0], [x0, y0], [x0, y1], [x1, y1]],
                    GRASS_COLOR,
                );
            }
        }
        for (i, tile) in track.tiles.iter().enumerate() {
            if !visible(&tile.quad) {
                continue;
            }
            let shade = (0.01 * (i % 3) as f64 * 255.0) as u8;
            let rgb = ROAD_COLOR.map(|c| c.saturating_add(shade));
            Self::draw_world_poly(r, cam, &tile.quad, rgb);
        }
        for border in &track.borders {
            if !visible(&border.quad) {
                continue;
            }
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
            let pts: smallvec::SmallVec<[[f64; 2]; 8]> = poly
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
        // Gym centers the rendered string at window (60, H - H*2.5/40); the atlas
        // glyphs are already in supersample space, so lay out the pen there directly.
        let ss = f64::from(SUPERSAMPLE) * FRAME as f64;
        let cx = (60.0 * ss / WINDOW_W) as f32;
        let cy = ((WINDOW_H - WINDOW_H * 2.5 / 40.0) * ss / WINDOW_H) as f32;
        blit_score_text(r, &score_text(score), cx, cy);
    }
}

/// Center-anchored atlas-glyph layout for the HUD score string.
fn blit_score_text(r: &mut Raster, text: &str, cx: f32, cy: f32) {
    let glyph = |ch: char| SCORE_GLYPHS.iter().find(|g| g.ch == ch);
    let width: f32 = text.chars().filter_map(glyph).map(|g| g.advance).sum();
    let height = SCORE_GLYPHS[0].height;
    let mut pen = cx - width / 2.0;
    let top = (cy - height as f32 / 2.0).round() as i32;
    for g in text.chars().filter_map(glyph) {
        r.blit_alpha(
            pen.round() as i32,
            top,
            g.width,
            g.height,
            g.alpha,
            [255, 255, 255],
        );
        pen += g.advance;
    }
}

thread_local! {
    // Rasterization scratch (590 KB at 4x supersampling); reused across encodes.
    static RASTER: std::cell::RefCell<Raster> =
        std::cell::RefCell::new(Raster::new(FRAME as u32, FRAME as u32, SUPERSAMPLE));
}

fn rasterize(live: &super::LiveState, r: &mut Raster) {
    let cam = Camera::new(live);
    let [vlo, vhi] = cam.view_aabb();
    let b = PLAYFIELD;
    if vlo[0] >= -b && vlo[1] >= -b && vhi[0] <= b && vhi[1] <= b {
        // View fully inside the playfield: the background quad covers every pixel,
        // so one solid flood replaces clear + path-filled quad byte-identically.
        r.fill_solid(BG_COLOR);
    } else {
        r.clear();
        CarRacingRenderer::draw_world_poly(
            r,
            &cam,
            &[[b, b], [b, -b], [-b, -b], [-b, b]],
            BG_COLOR,
        );
    }
    CarRacingRenderer::draw_road(r, &cam, &live.track);
    CarRacingRenderer::draw_car(r, &cam, &live.car);
    CarRacingRenderer::draw_indicators(r, live);
    CarRacingRenderer::draw_score(r, live);
}

impl FrameRenderer<CarRacingState> for CarRacingRenderer {
    fn frame_shape(&self) -> (usize, usize, usize) {
        (FRAME, FRAME, 3)
    }

    fn render(&self, state: &CarRacingState, dst: &mut [u8]) {
        let CarRacingState::Live(live) = state else {
            unreachable!("render on a pending CarRacing state (kept out of observations)");
        };
        RASTER.with_borrow_mut(|r| {
            rasterize(live, r);
            r.downsample_into(dst, FRAME, FRAME);
        });
    }
}

/// Pixel observation encoder: CHW (3, 96, 96), raw 0-255 as f32.
pub struct CarRacingPixels;

impl reinfors_core::ActionView for CarRacingPixels {}

impl reinfors_core::StateEncoder for CarRacingPixels {
    type State = CarRacingState;

    fn encode(&self, state: &CarRacingState, _agent: usize) -> Vec<f32> {
        let CarRacingState::Live(live) = state else {
            unreachable!("encode on a pending CarRacing state (kept out of observations)");
        };
        let mut out = Vec::new();
        RASTER.with_borrow_mut(|r| {
            rasterize(live, r);
            r.downsample_chw_f32(&mut out, FRAME, FRAME);
        });
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
    #[test]
    fn score_atlas_is_complete_and_well_formed() {
        let chars: Vec<char> = SCORE_GLYPHS.iter().map(|g| g.ch).collect();
        let mut expected: Vec<char> = "0123456789-".chars().collect();
        let mut sorted = chars.clone();
        sorted.sort_unstable();
        expected.sort_unstable();
        assert_eq!(sorted, expected, "atlas must cover exactly 0-9 and minus");
        for g in &SCORE_GLYPHS {
            assert_eq!(
                g.alpha.len(),
                g.width * g.height,
                "glyph {:?} bitmap size",
                g.ch
            );
            assert!(g.width > 0 && g.height > 0 && g.height == SCORE_GLYPHS[0].height);
            assert!(
                g.advance > 0.0 && g.advance < 64.0,
                "glyph {:?} advance",
                g.ch
            );
            assert!(g.alpha.iter().any(|&a| a > 0), "glyph {:?} is blank", g.ch);
        }
    }

    #[test]
    fn score_text_blit_is_deterministic_and_draws_every_glyph() {
        let text = "-0123456789";
        let render = || {
            let mut r = Raster::new(FRAME as u32, FRAME as u32, SUPERSAMPLE);
            r.fill_solid([0, 0, 0]);
            blit_score_text(
                &mut r,
                text,
                (FRAME * SUPERSAMPLE as usize) as f32 / 2.0,
                40.0,
            );
            r
        };
        let (a, b) = (render(), render());
        assert_eq!(a.pixmap.data(), b.pixmap.data());

        // Every glyph's pen band must contain ink.
        let glyph = |ch: char| SCORE_GLYPHS.iter().find(|g| g.ch == ch).unwrap();
        let width: f32 = text.chars().map(|c| glyph(c).advance).sum();
        let mut pen = (FRAME * SUPERSAMPLE as usize) as f32 / 2.0 - width / 2.0;
        let data = a.pixmap.data();
        let pw = a.pixmap.width() as usize;
        for ch in text.chars() {
            let g = glyph(ch);
            let x0 = pen.round() as usize;
            let ink = (0..a.pixmap.height() as usize)
                .any(|y| (x0..x0 + g.width).any(|x| data[(y * pw + x) * 4] > 0));
            assert!(ink, "glyph {ch:?} left no ink in its band");
            pen += g.advance;
        }
    }

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
    fn chw_encode_matches_the_hwc_debug_path() {
        // The encoder downsamples straight to CHW; the debug/PNG tooling goes through
        // HWC. Divergence here would invalidate every HWC-based pixel test.
        let s = live(5, 60);
        let direct = CarRacingPixels.encode(&s, 0);
        let mut via_hwc = Vec::new();
        crate::render::hwc_to_chw_f32(&frame(&s), FRAME, FRAME, &mut via_hwc);
        assert_eq!(direct, via_hwc);
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
        assert_eq!(score_text(2.5), "0002", "ties to even");
        assert_eq!(score_text(3.5), "0004", "ties to even");
        assert_eq!(score_text(-12.5), "-012");
        assert_eq!(score_text(-1000.0), "-1000", "width 4 is a minimum");
        assert_eq!(score_text(12345.6), "12346");
        assert_eq!(score_text(-0.4), "-000", "python negative-zero quirk");
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
        let n_tiles = track.tiles.len();
        let visited_count = fixture["tile_visited_count"].as_u64().unwrap() as u32;
        let mut visited = vec![0u64; n_tiles.div_ceil(64)];
        for id in 0..visited_count as usize {
            visited[id / 64] |= 1 << (id % 64);
        }
        let mut live = super::super::LiveState {
            seed: 0,
            track,
            car: super::super::dynamics::CarWorld::new(p0.beta, p0.x, p0.y),
            tick: fixture["tick"].as_u64().unwrap() as u32,
            visited,
            visited_count,
            wheel_tiles: Default::default(),
            new_lap: false,
            done: false,
        };
        // The HUD score derives from visited_count and tick; cross-check against the
        // exported gym reward so a reconstruction mismatch is loud.
        let our_score = 1000.0 * f64::from(visited_count) / n_tiles as f64
            - 0.1 * fixture["tick"].as_f64().unwrap();
        let gym_reward = fixture["reward"].as_f64().unwrap();
        assert!(
            (our_score - gym_reward).abs() < 1.0,
            "reconstructed score {our_score} vs gym reward {gym_reward}"
        );

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
        crate::render::write_png(
            &dir.join("rust_frame.png"),
            &hwc,
            FRAME as u32,
            FRAME as u32,
        )
        .unwrap();

        let gym = tiny_skia::Pixmap::load_png(dir.join("gym_frame.png")).unwrap();
        assert_eq!((gym.width(), gym.height()), (FRAME as u32, FRAME as u32));
        let gd = gym.data();
        let mut total_abs = 0u64;
        let mut gross = 0usize;
        let mut side = vec![0u8; FRAME * (FRAME * 3) * 3];
        for y in 0..FRAME {
            for x in 0..FRAME {
                let (gi, ri) = ((y * FRAME + x) * 4, (y * FRAME + x) * 3);
                let mut px_diff = 0u32;
                for c in 0..3 {
                    let d = i32::from(gd[gi + c]).abs_diff(i32::from(hwc[ri + c]));
                    total_abs += u64::from(d);
                    px_diff = px_diff.max(d);
                }
                let row = y * FRAME * 3 * 3;
                for c in 0..3 {
                    side[row + x * 3 + c] = gd[gi + c];
                    side[row + (FRAME + x) * 3 + c] = hwc[ri + c];
                    side[row + (2 * FRAME + x) * 3 + c] = px_diff.min(255) as u8;
                }
                if px_diff > 32 {
                    gross += 1;
                }
            }
        }
        crate::render::write_png(
            &dir.join("side_by_side.png"),
            &side,
            (FRAME * 3) as u32,
            FRAME as u32,
        )
        .unwrap();
        let mean = total_abs as f64 / (FRAME * FRAME * 3) as f64;
        let gross_frac = gross as f64 / (FRAME * FRAME) as f64;
        println!(
            "mean abs diff {mean:.1}, gross(>32) pixel fraction {gross_frac:.3}; \
             wrote side_by_side.png (gym | rust | diff)"
        );
        // Tolerances are deliberately loose: renderers differ in AA, font glyphs, and
        // physics-trajectory drift; they bound "same scene", not pixel parity.
        assert!(
            mean < 30.0,
            "mean abs diff {mean:.1} exceeds the scene tolerance"
        );
        assert!(
            gross_frac < 0.25,
            "gross-diff fraction {gross_frac:.3} exceeds the scene tolerance"
        );
    }
}
