//! Procedural track generation, ported from Gymnasium's CarRacing `_create_track`.
//!
//! All transcendentals go through `libm` so a seed regenerates the identical track on
//! every platform (rapier's `enhanced-determinism` only covers physics math).

use reinfors_core::{Rng, SplitMix64};

pub const SCALE: f64 = 6.0;
pub const TRACK_RAD: f64 = 900.0 / SCALE;
pub const PLAYFIELD: f64 = 2000.0 / SCALE;
pub const FPS: f64 = 50.0;
pub const TRACK_DETAIL_STEP: f64 = 21.0 / SCALE;
pub const TRACK_TURN_RATE: f64 = 0.31;
pub const TRACK_WIDTH: f64 = 40.0 / SCALE;
const CHECKPOINTS: usize = 12;
const MAX_ATTEMPTS: u32 = 64;

const GRID_CELL: f64 = 10.0;
const GRID_DIM: usize = (2.0 * PLAYFIELD / GRID_CELL) as usize + 1;

#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub struct TrackPoint {
    pub beta: f64,
    pub x: f64,
    pub y: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct Tile {
    pub quad: [[f64; 2]; 4],
}

pub struct Track {
    pub points: Vec<TrackPoint>,
    pub tiles: Vec<Tile>,
    pub fallback: bool,
    grid: Vec<Vec<u32>>,
}

fn uniform(rng: &mut SplitMix64, lo: f64, hi: f64) -> f64 {
    lo + (hi - lo) * rng.unit()
}

fn tile_quad(p1: &TrackPoint, p2: &TrackPoint) -> Tile {
    let (c1, s1) = (libm::cos(p1.beta), libm::sin(p1.beta));
    let (c2, s2) = (libm::cos(p2.beta), libm::sin(p2.beta));
    Tile {
        quad: [
            [p1.x - TRACK_WIDTH * c1, p1.y - TRACK_WIDTH * s1],
            [p1.x + TRACK_WIDTH * c1, p1.y + TRACK_WIDTH * s1],
            [p2.x + TRACK_WIDTH * c2, p2.y + TRACK_WIDTH * s2],
            [p2.x - TRACK_WIDTH * c2, p2.y - TRACK_WIDTH * s2],
        ],
    }
}

impl Track {
    /// Deterministic generation from one chance outcome. Bounded retries; an analytic ring
    /// track is the fallback so every declared outcome yields a valid track.
    pub fn generate(seed: u32) -> Track {
        Track::generate_with_attempts(seed, MAX_ATTEMPTS)
    }

    pub(crate) fn generate_with_attempts(seed: u32, max_attempts: u32) -> Track {
        let mut rng = SplitMix64::new(0xCA75_EED0 ^ u64::from(seed));
        for _ in 0..max_attempts {
            if let Some(points) = try_generate(&mut rng) {
                return Track::from_points(points, false);
            }
        }
        Track::from_points(ring_points(), true)
    }

    fn from_points(points: Vec<TrackPoint>, fallback: bool) -> Track {
        let n = points.len();
        let tiles: Vec<Tile> = (0..n)
            .map(|i| tile_quad(&points[i], &points[if i == 0 { n - 1 } else { i - 1 }]))
            .collect();
        let mut grid = vec![Vec::new(); GRID_DIM * GRID_DIM];
        for (id, t) in tiles.iter().enumerate() {
            let xs = t.quad.iter().map(|p| p[0]);
            let ys = t.quad.iter().map(|p| p[1]);
            let (x0, x1) = (
                xs.clone().fold(f64::MAX, f64::min),
                xs.fold(f64::MIN, f64::max),
            );
            let (y0, y1) = (
                ys.clone().fold(f64::MAX, f64::min),
                ys.fold(f64::MIN, f64::max),
            );
            for cy in cell_of(y0)..=cell_of(y1) {
                for cx in cell_of(x0)..=cell_of(x1) {
                    grid[cy * GRID_DIM + cx].push(id as u32);
                }
            }
        }
        Track {
            points,
            tiles,
            fallback,
            grid,
        }
    }

    /// Tile ids whose AABB-covered cells contain `(x, y)`.
    pub fn candidate_tiles(&self, x: f64, y: f64) -> &[u32] {
        &self.grid[cell_of(y) * GRID_DIM + cell_of(x)]
    }
}

fn cell_of(v: f64) -> usize {
    (((v + PLAYFIELD) / GRID_CELL).max(0.0) as usize).min(GRID_DIM - 1)
}

fn ring_points() -> Vec<TrackPoint> {
    let n = 100;
    (0..n)
        .map(|i| {
            let a = 2.0 * std::f64::consts::PI * (i as f64) / (n as f64);
            TrackPoint {
                beta: a,
                x: TRACK_RAD * libm::cos(a),
                y: TRACK_RAD * libm::sin(a),
            }
        })
        .collect()
}

fn try_generate(rng: &mut SplitMix64) -> Option<Vec<TrackPoint>> {
    let tau = 2.0 * std::f64::consts::PI;
    let mut checkpoints = Vec::with_capacity(CHECKPOINTS);
    let mut start_alpha = 0.0;
    for c in 0..CHECKPOINTS {
        let noise = uniform(rng, 0.0, tau / CHECKPOINTS as f64);
        let mut alpha = tau * c as f64 / CHECKPOINTS as f64 + noise;
        let mut rad = uniform(rng, TRACK_RAD / 3.0, TRACK_RAD);
        if c == 0 {
            alpha = 0.0;
            rad = 1.5 * TRACK_RAD;
        }
        if c == CHECKPOINTS - 1 {
            alpha = tau * c as f64 / CHECKPOINTS as f64;
            start_alpha = tau * (-0.5) / CHECKPOINTS as f64;
            rad = 1.5 * TRACK_RAD;
        }
        checkpoints.push((alpha, rad * libm::cos(alpha), rad * libm::sin(alpha)));
    }

    let (mut x, mut y, mut beta) = (1.5 * TRACK_RAD, 0.0, 0.0);
    let mut dest_i = 0usize;
    let mut laps = 0;
    let mut track: Vec<(f64, f64, f64, f64)> = Vec::new();
    let mut no_freeze = 2500;
    let mut visited_other_side = false;
    loop {
        let mut alpha = libm::atan2(y, x);
        if visited_other_side && alpha > 0.0 {
            laps += 1;
            visited_other_side = false;
        }
        if alpha < 0.0 {
            visited_other_side = true;
            alpha += tau;
        }
        let (mut dest_alpha, mut dest_x, mut dest_y);
        loop {
            let mut failed = true;
            loop {
                (dest_alpha, dest_x, dest_y) = checkpoints[dest_i % CHECKPOINTS];
                if alpha <= dest_alpha {
                    failed = false;
                    break;
                }
                dest_i += 1;
                if dest_i.is_multiple_of(CHECKPOINTS) {
                    break;
                }
            }
            if !failed {
                break;
            }
            alpha -= tau;
        }
        let r1x = libm::cos(beta);
        let r1y = libm::sin(beta);
        let p1x = -r1y;
        let p1y = r1x;
        let dest_dx = dest_x - x;
        let dest_dy = dest_y - y;
        let mut proj = r1x * dest_dx + r1y * dest_dy;
        while beta - alpha > 1.5 * std::f64::consts::PI {
            beta -= tau;
        }
        while beta - alpha < -1.5 * std::f64::consts::PI {
            beta += tau;
        }
        let prev_beta = beta;
        proj *= SCALE;
        if proj > 0.3 {
            beta -= TRACK_TURN_RATE.min((0.001 * proj).abs());
        }
        if proj < -0.3 {
            beta += TRACK_TURN_RATE.min((0.001 * proj).abs());
        }
        x += p1x * TRACK_DETAIL_STEP;
        y += p1y * TRACK_DETAIL_STEP;
        track.push((alpha, prev_beta * 0.5 + beta * 0.5, x, y));
        if laps > 4 {
            break;
        }
        no_freeze -= 1;
        if no_freeze == 0 {
            break;
        }
    }

    let (mut i1, mut i2) = (-1i64, -1i64);
    let mut i = track.len() as i64;
    loop {
        i -= 1;
        if i == 0 {
            return None;
        }
        let pass_through_start =
            track[i as usize].0 > start_alpha && track[(i - 1) as usize].0 <= start_alpha;
        if pass_through_start && i2 == -1 {
            i2 = i;
        } else if pass_through_start && i1 == -1 {
            i1 = i;
            break;
        }
    }
    let track = &track[i1 as usize..(i2 - 1) as usize];
    if track.is_empty() {
        return None;
    }

    let first_beta = track[0].1;
    let first_perp_x = libm::cos(first_beta);
    let first_perp_y = libm::sin(first_beta);
    let well_glued = libm::sqrt(
        (first_perp_x * (track[0].2 - track[track.len() - 1].2)).powi(2)
            + (first_perp_y * (track[0].3 - track[track.len() - 1].3)).powi(2),
    );
    if well_glued > TRACK_DETAIL_STEP {
        return None;
    }
    Some(
        track
            .iter()
            .map(|&(_a, beta, x, y)| TrackPoint { beta, x, y })
            .collect(),
    )
}
