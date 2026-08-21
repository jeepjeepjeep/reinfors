//! Faithful reimplementation of Gymnasium's CarRacing-v3 (discrete actions).
//!
//! Physics runs on rapier2d with the tire model ported from `car_dynamics.py`; tile
//! contact is a geometric task-level approximation (per-wheel footprint overlap), not
//! physics sensors. Trajectories are not Gymnasium-compatible; see the catalogue entry.

pub mod codec;
pub mod dynamics;
pub mod track;

use std::num::NonZeroU32;
use std::sync::Arc;

use dynamics::CarWorld;
use reinfors_core::{ActionView, Actor, ChanceDist, Game, Reward, Space, StateEncoder, Transition};
use track::{Track, PLAYFIELD, TRACK_RAD};

pub const N_ACTIONS: usize = 5;
const SEED_SPACE: u32 = u32::MAX;
const OBS_DIM: usize = 21;

#[derive(Clone)]
pub struct LiveState {
    pub seed: u32,
    pub track: Arc<Track>,
    pub car: CarWorld,
    pub tick: u32,
    pub visited: Vec<u64>,
    pub visited_count: u32,
    pub wheel_tiles: [Vec<u32>; 4],
    pub done: bool,
}

#[derive(Clone, Default)]
pub enum CarRacingState {
    #[default]
    Pending,
    Live(Box<LiveState>),
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CarRacingEvent {
    pub new_tiles: u32,
    pub total_tiles: u32,
    pub off_playfield: bool,
}

/// Canonical coefficients match Gymnasium. The off-playfield override is structural:
/// custom weights rescale components but never change the replacement rule.
#[derive(Clone, Copy, Debug)]
pub struct CarRacingReward {
    pub tile: f64,
    pub step: f64,
    pub off_playfield: f64,
}

impl Default for CarRacingReward {
    fn default() -> Self {
        CarRacingReward {
            tile: 1000.0,
            step: -0.1,
            off_playfield: -100.0,
        }
    }
}

impl Reward for CarRacingReward {
    type Event = CarRacingEvent;

    fn step_reward(&self, e: &CarRacingEvent, _agent: usize) -> f64 {
        if e.off_playfield {
            self.off_playfield
        } else {
            self.tile * f64::from(e.new_tiles) / f64::from(e.total_tiles.max(1)) + self.step
        }
    }
}

#[derive(Clone)]
pub struct CarRacing {
    pub lap_complete_percent: f64,
    pub max_ticks: Option<usize>,
}

impl Default for CarRacing {
    fn default() -> Self {
        CarRacing {
            lap_complete_percent: 0.95,
            max_ticks: Some(1000),
        }
    }
}

impl CarRacing {
    pub fn validate(&self) -> Result<(), String> {
        if !(0.0..=1.0).contains(&self.lap_complete_percent) || self.lap_complete_percent.is_nan() {
            return Err(format!(
                "lap_complete_percent must be in [0, 1], got {}",
                self.lap_complete_percent
            ));
        }
        if self.max_ticks == Some(0) {
            return Err("max_ticks must be at least 1".to_string());
        }
        Ok(())
    }

    pub(crate) fn realize(&self, seed: u32) -> LiveState {
        self.realize_from(Arc::new(Track::generate(seed)), seed)
    }

    #[cfg(test)]
    pub(crate) fn realize_with_attempts(&self, seed: u32, max_attempts: u32) -> LiveState {
        self.realize_from(
            Arc::new(Track::generate_with_attempts(seed, max_attempts)),
            seed,
        )
    }

    fn realize_from(&self, track: Arc<Track>, seed: u32) -> LiveState {
        let p0 = track.points[0];
        let car = CarWorld::new(p0.beta, p0.x, p0.y);
        let n_tiles = track.tiles.len();
        let mut live = LiveState {
            seed,
            track,
            car,
            tick: 0,
            visited: vec![0u64; n_tiles.div_ceil(64)],
            visited_count: 0,
            wheel_tiles: Default::default(),
            done: false,
        };
        // Gym's reset() runs one no-action step, which registers the starting tiles.
        self.contact_pass(&mut live);
        live
    }

    /// Rebuild derived wheel-contact state after a snapshot restore, without touching the
    /// visited bitset (restored tiles are authoritative; geometry only refreshes contacts).
    pub(crate) fn contact_pass_derived(&self, live: &mut LiveState) {
        let visited = live.visited.clone();
        let count = live.visited_count;
        self.contact_pass(live);
        live.visited = visited;
        live.visited_count = count;
    }

    /// Refresh per-wheel tile sets from geometry; returns (new tiles, entered tile zero).
    fn contact_pass(&self, live: &mut LiveState) -> (u32, bool) {
        let mut new_tiles = 0u32;
        let mut entered_zero = false;
        for i in 0..4 {
            let quad = live.car.wheel_quad(i);
            let mut cur: Vec<u32> = Vec::with_capacity(4);
            let (mut x0, mut y0, mut x1, mut y1) = (f64::MAX, f64::MAX, f64::MIN, f64::MIN);
            for [x, y] in quad {
                x0 = x0.min(x);
                y0 = y0.min(y);
                x1 = x1.max(x);
                y1 = y1.max(y);
            }
            for &(cx, cy) in &[(x0, y0), (x1, y0), (x0, y1), (x1, y1)] {
                for &id in live.track.candidate_tiles(cx, cy) {
                    if !cur.contains(&id)
                        && quad_overlap(&quad, &live.track.tiles[id as usize].quad)
                    {
                        cur.push(id);
                    }
                }
            }
            for &id in &cur {
                if !live.wheel_tiles[i].contains(&id) {
                    if id == 0 {
                        entered_zero = true;
                    }
                    let (word, bit) = ((id / 64) as usize, id % 64);
                    if live.visited[word] & (1 << bit) == 0 {
                        live.visited[word] |= 1 << bit;
                        live.visited_count += 1;
                        new_tiles += 1;
                    }
                }
            }
            live.car.ctl[i].on_road = !cur.is_empty();
            live.wheel_tiles[i] = cur;
        }
        (new_tiles, entered_zero)
    }
}

/// Convex-quad overlap via separating axes from both quads' edges.
fn quad_overlap(a: &[[f64; 2]; 4], b: &[[f64; 2]; 4]) -> bool {
    for quad in [a, b] {
        for i in 0..4 {
            let [ax, ay] = quad[i];
            let [bx, by] = quad[(i + 1) % 4];
            let (nx, ny) = (by - ay, ax - bx);
            let proj = |q: &[[f64; 2]; 4]| {
                let mut lo = f64::MAX;
                let mut hi = f64::MIN;
                for [x, y] in q {
                    let d = nx * x + ny * y;
                    lo = lo.min(d);
                    hi = hi.max(d);
                }
                (lo, hi)
            };
            let (alo, ahi) = proj(a);
            let (blo, bhi) = proj(b);
            if ahi < blo || bhi < alo {
                return false;
            }
        }
    }
    true
}

impl Game for CarRacing {
    type State = CarRacingState;
    type Event = CarRacingEvent;

    fn num_agents(&self) -> usize {
        1
    }

    fn action_count(&self) -> usize {
        N_ACTIONS
    }

    fn actor(&self, state: &CarRacingState) -> Actor {
        match state {
            CarRacingState::Pending => Actor::Chance,
            CarRacingState::Live(_) => Actor::Agent(0),
        }
    }

    fn legal_actions(&self, state: &CarRacingState, _agent: usize) -> Vec<usize> {
        match state {
            CarRacingState::Pending => Vec::new(),
            CarRacingState::Live(l) if l.done => Vec::new(),
            CarRacingState::Live(_) => (0..N_ACTIONS).collect(),
        }
    }

    fn chance_enumerable(&self) -> bool {
        false
    }

    fn searchable_chance_enumerable(&self) -> bool {
        true
    }

    fn chance_node(&self, state: &CarRacingState) -> ChanceDist {
        assert!(
            matches!(state, CarRacingState::Pending),
            "chance_node on a realized CarRacing state"
        );
        ChanceDist::SampleOnlyUniform(NonZeroU32::new(SEED_SPACE).unwrap())
    }

    fn apply_chance_node(
        &self,
        state: &CarRacingState,
        outcome: usize,
    ) -> Transition<CarRacingState, CarRacingEvent> {
        assert!(
            matches!(state, CarRacingState::Pending),
            "apply_chance_node on a realized CarRacing state"
        );
        let live = self.realize(outcome as u32);
        Transition::silent(CarRacingState::Live(Box::new(live)), 1)
    }

    fn step(
        &self,
        state: &CarRacingState,
        actions: &[usize],
    ) -> Transition<CarRacingState, CarRacingEvent> {
        let CarRacingState::Live(live) = state else {
            unreachable!("step on a pending CarRacing state");
        };
        assert!(!live.done, "step on a terminal CarRacing state");
        let action = actions[0];
        let mut next = live.clone();

        next.car
            .steer(-0.6 * f64::from(action == 1) + 0.6 * f64::from(action == 2));
        next.car.gas(0.2 * f64::from(action == 3));
        next.car.brake(0.8 * f64::from(action == 4));
        next.car.step(1.0 / track::FPS);
        next.tick += 1;

        let (new_tiles, entered_zero) = self.contact_pass(&mut next);
        let total = next.track.tiles.len() as u32;

        let all_visited = next.visited_count == total;
        let lap = entered_zero
            && f64::from(next.visited_count) / f64::from(total) > self.lap_complete_percent;
        let (hx, hy) = next.car.hull_pos();
        let off = hx.abs() > PLAYFIELD || hy.abs() > PLAYFIELD;
        let terminal = all_visited || lap || off;
        next.done = terminal;

        let event = CarRacingEvent {
            new_tiles,
            total_tiles: total,
            off_playfield: off,
        };
        Transition {
            next_state: CarRacingState::Live(next),
            events: vec![Some(event)],
            terminal,
        }
    }

    fn initial_state(&self) -> CarRacingState {
        CarRacingState::Pending
    }

    fn truncation_horizon(&self) -> Option<usize> {
        self.max_ticks
    }
}

/// Diagnostic state-vector encoder: pose, velocities, wheel state, progress. The pixel
/// encoder arrives with the renderer; this one is kept for renderer-free training and
/// physics-vs-visual failure isolation.
pub struct CarRacingVec;

impl ActionView for CarRacingVec {}

impl StateEncoder for CarRacingVec {
    type State = CarRacingState;

    fn encode(&self, state: &CarRacingState, _agent: usize) -> Vec<f32> {
        let CarRacingState::Live(l) = state else {
            unreachable!("encode on a pending CarRacing state (kept out of observations)");
        };
        let (x, y) = l.car.hull_pos();
        let angle = l.car.hull_angle();
        let hull = &l.car.bodies[l.car.hull];
        let v = hull.linvel();
        let mut out = Vec::with_capacity(OBS_DIM);
        out.push((x / PLAYFIELD) as f32);
        out.push((y / PLAYFIELD) as f32);
        out.push(libm::cos(angle) as f32);
        out.push(libm::sin(angle) as f32);
        out.push(v.x / (2.0 * TRACK_RAD as f32));
        out.push(v.y / (2.0 * TRACK_RAD as f32));
        out.push(hull.angvel());
        for i in 0..4 {
            out.push((l.car.ctl[i].omega / 100.0) as f32);
            out.push(f32::from(u8::from(l.car.ctl[i].on_road)));
        }
        for i in 0..2 {
            out.push(l.car.ctl[i].steer as f32);
        }
        out.push(l.visited_count as f32 / l.track.tiles.len().max(1) as f32);
        out.push(l.tick as f32 / 1000.0);
        out.push(f32::from(u8::from(l.track.fallback)));
        out.push(l.car.hull_speed() as f32 / (2.0 * TRACK_RAD) as f32);
        debug_assert_eq!(out.len(), OBS_DIM);
        out
    }

    fn obs_shape(&self) -> (usize, usize, usize) {
        (1, 1, OBS_DIM)
    }

    fn observation_space(&self) -> Space {
        Space::unit_box(vec![1, 1, OBS_DIM])
    }
}

#[cfg(test)]
mod tests {
    use super::codec::CarRacingCodec;
    use super::*;
    use reinfors_core::{
        realize_initial_state, Dqn, Engine, EngineParams, EpsilonGreedyQ, StateCodec,
    };

    fn game() -> CarRacing {
        CarRacing::default()
    }

    fn live(seed: u32) -> CarRacingState {
        CarRacingState::Live(Box::new(game().realize(seed)))
    }

    fn drive(state: &CarRacingState, actions: &[usize]) -> (CarRacingState, Vec<CarRacingEvent>) {
        let g = game();
        let mut s = state.clone();
        let mut events = Vec::new();
        for &a in actions {
            let t = g.step(&s, &[a]);
            events.push(t.events[0].unwrap());
            s = t.next_state;
            if t.terminal {
                break;
            }
        }
        (s, events)
    }

    fn fingerprint(state: &CarRacingState) -> Vec<u8> {
        CarRacingCodec { game: game() }.encode(state)
    }

    #[test]
    fn same_seed_same_track_and_trajectory() {
        let a = live(7);
        let b = live(7);
        assert_eq!(fingerprint(&a), fingerprint(&b));
        let plan: Vec<usize> = (0..120).map(|i| [3, 3, 1, 3, 2, 0][i % 6]).collect();
        let (fa, _) = drive(&a, &plan);
        let (fb, _) = drive(&b, &plan);
        assert_eq!(fingerprint(&fa), fingerprint(&fb));
    }

    #[test]
    fn different_seeds_usually_differ() {
        let differing = (0..8u32)
            .filter(|&s| fingerprint(&live(s)) != fingerprint(&live(s + 100)))
            .count();
        assert!(differing >= 7, "only {differing}/8 seed pairs differed");
    }

    #[test]
    fn track_generation_is_sane_across_seeds() {
        for seed in 0..32u32 {
            let t = track::Track::generate(seed);
            assert!(!t.fallback, "seed {seed} fell back unexpectedly");
            assert!(t.tiles.len() > 100, "seed {seed}: {} tiles", t.tiles.len());
            for p in &t.points {
                assert!(p.x.abs() < PLAYFIELD && p.y.abs() < PLAYFIELD);
            }
        }
    }

    #[test]
    fn exhausted_retries_fall_back_to_the_ring() {
        let t = track::Track::generate_with_attempts(0, 0);
        assert!(t.fallback);
        assert_eq!(t.tiles.len(), 100);
        let g = game();
        let l = g.realize_with_attempts(0, 0);
        assert!(l.track.fallback);
    }

    #[test]
    fn first_step_reward_matches_gym_accounting() {
        let s = live(3);
        let (_, events) = drive(&s, &[0]);
        let e = events[0];
        assert!(!e.off_playfield);
        let r = CarRacingReward::default().step_reward(&e, 0);
        let expected = 1000.0 * f64::from(e.new_tiles) / f64::from(e.total_tiles) - 0.1;
        assert_eq!(r, expected);
    }

    #[test]
    fn off_playfield_reward_is_exactly_minus_100() {
        let g = game();
        let CarRacingState::Live(mut l) = live(3) else {
            unreachable!()
        };
        let (hx, _) = l.car.hull_pos();
        let dx = (PLAYFIELD + 5.0 - hx) as f32;
        let handles: Vec<_> = std::iter::once(l.car.hull)
            .chain(l.car.wheels.iter().copied())
            .collect();
        for h in handles {
            let t = l.car.bodies[h].translation();
            l.car.bodies[h].set_translation(rapier2d::prelude::Vector::new(t.x + dx, t.y), true);
        }
        let t = g.step(&CarRacingState::Live(l), &[0]);
        let e = t.events[0].unwrap();
        assert!(t.terminal && e.off_playfield);
        assert_eq!(CarRacingReward::default().step_reward(&e, 0), -100.0);
    }

    #[test]
    fn lap_needs_reentry_not_overlap() {
        let g = game();
        let CarRacingState::Live(mut l) = live(3) else {
            unreachable!()
        };
        let n = l.track.tiles.len() as u32;
        for w in l.visited.iter_mut() {
            *w = u64::MAX;
        }
        l.visited_count = n;
        for tiles in &mut l.wheel_tiles {
            tiles.push(0);
        }
        let overlap_step = g.step(&CarRacingState::Live(l.clone()), &[0]);
        let still_overlapping = l.wheel_tiles.iter().any(|t| t.contains(&0));
        if still_overlapping {
            assert!(
                overlap_step.terminal,
                "all tiles visited terminates regardless; guard below is the entry case"
            );
        }
        // The entry case: wheel sets say we are NOT on tile zero, so stepping while
        // physically over it must register an entry.
        let mut left = l.clone();
        for tiles in &mut left.wheel_tiles {
            tiles.clear();
        }
        left.visited_count = n - 1;
        left.visited[0] &= !1u64;
        let t = g.step(&CarRacingState::Live(left), &[0]);
        assert!(
            t.terminal,
            "re-entry with lap fraction met must complete the lap"
        );
    }

    #[test]
    fn multi_wheel_contact_counts_each_tile_once() {
        let CarRacingState::Live(l) = live(11) else {
            unreachable!()
        };
        let mut per_wheel_hits = 0usize;
        for tiles in &l.wheel_tiles {
            per_wheel_hits += tiles.len();
        }
        assert!(
            per_wheel_hits > l.visited_count as usize,
            "start pose should have wheels sharing tiles ({per_wheel_hits} contacts, {} visited)",
            l.visited_count
        );
    }

    #[test]
    fn snapshot_roundtrip_resumes_bit_identically() {
        let codec = CarRacingCodec { game: game() };
        let plan_a: Vec<usize> = (0..80).map(|i| [3, 3, 1, 0, 2, 3][i % 6]).collect();
        let plan_b: Vec<usize> = (0..40).map(|i| [3, 2, 0, 1][i % 4]).collect();
        let (mid, _) = drive(&live(5), &plan_a);
        let restored = codec.decode(&codec.encode(&mid)).unwrap();
        codec.validate_decoded_state(&restored, false).unwrap();
        let (end_direct, _) = drive(&mid, &plan_b);
        let (end_restored, _) = drive(&restored, &plan_b);
        assert_eq!(
            fingerprint(&end_direct),
            fingerprint(&end_restored),
            "compact snapshot restore diverged: warm-start state matters; move to scheme (b)/(c)"
        );
    }

    #[test]
    fn adversarial_snapshots_are_rejected() {
        let codec = CarRacingCodec { game: game() };
        let good = codec.encode(&live(5));
        assert!(codec.decode(&[]).is_err());
        assert!(codec.decode(&good[..good.len() / 2]).is_err());
        let mut wrong_version = good.clone();
        wrong_version[0] = 99;
        assert!(codec.decode(&wrong_version).is_err());
        let pending = codec
            .decode(&codec.encode(&CarRacingState::Pending))
            .unwrap();
        assert!(codec.validate_decoded_state(&pending, false).is_err());
    }

    fn engine_with(n_groups: usize) -> Engine<CarRacing, EpsilonGreedyQ, Dqn> {
        Engine::new(
            game(),
            Box::new(CarRacingVec),
            Box::new(CarRacingReward::default()),
            EpsilonGreedyQ::new(1, 0.3),
            Dqn::new(1, 1.0, 1, 0.99),
            EngineParams {
                n_games: 2,
                seed: 9,
                n_groups,
                ..Default::default()
            },
        )
    }

    #[test]
    fn engine_collects_ungrouped() {
        let (records, stats) =
            engine_with(1).collect(16, |_obs: Vec<f32>, n: usize| vec![0.0; n * N_ACTIONS]);
        assert!(records.len() >= 16);
        assert!(stats.decisions > 0);
    }

    #[test]
    fn engine_collects_grouped() {
        let host = reinfors_core::ServiceHost::spawn(|_p: usize, _obs: Vec<f32>, n: usize| {
            vec![0.0; n * N_ACTIONS]
        });
        let (records, stats) =
            engine_with(2).collect_grouped_hosted(16, reinfors_core::InferMode::Shared, &host);
        assert!(records.len() >= 16);
        assert!(stats.decisions > 0);
    }

    #[test]
    fn realized_root_is_live_and_playable() {
        let g = game();
        let mut rng = reinfors_core::SplitMix64::new(42);
        let s = realize_initial_state(&g, &mut rng);
        assert!(matches!(g.actor(&s), reinfors_core::Actor::Agent(0)));
        assert_eq!(g.legal_actions(&s, 0).len(), N_ACTIONS);
    }

    #[test]
    #[ignore = "benchmark quartet; run explicitly with -- --ignored --nocapture"]
    fn benchmark_quartet() {
        let codec = CarRacingCodec { game: game() };
        let s = live(5);
        let CarRacingState::Live(l) = &s else {
            unreachable!()
        };
        let t0 = std::time::Instant::now();
        let mut clones = 0u64;
        while t0.elapsed().as_millis() < 200 {
            let c = l.clone();
            std::hint::black_box(&c);
            clones += 1;
        }
        let clone_us = t0.elapsed().as_micros() as f64 / clones as f64;

        let g = game();
        let mut cur = s.clone();
        let t0 = std::time::Instant::now();
        let mut steps = 0u64;
        while t0.elapsed().as_millis() < 500 {
            let t = g.step(&cur, &[3]);
            cur = if t.terminal { s.clone() } else { t.next_state };
            steps += 1;
        }
        let step_us = t0.elapsed().as_micros() as f64 / steps as f64;

        let bytes = codec.encode(&s);
        let t0 = std::time::Instant::now();
        let mut restores = 0u64;
        while t0.elapsed().as_millis() < 500 {
            let r = codec.decode(&bytes).unwrap();
            std::hint::black_box(&r);
            restores += 1;
        }
        let restore_us = t0.elapsed().as_micros() as f64 / restores as f64;
        println!(
            "clone {clone_us:.1}us | step {step_us:.1}us (clone {:.0}%) | snapshot {} bytes | restore {restore_us:.0}us",
            100.0 * clone_us / step_us,
            bytes.len()
        );
    }
}
