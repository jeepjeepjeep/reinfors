//! Batched external-sampling Deep CFR traversal and sample generation. This is a solver rather
//! than an engine policy because unbiased advantage samples require opponent-and-chance reach:
//! traverser actions are expanded, not sampled through ordinary self-play visitation.

use std::collections::HashMap;
use std::sync::atomic::AtomicU64;
use std::sync::Arc;

use crate::encoder::StateEncoder;
use crate::game::{Actor, Game, Rng, Transition};
use crate::reward::Reward;
use crate::rng::SplitMix64;
use crate::rollout::infer_cache::InferCache;
use crate::solvers::best_response;

/// `(information key, encoded features, legal actions)` for each reachable information set.
pub type InfosetFeatures = Vec<(Vec<u8>, Vec<f32>, Vec<usize>)>;

/// One advantage-network training sample.
pub struct AdvantageSample {
    pub obs: Vec<f32>,
    pub iteration: u64,
    pub legal: Vec<usize>,
    pub targets: Vec<f64>,
}

/// One average-policy training sample.
pub struct StrategySample {
    pub player: usize,
    pub obs: Vec<f32>,
    pub iteration: u64,
    pub legal: Vec<usize>,
    pub probs: Vec<f64>,
}

/// Telemetry for one collection call.
#[derive(Default)]
pub struct DeepCfrStats {
    pub traversals: usize,
    pub advantage_samples: usize,
    pub strategy_samples: usize,
    pub infer_calls: usize,
    pub infer_rows: usize,
    pub infer_seconds: f64,
    pub collect_seconds: f64,
    pub cache_lookups: usize,
    pub cache_hits: usize,
}

fn matched_strategy(row: &[f64], legal: &[usize]) -> Vec<f64> {
    let clamped: Vec<f64> = legal.iter().map(|&a| row[a].max(0.0)).collect();
    let total: f64 = clamped.iter().sum();
    if total > 0.0 {
        return clamped.iter().map(|c| c / total).collect();
    }
    // Brown et al.'s fallback is pure argmax; tabular CFR deliberately falls back to uniform.
    let mut best = 0;
    for (i, &a) in legal.iter().enumerate().skip(1) {
        if row[a] > row[legal[best]] {
            best = i;
        }
    }
    let mut sigma = vec![0.0; legal.len()];
    sigma[best] = 1.0;
    sigma
}

fn sample_index(probs: &[f64], rng: &mut SplitMix64) -> usize {
    let u = rng.unit();
    let mut acc = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        acc += p;
        if u < acc {
            return i;
        }
    }
    probs.len() - 1
}

enum Frame<S> {
    Edge {
        r: f64,
    },
    Traverser {
        state: S,
        obs: Vec<f32>,
        legal: Vec<usize>,
        values: Vec<f64>,
        pending_edge: f64,
        awaiting_sigma: bool,
    },
    OpponentAwait {
        who: usize,
        state: S,
        obs: Vec<f32>,
        legal: Vec<usize>,
    },
}

enum Step<S> {
    Descend(S),
    Return(f64),
}

/// An explicit-stack traversal that yields when inference is required.
struct Machine<S> {
    stack: Vec<Frame<S>>,
    step: Option<Step<S>>,
    // The flag prevents the post-miss recheck from counting as a cache hit.
    blocked: Option<(usize, u128, Vec<f32>, bool)>,
    rng: SplitMix64,
    done: bool,
}

pub struct DeepCfrSolver<G: Game> {
    game: G,
    encoder: Box<dyn StateEncoder<State = G::State>>,
    reward: Box<dyn Reward<Event = G::Event>>,
    iteration: u64,
    seed: u64,
    collects: u64,
    caches: Vec<InferCache>,
}

impl<G: Game> DeepCfrSolver<G> {
    pub fn new(
        game: G,
        encoder: Box<dyn StateEncoder<State = G::State>>,
        reward: Box<dyn Reward<Event = G::Event>>,
        seed: u64,
    ) -> Self {
        assert!(
            (2..=super::cfr::MAX_CFR_PLAYERS).contains(&game.num_agents()),
            "Deep CFR supports 2..={} players; this game has {} agents. NOTE: for more than 2 \
             players the average policy carries NO Nash guarantee — regret minimization is \
             empirical there (see the module docs)",
            super::cfr::MAX_CFR_PLAYERS,
            game.num_agents()
        );
        assert!(
            game.information_states(),
            "Deep CFR requires information-state keys (Game::information_states)"
        );
        // A chance root does not reveal the game's decision dynamics.
        let realized = crate::game::realize_initial_state(&game, &mut SplitMix64::new(0x0517_B0BE));
        assert!(
            !matches!(game.actor(&realized), Actor::Simultaneous),
            "simultaneous games are not supported by Deep CFR (sequential turn-taking only)"
        );
        let generation = Arc::new(AtomicU64::new(0));
        // Observation keys are not network identities: each player's frozen network needs its
        // own cache or identical observations can cross-contaminate policies.
        let caches = (0..game.num_agents())
            .map(|_| InferCache::new(CACHE_ROWS, Arc::clone(&generation)))
            .collect();
        DeepCfrSolver {
            game,
            encoder,
            reward,
            iteration: 0,
            seed,
            collects: 0,
            caches,
        }
    }

    pub fn iteration(&self) -> u64 {
        self.iteration
    }

    /// Advance the iteration stamped on emitted samples.
    pub fn next_iteration(&mut self) {
        self.iteration += 1;
    }

    /// Roll back collection RNG state after a failed callback.
    pub fn rollback_collect(&mut self) {
        self.collects = self.collects.saturating_sub(1);
    }

    /// Collect external-sampling advantage and average-strategy samples.
    pub fn collect<F>(
        &mut self,
        player: usize,
        traversals: usize,
        mut infer: F,
    ) -> (Vec<AdvantageSample>, Vec<StrategySample>, DeepCfrStats)
    where
        F: FnMut(usize, Vec<f32>, usize) -> Vec<f64>,
    {
        assert!(
            player < self.game.num_agents(),
            "player must be below {}",
            self.game.num_agents()
        );
        assert!(
            self.iteration >= 1,
            "call next_iteration() before collecting (samples are weighted by the iteration)"
        );
        let started = std::time::Instant::now();
        self.collects += 1;
        for cache in &mut self.caches {
            cache.force_clear();
        }
        let mut advantage = Vec::new();
        let mut strategy = Vec::new();
        let mut stats = DeepCfrStats {
            traversals,
            ..DeepCfrStats::default()
        };
        let action_count = self.game.action_count();

        let mut machines: Vec<Machine<G::State>> = (0..traversals)
            .map(|k| {
                let salt = (self.collects << 32) ^ ((player as u64) << 24) ^ k as u64;
                Machine {
                    stack: Vec::new(),
                    step: Some(Step::Descend(self.game.initial_state())),
                    blocked: None,
                    rng: SplitMix64::new(
                        self.seed
                            ^ 0xDCF2_5EED_0000_0000
                            ^ salt.wrapping_mul(0x9E37_79B9_7F4A_7C15),
                    ),
                    done: false,
                }
            })
            .collect();

        loop {
            let mut misses: HashMap<usize, Vec<usize>> = HashMap::new();
            let mut all_done = true;
            for (mi, m) in machines.iter_mut().enumerate() {
                loop {
                    if m.done {
                        break;
                    }
                    all_done = false;
                    if let Some((who, key, _, missed)) = m.blocked {
                        if !missed {
                            stats.cache_lookups += 1;
                        }
                        if let Some(row) = self.caches[who].lookup(key) {
                            if !missed {
                                stats.cache_hits += 1;
                            }
                            self.resume(m, &row, player, &mut advantage, &mut strategy);
                        } else {
                            if let Some(b) = m.blocked.as_mut() {
                                b.3 = true;
                            }
                            misses.entry(who).or_default().push(mi);
                            break;
                        }
                    } else {
                        self.advance(m, player);
                    }
                }
            }
            if misses.is_empty() {
                if all_done {
                    break;
                }
                continue;
            }
            let mut players: Vec<usize> = misses.keys().copied().collect();
            players.sort_unstable();
            for who in players {
                let mut order: Vec<u128> = Vec::new();
                let mut obs_flat: Vec<f32> = Vec::new();
                let mut seen: HashMap<u128, ()> = HashMap::new();
                for &mi in &misses[&who] {
                    let (_, key, obs, _) = machines[mi].blocked.as_ref().expect("blocked");
                    if seen.insert(*key, ()).is_none() {
                        order.push(*key);
                        obs_flat.extend_from_slice(obs);
                    }
                }
                let rows = order.len();
                let infer_started = std::time::Instant::now();
                let out = infer(who, obs_flat, rows);
                stats.infer_seconds += infer_started.elapsed().as_secs_f64();
                assert_eq!(
                    out.len(),
                    rows * action_count,
                    "infer returned {} values for {rows} rows; expected {action_count} per row",
                    out.len()
                );
                stats.infer_calls += 1;
                stats.infer_rows += rows;
                for (i, key) in order.iter().enumerate() {
                    self.caches[who].insert(*key, &out[i * action_count..(i + 1) * action_count]);
                }
            }
        }
        stats.advantage_samples = advantage.len();
        stats.strategy_samples = strategy.len();
        stats.collect_seconds = started.elapsed().as_secs_f64();
        (advantage, strategy, stats)
    }

    fn resume(
        &self,
        m: &mut Machine<G::State>,
        row: &[f64],
        player: usize,
        advantage: &mut Vec<AdvantageSample>,
        strategy: &mut Vec<StrategySample>,
    ) {
        m.blocked = None;
        match m
            .stack
            .pop()
            .expect("a blocked machine has a waiting frame")
        {
            Frame::OpponentAwait {
                who,
                state,
                obs,
                legal,
            } => {
                let sigma = matched_strategy(row, &legal);
                // The simple average estimator records one successor player.
                if who == (player + 1) % self.game.num_agents() {
                    strategy.push(StrategySample {
                        player: who,
                        obs,
                        iteration: self.iteration,
                        legal: legal.clone(),
                        probs: sigma.clone(),
                    });
                }
                let action = legal[sample_index(&sigma, &mut m.rng)];
                let mut joint = vec![0; self.game.num_agents()];
                joint[who] = action;
                let (edge, next) = self.realize_transition(&state, &joint, player);
                m.stack.push(Frame::Edge { r: edge });
                m.step = Some(match next {
                    Some(s) => Step::Descend(s),
                    None => Step::Return(0.0),
                });
            }
            Frame::Traverser {
                obs,
                legal,
                values,
                awaiting_sigma,
                ..
            } => {
                debug_assert!(awaiting_sigma, "σ arrives only after all children returned");
                let sigma = matched_strategy(row, &legal);
                let vbar: f64 = sigma.iter().zip(&values).map(|(s, v)| s * v).sum();
                advantage.push(AdvantageSample {
                    obs,
                    iteration: self.iteration,
                    legal,
                    targets: values.iter().map(|v| v - vbar).collect(),
                });
                m.step = Some(Step::Return(vbar));
            }
            Frame::Edge { .. } => unreachable!("edge frames never block"),
        }
        self.advance(m, player);
    }

    fn advance(&self, m: &mut Machine<G::State>, player: usize) {
        while let Some(step) = m.step.take() {
            match step {
                Step::Descend(mut state) => loop {
                    match self.game.actor(&state) {
                        Actor::Chance => {
                            let outcome = self.game.chance_node(&state).draw(&mut m.rng);
                            let t = self.game.apply_chance_node(&state, outcome);
                            let r = crate::reward::edge_reward(&*self.reward, &t.events, player);
                            if t.terminal {
                                m.step = Some(Step::Return(r));
                                break;
                            }
                            if r != 0.0 {
                                m.stack.push(Frame::Edge { r });
                            }
                            state = t.next_state;
                        }
                        Actor::Agent(who) => {
                            let legal = self.game.legal_actions(&state, who);
                            debug_assert!(!legal.is_empty(), "decision states offer actions");
                            let obs = self.encoder.encode(&state, who);
                            if who == player {
                                m.stack.push(Frame::Traverser {
                                    state,
                                    obs,
                                    legal,
                                    values: Vec::new(),
                                    pending_edge: 0.0,
                                    awaiting_sigma: false,
                                });
                                self.start_child(m, player);
                            } else {
                                let key = InferCache::key(&obs);
                                m.blocked = Some((who, key, obs.clone(), false));
                                m.stack.push(Frame::OpponentAwait {
                                    who,
                                    state,
                                    obs,
                                    legal,
                                });
                            }
                            break;
                        }
                        Actor::Simultaneous => {
                            panic!("a simultaneous decision was reached mid-game: solvers support uniformly SEQUENTIAL games (the framework assumes one dynamics per game; mixing violates that contract)")
                        }
                    }
                },
                Step::Return(v) => match m.stack.pop() {
                    None => {
                        m.done = true;
                    }
                    Some(Frame::Edge { r }) => {
                        m.step = Some(Step::Return(v + r));
                    }
                    Some(Frame::Traverser {
                        state,
                        obs,
                        legal,
                        mut values,
                        pending_edge,
                        awaiting_sigma,
                    }) => {
                        debug_assert!(!awaiting_sigma);
                        values.push(pending_edge + v);
                        let complete = values.len() == legal.len();
                        m.stack.push(Frame::Traverser {
                            state,
                            obs,
                            legal,
                            values,
                            pending_edge: 0.0,
                            awaiting_sigma: complete,
                        });
                        if complete {
                            if let Some(Frame::Traverser { obs, .. }) = m.stack.last() {
                                let key = InferCache::key(obs);
                                m.blocked = Some((player, key, obs.clone(), false));
                            }
                        } else {
                            self.start_child(m, player);
                        }
                    }
                    Some(Frame::OpponentAwait { .. }) => {
                        unreachable!("an awaiting opponent frame is resolved by resume()")
                    }
                },
            }
            if m.blocked.is_some() {
                return;
            }
        }
    }

    fn start_child(&self, m: &mut Machine<G::State>, player: usize) {
        let Some(Frame::Traverser {
            state,
            legal,
            values,
            pending_edge,
            ..
        }) = m.stack.last_mut()
        else {
            unreachable!("start_child runs on a traverser frame")
        };
        let action = legal[values.len()];
        let mut joint = vec![0; self.game.num_agents()];
        joint[player] = action;
        let state = state.clone();
        let (edge, next) = self.realize_transition(&state, &joint, player);
        *pending_edge = edge;
        m.step = Some(match next {
            Some(s) => Step::Descend(s),
            None => Step::Return(0.0),
        });
    }

    fn realize_transition(
        &self,
        state: &G::State,
        joint: &[usize],
        player: usize,
    ) -> (f64, Option<G::State>) {
        let t = self.game.step(state, joint);
        let r = crate::reward::edge_reward(&*self.reward, &t.events, player);
        if t.terminal {
            return (r, None);
        }
        let Transition { next_state, .. } = t;
        (r, Some(next_state))
    }

    /// Features and legal actions for every reachable information set.
    pub fn infoset_features(
        &self,
    ) -> Result<InfosetFeatures, best_response::EnumerationCapExceeded> {
        Ok(best_response::enumerate_infosets(&self.game)?
            .into_iter()
            .map(|(key, state, agent)| {
                let obs = self.encoder.encode(&state, agent);
                let legal = self.game.legal_actions(&state, agent);
                (key, obs, legal)
            })
            .collect())
    }

    /// Exact exploitability of a profile; unlisted information sets play uniformly.
    pub fn exploitability_of(
        &self,
        probs: &HashMap<Vec<u8>, Vec<f64>>,
    ) -> Result<f64, best_response::EnumerationCapExceeded> {
        best_response::exploitability(&self.game, &*self.reward, &|key, legal| {
            probs
                .get(key)
                .cloned()
                .unwrap_or_else(|| vec![1.0 / legal as f64; legal])
        })
    }
}

const CACHE_ROWS: usize = 1 << 18;
