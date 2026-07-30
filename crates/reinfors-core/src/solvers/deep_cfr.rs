//! Deep CFR (Brown et al. 2019, external-sampling variant): the neural scale-up of the
//! tabular [`cfr`](crate::solvers::cfr) solver. Where tabular CFR reads and writes a regret
//! table row, this queries the user's per-player ADVANTAGE networks through the standard
//! `infer` seam and — instead of accumulating — EMITS training samples:
//!
//! - at the traverser's infosets: one advantage sample per visit, `(features, iteration,
//!   legal ids, targets)` with targets `v(a) − Σ_a σ(a)·v(a)` over the enumerated actions;
//! - at opponent infosets: one strategy sample `(features, iteration, legal ids, σ)` — the
//!   average-policy network (the playable product) trains on these.
//!
//! The user owns reservoir buffers, iteration-weighted losses, and (re)training — reinfors
//! ships the data generator (the DQN-family division of labor). Strategies are derived from
//! net outputs by regret matching over clamped advantages, with a pure-argmax fallback when
//! none is positive (Brown's scheme).
//!
//! **Why a solver and not an engine policy**: external sampling's unbiasedness requires
//! traverser-infoset visitation weighted by OPPONENT-AND-CHANCE reach only, with all of the
//! traverser's own actions expanded — the counterfactual measure. Engine episodes advance by
//! both players' sampled actions (self-play visitation includes the traverser's own reach),
//! which is the wrong distribution for these estimates. Root-restarted generative traversal
//! is the sampling scheme that makes frequency equal the right measure — it also conditions
//! opponent ranges on their actions for free (worlds where the opponent would have folded
//! simply don't reach this infoset).
//!
//! **Throughput**: per-node Python callbacks would be ruinous, so K traversals run in
//! LOCKSTEP as explicit-stack machines — a machine advances until it blocks on a net query
//! (opponent nodes block descent; the traverser's σ is only needed at unwind, so children
//! explore first), pending queries batch per round grouped by player, and a per-player
//! [`InferCache`] (nets are frozen within one `collect` call; caches clear at every call)
//! turns the heavily revisited early-tree infosets into cache hits. `infer` runs only on the
//! calling thread. Per-machine rng streams are derived deterministically, so results are
//! reproducible and independent of cache state or advancement interleaving.
//!
//! **Player count**: any 2..=10 sequential players. The 2-player zero-sum Nash story does
//! NOT survive past two players — there the traversals are empirical per-player regret
//! minimization (the Pluribus regime), measured by the exact NashConv instrument on
//! enumerable games and by frozen-policy exploiters at scale.
//!
//! Construction gates match tabular CFR: 2 players, [`Game::information_states`],
//! rng-free `initial_state` (chance fully declared by construction).

use std::collections::HashMap;
use std::sync::atomic::AtomicU64;
use std::sync::Arc;

use crate::encoder::StateEncoder;
use crate::game::{Actor, Game, Rng, Transition};
use crate::reward::Reward;
use crate::rng::SplitMix64;
use crate::rollout::infer_cache::InferCache;
use crate::solvers::best_response;

/// One advantage-network training sample, emitted at a traverser infoset visit.
pub struct AdvantageSample {
    pub obs: Vec<f32>,
    pub iteration: u64,
    /// Legal action ids; `targets` aligns with this list.
    pub legal: Vec<usize>,
    /// Sampled advantages `v(a) − v̄` per legal action.
    pub targets: Vec<f64>,
}

/// One average-policy training sample, emitted at an opponent infoset visit.
pub struct StrategySample {
    /// The acting (opponent) player the sample belongs to.
    pub player: usize,
    pub obs: Vec<f32>,
    pub iteration: u64,
    pub legal: Vec<usize>,
    /// The current strategy σ played at this visit; `probs` aligns with `legal`.
    pub probs: Vec<f64>,
}

/// One `collect` call's telemetry (the engine `CollectStats` idiom).
#[derive(Default)]
pub struct DeepCfrStats {
    pub traversals: usize,
    pub advantage_samples: usize,
    pub strategy_samples: usize,
    pub infer_calls: usize,
    pub infer_rows: usize,
    /// Time inside the `infer` callback only (the net's share of the wall clock).
    pub infer_seconds: f64,
    /// The whole `collect` call (traversal, chance sampling, caching, sample construction).
    pub collect_seconds: f64,
    pub cache_lookups: usize,
    pub cache_hits: usize,
}

/// Regret matching over one net row restricted to the legal ids: play proportionally to
/// clamped-positive advantages; when none is positive, PURE ARGMAX of the advantage (Brown's
/// fallback — not uniform).
fn matched_strategy(row: &[f64], legal: &[usize]) -> Vec<f64> {
    let clamped: Vec<f64> = legal.iter().map(|&a| row[a].max(0.0)).collect();
    let total: f64 = clamped.iter().sum();
    if total > 0.0 {
        return clamped.iter().map(|c| c / total).collect();
    }
    let mut best = 0;
    for (i, &a) in legal.iter().enumerate().skip(1) {
        if row[a] > row[legal[best]] {
            best = i; // strict: ties keep the FIRST maximal action (conventional argmax)
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

/// A traversal's explicit stack frame.
enum Frame<S> {
    /// A pure edge-reward accumulator (chance-chain rewards; the opponent's stepped edge).
    Edge { r: f64 },
    /// The traverser's node: children explored depth-first; σ requested at unwind.
    Traverser {
        state: S,
        obs: Vec<f32>,
        legal: Vec<usize>,
        values: Vec<f64>,
        pending_edge: f64,
        awaiting_sigma: bool,
    },
    /// An opponent node blocked on its σ query.
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

/// One in-flight traversal: an explicit-stack coroutine that advances until it blocks on a
/// net query or completes. Owns its rng stream (derived deterministically), so results are
/// independent of advancement interleaving and cache state.
struct Machine<S> {
    stack: Vec<Frame<S>>,
    step: Option<Step<S>>,
    /// `(player, cache key, obs, already-missed)` of the pending σ query — the flag keeps
    /// the post-miss re-check out of the hit statistics (a miss would otherwise manufacture
    /// exactly one "hit" when its own inserted row resolves it).
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
    collects: u64,           // salts per-machine rng streams across collect calls
    caches: Vec<InferCache>, // one per player (never share obs-keyed rows across nets)
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
        // Gate on the REALIZED root: a declared game's raw root is commonly a chance node,
        // which says nothing about decision dynamics — probing it would let a chance-root
        // simultaneous game through to a mid-collect panic. (Uniform dynamics per game is a
        // framework contract; a mid-game switch is caught by the loud runtime backstop.)
        let realized = crate::game::realize_initial_state(&game, &mut SplitMix64::new(0x0517_B0BE));
        assert!(
            !matches!(game.actor(&realized), Actor::Simultaneous),
            "simultaneous games are not supported by Deep CFR (sequential turn-taking only)"
        );
        let generation = Arc::new(AtomicU64::new(0));
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

    /// Advance to the next CFR iteration — the weight `t` stamped on emitted samples (the
    /// user's loss weights linearly by it, per Brown).
    pub fn next_iteration(&mut self) {
        self.iteration += 1;
    }

    /// Roll back the per-call rng salt after a FAILED collect (e.g. the caller's net raised
    /// mid-call and the returned samples were discarded): a retry then draws the same worlds
    /// a fresh solver would, keeping error paths transactional with respect to determinism.
    pub fn rollback_collect(&mut self) {
        self.collects = self.collects.saturating_sub(1);
    }

    /// Run `traversals` external-sampling traversals with `player` as the traverser, emitting
    /// advantage samples for `player` and strategy samples at opponent infosets.
    ///
    /// `infer(player, obs_flat, rows) -> advantages` serves the CURRENT advantage network of
    /// `player` on a row-major `[rows, obs_dim]` batch, returning `rows * action_count` f64s.
    /// Networks must be frozen for the duration of one call (the per-player caches assume it;
    /// they clear at every call, so retraining BETWEEN calls is the expected rhythm).
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
            cache.force_clear(); // nets may have been retrained since the previous call
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
            // Advance every machine as far as the (round-warm) caches allow.
            let mut misses: HashMap<usize, Vec<usize>> = HashMap::new(); // player -> machines
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
            // One batched call per player with pending queries, deduplicated by cache key.
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
            // The outer loop's cache path resumes the blocked machines against the warm cache.
        }
        stats.advantage_samples = advantage.len();
        stats.strategy_samples = strategy.len();
        stats.collect_seconds = started.elapsed().as_secs_f64();
        (advantage, strategy, stats)
    }

    /// Deliver a σ row to the machine's blocked frame, then keep advancing.
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
                strategy.push(StrategySample {
                    player: who,
                    obs,
                    iteration: self.iteration,
                    legal: legal.clone(),
                    probs: sigma.clone(),
                });
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

    /// Small-step the machine until it blocks on a net query or completes.
    fn advance(&self, m: &mut Machine<G::State>, player: usize) {
        while let Some(step) = m.step.take() {
            match step {
                Step::Descend(mut state) => loop {
                    match self.game.actor(&state) {
                        Actor::Chance => {
                            // Root deals and interior reveals: draw, accumulate the
                            // traverser's edge rewards, stop on a chain that settles the game.
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
                            // All children valued: request the traverser's σ for the baseline.
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

    /// Step the traverser-frame's next unexplored action.
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

    /// Apply one decision: the deterministic step. Chance-node chains on the resulting state
    /// are the descent loop's job.
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

    // ---------------- the exploitability instrument ----------------

    /// Every reachable information set with an exemplar state and its features — the input to
    /// the exact-exploitability instrument for a NET policy (enumerable games only: the walk
    /// is capped like best response). Returns `(key, features, legal ids)` per infoset.
    pub fn infoset_features(&self) -> Vec<(Vec<u8>, Vec<f32>, Vec<usize>)> {
        best_response::enumerate_infosets(&self.game)
            .into_iter()
            .map(|(key, state, agent)| {
                let obs = self.encoder.encode(&state, agent);
                let legal = self.game.legal_actions(&state, agent);
                (key, obs, legal)
            })
            .collect()
    }

    /// Exact exploitability of a policy given per-infoset action probabilities (aligned with
    /// each infoset's legal ids, as returned by [`infoset_features`](Self::infoset_features));
    /// unlisted infosets play uniform. Same instrument and definition as tabular CFR
    /// (NashConv / 2, zero at Nash).
    pub fn exploitability_of(&self, probs: &HashMap<Vec<u8>, Vec<f64>>) -> f64 {
        best_response::exploitability(&self.game, &*self.reward, &|key, legal| {
            probs
                .get(key)
                .cloned()
                .unwrap_or_else(|| vec![1.0 / legal as f64; legal])
        })
    }
}

/// Per-player cache capacity (rows). Poker traversals hammer the early tree, so hit rates are
/// high at modest sizes; rows are `action_count` f64s.
const CACHE_ROWS: usize = 1 << 18;
