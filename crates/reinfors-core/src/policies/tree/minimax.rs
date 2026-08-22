//! Depth-limited minimax/expectiminimax baseline.

use crate::codec::bytes::Reader;
use crate::game::{Game, Rng};
use crate::policies::tree::fold_search_stats;
use crate::policy::{ChanceMode, Policy};
use crate::rollout::engine::CollectStats;

use super::expectimax::search::{Opponent, SearchConfig};
use super::expectimax::{decode_search_eval, encode_search_eval, SearchEvaluation};

/// Depth-limited minimax/expectiminimax with callback-scored frontier leaves; deterministic
/// given its evaluator.
pub struct Minimax {
    cfg: SearchConfig,
}

impl Minimax {
    /// `gamma` is the paired learner's: search backup and episode targets share one discount.
    pub fn new(depth: i32, move_cap: Option<usize>, chance: ChanceMode, gamma: f64) -> Self {
        assert!(depth >= 1, "minimax needs at least one ply of lookahead");
        assert!(
            Self::supports_chance_mode(chance),
            "Minimax expands each node exactly once and cannot express per-traversal chance \
             modes; use Committed or ExpandAll"
        );
        if let Some(cap) = move_cap {
            assert!(cap >= 1, "top_k must keep at least one move per node");
        }
        Minimax {
            cfg: SearchConfig {
                gamma,
                beta: 1.0, // frontier ordering is irrelevant: every node is expanded

                expansion_budget: usize::MAX,
                top_k: usize::MAX,
                max_depth: depth,
                chance,
                opponent: Opponent::Adversarial {
                    move_cap: move_cap.unwrap_or(usize::MAX),
                },
            },
        }
    }

    pub fn supports_chance_mode(mode: ChanceMode) -> bool {
        !mode.requires_repeated_traversal()
    }
}

impl Policy for Minimax {
    type Evaluation = SearchEvaluation;
    type PolicyState = ();

    fn supports_imperfect_information(&self) -> bool {
        false
    }

    fn max_agents(&self, sequential: bool) -> Option<usize> {
        // Some(0) rejects every simultaneous game; the binding additionally requires two players.
        if sequential {
            Some(2)
        } else {
            Some(0)
        }
    }

    fn encode_eval(&self, eval: &SearchEvaluation, out: &mut Vec<u8>) {
        encode_search_eval(eval, out);
    }

    fn decode_eval(&self, r: &mut Reader, action_count: usize) -> Result<SearchEvaluation, String> {
        decode_search_eval(r, action_count, 1, false)
    }

    fn policy_state_to_u64(&self, _s: &()) -> u64 {
        0
    }

    fn policy_state_from_u64(&self, v: u64) -> Result<(), String> {
        if v != 0 {
            return Err(format!("minimax carries no policy state, got {v}"));
        }
        Ok(())
    }

    fn begin_episode(&self, _rng: &mut dyn Rng) {}

    type Search<S: Send> = super::expectimax::search::MultiStepper<S>;

    fn begin_search<G: Game + Sync>(
        &self,
        ctx: crate::policy::SearchCtx<'_, G>,
        state: &G::State,
        perspectives: &[usize],
    ) -> Self::Search<G::State>
    where
        G::State: Send,
    {
        super::expectimax::search::guard_game(ctx.game);
        super::expectimax::search::MultiStepper::new(
            perspectives
                .iter()
                .map(|&agent| super::expectimax::search::Stepper::new(state.clone(), agent))
                .collect(),
        )
    }

    fn round<G: Game + Sync>(
        &self,
        ctx: crate::policy::SearchCtx<'_, G>,
        search: &mut Self::Search<G::State>,
        out: &mut crate::policy::RequestSink,
    ) -> crate::policy::RoundStatus
    where
        G::State: Send,
    {
        super::expectimax::search::multi_round(
            search, ctx.game, ctx.enc, ctx.reward, &self.cfg, out, ctx.rng,
        )
    }

    fn absorb<G: Game + Sync>(
        &self,
        ctx: crate::policy::SearchCtx<'_, G>,
        search: &mut Self::Search<G::State>,
        rows: crate::policy::RowsView<'_>,
    ) where
        G::State: Send,
    {
        super::expectimax::search::multi_absorb(search, ctx.game, ctx.enc, &self.cfg, rows);
    }

    fn finish<G: Game + Sync>(
        &self,
        ctx: crate::policy::SearchCtx<'_, G>,
        search: Self::Search<G::State>,
    ) -> Vec<(SearchEvaluation, Vec<crate::learner::InteriorTarget>)>
    where
        G::State: Send,
    {
        search
            .steppers
            .into_iter()
            .map(|st| {
                let agent = st.agent();
                let legal = ctx.game.legal_actions(st.root_state(), agent);
                let (mut values, interior, stats) = super::expectimax::search::stepper_finish(
                    st,
                    ctx.game,
                    ctx.enc,
                    &self.cfg,
                    ctx.collect_interior,
                );
                values.truncate(1);
                (
                    SearchEvaluation {
                        values,
                        visits: Vec::new(),
                        legal,
                        stats,
                    },
                    interior,
                )
            })
            .collect()
    }

    fn select(&self, eval: &SearchEvaluation, _state: &mut (), _rng: &mut dyn Rng) -> usize {
        let row = &eval.values[0];
        debug_assert!(!eval.legal.is_empty());
        let mut best = eval.legal[0];
        for &a in &eval.legal {
            if row[a] > row[best] {
                best = a;
            }
        }
        best
    }

    fn fold_telemetry(&self, eval: &SearchEvaluation, stats: &mut CollectStats) {
        fold_search_stats(eval, stats);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoder::{ActionView, StateEncoder};
    use crate::game::{Actor, ChanceDist, Game, Transition};
    use crate::policies::tree::expectimax::search::search_many;
    use crate::policies::tree::expectimax::search::SearchConfig as Cfg;
    use crate::reward::Reward as RewardTrait;

    struct Pass;
    impl RewardTrait for Pass {
        type Event = f64;
        fn step_reward(&self, e: &f64, _: usize) -> f64 {
            *e
        }
    }

    // A two-ply duel from a literal table. P0 moves at the root (0 or 1); P1 replies; every
    // reply is terminal with the listed payout to P0 (zero-sum).
    //   root action 0 -> replies pay {+2, -3}: minimax value -3 (greedy immediate bait).
    //   root action 1 -> replies pay {+0.5, +1}: minimax value +0.5.
    #[derive(Clone)]
    struct DuelSt {
        ply: u8,
        first: usize,
    }
    struct Duel;
    impl Game for Duel {
        type State = DuelSt;
        type Event = f64;
        fn num_agents(&self) -> usize {
            2
        }
        fn action_count(&self) -> usize {
            2
        }
        fn actor(&self, s: &DuelSt) -> Actor {
            Actor::Agent(usize::from(s.ply))
        }
        fn legal_actions(&self, s: &DuelSt, agent: usize) -> Vec<usize> {
            if s.ply < 2 && agent == usize::from(s.ply) {
                vec![0, 1]
            } else {
                Vec::new()
            }
        }
        fn step(&self, s: &DuelSt, actions: &[usize]) -> Transition<DuelSt, f64> {
            if s.ply == 0 {
                Transition {
                    next_state: DuelSt {
                        ply: 1,
                        first: actions[0],
                    },
                    events: vec![Some(0.0), Some(0.0)],
                    terminal: false,
                }
            } else {
                let pay = match (s.first, actions[1]) {
                    (0, 0) => 2.0,
                    (0, 1) => -3.0,
                    (1, 0) => 0.5,
                    _ => 1.0,
                };
                Transition {
                    next_state: DuelSt {
                        ply: 2,
                        first: s.first,
                    },
                    events: vec![Some(pay), Some(-pay)],
                    terminal: true,
                }
            }
        }
        fn initial_state(&self) -> DuelSt {
            DuelSt { ply: 0, first: 0 }
        }
    }
    struct DuelEnc;
    impl ActionView for DuelEnc {}
    impl StateEncoder for DuelEnc {
        type State = DuelSt;
        fn encode(&self, s: &DuelSt, _: usize) -> Vec<f32> {
            vec![f32::from(s.ply), s.first as f32]
        }
        fn obs_shape(&self) -> (usize, usize, usize) {
            (1, 1, 2)
        }
    }

    fn adversarial_cfg(depth: i32, move_cap: usize, chance: ChanceMode) -> Cfg {
        Cfg {
            gamma: 1.0,
            beta: 1.0,
            expansion_budget: usize::MAX,
            top_k: usize::MAX,
            max_depth: depth,
            chance,
            opponent: Opponent::Adversarial { move_cap },
        }
    }

    fn zero_infer(_players: &[usize], _obs: Vec<f32>, n: usize) -> Vec<f64> {
        vec![0.0; n * 2]
    }

    #[test]
    fn opponent_replies_back_up_as_a_minimum_not_an_expectation() {
        let results = search_many(
            &Duel,
            &DuelEnc,
            &Pass,
            &adversarial_cfg(2, usize::MAX, ChanceMode::Committed { samples: 1 }),
            vec![(Duel.initial_state(), 0)],
            false,
            0,
            zero_infer,
        );
        let v = &results[0].0[0];
        // Uniform expectation would give -0.5 for action 0; max-max would give +2.
        assert!((v[0] - -3.0).abs() < 1e-12, "min over replies: {v:?}");
        assert!((v[1] - 0.5).abs() < 1e-12, "min over replies: {v:?}");
    }

    #[test]
    fn selection_avoids_the_greedy_bait() {
        let policy = Minimax::new(2, None, ChanceMode::Committed { samples: 1 }, 1.0);
        let eval = SearchEvaluation {
            values: vec![vec![-3.0, 0.5]],
            visits: Vec::new(),
            legal: vec![0, 1],
            stats: Default::default(),
        };
        let mut st = ();
        assert_eq!(
            policy.select(&eval, &mut st, &mut crate::rng::SplitMix64::new(0)),
            1
        );
    }

    // Chance below the searcher's move: risky pays {0.5: 0, 0.5: 3}, safe pays 1 — exact
    // expectiminimax under ExpandAll prefers risky at 1.5.
    #[derive(Clone)]
    struct RiskSt {
        ply: u8,
        risky: bool,
    }
    struct Risk;
    impl Game for Risk {
        type State = RiskSt;
        type Event = f64;
        fn num_agents(&self) -> usize {
            2
        }
        fn action_count(&self) -> usize {
            2
        }
        fn actor(&self, s: &RiskSt) -> Actor {
            if s.ply == 1 && s.risky {
                Actor::Chance
            } else {
                Actor::Agent(0)
            }
        }
        fn legal_actions(&self, s: &RiskSt, agent: usize) -> Vec<usize> {
            if s.ply == 0 && agent == 0 {
                vec![0, 1]
            } else {
                Vec::new()
            }
        }
        fn step(&self, s: &RiskSt, actions: &[usize]) -> Transition<RiskSt, f64> {
            assert_eq!(s.ply, 0);
            if actions[0] == 1 {
                Transition {
                    next_state: RiskSt {
                        ply: 1,
                        risky: true,
                    },
                    events: vec![Some(0.0), Some(0.0)],
                    terminal: false,
                }
            } else {
                Transition {
                    next_state: RiskSt {
                        ply: 1,
                        risky: false,
                    },
                    events: vec![Some(1.0), Some(-1.0)],
                    terminal: true,
                }
            }
        }
        fn chance_node(&self, _s: &RiskSt) -> ChanceDist {
            ChanceDist::Weighted(vec![0.5, 0.5])
        }
        fn apply_chance_node(&self, s: &RiskSt, outcome: usize) -> Transition<RiskSt, f64> {
            let pay = if outcome == 0 { 0.0 } else { 3.0 };
            Transition {
                next_state: RiskSt {
                    ply: 2,
                    risky: s.risky,
                },
                events: vec![Some(pay), Some(-pay)],
                terminal: true,
            }
        }
        fn initial_state(&self) -> RiskSt {
            RiskSt {
                ply: 0,
                risky: false,
            }
        }
    }
    struct RiskEnc;
    impl ActionView for RiskEnc {}
    impl StateEncoder for RiskEnc {
        type State = RiskSt;
        fn encode(&self, s: &RiskSt, _: usize) -> Vec<f32> {
            vec![f32::from(s.ply), f32::from(u8::from(s.risky))]
        }
        fn obs_shape(&self) -> (usize, usize, usize) {
            (1, 1, 2)
        }
    }

    #[test]
    fn expand_all_backs_up_the_exact_expectiminimax_value() {
        let results = search_many(
            &Risk,
            &RiskEnc,
            &Pass,
            &adversarial_cfg(2, usize::MAX, ChanceMode::ExpandAll),
            vec![(Risk.initial_state(), 0)],
            false,
            0,
            zero_infer,
        );
        let v = &results[0].0[0];
        assert_eq!(v[1], 1.5, "0.5*0 + 0.5*3 exactly: {v:?}");
        assert_eq!(v[0], 1.0);
    }

    // Alternating corridor: P0 wins (+1) with its second move. A zero evaluator finds the win
    // at depth 3 and is blind to it at depth 2 — the horizon-perfection property.
    #[derive(Clone)]
    struct CorrSt(u8);
    struct Corridor;
    impl Game for Corridor {
        type State = CorrSt;
        type Event = f64;
        fn num_agents(&self) -> usize {
            2
        }
        fn action_count(&self) -> usize {
            2
        }
        fn actor(&self, s: &CorrSt) -> Actor {
            Actor::Agent(usize::from(s.0 % 2))
        }
        fn legal_actions(&self, s: &CorrSt, agent: usize) -> Vec<usize> {
            if s.0 < 3 && agent == usize::from(s.0 % 2) {
                vec![0, 1]
            } else {
                Vec::new()
            }
        }
        fn step(&self, s: &CorrSt, _actions: &[usize]) -> Transition<CorrSt, f64> {
            let next = s.0 + 1;
            if next == 3 {
                Transition {
                    next_state: CorrSt(next),
                    events: vec![Some(1.0), Some(-1.0)],
                    terminal: true,
                }
            } else {
                Transition {
                    next_state: CorrSt(next),
                    events: vec![Some(0.0), Some(0.0)],
                    terminal: false,
                }
            }
        }
        fn initial_state(&self) -> CorrSt {
            CorrSt(0)
        }
    }
    struct CorrEnc;
    impl ActionView for CorrEnc {}
    impl StateEncoder for CorrEnc {
        type State = CorrSt;
        fn encode(&self, s: &CorrSt, _: usize) -> Vec<f32> {
            vec![f32::from(s.0)]
        }
        fn obs_shape(&self) -> (usize, usize, usize) {
            (1, 1, 1)
        }
    }

    #[test]
    fn a_zero_evaluator_is_horizon_perfect_and_horizon_blind() {
        let run = |depth: i32| {
            search_many(
                &Corridor,
                &CorrEnc,
                &Pass,
                &adversarial_cfg(depth, usize::MAX, ChanceMode::Committed { samples: 1 }),
                vec![(Corridor.initial_state(), 0)],
                false,
                0,
                zero_infer,
            )
            .remove(0)
            .0
        };
        let seen = run(3);
        assert!((seen[0][0] - 1.0).abs() < 1e-12, "win inside the horizon");
        let blind = run(2);
        assert_eq!(blind[0][0], 0.0, "win beyond the horizon is invisible");
    }

    // Beam direction: at depth 3 the opponent node is evaluated as a leaf (retaining its Q row)
    // before expansion. The leaf evaluator ranks reply 0 as worse-for-us, but the true terminals
    // make reply 1 worse. Full width takes the true minimum; a one-move beam keeps only the
    // leaf-ranked reply.
    #[derive(Clone)]
    struct BeamSt {
        ply: u8,
        reply: usize,
    }
    struct Beam;
    impl Game for Beam {
        type State = BeamSt;
        type Event = f64;
        fn num_agents(&self) -> usize {
            2
        }
        fn action_count(&self) -> usize {
            2
        }
        fn actor(&self, s: &BeamSt) -> Actor {
            Actor::Agent(usize::from(s.ply == 1))
        }
        fn legal_actions(&self, s: &BeamSt, agent: usize) -> Vec<usize> {
            match (s.ply, agent) {
                (0, 0) => vec![0],
                (1, 1) => vec![0, 1],
                (2, 0) => vec![0],
                _ => Vec::new(),
            }
        }
        fn step(&self, s: &BeamSt, actions: &[usize]) -> Transition<BeamSt, f64> {
            match s.ply {
                0 => Transition {
                    next_state: BeamSt { ply: 1, reply: 0 },
                    events: vec![Some(0.0), Some(0.0)],
                    terminal: false,
                },
                1 => Transition {
                    next_state: BeamSt {
                        ply: 2,
                        reply: actions[1],
                    },
                    events: vec![Some(0.0), Some(0.0)],
                    terminal: false,
                },
                _ => {
                    let pay = if s.reply == 0 { -1.0 } else { -4.0 };
                    Transition {
                        next_state: BeamSt {
                            ply: 3,
                            reply: s.reply,
                        },
                        events: vec![Some(pay), Some(-pay)],
                        terminal: true,
                    }
                }
            }
        }
        fn initial_state(&self) -> BeamSt {
            BeamSt { ply: 0, reply: 0 }
        }
    }
    struct BeamEnc;
    impl ActionView for BeamEnc {}
    impl StateEncoder for BeamEnc {
        type State = BeamSt;
        fn encode(&self, s: &BeamSt, _: usize) -> Vec<f32> {
            vec![f32::from(s.ply), s.reply as f32]
        }
        fn obs_shape(&self) -> (usize, usize, usize) {
            (1, 1, 2)
        }
    }

    #[test]
    fn a_one_move_beam_keeps_the_leaf_ranked_reply() {
        // The opponent-node leaf row (the opponent's own values) ranks reply 0 as their best,
        // so a one-move beam keeps it (paying us -1); the deeper truth is reply 1 at -4, which
        // full width still finds.
        let infer = |_players: &[usize], obs: Vec<f32>, n: usize| -> Vec<f64> {
            (0..n)
                .flat_map(|i| {
                    if obs[i * 2] as u8 == 1 {
                        vec![9.0, -9.0]
                    } else {
                        vec![0.0, 0.0]
                    }
                })
                .collect()
        };
        let run = |cap: usize| {
            search_many(
                &Beam,
                &BeamEnc,
                &Pass,
                &adversarial_cfg(3, cap, ChanceMode::Committed { samples: 1 }),
                vec![(Beam.initial_state(), 0)],
                false,
                0,
                infer,
            )
            .remove(0)
            .0
        };
        let full = run(usize::MAX);
        assert!((full[0][0] - -4.0).abs() < 1e-12, "true min: {full:?}");
        let beamed = run(1);
        assert!(
            (beamed[0][0] - -1.0).abs() < 1e-12,
            "beam keeps the leaf-ranked reply: {beamed:?}"
        );
    }

    #[test]
    fn opponent_horizons_are_evaluated_on_turn_and_negated() {
        // Depth 1 ends on the opponent's turn. Each leaf row is requested FOR the opponent
        // (player 1 — the same on-turn distribution TreeStrap trains) and holds the opponent's
        // own values: after root action 0 their best is +10, after action 1 it is +1, so the
        // searcher's zero-sum values are -10 and -1.
        use std::cell::RefCell;
        let routed: RefCell<Vec<usize>> = RefCell::new(Vec::new());
        let infer = |players: &[usize], obs: Vec<f32>, n: usize| -> Vec<f64> {
            routed.borrow_mut().extend_from_slice(players);
            (0..n)
                .flat_map(|i| {
                    if obs[i * 2 + 1] == 0.0 {
                        vec![-10.0, 10.0]
                    } else {
                        vec![-2.0, 1.0]
                    }
                })
                .collect()
        };
        let results = search_many(
            &Duel,
            &DuelEnc,
            &Pass,
            &adversarial_cfg(1, usize::MAX, ChanceMode::Committed { samples: 1 }),
            vec![(Duel.initial_state(), 0)],
            false,
            0,
            infer,
        );
        let v = &results[0].0[0];
        assert!((v[0] - -10.0).abs() < 1e-12, "negated mover max: {v:?}");
        assert!((v[1] - -1.0).abs() < 1e-12, "negated mover max: {v:?}");
        assert_eq!(
            *routed.borrow(),
            vec![1, 1],
            "opponent-horizon rows must be requested for the opponent"
        );
    }

    #[test]
    #[should_panic(expected = "search tree exceeds")]
    fn one_wide_final_ply_cannot_allocate_past_the_bound() {
        // 1100 root moves x 1100 terminal replies crosses the node bound mid-round; the
        // insertion-point check must fire even though no further round would run.
        #[derive(Clone)]
        struct W(u8);
        struct WideFinal;
        impl Game for WideFinal {
            type State = W;
            type Event = f64;
            fn num_agents(&self) -> usize {
                2
            }
            fn action_count(&self) -> usize {
                1100
            }
            fn actor(&self, s: &W) -> Actor {
                Actor::Agent(usize::from(s.0))
            }
            fn legal_actions(&self, s: &W, agent: usize) -> Vec<usize> {
                if s.0 < 2 && agent == usize::from(s.0) {
                    (0..1100).collect()
                } else {
                    Vec::new()
                }
            }
            fn step(&self, s: &W, _a: &[usize]) -> Transition<W, f64> {
                Transition {
                    next_state: W(s.0 + 1),
                    events: vec![Some(0.0), Some(0.0)],
                    terminal: s.0 == 1,
                }
            }
            fn initial_state(&self) -> W {
                W(0)
            }
        }
        struct WEnc;
        impl ActionView for WEnc {}
        impl StateEncoder for WEnc {
            type State = W;
            fn encode(&self, s: &W, _: usize) -> Vec<f32> {
                vec![f32::from(s.0)]
            }
            fn obs_shape(&self) -> (usize, usize, usize) {
                (1, 1, 1)
            }
        }
        let _ = search_many(
            &WideFinal,
            &WEnc,
            &Pass,
            &adversarial_cfg(2, usize::MAX, ChanceMode::Committed { samples: 1 }),
            vec![(W(0), 0)],
            false,
            0,
            |_p: &[usize], _o: Vec<f32>, n: usize| vec![0.0; n * 1100],
        );
    }

    #[test]
    #[should_panic(expected = "per-traversal chance modes")]
    fn minimax_rejects_always_resample() {
        let _ = Minimax::new(4, None, ChanceMode::AlwaysResample, 1.0);
    }

    #[test]
    #[should_panic(expected = "at least one ply")]
    fn minimax_rejects_zero_depth() {
        let _ = Minimax::new(0, None, ChanceMode::Committed { samples: 1 }, 1.0);
    }

    #[test]
    #[should_panic(expected = "undefined for simultaneous")]
    fn adversarial_search_rejects_simultaneous_decisions() {
        #[derive(Clone)]
        struct S(bool);
        struct Sim;
        impl Game for Sim {
            type State = S;
            type Event = f64;
            fn num_agents(&self) -> usize {
                2
            }
            fn action_count(&self) -> usize {
                2
            }
            fn actor(&self, _s: &S) -> Actor {
                Actor::Simultaneous
            }
            fn legal_actions(&self, s: &S, _agent: usize) -> Vec<usize> {
                if s.0 {
                    Vec::new()
                } else {
                    vec![0, 1]
                }
            }
            fn step(&self, _s: &S, _a: &[usize]) -> Transition<S, f64> {
                Transition {
                    next_state: S(true),
                    events: vec![Some(0.0), Some(0.0)],
                    terminal: true,
                }
            }
            fn initial_state(&self) -> S {
                S(false)
            }
        }
        struct E;
        impl ActionView for E {}
        impl StateEncoder for E {
            type State = S;
            fn encode(&self, s: &S, _: usize) -> Vec<f32> {
                vec![f32::from(u8::from(s.0))]
            }
            fn obs_shape(&self) -> (usize, usize, usize) {
                (1, 1, 1)
            }
        }
        let _ = search_many(
            &Sim,
            &E,
            &Pass,
            &adversarial_cfg(2, usize::MAX, ChanceMode::Committed { samples: 1 }),
            vec![(S(false), 0)],
            false,
            0,
            zero_infer,
        );
    }
}
