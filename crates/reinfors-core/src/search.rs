//! Best-first selective expectimax, ported to match `snake_RL`'s `SelectiveExpectimaxPlanner`.
//!
//! Values back up per ensemble head (K coherent worlds; K=1 for a single-head net), so the
//! expansion priority is value-of-information: expand where the heads disagree (sigma) and the leaf
//! most affects the root (path weight). Leaf values come from an injected evaluator (`infer`) — the
//! Python network forward in production. The opponent is either `Uniform` (eager 1/3 weights) or
//! `Distributional` (deferred: opponent observations are batched with the leaves through `infer`,
//! then their head-mean Q is turned into chance weights via softmax-with-floor).
//!
//! The tree is an arena (`Vec<Node>` indexed by `usize`) so the frontier can hold node references
//! without fighting the borrow checker.

use std::collections::HashSet;

use crate::action::{relative_to_absolute, Action, RELATIVE_ACTIONS};
use crate::obs::egocentric_parts;
use crate::reward::Reward;
use crate::snake::{Cell, Snake, SnakeEnv};

/// The agent's belief about the opponent's move distribution.
pub enum Opponent {
    /// Equal weight on each opponent action (no net dependency).
    Uniform,
    /// Softmax over the opponent's (head-mean) Q with a uniform floor: deferred, so opponent
    /// observations ride the same batched forward as the leaves. `p = (1-floor)*softmax(q/temp) + floor/n`.
    Distributional { temperature: f64, floor: f64 },
}

pub struct SearchParams {
    pub grid_size: i32,
    pub initial_length: usize,
    pub play_to_last: bool,
    pub win_food_lead: Option<usize>,
    pub gamma: f64,
    pub beta: f64,
    pub expansion_budget: usize,
    pub top_k: usize,
    pub max_depth: i32,
    pub reward: Reward,
    pub opponent: Opponent,
}

#[derive(Default, Clone, Copy, Debug)]
pub struct SearchStats {
    pub max_depth: i32,
    pub expansions: usize,
    pub leaves: usize,
    pub rounds: usize,
}

/// A committed agent action's chance branch. `weight` is the resolved chance probability; for a
/// deferred (distributional) opponent it is filled in during evaluation from `deferred = (opp obs
/// index this round, opponent action index)`.
struct Branch {
    weight: f64,
    deferred: Option<(usize, usize)>,
    reward: f64,
    child: usize,
}

struct Edge {
    branches: Vec<Branch>,
}

struct Node {
    snakes: [Snake; 2],
    food: HashSet<Cell>,
    obs: Vec<f32>, // leaf observation (empty for terminal nodes)
    depth: i32,
    terminal: bool,
    edges: Option<Vec<Edge>>, // None => unexpanded frontier leaf
    bootstrap: Vec<f64>,      // per-head net leaf value (empty until evaluated)
    value: Vec<f64>,          // per-head backed-up value (empty at terminals; treated as 0)
    sigma: f64,               // std over heads of the bootstrap — the VOI signal
    path_weight: f64,
}

/// What the opponent branching produced for one node, before evaluation resolves deferred weights.
enum BranchWeight {
    Fixed(f64),
    Deferred(usize, usize),
}

/// Run a best-first selective-expectimax search from `(snakes, food)` for `agent` (0 or 1). Returns
/// the root action values as `[K][A]` (per head, per relative action Forward/Left/Right) plus
/// diagnostics. `infer` maps a batch of flat observations to per-observation `[K][A]` action values.
pub fn selective_search<F>(
    p: &SearchParams,
    snakes: [Snake; 2],
    food: HashSet<Cell>,
    agent: usize,
    mut infer: F,
) -> (Vec<Vec<f64>>, SearchStats)
where
    F: FnMut(&[Vec<f32>]) -> Vec<Vec<Vec<f64>>>,
{
    let opp = 1 - agent;
    let mut arena: Vec<Node> = vec![Node {
        snakes,
        food,
        obs: Vec::new(),
        depth: 0,
        terminal: false,
        edges: None,
        bootstrap: Vec::new(),
        value: Vec::new(),
        sigma: 0.0,
        path_weight: 1.0,
    }];
    let mut frontier: Vec<usize> = vec![0];
    let mut stats = SearchStats::default();
    let mut n_heads = 1usize;
    let mut first = true;

    while !frontier.is_empty() && stats.expansions < p.expansion_budget {
        if !first {
            sort_frontier(&arena, &mut frontier, p);
        }
        first = false;
        let take = p
            .top_k
            .min(p.expansion_budget - stats.expansions)
            .min(frontier.len());
        let batch: Vec<usize> = frontier.drain(..take).collect();

        let mut opp_obs: Vec<Vec<f32>> = Vec::new();
        let mut new_leaves: Vec<usize> = Vec::new();
        for &ni in &batch {
            expand_node(&mut arena, ni, agent, opp, p, &mut opp_obs, &mut new_leaves);
            stats.expansions += 1;
            stats.max_depth = stats.max_depth.max(arena[ni].depth + 1);
        }
        stats.rounds += 1;

        let n_opp = opp_obs.len();
        let mut obs_batch = opp_obs;
        obs_batch.extend(new_leaves.iter().map(|&li| arena[li].obs.clone()));
        if !obs_batch.is_empty() {
            let q = infer(&obs_batch);
            n_heads = q[0].len();
            evaluate(&mut arena, &batch, &new_leaves, &q, n_opp, p, &mut stats);
        }

        for &li in &new_leaves {
            if arena[li].depth < p.max_depth {
                frontier.push(li);
            }
        }
    }

    resolve(&mut arena, 0, p.gamma, n_heads);
    let values = root_action_values(&arena, p.gamma, n_heads);
    (values, stats)
}

/// Resolve a round's batched forward: opponent rows -> chance weights, leaf rows -> per-head
/// bootstrap + sigma, then write branch weights and child path-weights.
fn evaluate(
    arena: &mut [Node],
    batch: &[usize],
    new_leaves: &[usize],
    q: &[Vec<Vec<f64>>],
    n_opp: usize,
    p: &SearchParams,
    stats: &mut SearchStats,
) {
    // Opponent move probabilities from the head-mean Q (shared across heads, so chance weights stay
    // scalar and sigma reflects only the agent's own value disagreement).
    let opp_probs: Vec<Vec<f64>> = (0..n_opp)
        .map(|i| match &p.opponent {
            Opponent::Distributional { temperature, floor } => {
                softmax_floor(&head_mean(&q[i]), *temperature, *floor)
            }
            Opponent::Uniform => Vec::new(), // uniform registers no opponent observations
        })
        .collect();

    for (j, &li) in new_leaves.iter().enumerate() {
        let leaf_q = &q[n_opp + j]; // [K][A]
        let boot: Vec<f64> = leaf_q
            .iter()
            .map(|row| row.iter().copied().fold(f64::NEG_INFINITY, f64::max))
            .collect();
        arena[li].sigma = std(&boot);
        arena[li].bootstrap = boot;
        stats.leaves += 1;
    }

    // Resolve weights and child path-weights. Collected first, then applied, to avoid borrowing two
    // arena entries (a node and its child) at once.
    let mut weight_updates: Vec<(usize, usize, usize, f64)> = Vec::new();
    let mut path_updates: Vec<(usize, f64)> = Vec::new();
    for &ni in batch {
        let parent_pw = arena[ni].path_weight;
        if let Some(edges) = &arena[ni].edges {
            for (ei, edge) in edges.iter().enumerate() {
                for (bi, b) in edge.branches.iter().enumerate() {
                    let w = match b.deferred {
                        Some((oi, si)) => opp_probs[oi][si],
                        None => b.weight,
                    };
                    weight_updates.push((ni, ei, bi, w));
                    path_updates.push((b.child, parent_pw * w * p.gamma));
                }
            }
        }
    }
    for (ni, ei, bi, w) in weight_updates {
        arena[ni].edges.as_mut().unwrap()[ei].branches[bi].weight = w;
    }
    for (c, pw) in path_updates {
        arena[c].path_weight = pw;
    }
}

fn sort_frontier(arena: &[Node], frontier: &mut [usize], p: &SearchParams) {
    let max_voi = frontier
        .iter()
        .map(|&i| arena[i].path_weight * arena[i].sigma)
        .fold(0.0, f64::max);
    let max_voi = if max_voi > 0.0 { max_voi } else { 1.0 };
    let key = |i: usize| -> (f64, f64, f64) {
        let n = &arena[i];
        let voi = p.beta * (n.path_weight * n.sigma) / max_voi;
        let depth_term = (1.0 - p.beta) * (n.depth as f64) / (p.max_depth as f64);
        (-(voi + depth_term), -(n.depth as f64), -n.path_weight)
    };
    frontier.sort_by(|&a, &b| key(a).partial_cmp(&key(b)).unwrap());
}

fn expand_node(
    arena: &mut Vec<Node>,
    ni: usize,
    agent: usize,
    opp: usize,
    p: &SearchParams,
    opp_obs: &mut Vec<Vec<f32>>,
    new_leaves: &mut Vec<usize>,
) {
    let snakes = arena[ni].snakes.clone();
    let food = arena[ni].food.clone();
    let depth = arena[ni].depth;

    // Opponent branching, shared across the agent's edges at this node.
    let branching: Vec<(Option<Action>, BranchWeight)> = if !snakes[opp].alive {
        vec![(None, BranchWeight::Fixed(1.0))]
    } else {
        let heading = snakes[opp].direction;
        match &p.opponent {
            Opponent::Uniform => {
                let prob = 1.0 / RELATIVE_ACTIONS.len() as f64;
                RELATIVE_ACTIONS
                    .iter()
                    .map(|&r| {
                        (
                            Some(relative_to_absolute(heading, r)),
                            BranchWeight::Fixed(prob),
                        )
                    })
                    .collect()
            }
            Opponent::Distributional { .. } => {
                let oi = opp_obs.len();
                opp_obs.push(egocentric_parts(&snakes, &food, p.grid_size, opp));
                RELATIVE_ACTIONS
                    .iter()
                    .enumerate()
                    .map(|(i, &r)| {
                        (
                            Some(relative_to_absolute(heading, r)),
                            BranchWeight::Deferred(oi, i),
                        )
                    })
                    .collect()
            }
        }
    };

    let agent_heading = snakes[agent].direction;
    let mut edges: Vec<Edge> = Vec::with_capacity(RELATIVE_ACTIONS.len());
    for &agent_rel in RELATIVE_ACTIONS.iter() {
        let agent_abs = relative_to_absolute(agent_heading, agent_rel);
        let mut branches: Vec<Branch> = Vec::with_capacity(branching.len());
        for (opp_abs, bw) in &branching {
            let mut moves: [Option<Action>; 2] = [None, None];
            moves[agent] = Some(agent_abs);
            if let Some(oa) = opp_abs {
                moves[opp] = Some(*oa);
            }
            // Food-free root => no eating => spawn closure is never called.
            let mut sim = SnakeEnv::from_parts(
                p.grid_size,
                p.initial_length,
                p.play_to_last,
                p.win_food_lead,
                snakes.clone(),
                food.clone(),
            );
            let events = sim.advance(moves, || None);
            let reward = p.reward.eval(&events[agent]);

            let child = if sim.done || !sim.snakes[agent].alive {
                push_node(arena, sim.snakes, sim.food, Vec::new(), depth + 1, true)
            } else {
                let obs = egocentric_parts(&sim.snakes, &sim.food, p.grid_size, agent);
                let idx = push_node(arena, sim.snakes, sim.food, obs, depth + 1, false);
                new_leaves.push(idx);
                idx
            };
            let (weight, deferred) = match *bw {
                BranchWeight::Fixed(f) => (f, None),
                BranchWeight::Deferred(oi, si) => (0.0, Some((oi, si))),
            };
            branches.push(Branch {
                weight,
                deferred,
                reward,
                child,
            });
        }
        edges.push(Edge { branches });
    }
    arena[ni].edges = Some(edges);
}

fn push_node(
    arena: &mut Vec<Node>,
    snakes: [Snake; 2],
    food: HashSet<Cell>,
    obs: Vec<f32>,
    depth: i32,
    terminal: bool,
) -> usize {
    arena.push(Node {
        snakes,
        food,
        obs,
        depth,
        terminal,
        edges: None,
        bootstrap: Vec::new(),
        value: Vec::new(),
        sigma: 0.0,
        path_weight: 1.0,
    });
    arena.len() - 1
}

/// One bottom-up pass caching per-head `node.value`: 0 at terminals, the bootstrap at frontier
/// leaves, the per-head max over agent actions of the chance-averaged edge value at decision nodes.
fn resolve(arena: &mut Vec<Node>, idx: usize, gamma: f64, k: usize) {
    if arena[idx].terminal {
        return; // value stays empty; treated as 0 in edge_value
    }
    if arena[idx].edges.is_none() {
        arena[idx].value = arena[idx].bootstrap.clone();
        return;
    }
    let edges = take_edges(arena, idx);
    for edge in &edges {
        for &(_, _, child) in edge {
            resolve(arena, child, gamma, k);
        }
    }
    let mut value = vec![f64::NEG_INFINITY; k];
    for edge in &edges {
        let ev = edge_value(arena, edge, gamma, k);
        for h in 0..k {
            value[h] = value[h].max(ev[h]);
        }
    }
    arena[idx].value = value;
}

/// Snapshot a node's edges as (weight, reward, child) so we can recurse with `&mut arena`.
fn take_edges(arena: &[Node], idx: usize) -> Vec<Vec<(f64, f64, usize)>> {
    arena[idx]
        .edges
        .as_ref()
        .unwrap()
        .iter()
        .map(|e| {
            e.branches
                .iter()
                .map(|b| (b.weight, b.reward, b.child))
                .collect()
        })
        .collect()
}

/// Per-head chance-averaged value of one edge: sum over branches of `w * (r + gamma * child.value)`,
/// with terminal children contributing `w * r` (their continuation value is 0).
fn edge_value(arena: &[Node], branches: &[(f64, f64, usize)], gamma: f64, k: usize) -> Vec<f64> {
    let mut acc = vec![0.0; k];
    for &(w, r, c) in branches {
        if arena[c].terminal {
            for a in acc.iter_mut() {
                *a += w * r;
            }
        } else {
            for (h, a) in acc.iter_mut().enumerate() {
                *a += w * (r + gamma * arena[c].value[h]);
            }
        }
    }
    acc
}

/// Root action values as `[K][A]` (per head, per relative action).
fn root_action_values(arena: &[Node], gamma: f64, k: usize) -> Vec<Vec<f64>> {
    let edges = match &arena[0].edges {
        None => return vec![vec![0.0; RELATIVE_ACTIONS.len()]; k],
        Some(_) => take_edges(arena, 0),
    };
    let per_action: Vec<Vec<f64>> = edges
        .iter()
        .map(|e| edge_value(arena, e, gamma, k))
        .collect(); // [A][K]
    (0..k)
        .map(|h| per_action.iter().map(|ev| ev[h]).collect())
        .collect() // -> [K][A]
}

fn head_mean(q: &[Vec<f64>]) -> Vec<f64> {
    let k = q.len();
    let a = q[0].len();
    (0..a)
        .map(|j| (0..k).map(|h| q[h][j]).sum::<f64>() / k as f64)
        .collect()
}

fn std(values: &[f64]) -> f64 {
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let var = values.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / n;
    var.sqrt()
}

/// `p = (1 - floor) * softmax(q / temperature) + floor / n`, matching `DistributionalSelfPlayOpponent`.
fn softmax_floor(q: &[f64], temperature: f64, floor: f64) -> Vec<f64> {
    let qmax = q.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let z: Vec<f64> = q
        .iter()
        .map(|&v| ((v - qmax) / temperature).exp())
        .collect();
    let zsum: f64 = z.iter().sum();
    let n = q.len() as f64;
    z.iter()
        .map(|&zi| (1.0 - floor) * zi / zsum + floor / n)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::snake::Snake;

    fn snake(cells: &[Cell], dir: Action) -> Snake {
        Snake {
            body: cells.iter().copied().collect(),
            direction: dir,
            alive: true,
        }
    }

    fn params() -> SearchParams {
        SearchParams {
            grid_size: 12,
            initial_length: 3,
            play_to_last: true,
            win_food_lead: None,
            gamma: 0.99,
            beta: 1.0,
            expansion_budget: 30,
            top_k: 4,
            max_depth: 6,
            reward: Reward {
                step: 0.0,
                food: 0.0,
                loss: -10.0,
                draw: -6.0,
                kill: 20.0,
                win: 20.0,
                survival: 0.0,
            },
            opponent: Opponent::Uniform,
        }
    }

    #[test]
    fn fatal_actions_score_the_loss_and_survivable_turns_score_higher() {
        // A in the top-left corner heading Left: Forward (Left) and Right (Up) both run off-grid;
        // only Left (Down) survives. With a zero value function the only signal is the death penalty.
        let snakes = [
            snake(&[(0, 0), (0, 1), (0, 2)], Action::Left),
            snake(&[(6, 6), (6, 7), (6, 8)], Action::Left), // opponent, far away
        ];
        let p = params();
        let (values, stats) = selective_search(&p, snakes, HashSet::new(), 0, |obs| {
            vec![vec![vec![0.0; 3]]; obs.len()]
        });
        let v = &values[0]; // single head
        assert!(
            (v[0] - (-10.0)).abs() < 1e-9,
            "fatal Forward should score the loss: {v:?}"
        );
        assert!(
            (v[2] - (-10.0)).abs() < 1e-9,
            "fatal Right should score the loss: {v:?}"
        );
        assert!(
            v[1] > v[0] && v[1] > v[2],
            "survivable Left should win: {v:?}"
        );
        assert!(stats.expansions > 0 && stats.rounds > 0);
    }

    #[test]
    fn ensemble_bootstrap_sigma_drives_voi_priority() {
        // Two heads that disagree -> nonzero sigma -> beta=1 priority can steer on it (smoke check
        // that multi-head search runs and produces per-head root values).
        let snakes = [
            snake(&[(6, 5), (6, 4), (6, 3)], Action::Right),
            snake(&[(2, 8), (2, 9), (1, 9)], Action::Left),
        ];
        let mut p = params();
        p.expansion_budget = 24;
        let (values, stats) = selective_search(&p, snakes, HashSet::new(), 0, |obs| {
            obs.iter()
                .map(|o| {
                    let s = o.iter().sum::<f32>() as f64;
                    vec![
                        vec![s.sin(), s.cos(), (s * 0.5).sin()],
                        vec![(s + 1.0).sin(), s.cos(), (s * 0.3).sin()],
                    ]
                })
                .collect()
        });
        assert_eq!(values.len(), 2); // two heads
        assert_eq!(values[0].len(), 3);
        assert!(stats.expansions > 0);
    }
}
