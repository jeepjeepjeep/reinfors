//! Best-first selective expectimax, ported to match `snake_RL`'s `SelectiveExpectimaxPlanner`
//! (single-head, `UniformOpponent`, food-free root — the configuration its own equivalence tests
//! use). Leaf values come from an injected evaluator (`infer`): in production that's the Python
//! network forward; the differential test passes the same value function the oracle uses.
//!
//! The tree is an arena (`Vec<Node>` indexed by `usize`) so the frontier can hold node references
//! without fighting the borrow checker. Agent plies are MAX nodes; the opponent move is a chance
//! node averaged over with uniform weights.

use std::collections::HashSet;

use crate::action::{relative_to_absolute, Action, RELATIVE_ACTIONS};
use crate::obs::egocentric_parts;
use crate::reward::Reward;
use crate::snake::{Cell, Snake, SnakeEnv};

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
}

#[derive(Default, Clone, Copy, Debug)]
pub struct SearchStats {
    pub max_depth: i32,
    pub expansions: usize,
    pub leaves: usize,
    pub rounds: usize,
}

struct Branch {
    weight: f64,
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
    bootstrap: f64,
    value: f64,
    sigma: f64, // always 0 for the single-head port; kept so the priority key matches the oracle
    path_weight: f64,
}

/// Run a best-first selective-expectimax search from `(snakes, food)` for `agent` (0 or 1), and
/// return the root action values (one per relative action: Forward, Left, Right) plus diagnostics.
///
/// `infer` maps a batch of flat observations to per-observation action values (length = 3); the leaf
/// bootstrap is the max over those, matching the planner.
pub fn selective_search<F>(
    p: &SearchParams,
    snakes: [Snake; 2],
    food: HashSet<Cell>,
    agent: usize,
    mut infer: F,
) -> (Vec<f64>, SearchStats)
where
    F: FnMut(&[Vec<f32>]) -> Vec<Vec<f64>>,
{
    let opp = 1 - agent;
    let mut arena: Vec<Node> = vec![Node {
        snakes,
        food,
        obs: Vec::new(),
        depth: 0,
        terminal: false,
        edges: None,
        bootstrap: 0.0,
        value: 0.0,
        sigma: 0.0,
        path_weight: 1.0,
    }];
    let mut frontier: Vec<usize> = vec![0];
    let mut stats = SearchStats::default();
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

        let mut new_leaves: Vec<usize> = Vec::new();
        for &ni in &batch {
            expand_node(&mut arena, ni, agent, opp, p, &mut new_leaves);
            stats.expansions += 1;
            stats.max_depth = stats.max_depth.max(arena[ni].depth + 1);
        }
        stats.rounds += 1;

        if !new_leaves.is_empty() {
            let obs_batch: Vec<Vec<f32>> =
                new_leaves.iter().map(|&li| arena[li].obs.clone()).collect();
            let q = infer(&obs_batch);
            for (&li, row) in new_leaves.iter().zip(q.iter()) {
                arena[li].bootstrap = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                stats.leaves += 1;
            }
        }

        for &li in &new_leaves {
            if arena[li].depth < p.max_depth {
                frontier.push(li);
            }
        }
    }

    resolve(&mut arena, 0, p.gamma);
    let values = root_action_values(&arena, p.gamma);
    (values, stats)
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
    // Stable sort, matching Python's list.sort on the same key tuple.
    frontier.sort_by(|&a, &b| key(a).partial_cmp(&key(b)).unwrap());
}

fn expand_node(
    arena: &mut Vec<Node>,
    ni: usize,
    agent: usize,
    opp: usize,
    p: &SearchParams,
    new_leaves: &mut Vec<usize>,
) {
    let snakes = arena[ni].snakes.clone();
    let food = arena[ni].food.clone();
    let depth = arena[ni].depth;
    let path_weight = arena[ni].path_weight;

    // Opponent branching: uniform over its relative actions, or a single null branch if it is dead.
    let branching: Vec<(Option<Action>, f64)> = if snakes[opp].alive {
        let heading = snakes[opp].direction;
        let prob = 1.0 / RELATIVE_ACTIONS.len() as f64;
        RELATIVE_ACTIONS
            .iter()
            .map(|&r| (Some(relative_to_absolute(heading, r)), prob))
            .collect()
    } else {
        vec![(None, 1.0)]
    };

    let agent_heading = snakes[agent].direction;
    let mut edges: Vec<Edge> = Vec::with_capacity(RELATIVE_ACTIONS.len());
    for &agent_rel in RELATIVE_ACTIONS.iter() {
        let agent_abs = relative_to_absolute(agent_heading, agent_rel);
        let mut branches: Vec<Branch> = Vec::with_capacity(branching.len());
        for &(opp_abs, weight) in &branching {
            let mut moves: [Option<Action>; 2] = [None, None];
            moves[agent] = Some(agent_abs);
            if let Some(oa) = opp_abs {
                moves[opp] = Some(oa);
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
            arena[child].path_weight = path_weight * weight * p.gamma;
            branches.push(Branch {
                weight,
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
        bootstrap: 0.0,
        value: 0.0,
        sigma: 0.0,
        path_weight: 1.0,
    });
    arena.len() - 1
}

/// One bottom-up pass caching `node.value`: 0 at terminals, the net bootstrap at frontier leaves,
/// the max over agent actions of the chance-averaged edge value at expanded decision nodes.
fn resolve(arena: &mut Vec<Node>, idx: usize, gamma: f64) -> f64 {
    if arena[idx].terminal {
        arena[idx].value = 0.0;
        return 0.0;
    }
    if arena[idx].edges.is_none() {
        let v = arena[idx].bootstrap;
        arena[idx].value = v;
        return v;
    }
    let edges = take_edges(arena, idx);
    for edge in &edges {
        for &(_, _, child) in edge {
            resolve(arena, child, gamma);
        }
    }
    let value = edges
        .iter()
        .map(|edge| edge_value(arena, edge, gamma))
        .fold(f64::NEG_INFINITY, f64::max);
    arena[idx].value = value;
    value
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

fn edge_value(arena: &[Node], branches: &[(f64, f64, usize)], gamma: f64) -> f64 {
    branches
        .iter()
        .map(|&(w, r, c)| w * (r + gamma * arena[c].value))
        .sum()
}

fn root_action_values(arena: &[Node], gamma: f64) -> Vec<f64> {
    match &arena[0].edges {
        None => vec![0.0; RELATIVE_ACTIONS.len()],
        Some(edges) => edges
            .iter()
            .map(|e| {
                let branches: Vec<(f64, f64, usize)> = e
                    .branches
                    .iter()
                    .map(|b| (b.weight, b.reward, b.child))
                    .collect();
                edge_value(arena, &branches, gamma)
            })
            .collect(),
    }
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
            vec![vec![0.0; 3]; obs.len()]
        });
        // values are [Forward, Left, Right].
        assert!(
            (values[0] - (-10.0)).abs() < 1e-9,
            "fatal Forward should score the loss: {values:?}"
        );
        assert!(
            (values[2] - (-10.0)).abs() < 1e-9,
            "fatal Right should score the loss: {values:?}"
        );
        assert!(
            values[1] > values[0] && values[1] > values[2],
            "survivable Left should win: {values:?}"
        );
        assert!(stats.expansions > 0 && stats.rounds > 0);
    }
}
