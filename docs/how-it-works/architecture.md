# Architecture

Reinfors separates reusable rules, search, record generation, and user-owned learning.

```text
                         caller-owned Python
                   ┌──────────────────────────┐
                   │ network · optimizer      │
                   │ replay · devices · RPC   │
                   └────────────┬─────────────┘
                                │ infer callback
                                │ NumPy in / NumPy out
┌───────────────────────────────┼────────────────────────────────┐
│ Rust                          ▼                                │
│ Game + Encoder + Reward → Policy + Learner → Engine → Batch   │
│          rules/state       search/records    parallel games   │
└────────────────────────────────────────────────────────────────┘

Env: caller-driven play and evaluation
Solver: algorithm-owned traversal for CFR and Deep CFR
```

## The two primary execution surfaces

`Engine` owns many active episodes. A policy asks for evaluations, the engine pools those
requests, and a learner converts completed decisions or trajectories into training records.
Use it for DQN, TreeStrap/MCTS/expectimax, and AlphaZero data generation.

`Env` owns one game instance and lets the caller supply each decision. Use it for evaluation,
interactive play, custom agents, and Gymnasium/PettingZoo adapters. It exposes observations,
legal actions, active agents, event traces, rewards, snapshots, and forks.

CFR-family algorithms use standalone solvers because the solver owns the traversal and its
state. Tabular CFR trains internally; Deep CFR calls per-player advantage inference and
returns samples for caller-owned buffers and optimization.

## Component responsibilities

| Component | Responsibility |
| --- | --- |
| `Game` | State transitions, active agents, legal actions, explicit chance distributions, terminal state, and information-state identity. |
| `Encoder` | Convert a valid game state and agent perspective into a fixed observation and action-head view. |
| `Reward` | Map ordered transition events to per-agent scalar rewards. |
| `Policy` | Choose decisions and define any search/evaluation process used to do so. |
| `Learner` | Decide which training records a policy trajectory produces. |
| `Engine` | Run games concurrently, batch inference, assemble records, cache evaluations, and report telemetry. |
| `Env` | Expose one caller-controlled game instance. |
| `Solver` | Own algorithm-specific traversals and persistent state outside the policy/learner engine model. |

The [Rust component guide](../extending/rust-components.md) describes the implementation
contracts in detail.

## Why the inference boundary uses arrays

The engine needs evaluated rows, not a particular model object. NumPy is the stable Python
ABI at that boundary and lets callers wrap PyTorch, JAX, ONNX Runtime, a custom accelerator,
or RPC. Legal actions remain a property of the game and are enforced around inference by
search, selection, and loss construction; masks are not model inputs.

This keeps the one hot seam small while leaving model architecture and deployment open.
