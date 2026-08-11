# Architecture

Reinfors separates reusable rules, search, record generation, and user-owned learning.

```text
CALLER-OWNED PYTHON

┌──────────────────────────────────────────────────────────────────────┐
│ Training loop calls Engine.collect(...) or reads a CollectStream     │
└──────────────────────────────────┬───────────────────────────────────┘
                                   │
══════════════════════ RUST / PYTHON BOUNDARY ═════════════════════════
                                   ▼
RUST: ENGINE AND COMPOSED COMPONENTS

┌──────────────────────────────────────────────────────────────────────┐
│ Engine                                                               │
│ owns active game slots, component instances, cache, batching,        │
│ record floor, stream queue, snapshots, and telemetry                 │
└──────────────────────────────────┬───────────────────────────────────┘
                                   ▼
              ┌───────────────────────────────┐
              │ Game                          │
              │ state · rules · legal actions │
              │ chance · terminal status      │
              └───────┬───────────────▲───────┘
      state + player  │               │ selected action / transition
      perspective     │               │
                      ▼               ├──────────────────────────┐
              ┌───────────────┐       │                  ┌───────┴────────┐
              │ Encoder       │       │                  │ Policy/search  │
              │ observation + │       └──────────────────│ uses Game +    │
              │ action view   │                          │ network data   │
              └───────┬───────┘                          └───────┬────────┘
                      │ encoded evaluation request               │
                      ▼                                          │
              ┌───────────────────────────────┐                  │
              │ Engine inference pool         │◄─────────────────┘
              │ batches requests from games,  │
              │ players, and search leaves    │
              └───────────────────┬───────────┘
                                  │ pooled observation array
══════════════════════ RUST / PYTHON BOUNDARY ═════════════════════════
                                  ▼
CALLER-OWNED PYTHON

              ┌──────────────────────────────────────────┐
              │ infer callback                           │
              │ NumPy → arbitrary network/device/RPC →   │
              │ policy values, action values, or logits  │
              └───────────────────┬──────────────────────┘
                                  │ prediction arrays
══════════════════════ RUST / PYTHON BOUNDARY ═════════════════════════
                                  ▼
RUST

              ┌──────────────────────────────────────────┐
              │ Engine routes each prediction row back   │
              │ to the policy/search that requested it   │
              └───────────────────┬──────────────────────┘
                                  │
                         search continues, or
                         an action is selected
                                  │
              ┌───────────────────▼──────────────────────┐
              │ Game applies actions and chance          │
              │ transitions, producing ordered events    │
              └──────────────┬────────────────┬──────────┘
                             │                │ events
                transition / │                ▼
                search data  │       ┌───────────────────┐
                             │       │ Reward            │
                             │       │ events → rewards  │
                             │       └─────────┬─────────┘
                             ▼                 ▼
              ┌──────────────────────────────────────────┐
              │ Learner                                  │
              │ Engine-supplied trajectory, decision,    │
              │ and search data + rewards → records      │
              └───────────────────┬──────────────────────┘
                                  │ records accumulate to requested floor
              ┌───────────────────▼──────────────────────┐
              │ completed Engine batch                   │
              │ (direct return or CollectStream queue)   │
              └───────────────────┬──────────────────────┘
                                  │
══════════════════════ RUST / PYTHON BOUNDARY ═════════════════════════
                                  ▼
CALLER-OWNED PYTHON

              replay/buffer → loss → optimizer → updated network
                                  │
                                  └──────────────► next collection cycle
```

## The primary execution surfaces

`Engine` owns many active episodes. A policy asks for evaluations, the engine pools those
requests, and a learner converts completed decisions or trajectories into training records.
Use it for policy-driven, learner-shaped data generation, including direct value learning and
search-guided methods (for example, DQN, TreeStrap, and AlphaZero).

`Env` owns one game instance and lets the caller supply each decision. Use it for evaluation,
interactive play, custom agents, and Gymnasium/PettingZoo adapters. It exposes observations,
legal actions, active agents, event traces, rewards, snapshots, and forks.

`Arena` owns concurrent, paired evaluation games. It pools native policy search across ready games
and can place one external contestant on bounded worker lanes. Use it when the deployed agent
includes search or when a subprocess engine must play alongside searched contestants.

Algorithms whose traversal does not fit policy-driven episode collection use standalone solvers (e.g. CFR).

## Component responsibilities

| Component | Responsibility |
| --- | --- |
| `Game` | State transitions, active agents, legal actions, explicit chance distributions, terminal state, and information-state identity. |
| `Encoder` | Convert a valid game state and agent perspective into a fixed observation and action-head view. |
| `Reward` | Map ordered transition events to per-agent scalar rewards. |
| `Policy` | Choose decisions and define any search/evaluation process used to do so. |
| `Learner` | Decide which training records a policy trajectory produces. |
| `Engine` | Run games concurrently, batch inference, assemble records, cache evaluations, and report telemetry. |
| `Arena` | Run paired evaluation games, pool searched inference, and schedule external agents. |
| `Env` | Expose one caller-controlled game instance. |
| `Solver` | Own algorithm-specific traversals and persistent state outside the policy/learner engine model. |

The [native component contracts](../extending/component-contracts.md) describe the implementation
contracts in detail.

## Why the inference boundary uses arrays

The engine needs evaluated rows, not a particular model object. NumPy is the stable Python
ABI at that boundary and lets callers wrap PyTorch, JAX, ONNX Runtime, a custom accelerator,
or RPC. Legal actions remain a property of the game and are enforced around inference by
search, selection, and loss construction; masks are not model inputs.

This keeps the single performance-critical seam small while leaving model architecture and
deployment open.

## Next steps

- Put the components together in the [training-loop guide](../guides/training.md).
- Read the mandatory callback shapes and validation rules in the
  [inference contract](../reference/inference-contract.md).
- Compare supported compositions in the [algorithm catalogue](../catalogue/algorithms.md).
