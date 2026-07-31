# Examples

Start with the smallest example that answers your question. The scripts are deliberately
plain Python: model and optimizer code remains visible because it is part of the experiment,
not a reinfors abstraction.

## First steps

| Example | What it demonstrates | Extra dependencies |
| --- | --- | --- |
| `examples/quickstart.py` | Synchronous MCTS collection with a NumPy callback | None |
| `examples/train_gridworld.py` | Complete GridWorld DQN training loop | `reinfors[train]` |
| `examples/streaming.py` | Bounded background collection and clean shutdown | None |
| `examples/telemetry_tensorboard.py` | Structured telemetry written to TensorBoard | `reinfors[train]` and `tensorboard` |

From a source checkout:

```bash
python examples/quickstart.py
python examples/train_gridworld.py
python examples/streaming.py
python examples/telemetry_tensorboard.py --updates 10
```

## End-to-end training

<!-- TODO: Simplify the advanced examples below. Each should foreground one reinfors workflow;
move experiment-specific evaluation, replay, CLI, and reporting machinery into separate reference
scripts where needed. -->

| Script | Composition | Concepts |
| --- | --- | --- |
| `examples/train_example.py` | Snake + TreeStrap + selective expectimax or UCT | Ensemble Q-values, searched targets, PyTorch training |
| `examples/train_alphazero_example.py` | Connect 4 + AlphaZero | Policy/value heads, visit targets, concurrent collection, evaluation |
| `examples/train_alphazero_snake.py` | Snake + AlphaZero | Simultaneous actions, stochastic search, unbounded values |
| `examples/train_dqn_holdem.py` | Hold'em + DQN | Imperfect observations, sparse legal-action targets |
| `examples/train_deep_cfr.py` | Poker + Deep CFR | Per-player advantage inference, caller-owned reservoirs |

```bash
python examples/train_example.py --iterations 20
python examples/train_alphazero_example.py --iterations 40 --depth 1
python examples/train_alphazero_snake.py --help
python examples/train_dqn_holdem.py --help
python examples/train_deep_cfr.py --help
```

## Solving and evaluation

| Example | What it demonstrates |
| --- | --- |
| `examples/solve_leduc.py` | Tabular CFR variants and exact exploitability |
| `examples/eval_az_h2h.py` | Head-to-head evaluation of saved AlphaZero networks |

```bash
python examples/solve_leduc.py --iterations 1000
python examples/eval_az_h2h.py --help
```

## Adapters and validation

The adapter tests under `tests/` are concise examples of Gymnasium and PettingZoo usage. For
algorithm-specific array meanings, consult [batch formats](../reference/batch-formats.md)
before copying a loss.

!!! note "Notebooks"

    Notebooks will be added only where an interactive narrative provides value beyond the
    maintained scripts. Keeping canonical training code in importable scripts makes it easier
    to test and less likely to drift.
