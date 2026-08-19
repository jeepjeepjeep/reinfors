# Examples

Start with the smallest script that answers your question. Model and optimizer code remains visible
because it belongs to the experiment rather than a reinfors abstraction.

## First steps

### Quickstart

`examples/quickstart.py` demonstrates synchronous [AlphaZero](../catalogue/algorithms.md#alphazero)
collection with a NumPy callback and no extra dependencies.

**Runtime:** seconds for the default smoke run on a laptop CPU.

```bash
python examples/quickstart.py
```

### Train GridWorld

`examples/train_gridworld.py` is the smallest complete [DQN](../catalogue/algorithms.md#dqn)
training loop. Install `reinfors[train]`.

**Runtime:** seconds for the ten-update default on a laptop CPU.

```bash
python examples/train_gridworld.py
```

### Streaming

`examples/streaming.py` demonstrates bounded
[TreeStrap + UCT MCTS](../catalogue/algorithms.md#treestrap-uct-mcts) collection, queue inspection,
and clean shutdown with no extra dependencies.

**Runtime:** seconds for the three-batch default on a laptop CPU.

```bash
python examples/streaming.py --updates 3
```

### TensorBoard telemetry

`examples/telemetry_tensorboard.py` writes search, inference, episode-length, and return metrics to
TensorBoard. Install `reinfors[train]`.

**Runtime:** seconds to about a minute for the default on a laptop CPU.

```bash
python examples/telemetry_tensorboard.py --updates 10
```

## End-to-end training

These scripts include more experiment machinery—replay, evaluation, CLI configuration, or
reporting—because they demonstrate complete research workflows.

### TreeStrap Snake

`examples/train_treestrap_snake.py` trains ensemble Q-values from TreeStrap targets using either
[selective expectimax](../catalogue/algorithms.md#treestrap-selective-expectimax) or
[UCT MCTS](../catalogue/algorithms.md#treestrap-uct-mcts).

**Runtime:** minutes for the default; use `--iterations 1` for a smoke test.

```bash
python examples/train_treestrap_snake.py --iterations 20
```

### Train PPO

`examples/train_ppo_connect4.py` trains [PPO](../catalogue/algorithms.md#ppo) self-play with a
shared actor-critic: clipped-ratio epochs over collected batches, legality-masked log-probs, GAE
advantages, and value clipping against the collection-time critic. Install `reinfors[train]`.

**Runtime:** the twenty-update default finishes in seconds on a laptop CPU.

```bash
python examples/train_ppo_connect4.py
```

### AlphaZero Connect 4

`examples/train_alphazero_example.py` covers policy/value heads, visit targets, synchronous or
concurrent collection, evaluation, and saving a network.

**Runtime:** minutes for the 40-iteration CPU default; one iteration checks plumbing.

```bash
python examples/train_alphazero_example.py
```

### AlphaZero Snake

`examples/train_alphazero_snake.py` applies AlphaZero to simultaneous actions, stochastic search,
and unbounded values.

**Runtime:** minutes or longer for the default, scaling strongly with simulations and collection size.

```bash
python examples/train_alphazero_snake.py
```

### DQN Hold'em

`examples/train_dqn_holdem.py` covers imperfect observations, replay, ensemble DQN, and sparse legal
actions. Its flags assemble the full
[Rainbow](https://arxiv.org/abs/1710.02298) stack and compose freely — `--double` (decoupled
target selection), `--per` (prioritized replay), `--dueling` (value/advantage heads), `--c51`
(distributional heads), `--noisy` (weight-noise exploration), and `--n-step` (multi-step
returns):

```bash
python examples/train_dqn_holdem.py --double --per --dueling --c51 --noisy --n-step 3 --heads 1
```

Every component except `--n-step` is caller-side; see the
[DqnBatch guidance](../reference/batch-formats.md#dqnbatch) for the recipes.

**Runtime:** minutes or longer for the default; use `--iterations 1 --eval-every 0` for a smoke test.

```bash
python examples/train_dqn_holdem.py
```

### Deep CFR training

`examples/train_deep_cfr.py` demonstrates per-player advantage inference, caller-owned reservoirs,
and advantage/strategy training.

**Runtime:** long-running at the default; reduce iterations and training steps for a smoke test.

```bash
python examples/train_deep_cfr.py
```

## Solving and evaluation

### Play Connect4

`examples/play_connect4.py` renders a terminal board and accepts numbered keyboard input through
`Env`. It plays against a seeded random opponent by default; pass `--opponent human` for two people.
No optional dependencies are required.

**Runtime:** continues until the game ends or the player enters `q`.

```bash
python examples/play_connect4.py
```

### Solve Leduc

`examples/solve_leduc.py` runs tabular CFR, CFR+, or external-sampling MCCFR and reports exact
exploitability where the game is enumerable.

**Runtime:** seconds to minutes, depending on the game, variant, iterations, and exact probes.

```bash
python examples/solve_leduc.py --iterations 1000
```

### AlphaZero head-to-head

`examples/eval_az_h2h.py` referees two saved AlphaZero Connect 4 networks with alternating seats and
sampled opening diversity.

**Runtime:** seconds for the default search-free referee after checkpoint loading.

```bash
python examples/eval_az_h2h.py a.pt b.pt --games 200 --opening-plies 4
```

### Arena evaluation

`examples/eval_arena.py` runs paired, seat-swapped Connect Four matches across concurrent slots. It
demonstrates batched AlphaZero search against an external agent, seeded openings, timeouts, and
pair-level uncertainty without optional dependencies.

**Runtime:** seconds for the default CPU smoke run.

```bash
python examples/eval_arena.py --games 20 --simulations 32 --slots 8
```

## Adapters and validation

Start with the [Gymnasium and PettingZoo guide](../guides/adapters.md); the adapter tests under
`tests/` provide additional compliance-level examples. For algorithm-specific array meanings,
consult [batch formats](../reference/batch-formats.md) before copying a loss.

## Next steps

- Build from the smallest DQN example with the [training guide](../guides/training.md).
- Compare saved agents with the [evaluation guide](../guides/evaluation.md) and
  [Arena guide](../guides/arena.md).
- Add experiment metrics with [telemetry and TensorBoard](../guides/telemetry.md).
