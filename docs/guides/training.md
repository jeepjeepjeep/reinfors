# Build a training loop

An engine composition fixes the game, reward, policy, learner, and collection topology. Your
Python loop supplies inference and consumes the typed batch produced by that learner.

## 1. Compose an engine

```python
import reinfors as rf

game = rf.games.Snake(grid_size=12, num_snakes=2, max_ticks=200)
engine = rf.Engine(
    game,
    rf.Reward(food=1.0, loss=-10.0, win=10.0, draw=-5.0),
    rf.policies.SelectiveExpectimax(
        expansion_budget=32,
        top_k=4,
        max_depth=6,
        n_heads=8,
        chance=rf.chance_modes.Committed(samples=1),
    ),
    rf.learners.TreeStrap(gamma=0.99, outcome_weight=0.3),
    n_games=32,
    seed=0,
)
```

Constructors validate incompatible compositions before collection begins. Export
`engine.resolved_config()` with experiment artifacts so defaults are recorded too.

## 2. Adapt your network once

The callback receives a contiguous or non-contiguous two-dimensional `float32` observation
batch. Reshape according to `game.observation_space().shape`, run inference without gradients,
and return the policy-specific `float64` output.

Do not apply a dense argmax or max over illegal actions in your own loss. DQN batches provide
sparse legal action ids for target calculation; search policies obtain legality directly from
the game.

## 3. Collect and optimize

```python
for step in range(num_updates):
    batch = engine.collect(records_per_update, infer)
    loss = train(batch)
    engine.weights_updated()
    report(batch.telemetry, loss)
```

Prefer named fields such as `batch.obs` and `batch.targets`. Positional unpacking remains for
compatibility but makes algorithm-specific code less self-documenting.

## Per-player models

Pass one callback per player when policies differ:

```python
batch = engine.collect(records_per_update, [blue_infer, red_infer])
```

The callback sequence length must match the game player count. Use `batch.players` to route
records into the right optimizer or replay buffer. Set `learn_players=[0]` on `Engine` for a
frozen-opponent experiment; all players still act, but only player 0 emits training rows.

## Framework examples

The maintained PyTorch scripts demonstrate full losses rather than hiding them in the
library:

- `scripts/train_example.py`: TreeStrap with selective expectimax or UCT MCTS;
- `scripts/train_alphazero_example.py`: policy/value loss, evaluation, and streaming;
- `scripts/train_dqn_holdem.py`: sparse legal-action DQN targets;
- `scripts/train_deep_cfr.py`: caller-owned Deep CFR buffers and networks.

See [examples](../examples/index.md) for commands and learning objectives.
