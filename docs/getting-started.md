# Getting started

This page uses NumPy and a synchronous `collect` call so the control flow is visible. No ML
framework is required.

## Install

Reinfors requires Python 3.10 or newer. Published wheels target Linux (x86-64 and ARM64), macOS
(Intel and Apple silicon), and 64-bit Windows. On another platform, `pip` may fall back to building
the source distribution, which requires the [Rust development toolchain](development/setup.md).

```bash
pip install reinfors
```

Install adapters only when you need them:

```bash
pip install "reinfors[gym]"
```

For a source checkout, see the [development setup](development/setup.md).

A **record** is one learner-produced training row. `n_games` is the number of episode slots
advanced in parallel, not a total episode count. The [glossary](reference/glossary.md) defines
record floors, heads, ticks, CSR, and search terms when you need them.

## Collect a search batch

```python
import numpy as np
import reinfors as rf

game = rf.games.Connect4()
engine = rf.Engine(
    game=game,
    reward=rf.Reward(win=1.0, loss=-1.0, draw=0.0),
    policy=rf.policies.AlphaZero(num_simulations=64),
    learner=rf.learners.AlphaZero(),
    n_games=16,
    seed=7,
)

n_actions = game.action_space().n

def infer(obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return policy logits and a value for each observation."""
    rows = obs.shape[0]
    logits = np.zeros((rows, n_actions), dtype=np.float64)
    values = np.zeros(rows, dtype=np.float64)
    return logits, values

batch = engine.collect(n_records=512, infer=infer)

print(batch.obs.shape)       # flattened float32 observations
print(batch.policy_targets.shape)  # search-improved move distributions
print(batch.value_targets.shape)   # completed-game outcomes
print(batch.players.shape)   # player perspective for every record
print(batch.telemetry)       # collection and search measurements
```

This is the AlphaZero loop: the callback supplies policy logits and position values, search improves
the policy, and completed games produce value targets. Zero logits give every legal move the same
prior here. See the [AlphaZero overview and source](catalogue/algorithms.md#alphazero) or the
[complete AlphaZero training example](examples/index.md#alphazero-connect-4).

## Replace the callback

The zero callback is deliberately uninteresting. A real callback can reshape observations,
move them to an accelerator, run any model, and return host NumPy arrays:

```python
def infer(obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = framework_tensor(obs).reshape(len(obs), *game.observation_space().shape)
    logits, values = model(x.to(device))
    return (
        logits.detach().cpu().numpy().astype(np.float64, copy=False),
        values.detach().cpu().numpy().astype(np.float64, copy=False),
    )
```

The exact output depends on the chosen policy. See the [inference contract](reference/inference-contract.md)
and [batch formats](reference/batch-formats.md). If the callback is rejected, match its exception in
[troubleshooting](reference/troubleshooting.md).

## Next steps

- Add an optimizer using the complete [training-loop guide](guides/training.md).
- Overlap collection and learning using [concurrent collection](guides/streaming.md).
- Browse runnable [examples](examples/index.md).
- Compare [algorithm compositions](catalogue/algorithms.md).
