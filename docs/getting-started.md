# Getting started

This page uses NumPy and a synchronous `collect` call so the control flow is visible. No ML
framework is required.

## Install

```bash
pip install reinfors
```

Install adapters only when you need them:

```bash
pip install "reinfors[gym]"
```

For a source checkout, see the [development setup](development/setup.md).

## Collect a search batch

```python
import numpy as np
import reinfors as rf

game = rf.games.Connect4()
engine = rf.Engine(
    game=game,
    reward=rf.Reward(win=1.0, loss=-1.0, draw=0.0),
    policy=rf.policies.Mcts(num_simulations=64),
    learner=rf.learners.TreeStrap(),
    n_games=16,
    seed=7,
)

n_actions = game.action_space().n

def infer(obs: np.ndarray) -> np.ndarray:
    """Return one action-value head for each observation."""
    return np.zeros((obs.shape[0], 1, n_actions), dtype=np.float64)

batch = engine.collect(n_records=512, infer=infer)

print(batch.obs.shape)       # flattened float32 observations
print(batch.targets.shape)   # searched action-value targets
print(batch.masks.shape)     # bootstrap-head membership
print(batch.players.shape)   # player perspective for every record
print(batch.telemetry)       # collection and search measurements
```

`512` is a record floor, not necessarily the exact returned row count: completed episode
and search work can make a batch slightly larger.

## Replace the callback

The zero callback is deliberately uninteresting. A real callback can reshape observations,
move them to an accelerator, run any model, and return host NumPy arrays:

```python
def infer(obs: np.ndarray) -> np.ndarray:
    x = framework_tensor(obs).reshape(len(obs), *game.observation_space().shape)
    q = model(x.to(device))
    return q.detach().cpu().numpy().astype(np.float64, copy=False)
```

The exact output depends on the chosen policy. See the [inference contract](reference/inference-contract.md)
and [batch formats](reference/batch-formats.md).

## Next steps

- Add an optimizer using the complete [training-loop guide](guides/training.md).
- Overlap collection and learning using [concurrent collection](guides/streaming.md).
- Browse runnable [examples](examples/index.md).
- Compare [algorithm compositions](catalogue/algorithms.md).
