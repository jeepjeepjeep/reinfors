# Build a training loop

This example trains a small DQN on GridWorld. Install the training dependencies first:

```bash
pip install "reinfors[train]"
```

With the published package, save the program below as `train_gridworld.py` and run
`python train_gridworld.py`. From a source checkout, the identical maintained copy is already
available:

```bash
python examples/train_gridworld.py
```

The program collects ten batches, applies one DQN update per batch, and prints the record count,
loss, and mean completed-episode return. It normally finishes in seconds rather than minutes on a
laptop CPU. Expect noisy loss and return values, not monotonic curves: this short run verifies the
complete data and optimization path rather than training a converged agent.

The [glossary](../reference/glossary.md) defines records, ensemble heads, truncation tails, and the
other reinfors-specific terms used below. The five-step
[collection round](../concepts/sampling-and-training.md#a-collection-round) explains how parallel
games and inference pooling produce this batch.

```python
"""Train a small DQN on GridWorld."""

import copy

import numpy as np
import reinfors as rf
import torch
from torch import nn
from torch.nn import functional as F

UPDATES = 10
RECORDS_PER_UPDATE = 256
GAMMA = 0.99
TARGET_SYNC = 5

torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configure reinfors
game = rf.games.GridWorld(size=5, goal_row=0, goal_col=4, max_ticks=50)  # 50 game-time action boundaries
obs_size = int(np.prod(game.observation_space().shape))
n_actions = game.action_space().n
engine = rf.Engine(
    game=game,
    reward=rf.Reward(step=-0.01, goal=1.0),  # Weight the events emitted by GridWorld.
    policy=rf.policies.EpsilonGreedyQ(n_heads=1, epsilon=0.1),  # Explore on 10% of decisions.
    learner=rf.learners.Dqn(),  # Emit one-step transition records for the DQN loss.
    n_games=8,  # Advance eight independent episode slots in parallel.
    seed=0,
)

# Build the network and inference callback
net = nn.Sequential(
    nn.Linear(obs_size, 64),
    nn.ReLU(),
    nn.Linear(64, n_actions),
).to(device)
target_net = copy.deepcopy(net).eval()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)


def infer(obs_batch: np.ndarray) -> np.ndarray:
    """Return one ensemble head's Q-values for each observation."""
    net.eval()
    with torch.no_grad():
        obs = torch.from_numpy(np.ascontiguousarray(obs_batch)).to(device)
        return net(obs).unsqueeze(1).cpu().double().numpy()


for update in range(1, UPDATES + 1):
    # Collect a batch of experience.
    batch = engine.collect(n_records=RECORDS_PER_UPDATE, infer=infer)

    obs = torch.as_tensor(batch.obs, device=device)
    actions = torch.as_tensor(batch.actions, device=device)
    rewards = torch.as_tensor(batch.rewards, dtype=torch.float32, device=device)
    next_obs = torch.as_tensor(batch.next_obs, device=device)
    can_bootstrap = torch.as_tensor(batch.can_bootstrap, device=device)

    # Compute DQN targets.
    net.train()
    chosen_q = net(obs).gather(1, actions[:, None]).squeeze(1)
    with torch.no_grad():
        next_value = target_net(next_obs).max(dim=1).values
        targets = rewards + GAMMA * torch.where(can_bootstrap, next_value, 0.0)

    # Update the online network.
    loss = F.smooth_l1_loss(chosen_q, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Periodically sync the target network.
    if update % TARGET_SYNC == 0:
        target_net.load_state_dict(net.state_dict())

    # Report completed-episode return beside the optimization signal.
    episode_returns = [returns[0] for returns, _length, _seeded in batch.telemetry["episodes"]]
    mean_return = np.mean(episode_returns) if episode_returns else float("nan")
    print(f"update={update} records={len(batch.obs)} loss={loss.item():.3f} mean_return={mean_return:.3f}")
```

## What reinfors provides

The engine runs GridWorld episodes concurrently using the epsilon-greedy policy, batches calls to
`infer`, and returns the transitions needed for the DQN update. The network, target network,
optimizer, and loss remain ordinary PyTorch and can be replaced without changing the engine.
The configured epsilon is fixed for this engine; see
[policy and learner knobs](../reference/python-api.md#policy-and-learner-knobs) before adding a
schedule.

The inference callback is the adapter between the two sides. Here the network produces one Q-value
per action, and `unsqueeze(1)` adds the head axis expected by `EpsilonGreedyQ(n_heads=1)`. The
head axis is an ensemble dimension rather than a distinct policy/value model output. The callback
returns NumPy `float64` values; action legality stays inside the game and policy. Use the
[troubleshooting table](../reference/troubleshooting.md) when a callback or composition is rejected.

Every GridWorld move is legal, so the target network can take a dense maximum over its four outputs.
`batch.can_bootstrap` is false when the TD target should contain only the immediate reward. Games
with variable legal actions must also maximize over their provided legal action IDs. The
[Hold'em DQN example](../examples/index.md#dqn-holdem) demonstrates that general case.

`batch.telemetry` contains collection timings, inference counts, and episode summaries; see
[telemetry](telemetry.md) for reporting and TensorBoard output.

## Taking the loop further

This first example deliberately trains directly on each collected batch. Typical experiments add a
replay buffer, minibatching, checkpoints, evaluation, and concurrent collection. The maintained
scripts build those pieces on the same interface:

- [Hold'em DQN](../examples/index.md#dqn-holdem): replay, minibatching, sparse legality,
  evaluation, and ensemble DQN;
- [AlphaZero](../examples/index.md#alphazero-connect-4): search with policy and value heads;
- [TreeStrap](../examples/index.md#treestrap-snake): selective expectimax or UCT MCTS;
- [Deep CFR](../examples/index.md#deep-cfr-training): caller-owned buffers and networks.

Use `engine.resolved_config()` alongside checkpoints to record all constructor defaults; the
[configuration and checkpoints guide](configuration-and-checkpoints.md) shows the complete restore
workflow. If the experiment enables inference caching, follow the
[cache lifecycle guide](configuration-and-checkpoints.md#inference-cache-lifecycle).

## Per-player models

For a separate two-player experiment—for example, Connect 4 with a frozen opponent—pass one
callback per player to that game's Engine. This is not a continuation of the single-player
GridWorld example above:

```python
batch = two_player_engine.collect(
    n_records=RECORDS_PER_UPDATE,
    infer=[blue_infer, red_infer],
)
```

The callback sequence length must match the game player count. Use `batch.players` to route records
to the corresponding optimizer or replay buffer. Set `learn_players=[0]` on `Engine` for a frozen
opponent experiment: all players still act, but only player 0 emits training rows.

## Next steps

- Add structured metrics with [telemetry and TensorBoard](telemetry.md).
- Overlap collection and optimization with [concurrent collection](streaming.md).
- Compare trained agents with the [evaluation guide](evaluation.md).
- Choose a loss from the learner-specific [batch formats](../reference/batch-formats.md) and browse
  the maintained [examples](../examples/index.md).
