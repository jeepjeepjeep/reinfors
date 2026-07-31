# Build a training loop

This example trains a small DQN on GridWorld. It is a complete, single-file program: install the
training dependencies, copy the code into a file, and run it.

```bash
pip install "reinfors[train]"
python train_gridworld.py
```

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
game = rf.games.GridWorld(size=5, goal_row=0, goal_col=4, max_ticks=50)
obs_size = int(np.prod(game.observation_space().shape))
n_actions = game.action_space().n
engine = rf.Engine(
    game=game,
    reward=rf.Reward(step=-0.01, goal=1.0),
    policy=rf.policies.EpsilonGreedyQ(n_heads=1, epsilon=0.1),
    learner=rf.learners.Dqn(),
    n_games=8,
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
    """Return one head of Q-values for each observation."""
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
    can_bootstrap = torch.as_tensor(np.diff(batch.next_legal_offsets) > 0, device=device)

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

    print(f"update={update} records={len(batch.obs)} loss={loss.item():.3f}")
```

## What reinfors provides

The engine runs GridWorld episodes concurrently using the epsilon-greedy policy, batches calls to
`infer`, and returns the transitions needed for the DQN update. The network, target network,
optimizer, and loss remain ordinary PyTorch and can be replaced without changing the engine.

The inference callback is the adapter between the two sides. Here the network produces one Q-value
per action, and `unsqueeze(1)` adds the single ensemble-head dimension expected by
`EpsilonGreedyQ(n_heads=1)`. The callback returns NumPy `float64` values; action legality stays
inside the game and policy.

Every GridWorld move is legal, so the target network can take a dense maximum over its four outputs.
The batch's sparse next-action offsets still identify rows that should not bootstrap, such as
terminal states. Games with variable legal actions must instead maximize over their provided legal
action IDs; `examples/train_dqn_holdem.py` demonstrates that general case.

`engine.collect()` returns at least `n_records`, rather than exactly that many records, because
complete episodes are retained. Its `batch.telemetry` field contains collection timings, inference
counts, and episode summaries; see [telemetry](telemetry.md) for reporting and TensorBoard output.

## Taking the loop further

This first example deliberately trains directly on each collected batch. Typical experiments add a
replay buffer, minibatching, checkpoints, evaluation, and concurrent collection. The maintained
scripts build those pieces on the same interface:

- `examples/train_dqn_holdem.py`: replay, minibatching, sparse legality, evaluation, and ensemble DQN;
- `examples/train_alphazero_example.py`: AlphaZero search with policy and value heads;
- `examples/train_example.py`: TreeStrap with selective expectimax or UCT MCTS;
- `examples/train_deep_cfr.py`: caller-owned Deep CFR buffers and networks.

Use `engine.resolved_config()` alongside checkpoints to record all constructor defaults. If you
enable `infer_cache`, call `engine.weights_updated()` after an optimizer step so cached outputs from
the previous weights are invalidated.

## Per-player models

Pass one callback per player when policies differ:

```python
batch = engine.collect(n_records=records_per_update, infer=[blue_infer, red_infer])
```

The callback sequence length must match the game player count. Use `batch.players` to route records
to the corresponding optimizer or replay buffer. Set `learn_players=[0]` on `Engine` for a frozen
opponent experiment: all players still act, but only player 0 emits training rows.

See [examples](../examples/index.md) for runnable commands and [batch formats](../reference/batch-formats.md)
for the fields produced by every learner.
