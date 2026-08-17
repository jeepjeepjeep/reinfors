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
# Default-mode compile is the pattern the V1 benchmark favored; measure it on your workload.
forward = torch.compile(net) if device.type == "cuda" else net


def infer(obs_batch: np.ndarray) -> np.ndarray:
    """Return one ensemble head's Q-values for each observation."""
    net.eval()
    with torch.no_grad():
        obs = torch.from_numpy(np.ascontiguousarray(obs_batch)).to(device)
        return forward(obs).unsqueeze(1).cpu().numpy()


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
