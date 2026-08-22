"""Train PPO on CarRacing pixels with a small CNN."""

import argparse

import numpy as np
import reinfors as rf
import torch
from torch import nn
from torch.nn import functional as F

parser = argparse.ArgumentParser()
parser.add_argument("--updates", type=int, default=200)
parser.add_argument("--records-per-update", type=int, default=1024)
parser.add_argument("--n-games", type=int, default=16)
parser.add_argument("--epochs", type=int, default=4)
parser.add_argument("--minibatch", type=int, default=128)
parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

CLIP = 0.2
VALUE_COEF = 0.5
ENTROPY_COEF = 0.01

torch.manual_seed(args.seed)
device = torch.device(args.device)

game = rf.games.CarRacing()
c, h, w = game.observation_space().shape
n_actions = game.action_space().n
engine = rf.Engine(
    game=game,
    reward=rf.Reward(tile=1000.0, step=-0.1, off_playfield=-100.0),
    policy=rf.policies.Ppo(),
    learner=rf.learners.Ppo(gamma=0.99, lam=0.95),
    n_games=args.n_games,
    seed=args.seed,
)


class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c, 16, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            flat = self.conv(torch.zeros(1, c, h, w)).shape[1]
        self.trunk = nn.Sequential(nn.Linear(flat, 256), nn.ReLU())
        self.policy = nn.Linear(256, n_actions)
        self.value = nn.Linear(256, 1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Raw 0-255 pixel values; normalization is this model's own choice.
        x = self.trunk(self.conv(obs / 255.0))
        return self.policy(x), self.value(x).squeeze(-1)


net = Net().to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=2.5e-4)


def infer(obs_batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    net.eval()
    with torch.no_grad():
        obs = torch.from_numpy(np.ascontiguousarray(obs_batch)).view(-1, c, h, w).to(device)
        logits, values = net(obs)
        return logits.cpu().numpy(), values.cpu().numpy()


for update in range(1, args.updates + 1):
    batch = engine.collect(n_records=args.records_per_update, infer=infer)

    obs = torch.as_tensor(batch.obs, device=device).view(-1, c, h, w)
    actions = torch.as_tensor(batch.actions, device=device)
    behavior_logp = torch.as_tensor(batch.behavior_log_probs, dtype=torch.float32, device=device)
    returns = torch.as_tensor(batch.returns, dtype=torch.float32, device=device)
    values_old = torch.as_tensor(batch.values, dtype=torch.float32, device=device)
    advantages = torch.as_tensor(batch.advantages, dtype=torch.float32, device=device)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    net.train()
    m = len(obs)
    for _ in range(args.epochs):
        perm = torch.randperm(m, device=device)
        for start in range(0, m, args.minibatch):
            idx = perm[start : start + args.minibatch]
            logits, values = net(obs[idx])
            # Every action is always legal in CarRacing, so no legality mask is needed.
            dist = torch.distributions.Categorical(logits=logits)
            logp = dist.log_prob(actions[idx])

            ratio = torch.exp(logp - behavior_logp[idx])
            clipped = torch.clamp(ratio, 1.0 - CLIP, 1.0 + CLIP)
            policy_loss = -torch.min(ratio * advantages[idx], clipped * advantages[idx]).mean()

            v_clip = values_old[idx] + torch.clamp(values - values_old[idx], -CLIP, CLIP)
            value_loss = torch.max(
                F.mse_loss(values, returns[idx], reduction="none"),
                F.mse_loss(v_clip, returns[idx], reduction="none"),
            ).mean()

            loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * dist.entropy().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    episodes = batch.telemetry["episodes"]
    rets = [r[0] for r, _len, _seeded in episodes]
    ret = f"{float(np.mean(rets)):+.1f}" if rets else "(no finished episodes)"
    print(
        f"update={update} records={m} policy_loss={policy_loss.item():.3f} "
        f"value_loss={value_loss.item():.3f} return={ret}"
    )
