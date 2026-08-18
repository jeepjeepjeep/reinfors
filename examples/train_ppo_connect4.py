"""Train PPO self-play on Connect 4."""

import numpy as np
import reinfors as rf
import torch
from torch import nn
from torch.nn import functional as F

UPDATES = 20
RECORDS_PER_UPDATE = 512
EPOCHS = 4
MINIBATCH = 128
CLIP = 0.2
VALUE_COEF = 0.5
ENTROPY_COEF = 0.01

torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

game = rf.games.Connect4()
obs_size = int(np.prod(game.observation_space().shape))
n_actions = game.action_space().n
engine = rf.Engine(
    game=game,
    reward=rf.Reward(win=1.0, loss=-1.0),
    policy=rf.policies.Ppo(),  # Samples the masked softmax; exploration comes from stochasticity.
    learner=rf.learners.Ppo(gamma=1.0, lam=0.95),  # GAE over each player's own decisions.
    n_games=16,
    seed=0,
)

trunk = nn.Sequential(nn.Linear(obs_size, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU())
policy_head = nn.Linear(128, n_actions)
value_head = nn.Linear(128, 1)
net = nn.ModuleDict({"trunk": trunk, "policy": policy_head, "value": value_head}).to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=3e-4)


def forward(obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    h = net["trunk"](obs)
    return net["policy"](h), net["value"](h).squeeze(-1)


def infer(obs_batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return policy logits and a value for each observation."""
    net.eval()
    with torch.no_grad():
        obs = torch.from_numpy(np.ascontiguousarray(obs_batch)).to(device)
        logits, values = forward(obs)
        return logits.cpu().numpy(), values.cpu().numpy()


def dense_legal_mask(batch: rf.PpoBatch) -> np.ndarray:
    counts = np.diff(batch.legal_offsets)
    rows = np.repeat(np.arange(len(batch.obs)), counts)
    mask = np.zeros((len(batch.obs), n_actions), dtype=bool)
    mask[rows, batch.legal_ids] = True
    return mask


for update in range(1, UPDATES + 1):
    batch = engine.collect(n_records=RECORDS_PER_UPDATE, infer=infer)

    obs = torch.as_tensor(batch.obs, device=device)
    actions = torch.as_tensor(batch.actions, device=device)
    behavior_logp = torch.as_tensor(batch.behavior_log_probs, dtype=torch.float32, device=device)
    returns = torch.as_tensor(batch.returns, dtype=torch.float32, device=device)
    values_old = torch.as_tensor(batch.values, dtype=torch.float32, device=device)
    legal = torch.as_tensor(dense_legal_mask(batch), device=device)
    advantages = torch.as_tensor(batch.advantages, dtype=torch.float32, device=device)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    net.train()
    m = len(obs)
    for _ in range(EPOCHS):
        perm = torch.randperm(m, device=device)
        for start in range(0, m, MINIBATCH):
            idx = perm[start : start + MINIBATCH]
            logits, values = forward(obs[idx])
            # Mask with the SAME legality the actor sampled under, or the ratio is wrong.
            logits = logits.masked_fill(~legal[idx], torch.finfo(logits.dtype).min)
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
    p0 = [r[0] for r, _len, _seeded in episodes]
    mean_p0 = float(np.mean(p0)) if p0 else float("nan")
    print(
        f"update={update} records={m} policy_loss={policy_loss.item():.3f} "
        f"value_loss={value_loss.item():.3f} p0_return={mean_p0:+.2f}"
    )
