# Gymnasium and PettingZoo adapters

Use the ecosystem adapters when an existing RL library or evaluation tool expects Gymnasium or
PettingZoo rather than reinfors' native `Env` surface.

The `gym` extra installs both Gymnasium and PettingZoo:

```bash
pip install "reinfors[gym]"
```

## Select the matching API

`reinfors.gym.make` inspects the game and chooses the standard interface that matches its decision
model:

| Game interaction | Returned interface | Built-in example |
| --- | --- | --- |
| Single-agent | `gymnasium.Env` | GridWorld |
| Simultaneous multi-agent | PettingZoo `ParallelEnv` | Snake |
| Sequential multi-agent | PettingZoo `AECEnv` | Connect 4 |

```python
import numpy as np
import reinfors as rf
from reinfors import gym

env = gym.make(
    game=rf.games.GridWorld(size=8),
    reward=rf.Reward(goal=1.0, step=-0.01),
    seed=0,
)
observation, info = env.reset(seed=0)
```

The [game catalogue](../catalogue/games.md) identifies the adapter expected for every built-in game.
Use `gym.gymnasium_env`, `gym.parallel_env`, or `gym.aec_env` directly when the interaction model is
already known.

`gym.make` takes an existing game handle. It is different from `rf.games.make("snake", ...)`, which
constructs a game from its registry name.

## PettingZoo AEC loop

Sequential games expose one selected agent at a time. The legal-action mask is part of that agent's
observation:

```python
env = gym.make(
    game=rf.games.Connect4(),
    reward=rf.Reward(win=1.0, loss=-1.0, draw=0.0),
)
env.reset(seed=0)

for agent in env.agent_iter():
    observation, reward, terminated, truncated, info = env.last()
    action = None
    mask = observation["action_mask"]
    if not (terminated or truncated) and mask.any():
        action = int(np.flatnonzero(mask)[0])
    env.step(action)
```

## PettingZoo Parallel loop

Simultaneous games receive one action for every current agent. Their observations remain plain
arrays, while masks live in the `infos` mapping:

```python
env = gym.make(
    game=rf.games.Snake(grid_size=10),
    reward=rf.Reward(food=1.0, loss=-1.0),
)
observations, infos = env.reset(seed=0)

while env.agents:
    actions = {
        agent: int(np.flatnonzero(infos[agent]["action_mask"])[0])
        for agent in env.agents
    }
    observations, rewards, terminations, truncations, infos = env.step(actions)
```

## Rewards and episode limits

Pass `reward=` to the adapter because the standard `step` APIs must return scalar rewards. The
game's named Rust events are weighted inside the native `Env`, and the adapter returns those values
from `env.rewards`. Omitting `reward` uses the game's defaults; check the
[game reward table](../catalogue/games.md) before relying on them.

`max_episode_steps` overrides the adapter time limit. When it is omitted, the adapter uses the
game's `truncation_horizon()`; when that is `None`, no time-limit truncation is applied.

## Legal-action masks

The multi-agent mask location follows each ecosystem's convention:

- PettingZoo AEC observations are dictionaries containing `observation` and `action_mask`.
- PettingZoo Parallel observations remain plain arrays; each mask is in
  `infos[agent]["action_mask"]`.

Masks use `int8`, contain one entry per discrete action, and mark legal actions with `1`. AEC
observations for agents that cannot currently move carry an all-zero mask. Sample or select only
from marked actions; the underlying `Env` rejects illegal actions. The single-agent Gymnasium
adapter currently returns plain observations and empty info dictionaries rather than an action
mask.

## Native `Env` or an adapter?

Use `rf.Env` for direct reinfors evaluation, snapshots and forks, explicit event traces, and
game-action control. Use these adapters when interoperability with Gymnasium or PettingZoo tooling
is the priority. Training through `Engine` remains the batched search-and-sampling path and does not
go through an adapter.

The adapter implementations are checked with Gymnasium's environment checker and PettingZoo's
official AEC and Parallel compliance tests.

## Next steps

- Drive a game directly with the [evaluation guide](evaluation.md).
- Choose a compatible built-in from the [game catalogue](../catalogue/games.md).
- Review reinfors' [fixed action-space boundary](../reference/limits.md#fixed-observation-and-action-spaces).
