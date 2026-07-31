# Evaluation and interactive play

Use `Env` when the caller, rather than an engine policy, chooses actions. It is the small
surface for arenas, scripted opponents, debugging, and adapters.

```python
import random
import reinfors as rf

rng = random.Random(0)
env = rf.Env(rf.games.Connect4(), seed=0)

while not env.done():
    actions = {
        agent: rng.choice(env.legal_actions(agent))
        for agent in env.active_agents()
    }
    events = env.step(actions)
    if events:
        print(events)
```

Sequential games expose one active agent; simultaneous games can expose several. One call to
`step` is a tick and returns the ordered event trace across its action and any following
chance chain. When the environment has a reward mapping, `env.rewards` gives per-agent
rewards for the most recent tick.

## Fair comparisons

- alternate player seats or randomize them from a recorded seed;
- evaluate without self-play exploration noise unless noise is part of the policy being
  measured;
- report confidence intervals and draws, not only a mean score;
- record the resolved game configuration, model identifier, and seeds;
- use enough opening diversity to avoid replaying one deterministic trajectory.

`env.fork()` creates an independent environment at the same state. With no seed it has an
identical future chance stream; with a new seed it explores a different future. This is useful
for paired action comparisons and counterfactual evaluation.

## Standard adapters

Install `reinfors[gym]` and use `reinfors.gym` for the supported Gymnasium,
PettingZoo AEC, and PettingZoo Parallel surfaces. The [game catalogue](../catalogue/games.md)
lists the adapter matching each game. Adapters have official compliance tests in the test
suite.

## Imperfect-information metrics

Tabular CFR exposes average strategies, expected values, best-response values, NashConv, and
exploitability. Exact metrics enumerate the game tree and reject games beyond the safety cap.
For N-player games, NashConv is a distance-from-equilibrium diagnostic; the two-player
exploitability interpretation does not carry over unchanged.
