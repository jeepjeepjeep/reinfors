# Python API

Reinfors keeps the public composition surface small and typed.

| Module or type | Purpose |
| --- | --- |
| `reinfors.games` | Typed and name-addressable game constructors |
| `reinfors.encoders` | Alternative observation/action views |
| `reinfors.policies` | Search and action-selection handles |
| `reinfors.learners` | Training-record producers |
| `reinfors.chance_modes` | Search treatment of explicit chance nodes |
| `reinfors.noise` | Root exploration-noise handles |
| `reinfors.solvers` | CFR and Deep CFR constructors |
| `reinfors.spaces` | `Box` and `Discrete` descriptors |
| `reinfors.gym` | Optional Gymnasium and PettingZoo adapters |
| `Engine` | Parallel policy-driven collection |
| `CollectStream` | Background collection handle returned by `Engine.collect_stream` |
| `EngineSnapshot`, `EnvSnapshot` | Opaque continuation and environment snapshots |
| `Env` | Caller-driven single-game play |
| `Reward` | Named event-weight configuration |
| `TreeStrapBatch`, `DqnBatch`, `AlphaZeroBatch` | Engine learner-specific training batches |
| `DeepCfrBatch` | Training samples returned by `solvers.DeepCfr.collect` |
| `engine_from_config` | Reconstruct an engine from resolved configuration |
| `core_build_profile` | Report whether the native extension is a debug or release build |

The authoritative signatures, defaults, array types, and method-level contract comments are
in `python/reinfors/_reinfors.pyi`, shipped with the package and understood by editors. The
machine-readable component inventory is `reinfors.catalog`; generated coverage tables are
the friendlier browsing surface.

## Engine constructor

`Engine` is the composition and parallel-collection boundary:

| Parameter | Meaning |
| --- | --- |
| `game` | Native game handle, including any selected observation encoder. |
| `reward` | Named event weights. `None` uses that game's defaults. |
| `policy` | Acting or search policy; determines the inference-output contract. |
| `learner` | Record producer paired with the policy; determines the returned batch type. |
| `n_games` | Independent episode slots advanced in parallel. |
| `seed` | Root seed for reproducible engine, episode, search, and learner-mask RNG streams. |
| `start_buffer` | Enable reached-state starts; currently supported by Snake only. |
| `start_buffer_capacity` | Retained states per occupied start-buffer cell. |
| `p_fresh` | Fraction of resets that use the ordinary initial state while the buffer is populated. |
| `infer_cache` | Entries in each persistent evaluation-cache partition; zero disables caching. |
| `learn_players` | Players that emit records; omitted means every player. Other players still act. |

Collection uses `collect(n_records=..., infer=...)`; concurrent collection uses
`collect_stream(collect_size=..., infer=..., depth=...)`. Read the
[inference contract](inference-contract.md) before implementing the callback and the
[streaming guide](../guides/streaming.md) before sharing mutable model weights with a collector.

## Policy and learner knobs

Ensemble-head count and record membership deliberately live on different components. `n_heads` belongs
to `EpsilonGreedyQ` or `SelectiveExpectimax` because the policy consumes and acts from that many
estimators. `bootstrap_p` belongs to `Dqn` or `TreeStrap` because the learner independently includes
each record in each head's bootstrap sample; `1.0` produces all-one `batch.masks`.

`EpsilonGreedyQ(epsilon=...)` uses one fixed exploration probability for the lifetime of an engine;
there is no in-place scheduler. To change epsilon:

- construct a new engine with the new value;
- expect live episode and search state to start again; and
- do not restore an old engine snapshot, because the changed configuration has a different
  fingerprint and restoration rejects it.

Discount ownership also differs by learner family:

- DQN batches contain immediate rewards and next states; apply the chosen gamma in the caller's TD
  loss. `rf.learners.Dqn` therefore has no `gamma` argument.
- `rf.learners.TreeStrap(gamma=...)` discounts its generated search/return targets internally.
- `rf.learners.AlphaZero(gamma=...)` discounts its realized value targets internally; its default
  `1.0` matches the usual undiscounted game-outcome target.

## Config-driven construction

Games, policies, learners, encoders, chance modes, and noise expose stable registry names. For
example:

```python
import reinfors as rf

print(rf.games.registered())
game = rf.games.make("snake", grid_size=12, num_snakes=4)
policy = rf.policies.make("mcts", num_simulations=128)
```

Use typed constructors in ordinary code and registries for experiment configuration. Use
`engine.resolved_config()` rather than recording only the inputs you happened to specify.

`reinfors.solvers` instead exports the concrete `Cfr` and `DeepCfr` classes; it has no registry.
`reinfors.spaces` contains descriptor types, not constructors. `reinfors.gym.make(game, ...)` adapts
an existing game handle, so it should not be confused with `reinfors.games.make(name, ...)`.
