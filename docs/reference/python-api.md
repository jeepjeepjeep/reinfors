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
| `Env` | Caller-driven single-game play |
| `Reward` | Named event-weight configuration |

The authoritative signatures, defaults, array types, and method-level contract comments are
in `python/reinfors/_reinfors.pyi`, shipped with the package and understood by editors. The
machine-readable component inventory is `reinfors.catalog`; generated coverage tables are
the friendlier browsing surface.

## Config-driven construction

Each component module supports stable registry names:

```python
import reinfors as rf

print(rf.games.registered())
game = rf.games.make("snake", grid_size=12, num_snakes=4)
policy = rf.policies.make("mcts", num_simulations=128)
```

Use typed constructors in ordinary code and registries for experiment configuration. Use
`engine.resolved_config()` rather than recording only the inputs you happened to specify.
