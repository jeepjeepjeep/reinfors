# Configuration and checkpoints

Reinfors separates the experiment recipe from its live state.

## Resolved configuration

`resolved_config()` returns a JSON-compatible description with constructor defaults filled
in. Persist it beside every run:

```python
import json

config = engine.resolved_config()
with open("engine-config.json", "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2, sort_keys=True)
```

`rf.engine_from_config(config)` reconstructs the composition. `config_fingerprint()` is a
canonical 128-bit identifier for comparison; treat it as opaque and do not reimplement its
hashing.

## Engine snapshots

An engine snapshot captures active episodes, random-number generators, policy state, partial
trajectories, the reached-state start buffer, and the weight generation. It deliberately
excludes network parameters and the inference cache.

```python
snapshot = engine.snapshot(policy_version="checkpoint-0042")
payload = snapshot.to_bytes()

restored = rf.EngineSnapshot.from_bytes(payload)
engine.restore(restored, expect_policy_version="checkpoint-0042")
```

Save the model and optimizer alongside the snapshot. The optional policy version detects a
mismatched model checkpoint. Restored collection is record-exact for the same inference
outputs, although cache exclusion means the callback call pattern may differ.

For an active stream, use the [lossless pause barrier](streaming.md#checkpointing-without-losing-records)
before taking the snapshot.

## Environment snapshots and forks

`Env.snapshot()` captures the native game state, chance RNG, and terminal status. Restore
rejects incompatible compositions, unsupported schemas, and malformed bytes.

```python
state = env.snapshot()
branch_a = env.fork()          # same future chance stream
branch_b = env.fork(seed=123)  # divergent future chance stream
env.restore(state)
```

Snapshots are opaque continuation artifacts, not a long-term game-state interchange format.
For human-readable inspection use `env.state()`; do not build a restore path from that
dictionary.
