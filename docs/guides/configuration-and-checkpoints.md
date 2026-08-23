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
hashing. Changing a policy setting such as epsilon produces a different composition rather than an
in-place schedule; see [policy and learner knobs](../reference/python-api.md#policy-and-learner-knobs)
for the snapshot consequence.

## Reached-state starts

The optional start buffer broadens training coverage by saving non-terminal states reached during
collection and starting some later episodes from them. It is currently available for Snake:

```python
engine = rf.Engine(
    game=rf.games.Snake(grid_size=12, max_ticks=500),
    reward=rf.Reward(food=1.0, loss=-1.0),
    policy=rf.policies.EpsilonGreedyQ(n_heads=1, epsilon=0.1),
    learner=rf.learners.Dqn(),
    n_games=16,
    start_buffer=True,
    start_buffer_capacity=1_000,
    p_fresh=0.05,
)
```

Snake groups reached states into difficulty cells based on snake length. The buffer keeps a bounded
reservoir of `start_buffer_capacity` states per cell. For a buffer-backed reset, it samples exactly
uniformly among occupied cells, then uniformly among the retained states in that cell. `p_fresh` is
the fraction of resets that still use the game's normal initial state. While the buffer is empty,
all starts are fresh. Snake only buffers states while every snake remains alive; once any snake has
died, subsequent states from that episode are skipped. The buffer therefore broadens coverage over
full-game states, not post-elimination continuations.

The feature changes the distribution of training states, not the transition targets. Episode
telemetry reports `(returns, length, seeded)`; `seeded` is true when that episode began from the
buffer. The buffer and its RNG are included in engine snapshots and resolved configuration.

## Inference cache lifecycle

Set `infer_cache` to the entry capacity of each player's persistent evaluation-cache partition; zero
(the default) disables it:

```python
engine = rf.Engine(
    game=game,
    reward=reward,
    policy=policy,
    learner=learner,
    n_games=16,
    infer_cache=100_000,
)
```

Cached outputs are correct only while their network weights remain unchanged. Transposition-rich
workloads such as Chess are natural candidates, but measure `cache_hits` rather than assuming a
benefit. Invalidate after an optimizer step or collector-weight synchronization:

```python
engine.weights_updated()          # Shared callback, or every per-player partition.
engine.weights_updated(player=1)  # Per-player callbacks: only player 1 changed.
```

Use `player=` only with a per-player callback sequence. A shared callable uses the shared partition,
so player-specific invalidation can leave its rows stale. `weights_updated` is thread-safe during
`collect_stream`, takes effect at the next safe round boundary, and is a no-op when caching is off.
Monitor `cache_lookups` and `cache_hits` in
[telemetry](../reference/telemetry-fields.md). Cache contents are excluded from snapshots, so a
restore can change callback patterns without changing records.

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

Save the model and optimizer alongside the snapshot. Set `policy_version` to the identifier of that
exact model file—such as its checkpoint name, content hash, or artifact version—and pass the same
identifier as `expect_policy_version` when restoring. The engine compares the strings; it cannot
inspect caller-owned model weights. For games with bit-transparent state codecs (every built-in
except CarRacing), restored collection is record-exact for the same inference outputs, although
cache exclusion means the callback call pattern may differ. CarRacing restores by deterministic
reconstruction instead — resumed collection is reproducible from the snapshot but not
record-identical to the uninterrupted run; see the
[CarRacing notes](../catalogue/games.md#carracing-notes).

For an active stream, use the [lossless pause barrier](streaming.md#checkpointing-without-losing-records)
before taking the snapshot. `pause()` terminates that stream; after saving or restoring, start a new
`collect_stream` to resume concurrent collection.

## Resume after a crash

A process crash cannot preserve batches queued or in flight inside a collection stream. Choose a
checkpoint cadence from the amount of completed training and collection work you can afford to
repeat, and use the pause barrier for each graceful checkpoint.

Persist one matched checkpoint set:

- the model weights and optimizer state;
- the engine snapshot bytes and their exact `policy_version`;
- the resolved configuration and configuration fingerprint;
- caller-owned replay buffers, schedulers, counters, and any other experiment state.

On restart, reconstruct the Engine and let `restore` validate its internal composition fingerprint
and the caller-owned model version:

```python
import reinfors as rf

snapshot = rf.EngineSnapshot.from_bytes(snapshot_payload)
engine = rf.engine_from_config(saved_config)
engine.restore(snapshot, expect_policy_version=model_version)
```

Do not compare `engine.config_fingerprint()` with `snapshot.fingerprint`: they intentionally hash
different representations. `restore` performs the correct internal composition check and raises
`snapshot is from a different composition` when they do not match.

Load the exact model, optimizer, replay, and scheduler state associated with `model_version`, then
create a new stream. The engine snapshot cannot recover caller-owned state, and the model checkpoint
cannot recover native episode/search state.

Publish each checkpoint as a matched set: write a versioned staging directory on the same
filesystem, atomically rename it with [`os.replace`](https://docs.python.org/3/library/os.html#os.replace),
then update a `LATEST` pointer last. After a crash, ignore incomplete staging directories and resume
only the published version.

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

## Next steps

- Use the [stream pause barrier](streaming.md#checkpointing-without-losing-records) before snapshotting
  concurrent collection.
- Record collection and episode state with [telemetry](telemetry.md).
- Keep model identifiers aligned while running the [evaluation workflow](evaluation.md).
