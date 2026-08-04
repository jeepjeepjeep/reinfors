# Troubleshooting

Reinfors validates the native/Python boundary strictly so malformed network outputs fail near the
callback rather than corrupting search. Match a traceback fragment here first:

| Exception contains | Likely cause | Go to |
| --- | --- | --- |
| `unknown reward key` | The game does not emit that named reward event | [Construction is rejected](#construction-is-rejected) |
| `incompatible policy/learner composition` | The policy and learner cannot form an Engine | [Construction is rejected](#construction-is-rejected) |
| `not compatible with`, `would be clairvoyant`, or `supports at most` | The algorithm does not support that game's semantics | [Construction is rejected](#construction-is-rejected) |
| `win_food_lead is a two-snake rule` | `win_food_lead` was combined with `num_snakes != 2` | [Construction is rejected](#construction-is-rejected) |
| `expected … per-player infer callables` | The callback sequence does not match the player count | [Per-player callback count](#per-player-callback-count) |
| `infer output must be a float64 NumPy array` or `AlphaZero infer output must be` | Wrong callback dtype or rank | [Callback dtype or shape errors](#callback-dtype-or-shape-errors) |
| `infer returned shape` or `AlphaZero infer must return` | Wrong callback rows, heads, or action count | [Wrong row or action count](#wrong-row-or-action-count) |
| `outputs must contain only finite values` | The callback returned NaN or infinity | [NaN or infinity from inference](#nan-or-infinity-from-inference) |
| `this engine is held by a collect_stream` | A stream still owns the engine | [The engine is held by a stream](#the-engine-is-held-by-a-stream) |
| `snapshot is from a different composition` | Snapshot and Engine fingerprints differ | [Snapshot restoration is rejected](#snapshot-restoration-is-rejected) |
| `policy_version mismatch` | The loaded model is not the snapshot's model | [Snapshot restoration is rejected](#snapshot-restoration-is-rejected) |

## Construction is rejected

`unknown reward key` names the rejected key and the valid keys for that game. Choose event names
from the [game catalogue](../catalogue/games.md); unknown names are never ignored.

`incompatible policy/learner composition` means the two handles do not share a training contract.
Use an exact pairing from the [algorithm catalogue](../catalogue/algorithms.md) and confirm the game
supports it in the [compatibility matrix](../catalogue/compatibility.md).

Errors containing `not compatible with`, `would be clairvoyant`, or `supports at most` reject a
game/algorithm pairing because of information, player-count, or dynamics requirements. Choose a
supported row from the same compatibility matrix rather than bypassing the constructor check.

`win_food_lead` is defined only for exactly two snakes. Leave it as `None` when `num_snakes` is
greater than two; the constructor rejects the combination rather than inventing a multi-player
interpretation of the lead.

## Per-player callback count

A bare callable is shared by every player. A sequence selects per-player inference instead and must
contain exactly one callable per game player, in player order. This rule applies to Engine collection
and Deep CFR; a short sequence never silently reuses its final callback.

## Callback dtype or shape errors

Engine observations arrive as a two-dimensional `float32` NumPy array. Engine outputs must be
`float64`; Q-family policies require `(rows, heads, actions)`, while AlphaZero requires a tuple of
logits `(rows, actions)` and values `(rows,)`. Deep CFR instead returns `(rows, actions)`.

Framework tensors must be detached, moved to CPU, and converted to NumPy before returning. Do not
remove a singleton Q head: `(rows, 1, actions)` is distinct from `(rows, actions)`. The complete
shapes are in the [inference contract](inference-contract.md). Dtype/rank errors report the returned
object's observed Python type, dtype, and shape.

## Wrong row or action count

Return exactly one output row for every input row, even when a row has no actions you wish to train.
The action dimension is the game's fixed action vocabulary, not the number of currently legal
actions. Legality is applied around the network.

## NaN or infinity from inference

Callbacks must return finite values. Reinfors rejects NaN or infinity before those values enter
search. Check normalization, empty reductions, masked logits, and division by zero in the caller's
forward path.

## The engine is held by a stream

An active `CollectStream` exclusively borrows its engine. Calling `engine.collect`, starting another
stream, restoring, or snapshotting during that borrow raises `this engine is held by a
collect_stream`. Use a context manager and call `pause()` for a lossless checkpoint barrier or
`stop()` when queued work may be discarded. A paused stream is finished; create a new stream to
continue. See [concurrent collection](../guides/streaming.md).

## Snapshot restoration is rejected

`snapshot is from a different composition` means `restore` found that the snapshot's internal
composition fingerprint does not match the destination Engine. Reconstruct the Engine from the
resolved configuration saved with that snapshot. Do not compare `engine.config_fingerprint()` with
`snapshot.fingerprint`; those public values intentionally hash different representations.

`policy_version mismatch` means `expect_policy_version` does not name the model recorded in the
snapshot. Load that exact model checkpoint and pass its identifier. Do not remove the check merely
to make restoration proceed: native search state and external weights would then describe different
points in training. See the [crash-resume checklist](../guides/configuration-and-checkpoints.md#resume-after-a-crash).

## Training runs but the agent plays badly

This class of failure may not raise an exception:

- With a transforming encoder, convert between the network-head and game-action frames. A missed
  conversion can select legal but wrong actions; see [action frames](glossary.md#action-frames).
- Confirm reward keys and weights against the [game catalogue](../catalogue/games.md). A valid
  all-zero mapping produces no learning signal.
- For DQN, use `batch.can_bootstrap` and reduce losses over included bootstrap-mask entries.
- `mean_sigma` and `mean_disagreement` are always `0.0` at `n_heads=1`; they measure ensemble
  variation only at `n_heads >= 2`.

## Changes are missing in a source checkout

Python binding edits and native Rust changes do not affect an already-built extension. From the
repository environment, rebuild with:

```bash
maturin develop
```

Then rerun the smallest relevant test before the full suite. Pure Markdown and Python-only changes
do not require rebuilding Rust. Use `maturin develop --release` only for performance measurements;
the default debug build gives the faster development cycle described in
[development setup](../development/setup.md).

## Next steps

- Verify exact fields in [batch formats](batch-formats.md).
- Interpret throughput and search counters with [telemetry fields](telemetry-fields.md).
- Report a reproducible defect with the resolved configuration, build profile, seed, and complete
  exception traceback.
