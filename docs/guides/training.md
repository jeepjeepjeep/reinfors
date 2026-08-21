# Build a training loop

This example trains a small DQN on GridWorld. Install the training dependencies first:

```bash
pip install "reinfors[train]"
```

With the published package, save the program below as `train_gridworld.py` and run
`python train_gridworld.py`. From a source checkout, the identical maintained copy is already
available:

```bash
python examples/train_gridworld.py
```

The program collects ten batches, applies one DQN update per batch, and prints the record count,
loss, and mean completed-episode return. It normally finishes in seconds rather than minutes on a
laptop CPU. Expect noisy loss and return values, not monotonic curves: this short run verifies the
complete data and optimization path rather than training a converged agent.

The [glossary](../reference/glossary.md) defines records, ensemble heads, truncation tails, and the
other reinfors-specific terms used below. The five-step
[collection round](../concepts/sampling-and-training.md#a-collection-round) explains how parallel
games and inference pooling produce this batch.

```python
"""Train a small DQN on GridWorld."""

import copy

import numpy as np
import reinfors as rf
import torch
from torch import nn
from torch.nn import functional as F

UPDATES = 10
RECORDS_PER_UPDATE = 256
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
    learner=rf.learners.Dqn(gamma=0.99),  # Emit transition records; gamma lives engine-side.
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
    discounts = torch.as_tensor(batch.discounts, dtype=torch.float32, device=device)

    # Compute DQN targets. `discounts` is the engine's gamma^k per record (0 at terminals),
    # so the same line is correct for 1-step and n-step collection.
    net.train()
    chosen_q = net(obs).gather(1, actions[:, None]).squeeze(1)
    with torch.no_grad():
        next_value = target_net(next_obs).max(dim=1).values
        targets = rewards + discounts * torch.where(can_bootstrap, next_value, 0.0)

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
```

## What reinfors provides

The engine runs GridWorld episodes concurrently using the epsilon-greedy policy, batches calls to
`infer`, and returns the transitions needed for the DQN update. The network, target network,
optimizer, and loss remain ordinary PyTorch and can be replaced without changing the engine.
The configured epsilon is fixed for this engine; see
[policy and learner knobs](../reference/python-api.md#policy-and-learner-knobs) before adding a
schedule.

The inference callback is the adapter between the two sides. Here the network produces one Q-value
per action, and `unsqueeze(1)` adds the head axis expected by `EpsilonGreedyQ(n_heads=1)`. The
head axis is an ensemble dimension rather than a distinct policy/value model output. The callback
returns the network's native NumPy `float32` values, which reinfors widens exactly; action legality
stays inside the game and policy. Use the
[troubleshooting table](../reference/troubleshooting.md) when a callback or composition is rejected.

Every GridWorld move is legal, so the target network can take a dense maximum over its four outputs.
`batch.can_bootstrap` is false when the TD target should contain only the record's reward sum. Games
with variable legal actions must also maximize over their provided legal action IDs. The
[Hold'em DQN example](../examples/index.md#dqn-holdem) demonstrates that general case.

`batch.telemetry` contains collection timings, inference counts, and episode summaries; see
[telemetry](telemetry.md) for reporting and TensorBoard output.

## Taking the loop further

This first example deliberately trains directly on each collected batch. Typical experiments add a
replay buffer, minibatching, checkpoints, evaluation, and concurrent collection. The maintained
scripts build those pieces on the same interface:

- [Hold'em DQN](../examples/index.md#dqn-holdem): replay, minibatching, sparse legality,
  evaluation, and ensemble DQN;
- [AlphaZero](../examples/index.md#alphazero-connect-4): search with policy and value heads;
- [TreeStrap](../examples/index.md#treestrap-snake): selective expectimax or UCT MCTS;
- [Deep CFR](../examples/index.md#deep-cfr-training): caller-owned buffers and networks.

Use `engine.resolved_config()` alongside checkpoints to record all constructor defaults; the
[configuration and checkpoints guide](configuration-and-checkpoints.md) shows the complete restore
workflow. If the experiment enables inference caching, follow the
[cache lifecycle guide](configuration-and-checkpoints.md#inference-cache-lifecycle).

## Scheduler knobs (`batch_size`, `n_threads`)

The collect loop is a threshold scheduler: worker threads run search rounds and feed
inference requests into a shared queue, and the callback fires on the collecting thread
the moment `batch_size` rows are queued (default `max(1, n_games/2)`) — or earlier when
no search can progress without results — so tree work overlaps inference. Raise
`batch_size` toward your accelerator's sweet spot for larger, fewer calls at the cost of
some latency; `telemetry["infer_rows"] / telemetry["infer_calls"]` reports the realized
mean. `n_threads` sets the worker count. Both are part of the resolved configuration and
`config_fingerprint()` — changing them changes which games advance before the floor and
therefore the returned window's composition — but they are excluded from the snapshot
compatibility fingerprint, so checkpoints restore across different values. Collection is
exactly reproducible at `n_threads=1` with a deterministic callback; at `n_threads>1`,
task completion order varies between runs — a valid collection either way.

## Overlapping search and inference (`n_groups`)

By default the engine's collect loop alternates between tree work on the CPU and your `infer`
callback: while the accelerator runs a batch, the engine waits, and vice versa. With
`n_groups=2` the games split into two fixed groups, each collecting on its own worker
thread with inference forwarded to a service thread that owns the callback — one group's
tree work runs while the other group's batch is inside the callback.

With per-group search time `S` and inference time `I`, steady-state throughput improves from
one group-round per `S + I` to one per `max(S, I)` — a gain of `(S + I) / max(S, I)`, largest
when the two stages are *balanced* (up to 2x at `S ≈ I`) and small when either stage
dominates. Per-game decision latency moves from `S + I` to roughly `2 · max(S, I)`: similar
when balanced, approaching 2x when one stage dominates. Measure your own split before
reaching for this knob: `telemetry["infer_seconds"]` against wall time gives `I`'s share.

Sizing: keep each *group's* callback batches near your accelerator's sweet spot, which
usually means doubling `n_games` rather than splitting it. Note that games-per-group only
approximates rows-per-callback — simultaneous and MaxN searches evaluate multiple
perspectives per node, exhaustive chance fans stage every outcome, and cache hits, in-batch
deduplication, and terminal simulations all remove rows. Check the realized mean with
`telemetry["infer_rows"] / telemetry["infer_calls"]` rather than assuming it (with
`pad_rows_to`, add `padded_rows` to the numerator for the physical size); a
`n_games=128, n_groups=2` starting point for a sequential game at a batch-64 sweet spot is a
workload-specific example, not a rule. If the callback needs a *constant* row count
(compiled/graph-captured forwards), `pad_rows_to` in the
[inference contract](../reference/inference-contract.md#input) fixes every call at exactly
that row count — zero-padding short calls, chunking oversized ones — at the cost of
discarded pad-row compute.

Grouped collects are run-to-run nondeterministic while shared state is live — the shared
inference cache, the start buffer, and weight refreshes — which is the same status real
accelerator training already has. With deterministic inference and none of those in play
(e.g. cacheless test configurations) they are exact: per-group record floors and
persistent per-group rng streams (carried in snapshots) make results independent of
scheduling. Splitting one collect into several remains observable — record floors reset
per call. Reproduce anomalies with `n_groups=1`, which stays
exactly deterministic. Digests differ from `n_groups=1` — it is a different composition,
and `resolved_config()`/`config_fingerprint()` record it.
On a weight refresh (`weights_updated()`), rows already in flight finish their round. Once
a round observes the new generation (each round syncs at its boundary, before any lookup),
older entries are cleared and no longer served; a refresh landing *mid-round* takes effect
at the next boundary, so that round's remaining lookups may still see pre-refresh entries —
the same one-round staleness window the ungrouped collect has.

Grouping is policy- and learner-agnostic: any composition collects grouped, including
truncation-tail bootstrapping (tail forwards are ordinary inference requests). The one
remaining restriction — a single shared callback, not per-player routing — is checked when
a collect or stream begins, since the callback shape is only known then.

## Compiling the inference callback

The PyTorch forward inside the callback can be compiled normally; no reinfors API
changes are required. The example above builds
`forward = torch.compile(net) if device.type == "cuda" else net` and calls it inside
`infer`. Default mode over the engine's natural, varying batch sizes is the pattern the
V1 benchmark favored —
[+19.6% completed training states/s](https://github.com/jeepjeepjeep/reinfors-benchmarks/blob/main/docs/configuring-the-engines.md#reinfors-throughput-levers)
at its operating point, measured on one chess ResNet workload on an A10G; measure it on
your own workload. Graph-capture modes recapture or recompile per batch shape; a
constant row count avoids that, which is what `pad_rows_to` in the
[inference contract](../reference/inference-contract.md#input) provides. The first
calls pay compilation latency, so short CPU example runs skip it.

## Per-player models

For a separate two-player experiment—for example, Connect 4 with a frozen opponent—pass one
callback per player to that game's Engine. This is not a continuation of the single-player
GridWorld example above:

```python
batch = two_player_engine.collect(
    n_records=RECORDS_PER_UPDATE,
    infer=[blue_infer, red_infer],
)
```

The callback sequence length must match the game player count. Use `batch.players` to route records
to the corresponding optimizer or replay buffer. Set `learn_players=[0]` on `Engine` for a frozen
opponent experiment: all players still act, but only player 0 emits training rows.

## Next steps

- Add structured metrics with [telemetry and TensorBoard](telemetry.md).
- Overlap collection and optimization with [concurrent collection](streaming.md).
- Compare trained agents with the [evaluation guide](evaluation.md).
- Choose a loss from the learner-specific [batch formats](../reference/batch-formats.md) and browse
  the maintained [examples](../examples/index.md).
