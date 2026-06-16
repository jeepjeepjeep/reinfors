"""The end-to-end training layer: feed `Engine.collect` into a PyTorch optimisation loop.

Rust owns data generation (env dynamics, observations, the batched selective search); this module
owns the value network forward and its gradient step. They meet at the `infer` callback — the search
calls it once per pooled round, so each `collect` automatically searches with the *current* weights
(the actor-learner weight sync is implicit: there is one live network).

The network (`BootstrappedQNetwork`) is a faithful port of `snake_RL`'s ensemble Q-net — a shared
CNN trunk plus K randomized-prior heads — so reinfors can both train from scratch and load oracle
checkpoints. The loss (`treestrap_loss`) is the per-head masked Huber of its `BootstrappedTreeStrapTrainer`.

torch is an optional dependency (`pip install reinfors[train]`); importing this module requires it.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Optimizer

ObsShape = Sequence[int]  # (C, H, W)


class GridTrunk(nn.Module):
    """Shared CNN feature extractor: 2 strided convs + flatten + linear projection -> ReLU.

    Maps an (B, C, H, W) observation to a (B, feature_dim) vector. Matches `snake_RL`'s `GridTrunk`.
    """

    def __init__(self, obs_shape: ObsShape, feature_dim: int = 128) -> None:
        super().__init__()
        assert len(obs_shape) == 3, f"obs_shape must be (C, H, W), got: {obs_shape}"
        self.feature_dim = feature_dim
        self.conv = nn.Sequential(
            nn.Conv2d(obs_shape[0], 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        with torch.no_grad():
            conv_flat_dim = self.conv(torch.zeros(1, *obs_shape)).numel()
        self.proj = nn.Sequential(nn.Flatten(), nn.Linear(conv_flat_dim, feature_dim), nn.ReLU())

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.proj(self.conv(obs))


class PriorScaledHead(nn.Module):
    """Randomized-prior head (Osband 2018): `trainable(x) + prior_scale * prior(x).detach()`.

    The prior is frozen, so it is a permanent per-head offset that keeps heads disagreeing where data
    is thin — the epistemic-uncertainty signal the selective search expands on. Matches the oracle.
    """

    def __init__(self, in_features: int, out_features: int, prior_scale: float = 1.0) -> None:
        super().__init__()
        self.prior_scale = prior_scale
        self.trainable = nn.Linear(in_features, out_features)
        self.prior = nn.Linear(in_features, out_features)
        self.prior.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.trainable(x) + self.prior_scale * self.prior(x).detach()


class BootstrappedQNetwork(nn.Module):
    """Ensemble Q-network: shared `GridTrunk` + K `PriorScaledHead`s. forward(obs) -> (B, K, A).

    Heads are applied with one batched matmul (einsum over stacked head weights) rather than a Python
    loop — numerically identical to the per-head application, far fewer kernel launches. Matches
    `snake_RL`'s `BootstrappedQNetwork`, so its `state_dict` is interchangeable with the oracle's.
    """

    def __init__(self, obs_shape: ObsShape, n_actions: int, n_heads: int, prior_scale: float = 1.0) -> None:
        super().__init__()
        self.obs_shape = tuple(obs_shape)
        self.n_actions = n_actions
        self.n_heads = n_heads
        self.prior_scale = prior_scale
        self.trunk = GridTrunk(obs_shape)
        self.heads = nn.ModuleList(
            [PriorScaledHead(self.trunk.feature_dim, n_actions, prior_scale) for _ in range(n_heads)]
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.trunk(obs)  # (B, d)
        heads: list[PriorScaledHead] = list(self.heads)
        w_tr = torch.stack([h.trainable.weight for h in heads])
        b_tr = torch.stack([h.trainable.bias for h in heads])
        w_pr = torch.stack([h.prior.weight for h in heads])
        b_pr = torch.stack([h.prior.bias for h in heads])
        trainable = torch.einsum("bd,kad->bka", features, w_tr) + b_tr
        prior = (torch.einsum("bd,kad->bka", features, w_pr) + b_pr).detach()
        return trainable + self.prior_scale * prior  # (B, K, A)


def treestrap_loss(q: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per-head masked Huber, matching `snake_RL`'s `BootstrappedTreeStrapTrainer`: each head regresses
    its own searched targets, on the records its bootstrap mask selects, normalised by active head-records.

    `q`/`target` are (B, K, A); `mask` is (B, K). Returns a scalar loss.
    """
    huber = F.smooth_l1_loss(q, target, reduction="none")  # (B, K, A)
    return (mask.unsqueeze(-1) * huber).sum() / (mask.sum().clamp(min=1.0) * q.shape[-1])


def make_infer(net: BootstrappedQNetwork, device: str | torch.device = "cpu") -> Callable[[np.ndarray], np.ndarray]:
    """Build the `infer` callback the search calls: an (N, C*H*W) float32 batch -> (N, K, A) float64
    per-head action values, a no-grad forward of `net` on `device`. Closes over the live network, so
    successive searches automatically use the latest weights.

    Runs the forward in eval mode (correct inference if BatchNorm/Dropout are ever added) but restores
    the net's prior mode afterwards, so `infer` is side-effect-free w.r.t. training mode. Upcasts to
    float64 on the host *after* the device->host copy, so that copy moves float32, not float64.

    Moves the net onto `device` once up front, so the callback is self-contained on CPU or GPU."""
    net.to(device)
    c, h, w = net.obs_shape

    def infer(obs_batch: np.ndarray) -> np.ndarray:
        was_training = net.training
        net.eval()
        with torch.no_grad():
            x = torch.from_numpy(np.ascontiguousarray(obs_batch)).reshape(-1, c, h, w).to(device)
            q = net(x)
        net.train(was_training)
        return q.cpu().double().numpy()

    return infer


class ReplayBuffer:
    """Off-policy replay of TreeStrap records — a faithful port of `snake_RL`'s
    `EnsembleTreeStrapBuffer`. Each record is one contiguous float32 row `[obs ‖ K·A target ‖ K mask]`
    in a ring buffer, so a sampled minibatch reaches the device in a single transfer and the trainer
    splits it there. Reusing each record across many gradient steps amortises the (expensive) search
    that produced it — without a buffer every record costs a full search and is used exactly once.
    """

    def __init__(self, capacity: int, obs_dim: int, n_heads: int, n_actions: int, seed: int = 0) -> None:
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.n_heads = n_heads
        self.n_actions = n_actions
        self._target_w = n_heads * n_actions
        self._row = obs_dim + self._target_w + n_heads
        self._data = np.zeros((capacity, self._row), dtype=np.float32)
        self._rng = np.random.default_rng(seed)
        self._idx = 0
        self._size = 0

    @property
    def size(self) -> int:
        return self._size

    def push_batch(self, obs: np.ndarray, target: np.ndarray, mask: np.ndarray) -> None:
        """Append a `collect` batch: `obs` (M, obs_dim), `target` (M, K, A), `mask` (M, K). Equivalent
        to M sequential single-record pushes (ring-overwrites oldest); requires M <= capacity."""
        m = obs.shape[0]
        if m == 0:
            return
        rows = np.empty((m, self._row), dtype=np.float32)
        rows[:, : self.obs_dim] = obs
        rows[:, self.obs_dim : self.obs_dim + self._target_w] = np.asarray(target, dtype=np.float32).reshape(m, -1)
        rows[:, self.obs_dim + self._target_w :] = mask
        pos = (self._idx + np.arange(m)) % self.capacity
        self._data[pos] = rows
        self._idx = (self._idx + m) % self.capacity
        self._size = min(self._size + m, self.capacity)

    def sample(self, batch_size: int) -> torch.Tensor:
        """A `(batch_size, row)` float32 tensor — obs, target, and mask travel to the device together."""
        idx = self._rng.integers(0, self._size, size=batch_size)
        return torch.from_numpy(self._data[idx])


@dataclass
class StepMetrics:
    """One gradient step's diagnostics, mirroring `snake_RL`'s `TrainStepMetrics`: the masked-Huber
    `loss`, the mean predicted Q over the batch, and the mean searched target."""

    loss: float
    mean_q: float
    mean_target_q: float


@dataclass
class CollectReport:
    """One `collect` call's data-generation telemetry, for logging the self-play learning curve and
    throughput: the `Engine.collect` telemetry dict (finished-episode summaries + per-decision search
    aggregates), the number of `records` produced, and the wall-clock `seconds` the collect took."""

    iteration: int
    records: int
    seconds: float
    telemetry: dict[str, object]


def train(
    engine: object,
    net: BootstrappedQNetwork,
    optimizer: Optimizer,
    *,
    iterations: int | None = None,
    max_episodes: int | None = None,
    collect_size: int,
    batch_size: int,
    grad_steps_per_collect: int = 1,
    buffer_capacity: int = 100_000,
    min_buffer_size: int | None = None,
    device: str | torch.device = "cpu",
    grad_clip: float = 10.0,
    seed: int = 0,
    on_step: Callable[[int, StepMetrics], None] | None = None,
    on_collect: Callable[[int, CollectReport], None] | None = None,
) -> list[float]:
    """Off-policy actor-learner loop; returns the per-gradient-step losses.

    Each iteration searches with the current network — `engine.collect(collect_size, infer)` — pushes
    the TreeStrap records `(obs, targets, masks)` into a replay buffer, then (once it holds at least
    `min_buffer_size`) takes `grad_steps_per_collect` gradient steps, each on a fresh minibatch sampled
    from the buffer (per-head masked Huber, gradient-clipped). Because `infer` reads the live `net`,
    the next collect already searches with the updated weights — the actor-learner sync is implicit.

    The loop runs until it has done `iterations` collects and/or completed `max_episodes` finished
    episodes (whichever bound is hit first); at least one must be set. `max_episodes` is the budget that
    matches snake_RL's `num_episodes` — the number of *self-play episodes* generated, independent of how
    many collects or gradient steps that takes (which depend on episode length and `--n-games`).

    Reusing buffered records across steps amortises the search (the dominant cost): the reuse factor is
    roughly `grad_steps_per_collect * batch_size / collect_size`, and replay decorrelates updates —
    matching `EnsembleTreeStrapRunner`'s dynamics rather than the single-pass on-policy approximation.

    Telemetry (optional, no logging dependency in the library): `on_step(step, StepMetrics)` fires per
    gradient step, `on_collect(iteration, CollectReport)` per collect (including the warm-up collects
    before the buffer fills, so the data-generation curve and throughput start at step 0). A caller
    wires these to a TensorBoard `SummaryWriter`.
    """
    if iterations is None and max_episodes is None:
        raise ValueError("set iterations and/or max_episodes")
    net.to(device)
    net.train()  # gradient steps run in train mode; `infer` is mode-neutral (saves/restores)
    obs_dim = math.prod(net.obs_shape)
    k, a = net.n_heads, net.n_actions
    buffer = ReplayBuffer(buffer_capacity, obs_dim, k, a, seed)
    min_buffer = min_buffer_size if min_buffer_size is not None else batch_size
    infer = make_infer(net, device)
    c, h, w = net.obs_shape
    losses: list[float] = []
    episodes_done = 0
    it = 0
    while iterations is None or it < iterations:
        t0 = time.perf_counter()
        obs, target, mask, telemetry = engine.collect(collect_size, infer)  # type: ignore[attr-defined]
        collect_seconds = time.perf_counter() - t0
        buffer.push_batch(obs, target, mask)
        episodes_done += len(telemetry["episodes"])
        if on_collect is not None:
            on_collect(it, CollectReport(it, int(obs.shape[0]), collect_seconds, telemetry))
        if buffer.size >= min_buffer:
            for _ in range(grad_steps_per_collect):
                batch = buffer.sample(batch_size).to(device)  # one host->device transfer
                obs_t = batch[:, :obs_dim].reshape(-1, c, h, w)
                target_t = batch[:, obs_dim : obs_dim + k * a].reshape(-1, k, a)
                mask_t = batch[:, obs_dim + k * a :]
                q = net(obs_t)
                loss = treestrap_loss(q, target_t, mask_t)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), max_norm=grad_clip)
                optimizer.step()
                losses.append(float(loss.item()))
                if on_step is not None:
                    on_step(
                        len(losses),
                        StepMetrics(losses[-1], float(q.mean().item()), float(target_t.mean().item())),
                    )
        it += 1
        if max_episodes is not None and episodes_done >= max_episodes:
            break
    return losses
