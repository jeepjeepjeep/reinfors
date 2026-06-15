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

from collections.abc import Callable, Sequence

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
    successive searches automatically use the latest weights."""
    c, h, w = net.obs_shape

    def infer(obs_batch: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x = torch.from_numpy(np.ascontiguousarray(obs_batch)).reshape(-1, c, h, w).to(device)
            q = net(x)
        return q.double().cpu().numpy()

    return infer


def train(
    engine: object,
    net: BootstrappedQNetwork,
    optimizer: Optimizer,
    *,
    iterations: int,
    batch_size: int,
    device: str | torch.device = "cpu",
    grad_clip: float = 10.0,
    on_step: Callable[[int, float], None] | None = None,
) -> list[float]:
    """Run the actor-learner loop for `iterations` steps and return the per-step losses.

    Each step: `engine.collect(batch_size, infer)` searches the parallel games with the current
    network and returns TreeStrap records `(obs, targets, masks)`; the network regresses onto them
    (per-head masked Huber, gradient-clipped). Because `infer` reads the live `net`, the next collect
    already searches with the updated weights — no explicit actor-learner weight sync.
    """
    net.to(device)
    infer = make_infer(net, device)
    c, h, w = net.obs_shape
    losses: list[float] = []
    for step in range(iterations):
        obs, target, mask = engine.collect(batch_size, infer)  # type: ignore[attr-defined]
        obs_t = torch.from_numpy(np.ascontiguousarray(obs)).reshape(-1, c, h, w).to(device)
        target_t = torch.from_numpy(np.ascontiguousarray(target)).float().to(device)
        mask_t = torch.from_numpy(np.ascontiguousarray(mask)).to(device)
        q = net(obs_t)  # (B, K, A)
        loss = treestrap_loss(q, target_t, mask_t)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), max_norm=grad_clip)
        optimizer.step()
        losses.append(float(loss.item()))
        if on_step is not None:
            on_step(step, losses[-1])
    return losses
