"""Type stubs for the compiled `reinfors._reinfors` extension module.

Hand-maintained for now; once the API surface grows we can generate these from the Rust bindings.
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray

def core_version() -> str: ...
def core_build_profile() -> str: ...  # "debug" | "release"; perf numbers are only valid on "release"

class Reward:
    # A generic named-weight reward; each game validates the keys it understands (snake: step/food/
    # loss/draw/kill/win/survival; connect4: win/loss/draw; gridworld: step/goal).
    def __init__(self, **weights: float) -> None: ...

# Space descriptors a game advertises (re-exported as rf.spaces.Box / rf.spaces.Discrete).
class Box:
    shape: tuple[int, ...]
    low: NDArray[np.float32]
    high: NDArray[np.float32]

class Discrete:
    n: int

# Opaque composition handles, built via the staticmethod constructors and passed to `Engine`.
class GameHandle:
    @staticmethod
    def Snake(
        grid_size: int = ...,
        initial_length: int = ...,
        food: int = ...,
        play_to_last: bool = ...,
        win_food_lead: int | None = ...,
        max_ticks: int | None = ...,
    ) -> GameHandle: ...
    @staticmethod
    def Connect4() -> GameHandle: ...
    @staticmethod
    def GridWorld(
        size: int = ...,
        goal_row: int = ...,
        goal_col: int = ...,
        max_ticks: int | None = ...,
    ) -> GameHandle: ...
    def observation_space(self) -> Box: ...
    def action_space(self) -> Discrete: ...
    # The truncation horizon (episode cap); None = never truncate (only for games that always end,
    # like Connect-4). Loop-prone games default to a finite cap.
    def truncation_horizon(self) -> int | None: ...

class PolicyHandle:
    @staticmethod
    def SelectiveExpectimax(
        expansion_budget: int = ...,
        top_k: int = ...,
        max_depth: int = ...,
        beta: float = ...,
        food_samples: int = ...,
        n_heads: int = ...,
        epsilon: float = ...,
        opponent: str = ...,
        opp_temperature: float = ...,
        opp_floor: float = ...,
    ) -> PolicyHandle: ...
    @staticmethod
    def EpsilonGreedyQ(n_heads: int = ..., epsilon: float = ...) -> PolicyHandle: ...
    # MCTS (UCT); pairs with TreeStrap; sequential/single-agent games only. act_by: "value" | "visits".
    @staticmethod
    def Mcts(
        num_simulations: int = ...,
        uct_c: float = ...,
        max_depth: int = ...,
        act_by: str = ...,
    ) -> PolicyHandle: ...

class LearnerHandle:
    @staticmethod
    def TreeStrap(
        gamma: float = ...,
        outcome_weight: float = ...,
        bootstrap_p: float = ...,
        interior_targets: bool = ...,
    ) -> LearnerHandle: ...
    @staticmethod
    def Dqn(bootstrap_p: float = ...) -> LearnerHandle: ...

class TreeStrapBatch:
    """`Engine.collect` result for the TreeStrap family. Also unpacks positionally as
    `obs, targets, masks, telemetry = batch`."""

    obs: NDArray[np.float32]
    targets: NDArray[np.float64]
    masks: NDArray[np.float32]
    telemetry: dict[str, Any]
    def __len__(self) -> int: ...
    def __getitem__(self, i: int) -> Any: ...

class DqnBatch:
    """`Engine.collect` result for the DQN family. Also unpacks positionally as
    `obs, actions, rewards, next_obs, dones, masks, telemetry = batch`."""

    obs: NDArray[np.float32]
    actions: NDArray[np.int64]
    rewards: NDArray[np.float64]
    next_obs: NDArray[np.float32]
    dones: NDArray[np.bool_]
    masks: NDArray[np.float32]
    telemetry: dict[str, Any]
    def __len__(self) -> int: ...
    def __getitem__(self, i: int) -> Any: ...

class Engine:
    def __init__(
        self,
        game: GameHandle,
        reward: Reward | None,
        policy: PolicyHandle,
        learner: LearnerHandle,
        n_games: int,
        seed: int = ...,
        # Reached-state start buffer (off by default; snake only): seed a fraction of episodes from
        # previously-reached states to flatten start-state coverage. `p_fresh` is the fraction that
        # still start fresh from `initial_state`.
        start_buffer: bool = ...,
        start_buffer_capacity: int = ...,
        p_fresh: float = ...,
    ) -> None: ...
    # The batch is learner-shaped: the TreeStrap family yields a `TreeStrapBatch`, the DQN family a
    # `DqnBatch`. Both expose named fields and also unpack positionally (back-compat with the old tuple).
    # `infer` is a Python callable, or (with the `nn` extra) a `Net` for a callback-free forward.
    def collect(self, n_records: int, infer: Any) -> TreeStrapBatch | DqnBatch: ...
    # Fully-in-Rust training loop (TreeStrap only): `steps` rounds of (collect `collect_size` records via
    # the trainer's net, then `reuse * records / batch_size` mini-batch steps). Returns one telemetry dict
    # per collect: `losses` (per grad step), `records`, `collect_seconds`, `episodes`, and the `search`
    # aggregates — the keys `collect` reports. `batch_size=None` → one full-batch step. See `reinfors.nn`.
    def train(
        self,
        trainer: TreeStrapTrainer,
        steps: int,
        collect_size: int,
        batch_size: int | None = ...,
        reuse: float = ...,
    ) -> list[dict[str, Any]]: ...

class Env:
    """A caller-driven single-game instance (the inverse of `Engine`): you supply each tick's actions."""

    def __init__(self, game: GameHandle, reward: Reward | None = ..., seed: int = ...) -> None: ...
    def reset(self) -> None: ...
    def done(self) -> bool: ...
    def num_agents(self) -> int: ...
    def action_count(self) -> int: ...
    def active_agents(self) -> list[int]: ...
    def legal_actions(self, agent: int) -> list[int]: ...
    def observe(self, agent: int) -> NDArray[np.float32]: ...
    def observation_space(self) -> Box: ...
    def state(self) -> dict[str, Any]: ...
    # Per-agent events (game-specific: snake → dict, connect4 → str, gridworld → dict). The Env holds
    # no reward; a game-aware caller reads the outcome from these.
    def step(self, actions: dict[int, int]) -> list[Any]: ...
    # Per-agent scalar rewards for the most recent `step`, or None if built without a `reward` (the
    # reward-free play/eval default) or before the first `step`. The training-facing `reinfors.gym`
    # adapters read this so the event→reward mapping stays in Rust.
    @property
    def rewards(self) -> list[float] | None: ...

# Rust-native value net (candle — pure Rust, in the wheel by default). Declared here for typing; absent
# at runtime only in a `--no-default-features` build — `reinfors.nn` guards for that. Pass an instance to
# `Engine.collect` in place of an `infer` callable to run the forward in Rust (no callback).
class Net:
    # device: "cpu" | "metal" | "cuda" | "auto" (GPU backends must be compiled in).
    @staticmethod
    def conv(obs_shape: tuple[int, int, int], n_actions: int, n_heads: int, device: str = ...) -> Net: ...
    @staticmethod
    def mlp(in_dim: int, hidden: int, n_actions: int, n_heads: int, device: str = ...) -> Net: ...
    @property
    def n_heads(self) -> int: ...
    @property
    def n_actions(self) -> int: ...
    def forward(self, obs: NDArray[np.float32]) -> NDArray[np.float64]: ...
    def get_weights(self) -> list[NDArray[np.float32]]: ...
    def set_weights(self, weights: list[NDArray[np.float32]]) -> None: ...

# Adam + masked-Huber trainer over a `Net`'s parameters (nn extra). Fused via `Engine.train`, or stepped
# from Python via `update` on a collected batch. Same runtime-optionality as `Net`.
class TreeStrapTrainer:
    def __init__(self, net: Net, lr: float = ...) -> None: ...
    # Trains the net this trainer was built from (which it owns).
    def update(
        self,
        obs: NDArray[np.float32],
        targets: NDArray[np.float64],
        masks: NDArray[np.float32],
    ) -> float: ...
