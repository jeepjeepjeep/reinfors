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
    # temperature > 0 (AlphaZero-style) samples the first temperature_drop plies of each episode
    # ∝ visits^(1/temperature) for training self-play diversity (None = whole episode); 0 = greedy.
    @staticmethod
    def Mcts(
        num_simulations: int = ...,
        uct_c: float = ...,
        max_depth: int = ...,
        act_by: str = ...,
        temperature: float = ...,
        temperature_drop: int | None = ...,
    ) -> PolicyHandle: ...
    # AlphaZero (PUCT); pairs with learners.AlphaZero; sequential/single-agent games only. The infer
    # callback returns a (policy_logits (N, A) f64, values (N,) f64) tuple — one forward, both heads.
    # Root Dirichlet noise (noise_epsilon/noise_alpha) + acting temperature drive self-play diversity;
    # acting is by visit count. temperature_drop=None applies the temperature to whole episodes.
    @staticmethod
    def AlphaZero(
        num_simulations: int = ...,
        c_puct: float = ...,
        max_depth: int = ...,
        noise_epsilon: float = ...,
        noise_alpha: float = ...,
        temperature: float = ...,
        temperature_drop: int | None = ...,
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
    # AlphaZero record production: (obs, pi, z) — pi = tau=1 root visit distribution, z = discounted
    # realized return (gamma=1 + win/loss rewards = the paper's z). Pairs with policies.AlphaZero.
    @staticmethod
    def AlphaZero(gamma: float = ...) -> LearnerHandle: ...

class AlphaZeroBatch:
    """`Engine.collect` result for the AlphaZero family. Also unpacks positionally as
    `obs, policy_targets, value_targets, telemetry = batch`."""

    obs: NDArray[np.float32]
    policy_targets: NDArray[np.float64]
    value_targets: NDArray[np.float64]
    telemetry: dict[str, Any]
    def __len__(self) -> int: ...
    def __getitem__(self, i: int) -> Any: ...

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
    def collect(self, n_records: int, infer: Any) -> TreeStrapBatch | DqnBatch | AlphaZeroBatch: ...

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
