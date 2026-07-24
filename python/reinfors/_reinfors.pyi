"""Type stubs for the compiled `reinfors._reinfors` extension module.

Hand-maintained for now; once the API surface grows we can generate these from the Rust bindings.
"""

from collections.abc import Iterator
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
    # Standard chess (cozy-chess rules). Actions = the AlphaZero 8x8x73 = 4672 encoding (~35 legal
    # per position — the tree searches mask to the legal set). `encoder` is an rf.encoders.* handle
    # picking the observation view (default MinimalChess, (19, 8, 8)); the state's history
    # bookkeeping follows the selected encoder. max_ticks defaults to 512 (weak-net self-play
    # shuffles inside the fifty-move window).
    @staticmethod
    def Chess(max_ticks: int | None = ..., encoder: EncoderHandle | None = ...) -> GameHandle: ...
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

class EncoderHandle:
    # Observation-encoder handles, passed to a game handle's `encoder=` kwarg. Game-specific; any
    # state bookkeeping a view needs is enabled in the game automatically when selected.
    @staticmethod
    def MinimalChess() -> EncoderHandle: ...
    # AlphaZero's chess view: 14*history_length + 7 planes; history_length=8 = the paper's 119.
    @staticmethod
    def AlphaZeroChess(history_length: int = ...) -> EncoderHandle: ...

class PolicyHandle:
    # Best-first selective expectimax (expand-once). chance_mode: "committed" (default; the
    # historical food_samples estimator) | "expand_all" (exact fan). "always_resample" is rejected —
    # an expand-once search has no traversal event to redraw on.
    @staticmethod
    def SelectiveExpectimax(
        expansion_budget: int = ...,
        top_k: int = ...,
        max_depth: int = ...,
        beta: float = ...,
        chance_mode: str = ...,
        chance_samples: int = ...,
        n_heads: int = ...,
        epsilon: float = ...,
        opponent: str = ...,
        opp_temperature: float = ...,
        opp_floor: float = ...,
    ) -> PolicyHandle: ...
    @staticmethod
    def EpsilonGreedyQ(n_heads: int = ..., epsilon: float = ...) -> PolicyHandle: ...
    # MCTS (UCT); pairs with TreeStrap; sequential, single-agent, AND simultaneous (decoupled/DUCT
    # per-agent statistics) games. act_by: "value" | "visits".
    # temperature > 0 (AlphaZero-style) samples the first temperature_drop plies of each episode
    # ∝ visits^(1/temperature) for training self-play diversity (None = whole episode); 0 = greedy.
    # chance_mode (declared-chance games): "always_resample" (fresh draw ∝ p per descent, unbiased
    # default) | "committed" (freeze chance_samples draws per edge — food_samples-style, for wide
    # fans) | "expand_all" (evaluate every outcome at expansion — exact, narrow fans).
    @staticmethod
    def Mcts(
        num_simulations: int = ...,
        uct_c: float = ...,
        max_depth: int = ...,
        act_by: str = ...,
        temperature: float = ...,
        temperature_drop: int | None = ...,
        chance_mode: str = ...,
        chance_samples: int = ...,
    ) -> PolicyHandle: ...
    # AlphaZero (PUCT); pairs with learners.AlphaZero; sequential, single-agent, and simultaneous
    # (DUCT) games — noise_scope: "requester" (default) | "both" picks which root priors the
    # Dirichlet noise perturbs in a simultaneous tree. The infer
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
        chance_mode: str = ...,
        chance_samples: int = ...,
        noise_scope: str = ...,
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
    def __iter__(self) -> Iterator[Any]: ...  # positional unpacking (runtime: sequence protocol)

class TreeStrapBatch:
    """`Engine.collect` result for the TreeStrap family. Also unpacks positionally as
    `obs, targets, masks, telemetry = batch`."""

    obs: NDArray[np.float32]
    targets: NDArray[np.float64]
    masks: NDArray[np.float32]
    telemetry: dict[str, Any]
    def __len__(self) -> int: ...
    def __getitem__(self, i: int) -> Any: ...
    def __iter__(self) -> Iterator[Any]: ...  # positional unpacking (runtime: sequence protocol)

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
    def __iter__(self) -> Iterator[Any]: ...  # positional unpacking (runtime: sequence protocol)

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
        # Net-evaluation cache (entries; 0 = off): position-keyed reuse of net rows across the
        # search, cleared when weights_updated() is called. Raises effective throughput in
        # transposition-rich games; behavior-identical given fixed weights.
        infer_cache: int = ...,
    ) -> None: ...
    # Tell the engine the net's weights changed (call after every weight sync — e.g. right after
    # load_state_dict onto the collector net). Clears the infer cache at the next round boundary;
    # thread-safe, callable while a collect_stream is active; no-op when the cache is off.
    def weights_updated(self) -> None: ...
    # The batch is learner-shaped: the TreeStrap family yields a `TreeStrapBatch`, the DQN family a
    # `DqnBatch`. Both expose named fields and also unpack positionally (back-compat with the old tuple).
    def collect(self, n_records: int, infer: Any) -> TreeStrapBatch | DqnBatch | AlphaZeroBatch: ...
    # Continuous background collection: a worker thread runs collect after collect into a bounded
    # queue of `depth` finished batches (None = unbounded, the continuous-actor topology). The engine
    # is held by the stream until stop(); collect() errors meanwhile. Weight staleness is the
    # caller's: sync a collector-net copy at your own cadence.
    def collect_stream(self, collect_size: int, infer: Any, depth: int | None = ...) -> CollectStream: ...

class CollectStream:
    """A running background collection. `next()` blocks (GIL released) for the worker's next batch;
    iterating yields batches until `stop()`. Context-manager exit stops the stream and returns the
    engine to its `Engine`.

    Single-consumer: one thread loops `next()` and owns `stop()`. Other threads run freely during
    the wait but must not touch the stream object (a concurrent `stop()` raises RuntimeError
    "Already borrowed" and could not interrupt a blocked `next()` anyway). A stream dropped without
    `stop()` permanently forfeits its engine — prefer the `with` form."""

    def next(self) -> TreeStrapBatch | DqnBatch | AlphaZeroBatch: ...
    def pending(self) -> int: ...
    def stop(self) -> None: ...
    def __iter__(self) -> CollectStream: ...
    def __next__(self) -> TreeStrapBatch | DqnBatch | AlphaZeroBatch: ...
    def __enter__(self) -> CollectStream: ...
    def __exit__(self, *args: Any) -> None: ...

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
