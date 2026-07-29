"""Type stubs for the compiled `reinfors._reinfors` extension module.

Hand-maintained for now; once the API surface grows we can generate these from the Rust bindings.
"""

from collections.abc import Iterator
from typing import Any

import numpy as np
from numpy.typing import NDArray

def core_version() -> str: ...

# Chess interop (pure; for referees/tools): standard-UCI <-> 8x8x73 action ids, grounded in a
# FEN so castling/promotions disambiguate exactly. ValueError on bad FEN / illegal move.
def chess_uci_action(uci: str, fen: str) -> int: ...
def chess_action_uci(action: int, fen: str) -> str: ...
def core_build_profile() -> str: ...  # "debug" | "release"; perf numbers are only valid on "release"

class EngineSnapshot:
    # Quiescent engine capture (episodes, RNGs, policy state, partial trajectories, start
    # buffer, weights generation). Record-exact contract: restored collects yield byte-identical
    # records; the infer cache is excluded (call pattern may differ, results cannot).
    fingerprint: str
    schema_version: int
    weights_generation: int
    policy_version: str | None
    def to_bytes(self) -> bytes: ...
    @staticmethod
    def from_bytes(data: bytes) -> EngineSnapshot: ...

class EnvSnapshot:
    # Opaque, restorable Env capture (native state via the game codec + RNG + terminal status).
    # Produced by Env.snapshot(); validated while decoding — malformed blobs raise ValueError.
    fingerprint: str
    schema_version: int
    def to_bytes(self) -> bytes: ...
    @staticmethod
    def from_bytes(data: bytes) -> EnvSnapshot: ...

class Reward:
    # A generic named-weight reward; each game validates the keys it understands (snake: step/food/
    # loss/draw/kill/win/survival; connect4: win/loss/draw; gridworld: step/goal). Weights must be
    # finite (NaN/inf raise ValueError at construction).
    def __init__(self, **weights: float) -> None: ...

# Space descriptors a game advertises (re-exported as rf.spaces.Box / rf.spaces.Discrete).
class Box:
    shape: tuple[int, ...]
    low: NDArray[np.float32]
    high: NDArray[np.float32]

class Discrete:
    n: int

class Cfr:
    """Counterfactual regret minimization over a 2-player declared-chance game with
    information-state keys. Variants: "vanilla", "plus" (CFR+), "external_mccfr". The output
    is the AVERAGE strategy, keyed by `Env.information_state_key` bytes."""

    def __init__(self, game: GameHandle, variant: str = ..., seed: int = ...) -> None: ...
    def iterate(self, n: int) -> None: ...
    @property
    def iterations(self) -> int: ...
    @property
    def num_infosets(self) -> int: ...
    def exploitability(self) -> float: ...
    def expected_value(self, player: int) -> float: ...
    def average_strategy(self, key: bytes) -> tuple[list[int], list[float]] | None: ...
    def save(self) -> bytes: ...
    def load(self, bytes: bytes) -> None: ...

# Opaque composition handles, built via the staticmethod constructors and passed to `Engine`.
class GameHandle:
    @staticmethod
    def TexasHoldem(
        num_players: int = ...,
        stack: int = ...,
        small_blind: int = ...,
        big_blind: int = ...,
    ) -> GameHandle: ...
    # One episode = one hand at fresh stacks; chip-delta rewards (zero-sum), reward key: scale.
    # Hidden information: search policies reject it; train with EpsilonGreedyQ + Dqn.
    @staticmethod
    def KuhnPoker() -> GameHandle: ...
    # 3-card analytic testbed (12 infosets); hidden information; reward key: scale.
    @staticmethod
    def LeducPoker() -> GameHandle: ...
    # 6-card two-round benchmark; hidden information; reward key: scale.
    @staticmethod
    def Snake(
        grid_size: int = ...,
        initial_length: int = ...,
        food: int = ...,
        play_to_last: bool = ...,
        win_food_lead: int | None = ...,
        max_ticks: int | None = ...,
        num_snakes: int = ...,
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
    # Backgammon: OpenSpiel-compatible 1352-action encoding, declared dice chance (21 rolls,
    # non-uniform), no doubling cube. Reward keys: win/gammon/backgammon (defaults 1/2/3, zero-sum).
    @staticmethod
    def Backgammon(max_ticks: int | None = ...) -> GameHandle: ...
    # The goal defaults to the far corner (size-1, size-1), derived from `size`; explicit
    # coordinates must lie inside the grid. Invalid configs raise ValueError at construction.
    @staticmethod
    def GridWorld(
        size: int = ...,
        goal_row: int | None = ...,
        goal_col: int | None = ...,
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
    # Mover-relative chess view (19, 8, 8): board seen from the mover's side, action head indexed
    # under the same symmetry (role equivariance; the AlphaZero paper's convention).
    @staticmethod
    def RelativeChess() -> EncoderHandle: ...
    # OpenSpiel's chess observation replicated exactly (20, 8, 8) — the interop/benchmark view.
    @staticmethod
    def OpenSpielChess() -> EncoderHandle: ...
    # The encoder's action map (identity for absolute encoders) — for driving a net outside the
    # engine: read logits at head_index(a, agent) for each legal game action a.
    def head_index(self, action: int, agent: int) -> int: ...
    def game_action(self, head: int, agent: int) -> int: ...
    # AlphaZero's chess view: 14*history_length + 7 planes; history_length=8 = the paper's 119.
    @staticmethod
    def AlphaZeroChess(history_length: int = ...) -> EncoderHandle: ...

class ChanceModeHandle:
    # How a search consumes declared chance (rf.chance_modes.*; policy `chance=` kwarg). Expand-once
    # searches (SelectiveExpectimax) reject per-traversal modes (AlwaysResample) at construction.
    @staticmethod
    def AlwaysResample() -> ChanceModeHandle: ...
    @staticmethod
    def Committed(samples: int = ...) -> ChanceModeHandle: ...
    @staticmethod
    def ExpandAll() -> ChanceModeHandle: ...

class NoiseHandle:
    # Root exploration noise (rf.noise.*; AlphaZero `noise=` kwarg — None disables, omitted = the
    # self-play default Dirichlet(0.25, 0.3, "requester")). scope: "requester" | "all".
    @staticmethod
    def Dirichlet(epsilon: float = ..., alpha: float = ..., scope: str = ...) -> NoiseHandle: ...

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
        chance: ChanceModeHandle | None = ...,
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
        chance: ChanceModeHandle | None = ...,
    ) -> PolicyHandle: ...
    # AlphaZero (PUCT); pairs with learners.AlphaZero; sequential, single-agent, and simultaneous
    # (DUCT) games — noise_scope: "requester" (default) | "all" picks which root priors the
    # Dirichlet noise perturbs in a simultaneous tree. The infer
    # callback returns a (policy_logits (N, A) f64, values (N,) f64) tuple — one forward, both heads.
    # Root Dirichlet noise (noise_epsilon/noise_alpha) + acting temperature drive self-play diversity;
    # acting is by visit count. temperature_drop=None applies the temperature to whole episodes.
    @staticmethod
    def AlphaZero(
        num_simulations: int = ...,
        c_puct: float = ...,
        max_depth: int = ...,
        temperature: float = ...,
        temperature_drop: int | None = ...,
        chance: ChanceModeHandle | None = ...,
        noise: NoiseHandle | None = ...,
        sequential_backup: str = ...,
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
    `obs, policy_targets, value_targets, policy_weights, telemetry = batch`.
    `policy_weights` masks the policy loss: 1.0 on acting-agent rows, 0.0 on value-only rows
    (non-mover perspectives, emitted by sequential N>2 games; their pi rows are inert zeros).
    Policy term: `(w * cross_entropy(logits, pi)).sum() / w.sum()`; every row trains the value
    head. 2p and simultaneous compositions emit all-ones weights."""

    obs: NDArray[np.float32]
    policy_targets: NDArray[np.float64]
    value_targets: NDArray[np.float64]
    policy_weights: NDArray[np.float64]
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
    # Legality in CSR form (record i's ids = ids[offsets[i]:offsets[i+1]]) — sparse because dense
    # (M, A) masks dwarf the observations on wide action spaces (~37 GB per 1M chess transitions).
    # THE bootstrap rule: bootstrap iff record i's next slice is NON-EMPTY (empty = terminal OR
    # alternating-game truncation tail -> target = r). `dones` is an episode flag, not target math
    # ((1 - done) * max meets -inf as NaN). Complete safe target, densified per minibatch:
    #   counts = np.diff(next_legal_offsets); rows = np.repeat(np.arange(M), counts)
    #   mask = np.zeros((M, A), bool); mask[rows, next_legal_ids] = True
    #   q  = np.where(mask, q_next, -np.inf).max(-1)
    #   td = rewards + gamma * np.where(np.isfinite(q), q, 0.0)
    legal_ids: NDArray[np.int64]
    legal_offsets: NDArray[np.int64]
    next_legal_ids: NDArray[np.int64]
    next_legal_offsets: NDArray[np.int64]
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
    # The fully resolved immutable composition (defaults included), JSON-compatible;
    # rf.engine_from_config(engine.resolved_config()) reconstructs an equivalent engine.
    def resolved_config(self) -> dict[str, Any]: ...
    # 128-bit hex over reinfors-produced canonical bytes; compare, never recompute.
    def config_fingerprint(self) -> str: ...
    def snapshot(self, policy_version: str | None = ...) -> EngineSnapshot: ...
    def restore(self, snapshot: EngineSnapshot, expect_policy_version: str | None = ...) -> None: ...
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
    # Lossless checkpoint barrier: stops new collects, finishes the in-flight one with real
    # inference, returns every remaining batch; the engine (returned to its Engine) then matches
    # the delivered batches exactly — snapshot() right after is record-exact.
    def pause(self) -> list[Any]: ...
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
    def information_state_key(self, agent: int) -> bytes: ...
    # Composition (game incl. encoder + resolved reward) and its fingerprint (embedded in
    # snapshots; excludes the reinfors version so snapshots survive upgrades).
    def resolved_config(self) -> dict[str, Any]: ...
    def config_fingerprint(self) -> str: ...
    def snapshot(self) -> EnvSnapshot: ...
    # Rejects other compositions (fingerprint), unsupported schemas, malformed state bytes.
    # Lands at a step boundary: rewards is None until the next step.
    def restore(self, snapshot: EnvSnapshot) -> None: ...
    # Independent env at this exact point. Clone-exact by default (identical future chance
    # stream); pass seed for a divergent fork.
    def fork(self, seed: int | None = ...) -> Env: ...
    # Per-agent events (game-specific: snake → dict, connect4 → str, gridworld → dict). The Env holds
    # no reward; a game-aware caller reads the outcome from these.
    def step(self, actions: dict[int, int]) -> list[Any]: ...
    # Per-agent scalar rewards for the most recent `step`, or None if built without a `reward` (the
    # reward-free play/eval default) or before the first `step`. The training-facing `reinfors.gym`
    # adapters read this so the event→reward mapping stays in Rust.
    @property
    def rewards(self) -> list[float] | None: ...
