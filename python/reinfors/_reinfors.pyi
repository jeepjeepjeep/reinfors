"""Type stubs for the compiled `reinfors._reinfors` extension module.

Hand-maintained for now; once the API surface grows we can generate these from the Rust bindings.
"""

from collections.abc import Iterator
from typing import Any

import numpy as np
from numpy.typing import NDArray

def core_version() -> str: ...

# Chess interop (pure; for referees/tools): standard-UCI <-> 8x8x73 action ids, grounded in a
# FEN so castling/promotions disambiguate exactly. Castling uses standard UCI (`e1g1`, not
# cozy-chess's internal king-takes-rook form). ValueError on bad FEN / illegal move.
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
    # A generic named-weight reward; each game validates its own event keys. See the game catalogue
    # for supported keys and defaults. Weights must be finite (NaN/inf raise ValueError).
    def __init__(self, **weights: float) -> None: ...

# Space descriptors a game advertises (re-exported as rf.spaces.Box / rf.spaces.Discrete).
class Box:
    shape: tuple[int, ...]
    low: NDArray[np.float32]
    high: NDArray[np.float32]

class Discrete:
    n: int

class Cfr:
    """Counterfactual regret minimization over compatible sequential declared-chance games with
    information-state keys. Variants: "vanilla", "plus" (CFR+), "external_mccfr". The output
    is the AVERAGE strategy, keyed by `Env.information_state_key` bytes. See the generated
    compatibility catalogue for built-in compositions. Solver values and exact metrics use the
    game's native utility (raw chips for poker), independent of any `rf.Reward` scaling."""

    def __init__(self, game: GameHandle, variant: str = ..., seed: int = ...) -> None: ...
    def iterate(self, n: int) -> None: ...
    @property
    def iterations(self) -> int: ...
    @property
    def num_infosets(self) -> int: ...
    @property
    def num_players(self) -> int: ...
    # NashConv / num_players (pyspiel naming). N>2: distance from equilibrium, NO guarantee.
    # Exact metrics enumerate the tree: ValueError past the cap (e.g. full hold'em, 7p+ Kuhn).
    def exploitability(self) -> float: ...
    # sum_i (br_i - v_i): zero exactly at a Nash equilibrium.
    def nash_conv(self) -> float: ...
    # Each player's exact best-response value vs the others' average profile.
    def best_response_values(self) -> list[float]: ...
    def expected_value(self, player: int) -> float: ...
    # None means the infoset was unvisited; callers playing an MCCFR profile should use uniform legal play.
    def average_strategy(self, key: bytes) -> tuple[list[int], list[float]] | None: ...
    def save(self) -> bytes: ...
    def load(self, bytes: bytes) -> None: ...

class DeepCfrBatch:
    advantage_obs: NDArray[np.float32]
    advantage_iterations: NDArray[np.int64]
    advantage_legal_offsets: NDArray[np.int64]
    advantage_legal_ids: NDArray[np.int64]
    advantage_targets: NDArray[np.float64]
    strategy_obs: NDArray[np.float32]
    strategy_iterations: NDArray[np.int64]
    strategy_players: NDArray[np.int64]
    strategy_legal_offsets: NDArray[np.int64]
    strategy_legal_ids: NDArray[np.int64]
    strategy_probs: NDArray[np.float64]
    telemetry: dict[str, Any]

class DeepCfr:
    """Deep CFR data generator (external sampling): traversals query the current advantage
    networks through `infer` (one callable, or a per-player sequence) and emit advantage and
    strategy training samples. Each callback maps float32 `(rows, observation_size)` to float32 or
    float64 advantages `(rows, actions)`. Buffers, weighting, and training are the caller's."""

    def __init__(self, game: GameHandle, seed: int = ...) -> None: ...
    # Call exactly once before every iteration's per-player collect calls; the resulting iteration
    # number is the linear-CFR sample weight.
    def next_iteration(self) -> None: ...
    @property
    def iteration(self) -> int: ...
    def resolved_config(self) -> dict[str, Any]: ...
    # infer/policy_infer outputs may be float64 or float32 (exact widening).
    def collect(self, player: int, traversals: int, infer: Any) -> DeepCfrBatch: ...
    # Enumerable games only; raises ValueError past the exact-enumeration cap. Rows are scores:
    # negatives are clamped, legal entries renormalized, and degenerate rows become uniform.
    # Results use native game utility (raw chips for poker), not an rf.Reward scale.
    def exploitability(self, policy_infer: Any) -> float: ...

# Opaque composition handles, built via the staticmethod constructors and passed to `Engine`.
class GameHandle:
    # One episode is one hand at fresh stacks. Events are zero-sum per-seat chip deltas; `scale`
    # converts units (for example, `rf.Reward(scale=1 / big_blind)` reports rewards in blinds).
    # The button is redrawn each hand so seats rotate positions. Search policies reject its hidden
    # state; train with EpsilonGreedyQ + Dqn or use a solver workflow.
    @staticmethod
    def TexasHoldem(
        num_players: int = ...,
        stack: int = ...,
        small_blind: int = ...,
        big_blind: int = ...,
        encoder: EncoderHandle | None = ...,
    ) -> GameHandle: ...
    # 3-card analytic testbed (12 infosets); hidden information; reward key: scale.
    @staticmethod
    def KuhnPoker(players: int = ..., encoder: EncoderHandle | None = ...) -> GameHandle: ...
    # 6-card two-round benchmark; hidden information; reward key: scale.
    @staticmethod
    def LeducPoker(encoder: EncoderHandle | None = ...) -> GameHandle: ...
    # Simultaneous multi-snake game with declared food chance; max_ticks is its rollout horizon.
    @staticmethod
    def Snake(
        grid_size: int = ...,
        initial_length: int = ...,
        food: int = ...,
        play_to_last: bool = ...,
        win_food_lead: int | None = ...,
        max_ticks: int | None = ...,
        num_snakes: int = ...,
        encoder: EncoderHandle | None = ...,
    ) -> GameHandle: ...
    @staticmethod
    def Connect4(encoder: EncoderHandle | None = ...) -> GameHandle: ...
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
    def Backgammon(max_ticks: int | None = ..., encoder: EncoderHandle | None = ...) -> GameHandle: ...
    # The goal defaults to the far corner (size-1, size-1), derived from `size`; explicit
    # coordinates must lie inside the grid. Invalid configs raise ValueError at construction.
    @staticmethod
    def GridWorld(
        size: int = ...,
        goal_row: int | None = ...,
        goal_col: int | None = ...,
        max_ticks: int | None = ...,
        encoder: EncoderHandle | None = ...,
    ) -> GameHandle: ...
    def observation_space(self) -> Box: ...
    def action_space(self) -> Discrete: ...
    @property
    def encoder(self) -> EncoderHandle: ...
    # The declared truncation horizon (episode cap), or None when the game declares no horizon.
    def truncation_horizon(self) -> int | None: ...

class EncoderHandle:
    # Every game handle carries one compatible observation encoder. Constructors use the game's
    # registered default when `encoder=None`.
    @property
    def name(self) -> str: ...
    @staticmethod
    def Snake() -> EncoderHandle: ...
    @staticmethod
    def Connect4() -> EncoderHandle: ...
    @staticmethod
    def MinimalChess() -> EncoderHandle: ...
    # Mover-relative chess view (19, 8, 8): board seen from the mover's side, action head indexed
    # under the same symmetry (role equivariance; the AlphaZero paper's convention).
    @staticmethod
    def RelativeChess() -> EncoderHandle: ...
    # OpenSpiel's chess observation replicated exactly (20, 8, 8), including its omission of
    # en-passant rights — positions differing only by those rights encode identically.
    @staticmethod
    def OpenSpielChess() -> EncoderHandle: ...
    # The encoder's action map (identity for absolute encoders) — for driving a net outside the
    # engine: read logits at head_index(a, agent) for each legal game action a.
    def head_index(self, action: int, agent: int) -> int: ...
    def game_action(self, head: int, agent: int) -> int: ...
    # AlphaZero's chess view: 14*history_length + 7 planes; history_length=8 = the paper's 119.
    @staticmethod
    def AlphaZeroChess(history_length: int = ...) -> EncoderHandle: ...
    @staticmethod
    def Backgammon() -> EncoderHandle: ...
    @staticmethod
    def TexasHoldem() -> EncoderHandle: ...
    @staticmethod
    def KuhnPoker() -> EncoderHandle: ...
    @staticmethod
    def LeducPoker() -> EncoderHandle: ...
    @staticmethod
    def GridWorld() -> EncoderHandle: ...

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
    # self-play default Dirichlet(0.25, 0.3, "requester")). In simultaneous trees, "requester"
    # perturbs only the requesting player's root prior; "all" perturbs every player's root prior.
    @staticmethod
    def Dirichlet(epsilon: float = ..., alpha: float = ..., scope: str = ...) -> NoiseHandle: ...

class PolicyHandle:
    # Best-first selective expectimax (expand-once). Pass `chance=rf.chance_modes.Committed(samples=...)`
    # (default samples=1) or `chance=rf.chance_modes.ExpandAll()`. AlwaysResample is rejected because
    # an expand-once search has no traversal event to redraw on.
    @staticmethod
    def SelectiveExpectimax(
        expansion_budget: int = 64,
        top_k: int = 8,
        max_depth: int = 12,
        beta: float = 1.0,
        chance: ChanceModeHandle | None = None,
        n_heads: int = 1,
        epsilon: float = 0.0,
        opponent: str = "uniform",
        opp_temperature: float = 1.0,
        opp_floor: float = 0.0,
    ) -> PolicyHandle: ...
    @staticmethod
    def EpsilonGreedyQ(n_heads: int = 1, epsilon: float = 0.1) -> PolicyHandle: ...
    # MCTS (UCT) for compatible sequential, single-agent, and simultaneous (decoupled/DUCT
    # per-agent statistics) compositions. See the algorithm catalogue for its learner pairing.
    # act_by: "value" | "visits".
    # temperature > 0 (AlphaZero-style) samples the first temperature_drop plies of each episode
    # ∝ visits^(1/temperature) for training self-play diversity (None = whole episode); 0 = greedy.
    # `chance=` accepts AlwaysResample() (fresh draw per descent; unbiased default),
    # Committed(samples=...) (freeze that many draws per edge for depth on wide fans), or
    # ExpandAll() (evaluate every outcome; exact for narrow fans).
    @staticmethod
    def Mcts(
        num_simulations: int = 64,
        uct_c: float = 2.0,
        max_depth: int = 64,
        act_by: str = "value",
        temperature: float = 0.0,
        temperature_drop: int | None = None,
        chance: ChanceModeHandle | None = None,
    ) -> PolicyHandle: ...
    # AlphaZero (PUCT) for compatible sequential, single-agent, and simultaneous (DUCT)
    # compositions; see the algorithm catalogue for its learner pairing. `noise=` accepts an
    # rf.noise.Dirichlet(epsilon=..., alpha=..., scope="requester" | "all") handle or None. The infer
    # callback returns (policy_logits (N, width>=A), values (N,)); each array may be f32 or f64.
    # Columns after A are ignored, allowing a padded policy head without a device-side slice.
    # Root Dirichlet noise plus acting temperature drive self-play diversity;
    # acting is by visit count. temperature_drop=None applies the temperature to whole episodes.
    @staticmethod
    def AlphaZero(
        num_simulations: int = 64,
        c_puct: float = 1.5,
        max_depth: int = 64,
        temperature: float = 1.0,
        temperature_drop: int | None = 8,
        chance: ChanceModeHandle | None = None,
        noise: NoiseHandle | None = ...,
        sequential_backup: str = "auto",
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
    players: NDArray[np.int64]
    policy_targets: NDArray[np.float64]
    value_targets: NDArray[np.float64]
    policy_weights: NDArray[np.float64]
    # Legality of each row's state, HEAD frame (π's frame), CSR: row i's ids =
    # legal_ids[legal_offsets[i]:legal_offsets[i+1]] — the mask for legal-only policy losses
    # (softmax over legal actions; densify per minibatch:
    #   counts = np.diff(legal_offsets); rows = np.repeat(np.arange(M), counts)
    #   mask = np.zeros((M, A), bool); mask[rows, legal_ids] = True
    # then logits.masked_fill(~mask, torch.finfo(logits.dtype).min) before log_softmax —
    # finfo, not a constant: -2**16 overflows fp16). Empty rows = value-only
    # records. Named-access only; positional unpacking is unchanged.
    legal_ids: NDArray[np.int64]
    legal_offsets: NDArray[np.int64]
    telemetry: dict[str, Any]
    def __len__(self) -> int: ...
    def __getitem__(self, i: int) -> Any: ...
    def __iter__(self) -> Iterator[Any]: ...  # positional unpacking (runtime: sequence protocol)

class TreeStrapBatch:
    """`Engine.collect` result for the TreeStrap family. Also unpacks positionally as
    `obs, targets, masks, telemetry = batch`."""

    obs: NDArray[np.float32]
    players: NDArray[np.int64]
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
    players: NDArray[np.int64]
    actions: NDArray[np.int64]
    rewards: NDArray[np.float64]
    next_obs: NDArray[np.float32]
    dones: NDArray[np.bool_]
    # Authoritative TD bootstrap rule: false at terminals and no-successor truncation tails.
    can_bootstrap: NDArray[np.bool_]
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
        # Restrict training-record emission to these players (frozen opponents keep acting but
        # leave no records). Default: all players learn; records carry their player either way.
        learn_players: list[int] | None = ...,
        # Double-buffered collect (1 = off): 2 splits the games into two fixed groups whose
        # search rounds alternate, overlapping tree work with inference (the callback runs on a
        # submitter thread). Requires policies.Mcts/AlphaZero, a single shared callback, and no
        # truncation-tail bootstrapping. Deterministic per seed; digests differ from n_groups=1
        # (a different composition — the fingerprint records it). Size groups to keep the GPU
        # batch at its sweet spot: n_games=128 with n_groups=2 gives two 64-row groups.
        n_groups: int = ...,
    ) -> None: ...
    # Tell the engine the net's weights changed (call after every weight sync — e.g. right after
    # load_state_dict onto the collector net). Clears the infer cache at the next round boundary;
    # thread-safe, callable while a collect_stream is active; no-op when the cache is off.
    def weights_updated(self, player: int | None = ...) -> None: ...
    # The fully resolved immutable composition (defaults included), JSON-compatible;
    # rf.engine_from_config(engine.resolved_config()) reconstructs an equivalent engine.
    def resolved_config(self) -> dict[str, Any]: ...
    # 128-bit hex over reinfors-produced canonical bytes; compare, never recompute.
    def config_fingerprint(self) -> str: ...
    def snapshot(self, policy_version: str | None = ...) -> EngineSnapshot: ...
    def restore(self, snapshot: EngineSnapshot, expect_policy_version: str | None = ...) -> None: ...
    # The batch is learner-shaped: the TreeStrap family yields a `TreeStrapBatch`, the DQN family a
    # `DqnBatch`. Both expose named fields and also unpack positionally (back-compat with the old tuple).
    # infer outputs may be float64 or float32 (f32 widens exactly -> bit-identical batches;
    # f32 skips the producer-side conversion — the GPU fast path). The concrete batch family is
    # selected by the runtime learner handle, so this remains Any until Engine carries that
    # relationship as a generic type parameter.
    def collect(self, n_records: int, infer: Any) -> Any: ...
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
    "Already borrowed" and could not interrupt a blocked `next()` anyway). If the stream is dropped
    without `stop()`, the engine remains unavailable for the rest of the process: it cannot collect,
    snapshot, restore, or start another stream. Prefer the `with` form."""

    # Like Engine.collect, the concrete batch family is selected by the runtime learner handle.
    def next(self) -> Any: ...
    def pending(self) -> int: ...
    # Lossless checkpoint barrier: stops new collects, finishes the in-flight one with real
    # inference, returns every remaining batch; the engine (returned to its Engine) then matches
    # the delivered batches exactly — snapshot() right after is record-exact.
    def pause(self) -> list[Any]: ...
    def stop(self) -> None: ...
    def __iter__(self) -> CollectStream: ...
    def __next__(self) -> Any: ...
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
    # Trusted inspection state, not an agent observation: hidden cards are included. Never feed it
    # to a poker agent. Connect4's board is [row][column] with row 0 at the bottom.
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
    # The tick's ordered (agent, event) trace: every emission across the tick's edges (events are
    # per-edge and incremental — an edge emits only what it causally determines, so quiet ticks
    # return []). Event payloads are game-specific. Backgammon's `margin` is 1 plain, 2 gammon,
    # 3 backgammon; poker events are per-seat chip deltas.
    # The Env holds no reward; a game-aware caller reads the outcome from these.
    def step(self, actions: dict[int, int]) -> list[tuple[int, Any]]: ...
    # Per-agent scalar rewards for the most recent `step`, or None if built without a `reward` (the
    # reward-free play/eval default) or before the first `step`. The training-facing `reinfors.gym`
    # adapters read this so the event→reward mapping stays in Rust.
    @property
    def rewards(self) -> list[float] | None: ...
