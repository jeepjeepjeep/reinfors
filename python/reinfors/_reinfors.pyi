"""Type stubs for the compiled `reinfors._reinfors` extension module.

Hand-maintained for now; once the API surface grows we can generate these from the Rust bindings.
"""

from typing import Any

Cell = tuple[int, int]
EventTuple = tuple[bool, bool, str | None, bool, bool, bool, bool]
StatsTuple = tuple[int, int, int, int]
InteriorTarget = tuple[list[float], list[list[float]]]  # (obs [5*g*g], values [K][A])
SearchOutput = tuple[list[list[float]], list[InteriorTarget], StatsTuple]

def core_version() -> str: ...
def selective_search_many(
    envs: list[SnakeEnv],
    agents: list[int],
    gamma: float,
    beta: float,
    expansion_budget: int,
    top_k: int,
    max_depth: int,
    reward: tuple[float, float, float, float, float, float, float],
    opponent: str,
    opp_temperature: float,
    opp_floor: float,
    infer: Any,
    collect_interior: bool = ...,
    food_samples: int = ...,
    seed: int = ...,
) -> list[SearchOutput]: ...
def blend_outcome_targets(
    search_values: object,  # (T, K, A) float64
    actions: list[int],
    rewards: list[float],
    gamma: float,
    outcome_weight: float,
    tail: list[float],
) -> object: ...  # (T, K, A) float64

class SnakeEnv:
    def __init__(self, grid_size: int, initial_length: int, play_to_last: bool, win_food_lead: int | None) -> None: ...
    def set_food(self, cells: list[Cell]) -> None: ...
    def set_snakes(
        self, a_body: list[Cell], a_dir: int, a_alive: bool, b_body: list[Cell], b_dir: int, b_alive: bool
    ) -> None: ...
    def step(self, actions: tuple[int | None, int | None], spawns: list[Cell]) -> list[EventTuple]: ...
    def bodies(self) -> tuple[list[Cell], list[Cell]]: ...
    def directions(self) -> tuple[int, int]: ...
    def alive(self) -> tuple[bool, bool]: ...
    def food(self) -> list[Cell]: ...
    def is_done(self) -> bool: ...
    def obs(self, agent: int) -> object: ...  # numpy.ndarray[Any, dtype[float32]]
    def selective_search(
        self,
        agent: int,
        gamma: float,
        beta: float,
        expansion_budget: int,
        top_k: int,
        max_depth: int,
        reward: tuple[float, float, float, float, float, float, float],
        opponent: str,
        opp_temperature: float,
        opp_floor: float,
        infer: Any,
        collect_interior: bool = ...,
        food_samples: int = ...,
        seed: int = ...,
    ) -> SearchOutput: ...

class Engine:
    def __init__(
        self,
        n_games: int,
        grid_size: int,
        initial_length: int,
        play_to_last: bool,
        win_food_lead: int | None,
        initial_food_count: int,
        gamma: float,
        beta: float,
        expansion_budget: int,
        top_k: int,
        max_depth: int,
        reward: tuple[float, float, float, float, float, float, float],
        opponent: str,
        opp_temperature: float,
        opp_floor: float,
        epsilon: float,
        max_ticks: int,
        n_heads: int,
        outcome_weight: float,
        interior_targets: bool,
        bootstrap_p: float,
        seed: int,
        food_samples: int = ...,
    ) -> None: ...
    # collect returns (obs [M, 5*g*g] f32, targets [M, K, 3] f64, masks [M, K] f32) numpy arrays plus a
    # telemetry dict (finished-episode summaries + per-call search aggregates).
    def collect(self, n_records: int, infer: Any) -> tuple[Any, Any, Any, dict[str, Any]]: ...

class Connect4Engine:
    def __init__(
        self,
        win_reward: float,
        loss_reward: float,
        draw_reward: float,
        gamma: float,
        beta: float,
        expansion_budget: int,
        top_k: int,
        max_depth: int,
        opponent: str,
        opp_temperature: float,
        opp_floor: float,
        epsilon: float,
        max_ticks: int,
        n_heads: int,
        outcome_weight: float,
        interior_targets: bool,
        bootstrap_p: float,
        seed: int,
        n_games: int,
        food_samples: int = ...,
    ) -> None: ...
    # collect returns (obs [M, 2*6*7] f32, targets [M, K, 7] f64, masks [M, K] f32) + telemetry dict.
    def collect(self, n_records: int, infer: Any) -> tuple[Any, Any, Any, dict[str, Any]]: ...

class GridWorldEngine:
    def __init__(
        self,
        size: int,
        goal_row: int,
        goal_col: int,
        step_reward: float,
        goal_reward: float,
        gamma: float,
        beta: float,
        expansion_budget: int,
        top_k: int,
        max_depth: int,
        opponent: str,
        opp_temperature: float,
        opp_floor: float,
        epsilon: float,
        max_ticks: int,
        n_heads: int,
        outcome_weight: float,
        interior_targets: bool,
        bootstrap_p: float,
        seed: int,
        n_games: int,
        food_samples: int = ...,
    ) -> None: ...
    # collect returns (obs [M, 2*size*size] f32, targets [M, K, 4] f64, masks [M, K] f32) + telemetry.
    def collect(self, n_records: int, infer: Any) -> tuple[Any, Any, Any, dict[str, Any]]: ...

class DqnGridWorldEngine:
    def __init__(
        self,
        size: int,
        goal_row: int,
        goal_col: int,
        step_reward: float,
        goal_reward: float,
        epsilon: float,
        n_heads: int,
        bootstrap_p: float,
        max_ticks: int,
        seed: int,
        n_games: int,
    ) -> None: ...
    # collect returns off-policy transitions: (obs [M, dim] f32, actions [M] i64, rewards [M] f64,
    # next_obs [M, dim] f32, dones [M] bool, masks [M, K] f32) + telemetry.
    def collect(self, n_records: int, infer: Any) -> tuple[Any, Any, Any, Any, Any, Any, dict[str, Any]]: ...
