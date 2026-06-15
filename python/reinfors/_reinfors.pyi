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
    ) -> None: ...
    # collect returns (obs [M, 5*g*g] f32, targets [M, K, 3] f64, masks [M, K] f32) numpy arrays.
    def collect(self, n_records: int, infer: Any) -> tuple[object, object, object]: ...
