"""Type stubs for the compiled `reinfors._reinfors` extension module.

Hand-maintained for now; once the API surface grows we can generate these from the Rust bindings.
"""

from typing import Any

Cell = tuple[int, int]
EventTuple = tuple[bool, bool, str | None, bool, bool, bool, bool]
StatsTuple = tuple[int, int, int, int]
SearchOutput = tuple[list[list[float]], StatsTuple]

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
) -> list[SearchOutput]: ...

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
    ) -> SearchOutput: ...

class Engine:
    def __init__(
        self,
        n_games: int,
        grid_size: int,
        initial_length: int,
        play_to_last: bool,
        win_food_lead: int | None,
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
        seed: int,
    ) -> None: ...
    # collect returns (obs [M, 5*g*g] float32, targets [M, K, 3] float64) numpy arrays.
    def collect(self, n_records: int, infer: Any) -> tuple[object, object]: ...
