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

class Reward:
    # A generic named-weight reward; each game validates the keys it understands (snake: step/food/
    # loss/draw/kill/win/survival; connect4: win/loss/draw; gridworld: step/goal).
    def __init__(self, **weights: float) -> None: ...

# Opaque composition handles, built via the staticmethod constructors and passed to `Engine`.
class GameHandle:
    @staticmethod
    def Snake(
        grid_size: int = ...,
        initial_length: int = ...,
        food: int = ...,
        play_to_last: bool = ...,
        win_food_lead: int | None = ...,
        reward: Reward | None = ...,
    ) -> GameHandle: ...
    @staticmethod
    def Connect4(reward: Reward | None = ...) -> GameHandle: ...
    @staticmethod
    def GridWorld(
        size: int = ...,
        goal_row: int = ...,
        goal_col: int = ...,
        reward: Reward | None = ...,
    ) -> GameHandle: ...

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

class Engine:
    def __init__(
        self,
        game: GameHandle,
        policy: PolicyHandle,
        learner: LearnerHandle,
        n_games: int,
        max_ticks: int,
        seed: int = ...,
    ) -> None: ...
    # The record batch is learner-shaped: the TreeStrap family yields (obs, targets, masks, telemetry);
    # the DQN family yields (obs, actions, rewards, next_obs, dones, masks, telemetry).
    def collect(self, n_records: int, infer: Any) -> Any: ...
