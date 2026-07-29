"""The non-snake games (Connect-4, GridWorld) and the DQN family driven through the unified `Engine`:
each composition collects records of the right shape, telemetry carries one reward per agent, and the
name registries resolve. These are engine-contract tests: a dummy numpy `infer` (zeros of the right
K/A) keeps them torch-free — the model and gradient step live in the consumer, not reinfors.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
import reinfors as rf

_K = 2
_TELEMETRY_KEYS = {
    "episodes",
    "decisions",
    "max_depth",
    "mean_leaves",
    "mean_rounds",
    "mean_expansions",
    "mean_sigma",
    "mean_disagreement",
}


def _selective() -> rf._reinfors.PolicyHandle:
    return rf.policies.SelectiveExpectimax(
        expansion_budget=24,
        top_k=4,
        max_depth=6,
        beta=1.0,
        chance=rf.chance_modes.Committed(samples=1),
        n_heads=_K,
        epsilon=0.0,
        opponent="uniform",
        opp_temperature=1.0,
        opp_floor=0.1,
    )


def _treestrap() -> rf._reinfors.LearnerHandle:
    return rf.learners.TreeStrap(gamma=0.99, outcome_weight=0.5, bootstrap_p=1.0, interior_targets=False)


def _connect4_engine() -> rf.Engine:
    # Connect-4 is bounded (it always terminates), so it needs no truncation horizon.
    return rf.Engine(
        rf.games.Connect4(),
        rf.Reward(win=1.0, loss=-1.0, draw=0.0),
        _selective(),
        _treestrap(),
        n_games=2,
        seed=0,
    )


def _gridworld_engine() -> rf.Engine:
    return rf.Engine(
        rf.games.GridWorld(size=5, goal_row=0, goal_col=1, max_ticks=30),
        rf.Reward(step=0.0, goal=1.0),
        _selective(),
        _treestrap(),
        n_games=2,
        seed=0,
    )


def _dqn_engine() -> rf.Engine:
    return rf.Engine(
        rf.games.GridWorld(size=5, goal_row=0, goal_col=1, max_ticks=10),
        rf.Reward(step=0.0, goal=1.0),
        rf.policies.EpsilonGreedyQ(n_heads=_K, epsilon=0.1),
        rf.learners.Dqn(bootstrap_p=1.0),
        n_games=3,
        seed=0,
    )


def _dummy_infer(a: int) -> Callable[[np.ndarray], np.ndarray]:
    def infer(arr: np.ndarray) -> np.ndarray:
        return np.zeros((arr.shape[0], _K, a), dtype=np.float64)

    return infer


@pytest.mark.parametrize(
    ("make_engine", "action_count"),
    [(_connect4_engine, 7), (_gridworld_engine, 4)],
)
def test_collect_shapes_and_telemetry(make_engine: Callable[[], rf.Engine], action_count: int) -> None:
    engine = make_engine()
    obs, tgt, mask, telemetry = engine.collect(40, _dummy_infer(action_count))
    m = obs.shape[0]
    assert m >= 40
    assert tgt.shape == (m, _K, action_count) and tgt.dtype == np.float64
    assert mask.shape == (m, _K)
    assert set(telemetry) >= _TELEMETRY_KEYS
    assert telemetry["decisions"] > 0


@pytest.mark.parametrize(
    ("make_engine", "action_count", "num_agents"),
    [(_connect4_engine, 7, 2), (_gridworld_engine, 4, 1)],
)
def test_episode_reward_is_per_agent(make_engine: Callable[[], rf.Engine], action_count: int, num_agents: int) -> None:
    # The engine generalizes to any agent count: each finished episode's telemetry carries one reward
    # per agent — length 1 for single-agent GridWorld, 2 for Connect-4 — not a hardcoded pair.
    engine = make_engine()
    _, _, _, telemetry = engine.collect(150, _dummy_infer(action_count))
    episodes = telemetry["episodes"]
    assert len(episodes) > 0
    for rewards, length, seeded in episodes:
        assert len(rewards) == num_agents
        assert length >= 1 and all(np.isfinite(r) for r in rewards)
        assert seeded is False  # no start buffer configured -> every episode is a fresh start


def test_start_buffer_is_off_by_default_snake_only_and_tags_seeded() -> None:
    # The reached-state start buffer is off by default and snake-only in v1. Enabled with p_fresh=0 and
    # a short horizon, it fills and then seeds episodes from reached states (tagged in telemetry).
    engine = rf.Engine(
        rf.games.Snake(grid_size=8, max_ticks=5),
        rf.Reward(food=1.0, loss=-1.0),
        _selective(),
        _treestrap(),
        n_games=4,
        start_buffer=True,
        start_buffer_capacity=64,
        p_fresh=0.0,
    )
    seeded_any = False
    for _ in range(4):
        _, _, _, telemetry = engine.collect(200, _dummy_infer(3))
        seeded_any |= any(seeded for _, _, seeded in telemetry["episodes"])
    assert seeded_any, "p_fresh=0 should seed some episodes once the buffer fills"
    # No cell key for non-snake games -> enabling the buffer is rejected.
    with pytest.raises(ValueError, match="only supported for the snake"):
        rf.Engine(
            rf.games.Connect4(),
            rf.Reward(win=1.0),
            _selective(),
            _treestrap(),
            n_games=2,
            start_buffer=True,
        )


def test_registries_list_the_built_in_names() -> None:
    assert rf.registered_games() == ["backgammon", "chess", "connect4", "gridworld", "snake", "texas_holdem"]
    assert rf.registered_policies() == ["alphazero", "epsilon_greedy_q", "mcts", "selective_expectimax"]
    assert rf.registered_learners() == ["alphazero", "dqn", "treestrap"]


def test_game_handles_advertise_spaces() -> None:
    # A handle reports the observation Box (whose shape sizes the network input) and action Discrete,
    # so a network can be sized from the game instead of hard-coding its dimensions.
    obs = rf.games.Snake(grid_size=12).observation_space()
    assert isinstance(obs, rf.spaces.Box) and obs.shape == (5, 12, 12)
    assert obs.low.shape == obs.shape == obs.high.shape  # bounds broadcast to the obs shape
    assert (obs.low == 0.0).all() and (obs.high == 1.0).all()  # one-hot planes -> [0, 1]
    act = rf.games.Snake(grid_size=12).action_space()
    assert isinstance(act, rf.spaces.Discrete) and act.n == 3
    # The non-snake games advertise their own shapes (mirrors the Rust `spaces` test).
    assert rf.games.Connect4().observation_space().shape == (2, 6, 7)
    assert rf.games.Connect4().action_space().n == 7
    assert rf.games.GridWorld(size=5).observation_space().shape == (2, 5, 5)
    assert rf.games.GridWorld(size=5).action_space().n == 4


def test_loop_prone_games_default_to_a_finite_truncation_horizon() -> None:
    # A game that can loop forever (snake circling, gridworld never reaching the goal) MUST default to a
    # finite horizon, else Engine.collect would spin on a non-terminating episode. Connect-4 always ends
    # on its own, so it truncates never. `max_ticks=None` is the explicit opt-in to "never truncate".
    assert rf.games.Snake().truncation_horizon() == 1000
    assert rf.games.GridWorld().truncation_horizon() == 1000
    assert rf.games.Connect4().truncation_horizon() is None
    assert rf.games.Snake(max_ticks=None).truncation_horizon() is None  # explicit opt-out
    assert rf.games.Snake(max_ticks=250).truncation_horizon() == 250


def test_make_constructs_and_rejects_unknown() -> None:
    # The name-addressable path builds the same handles the typed constructors do.
    engine = rf.Engine(
        rf.make_game("connect4"),
        rf.Reward(win=1.0, loss=-1.0, draw=0.0),
        rf.make_policy("selective_expectimax", n_heads=_K),
        rf.make_learner("treestrap"),
        n_games=2,
    )
    _, _, _, telemetry = engine.collect(20, _dummy_infer(7))
    assert telemetry["decisions"] > 0
    with pytest.raises(KeyError):
        rf.make_game("pong")
    with pytest.raises(KeyError):
        rf.make_policy("a2c")


def test_engine_from_config_round_trips_a_yaml_shaped_dict() -> None:
    # A config shaped like parsed YAML — a nested `reward` mapping, not a pre-built handle — builds a
    # working engine: engine_from_config wraps the reward dict into rf.Reward automatically.
    config = {
        "game": {"name": "snake", "grid_size": 8, "reward": {"food": 1.0, "loss": -10.0}},
        "policy": {"name": "selective_expectimax", "n_heads": _K, "expansion_budget": 16, "max_depth": 6},
        "learner": {"name": "treestrap", "gamma": 0.99},
        "engine": {"n_games": 2, "max_ticks": 10, "seed": 0},
    }
    engine = rf.engine_from_config(config)
    _, _, _, telemetry = engine.collect(20, _dummy_infer(3))
    assert telemetry["decisions"] > 0
    # Reward validation still fires through the config path (the wrapped dict isn't trusted blindly).
    bad = {**config, "game": {"name": "snake", "grid_size": 8, "reward": {"goal": 1.0}}}
    with pytest.raises(ValueError, match="unknown reward key"):
        rf.engine_from_config(bad)


def test_reward_rejects_keys_not_valid_for_the_game() -> None:
    # The generic Reward is validated per game at Engine construction (the reward is decoupled from the
    # game now): any key the game doesn't define is an error (not silently ignored). Valid keys work.
    def engine(game: rf._reinfors.GameHandle, reward: rf.Reward) -> rf.Engine:
        return rf.Engine(game, reward, _selective(), _treestrap(), n_games=1)

    engine(rf.games.Snake(grid_size=8), rf.Reward(food=1.0, loss=-10.0))  # snake keys: ok
    engine(rf.games.Connect4(), rf.Reward(win=1.0))  # connect4 keys: ok
    with pytest.raises(ValueError, match="unknown reward key"):
        engine(rf.games.Snake(grid_size=8), rf.Reward(goal=1.0))  # 'goal' is gridworld's, not snake's
    with pytest.raises(ValueError, match="unknown reward key"):
        engine(rf.games.Connect4(), rf.Reward(food=1.0))  # 'food' is snake's, not connect4's


def test_incompatible_policy_learner_pairing_is_rejected() -> None:
    # A search learner with a Q policy (mismatched evaluation type) must fail at Engine construction.
    with pytest.raises(ValueError):
        rf.Engine(
            rf.games.GridWorld(size=5),
            rf.Reward(goal=1.0),
            rf.policies.EpsilonGreedyQ(n_heads=_K),
            rf.learners.TreeStrap(),
            n_games=1,
        )


def _mcts_infer(action_count: int) -> Callable[[np.ndarray], np.ndarray]:
    # MCTS is single-head (n_heads=1), so its net returns (N, 1, A).
    def infer(arr: np.ndarray) -> np.ndarray:
        return np.zeros((arr.shape[0], 1, action_count), dtype=np.float64)

    return infer


@pytest.mark.parametrize(
    ("game", "action_count"),
    [
        (rf.games.Connect4(), 7),
        (rf.games.GridWorld(size=5, goal_row=0, goal_col=1, max_ticks=20), 4),
    ],
)
def test_mcts_pairs_with_treestrap_on_sequential_games(game: rf._reinfors.GameHandle, action_count: int) -> None:
    # The MCTS planner drives sequential / single-agent games through TreeStrap: single-head targets.
    reward = rf.Reward(win=1.0, loss=-1.0) if action_count == 7 else rf.Reward(goal=1.0)
    engine = rf.Engine(
        game,
        reward,
        rf.policies.Mcts(num_simulations=24, act_by="visits"),
        rf.learners.TreeStrap(),
        n_games=2,
        seed=0,
    )
    obs, tgt, mask, telemetry = engine.collect(30, _mcts_infer(action_count))
    m = obs.shape[0]
    assert m >= 30
    assert tgt.shape == (m, 1, action_count)  # MCTS is single-head
    assert mask.shape == (m, 1)
    assert telemetry["decisions"] > 0


def test_mcts_collects_on_simultaneous_snake() -> None:
    # DUCT: UCT+TreeStrap now pairs with snake (simultaneous + declared chance).
    def infer(arr: np.ndarray) -> np.ndarray:
        return np.zeros((arr.shape[0], 1, 3))

    engine = rf.Engine(
        rf.games.Snake(grid_size=8, max_ticks=30),
        rf.Reward(food=1.0, loss=-1.0),
        rf.policies.Mcts(num_simulations=8, chance=rf.chance_modes.Committed(samples=2)),
        rf.learners.TreeStrap(),
        n_games=2,
        seed=0,
    )
    obs, _targets, _masks, telemetry = engine.collect(40, infer)
    assert obs.shape[0] >= 40
    assert telemetry["decisions"] > 0


def _mcts_engine(temperature: float, seed: int = 0, drop: int | None = None) -> rf.Engine:
    return rf.Engine(
        rf.games.Connect4(),
        rf.Reward(win=1.0, loss=-1.0),
        rf.policies.Mcts(num_simulations=16, temperature=temperature, temperature_drop=drop),
        rf.learners.TreeStrap(),
        n_games=1,
        seed=seed,
    )


def test_mcts_temperature_zero_replays_the_same_game() -> None:
    # Greedy acting from a fixed start with a fixed net is one deterministic game replayed forever —
    # the no-self-play-diversity failure mode the temperature exists to fix.
    _, _, _, tel = _mcts_engine(0.0).collect(120, _mcts_infer(7))
    lengths = [length for _r, length, _s in tel["episodes"]]
    assert len(lengths) >= 3 and len(set(lengths)) == 1


def test_mcts_temperature_diversifies_self_play_and_stays_seeded() -> None:
    # temperature > 0 samples opening plies from the seeded acting RNG: episodes within a collect now
    # differ (diversity), while the same seed still reproduces the identical collect (determinism).
    _, _, _, t1 = _mcts_engine(1.0).collect(120, _mcts_infer(7))
    games1 = [(tuple(r), length) for r, length, _s in t1["episodes"]]
    assert len(set(games1)) > 1
    _, _, _, t2 = _mcts_engine(1.0).collect(120, _mcts_infer(7))
    assert games1 == [(tuple(r), length) for r, length, _s in t2["episodes"]]


def test_mcts_temperature_drop_zero_is_greedy() -> None:
    # drop=0 means no plies are sampled — identical behavior to temperature 0.
    _, _, _, tel = _mcts_engine(1.0, drop=0).collect(120, _mcts_infer(7))
    lengths = [length for _r, length, _s in tel["episodes"]]
    assert len(set(lengths)) == 1


def test_mcts_rejects_bad_temperature() -> None:
    with pytest.raises(ValueError, match="temperature"):
        _mcts_engine(-0.5)


def test_mcts_rejects_unknown_act_by() -> None:
    with pytest.raises(ValueError, match="act_by"):
        rf.policies.Mcts(act_by="greedy")


@pytest.mark.parametrize(
    "bad",
    [
        {"expansion_budget": 0},
        {"top_k": 0},
        {"max_depth": 0},
        {"beta": 1.5},
    ],
)
def test_engine_rejects_degenerate_search_params(bad: dict[str, float]) -> None:
    # SelectiveExpectimax search knobs are validated at Engine construction (the core does not).
    kw: dict[str, Any] = {
        "expansion_budget": 24,
        "top_k": 4,
        "max_depth": 6,
        "beta": 1.0,
        "chance": rf.chance_modes.Committed(samples=1),
        "n_heads": _K,
        **bad,
    }
    with pytest.raises(ValueError):
        rf.Engine(
            rf.games.Snake(grid_size=8),
            rf.Reward(food=1.0),
            rf.policies.SelectiveExpectimax(**kw),
            rf.learners.TreeStrap(),
            n_games=1,
        )


# --- Model-free DQN: a second algorithm through the same unified engine, with a different record shape
# (off-policy transitions instead of TreeStrap targets) — the seam + binding generalization. ---


def test_dqn_engine_emits_well_formed_transitions() -> None:
    engine = _dqn_engine()
    dim = 2 * 5 * 5
    obs, actions, rewards, next_obs, dones, masks, telemetry = engine.collect(60, _dummy_infer(4))
    m = obs.shape[0]
    assert m >= 60
    assert obs.shape == (m, dim) and obs.dtype == np.float32
    assert actions.shape == (m,) and actions.dtype == np.int64
    assert rewards.shape == (m,) and rewards.dtype == np.float64
    assert next_obs.shape == (m, dim) and next_obs.dtype == np.float32
    assert dones.shape == (m,) and dones.dtype == bool
    assert masks.shape == (m, _K) and np.isin(masks, (0.0, 1.0)).all()
    assert (actions >= 0).all() and (actions < 4).all()
    assert "episodes" in telemetry and telemetry["decisions"] > 0


def test_dqn_transitions_drive_a_td_step() -> None:
    # The transition record is usable in a gradient update end to end (finite loss) — the seam's *other*
    # record shape + loss, which the TreeStrap example can't cover. Uses a tiny inline net (reinfors
    # ships none) as its own TD target: enough to prove the records train, not a full DQN.
    pytest.importorskip("torch")
    import torch

    class TinyQ(torch.nn.Module):
        def __init__(self, dim: int, n_heads: int, n_actions: int) -> None:
            super().__init__()
            self.n_heads, self.n_actions = n_heads, n_actions
            self.fc = torch.nn.Linear(dim, n_heads * n_actions)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out: torch.Tensor = self.fc(x).view(-1, self.n_heads, self.n_actions)
            return out

    dim, n_actions = 2 * 5 * 5, 4
    batch = _dqn_engine().collect(64, _dummy_infer(n_actions))
    obs, actions, rewards, next_obs, _dones, masks, _ = batch
    net = TinyQ(dim, _K, n_actions)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)

    q = net(torch.from_numpy(obs))  # (M, K, A)
    a = torch.from_numpy(actions).long()
    r = torch.from_numpy(rewards).float()
    mask = torch.from_numpy(masks).float()
    # The documented masked target (the reference pattern — NOT (1 - done) * max, which meets the
    # masked max's -inf as NaN): densify the CSR legality per minibatch, bootstrap iff non-empty.
    m = obs.shape[0]
    counts = np.diff(batch.next_legal_offsets)  # type: ignore[union-attr]
    rows = np.repeat(np.arange(m), counts)
    dense = np.zeros((m, n_actions), dtype=bool)
    dense[rows, batch.next_legal_ids] = True  # type: ignore[union-attr]
    next_legal = torch.from_numpy(dense)
    with torch.no_grad():
        q_next = net(torch.from_numpy(next_obs))  # (M, K, A)
        masked = q_next.masked_fill(~next_legal[:, None, :], float("-inf")).max(dim=-1).values
        target = r[:, None] + 0.99 * torch.where(torch.isfinite(masked), masked, 0.0)  # (M, K)
    chosen = q.gather(-1, a[:, None, None].expand(-1, _K, 1)).squeeze(-1)  # (M, K)
    loss = (mask * (chosen - target) ** 2).sum() / mask.sum().clamp(min=1.0)
    opt.zero_grad()
    loss.backward()
    opt.step()
    assert torch.isfinite(loss)


def test_expectimax_rejects_per_traversal_chance_mode() -> None:
    # Expand-once search: no traversal event to redraw on — the paradigm property, not a mode list.
    with pytest.raises(ValueError, match="per-traversal"):
        rf.policies.SelectiveExpectimax(chance=rf.chance_modes.AlwaysResample())


def test_expectimax_expand_all_runs_on_snake() -> None:
    # The exact fan on a real declared distribution (snake respawn) — small grid keeps it cheap.
    def infer(arr: np.ndarray) -> np.ndarray:
        return np.zeros((arr.shape[0], 1, 3))

    engine = rf.Engine(
        rf.games.Snake(grid_size=6, max_ticks=30),
        rf.Reward(food=1.0, loss=-1.0),
        rf.policies.SelectiveExpectimax(expansion_budget=12, top_k=2, chance=rf.chance_modes.ExpandAll()),
        rf.learners.TreeStrap(),
        n_games=2,
        seed=0,
    )
    obs, _targets, _masks, telemetry = engine.collect(40, infer)
    assert obs.shape[0] >= 40
    assert telemetry["decisions"] > 0
