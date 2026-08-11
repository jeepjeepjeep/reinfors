"""Arena — reinfors-driven parallel evaluation: N games, pooled batched search.

Two contestants play ``n_games`` (an even number) over ``n_slots`` concurrent games.
Openings are generated once per game pair and both games restore the same snapshot with
seats swapped, so opening imbalance cancels out of the relative score. Searched
contestants decide through ``PolicyHandle.choose`` — one pooled search per contestant per
scheduler round, leaves batched into a single ``infer`` call. External contestants
(``External``) run their own engines on worker lanes.

Determinism: a searched-only Arena run is exactly reproducible per master seed (its
scheduling and batch composition are deterministic). External-seat runs are
statistically seeded but not replayable — external completion timing changes search
batches and RNG assignment. Accelerator inference may independently weaken either
guarantee.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, wait
from dataclasses import dataclass, field
from functools import partial
from math import sqrt
from time import monotonic, sleep
from typing import Any

from . import _reinfors

_MIX = 0x9E3779B97F4A7C15
_MASK = (1 << 64) - 1


def _mix64(*parts: int) -> int:
    """SplitMix64 over a folded tuple: stable across processes, no PYTHONHASHSEED."""
    x = 0
    for p in parts:
        x = (x ^ (p & _MASK)) * _MIX & _MASK
    for _ in range(2):
        x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9 & _MASK
        x = (x ^ (x >> 27)) * 0x94D049BB133111EB & _MASK
        x ^= x >> 31
    return x


@dataclass(frozen=True)
class View:
    """What an external agent sees when asked to act."""

    obs: Any
    legal_actions: list[int]
    agent: int
    ticks: int


class External:
    """An externally-driven contestant: ``factory()`` builds one agent per game.

    An agent must provide ``act(view) -> int`` and may provide ``on_action(action)``
    (called, in order, for every action executed in its game — feed these to stdin-driven
    engines) and ``close()``. At most ``workers`` agents are live at once; each worker
    lane runs its game's calls serially, so pipe I/O never stalls the Arena loop and
    per-game call order is preserved. ``timeout`` (seconds) bounds a single ``act`` call;
    a hung agent fails the run instead of hanging it.
    """

    def __init__(
        self,
        factory: Callable[[], Any],
        workers: int = 1,
        timeout: float | None = None,
    ) -> None:
        if workers < 1:
            raise ValueError("workers must be >= 1")
        if timeout is not None and timeout <= 0:
            raise ValueError("timeout must be positive")
        self.factory = factory
        self.workers = workers
        self.timeout = timeout


class _Lane(threading.Thread):
    """One worker lane: owns at most one live agent, executes its calls in order."""

    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.calls: queue.Queue[tuple[Callable[[], Any], Future[Any]] | None] = queue.Queue()
        self.start()

    def run(self) -> None:
        while True:
            item = self.calls.get()
            if item is None:
                return
            fn, future = item
            if not future.set_running_or_notify_cancel():
                continue
            try:
                future.set_result(fn())
            except BaseException as e:
                future.set_exception(e)

    def submit(self, fn: Callable[[], Any]) -> Future[Any]:
        future: Future[Any] = Future()
        self.calls.put((fn, future))
        return future

    def stop(self) -> None:
        self.calls.put(None)


@dataclass
class GameResult:
    game_id: int
    opening_id: int
    seats: tuple[int, ...]  # seats[agent] = contestant index
    payoffs: tuple[float, ...]  # by contestant index
    length: int
    actions: list[int]


@dataclass
class ArenaResult:
    games: list[GameResult]
    n_contestants: int = 2

    def payoff(self, contestant: int) -> tuple[float, float]:
        """Pair-level mean payoff and stderr for a contestant.

        The two seat-swapped games of a pair share an opening and are correlated, so the
        estimate averages each pair first and takes the spread across pairs.
        """
        pairs: dict[int, list[float]] = {}
        for g in self.games:
            pairs.setdefault(g.opening_id, []).append(g.payoffs[contestant])
        scores = [sum(v) / len(v) for v in pairs.values()]
        n = len(scores)
        mean = sum(scores) / n
        if n < 2:
            return mean, float("inf")
        var = sum((s - mean) ** 2 for s in scores) / (n - 1)
        return mean, sqrt(var / n)

    def seat_payoffs(self, contestant: int) -> dict[int, float]:
        """Mean payoff split by the agent seat the contestant occupied."""
        by_seat: dict[int, list[float]] = {}
        for g in self.games:
            seat = g.seats.index(contestant)
            by_seat.setdefault(seat, []).append(g.payoffs[contestant])
        return {seat: sum(v) / len(v) for seat, v in by_seat.items()}


@dataclass
class _Game:
    game_id: int
    opening_id: int
    seats: tuple[int, ...]
    env: Any
    payoffs: list[float]  # by agent seat
    actions: list[int] = field(default_factory=list)
    agent: Any = None  # external contestant's per-game agent
    lane: _Lane | None = None
    pending: Future[Any] | None = None
    pending_since: float = 0.0


class Arena:
    """See the module docstring. ``contestants`` entries are either
    ``(policy, infer, gamma)`` — a searched seat, with gamma the discount the model was
    trained with — or an ``External``."""

    def __init__(
        self,
        game: Any,
        reward: Any,
        contestants: Sequence[Any],
        n_slots: int = 16,
        start: Any = None,
        seed: int = 0,
        batch_delay: float = 0.0,
    ) -> None:
        if len(contestants) != 2:
            raise ValueError("Arena v1 takes exactly two contestants")
        probe = _reinfors.Env(game, reward, seed=0)
        if probe.num_agents() != 2:
            raise ValueError("Arena v1 plays 2-agent sequential games")
        for c in contestants:
            if isinstance(c, External):
                continue
            if not (isinstance(c, tuple) and len(c) == 3):
                raise ValueError(
                    "a searched contestant is (policy, infer, gamma) — gamma is the discount the model was trained with"
                )
        self._game = game
        self._reward = reward
        self._contestants = list(contestants)
        self._n_slots = n_slots
        self._start = start
        self._seed = seed
        self._batch_delay = batch_delay

    def play(self, n_games: int) -> ArenaResult:
        if n_games < 2 or n_games % 2 != 0:
            raise ValueError("n_games must be even (openings are played from both seats)")
        lanes: dict[int, list[_Lane]] = {
            ci: [_Lane() for _ in range(c.workers)] for ci, c in enumerate(self._contestants) if isinstance(c, External)
        }
        try:
            return self._play(n_games, lanes)
        finally:
            for lane_pool in lanes.values():
                for lane in lane_pool:
                    lane.stop()

    # scheduling ---------------------------------------------------------------

    def _play(self, n_games: int, lanes: dict[int, list[_Lane]]) -> ArenaResult:
        openings: dict[int, Any] = {}
        free_lanes = {ci: list(pool) for ci, pool in lanes.items()}
        next_game = 0
        active: list[_Game] = []
        finished: list[GameResult] = []

        while len(finished) < n_games:
            while next_game < n_games and len(active) < self._n_slots:
                game = self._start_game(next_game, openings, free_lanes)
                if game is None:
                    break  # external lanes exhausted; retry after a game finishes
                active.append(game)
                next_game += 1

            self._drain_external(active, finished, free_lanes)
            ready = self._searched_ready(active)
            if ready and self._batch_delay > 0 and self._external_pending(active):
                sleep(self._batch_delay)
                self._drain_external(active, finished, free_lanes)
                ready = self._searched_ready(active)
            if ready:
                for ci, games in sorted(ready.items()):
                    self._choose_and_step(ci, games, active, finished, free_lanes)
                continue
            self._dispatch_external(active)
            self._await_external(active)

        finished.sort(key=lambda g: g.game_id)
        return ArenaResult(games=finished)

    def _start_game(
        self,
        game_id: int,
        openings: dict[int, Any],
        free_lanes: dict[int, list[_Lane]],
    ) -> _Game | None:
        opening_id = game_id // 2
        seats = (0, 1) if game_id % 2 == 0 else (1, 0)
        external = [ci for ci in free_lanes if ci in seats]
        for ci in external:
            if not free_lanes[ci]:
                return None
        env = _reinfors.Env(self._game, self._reward, seed=_mix64(self._seed, game_id, 1))
        if self._start is not None:
            if opening_id not in openings:
                openings[opening_id] = self._start.generate(self._game, self._reward, _mix64(self._seed, opening_id, 2))
            env.restore(openings[opening_id])
        game = _Game(
            game_id=game_id,
            opening_id=opening_id,
            seats=seats,
            env=env,
            payoffs=[0.0, 0.0],
        )
        for ci in external:
            game.lane = free_lanes[ci].pop()
            game.agent = self._contestants[ci].factory()
        return game

    def _mover(self, game: _Game) -> int:
        seat: int = game.env.active_agents()[0]
        return game.seats[seat]

    def _searched_ready(self, active: list[_Game]) -> dict[int, list[_Game]]:
        ready: dict[int, list[_Game]] = {}
        for game in active:
            if game.pending is not None:
                continue
            ci = self._mover(game)
            if not isinstance(self._contestants[ci], External):
                ready.setdefault(ci, []).append(game)
        for games in ready.values():
            games.sort(key=lambda g: g.game_id)
        return ready

    def _external_pending(self, active: list[_Game]) -> bool:
        return any(g.pending is not None for g in active)

    def _choose_and_step(
        self,
        ci: int,
        games: list[_Game],
        active: list[_Game],
        finished: list[GameResult],
        free_lanes: dict[int, list[_Lane]],
    ) -> None:
        policy, infer, gamma = self._contestants[ci]
        seed = _mix64(self._seed, ci, 3, len(finished), games[0].game_id)
        actions = policy.choose([g.env for g in games], infer, seed=seed, gamma=gamma)
        for game, action in zip(games, actions, strict=True):
            self._step(game, action, active, finished, free_lanes)

    def _dispatch_external(self, active: list[_Game]) -> None:
        for game in active:
            if game.pending is not None:
                continue
            ci = self._mover(game)
            if not isinstance(self._contestants[ci], External):
                continue
            env = game.env
            agent = game.env.active_agents()[0]
            view = View(
                obs=env.observe(agent),
                legal_actions=env.legal_actions(agent),
                agent=agent,
                ticks=env.ticks,
            )
            bot = game.agent
            assert game.lane is not None
            game.pending = game.lane.submit(partial(bot.act, view))
            game.pending_since = monotonic()

    def _await_external(self, active: list[_Game]) -> None:
        pending = [g.pending for g in active if g.pending is not None]
        if not pending:
            return
        timeouts = [c.timeout for c in self._contestants if isinstance(c, External) and c.timeout]
        wait(pending, timeout=min(timeouts) if timeouts else None, return_when=FIRST_COMPLETED)
        if all(f is not None and not f.done() for f in pending) and timeouts:
            self._check_timeouts(active)

    def _check_timeouts(self, active: list[_Game]) -> None:
        now = monotonic()
        for game in active:
            if game.pending is None or game.pending.done():
                continue
            ci = self._mover(game)
            timeout = self._contestants[ci].timeout
            if timeout is not None and now - game.pending_since > timeout:
                self._close_agents(active)
                raise TimeoutError(
                    f"external contestant {ci} exceeded its {timeout}s move timeout in game {game.game_id}"
                )

    def _drain_external(
        self,
        active: list[_Game],
        finished: list[GameResult],
        free_lanes: dict[int, list[_Lane]],
    ) -> None:
        for game in list(active):
            if game.pending is None or not game.pending.done():
                continue
            future, game.pending = game.pending, None
            try:
                action = future.result()
            except Exception as e:
                self._close_agents(active)
                raise RuntimeError(f"external contestant {self._mover(game)} failed in game {game.game_id}: {e}") from e
            agent = game.env.active_agents()[0]
            if action not in game.env.legal_actions(agent):
                self._close_agents(active)
                raise RuntimeError(f"external contestant chose illegal action {action} in game {game.game_id}")
            self._step(game, action, active, finished, free_lanes)
        self._check_timeouts(active)

    def _step(
        self,
        game: _Game,
        action: int,
        active: list[_Game],
        finished: list[GameResult],
        free_lanes: dict[int, list[_Lane]],
    ) -> None:
        agent = game.env.active_agents()[0]
        game.env.step({agent: action})
        game.actions.append(action)
        rewards = game.env.rewards
        if rewards is not None:
            for seat, r in enumerate(rewards):
                game.payoffs[seat] += r
        if game.agent is not None and hasattr(game.agent, "on_action"):
            bot = game.agent
            assert game.lane is not None
            game.lane.submit(partial(bot.on_action, action))
        if game.env.done():
            self._finish(game, active, finished, free_lanes)

    def _finish(
        self,
        game: _Game,
        active: list[_Game],
        finished: list[GameResult],
        free_lanes: dict[int, list[_Lane]],
    ) -> None:
        active.remove(game)
        payoffs_by_contestant = [0.0, 0.0]
        for seat, payoff in enumerate(game.payoffs):
            payoffs_by_contestant[game.seats[seat]] = payoff
        finished.append(
            GameResult(
                game_id=game.game_id,
                opening_id=game.opening_id,
                seats=game.seats,
                payoffs=tuple(payoffs_by_contestant),
                length=game.env.ticks,
                actions=game.actions,
            )
        )
        if game.agent is not None:
            bot, lane = game.agent, game.lane
            assert lane is not None
            if hasattr(bot, "close"):
                lane.submit(bot.close).result()
            ci = self._mover_contestant_for_lane(game)
            free_lanes[ci].append(lane)
            game.agent = None
            game.lane = None

    def _mover_contestant_for_lane(self, game: _Game) -> int:
        for ci, c in enumerate(self._contestants):
            if isinstance(c, External) and ci in game.seats:
                return ci
        raise AssertionError("lane held by a game with no external contestant")

    def _close_agents(self, active: list[_Game]) -> None:
        for game in active:
            if game.agent is not None and hasattr(game.agent, "close"):
                try:
                    game.agent.close()
                except Exception:
                    pass
