"""Arena — reinfors-driven parallel evaluation: N games, pooled batched search.

Two contestants play ``n_games`` (an even number) over ``n_slots`` concurrent games.
Openings are generated once per game pair and both games restore the same snapshot with
seats swapped, so opening imbalance cancels out of the relative score. Searched
contestants decide through ``PolicyHandle.choose`` — one pooled search per contestant per
scheduler round, leaves batched into a single ``infer`` call. External contestants
(``External``) run their own engines on worker lanes; external moves are dispatched
before searched batches run, so external engines compute while the GPU searches.

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
from math import isfinite, sqrt
from time import monotonic, sleep
from typing import Any

from . import _reinfors

_MIX = 0x9E3779B97F4A7C15
_MASK = (1 << 64) - 1
_ABORT_GRACE = 5.0  # best-effort close budget when a run is already failing


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
    (called, in order, for every action executed in its game — opening moves included —
    feed these to stdin-driven engines) and ``close()``. At most ``workers`` agents are
    live at once; each worker lane runs its game's calls serially, so pipe I/O never
    stalls the Arena loop and per-game call order is preserved. ``timeout`` (seconds)
    is a total per-turn budget measured from dispatch — it covers queued ``on_action``
    notifications ahead of the ``act`` call as well as the call itself — and equally
    bounds end-of-game finalization (pending notifications plus ``close``), so a hung
    agent fails the run instead of hanging it.
    """

    def __init__(
        self,
        factory: Callable[[], Any],
        workers: int = 1,
        timeout: float | None = None,
    ) -> None:
        if workers < 1:
            raise ValueError("workers must be >= 1")
        if timeout is not None and not (isfinite(timeout) and timeout > 0):
            raise ValueError("timeout must be finite and positive")
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
    actions: list[int]  # the full game, opening moves included


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
    notifications: list[Future[Any]] = field(default_factory=list)


@dataclass
class _Finalizing:
    """A finished game whose lane is still working off notifications and close()."""

    game: _Game
    ci: int
    futures: list[Future[Any]]
    deadline: float | None


class Arena:
    """See the module docstring. ``contestants`` entries are either
    ``(policy, infer, gamma)`` — a searched seat, with gamma the discount the model was
    trained with — or an ``External`` (at most one in v1)."""

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
        if sum(isinstance(c, External) for c in contestants) > 1:
            raise ValueError("Arena v1 supports at most one External contestant")
        if n_slots < 1:
            raise ValueError("n_slots must be >= 1")
        if not (isfinite(batch_delay) and batch_delay >= 0.0):
            raise ValueError("batch_delay must be finite and >= 0")
        probe = _reinfors.Env(game, reward, seed=0)
        if probe.num_agents() != 2 or len(probe.active_agents()) != 1:
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
        active: list[_Game] = []
        finalizing: list[_Finalizing] = []
        try:
            return self._play(n_games, lanes, active, finalizing)
        finally:
            self._shutdown(active, finalizing)
            for lane_pool in lanes.values():
                for lane in lane_pool:
                    lane.stop()

    # scheduling ---------------------------------------------------------------

    def _play(
        self,
        n_games: int,
        lanes: dict[int, list[_Lane]],
        active: list[_Game],
        finalizing: list[_Finalizing],
    ) -> ArenaResult:
        openings: dict[int, tuple[Any, list[int]]] = {}
        free_lanes = {ci: list(pool) for ci, pool in lanes.items()}
        choose_calls = dict.fromkeys(range(len(self._contestants)), 0)
        next_game = 0
        finished: list[GameResult] = []

        while len(finished) < n_games or finalizing:
            self._poll_finalizing(finalizing, free_lanes)
            while next_game < n_games and len(active) < self._n_slots:
                game = self._start_game(next_game, openings, free_lanes)
                if game is None:
                    break  # external lanes exhausted; retry after a game finishes
                active.append(game)
                next_game += 1

            self._drain_external(active, finished, free_lanes, finalizing)
            # externals first: their engines compute while the GPU searches below
            self._dispatch_external(active)
            ready = self._searched_ready(active)
            if ready and self._batch_delay > 0 and self._external_pending(active):
                sleep(self._batch_delay)
                self._drain_external(active, finished, free_lanes, finalizing)
                self._dispatch_external(active)
                ready = self._searched_ready(active)
            if ready:
                for ci, games in sorted(ready.items()):
                    choose_calls[ci] += 1
                    self._choose_and_step(ci, choose_calls[ci], games, active, finished, finalizing)
                continue
            self._await_external(active, finalizing)

        finished.sort(key=lambda g: g.game_id)
        return ArenaResult(games=finished)

    def _start_game(
        self,
        game_id: int,
        openings: dict[int, tuple[Any, list[int]]],
        free_lanes: dict[int, list[_Lane]],
    ) -> _Game | None:
        opening_id = game_id // 2
        seats = (0, 1) if game_id % 2 == 0 else (1, 0)
        external = [ci for ci in free_lanes if ci in seats]
        for ci in external:
            if not free_lanes[ci]:
                return None
        env = _reinfors.Env(self._game, self._reward, seed=_mix64(self._seed, game_id, 1))
        opening_actions: list[int] = []
        if self._start is not None:
            if opening_id not in openings:
                openings[opening_id] = self._start.generate(self._game, self._reward, _mix64(self._seed, opening_id, 2))
            snapshot, opening_actions = openings[opening_id]
            env.restore(snapshot)
        game = _Game(
            game_id=game_id,
            opening_id=opening_id,
            seats=seats,
            env=env,
            payoffs=[0.0, 0.0],
            actions=list(opening_actions),
        )
        for ci in external:
            lane = free_lanes[ci].pop()
            try:
                game.agent = self._contestants[ci].factory()
            except Exception as e:
                free_lanes[ci].append(lane)
                raise RuntimeError(f"external contestant {ci} factory failed for game {game_id}: {e}") from e
            game.lane = lane
            # a stdin-driven engine must see the opening before its first act()
            if opening_actions and hasattr(game.agent, "on_action"):
                for action in opening_actions:
                    game.notifications.append(lane.submit(partial(game.agent.on_action, action)))
        return game

    def _mover(self, game: _Game) -> int:
        agents = game.env.active_agents()
        if len(agents) != 1:
            raise RuntimeError(f"game {game.game_id}: {len(agents)} active agents — Arena plays sequential games")
        seat: int = agents[0]
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
        call_no: int,
        games: list[_Game],
        active: list[_Game],
        finished: list[GameResult],
        finalizing: list[_Finalizing],
    ) -> None:
        policy, infer, gamma = self._contestants[ci]
        seed = _mix64(self._seed, ci, 3, call_no)
        actions = policy.choose([g.env for g in games], infer, seed=seed, gamma=gamma)
        for game, action in zip(games, actions, strict=True):
            self._step(game, action, active, finished, finalizing)

    def _dispatch_external(self, active: list[_Game]) -> None:
        for game in active:
            if game.pending is not None:
                continue
            ci = self._mover(game)
            if not isinstance(self._contestants[ci], External):
                continue
            self._check_notifications(game)
            env = game.env
            agent_seat = env.active_agents()[0]
            view = View(
                obs=env.observe(agent_seat),
                legal_actions=env.legal_actions(agent_seat),
                agent=agent_seat,
                ticks=env.ticks,
            )
            bot = game.agent
            assert game.lane is not None
            game.pending = game.lane.submit(partial(bot.act, view))
            game.pending_since = monotonic()

    def _await_external(self, active: list[_Game], finalizing: list[_Finalizing]) -> None:
        pending = [g.pending for g in active if g.pending is not None]
        for entry in finalizing:
            pending.extend(f for f in entry.futures if not f.done())
        if not pending:
            return
        timeouts = [c.timeout for c in self._contestants if isinstance(c, External) and c.timeout]
        wait(pending, timeout=min(timeouts) if timeouts else None, return_when=FIRST_COMPLETED)
        self._check_timeouts(active)

    def _check_timeouts(self, active: list[_Game]) -> None:
        now = monotonic()
        for game in active:
            if game.pending is None or game.pending.done():
                continue
            ci = self._mover(game)
            timeout = self._contestants[ci].timeout
            if timeout is not None and now - game.pending_since > timeout:
                raise TimeoutError(
                    f"external contestant {ci} exceeded its {timeout}s move timeout in game {game.game_id}"
                )

    def _check_notifications(self, game: _Game) -> None:
        remaining = []
        for future in game.notifications:
            if not future.done():
                remaining.append(future)
                continue
            error = future.exception()
            if error is not None:
                raise RuntimeError(f"external contestant on_action failed in game {game.game_id}: {error}") from error
        game.notifications = remaining

    def _drain_external(
        self,
        active: list[_Game],
        finished: list[GameResult],
        free_lanes: dict[int, list[_Lane]],
        finalizing: list[_Finalizing],
    ) -> None:
        for game in list(active):
            self._check_notifications(game)
            if game.pending is None or not game.pending.done():
                continue
            future, game.pending = game.pending, None
            try:
                action = future.result()
            except Exception as e:
                raise RuntimeError(f"external contestant {self._mover(game)} failed in game {game.game_id}: {e}") from e
            agent_seat = game.env.active_agents()[0]
            if action not in game.env.legal_actions(agent_seat):
                raise RuntimeError(f"external contestant chose illegal action {action} in game {game.game_id}")
            self._step(game, action, active, finished, finalizing)
        self._check_timeouts(active)

    def _step(
        self,
        game: _Game,
        action: int,
        active: list[_Game],
        finished: list[GameResult],
        finalizing: list[_Finalizing],
    ) -> None:
        agent_seat = game.env.active_agents()[0]
        game.env.step({agent_seat: action})
        game.actions.append(action)
        rewards = game.env.rewards
        if rewards is not None:
            for seat, r in enumerate(rewards):
                game.payoffs[seat] += r
        if game.agent is not None and hasattr(game.agent, "on_action"):
            assert game.lane is not None
            game.notifications.append(game.lane.submit(partial(game.agent.on_action, action)))
        if game.env.done():
            self._finish(game, active, finished, finalizing)

    def _finish(
        self,
        game: _Game,
        active: list[_Game],
        finished: list[GameResult],
        finalizing: list[_Finalizing],
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
        if game.agent is None:
            return
        # Nonblocking handoff: close is queued NOW (the lane runs it after the pending
        # notifications, even if one of them fails), and the scheduler reclaims the lane
        # once the lane works the queue off — completions never stall GPU batches.
        ci = self._external_contestant(game)
        timeout = self._contestants[ci].timeout
        futures = list(game.notifications)
        if hasattr(game.agent, "close"):
            assert game.lane is not None
            futures.append(game.lane.submit(game.agent.close))
        game.notifications = []
        finalizing.append(
            _Finalizing(
                game=game,
                ci=ci,
                futures=futures,
                deadline=None if timeout is None else monotonic() + timeout,
            )
        )

    def _poll_finalizing(self, finalizing: list[_Finalizing], free_lanes: dict[int, list[_Lane]]) -> None:
        for entry in list(finalizing):
            for future in entry.futures:
                if future.done() and future.exception() is not None:
                    error = future.exception()
                    raise RuntimeError(
                        f"external contestant {entry.ci} failed finishing game {entry.game.game_id}: {error}"
                    ) from error
            if all(f.done() for f in entry.futures):
                lane = entry.game.lane
                assert lane is not None
                entry.game.agent = None
                entry.game.lane = None
                free_lanes[entry.ci].append(lane)
                finalizing.remove(entry)
            elif entry.deadline is not None and monotonic() > entry.deadline:
                raise TimeoutError(
                    f"external contestant {entry.ci} exceeded its timeout finishing game {entry.game.game_id}"
                )

    def _external_contestant(self, game: _Game) -> int:
        for ci, c in enumerate(self._contestants):
            if isinstance(c, External) and ci in game.seats:
                return ci
        raise AssertionError("external bookkeeping on a game with no external contestant")

    def _shutdown(self, active: list[_Game], finalizing: list[_Finalizing]) -> None:
        """Best-effort agent cleanup on abort: serialized through each lane, bounded,
        and silent — the primary error is already propagating."""
        closing = []
        for game in active:
            if game.agent is not None and game.lane is not None:
                if hasattr(game.agent, "close"):
                    closing.append(game.lane.submit(game.agent.close))
                game.agent = None
                game.lane = None
        for entry in finalizing:
            # close is already queued on the lane; just give it the grace window
            closing.extend(f for f in entry.futures if not f.done())
            entry.game.agent = None
            entry.game.lane = None
        if closing:
            wait(closing, timeout=_ABORT_GRACE)
