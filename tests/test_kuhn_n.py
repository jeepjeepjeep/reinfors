"""N-player Kuhn: binding basics, and (dev-oracle, skipped without pyspiel) the whole
3-player tree walked side by side with pyspiel's kuhn_poker(players=3) — acting order, legal
sets, terminality, returns, and the information-set PARTITION (two states share a pyspiel
infoset string iff they share our key)."""

from typing import Any

import numpy as np
import pytest
import reinfors as rf


def test_construction_and_surfaces() -> None:
    g3 = rf.games.KuhnPoker(players=3)
    env = rf.Env(g3, seed=0)
    assert env.num_agents() == 3
    assert env.observation_space().shape == (9, 1, 1)
    with pytest.raises(ValueError, match="players"):
        rf.games.KuhnPoker(players=1)
    with pytest.raises(ValueError, match="players"):
        rf.games.KuhnPoker(players=11)
    # Default stays the historical 2-player game.
    assert rf.Env(rf.games.KuhnPoker(), seed=0).observation_space().shape == (6, 1, 1)


def test_env_plays_a_three_player_hand() -> None:
    env = rf.Env(rf.games.KuhnPoker(players=3), seed=3)
    acted = []
    while not env.done():
        (agent,) = env.active_agents()
        acted.append(agent)
        assert env.legal_actions(agent) == [0, 1]
        env.step({agent: 0})  # everyone passes
    assert acted == [0, 1, 2], "cyclic acting from player 0"


def test_engine_collects_three_player_kuhn() -> None:
    engine = rf.Engine(
        rf.games.KuhnPoker(players=3),
        rf.Reward(),
        rf.policies.EpsilonGreedyQ(n_heads=1, epsilon=0.2),
        rf.learners.Dqn(),
        n_games=4,
        seed=5,
    )
    batch = engine.collect(60, lambda obs: np.zeros((obs.shape[0], 1, 2)))
    assert set(batch.players.tolist()) == {0, 1, 2}
    assert engine.resolved_config()["game"]["players"] == 3
    rebuilt = rf.engine_from_config(engine.resolved_config())
    assert rebuilt.resolved_config()["game"]["players"] == 3


def test_engine_snapshot_round_trips_three_player_kuhn() -> None:
    """The engine codec must carry the 3-player game (a 2-player codec rejects 3-card deals
    at decode with "every card must be dealt")."""
    engine = rf.Engine(
        rf.games.KuhnPoker(players=3),
        rf.Reward(),
        rf.policies.EpsilonGreedyQ(n_heads=1, epsilon=0.2),
        rf.learners.Dqn(),
        n_games=4,
        seed=5,
        n_threads=1,
    )

    def infer(obs: np.ndarray) -> np.ndarray:
        return np.zeros((obs.shape[0], 1, 2))

    def sig(n: int) -> dict[str, bytes]:
        b = engine.collect(n, infer)
        arrays = ((k, getattr(b, k, None)) for k in dir(b))
        return {k: np.ascontiguousarray(v).tobytes() for k, v in arrays if isinstance(v, np.ndarray)}

    sig(30)  # advance mid-hand so live states go through the codec
    snap = engine.snapshot()
    ahead = sig(40)
    engine.restore(snap)
    assert sig(40) == ahead


def _env_at_deal(cards: list[int]) -> "rf.Env":
    """Our env realized at the exact target deal (24 deals; reseeding is cheap)."""
    for seed in range(5000):
        env = rf.Env(rf.games.KuhnPoker(players=3), seed=seed)
        if [int(c) for c in env.state()["cards"]] == cards:
            return env
    raise AssertionError(f"deal {cards} not reached by reseeding")


def test_full_tree_parity_with_pyspiel_three_player() -> None:
    pyspiel = pytest.importorskip("pyspiel")
    import itertools

    ps = pyspiel.load_game("kuhn_poker(players=3)")
    partition: dict[str, bytes] = {}
    reverse: dict[bytes, str] = {}
    terminals = 0

    def walk(node: Any, env: "rf.Env") -> None:
        nonlocal terminals
        if node.is_terminal():
            terminals += 1
            assert env.done(), "terminality must agree"
            return
        actor = node.current_player()
        assert env.active_agents() == [actor], "acting order must agree"
        assert env.legal_actions(actor) == node.legal_actions()
        key = bytes(env.information_state_key(actor))
        info = node.information_state_string()
        if info in partition:
            assert partition[info] == key, f"partition split at {info!r}"
        else:
            assert key not in reverse, f"partition merge: {info!r} vs {reverse[key]!r}"
            partition[info] = key
            reverse[key] = info
        for a in node.legal_actions():
            snap = env.snapshot()
            env.step({actor: a})
            child = node.child(a)
            if child.is_terminal():
                ours = [0.0, 0.0, 0.0]
                # Replay the whole line to accumulate every emission (single terminal edge
                # here, but fold the trace properly anyway).
                # The step above already consumed it; recompute via a fresh replay:
                env2 = _env_at_deal([int(c) for c in node.history()[:3]])
                for h in [int(x) for x in child.history()[3:]]:
                    (agent,) = env2.active_agents()
                    for who, ev in env2.step({agent: h}):
                        ours[who] += ev
                assert ours == list(child.returns()), f"returns diverge on {child.history()}"
            walk(child, env)
            env.restore(snap)

    for deal in itertools.permutations(range(4), 3):
        node = ps.new_initial_state()
        for c in deal:
            node.apply_action(c)
        walk(node, _env_at_deal(list(deal)))

    assert terminals == 24 * 13, "24 deals x 13 terminal betting lines each"
    assert len(partition) == len(reverse)
