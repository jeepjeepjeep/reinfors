# Evaluate agents with Arena

`Arena` plays paired matches over concurrent game slots. Searched contestants use the same native
policies as training, with inference pooled across every game ready for that contestant. External
contestants run on worker lanes, allowing a subprocess engine to think while searched games continue.

Use `Arena` to evaluate deployed search configurations. Use [`Env`](evaluation.md#drive-an-environment-directly)
when you need a hand-written referee loop, raw network actions, or interactive play.

## Run the maintained example

The example pits AlphaZero search with a uniform placeholder network against a scripted Connect Four
agent. Replace `infer` with a checkpoint-backed callback for a real comparison:

```bash
python examples/eval_arena.py --games 20 --simulations 32 --slots 8
```

Games are played in pairs from the same opening with contestant seats swapped. The reported standard
error is therefore computed over opening pairs, not games.

```python
"""Run a paired Arena match between searched and external Connect Four agents."""

from __future__ import annotations

import argparse

import numpy as np
import reinfors as rf


class CenterBot:
    """A stateless external agent; subprocess-backed engines use the same contract."""

    def act(self, view: rf.arena.View) -> int:
        return min(view.legal_actions, key=lambda action: (abs(action - 3), action))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=8)
    parser.add_argument("--simulations", type=int, default=8)
    parser.add_argument("--slots", type=int, default=4)
    args = parser.parse_args()

    game = rf.games.Connect4()
    reward = rf.Reward(win=1.0, loss=-1.0)
    n_actions = game.action_space().n

    def infer(obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Replace these uniform outputs with the searched contestant's network."""
        rows = len(obs)
        logits = np.zeros((rows, n_actions), dtype=np.float32)
        values = np.zeros(rows, dtype=np.float32)
        return logits, values

    searched = (
        rf.policies.AlphaZero(
            num_simulations=args.simulations,
            temperature=0.0,
            noise=None,
        ),
        infer,
        1.0,  # gamma used when search backs up rewards
    )
    external = rf.arena.External(CenterBot, workers=args.slots, timeout=10.0)
    arena = rf.Arena(
        game,
        reward,
        contestants=[searched, external],
        n_slots=args.slots,
        start=rf.starts.RandomStartingMoves(2),
        seed=0,
    )

    result = arena.play(args.games)
    mean, stderr = result.payoff(0)
    score = mean / 2.0 + 0.5
    pairs = len(result.games) // 2
    pair_label = "pair" if pairs == 1 else "pairs"
    print(f"searched score={score:.3f} +/- {stderr / 2.0:.3f} over {pairs} opening {pair_label}")
    print(f"searched payoff by seat: {result.seat_payoffs(0)}")


if __name__ == "__main__":
    main()
```

## Contestants

A searched contestant is `(policy, infer, gamma)`. `Arena` calls `policy.choose` once for all ready
environments belonging to that contestant, so search leaves share the ordinary batched inference
contract. Configure evaluation behavior on the policy itself: for AlphaZero, fixed-strength play
normally uses a simulation budget, `temperature=0.0`, and `noise=None`.

`gamma` is the discount used while search backs up rewards. Match it to the model's training
configuration; use `1.0` for undiscounted terminal outcomes.

An external contestant wraps a per-game factory:

```python
external = rf.arena.External(factory, workers=8, timeout=30.0)
```

Each factory instance must implement `act(view) -> action`. `view` contains `obs`, `legal_actions`,
`agent`, and `ticks`. An instance may also implement:

- `on_action(action)`, called in order for every move in its game, including opening moves;
- `close()`, queued after pending notifications when the game finishes or cleanup begins.

This supports stateful stdin/stdout engines: launch one subprocess in the factory, submit positions or
moves from `on_action`, and read its next move in `act`. Exceptions, illegal actions, and timeouts fail
the match with game context.

Worker lanes are Python threads. They suit subprocess and other blocking I/O because those operations
release the GIL; CPU-bound Python agents do not gain multicore execution and should own worker
processes instead. A timeout bounds Arena's wait, but cannot forcibly interrupt arbitrary agent code.

`close()` is best-effort during an abort. Calls on one lane are serialized, so if `act()` or
`on_action()` blocks, the queued `close()` cannot run; after the abort grace period Arena stops waiting
and the daemon lane may remain blocked. A subprocess-backed agent must therefore provide out-of-band
cleanup rather than rely on `close()`: for example, register every child when it starts and terminate
its process group from the caller's own `finally`. `close()` remains the ordinary-completion cleanup
path.

## Concurrency and batching

`n_slots` limits live games. `External(..., workers=K)` separately limits live external-agent
instances; with one external seat per game, the smaller limit controls effective concurrency. Arena
dispatches external moves before searched batches so their computation can overlap.

Set `batch_delay` only after measuring. A small delay can admit external completions into a larger
searched batch, trading decision latency for accelerator utilization. Record `n_slots`, worker count,
search configuration, and device placement with evaluation results.

## Openings and results

`play(n_games)` requires an even number. `RandomStartingMoves(n)` draws one seeded legal opening per
pair, restores its snapshot for both games, and swaps the contestants' seats. Openings that finish the
game are resampled. If no live opening is found within `max_retries`, generation raises `ValueError`;
an opening length that games rarely survive should therefore fail loudly rather than hang. External
agents receive the opening through `on_action` before their first `act`.

`ArenaResult.games` is ordered by game id. Each `GameResult` contains its opening id, seat map,
contestant-ordered payoffs, length, and complete action trace. `result.payoff(i)` returns contestant
`i`'s mean reward and pair-level standard error; `result.seat_payoffs(i)` reports its mean by player
seat. The example's `(mean / 2) + 0.5` conversion is specific to win/loss rewards of `+1/-1`.

## Reproducibility and current limits

With deterministic inference, searched-only matches replay from the Arena seed. Matches containing an
external contestant are statistically seeded but not exactly replayable: completion timing changes
searched batch composition and RNG assignment. Accelerator kernels may add their own nondeterminism.

Arena currently supports exactly two contestants in two-agent sequential games, with at most one
external contestant. It does not produce Engine telemetry or training records; log game results,
resolved policy configurations, model identifiers, seeds, and the execution configuration in the
caller.
