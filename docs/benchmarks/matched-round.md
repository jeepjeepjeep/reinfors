# The matched round

The headline experiment: each stack trains chess from scratch for **2 hours of wall-clock**
on the whole machine (legs run sequentially, never sharing the box), at its own measured
operating point, under the full matched-knob protocol — then the two resulting models play
each other.

## Training throughput

| | OpenSpiel | reinfors |
|---|---|---|
| states collected (2h) | _TBD_ | _TBD_ |
| sustained states/s | _TBD_ | _TBD_ |
| learn steps (minibatch-equivalents) | _TBD_ | _TBD_ |
| gradient-samples per state (verified reuse) | _TBD_ | _TBD_ |
| final cache hit rate | _TBD_ | _TBD_ |

Learner telemetry panels (losses vs wall-clock and vs states collected, throughput
trajectories, cache-hit evolution, game-length trends) are rendered from both learners'
structured logs by one script; the loss curves are definitionally aligned but each is
measured on its own self-play distribution — they show per-system learning progress, not
head-to-head quality.

## Head-to-head

Protocol: the two final models play a match at 64 simulations per move on both sides,
their side running its own unmodified game runner (search solver disabled, so the match is
net-vs-net), our side driven through a bridge that mirrors the game state and submits moves
over its human-input interface. Openings are seeded uniform-random fixed lines, each played
once per color (paired scoring); artifacts are each side's last complete checkpoint before
the deadline kill; every game is exported as PGN with full run metadata.

| | result |
|---|---|
| games (opening pairs) | _TBD_ |
| W / D / L (reinfors perspective) | _TBD_ |
| score ± SE (paired) | _TBD_ |
| implied Elo difference (95% CI) | _TBD_ |

Interpretation guardrails, fixed in advance: with the throughput difference measured at the
scale it is, the expected strength edge after 2 hours is small relative to 50–100-game
match resolution, so a near-parity score is an expected outcome, not a null result; the
claim under test is "throughput preserved by the modular boundary", and the match's job is
to bound the strength cost of whatever differences remain, not to crown a winner.

## Strength over time

Both stacks checkpoint on comparable cadences throughout the round, so strength-vs-time
curves (each intermediate checkpoint against a fixed reference opponent, and/or
cross-stack at matched wall-clock) are recoverable from the same artifacts:

_TBD — planned; artifacts already collected per round._

## Open items

Carried caveats, stated rather than hidden:

- **single seed per side** — training-run variance in AlphaZero-style setups is real;
  multi-seed rounds are the planned fix and the reason match-level precision is not
  over-interpreted;
- match length: ~50 games resolves ~±0.05 in score; extended matches (fresh opening seeds
  pool cleanly with earlier games) are queued for tighter intervals;
- one game (chess) at one net size; a second, cheaper game (connect4) is planned as a
  calibration point where both stacks' dynamics differ substantially from chess.
