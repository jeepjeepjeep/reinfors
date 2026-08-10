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

Interpretation uses the pair-level standard error and confidence interval above; no result
is treated as more precise than the match supports.

## Open items

- **single seed per side** — training-run variance in AlphaZero-style setups is real, and
  match-level precision is not over-interpreted because of it;
- one game (chess) at one net size.
