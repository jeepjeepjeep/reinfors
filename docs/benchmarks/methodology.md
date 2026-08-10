# Methodology

The measurement discipline shared by every benchmark family. Rules specific to comparing
two frameworks live in the [comparison protocol](openspiel/protocol.md).

## Termination: hard kill, interior windows

Each leg is killed (SIGKILL) at its deadline, on both stacks. Rates use counter deltas
between timestamped interior samples; runs with fewer than two interior samples fail
rather than fall back to totals. This avoids counting work completed while draining after
the deadline, which inflated one side by 18–31% in an early sweep.

## The training-relevant rate is states/s, not rows/s

A *row* is one position forwarded through the net: search effort. A *state* is one training
example delivered to the learner — and in AlphaZero-style training a position only becomes
an example when its game **finishes** (the value target is the realized outcome). Under a
hard deadline, in-flight games count for nothing. Configurations routinely trade these
against each other (more parallel games → more rows/s, slower per-game progress → fewer
completed states/s), so topology selection and headline comparisons use states/s at matched
search budget; rows/s is recorded as diagnosis.

## Determinism, repeats, and uncertainty

Collection is seeded and bit-reproducible per configuration under fixed weights and
deterministic inference; every run records its seed and resolved configuration.

Repeats are tiered by cost: tuning and characterization measurements (kernel curves,
sweep legs) are repeated — at least three legs, reported as median with the spread — while
the two-hour rounds are explicitly **single-seed per side**, stated as such wherever their
results appear. Match results report a standard error (computed over opening pairs);
sweep-derived selections state the margin between the chosen configuration and the
runner-up. No published number appears without either a repeat-derived spread or a
single-run label.
