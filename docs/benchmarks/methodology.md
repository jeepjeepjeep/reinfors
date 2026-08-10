# Methodology

The measurement discipline shared by every benchmark family — internal grids and the
cross-framework comparison alike. Several rules exist because an earlier version of
the benchmark violated them and produced numbers that flattered one configuration;
each states what it prevents. Rules specific to comparing *two frameworks* fairly
live with the [OpenSpiel comparison protocol](openspiel/protocol.md).

## Termination: hard kill, interior windows

Every timed leg is terminated by SIGKILL at its deadline, on **both** stacks, and rates are
computed from counter deltas between timestamped samples strictly inside the window — never
by dividing a total by the nominal duration.

*Why:* graceful shutdown lets a stack keep producing while it drains in-flight work. In an
early sweep, one side was stopped with SIGINT plus a grace period and internally-timed
totals, inflating its rates by 18–31% relative to the hard-killed side. Discovered when its
single inference thread logged more busy-seconds than the leg's nominal length. If interior
samples are missing, the result is reported as failed — never reconstructed.

## The training-relevant rate is states/s, not rows/s

A *row* is one position forwarded through the net: search effort. A *state* is one training
example delivered to the learner — and in AlphaZero-style training a position only becomes
an example when its game **finishes** (the value target is the realized outcome). Under a
hard deadline, in-flight games count for nothing. Configurations routinely trade these
against each other (more parallel games → more rows/s, slower per-game progress → fewer
completed states/s), so topology selection and headline comparisons use states/s at matched
search budget; rows/s is recorded as diagnosis.

## Determinism and repeats

Collection is seeded and bit-reproducible per configuration under fixed weights and
deterministic inference; every run records its seed and resolved configuration, and
single-seed results carry that caveat explicitly rather than implying run-to-run
generality.
