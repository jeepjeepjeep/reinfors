# References

What reinfors builds on. Legal redistribution notices live separately in the
repository's `THIRD-PARTY-NOTICES`.

## Algorithms

Per-algorithm sources are maintained in one place — the
[algorithms catalogue](../catalogue/algorithms.md), generated from `rf.catalog`
(each entry's **Sources** line covers the base algorithm and the implemented
variants: CFR+ and external-sampling MCCFR for `rf.solvers.Cfr`, Deep CFR for
`rf.solvers.DeepCfr`, decoupled UCT for simultaneous-move search, Bootstrapped
DQN for the multi-head design). `SelectiveExpectimax` is in-house design:
uncertainty-guided selective expansion over classical expectimax search.

## Encodings and layouts

- **OpenSpiel chess observation** (`rf.encoders.OpenSpielChess()`) — layout
  pinned to [OpenSpiel](https://github.com/google-deepmind/open_spiel)'s
  `chess.cc` observation tensor for cross-stack comparability; Lanctot et al.,
  ["OpenSpiel: A Framework for Reinforcement Learning in Games"](https://arxiv.org/abs/1908.09453),
  2019.
- **AZ-119 chess planes** (`rf.encoders.AlphaZeroChess()`) — the 119-plane
  input of Silver et al.,
  ["A general reinforcement learning algorithm that masters chess, shogi, and
  Go through self-play"](https://www.science.org/doi/10.1126/science.aar6404),
  Science, 2018, with documented deviations (absolute frame, side-to-move
  plane, newest-first history).
- **Backgammon encoding** (`rf.encoders.Backgammon()`) — Tesauro,
  ["Temporal Difference Learning and TD-Gammon"](https://doi.org/10.1145/203330.203343),
  Communications of the ACM, 1995.

## Validation and comparison

- **Mandatory internal suites** gate every game and encoding: rules, legality,
  codec round-trips, and the no-panic boundary sweep run in CI unconditionally.
- **Optional OpenSpiel oracles**: the poker suites compare against `pyspiel`
  when it is installed (`pytest.importorskip` — a development-time oracle, not
  a CI gate). The chess observation layout is pinned to OpenSpiel's
  implementation and exercised against it in the companion
  [reinfors-benchmarks](https://github.com/jeepjeepjeep/reinfors-benchmarks)
  repository, which also holds the benchmark protocol and results.
