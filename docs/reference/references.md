# References

What reinfors builds on, keyed to the code surface that implements it. Legal
redistribution notices live separately in the repository's `THIRD-PARTY-NOTICES`.

## Algorithms

- **AlphaZero** (`policies.AlphaZero`, `learners.AlphaZero`) — Silver et al.,
  ["A general reinforcement learning algorithm that masters chess, shogi, and Go
  through self-play"](https://www.science.org/doi/10.1126/science.aar6404),
  Science, 2018. Dirichlet root noise and the temperature-drop move-selection
  schedule follow the same work.
- **MCTS / UCT** (`policies.Mcts`) — Kocsis and Szepesvári,
  ["Bandit Based Monte-Carlo Planning"](https://doi.org/10.1007/11871842_29),
  ECML, 2006. The PUCT selection rule follows the AlphaGo Zero formulation
  (Silver et al., [Nature, 2017](https://doi.org/10.1038/nature24270)).
- **TreeStrap** (`learners.TreeStrap`) — Veness, Silver, Blair, and Uther,
  ["Bootstrapping from Game Tree Search"](https://papers.nips.cc/paper/3722-bootstrapping-from-game-tree-search),
  NeurIPS, 2009.
- **CFR** (`CfrSolver`) — Zinkevich, Johanson, Bowling, and Piccione,
  ["Regret Minimization in Games with Incomplete Information"](https://papers.nips.cc/paper/3306-regret-minimization-in-games-with-incomplete-information),
  NeurIPS, 2007.
- **Deep CFR** (`DeepCfr`) — Brown, Lerer, Gross, and Sandholm,
  ["Deep Counterfactual Regret Minimization"](https://proceedings.mlr.press/v97/brown19b.html),
  ICML, 2019.
- **DQN** (`learners.Dqn`, `policies.EpsilonGreedyQ`) — Mnih et al.,
  ["Human-level control through deep reinforcement learning"](https://doi.org/10.1038/nature14236),
  Nature, 2015.
- **SelectiveExpectimax** (`policies.SelectiveExpectimax`) — in-house design:
  uncertainty-guided selective expansion over classical expectimax search.

## Encodings and layouts

- **OpenSpiel chess observation** (`encoders.OpenSpielChess`) — layout pinned to
  [OpenSpiel](https://github.com/google-deepmind/open_spiel)'s `chess.cc`
  observation tensor for cross-stack parity; Lanctot et al.,
  ["OpenSpiel: A Framework for Reinforcement Learning in Games"](https://arxiv.org/abs/1908.09453),
  2019.
- **AZ-119 chess planes** (`encoders.AlphaZeroChess`) — the 119-plane input of
  Silver et al. 2018, with documented deviations (absolute frame, side-to-move
  plane, newest-first history).
- **Backgammon encoding** (`BackgammonTesauro`) — Tesauro,
  ["Temporal Difference Learning and TD-Gammon"](https://doi.org/10.1145/203330.203343),
  Communications of the ACM, 1995.

## Validation and comparison targets

- **OpenSpiel** — the benchmark comparison target (protocol, evidence, and
  results in the companion
  [reinfors-benchmarks](https://github.com/jeepjeepjeep/reinfors-benchmarks)
  repository) and the `pyspiel` parity suites that gate the chess, backgammon,
  and poker implementations.
