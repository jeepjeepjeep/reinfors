<p align="center">
  <img src="https://raw.githubusercontent.com/jeepjeepjeep/reinfors/main/assets/reinfors-banner.svg" alt="Animated reinfors logo" width="100%">
</p>

High-throughput reinforcement-learning search and sampling in Rust, with caller-owned
Python networks and training.

Reinfors runs game dynamics, search, episode orchestration, and batch assembly in a
parallel Rust backend. Your inference callback is the boundary: it receives pooled NumPy
observations and returns model outputs, so the network, framework, optimizer, replay,
hardware placement, and distributed topology remain yours.

[Game semantics](https://jeepjeepjeep.github.io/reinfors/catalogue/games/) cover single- and multi-agent, zero-sum, cooperative, and
general-sum tasks, with turn-taking or simultaneous decisions, explicit chance, and perfect
or imperfect information. [Algorithms](https://jeepjeepjeep.github.io/reinfors/catalogue/algorithms/) span policy-driven value learning,
search-guided learning, and standalone game-theoretic solving.

```python
import numpy as np
import reinfors as rf

game = rf.games.Connect4()
engine = rf.Engine(
    game=game,
    reward=rf.Reward(win=1.0, loss=-1.0),
    policy=rf.policies.Minimax(depth=4),
    learner=rf.learners.TreeStrap(),
    n_games=32,  # parallel episode slots
    seed=0,
)

actions = game.action_space().n

def infer(obs: np.ndarray) -> np.ndarray:
    # The network is the search's evaluation function: it scores every frontier leaf,
    # pooled across games. Replace with PyTorch, JAX, an accelerator service, or any
    # other backend; TreeStrap's targets train it toward its own deeper search results.
    return np.zeros((len(obs), 1, actions), dtype=np.float32)

batch = engine.collect(n_records=1024, infer=infer)
print(batch.obs.shape, batch.targets.shape, batch.telemetry)
```

## Performance

Reinfors' native backend makes it much more performant than Python-loop RL
libraries such as Gymnasium. Benchmarking reinfors' `car_racing` port against
Gymnasium's `CarRacing-v3` — separate implementations of the same design, with
matched action space and 96x96 RGB observation content, stepped in
single-threaded loops — reinfors runs **~20x** more environment steps per second
(medians of three 30s trials):

| Steps/s, single-threaded | Apple M1 Max | AMD EPYC 7R32 (g5.2xlarge, SMT off, pinned) |
| --- | --- | --- |
| reinfors `car_racing` | 3,850 | 2,069 |
| Gymnasium `CarRacing-v3` | 195 | 148 |
| **speedup** | **19.7x** | **14.0x** |

Reproduce with [`scripts/bench_carracing_throughput.py`](scripts/bench_carracing_throughput.py), which reports medians,
ranges, and machine provenance.

Parallelising multiplies this further, and in reinfors it is trivial — set
`n_threads` — while [`collect_stream`](https://jeepjeepjeep.github.io/reinfors/guides/streaming/) overlaps collection with training as the
normal operating mode:

<p align="center">
  <img src="https://github.com/jeepjeepjeep/reinfors/releases/download/v0.3.0/carracing-throughput.gif" alt="One Gymnasium car completes a single lap while reinfors completes 20" width="90%">
</p>

Benchmarked against OpenSpiel's all-C++ libtorch AlphaZero on a matched chess training
workload — two-hour rounds at each stack's best measured configuration, then
head-to-head play between the resulting models — reinfors sustained **9.9% higher
training throughput**, and its trained networks scored **0.605 ± 0.020** against
OpenSpiel's over 300 paired games (**+74 Elo**). Protocol, evidence, and full results:
[reinfors-benchmarks](https://github.com/jeepjeepjeep/reinfors-benchmarks).

Reinfors is not designed to maximize throughput at any cost or to outperform a bespoke, fully
fused JAX/XLA pipeline on the fixed workload it specializes for. It targets a practical balance:
native simulation, search, and batching, while games, algorithms, networks, and deployment remain
modular enough for broad experimentation.

## Install

```bash
pip install reinfors
```

Optional adapters and training dependencies are separate:

```bash
pip install "reinfors[gym]"   # Gymnasium and PettingZoo adapters
pip install "reinfors[train]" # PyTorch examples
```

Contributing or building from source? Start with the
[contributing guide](https://github.com/jeepjeepjeep/reinfors/blob/main/CONTRIBUTING.md) and the
[development setup guide](https://jeepjeepjeep.github.io/reinfors/development/setup/).

## Where next?

- [Get started](https://jeepjeepjeep.github.io/reinfors/getting-started/)
- [Understand sampling and injectable training](https://jeepjeepjeep.github.io/reinfors/concepts/sampling-and-training/)
- [Choose a game](https://jeepjeepjeep.github.io/reinfors/catalogue/games/), [algorithm](https://jeepjeepjeep.github.io/reinfors/catalogue/algorithms/), or
  [built-in composition](https://jeepjeepjeep.github.io/reinfors/catalogue/compatibility/)
- [Run the examples](https://jeepjeepjeep.github.io/reinfors/examples/)
- [Evaluate searched agents with Arena](https://jeepjeepjeep.github.io/reinfors/guides/arena/)
- [Add a Rust game or algorithm](https://jeepjeepjeep.github.io/reinfors/extending/)
- [Read the complete documentation](https://jeepjeepjeep.github.io/reinfors/)

## Stability

reinfors is pre-1.0: **any 0.x release may change any API, behavior, or serialized format**
(including snapshot and config layouts) without deprecation. Pin an exact version
(`reinfors==0.x.y`) and read release notes when upgrading. What does hold at every version:
constructors validate their inputs, and no public Python input reaches a Rust panic — both
enforced by adversarial test sweeps in CI.

## Citation

If you use reinfors in your research, cite it via the repository's
[CITATION.cff](https://github.com/jeepjeepjeep/reinfors/blob/main/CITATION.cff) (GitHub's "Cite this repository" button renders it
as BibTeX/APA). What reinfors itself builds on is catalogued in
[References](https://jeepjeepjeep.github.io/reinfors/reference/references/).

## License

Licensed under either of [MIT](https://github.com/jeepjeepjeep/reinfors/blob/main/LICENSE-MIT) or [Apache-2.0](https://github.com/jeepjeepjeep/reinfors/blob/main/LICENSE-APACHE), at your
option. Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in this work shall be dual-licensed as above, without any additional terms
or conditions.
