# reinfors

<p align="center">
  <img src="https://raw.githubusercontent.com/jeepjeepjeep/reinfors/main/assets/reinfors-banner.svg" alt="Animated reinfors logo" width="100%">
</p>

High-throughput reinforcement-learning search and sampling in Rust, with caller-owned
Python networks and training.

Reinfors runs game dynamics, search, episode orchestration, and batch assembly in a
parallel Rust backend. Your inference callback is the boundary: it receives pooled NumPy
observations and returns model outputs, so the network, framework, optimizer, replay,
hardware placement, and distributed topology remain yours.

[Benchmarks](docs/benchmarks/index.md) find this boundary negligible beside search and
training, so an all-Rust training path would sacrifice flexibility for little measured gain.

Reinfors is not designed to maximize throughput at any cost or to outperform a bespoke, fully
fused JAX/XLA pipeline on the fixed workload it specializes for. It targets a practical balance:
native simulation, search, and batching, while games, algorithms, networks, and deployment remain
modular enough for broad experimentation.

```python
import numpy as np
import reinfors as rf

game = rf.games.Connect4()
engine = rf.Engine(
    game=game,
    reward=rf.Reward(win=1.0, loss=-1.0),
    policy=rf.policies.Mcts(num_simulations=64),
    learner=rf.learners.TreeStrap(),
    n_games=32,  # parallel episode slots
    seed=0,
)

actions = game.action_space().n

def infer(obs: np.ndarray) -> np.ndarray:
    # Replace with PyTorch, JAX, an accelerator service, or any other backend.
    return np.zeros((len(obs), 1, actions), dtype=np.float32)

batch = engine.collect(n_records=1024, infer=infer)
print(batch.obs.shape, batch.targets.shape, batch.telemetry)
```

## Why reinfors?

- Search and sampling are native, multithreaded, and batch network requests across games
  and search leaves.
- `collect` supports a simple synchronous loop; `collect_stream` runs parallel Rust search
  concurrently with Python training, with configurable queueing and bounded backpressure.
- Networks are injectable per player and are not tied to a framework or device topology.
- Composable Rust traits make new games and algorithms straightforward to add, with safer,
  simpler native extension than comparable C++ infrastructure.
- Games include deterministic, stochastic, simultaneous, N-player, and
  imperfect-information environments.
- Algorithms span policy-driven value learning, search-guided learning, and standalone
  game-theoretic solving; see the [algorithm catalogue](docs/catalogue/algorithms.md) for current
  implementations.
- Resolved configurations, snapshots, structured batches, and telemetry support
  reproducible experiments.

## Install

```bash
pip install reinfors
```

Optional adapters and training dependencies are separate:

```bash
pip install "reinfors[gym]"   # Gymnasium and PettingZoo adapters
pip install "reinfors[train]" # PyTorch examples
```

Reinfors is pre-1.0. Public contracts are documented, but breaking changes may still occur.

## Where next?

- [Get started](docs/getting-started.md)
- [Understand sampling and injectable training](docs/concepts/sampling-and-training.md)
- [Choose a game](docs/catalogue/games.md), [algorithm](docs/catalogue/algorithms.md), or
  [built-in composition](docs/catalogue/compatibility.md)
- [Run the examples](docs/examples/index.md)
- [Add a Rust game or algorithm](docs/extending/index.md)
- [Read the complete documentation](docs/index.md)

The benchmark area is being rebuilt with controlled, reproducible measurements. Results
will be published only with their hardware, configuration, and methodology.

## License

[MIT](LICENSE)
