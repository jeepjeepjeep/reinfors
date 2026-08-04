# Python bindings

Python exposes opaque native handles rather than mirroring Rust state. The built-in handle types are
defined in `crates/reinfors-py/src/lib.rs`, then given typed and name-addressable Python surfaces in
`python/reinfors`.

## Register a built-in game

GridWorld is a complete, compact example. Search its integration points before editing the large
binding module:

```bash
rg -n "GameSpec::GridWorld|fn gridworld" crates/reinfors-py/src/lib.rs
```

### 1. Add the native dispatch

In `crates/reinfors-py/src/lib.rs`:

1. Import the game, encoder, event, and reward from `reinfors-games`.
2. Add its validated configuration to `GameSpec` and handle it in `num_agents` and `spaces`.
3. Add it to `game_cfg`, `reward_schema`, `build_reward`, and `RewardBox`.
4. Add concrete arms to `build_engine` and `build_env`, including its codec when snapshots are
   supported.
5. Include any game-specific fields exposed by common handle methods, such as
   `truncation_horizon`.

The public constructor is a static method on `GameHandle`. GridWorld's arm validates before storing
the spec and is the pattern to copy:

```rust
#[staticmethod]
#[pyo3(signature = (size=5, goal_row=None, goal_col=None, max_ticks=1000))]
#[pyo3(name = "GridWorld")]
fn gridworld(
    size: i32,
    goal_row: Option<i32>,
    goal_col: Option<i32>,
    max_ticks: Option<usize>,
) -> PyResult<Self> {
    check_max_ticks(max_ticks)?;
    let corner = size.saturating_sub(1);
    let goal = (goal_row.unwrap_or(corner), goal_col.unwrap_or(corner));
    GridWorld { size, goal, max_ticks }
        .validate()
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    Ok(GameHandle {
        spec: GameSpec::GridWorld { size, goal, max_ticks },
    })
}
```

Keeping the configuration in `GameSpec` allows the binding to reconstruct the concrete generic
composition only when an `Engine` or `Env` is built. Constructor failures must become Python
exceptions rather than Rust panics.

### 2. Add the Python surface

Update these three files:

- `python/reinfors/_reinfors.pyi`: add the typed `GameHandle` static constructor;
- `python/reinfors/games.py`: export the constructor and add its stable snake-case name to
  `_REGISTRY`;
- `python/reinfors/catalog.py`: add compatibility and user-facing metadata to `GAMES`.

The registry supplies both `rf.games.GridWorld(...)` and config-driven construction through
`rf.games.make("gridworld", ...)`. Its import-time assertion against the catalogue prevents a
registered game from silently disappearing from generated documentation.

### 3. Verify the installed extension

Complete the [native rebuild check](../development/setup.md#rebuild-after-native-changes), then
verify the registry entry:

```bash
python -c 'import reinfors as rf; assert "my_game" in rf.games.registered()'
```

Replace `my_game` with the stable registry name.

### 4. Test every public path

At minimum, test:

- direct construction and invalid arguments;
- `rf.games.make(name, **kwargs)` and `registered()`;
- `Engine` and `Env` composition;
- resolved-config reconstruction;
- snapshot round trips when a codec is provided;
- Gymnasium or PettingZoo compliance when the game's dynamics fit an adapter.

GridWorld coverage is spread across `tests/test_games.py`, `tests/test_constructor_validation.py`,
`tests/test_env_snapshots.py`, `tests/test_gym.py`, and `tests/test_catalog.py`.

## Other handle-based components

Policies, learners, encoders, chance modes, and noise follow the same overall path:

1. add a validated PyO3 factory/spec arm in `crates/reinfors-py/src/lib.rs`;
2. add the typed constructor to `python/reinfors/_reinfors.pyi`;
3. export it from its module under `python/reinfors`;
4. register its stable name in that module's `_REGISTRY`;
5. add catalogue metadata when it is a user-facing algorithm;
6. test direct and name-based construction, composition, resolved config, and invalid input.

Use an existing component in the same family as the exemplar. Its `Spec` occurrences in
`crates/reinfors-py/src/lib.rs` reveal every type-erased dispatch point that needs an arm.

## Solvers are different

`Solver` is an architectural category, not a shared Rust trait or handle registry. The current
Python solvers are concrete PyO3 classes:

- native implementations: `crates/reinfors-core/src/solvers/cfr.rs` and `deep_cfr.rs`;
- binding classes: `PyCfr` and `PyDeepCfr` in `crates/reinfors-py/src/lib.rs`;
- module registration: `_reinfors` adds each class with `m.add_class` near the end of that file;
- typed surface: `Cfr` and `DeepCfr` in `python/reinfors/_reinfors.pyi`;
- public exports: direct aliases in `python/reinfors/solvers.py`;
- tests: `tests/test_cfr.py` and `tests/test_deep_cfr.py`.

There is no `_REGISTRY`, `make`, or generic solver registration hook in
`python/reinfors/solvers.py`. Adding a new solver therefore means binding its concrete API, adding it
to the PyO3 module, stub, and public exports, then documenting and testing that API. If the solver
only supports some games, its constructor must validate those compositions and direct users to the
compatibility documentation rather than embed a game list that will become stale.

## Binding design rules

- Keep Python callbacks at a small number of explicit seams; do not add per-node Python calls.
- Convert callback failures and constructor errors into contextual Python exceptions.
- Validate array rank, shape, dtype, row count, and finite values before native indexing.
- Preserve named batch fields when adding data; positional tuple order is compatibility-only.
- Keep catalogue metadata pure Python so documentation can build without compiling Rust.
- Update type stubs in the same change as runtime bindings.

## Update documentation

User-facing components also require catalogue metadata. Follow the single
[catalogue-generation workflow](../development/documentation.md#update-component-coverage) rather
than editing generated pages.
