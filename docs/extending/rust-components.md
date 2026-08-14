# Add a game

This walkthrough implements one built-in game from native rules through its Python surface. Read the
[component contracts](component-contracts.md) first when deciding whether the change is a game,
encoder, reward, policy, learner, or solver.

## Walkthrough

GridWorld is the best template for a small game. Its rules, event, reward, encoder, codec, and unit
tests are all in `crates/reinfors-games/src/gridworld.rs`.

### 1. Define rules, state, and events

Add `crates/reinfors-games/src/my_game.rs`. Keep configuration on the game, mutable episode data in
the state, and reward-relevant outcomes in the event:

```rust
#[derive(Clone)]
pub struct MyState {
    // Native state used by the rules.
}

pub struct MyEvent {
    // What one transition causally decided for one agent.
}

pub struct MyGame {
    // Validated rules configuration.
}

impl MyGame {
    pub fn validate(&self) -> Result<(), String> {
        // Reject invalid user configuration here.
        Ok(())
    }
}
```

Then implement the authoritative state machine from `crates/reinfors-core/src/game.rs`:

```rust
impl Game for MyGame {
    type State = MyState;
    type Event = MyEvent;

    fn num_agents(&self) -> usize { /* ... */ }
    fn action_count(&self) -> usize { /* ... */ }
    fn actor(&self, state: &MyState) -> Actor { /* ... */ }
    fn legal_actions(&self, state: &MyState, agent: usize) -> Vec<usize> { /* ... */ }
    fn step(&self, state: &MyState, actions: &[usize]) -> Transition<MyState, MyEvent> {
        /* ... */
    }
    fn initial_state(&self) -> MyState { /* ... */ }
}
```

`actor` identifies an agent, simultaneous agents, or nature. If it can return `Actor::Chance`, also
implement `chance_node` and `apply_chance_node`; GridWorld shows a uniform root draw, while
`crates/reinfors-games/src/backgammon.rs` shows chained weighted draws. Imperfect-information games
also implement `information_states`, `information_state_key`, and `perfect_information`; use
`crates/reinfors-games/src/kuhn.rs` as the small exemplar.

Important invariants:

- legal actions and active agents agree with the node returned by `actor`;
- a transition has one event slot per agent and emits only what that edge determines;
- root chance resolves to a non-terminal decision state without emitting events;
- weighted chance probabilities are finite and positive;
- accepted states are safe for every game method.

Prefer a chain of narrow chance nodes over eagerly constructing a combinatorial outcome when draws
are naturally sequential.

How the engine and the policies consume chance, so a game knows what it is signing up for:

- **The rollout samples real chance.** Root chance chains are realized before any policy sees a
  state (`realize_initial_state`), and after every played action the episode samples the complete
  chance chain to the next decision or terminal state (`step_env`); both are cycle-guarded. The
  actual trajectory always draws from the game's declared distributions — policies never influence
  it, and non-search policies (e.g. DQN's) meet chance only this way. No policy ever sees a chance
  state as its decision root; MCTS asserts exactly that.
- **Search policies additionally choose how to model hypothetical chance during planning**, via
  their [chance mode](../catalogue/algorithms.md): MCTS / AlphaZero expand explicit chance nodes
  inside the tree; SelectiveExpectimax applies its chance mode while flattening chance chains along
  stepped edges and never represents chance states as tree nodes; Minimax expands each node exactly
  once, so it accepts only chance modes expressible in a single expansion and rejects per-traversal
  resampling at construction.

A game that emits `Actor::Chance` therefore works unmodified across policies: the realized
trajectory always samples the game's distributions, and only the planning-time modeling of
outcomes differs — selected on the policy, never in the game.

### 2. Add the representation and reward

An encoder implements both traits in `crates/reinfors-core/src/encoder.rs`:

```rust
pub struct MyEncoder;

impl ActionView for MyEncoder {} // Identity mapping from game actions to network-head indices.

impl StateEncoder for MyEncoder {
    type State = MyState;

    fn encode(&self, state: &MyState, agent: usize) -> Vec<f32> { /* ... */ }
    fn obs_shape(&self) -> (usize, usize, usize) { /* ... */ }
}
```

`encode` returns a flat channel-major buffer matching `(C, H, W)`. Override `ActionView` when an
agent-relative observation also changes how actions map to the network head. Test that mapping with
`reinfors_core::check_action_view`. For a complete alternative-view example, follow
[Add an encoder](encoders.md).

Implement `Reward` from `crates/reinfors-core/src/reward.rs` separately from the rules:

```rust
pub struct MyReward { /* configurable weights */ }

impl Reward for MyReward {
    type Event = MyEvent;

    fn step_reward(&self, event: &MyEvent, agent: usize) -> f64 { /* ... */ }
}
```

This separation lets users change an objective without duplicating game dynamics.

### 3. Add snapshot support

Built-in games should implement `StateCodec` from `crates/reinfors-core/src/codec.rs` so `Env` and
`Engine` snapshots can preserve their state. GridWorld uses the shared serde helpers, versions its
payload, recomputes derived fields during decode, and validates only the invariants needed to make
game operations safe. A codec does not need to prove that a state is reachable through legal play.

### 4. Export and compile the native game

Declare the module and re-export its public types from `crates/reinfors-games/src/lib.rs`. Compile
the game before touching the binding layer:

```bash
cargo check -p reinfors-games
```

This checkpoint isolates native module, type, and trait errors from later PyO3 registration errors.

### 5. Test the new game

Put focused rules tests beside the implementation. GridWorld's tests cover movement, legality,
chance, rewards, encoding, truncation, codec round trips, and native engine composition. Use it to
decide what to test, but filter the command to your new module:

```bash
cargo test -p reinfors-games my_game::tests
```

Replace `my_game` with the module name. Check that Cargo actually ran the expected tests: a successful
command that reports zero matching tests has not validated the game. Then run the full native game
suite:

```bash
cargo test -p reinfors-games
```

### 6. Bind and test Python

Follow [Register a built-in game](python-bindings.md#register-a-built-in-game) to expose the game.
Then use the canonical [native rebuild check](../development/setup.md#rebuild-after-native-changes)
before testing the installed Python surface:

```bash
python -c 'import reinfors as rf; assert "my_game" in rf.games.registered()'
```

Replace `my_game` with the stable registry name. The Python-side tests for the GridWorld exemplar
are easiest to find with:

```bash
rg -n "GridWorld|gridworld" tests
```

## Next steps

- Run the complete local gates from [development setup](../development/setup.md).
- Add user-facing metadata through the
  [documentation workflow](../development/documentation.md#update-component-coverage).
