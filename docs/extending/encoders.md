# Add an encoder

An encoder changes how an existing game is presented to a network without changing the game's
rules. It owns both the observation tensor and, when necessary, the mapping between game action ids
and network-head columns.

This walkthrough adds a second Connect4 view. The built-in encoder already uses separate agent and
opponent planes. The new `MirroredConnect4Planes` keeps those planes but reflects
the board horizontally for player 1, giving both players a consistent player-relative convention.
That reflection also maps game column `a` to policy-head column `6 - a` for player 1.

The completed Python surface will be:

```python
import reinfors as rf

encoder = rf.encoders.MirroredConnect4()
game = rf.games.Connect4(encoder=encoder)

assert game.encoder.name == "mirrored_connect4"
assert encoder.head_index(action=0, agent=0) == 0
assert encoder.head_index(action=0, agent=1) == 6
```

## 1. Implement the native view

Add the encoder beside the existing `Connect4Planes` in
`crates/reinfors-games/src/connect4.rs`. Keeping game-specific encoders with their state type makes
the representation contract easy to review and test.

```rust
pub struct MirroredConnect4Planes;

impl MirroredConnect4Planes {
    fn view_col(column: usize, agent: usize) -> usize {
        if agent == 0 {
            column
        } else {
            COLS - 1 - column
        }
    }
}

impl ActionView for MirroredConnect4Planes {
    fn head_index(&self, action: usize, agent: usize) -> usize {
        Self::view_col(action, agent)
    }

    fn game_action(&self, head: usize, agent: usize) -> usize {
        Self::view_col(head, agent)
    }
}

impl StateEncoder for MirroredConnect4Planes {
    type State = Connect4State;

    fn encode(&self, state: &Connect4State, agent: usize) -> Vec<f32> {
        let mine = (agent + 1) as u8;
        let plane = ROWS * COLS;
        let mut obs = vec![0.0; 2 * plane];

        for (game_index, &piece) in state.cells.iter().enumerate() {
            let row = game_index / COLS;
            let game_col = game_index % COLS;
            let view_index = row * COLS + Self::view_col(game_col, agent);
            if piece == mine {
                obs[view_index] = 1.0;
            } else if piece != 0 {
                obs[plane + view_index] = 1.0;
            }
        }
        obs
    }

    fn obs_shape(&self) -> (usize, usize, usize) {
        (2, ROWS, COLS)
    }

    fn observation_space(&self) -> Space {
        Space::unit_box(vec![2, ROWS, COLS])
    }
}
```

`encode` returns a flat channel-major buffer matching `(C, H, W)`. `head_index` maps a canonical
game action into the network head; `game_action` is its inverse. Both action maps must be pure
functions of `(action, agent)`, never of the current state.

## 2. Test the representation contract

Add focused tests in the same module. The shared checker proves that every agent's action mapping is
in range, bijective, and invertible. A small position test then connects that mapping to the
observation transform.

```rust
#[test]
fn mirrored_view_keeps_observations_and_actions_aligned() {
    let encoder = MirroredConnect4Planes;
    reinfors_core::check_action_view(&encoder, COLS, 2);

    assert_eq!(encoder.head_index(0, 0), 0);
    assert_eq!(encoder.head_index(0, 1), 6);
    assert_eq!(encoder.game_action(6, 1), 0);

    let state = Connect4
        .step(&Connect4.initial_state(), &[0, 0])
        .next_state;
    let player_one = encoder.encode(&state, 1);

    // Player 0's piece in game column 0 appears in player 1's opponent plane, view column 6.
    assert_eq!(player_one[ROWS * COLS + 6], 1.0);
}
```

Run the native checkpoint before editing the bindings:

```bash
cargo test -p reinfors-games connect4
```

Re-export `MirroredConnect4Planes` from `crates/reinfors-games/src/lib.rs` so the binding crate can
import it.

## 3. Add the binding dispatch

Python handles store type-erased encoder specifications until an `Engine` or `Env` constructs the
concrete Rust composition. Find every existing Connect4 dispatch point first:

```bash
rg -n "EncoderSpec::Connect4|Connect4Planes|GameSpec::Connect4" crates/reinfors-py/src/lib.rs
```

Follow the existing multi-view Chess pattern. Introduce a game-specific variant and one helper that
materializes the concrete encoder:

```rust
#[derive(Clone, Copy)]
enum Connect4EncoderSpec {
    Standard,
    Mirrored,
}

#[derive(Clone, Copy)]
enum EncoderSpec {
    // ...
    Connect4(Connect4EncoderSpec),
    // ...
}

fn connect4_encoder(
    spec: Connect4EncoderSpec,
) -> Box<dyn StateEncoder<State = Connect4State>> {
    match spec {
        Connect4EncoderSpec::Standard => Box::new(Connect4Planes),
        Connect4EncoderSpec::Mirrored => Box::new(MirroredConnect4Planes),
    }
}
```

The builders currently receive `GameSpec`, so carry the selected concrete view there too:

```rust
enum GameSpec {
    // ...
    Connect4 { encoder: Connect4EncoderSpec },
    // ...
}
```

Then update the compiler-visible dispatch points:

1. Import the new native encoder and keep `Standard` as `GameSpec::Connect4`'s default. Allow both
   Connect4 encoder variants in `accepts_encoder`.
2. Persist the selected `Connect4EncoderSpec` in `GameSpec`, as the Chess arm does, so `spaces`,
   `build_engine`, `build_env`, snapshots, forks, and resolved configuration all reconstruct the
   same view.
3. Give both variants stable `cfg` and `name` values and the same action count.
4. Add `EncoderHandle::MirroredConnect4` and return its new specification.
5. Route `EncoderHandle.head_index` and `game_action` through `connect4_encoder`. Do not leave the
   new variant in the identity fallback: the Python methods must expose the same action frame used
   internally by search and learning.
6. Update every now-non-exhaustive `GameSpec::Connect4` match. Preserve the encoder selection only
   where observations are constructed; rules, rewards, and codecs remain encoder-independent.

Keeping these matches exhaustive is intentional: a future encoder cannot silently inherit an
identity action mapping.

## 4. Export the Python handle

Update the public surface in three places:

```python
# python/reinfors/_reinfors.pyi, on EncoderHandle
@staticmethod
def MirroredConnect4() -> EncoderHandle: ...
```

```python
# python/reinfors/encoders.py
MirroredConnect4 = _reinfors.EncoderHandle.MirroredConnect4

_REGISTRY = {
    # ...
    "mirrored_connect4": MirroredConnect4,
}
```

Add matching `EncoderInfo` metadata to `python/reinfors/catalog.py`, then regenerate the catalogue:

```bash
python scripts/generate_docs.py
```

The constructor name, registry key, `EncoderSpec::name`, resolved configuration, and catalogue key
must agree. The registry assertion and generated-document check catch drift between the Python
surface and catalogue.

## 5. Rebuild and test the public paths

Rebuild the extension using the [development setup](../development/setup.md#rebuild-after-native-changes),
then test direct construction, name-based construction, compatibility rejection, spaces, action
mapping, resolved-config reconstruction, and `Env`/`Engine` composition.

```python
import reinfors as rf

direct = rf.encoders.MirroredConnect4()
named = rf.encoders.make("mirrored_connect4")

assert direct.name == named.name == "mirrored_connect4"
assert direct.head_index(action=0, agent=1) == 6
assert direct.game_action(head=6, agent=1) == 0

game = rf.games.Connect4(encoder=direct)
assert game.encoder.name == "mirrored_connect4"
assert game.observation_space().shape == (2, 6, 7)
```

Also assert that another game's constructor rejects this handle. That verifies compatibility at the
Python boundary rather than relying only on Rust's generic types.

## Action-frame rule

`Game.legal_actions` and `Env.legal_actions()` use canonical game ids. Network outputs and training
batches use encoder head ids. Search, policies, and learners cross that boundary through
`head_index`; callers driving a network through `Env` must do the same explicitly. See
[action frames](../reference/glossary.md#action-frames) for the complete contract.

## Next steps

- Review the surrounding [native component contracts](component-contracts.md).
- Follow the general [Python binding rules](python-bindings.md#other-handle-based-components).
- Run all repository gates from [development setup](../development/setup.md#verify-changes).
