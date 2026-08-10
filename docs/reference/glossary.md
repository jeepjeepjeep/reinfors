# Glossary

These terms describe reinfors' public data and execution model.

## Core components

**Game**
: Native rules and immutable game configuration. A game defines state transitions, active agents or
  chance, legal actions, terminality, events, and its observation/action spaces; it does not choose
  actions or train a model.

**Encoder**
: The view from native game state to a player's network observation. An encoder may also translate
  between canonical game-action ids and network-head columns; it changes representation, not rules.

**Reward**
: The mapping from a game's named transition events to per-agent scalar rewards. Reward weights
  define the training objective without changing the game's rules, events, or terminal states.

**Policy**
: The acting rule. A policy may select directly from network values or run search, and it defines
  the inference output it consumes. It chooses actions but does not construct the training loss.

**Learner**
: The record producer paired with a policy. It turns transitions, trajectories, or search results
  into learner-shaped training rows. The Python caller still owns the network, optimizer, replay
  buffer, and gradient update.

**`Engine`**
: The batched collector. It advances many episode slots using one game/policy/learner composition,
  pools network requests through the inference callback, and returns training batches.

**`Env`**
: One caller-driven game instance for evaluation, scripted play, adapters, snapshots, and forks.
  The caller supplies actions directly; `Env` does not run an Engine policy or emit learner batches.

**Solver**
: A standalone algorithm that owns its traversal and persistent state instead of using the
  Policy/Learner/Engine collection pipeline. Current examples include CFR and Deep CFR.

## Collection

**Record**
: One learner-produced training row. Its meaning depends on the learner: for example, a DQN record
  is a transition and an AlphaZero record is an observation with policy and value targets.

**Record floor**
: The minimum number of records requested from `Engine.collect`. The engine preserves completed
  episode and search work, so the returned batch can contain more rows than the floor.

**`n_games`**
: The number of episode slots the engine advances in parallel. Slots reset and start new episodes as
  collection continues; `n_games=16` does not mean that collection stops after sixteen episodes.

**Tick**
: One game-time boundary: the active agent action, or simultaneous joint action, plus any following
  chain of chance draws until the next decision or terminal state. Events and rewards from those
  edges belong to the same tick. `max_ticks` limits these boundaries, not search nodes or inference
  calls.

**Truncation tail**
: A DQN record at an episode-length boundary in an alternating game whose post-move state belongs to
  the opponent. The record's player has no legal next action, so the row is non-bootstrapping even
  though the game did not reach a rules-terminal state; its TD target contains only immediate reward.

## Game information

**Information state**
: Everything an agent is allowed to distinguish at one decision: its observations, private
  information, and relevant remembered history. Imperfect-information solvers group game states by
  `information_state_key`; equal keys must represent decisions indistinguishable to that agent.

## Network outputs and ensembles

**Network output head**
: A semantically different output of one model, such as AlphaZero's policy head and value head.

**Ensemble head (`n_heads`)**
: One independently bootstrapped Q/value estimator. “Head” in this context is not a semantically
  distinct model output such as AlphaZero's policy head or value head.

**Bootstrap mask**
: A `(records, heads)` membership array. `masks[r, k] == 1` means training record `r`
  contributes to ensemble head `k`; zero excludes that record/head pair from the loss. Compute a
  loss per pair, multiply by `masks`, then reduce over included pairs.

## Sparse legal actions

**CSR**
: Compressed sparse row storage. Legal action ids for every record are packed into one `ids` array;
  an `offsets` array of length `records + 1` marks each row. Record `i` contains:

  ```python
  row_legal_actions = ids[offsets[i] : offsets[i + 1]]
  ```

  This avoids allocating a dense `(records, action_count)` legality mask for games with wide action
  vocabularies.

## Action frames

**Game-action frame**
: The canonical action ids accepted by `Env.step` and returned by `Env.legal_actions`. These ids
  describe the native game regardless of the observing player's perspective.

**Network-head frame**
: The action-column ids exposed to a network after applying the selected encoder. Inference columns
  and every action-indexed Engine field—including DQN actions and legal ids, TreeStrap targets, and
  AlphaZero policy targets—use this frame. Identity encoders make the two frames numerically equal;
  transforming encoders such as `RelativeChess` do not.

Move from a game id to a network column with `encoder.head_index(game_action, agent)`. Move from a
selected network column back to an environment action with `encoder.game_action(head_action,
agent)`. Failing to convert can select a legal but silently wrong move.

## Chance modes

Search policies receive one of the handles in `rf.chance_modes`:

**`AlwaysResample()`**
: Draw one outcome from the declared distribution on every tree descent. This is an unbiased sampled
  estimate and the default for MCTS and AlphaZero.

**`Committed(samples=k)`**
: Draw `k` outcomes for an edge, keep them fixed, and search more deeply inside that sampled chance
  model. This controls work on wide chance fans and is SelectiveExpectimax's default at `k=1`.

**`ExpandAll()`**
: Enumerate and probability-weight every declared outcome. This is exact for a narrow chance fan and
  rejects fans above the canonical [enumeration limit](limits.md#enumeration-limits).

SelectiveExpectimax expands a node once, so it supports `Committed` and `ExpandAll` but rejects
`AlwaysResample` at construction. MCTS and AlphaZero revisit nodes and support all three modes.

## Exploration noise

**Exploration noise**
: A deliberate perturbation used to diversify action selection, distinct from game chance. The
  current `rf.noise` surface supplies AlphaZero root Dirichlet noise over legal-action priors.

## Experiment identity

**Resolved configuration**
: The JSON-compatible result of `engine.resolved_config()`, with constructor defaults filled in. It
  records the complete native composition needed by `rf.engine_from_config(...)`.

**Configuration fingerprint (`config_fingerprint()`)**
: An opaque identifier computed from the resolved configuration. It identifies the complete engine
  composition for equality checks; callers should compare it rather than reproduce its hashing.

**Policy version**
: A caller-chosen model identifier stored in an engine snapshot. Pair it with the exact model file
  and pass it as `expect_policy_version` during restore; reinfors compares the identifier but cannot
  inspect caller-owned weights.

## Search and limits

**MaxN backup**
: A sequential N-player tree backup that carries one value per player. At player `i`'s decision,
  search selects the child maximizing component `i`, rather than applying a two-player negamax sign
  flip.

**Safety cap**
: A hard bound checked before an exact operation would allocate or enumerate excessive work.
  Crossing one raises a configuration or runtime error; sampled alternatives do not enumerate the
  capped fan. The numeric values live in [enumeration limits](limits.md#enumeration-limits).

## Next steps

- Apply record, head, legality, and action-frame terms in [batch formats](batch-formats.md).
- Implement a network against the [inference contract](inference-contract.md).
- Return to the runnable [training guide](../guides/training.md).
