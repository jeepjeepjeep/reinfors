# Add a policy or learner

A policy owns evaluation and action selection; a learner converts evaluations and finished
trajectories into typed training records. They meet at the policy's `Evaluation` type, so the
two usually ship as a pair: a learner is written against the evaluation a policy produces.

The smallest complete pair is model-free:
`crates/reinfors-core/src/policies/modelfree/epsilon_greedy_q.rs` with
`crates/reinfors-core/src/learners/dqn.rs`. Search policies live under
`crates/reinfors-core/src/policies/tree/`; use MCTS as the exemplar there. Read this page with
one of those files open.

## Walkthrough

### 1. Implement `Policy`

The trait is in `crates/reinfors-core/src/policy.rs`. Its surface, in the order that matters:

- `type Evaluation` — what evaluating one `(state, agent)` request produces. This is the
  currency between policy and learner, and it is serialized (`encode_eval` / `decode_eval`)
  when engine snapshots capture partial trajectories' buffered evaluations.
- `type PolicyState` — per-episode policy state (move counters, temperature schedules). It
  must round-trip through a `u64` (`policy_state_to_u64` / `policy_state_from_u64`) so
  snapshots stay compact; keep it that small by design.
- **Capability claims have no defaults, deliberately**: `max_agents(sequential)` and
  `supports_imperfect_information` force every new policy to state what it supports. The
  engine's composition gates read these — a policy that searches the true state of a
  hidden-information game is rejected at `Engine` construction, with a pointer to the
  [compatibility catalogue](../catalogue/compatibility.md).
- `evaluate(...)` — the batch heart: a `Vec<(state, agent)>` of requests in, evaluations out.
  **Return exactly one evaluation per request, in request order** — the engine pairs them
  positionally and asserts the lengths match. All inference goes through the passed
  `Evaluator`; a policy never calls the network directly, which is what lets the engine pool
  requests across games into batched callback calls.
- `select(eval, policy_state, rng)` — one action from one evaluation. **The returned action
  must be legal in the game-action frame** (a game action id, not a network-head index).
- `fold_telemetry` (optional) — surface search statistics into `CollectStats`.

Draw randomness only from the `rng` arguments: a policy must be deterministic given its
requests, seed, inference outputs, and RNG state. Hidden randomness (globals, hash-order
iteration) breaks the engine's determinism guarantees.

### 2. Implement `Learner`

`Learner<E>` (`crates/reinfors-core/src/learner.rs`) is generic over the evaluation type it
consumes:

- `type Record` plus `eval_records` / `episode_records` — the two record sources: per-evaluation
  and per-finished-episode.
- Behavior knobs with defaults: `needs_next_obs`, `needs_interior`, `uses_episode_tail` (+
  `tail_from_row` for truncation bootstraps), `value_only_evaluation`.
- **Network-facing action ids go through the supplied `ActionView`** (as `tail_from_row`'s
  default does with `head_index`): encoders may reorder or re-orient the action head, so a
  learner that writes raw game-action ids into records corrupts them under those encoders.

Document the record's shape, dtype, legal-action treatment, and terminal/truncation behavior —
that description becomes the [batch-formats reference](../reference/batch-formats.md) entry.

A new `Record` type must also cross the Python boundary: implement the binding's private
`RecordBatch` conversion (`into_py_batch` in `crates/reinfors-py/src/lib.rs`), usually with a
`#[pyclass]` batch struct registered with the PyO3 module (the `m.add_class` block at the end
of `lib.rs`) and exposed through the stub and top-level API (`__init__.py`'s import list and
`__all__`) — the engine wrapper's `L::Record: RecordBatch` bound fails to compile without
it. A policy that introduces a new callback-output contract likewise selects or extends
`InferLayout`, which sizes and validates the callback's return shape. Reusing an existing
layout across policies is the cheap path — PPO reuses `PolicyValue` — but contract errors
name the composed policy through the label carried in the layout variant, and the
[troubleshooting table](../reference/troubleshooting.md) matches those strings: keep existing
labels byte-stable and add the new label's fragments to the table.

### 3. Bind and compose

Follow the [handle-based binding path](python-bindings.md#other-handle-based-components) with
one policy-specific addition. The binding dispatches compositions per **(policy, learner)
pair**: `build_for_game` in `crates/reinfors-py/src/lib.rs` has one arm per supported pairing,
and unsupported pairings fall through to the incompatible-composition error. A new policy
compatible with an existing learner adds an arm; a new pair adds one arm and its capability
gates. Two further exhaustive matches always need arms — the `choose_batch` policy dispatch
and the choose path's expected-heads/layout map — and the compiler names any others: grep an
existing `PolicySpec` variant to find every dispatch point, then let `cargo check`'s
non-exhaustive-match errors confirm you found them all.

Composition is also where resource validation lives: budget-like constructor parameters
(simulation counts, expansion budgets, depths) enroll in `check_search_budgets`, and anything
that multiplies per-call or per-node memory (an `n_heads`-like parameter) needs a cap there —
see the existing arms. The config JSON rendering must round-trip through
`rf.engine_from_config(engine.resolved_config())` — add round-trip coverage for the new
composition; the existing tests exercise representative families, not every pairing.

### 4. Let the tests enforce enrollment

Registration is self-policing:

- The `_REGISTRY` entry (with its `catalog.py` twin — a parity assert fails the import if they
  diverge) auto-enrolls the constructor in the no-panic sweep. A parameter whose default hides
  its type fails test collection until it gets an edge-value bank, and every policy/learner
  name must have a composition hook (engine build + one collect) or collection fails with
  `"policies.X has no composition hook: enroll it in USE"` — both in
  `tests/test_constructor_validation.py`.
- Two more assertions fail until updated, by design: `tests/test_catalog.py`'s
  `build_workflow` needs an arm constructing the new composition (that arm is what makes the
  generated compatibility matrix construction-tested), and
  `tests/test_games.py::test_registries_list_the_built_in_names` pins the exact registry name
  lists.
- Add family tests alongside the sweep: record shapes and dtypes, determinism for a seed, and
  whatever the learner's targets claim (see `tests/test_engine.py` for the TreeStrap pattern
  of tying collected targets to the search's own outputs).

### 5. Document

In `python/reinfors/catalog.py`: add the name to `POLICIES` and/or `LEARNERS`, the algorithm
entry (description, reference citation) to `ALGORITHMS`, and the algorithm to each compatible
game's `GameInfo.algorithms` — then run `python scripts/generate_docs.py`. The
[compatibility matrix](../catalogue/compatibility.md) is generated from those entries; never
edit it directly. The
[compatibility questions](index.md#compatibility-questions-to-answer-first) are the checklist
of facts to state.
