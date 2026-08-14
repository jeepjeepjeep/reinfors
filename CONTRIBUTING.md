# Contributing

Thanks for considering a contribution. Issues and pull requests are welcome; for anything
larger than a bug fix, open an issue first so the approach can be agreed before you invest
in an implementation.

## Setup

Follow the [development setup guide](docs/development/setup.md) for the editable build
(Cargo workspace + PyO3 extension via Maturin) and test invocations.

## Checks a pull request must pass

CI runs three jobs; all are reproducible locally:

- **rust** — `cargo fmt --check`, `cargo clippy --all-targets --all-features -- -D warnings`,
  `cargo test -p reinfors-core -p reinfors-games`
- **python** — `pytest` against an editable build
- **docs** — `python scripts/generate_docs.py --check`, `python scripts/check_docs.py`, and a
  strict `mkdocs build`

Typed Python surfaces must satisfy both mypy and pyright. Installing the repo's
[pre-commit](https://pre-commit.com) hooks (`pre-commit install`) covers the formatting and
lint gates at commit time.

## Code conventions

- Comments are reserved for constraints the code cannot express; explanations live in `docs/`.
  Before adding a comment, ask what breaks if it is deleted — if the answer is nothing, leave
  it out.
- Every public Python input must raise an ordinary Python exception, never a Rust panic.
  Constructors registered in the handle modules are enrolled in the adversarial sweep
  automatically; new parameters with non-obvious types fail test collection until they are
  given an edge-value bank (see `tests/test_constructor_validation.py`).

## Adding a game

A new game is a `Game` implementation plus one entry in each relevant registry
(`python/reinfors/games.py`, the encoder registry, `catalog.py`); the registry parity asserts
and the sweep pick it up from there. The full walkthrough, including the invariants a game
must uphold and how the engine consumes chance, is in
[extending reinfors](docs/extending/rust-components.md). Games with an external reference
implementation should ship a parity test against it, following the existing
`tests/test_*_parity.py` suites.

## Adding a policy or learner

Policies and learners usually ship as a pair, meeting at the policy's evaluation type. The
walkthrough is [add a policy or learner](docs/extending/policies-and-learners.md): the two
core traits (capability claims are deliberate, with no defaults), the per-(policy, learner)
composition arm in the binding, and the self-enforcing registration — the sweep fails test
collection until a new constructor's parameters are banked and its composition hook exists.
State the algorithm's reference in `catalog.py` and its row in the compatibility matrix.

## Licensing

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion
in this work shall be dual-licensed under MIT OR Apache-2.0, without any additional terms or
conditions.
