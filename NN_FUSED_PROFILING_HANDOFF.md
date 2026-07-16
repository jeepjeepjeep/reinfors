# Handoff: reinfors-nn fused-training performance investigation

**Date:** 2026-07-15 · **Branch (both repos on):** `reinfors-nn`
**Repos:** `/Users/adam/reinforcement_learning/reinfors` (default) · `/Users/adam/reinforcement_learning/snake_RL`

This file exists because the previous session gave a run of sloppy/partly-wrong answers and the user
(rightly) reset context. Read it skeptically; where it states a fact it was measured, and the known
mistakes are called out so you don't repeat them.

---

## The question being investigated

`snake_RL/scripts/train_reinfors_fast.py` (trains reinfors' **candle** Rust-native net fully in Rust via
`engine.train`) runs **~2× slower** than `snake_RL/scripts/train_reinfors.py` (uses reinfors only for data
generation; snake_RL's **torch** net does the learning, driven via `engine.collect` + a Python `infer`
callback). The user's numbers: ~8k collect rec/s (slow/torch) vs ~4k (fast/candle). This "shouldn't"
happen — the fused path pays **zero** Python/GIL/boundary cost, so where does the time go?

The user upgraded macOS 14→26 specifically because we (wrongly) believed candle conv was "slow on CPU,
fast on GPU." It is **not** — see findings.

---

## Current state of the code

### Uncommitted changes (branch `reinfors-nn`) — KEEP-OR-REVERT DECISION NEEDED
Two files, both adding **backend-neutral infer timing telemetry**. This is genuinely useful profiling
infrastructure regardless of the candle verdict. Recommend keeping; get user confirmation.

- `crates/reinfors-core/src/engine.rs`: `CollectStats` gained `infer_seconds: f64`, `infer_calls:
  usize`, `infer_rows: usize`. Inside `collect()`, the `infer` closure is wrapped once in a `timed`
  closure that brackets every forward invocation with `Instant::now()` and accumulates seconds/calls/rows.
  (NLL releases the wrapper's borrows after the loop so the totals are read into `stats` afterward.)
- `crates/reinfors-py/src/lib.rs`: `build_telemetry` emits `infer_seconds`/`infer_calls`/`infer_rows`.
  Because all three collect paths (`run_collect` callback, `run_collect_native`, `train_native`) funnel
  through `build_telemetry`, **`engine.train`'s per-collect telemetry dicts already carry these fields.**

Everything else on `reinfors-nn` is committed (HEAD `711ae53` "nn: guard empty batch in trainer.update;
drop dead code + stale gitignore"). `python/reinfors/nn.py` was edited then **reverted** this session —
it is back at HEAD (see "known-stale docs" below).

### Build state — IMPORTANT GOTCHA
Installed via: `cd reinfors && .venv/bin/maturin develop --release --features nn-metal,nn-accelerate`
(macOS is now 26.5.2, so candle Metal initializes — the old `MTLResidencySetDescriptor` panic is gone).
The compiled `.so` lives at `reinfors/python/reinfors/_reinfors.abi3.so` and is shared editable by both
`reinfors/.venv` and `snake_RL/.venv`.

**GOTCHA:** running any `uv`/`uvx` command *inside the reinfors dir* triggers a `uv` project resync that
**rebuilds reinfors with DEFAULT features**, silently dropping `nn-metal`/`nn-accelerate` (you'll then get
`ValueError: built without Metal support`, and CPU matmul slows down). To test without clobbering:
- use `reinfors/.venv/bin/python` **directly** (not `uv run`), or
- `uv run` from the **snake_RL** dir (does not rebuild reinfors).
After any clobber, rerun the maturin command above.

---

## What was actually measured (trust these)

Harness: `snake_RL/scripts/profile_collect_SCRATCH.py` (untracked; copy of the scratch tool).
Run: `uv run python scripts/profile_collect_SCRATCH.py configs/ensemble_treestrap.yaml --reps 6`
Config: snake, obs_shape (5,20,20), 3 actions, 10 heads, collect_size 4096 → ~5040 records/collect,
~46 forward calls/collect, ~1030 rows/call. **6 collects each, mean ± std.**

| config | throughput (rec/s) | wall ms | infer ms | search ms | **infer µs/row** |
|---|---|---|---|---|---|
| torch [mps]  | 9141 ± 1736 | 575 ± 125 | 433 ± 128 | 142 ± 12 | **8.92** |
| torch [cpu]  | 2415 ± 34   | 2088 ± 30 | 1961 ± 30 | 127 ± 1  | **40.25** |
| candle [cpu] | 5070 ± 103  | 994 ± 20  | 865 ± 20  | 129 ± 2  | **18.82** |
| candle [metal] | 2384 ± 69 | 2116 ± 62 | 1973 ± 62 | 143 ± 4 | **41.09** |

torch [mps] callback internals: host→device 52, forward 351 (7.22 µs/row), device→host 27,
boundary/GIL (= Rust-measured infer − Python-measured callback body) **3.4 ± 0.3 ms**.

### Metric definitions (so you don't misread the table)
- Two clocks: **Clock A** = Rust `Instant` bracketing each forward inside `collect` → `infer_seconds`.
  **Clock B** = Python `perf_counter` around the whole `engine.collect` FFI call → `wall`.
- `infer` = `infer_seconds` (Clock A) = total time the search spent in the net forward per collect.
  candle path: pure candle Rust forward (incl. its `to_cpu` sync). torch path: the **whole** pyo3
  crossing + the Python callback body (numpy marshaling + torch).
- `search` = `wall − infer` = everything that isn't the forward (game sim, tree expansion/backup, target
  assembly, record building, **plus** the outer pyo3 call + result marshaling). It is *not* pure search.
- `boundary/GIL` = `infer_seconds − py_callback_total` = the pyo3+numpy marshaling the fused path avoids.
- `infer µs/row` = `infer_seconds / infer_rows` — the cleanest cross-backend number (intrinsic to
  net+device, independent of batching/record count). **A "row" = one state forwarded; a "record" = one
  training example. They differ ~9× (47k rows vs 5040 records/collect).**

### Conclusions that hold up
1. **candle CPU conv is ~2.1× FASTER than torch CPU conv** (18.82 vs 40.25 µs/row). The earlier claim
   "candle conv is slow on CPU" was WRONG.
2. **torch on MPS (8.92) is the fastest.** That, and only that, is why `train_reinfors.py` wins:
   torch's MPS gives a ~5× conv speedup (40→7 µs/row).
3. **candle Metal is a REGRESSION vs candle CPU** (41 vs 18.82 µs/row, 2.2× slower). For candle, GPU is a
   *net loss* on this conv — opposite of torch. This is the real gap.
4. **Python boundary is ~0.6% (3.4 ms / 575 ms).** The fused architecture's speed advantage over the
   callback is negligible; the 2× gap is entirely the net forward (backend + device), not fused-vs-callback.
5. **search cost is ~127–143 ms across all four** (±1–12 ms; ~12% spread from trajectory divergence —
   different net weights → different game lengths / SelectiveExpectimax pruning). Not the driver.

---

## THE METHODOLOGICAL FLAW to fix first (this is why the user reset)

The harness profiled the candle path with **`engine.collect(n, cnet)`**. But `train_reinfors_fast.py`'s
**whole point** is that it uses **`engine.train`** — a fully-fused Rust loop with **no `engine.collect`
call from Python and no records marshaled to Python**. So the harness did not measure what the fast script
actually runs.

Mitigating facts: `infer_seconds` (the metric everything rests on) is identical whether driven by
`collect` or `train` (same `net.forward`, same search). The marshaling the fused loop saves lands in the
`search` bucket. So the *conclusions above still hold*. But the user is correct that a faithful profile
must drive `engine.train`.

**Fix:** re-profile the candle path via `engine.train(trainer, steps=N, collect_size=4096, batch_size=…,
reuse=1.0)`. Its returned per-collect dicts **already include** `infer_seconds`/`infer_calls`/`infer_rows`
(via `build_telemetry`) plus `collect_seconds` (Rust-measured, excludes grad steps) and `losses`. So you
can decompose the *real* fused loop:
  - collect wall (Rust) = `collect_seconds`
  - forward = `infer_seconds`; search = `collect_seconds − infer_seconds`
  - grad-step time = (sum over dicts of wall) − Σ`collect_seconds`, or time it separately.
No Python callback exists on this path, so there is no boundary term to measure (that's the point).
For the torch comparison, keep using `engine.collect` + the instrumented callback (it has no `train`
equivalent — snake_RL owns that net's optimizer).

---

## Web research on candle Metal (done 2026-07-15; treat as "is there a fix?", not as primary evidence)

The core fact (candle Metal conv slow) is **measured by us**, not taken from the web — important, because
last session I wrongly extrapolated an online "conv slow" claim into "slower than torch," which our data
disproved. What the web adds: **no known conv2d-on-Metal fix exists.**
- Open, unresolved: [#2659](https://github.com/huggingface/candle/issues/2659) (Metal forward time
  degrades over iterations — but our runs are stable, likely not our issue),
  [#1780](https://github.com/huggingface/candle/issues/1780) (`to_cpu` slow, traced to
  `wait_until_completed` command-buffer flush — consistent with our per-call sync cost, since the search
  pulls values to host every round), [#3052](https://github.com/huggingface/candle/issues/3052) /
  [#1139](https://github.com/huggingface/candle/issues/1139) (general "Metal ~5× slower than torch").
- Shipped Metal perf work ([PR #2615](https://github.com/huggingface/candle/pull/2615)) targets
  **quantized LLM matmul**, not conv. Upgrading candle is unlikely to fix conv-on-Metal.
- CPU conv issue [#3119](https://github.com/huggingface/candle/issues/3119) is real but about candle-CPU
  being slower *than it could be*, NOT slower than torch (our measurement: candle-CPU beats torch-CPU).

---

## Other real findings / latent bugs
- **batch-1 Metal crash:** `net.forward` with n==1 on Metal panics — candle's `gemv` kernel fails to
  compile on this toolchain (`use of undeclared identifier 'bfloat'`), an `.unwrap()` at
  `crates/reinfors-nn/src/lib.rs:195`. Batch ≥2 uses `gemm` and works, so training (pooled batches ~1030)
  never hits it, but it's a latent hard crash. Would be a clean-error fix.
- **known-stale docs:** `python/reinfors/nn.py` at HEAD still claims "candle CPU conv2d is slow (~8-10x
  behind PyTorch)… for a conv net prefer GPU." Both halves are now contradicted by measurement
  (candle-CPU beats torch-CPU; candle-Metal is a regression). Left unchanged because the user said
  **"Don't update docs"** and considers the fused path "completely pointless" as-is. Flag for a decision;
  do not silently "fix" into another positive-spin claim.

---

## User's stance & open decision
The user's current read: **"the fused path is completely pointless… it is clearly still broken."** For a
conv net on GPU (where real training runs) that is fair: candle Metal is a regression with no upstream
fix. candle-CPU-beats-torch-CPU + no-libtorch-dependency is real but the user does not consider it worth
positioning. Do **not** re-litigate this with "but it's faster in case X" unless you have new measured
evidence. Likely next directions the user may want (ask, don't assume):
  (a) re-profile via `engine.train` to close the methodology gap and confirm the numbers, then decide;
  (b) drop/shelve the fused-conv path; keep the telemetry hook;
  (c) investigate whether candle Metal conv can be made fast (uncertain, no upstream help — likely not
      worth it);
  (d) correct or remove the stale nn.py perf docs.

---

## Workflow constraints (from user memory — obey)
- Commit email `adamianroberts@users.noreply.github.com`; the real gmail is blocked (GH007).
- Concise commit messages: terse subject, usually no body, **no Co-Authored-By trailer**.
- Default to the **reinfors** repo unless told otherwise; ask if ambiguous.
- **Measure, don't reason** about perf; benchmark **release** builds only.
- Minimal comments/docstrings (only non-obvious "why").
- Commit/push only when the user asks.

## Artifacts
- Handoff: this file (`reinfors/NN_FUSED_PROFILING_HANDOFF.md`, untracked).
- Harness: `snake_RL/scripts/profile_collect_SCRATCH.py` (untracked) — **measures `collect`; must be
  extended to drive `engine.train` for the candle path per the flaw above.**
