# Reproducing benchmarks

Reproduction commands and downloadable artifacts will be added with the new benchmark
results. The target workflow is:

```text
checkout exact commit
→ build the release wheel
→ capture system metadata
→ run correctness preflight
→ execute a versioned benchmark manifest
→ write raw JSON/CSV
→ regenerate tables and figures
```

The benchmark harness will live outside prose pages so commands, configs, and parsers can be
tested. This page will remain a short entry point to those artifacts rather than duplicating
them.

Current status: methodology defined; A10G measurements and publication artifacts pending.
