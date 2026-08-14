# Security policy

## Supported versions

Only the latest 0.x release receives fixes; reinfors is pre-1.0 (see the README's
[Stability](README.md#stability) section).

## Reporting a vulnerability

Report privately via GitHub's private vulnerability reporting: the **Security** tab of this
repository → **Report a vulnerability**. Please do not open public issues for security
reports. This is a solo-maintained project; reports are acknowledged on a best-effort basis,
normally within a week.

## Scope

The load-bearing boundary is the Python↔Rust surface: reinfors promises that no public
Python input — constructor arguments, method calls, config dicts, or snapshot bytes —
reaches a Rust panic, aborts the process, or triggers unbounded allocation. Reproducible
violations of that contract are in scope, as are memory-safety issues in the extension.
Slow or resource-hungry but bounded computation from extreme-but-valid parameters is not
considered a vulnerability.
