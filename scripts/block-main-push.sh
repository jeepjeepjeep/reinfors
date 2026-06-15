#!/usr/bin/env bash
# pre-push guard: reject direct pushes to main — changes must go through a pull request.
#
# Wired in via pre-commit's pre-push stage (see .pre-commit-config.yaml), so it coexists with the
# pre-commit-stage hooks. pre-commit sets PRE_COMMIT_REMOTE_BRANCH to the ref being pushed to.
#
# This is a client-side guard: it only runs in clones that installed the pre-push hook
# (`pre-commit install --hook-type pre-push`) and can be bypassed with `git push --no-verify`.
# It stops accidental direct pushes; it is not a server-side guarantee.

if [ "${PRE_COMMIT_REMOTE_BRANCH:-}" = "refs/heads/main" ]; then
  echo "" >&2
  echo "  ✗ Direct pushes to 'main' are blocked — open a pull request instead." >&2
  echo "    (emergency override: git push --no-verify)" >&2
  echo "" >&2
  exit 1
fi
exit 0
