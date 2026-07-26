#!/usr/bin/env bash
# scripts/mutants_diff_files.sh
#
# Generate a unified diff of changed Rust source files for scoped mutation
# testing (Issue #1891, Solution B). The output is consumed by:
#
#   cargo mutants --in-diff <diff-file>
#
# `--in-diff` restricts mutation generation to only the *lines* touched by the
# diff, which is far more precise than `--file` (which would mutate every
# function in a changed file).
#
# Usage:
#   scripts/mutants_diff_files.sh [BASE_REF] [OUTPUT_FILE]
#
# Arguments:
#   BASE_REF     — git ref to diff against (default: origin/develop)
#   OUTPUT_FILE  — where to write the unified diff (default: mutants_diff.patch)
#
# Environment:
#   In GitHub Actions the checkout action must use fetch-depth: 0 (or at least
#   enough history to resolve the base ref) so the diff is available locally.
#
# Exit codes:
#   0 — diff written to OUTPUT_FILE (may be empty if no .rs files changed)
#   1 — git command failure
#
# Prints to stdout: the number of changed .rs files (for the workflow to decide
# whether to skip the mutation job entirely).

set -euo pipefail

BASE_REF="${1:-origin/develop}"
OUTPUT_FILE="${2:-mutants_diff.patch}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Ensure the base ref is resolvable. On a shallow CI checkout the remote ref
# may not be present; try to fetch it (best-effort — swallow errors).
if ! git rev-parse --verify "$BASE_REF" >/dev/null 2>&1; then
    BASE_BRANCH="${BASE_REF#origin/}"
    echo "info: base ref '$BASE_REF' not found locally; attempting fetch of '$BASE_BRANCH'…" >&2
    git fetch origin "$BASE_BRANCH" --depth=200 2>/dev/null || true
fi

# Fall back to merge-base comparison so only PR-introduced changes are captured
# (not commits already merged into the base since the branch point).
if git rev-parse --verify "$BASE_REF" >/dev/null 2>&1; then
    MERGE_BASE="$(git merge-base "$BASE_REF" HEAD 2>/dev/null || echo "$BASE_REF")"
    DIFF_RANGE="${MERGE_BASE}...HEAD"
else
    # Last resort: diff against HEAD~1 (single-commit fallback).
    echo "warn: could not resolve '$BASE_REF'; falling back to HEAD~1" >&2
    DIFF_RANGE="HEAD~1"
fi

# Collect the diff for .rs files under src/ (the main crate). We explicitly
# filter to Rust source so non-code changes (docs, configs) don't trigger a
# mutation run that has nothing to mutate.
git diff "$DIFF_RANGE" --diff-filter=d -- 'src/*.rs' 'src/**/*.rs' > "$OUTPUT_FILE" || true

# Count the number of changed .rs files.
# grep -c always prints a count; on zero matches it exits 1, so we capture
# stdout and strip any trailing newline instead of relying on exit codes.
CHANGED_FILES=$(git diff "$DIFF_RANGE" --diff-filter=d --name-only -- 'src/*.rs' 'src/**/*.rs' \
    | wc -l | tr -d ' ')

echo "$CHANGED_FILES"

if [[ "$CHANGED_FILES" -eq 0 ]]; then
    echo "info: no Rust source files changed under src/; diff is empty." >&2
fi
