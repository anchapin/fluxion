#!/usr/bin/env bash
# scripts/ci-local.sh — local pre-push CI validation via `act`
#
# Runs a curated subset of CI workflows locally before pushing to GitHub.
# Catches workflow-shape failures (wrong action refs, missing env, typos in
# step names) without burning a slot in the GH-hosted runner queue.
#
# Usage:
#   ./scripts/ci-local.sh                       # run the default suite
#   ./scripts/ci-local.sh fmt                   # single workflow
#   ./scripts/ci-local.sh scorecard docs        # multiple workflows
#   ./scripts/ci-local.sh -h                    # help
#
# What runs by default:
#   - scorecard-drift (Python-only, ~30s)
#   - docs-hygiene    (Python+Node, ~1m)
#   - architecture_drift (Python only, ~30s)
#   - scripts-tests   (pytest harness, ~1m)
#
# What does NOT run here:
#   - Anything that depends on self-hosted runner labels ([self-hosted,*])
#   - Rust matrix jobs (use `cargo test` directly for those)
#   - Anything with timeout-minutes > 10 (host RAM/time constraints)
#
# Caveats:
#   - macOS/Windows runners don't exist on Linux Docker; jobs targeting
#     those runners will SKIP locally (matches GH behavior, sorta).
#   - Some actions behave differently outside GH (e.g., checkout in detached
#     worktree). Expect some false positives — use the GH queue for the
#     authoritative run.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Always read .actrc from repo root for image pinning
if [ ! -f "$REPO_ROOT/.actrc" ]; then
  echo "WARN: .actrc missing in repo root — act will use defaults" >&2
fi

usage() {
  sed -n '2,30p' "$0" | sed -e 's/^# \{0,1\}//'
  exit 0
}

# Lightweight curated jobs
declare -A WORKFLOWS=(
  [fmt]="scorecard-drift.yml"
  [scorecard]="scorecard-drift.yml"
  [docs]="docs-hygiene.yml"
  [architecture]="architecture_drift.yml"
  [scripts]="scripts-tests.yml"
)

DEFAULT_SUITE=(fmt scorecard docs architecture scripts)

if [ $# -eq 0 ] || [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  usage
fi

# Check act is available
if ! command -v act >/dev/null 2>&1; then
  echo "ERR: 'act' not in PATH. Install: https://nektosact.com/installation/" >&2
  exit 2
fi

# Check docker is running
if ! docker info >/dev/null 2>&1; then
  echo "ERR: Docker daemon not reachable. Start Docker first." >&2
  exit 2
fi

# Build list of (workflow_file, job_name) pairs
TARGETS=()
for arg in "$@"; do
  if [ "${WORKFLOWS[$arg]+exists}" ]; then
    TARGETS+=("${WORKFLOWS[$arg]}")
  elif [[ "$arg" == *.yml ]]; then
    TARGETS+=("$arg")
  else
    echo "ERR: unknown workflow '$arg'. Options: ${!WORKFLOWS[*]} or path/to/*.yml" >&2
    exit 2
  fi
done

# Default to the suite
if [ ${#TARGETS[@]} -eq 0 ]; then
  for k in "${DEFAULT_SUITE[@]}"; do
    TARGETS+=("${WORKFLOWS[$k]}")
  done
fi

echo "==> Running act on: ${TARGETS[*]}"
echo

# Build the act command. We use:
#   -W <each workflow file>     restrict to listed workflows
#   --no-skip-checkout          make sure the checkout step runs (verify the SHA exists)
#   -q                           quiet (each job's stdout goes to its own container)
ARGS=( -q --no-skip-checkout )
for wf in "${TARGETS[@]}"; do
  ARGS+=( -W ".github/workflows/$wf" )
done

exec act "${ARGS[@]}"
