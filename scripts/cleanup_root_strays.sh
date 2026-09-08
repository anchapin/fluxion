#!/usr/bin/env bash
# scripts/cleanup_root_strays.sh
#
# Operator-action helper for issue #3438 — delete stale root transient
# artifacts that trip ``scripts/check_root_hygiene.py`` on the operator's
# local checkout. The two known strays at the time of writing are:
#
#   - ``case_900ff_profile_hourly.csv`` — 315.5K stale sensitivity-profile
#     dump from a CLI sensitivity run predating the #3303 fix (the
#     sensitivity command no longer writes scratch reports to cwd; it now
#     writes to ``tmp/``).
#   - ``hourly_output.csv`` — 191B stale hourly timeseries from the same
#     pre-#3303 CLI sensitivity run.
#
# These files live only on the operator's machine (they are untracked and
# matched by ``.gitignore``'s ``*_profile_hourly.csv`` and
# ``hourly_output*.csv`` patterns). They cannot be cleaned by a PR; the
# fix is an operator action. This script makes that action scriptable
# and repeatable across wave-orchestrator sessions so future sweeps do
# not re-surface the same 3-failure noise.
#
# Safety policy (defensive, narrow):
#
#   1. Only files that match BOTH conditions are eligible for deletion:
#        a) file name at the repo root matches a known stale pattern
#           (the patterns below); AND
#        b) ``git check-ignore`` reports the file as gitignored (so the
#           operator has already declared it as not-source); AND
#        c) ``git ls-files --error-unmatch`` reports the file as
#           untracked (so we never delete anything the operator has
#           committed; tracked-but-ignored files require ``git rm
#           --cached`` instead — see issue #3076).
#   2. The script refuses to delete anything else, including:
#        - no-extension blobs (potential compiled binaries — issue #2466);
#        - tracked files (recoverable from git, but never auto-deleted);
#        - files matching a blocked extension that the operator has not
#          explicitly gitignored (likely real scratch — operator decides).
#   3. Default mode is dry-run; ``--apply`` performs the deletions.
#
# Usage::
#
#     ./scripts/cleanup_root_strays.sh             # dry-run, prints plan
#     ./scripts/cleanup_root_strays.sh --list     # print only the candidate paths
#     ./scripts/cleanup_root_strays.sh --apply     # actually delete eligible strays
#     ./scripts/cleanup_root_strays.sh --apply --yes   # skip the "press enter" prompt
#
# Exit codes:
#     0 — clean: no eligible strays found (or all deleted under --apply).
#     1 — eligible strays remain after --apply (should not happen — bail).
#     2 — tool / git error (not in a git repo, git missing, ...).

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "${REPO_ROOT}" ]]; then
    echo "ERROR: not inside a Git working tree." >&2
    exit 2
fi
cd "${REPO_ROOT}"

# Patterns are the two operator strays from issue #3438 plus future
# variants that match the same gitignore classes already pinned at
# .gitignore lines 170 (`hourly_output*.csv`) and 220 (`*_profile_hourly.csv`).
# Keep these in sync with .gitignore.
KNOWN_STRAY_PATTERNS=(
    "case_900ff_profile_hourly.csv"
    "hourly_output.csv"
)

MODE="dry-run"
ASSUME_YES="false"
LIST_ONLY="false"

for arg in "$@"; do
    case "${arg}" in
        --apply)
            MODE="apply"
            ;;
        --list)
            LIST_ONLY="true"
            ;;
        --yes|-y)
            ASSUME_YES="true"
            ;;
        -h|--help)
            sed -n '2,55p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "ERROR: unknown flag: ${arg}" >&2
            echo "Run with --help for usage." >&2
            exit 2
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Classify each candidate by the three safety predicates.
# ---------------------------------------------------------------------------

declare -a ELIGIBLE=()
declare -a SKIPPED=()

classify() {
    local name="$1"
    local path="${REPO_ROOT}/${name}"

    if [[ ! -f "${path}" ]]; then
        return 0
    fi

    local ignored_rc=0
    git check-ignore --quiet -- "${name}" 2>/dev/null || ignored_rc=$?
    local is_gitignored="false"
    if [[ "${ignored_rc}" -eq 0 ]]; then
        is_gitignored="true"
    fi

    local tracked_rc=0
    git ls-files --error-unmatch -- "${name}" >/dev/null 2>&1 || tracked_rc=$?
    local is_tracked="false"
    if [[ "${tracked_rc}" -eq 0 ]]; then
        is_tracked="true"
    fi

    if [[ "${is_gitignored}" == "true" && "${is_tracked}" == "false" ]]; then
        ELIGIBLE+=("${name}")
    else
        local reason
        if [[ "${is_tracked}" == "true" ]]; then
            reason="tracked (use 'git rm --cached' instead)"
        else
            reason="not in .gitignore (operator decision required)"
        fi
        SKIPPED+=("${name}: ${reason}")
    fi
}

for pattern in "${KNOWN_STRAY_PATTERNS[@]}"; do
    classify "${pattern}"
done

# ---------------------------------------------------------------------------
# Reporting + action.
# ---------------------------------------------------------------------------

if [[ "${LIST_ONLY}" == "true" ]]; then
    printf '%s\n' "${ELIGIBLE[@]+"${ELIGIBLE[@]}"}"
    exit 0
fi

echo "=== Root Stray Cleanup (issue #3438) ==="
echo "Repo: ${REPO_ROOT}"
echo "Mode: ${MODE}"
echo

if [[ ${#ELIGIBLE[@]} -eq 0 && ${#SKIPPED[@]} -eq 0 ]]; then
    echo "PASS: no eligible root strays found."
    exit 0
fi

if [[ ${#ELIGIBLE[@]} -gt 0 ]]; then
    echo "Eligible for cleanup (gitignored, untracked, matches a known stale name):"
    for f in "${ELIGIBLE[@]}"; do
        echo "  - ${f}"
    done
    echo
fi

if [[ ${#SKIPPED[@]} -gt 0 ]]; then
    echo "Skipped (NOT eligible for auto-cleanup):"
    for s in "${SKIPPED[@]}"; do
        echo "  - ${s}"
    done
    echo
fi

if [[ "${MODE}" == "dry-run" ]]; then
    echo "DRY-RUN: no files were deleted. Re-run with --apply to clean."
    echo "After --apply, rerun: python3 scripts/check_root_hygiene.py"
    exit 0
fi

# MODE == "apply"
if [[ "${ASSUME_YES}" != "true" ]]; then
    echo "About to delete ${#ELIGIBLE[@]} file(s). Press Enter to continue, Ctrl-C to abort."
    read -r _
fi

deleted=0
for f in "${ELIGIBLE[@]}"; do
    if rm -- "${f}"; then
        echo "  deleted: ${f}"
        deleted=$((deleted + 1))
    else
        echo "  FAILED to delete: ${f}" >&2
    fi
done

echo
echo "Cleanup complete: ${deleted}/${#ELIGIBLE[@]} file(s) deleted."
echo "Verify gate: python3 scripts/check_root_hygiene.py"

# Re-classify so the operator sees whether anything remains.
remaining=0
for f in "${ELIGIBLE[@]}"; do
    if [[ -f "${REPO_ROOT}/${f}" ]]; then
        remaining=$((remaining + 1))
    fi
done

if [[ ${remaining} -gt 0 ]]; then
    echo "WARNING: ${remaining} file(s) still present after rm (file lock?)." >&2
    exit 1
fi
exit 0