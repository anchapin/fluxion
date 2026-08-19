#!/usr/bin/env bash
# scripts/cleanup_stale_worktrees.sh
#
# Cleanup stale worktrees and fix/issue-* branches from prior orchestration
# sessions (issue #3069).
#
# The wave-orchestrator spawns one worktree per `fix/issue-*` branch and
# only cleans each one up if the wave explicitly finishes. Aborted waves
# leave behind a worktree + branch pair every time; the 2026-08-16 wave
# exposed 470 stale `fix/issue-*` branches and 23 stale worktrees from
# prior sessions, costing ~10 GB of disk pressure. This script classifies
# every cleanup target with deterministic safety checks and prints a plan
# that the operator can audit before any mutation. --apply performs the
# deletions.
#
# Usage:
#   ./scripts/cleanup_stale_worktrees.sh [--apply] [--keep-empty-commits]
#                                        [--keep-unmerged] [--json]
#                                        [--output <path>]
#
# Modes:
#   (default)     Dry-run. Prints a plan to stdout; exits 0 if all targets
#                 are safe to delete, 2 if some targets would be skipped.
#   --apply       Actually delete worktrees / branches / remote branches.
#
# Filters:
#   --keep-empty-commits   Skip branches whose only divergence from develop
#                          is an empty commit (default: delete — matches
#                          the #3069 e265c62 pattern).
#   --keep-unmerged        Keep unmerged branches (no-op; default already
#                          skips unmerged branches). Accepted for symmetry
#                          with --keep-empty-commits and future-proofing.
#
# Output:
#   (default)     Human-readable summary to stdout.
#   --json        JSON-only output to stdout (for CI / audit pipelines).
#   --output <p>  Write the JSON report to <p> in addition to stdout.
#
# Exit codes:
#   0 = Clean dry-run: all targets safe to delete (would all succeed).
#   1 = Errors (not a git repo, no develop branch, git failures, etc.).
#   2 = Some targets would be skipped (unmerged, unpushed, protected).
#
# Idempotency: safe to re-run. Already-deleted targets are skipped silently.
#
# Safety:
#   - main / develop are NEVER deleted
#   - The current worktree is NEVER deleted
#   - Any branch with commits not reachable from $BASE_BRANCH is skipped
#     (race-free merge-base check, see has_unpushed_commits; #3119).
#     Missing origin/<branch> tracking does NOT by itself cause a skip —
#     the script verifies the local tip == merge-base with $BASE_BRANCH
#     so a tracking-ref deletion between dry-run and --apply does not
#     reclassify a genuinely-merged branch.
#   - The default mode is dry-run; --apply is required for any mutation
#   - Every action is printed before being taken
#
# See docs/agents/worktree-cleanup.md for the operator guide.

set -euo pipefail

# ---- argument parsing -------------------------------------------------------

APPLY=false
KEEP_EMPTY_COMMITS=false
KEEP_UNMERGED=false
JSON_OUTPUT=false
OUTPUT_PATH=""

print_usage() {
    sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed -e '/^set -euo/d' | head -n -1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --apply)
            APPLY=true
            shift
            ;;
        --keep-empty-commits)
            KEEP_EMPTY_COMMITS=true
            shift
            ;;
        --keep-unmerged)
            KEEP_UNMERGED=true
            shift
            ;;
        --json)
            JSON_OUTPUT=true
            shift
            ;;
        --output)
            OUTPUT_PATH="${2:-}"
            if [[ -z "$OUTPUT_PATH" ]]; then
                echo "ERROR: --output requires a path" >&2
                exit 1
            fi
            shift 2
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            echo "ERROR: unknown argument: $1" >&2
            print_usage >&2
            exit 1
            ;;
    esac
done

# ---- preconditions ----------------------------------------------------------

if ! command -v git > /dev/null 2>&1; then
    echo "ERROR: git not found in PATH" >&2
    exit 1
fi

if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo "ERROR: not a git repository (no .git directory)" >&2
    exit 1
fi

# Find the main repo root. `git rev-parse --git-common-dir` returns the
# shared .git/ directory shared across all worktrees; the main repo root is
# its parent. This works whether the script is invoked from the main repo
# or from any of its worktrees.
GIT_COMMON_DIR="$(git rev-parse --git-common-dir)"
case "$GIT_COMMON_DIR" in
    /*) ;;
    *)  GIT_COMMON_DIR="$(pwd)/$GIT_COMMON_DIR" ;;
esac
GIT_COMMON_DIR="$(cd "$GIT_COMMON_DIR" && pwd)"
MAIN_REPO_ROOT="$(cd "$GIT_COMMON_DIR/.." && pwd)"

CURRENT_DIR="$(pwd)"
CURRENT_TOPLEVEL="$(git rev-parse --show-toplevel 2>/dev/null || echo "$CURRENT_DIR")"
CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "HEAD")"

# Determine the base branch (develop > origin/develop > main > origin/main).
if git -C "$MAIN_REPO_ROOT" rev-parse --verify develop > /dev/null 2>&1; then
    BASE_BRANCH="develop"
elif git -C "$MAIN_REPO_ROOT" rev-parse --verify origin/develop > /dev/null 2>&1; then
    BASE_BRANCH="origin/develop"
elif git -C "$MAIN_REPO_ROOT" rev-parse --verify main > /dev/null 2>&1; then
    BASE_BRANCH="main"
elif git -C "$MAIN_REPO_ROOT" rev-parse --verify origin/main > /dev/null 2>&1; then
    BASE_BRANCH="origin/main"
else
    echo "ERROR: no develop or main branch (local or origin)" >&2
    exit 1
fi

# Allowed extra worktree prefixes (always skipped — see AGENTS.md
# "Local-only runtime dirs" for context).
SUPERSET_PREFIX="${HOME}/.superset"
PLANNING_PREFIX="${HOME}/.planning/worktrees"

# ---- helpers ----------------------------------------------------------------

# is_merged_into_base <branch>
#   Returns 0 if <branch> is fully merged into $BASE_BRANCH (no commits
#   on <branch> not on $BASE_BRANCH).
is_merged_into_base() {
    local branch="$1"
    local diff
    diff="$(git -C "$MAIN_REPO_ROOT" log "${BASE_BRANCH}..${branch}" --oneline 2>/dev/null || true)"
    [[ -z "$diff" ]]
}

# has_empty_commit_only <branch>
#   Returns 0 if <branch> has commits not on $BASE_BRANCH, but the diff
#   between $BASE_BRANCH and <branch> is empty (matches the #3069
#   e265c62 pattern: a branch that exists but contributes no changes).
has_empty_commit_only() {
    local branch="$1"
    local log
    log="$(git -C "$MAIN_REPO_ROOT" log "${BASE_BRANCH}..${branch}" --oneline 2>/dev/null || true)"
    if [[ -z "$log" ]]; then
        return 1  # merged, not empty-commit
    fi
    # Diff against the merge base. `git diff --quiet base..branch` returns
    # 0 only when there are no file changes anywhere along the path.
    git -C "$MAIN_REPO_ROOT" diff --quiet "${BASE_BRANCH}..${branch}" -- 2>/dev/null
}

# has_unpushed_commits <branch>
#   Returns 0 if <branch> has commits not on $BASE_BRANCH. The check is
#   race-free (issue #3119): it does NOT treat "no origin/<branch>
#   tracking ref" as a blanket skip, because that state can flip between
#   the dry-run and the --apply run (e.g. operator runs `git update-ref
#   -d`, `git fetch --prune`, or a CI job prunes the ref out-of-band).
#   Instead:
#
#     1. If origin/<branch> exists and is an ancestor of $BASE_BRANCH
#        → the remote state is fully merged, no unpushed commits.
#     2. If origin/<branch> is missing AND the local branch tip equals
#        merge-base($BASE_BRANCH, <branch>) → the local branch is a
#        stale mirror of develop (no unique commits), safe to delete.
#     3. Otherwise (origin/<branch> ahead of $BASE_BRANCH with local
#        commits past remote, or local branch has unique commits not on
#        $BASE_BRANCH) → returns 0 (has unpushed), skipped.
#
#   This matches the issue's recommended `merge-base --is-ancestor`
#   check: a branch is safe to delete iff every reachable commit is
#   already in $BASE_BRANCH's history.
has_unpushed_commits() {
    local branch="$1"
    local remote_branch="origin/${branch}"

    # Case 1: remote tracking ref exists.
    if git -C "$MAIN_REPO_ROOT" rev-parse --verify "$remote_branch" > /dev/null 2>&1; then
        if git -C "$MAIN_REPO_ROOT" merge-base --is-ancestor \
                "$remote_branch" "$BASE_BRANCH" > /dev/null 2>&1; then
            # Remote tip is in $BASE_BRANCH's history. By the time we get
            # here, the caller has already established (via
            # is_merged_into_base / has_empty_commit_only) that the
            # local branch tip is also reachable from $BASE_BRANCH, so
            # the whole branch is in develop and safe to delete.
            return 1
        fi
        # Remote exists but is ahead of $BASE_BRANCH — fall back to the
        # original log-based check: are there commits on $branch not on
        # origin/<branch>?
        local diff
        diff="$(git -C "$MAIN_REPO_ROOT" log "${remote_branch}..${branch}" --oneline 2>/dev/null || true)"
        [[ -n "$diff" ]]
        return
    fi

    # Case 2: no remote tracking ref. Resolve the local branch tip and
    # compare it to merge-base($BASE_BRANCH, <branch>). Equal tips mean
    # the local branch has no unique commits beyond develop, so it is
    # safe to delete even though the tracking ref is gone.
    if ! git -C "$MAIN_REPO_ROOT" rev-parse --verify "$branch^{commit}" > /dev/null 2>&1; then
        return 0  # branch doesn't exist locally — treat as unpushed
    fi
    local common branch_tip
    common="$(git -C "$MAIN_REPO_ROOT" merge-base "$BASE_BRANCH" "$branch" 2>/dev/null || true)"
    branch_tip="$(git -C "$MAIN_REPO_ROOT" rev-parse "$branch^{commit}" 2>/dev/null || true)"
    if [[ -n "$common" ]] && [[ "$branch_tip" == "$common" ]]; then
        return 1  # branch tip == merge-base → no unique commits
    fi
    return 0  # branch has unique commits not on $BASE_BRANCH → skip
}

# json_escape <string>
#   Minimal JSON string escaping for report fields. Handles the four
#   characters that realistically appear in branch names + paths /
#   reasons (backslash, double quote, tab, newline, carriage return).
json_escape() {
    local s="${1-}"
    s="${s//\\/\\\\}"
    s="${s//\"/\\\"}"
    s="${s//	/\\t}"
    s="${s//$'\n'/\\n}"
    s="${s//$'\r'/\\r}"
    printf '%s' "$s"
}

# ---- plan construction ------------------------------------------------------

# Parallel arrays keep the dispatch in pure bash. Each plan item corresponds
# to one target (worktree or branch) and is rendered as a JSON object.
declare -a PLAN_KIND
declare -a PLAN_TARGET
declare -a PLAN_BRANCH
declare -a PLAN_ACTION
declare -a PLAN_REASON
declare -a PLAN_CMDS

WT_PATHS=()
WT_BRANCHES=()
WT_IS_MAIN=()

current_path=""
while IFS= read -r line; do
    case "$line" in
        "worktree "*)
            current_path="${line#worktree }"
            ;;
        "branch "*)
            current_branch="${line#branch refs/heads/}"
            if [[ -n "${current_path:-}" && -n "${current_branch:-}" ]]; then
                WT_PATHS+=("$current_path")
                WT_BRANCHES+=("$current_branch")
                if [[ ${#WT_PATHS[@]} -eq 1 ]]; then
                    WT_IS_MAIN+=(1)
                else
                    WT_IS_MAIN+=(0)
                fi
                current_path=""
            fi
            ;;
    esac
done < <(git -C "$MAIN_REPO_ROOT" worktree list --porcelain)

# Discover fix/issue-* branches.
BRANCH_NAMES=()
while IFS= read -r branch; do
    branch="${branch#"${branch%%[![:space:]]*}"}"  # trim leading whitespace
    [[ -z "$branch" ]] && continue
    BRANCH_NAMES+=("$branch")
done < <(git -C "$MAIN_REPO_ROOT" branch --list 'fix/issue-*')

# Map branch → worktree path (for "skip — has worktree" cases).
declare -A BRANCH_TO_WT_PATH
for i in "${!WT_PATHS[@]}"; do
    BRANCH_TO_WT_PATH["${WT_BRANCHES[$i]}"]="${WT_PATHS[$i]}"
done

# Classify one worktree. Appends to the plan arrays.
classify_worktree() {
    local idx="$1"
    local path="${WT_PATHS[$idx]}"
    local branch="${WT_BRANCHES[$idx]}"
    local is_main="${WT_IS_MAIN[$idx]}"

    local action="skip"
    local reason=""

    if [[ "$is_main" -eq 1 ]]; then
        reason="main worktree"
    elif [[ -n "$SUPERSET_PREFIX" && "$path" == "$SUPERSET_PREFIX"* ]]; then
        reason="~/.superset/ external tool worktree (allowed)"
    elif [[ -n "$PLANNING_PREFIX" && "$path" == "$PLANNING_PREFIX"* ]]; then
        reason="~/.planning/worktrees/ agent runtime (allowed)"
    elif [[ "$path" == "$CURRENT_TOPLEVEL" ]]; then
        reason="current worktree"
    elif [[ "$branch" == "main" || "$branch" == "develop" ]]; then
        reason="protected branch"
    elif is_merged_into_base "$branch"; then
        if has_unpushed_commits "$branch"; then
            reason="merged into $BASE_BRANCH but has unpushed commits"
        else
            action="delete"
            reason="merged into $BASE_BRANCH"
        fi
    elif has_empty_commit_only "$branch"; then
        if has_unpushed_commits "$branch"; then
            reason="empty-commit branch with unpushed commits"
        elif $KEEP_EMPTY_COMMITS; then
            reason="empty-commit branch (kept by --keep-empty-commits)"
        else
            action="delete"
            reason="empty-commit branch (safe to delete per #3069 e265c62 pattern)"
        fi
    elif has_unpushed_commits "$branch"; then
        reason="unmerged with unpushed commits"
    else
        reason="unmerged into $BASE_BRANCH"
    fi

    PLAN_KIND+=("worktree")
    PLAN_TARGET+=("$path")
    PLAN_BRANCH+=("$branch")
    PLAN_ACTION+=("$action")
    PLAN_REASON+=("$reason")

    if [[ "$action" == "delete" ]]; then
        # Build the deletion command suite. Each segment is `;`-joined so the
        # later failures don't abort earlier ones (e.g. we still want to
        # attempt `git branch -D` even if `git worktree remove` fails).
        local wt_remove="git -C \"$MAIN_REPO_ROOT\" worktree remove --force \"$path\""
        local branch_delete="git -C \"$MAIN_REPO_ROOT\" branch -D \"$branch\""
        local remote_delete="git -C \"$MAIN_REPO_ROOT\" push origin --delete \"$branch\" 2>/dev/null || true"
        PLAN_CMDS+=("${wt_remove} 2>/dev/null || true; ${branch_delete} 2>/dev/null || true; ${remote_delete}")
    else
        PLAN_CMDS+=("")
    fi
}

# Classify one branch (no worktree).
classify_branch() {
    local branch="$1"

    local action="skip"
    local reason=""

    if [[ -n "${BRANCH_TO_WT_PATH[$branch]:-}" ]]; then
        reason="has worktree at ${BRANCH_TO_WT_PATH[$branch]} — handled by worktree cleanup"
    elif [[ "$branch" == "$CURRENT_BRANCH" ]]; then
        reason="current branch"
    elif [[ "$branch" == "main" || "$branch" == "develop" ]]; then
        reason="protected branch"
    elif is_merged_into_base "$branch"; then
        if has_unpushed_commits "$branch"; then
            reason="merged into $BASE_BRANCH but has unpushed commits"
        else
            action="delete"
            reason="merged into $BASE_BRANCH"
        fi
    elif has_empty_commit_only "$branch"; then
        if has_unpushed_commits "$branch"; then
            reason="empty-commit branch with unpushed commits"
        elif $KEEP_EMPTY_COMMITS; then
            reason="empty-commit branch (kept by --keep-empty-commits)"
        else
            action="delete"
            reason="empty-commit branch (safe to delete per #3069 e265c62 pattern)"
        fi
    elif has_unpushed_commits "$branch"; then
        reason="unmerged with unpushed commits"
    else
        reason="unmerged into $BASE_BRANCH"
    fi

    PLAN_KIND+=("branch")
    PLAN_TARGET+=("$branch")
    PLAN_BRANCH+=("$branch")
    PLAN_ACTION+=("$action")
    PLAN_REASON+=("$reason")

    if [[ "$action" == "delete" ]]; then
        local branch_delete="git -C \"$MAIN_REPO_ROOT\" branch -D \"$branch\""
        local remote_delete="git -C \"$MAIN_REPO_ROOT\" push origin --delete \"$branch\" 2>/dev/null || true"
        PLAN_CMDS+=("${branch_delete} 2>/dev/null || true; ${remote_delete}")
    else
        PLAN_CMDS+=("")
    fi
}

# Aggregate "skip" classifications that the operator should pay attention to.
# Only count branches/worktrees we'd ACTION on (not the always-skipped
# "main worktree" / "external worktree" cases) — those are scaffold, not
# blockers.
plan_skip_blocks() {
    local count=0
    for i in "${!PLAN_KIND[@]}"; do
        if [[ "${PLAN_ACTION[$i]}" != "delete" ]]; then
            # Only count "blocking" skips: unmerged or unpushed, not the
            # always-skipped allowed-external cases.
            case "${PLAN_REASON[$i]}" in
                "main worktree"|*"allowed"*|"current worktree"|"current branch"|"protected branch")
                    ;;
                *)
                    count=$((count + 1))
                    ;;
            esac
        fi
    done
    echo "$count"
}

# ---- run classification -----------------------------------------------------

for i in "${!WT_PATHS[@]}"; do
    classify_worktree "$i"
done

for branch in "${BRANCH_NAMES[@]}"; do
    classify_branch "$branch"
done

total=${#PLAN_KIND[@]}
delete_count=0
for action in "${PLAN_ACTION[@]}"; do
    if [[ "$action" == "delete" ]]; then
        delete_count=$((delete_count + 1))
    fi
done
skip_count=$((total - delete_count))
blocking_skips=$(plan_skip_blocks)

# ---- build JSON report ------------------------------------------------------

now_iso="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

json_items=""
for i in "${!PLAN_KIND[@]}"; do
    item=$(printf '{"kind":"%s","target":"%s","branch":"%s","action":"%s","reason":"%s"}' \
        "$(json_escape "${PLAN_KIND[$i]}")" \
        "$(json_escape "${PLAN_TARGET[$i]}")" \
        "$(json_escape "${PLAN_BRANCH[$i]}")" \
        "$(json_escape "${PLAN_ACTION[$i]}")" \
        "$(json_escape "${PLAN_REASON[$i]}")")
    if [[ -z "$json_items" ]]; then
        json_items="$item"
    else
        json_items="${json_items},${item}"
    fi
done

json_report=$(cat <<EOF
{
  "timestamp": "${now_iso}",
  "apply": ${APPLY},
  "keep_empty_commits": ${KEEP_EMPTY_COMMITS},
  "keep_unmerged": ${KEEP_UNMERGED},
  "base_branch": "${BASE_BRANCH}",
  "main_repo_root": "${MAIN_REPO_ROOT}",
  "current_worktree": "${CURRENT_TOPLEVEL}",
  "summary": {
    "total": ${total},
    "delete": ${delete_count},
    "skip": ${skip_count},
    "blocking_skips": ${blocking_skips}
  },
  "plan": [${json_items}]
}
EOF
)

# ---- emit output ------------------------------------------------------------

if $JSON_OUTPUT; then
    echo "$json_report"
else
    cat <<EOF
============================================
  Stale Worktree/Branch Cleanup
============================================
timestamp:  ${now_iso}
mode:       $(if $APPLY; then echo "APPLY"; else echo "dry-run"; fi)
base_branch: ${BASE_BRANCH}
repo_root:  ${MAIN_REPO_ROOT}
keep_empty_commits: ${KEEP_EMPTY_COMMITS}
keep_unmerged:      ${KEEP_UNMERGED}

Summary:
  total targets:  ${total}
  would delete:   ${delete_count}
  would skip:     ${skip_count}
  blocking skips: ${blocking_skips}  (unmerged / unpushed branch cleanup targets)

EOF

    if [[ "$total" -gt 0 ]]; then
        echo "Plan:"
        for i in "${!PLAN_KIND[@]}"; do
            if [[ "${PLAN_ACTION[$i]}" == "delete" ]]; then
                marker="DEL "
            else
                marker="SKIP"
            fi
            printf "  %s [%s] %-58s [%s]\n      reason: %s\n" \
                "$marker" \
                "${PLAN_KIND[$i]}" \
                "${PLAN_TARGET[$i]}" \
                "${PLAN_BRANCH[$i]}" \
                "${PLAN_REASON[$i]}"
        done
    else
        echo "No worktrees or fix/issue-* branches found."
    fi

    echo ""
    if ! $APPLY; then
        echo "This was a dry-run. Re-run with --apply to execute the deletions."
    fi
fi

# ---- write report to file (if requested) ------------------------------------

if [[ -n "$OUTPUT_PATH" ]]; then
    mkdir -p "$(dirname "$OUTPUT_PATH")"
    printf '%s\n' "$json_report" > "$OUTPUT_PATH"
    if ! $JSON_OUTPUT; then
        echo ""
        echo "Report written to: ${OUTPUT_PATH}"
    fi
elif $APPLY && $JSON_OUTPUT; then
    # --apply + --json without --output: still write a default report so
    # the audit trail is preserved.
    default_report="${MAIN_REPO_ROOT}/target/cleanup_stale_worktrees_report.json"
    mkdir -p "$(dirname "$default_report")"
    printf '%s\n' "$json_report" > "$default_report"
fi

# ---- execute plan if --apply ------------------------------------------------

if $APPLY; then
    if [[ "$JSON_OUTPUT" != "true" ]]; then
        echo ""
        echo "Executing deletions..."
    fi
    for i in "${!PLAN_KIND[@]}"; do
        if [[ "${PLAN_ACTION[$i]}" == "delete" ]]; then
            cmds="${PLAN_CMDS[$i]}"
            if [[ -n "$cmds" ]]; then
                if [[ "$JSON_OUTPUT" != "true" ]]; then
                    echo "  [${PLAN_KIND[$i]}] ${PLAN_TARGET[$i]}: ${PLAN_REASON[$i]}"
                fi
                # Use bash -c so the multi-segment string is parsed as a
                # single shell command. Set +e locally so a failure on one
                # segment doesn't abort the whole cleanup run.
                (set +e; bash -c "$cmds") || true
            fi
        fi
    done
    if [[ "$JSON_OUTPUT" != "true" ]]; then
        echo "Done."
    fi
fi

# ---- exit code --------------------------------------------------------------

# 0 = all targets safe to delete (no blocking skips)
# 2 = some targets would be skipped (unmerged, unpushed, etc.)
if [[ "$blocking_skips" -gt 0 ]]; then
    exit 2
fi
exit 0
