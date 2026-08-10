#!/usr/bin/env bash
# scripts/wt-add.sh
#
# Race-safe git worktree creation for the Wave Orchestrator.
# Resolves fluxion #2489: the race between `git pull` updating the local
# `develop` ref and `git worktree add` reading that ref left new branches
# based on stale develop, forcing a manual `git reset --hard origin/develop`.
#
# This helper:
#   1. fetches the base branch from the remote (atomically updating
#      <remote>/<base>, which is what we read -- never the local ref),
#   2. creates the worktree + branch DIRECTLY from <remote>/<base>,
#   3. verifies the new branch HEAD == <remote>/<base> HEAD before success.
#
# Usage:
#   wt-add.sh <issue-num> <slug> [worktree-root]
#   wt-add.sh --check
#   wt-add.sh --dry-run <issue-num> <slug>
#   wt-add.sh -h | --help
#
# Arguments:
#   issue-num      GitHub issue number (e.g. 2489)
#   slug           branch slug, no spaces (e.g. worktree-race-safe-sync)
#   worktree-root  parent dir for the new worktree (default: ../worktrees)
#
# Environment:
#   WT_REMOTE       remote to fetch from   (default: origin)
#   WT_BASE_BRANCH  base branch on remote   (default: develop)
#
# Exit codes:
#   0  success / --check invariant holds
#   1  misuse / invariant violated
#   2  worktree or branch already exists

set -euo pipefail

REMOTE="${WT_REMOTE:-origin}"
BASE_BRANCH="${WT_BASE_BRANCH:-develop}"

usage() {
    cat >&2 <<EOF
Usage:
  wt-add.sh <issue-num> <slug> [worktree-root]
  wt-add.sh --check
  wt-add.sh --dry-run <issue-num> <slug>
  wt-add.sh -h | --help

Creates <worktree-root>/issue-<num>-<slug> on a new branch
fix/issue-<num>-<slug> based on ${REMOTE}/${BASE_BRANCH}, then verifies the
new HEAD matches ${REMOTE}/${BASE_BRANCH}. Default worktree-root is ../worktrees.

  --check    Self-test: fetch and report whether local ${BASE_BRANCH} matches
             ${REMOTE}/${BASE_BRANCH}. Creates no worktree. Exits 0 if synced.
  --dry-run  Print planned commands without creating the worktree.

Env: WT_REMOTE (default ${REMOTE}), WT_BASE_BRANCH (default ${BASE_BRANCH})
EOF
    exit "${1:-1}"
}

die() { echo "wt-add: error: $*" >&2; exit 1; }
note() { echo "wt-add: $*"; }

# Resolve the repo root via git (works from a worktree too). Exits if not a repo.
require_git_repo() {
    git rev-parse --git-dir >/dev/null 2>&1 || die "not inside a git repository"
}

# Fetch <BASE_BRANCH> from <REMOTE>. The remote-tracking ref <REMOTE>/<BASE_BRANCH>
# is updated atomically by fetch -- that is the ref we read everywhere below.
sync_remote_ref() {
    note "fetching ${REMOTE}/${BASE_BRANCH}"
    git fetch "$REMOTE" "$BASE_BRANCH" \
        || die "fetch ${REMOTE} ${BASE_BRANCH} failed"
}

# Compute the worktree path and branch name from issue number + slug.
# Echoes "<path> <branch>" as an absolute path. Validates the slug.
# Side-effect free (safe for --dry-run): creates no directories.
resolve_names() {
    local num="$1" slug="$2" root="${3:-../worktrees}"
    [[ "$num" =~ ^[0-9]+$ ]] || die "issue-num must be numeric (got '$num')"
    [[ -n "$slug" ]] || die "slug is required"
    [[ "$slug" =~ ^[A-Za-z0-9._-]+$ ]] \
        || die "slug may only contain [A-Za-z0-9._-] (got '$slug')"
    # Resolve root to an absolute path relative to CWD for stable reporting,
    # without creating it (caller decides whether to mkdir).
    case "$root" in
        /*) : ;;                                    # already absolute
        *)  root="$PWD/$root" ;;                    # make absolute
    esac
    echo "${root}/issue-${num}-${slug} fix/issue-${num}-${slug}"
}

cmd_create() {
    local num="$1" slug="$2" root="${3:-}"
    require_git_repo
    sync_remote_ref

    local remote_head
    remote_head="$(git rev-parse --verify "${REMOTE}/${BASE_BRANCH}" 2>/dev/null)" \
        || die "ref ${REMOTE}/${BASE_BRANCH} not found after fetch"
    note "target ${REMOTE}/${BASE_BRANCH} @ ${remote_head:0:12}"

    local names path branch
    names="$(resolve_names "$num" "$slug" "$root")"
    path="${names%% *}"
    branch="${names##* }"

    if git rev-parse --verify --quiet "$branch" >/dev/null; then
        echo "wt-add: branch '${branch}' already exists" >&2
        exit 2
    fi
    if [[ -e "$path" ]]; then
        echo "wt-add: path '${path}' already exists" >&2
        exit 2
    fi

    # Ensure the parent directory exists (git creates the leaf worktree).
    mkdir -p "$(dirname "$path")"

    # Bases the new branch on <REMOTE>/<BASE_BRANCH> directly. Reading the
    # remote-tracking ref (updated atomically by fetch) is what closes the race.
    note "git worktree add '$path' -b '$branch' ${REMOTE}/${BASE_BRANCH}"
    git worktree add --force "$path" -b "$branch" "${REMOTE}/${BASE_BRANCH}" \
        || die "git worktree add failed"

    # Invariant: the freshly created branch must sit exactly on <remote>/<base>.
    local worktree_head
    worktree_head="$(git -C "$path" rev-parse HEAD)"
    if [[ "$worktree_head" != "$remote_head" ]]; then
        # Roll back the partial worktree so callers can retry cleanly.
        git worktree remove --force "$path" >/dev/null 2>&1 || true
        git branch -D "$branch" >/dev/null 2>&1 || true
        die "worktree HEAD ${worktree_head:0:12} != ${REMOTE}/${BASE_BRANCH} ${remote_head:0:12}; rolled back"
    fi

    note "OK: $path on $branch @ ${worktree_head:0:12}"
}

cmd_dry_run() {
    local num="$1" slug="$2" root="${3:-}"
    require_git_repo
    local names path branch
    names="$(resolve_names "$num" "$slug" "$root")"
    path="${names%% *}"
    branch="${names##* }"
    echo "# planned (no changes will be made):"
    echo "git fetch $REMOTE $BASE_BRANCH"
    echo "git worktree add '$path' -b '$branch' $REMOTE/$BASE_BRANCH"
    echo "test \"\$(git -C '$path' rev-parse HEAD)\" = \"\$(git rev-parse $REMOTE/$BASE_BRANCH)\""
}

cmd_check() {
    require_git_repo
    sync_remote_ref
    local local_ref remote_ref
    local_ref="$(git rev-parse --verify --quiet "${BASE_BRANCH}" 2>/dev/null || true)"
    remote_ref="$(git rev-parse --verify --quiet "${REMOTE}/${BASE_BRANCH}" 2>/dev/null)" \
        || die "ref ${REMOTE}/${BASE_BRANCH} not found after fetch"
    if [[ -z "$local_ref" ]]; then
        echo "wt-add: local '${BASE_BRANCH}' does not exist; create-from-remote still safe" >&2
        echo "  git fetch $REMOTE $BASE_BRANCH:$BASE_BRANCH"
        exit 1
    fi
    if [[ "$local_ref" == "$remote_ref" ]]; then
        note "OK: ${BASE_BRANCH} synced to ${REMOTE}/${BASE_BRANCH} @ ${local_ref:0:12}"
        exit 0
    fi
    echo "wt-add: FAIL: ${BASE_BRANCH} (${local_ref:0:12}) != ${REMOTE}/${BASE_BRANCH} (${remote_ref:0:12})" >&2
    echo "        create-from-remote is still safe; to sync the local ref run:" >&2
    echo "        git fetch $REMOTE $BASE_BRANCH:$BASE_BRANCH   (fast-forward only)" >&2
    exit 1
}

main() {
    case "${1:-}" in
        -h|--help) usage 0 ;;
        --check) shift; [[ $# -eq 0 ]] || usage 1; cmd_check ;;
        --dry-run)
            shift
            [[ $# -ge 2 && $# -le 3 ]] || usage 1
            cmd_dry_run "$1" "$2" "${3:-}"
            ;;
        "")
            usage 1
            ;;
        *)
            [[ $# -ge 2 && $# -le 3 ]] || usage 1
            cmd_create "$1" "$2" "${3:-}"
            ;;
    esac
}

main "$@"
