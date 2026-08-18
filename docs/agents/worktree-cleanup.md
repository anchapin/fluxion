# Stale worktree & branch cleanup (issue #3069)

Operator guide for `scripts/cleanup_stale_worktrees.sh` — the **dry-run
by default** cleanup tool that identifies stale `fix/issue-*` branches
and orchestration worktrees accumulated by the wave-orchestrator.
The script is fail-closed: it prints a plan, and only mutates the
repo when invoked with `--apply`. Idempotent and safe to re-run.

## What the script does

For every worktree and every `fix/issue-*` branch in the current repo,
the script classifies the target into one of two actions — **delete**
or **skip** — and prints a deterministic plan. The same classification
runs identically under dry-run and `--apply`; the only difference is
that `--apply` actually executes the deletions. The script never
deletes `main` or `develop`, never removes the current worktree, and
never deletes a branch that has unpushed commits or no remote
tracking.

## Why 470 stale branches accumulated

The wave-orchestrator (rationalised by `github-wave-orchestrator` and
`parallel-issue-workflow`) spawns one worktree per `fix/issue-*` branch
and only cleans each one up if the wave explicitly finishes. Aborted
or crashed waves leave behind a worktree + branch pair every time,
and the 2026-08-16 wave surfaced 470 stale `fix/issue-*` branches and
23 stale worktrees from prior sessions. At ~500 MB per worktree, this
is ~10 GB of recoverable disk pressure. The volume also clips
`git branch --list` and `git worktree list` output, making future
agent sessions harder to audit (an agent picking up an abandoned
worktree has no way to know it's stale other than by looking at
upstream + merge-base).

## How to run

```bash
# 1. Dry-run (default): print the plan, exit 0 if all targets safe.
./scripts/cleanup_stale_worktrees.sh

# 2. JSON report for CI / audit:
./scripts/cleanup_stale_worktrees.sh --json --output target/cleanup_report.json

# 3. Actually delete: re-run with --apply.
./scripts/cleanup_stale_worktrees.sh --apply

# 4. Apply + skip empty-commit branches (e.g. for the #3069 e265c62 case,
#    if you want to manually triage them one at a time):
./scripts/cleanup_stale_worktrees.sh --apply --keep-empty-commits
```

`--apply` is the only flag that mutates state. All other flags (`--json`,
`--output`, `--keep-empty-commits`, `--keep-unmerged`) only change the
output or skip-rule selection.

## When to run it

- **Before major orchestrations** (e.g. before kicking off a new
  wave-orchestrator that its own state file expects to start clean).
- **Quarterly as a hygiene check**, or whenever `git worktree list`
  shows more than ~5 entries.
- **Ad-hoc after a wave-orchestrator crash** that left orphaned
  worktrees behind.
- **NOT during an active wave** — the wave-orchestrator's own worktree
  bookkeeping (`BRANCH_DIR` / `WORKTREE_DIR`) may still reference the
  branches you would delete. The plan still runs as a dry-run during
  an active wave, but `--apply` should wait for the wave to finish.

## What the script does NOT do

- **Does NOT delete branches with unmerged commits** — the only
  exception is branches whose only divergence from `develop` is an
  empty commit (matches the #3069 `e265c62` pattern). Pass
  `--keep-empty-commits` to also skip those.
- **Does NOT delete main or develop** — these are protected by name.
- **Does NOT remove the current worktree** — the script refuses to
  call `git worktree remove` on the path returned by `git rev-parse
  --show-toplevel`.
- **Does NOT delete branches with unpushed commits** — if `origin/<branch>`
  exists and is behind the local branch, the branch is skipped. If
  `origin/<branch>` does not exist at all, the branch is treated as
  having unpushed commits and is skipped (the conservative default).
- **Does NOT touch worktrees under `~/.superset/` or `~/.planning/worktrees/`**
  — these are external tool / agent runtime worktrees (see AGENTS.md
  §Repository Hygiene) and are always preserved.
- **Does NOT silently succeed** — if the script cannot classify a
  target (e.g. git failure, no develop branch), it exits 1 with a
  clear error message pointing at the failure.

## Exit codes

| Code | Meaning |
|------|---------|
| 0 | Dry-run: every examined target is safe to delete (no blocking skips). |
| 1 | Git failure, not a git repo, no develop/main branch, etc. |
| 2 | Some targets would be skipped (unmerged, unpushed, etc.). |

Exit 2 is a "dry-run completed successfully but found blocking
items" signal — it is the expected exit code when there are
unmerged/unpushed branches in the repo. CI pipelines that consume
the JSON report should not treat exit 2 as a failure.

## Safety model

The script is fail-closed by design. Every action is printed before
it is taken, and the default mode is dry-run. Targets are deleted
only when ALL of the following hold:

1. The branch is fully merged into `develop` (or `origin/develop`,
   falling back to `main` / `origin/main`) OR the branch's only
   divergence from develop is an empty commit (matches #3069).
2. The branch has no unpushed commits (`git log origin/<branch>..<branch>`
   is empty) AND `origin/<branch>` exists OR the branch has no remote
   tracking (in which case it's skipped).
3. The branch is not `main`, `develop`, or the current branch.
4. The worktree (if any) is not the path returned by `git rev-parse
   --show-toplevel`.

When `--apply` is used, the deletion sequence is:

```
git worktree remove --force <path>     # for worktrees
git branch -D <name>                   # for both branches and worktrees
git push origin --delete <name>        # for remote branch cleanup
```

Each step is `|| true`-guarded so that a failure on one step (e.g. the
remote branch was already deleted) does not abort the rest.

## Cleanup history

Audit log of real `cleanup_stale_worktrees.sh --apply` runs against the
`anchapin/fluxion` repo. Each entry is dated and records the pre/post
branch count, the scope of the deletion wave, and any operational
observations that future runs should know about.

### 2026-08-18 — wave-orchestration hygiene cleanup (issue #3106)

Triggered by issue #3106 (chore: execute
`scripts/cleanup_stale_worktrees.sh --apply`). The #3069 PR had
delivered the script but explicitly skipped the actual cleanup step,
leaving 468+ stale `fix/issue-*` branches accumulated across
wave-orchestration sessions 2026-08-16 / 17.

| Metric | Value |
|---|---|
| Issue | [#3106](https://github.com/anchapin/fluxion/issues/3106) |
| Operator | chore-3106 agent (orchestration-driven) |
| Base branch | `develop` (`a3e21c4`) |
| Pre-cleanup `fix/issue-*` count | **471** |
| Dry-run prediction | 64 to delete · 411 to skip · 408 blocking skips |
| Post-cleanup `fix/issue-*` count | **407** |
| Branches deleted (script) | 63 |
| Branches deleted (manual, post-script) | 1 (`fix/issue-672-epic-v13-coord` — see note below) |
| Total stale branches removed | **64** (matches dry-run prediction) |
| Worktrees before / after | 5 / 5 (no worktree deletions; the 5 entries were the main `~/Projects/fluxion`, the `~/.superset/` external worktree, and three wave-1 worktrees for issues #3104 / #3105 / #3106 — all preserved by the script's safety guards) |

**Operational notes (worth recording for future runs):**

1. **First `--apply` run timed out at the shell 120 s default.** The
   script invokes `git push origin --delete <branch>` once per delete,
   which is the slow step on this repo (HTTPS + GitHub auth + ~64
   sequential round trips). Workaround: launch the script in the
   background and wait for completion, or pipe through `tee` to a log
   file. The script's `|| true` guards make it safe to interrupt and
   re-run — it is idempotent and will pick up where it left off.

2. **One dry-run "DEL" target was reclassified as "SKIP" between
   the dry-run and `--apply`.** The branch `fix/issue-672-epic-v13-coord`
   was listed as DEL in the dry-run but came back as
   `merged into develop but has unpushed commits` in the second
   `--apply` run. Root cause: the script's `has_unpushed_commits`
   helper treats "no `origin/<branch>` tracking ref" as a conservative
   *skip*, and the remote tracking for that branch had been removed
   out-of-band between the dry-run and the apply. Confirmed
   `git merge-base --is-ancestor fix/issue-672-epic-v13-coord origin/develop`
   returned 0 (true), so the branch was genuinely merged — deleted it
   manually with `git branch -D`. Future operators: if a DEL target
   gets reclassified as SKIP, check `origin/<branch>` first; if the
   branch is still merged into `develop` (use `git merge-base
   --is-ancestor`), it is safe to delete manually.

3. **Wave-1 worktrees were correctly preserved.** Branches
   `fix/issue-3104-*`, `fix/issue-3105-*`, and `fix/issue-3106-*`
   (the current wave's own work) had no `origin/<branch>` tracking
   yet (not yet pushed), so the script conservatively skipped all of
   them as "unpushed commits". This is exactly the intended
   behaviour — never delete current work. Once those PRs land and
   the wave-1 branches are merged into `develop`, the next cleanup
   pass will reclaim them.

4. **Idempotency confirmed.** Re-running the script after the
   apply-during-the-apply produced the expected post-cleanup state:
   `would delete: 0`, `would skip: 412` (all blocked on unmerged or
   unpushed). No false re-deletions.

## Related files

- `scripts/cleanup_stale_worktrees.sh` — the script itself.
- `scripts/ci/test_cleanup_stale_worktrees.py` — pytest harness
  covering all five acceptance criteria + idempotency + safety
  guards + `--keep-empty-commits` opt-out.
- `scripts/disk-space-check.sh` — sibling pre-flight check that
  estimates recovered disk space after cleanup.
- `AGENTS.md` §Repository Hygiene — the canonical list of
  always-allowed external worktree prefixes.
- Issue #3069 — original wave-orchestration-end observation.
