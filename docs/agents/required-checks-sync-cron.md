# Required Checks Sync (Live Mode) — Issue #3123
> **Summary 1/7:** `scripts/check_required_checks_sync.py` ships with a static-only mode (PR-blocking CI) and a live-mode hook (`FLUXION_CHECK_LIVE_PROTECTION=1`) added in #3116 that `gh api`-queries the live `develop` branch protection.
> **Summary 2/7:** #3123 wires that hook into a scheduled workflow (`.github/workflows/required-checks-sync-cron.yml`) so a "YAML says 13 required checks, develop actually has 0" gap cannot recur silently between PRs.
> **Summary 3/7:** The workflow runs daily at 06:00 UTC plus `workflow_dispatch`; on PR triggers the static check is already covered by `.github/workflows/scripts-tests.yml` so the cron stays schedule-only.
> **Summary 4/7:** Auth uses an org-level PAT (`FLUXION_BRANCH_PROTECTION_PAT`, `repo` scope) because the default `GITHUB_TOKEN` in scheduled workflows is read-only on most scopes; passed via `GH_TOKEN` to the script's `gh api` call.
> **Summary 5/7:** Secret rotation policy: the PAT is created by an org owner, scoped to `anchapin/fluxion` only with `repo` access, and rotated **quarterly** (next rotation due at the end of each calendar quarter — tracked in the org secrets dashboard).
> **Summary 6/7:** Failure alert path: the script exits non-zero → the scheduled workflow fails → GitHub sends workflow-failure notifications to subscribers; the failure step also emits a `::error::` annotation with remediation steps.
> **Summary 7/7:** Job name is intentionally NOT in `release_gates.yaml::ci.required_checks` because scheduled-only workflows cannot produce a PR-blockable check run (the script's invariant #3 enforces this on the static side).

- **Owner:** Issue #3123 — cron-mode wiring for the live branch-protection check.
- **Related:** #3116 (added the live-mode hook in the script), #2866 (created the static sync script), #2526 (least-privilege `GITHUB_TOKEN` defaults).

## Trigger

The workflow is configured in `.github/workflows/required-checks-sync-cron.yml`:

| Trigger | Schedule | Purpose |
|---|---|---|
| `schedule` | `0 6 * * *` (daily 06:00 UTC) | Catch drift between YAML and develop's live branch protection before the next PR lands. |
| `workflow_dispatch` | Manual | Operator-run after a `gh api PUT` against branch protection, or to verify a fix landed. |

There is **no `pull_request` trigger** on this workflow. The PR-blocking static-mode invocation lives in `.github/workflows/scripts-tests.yml` (already wired in #2866); adding a PR trigger here would duplicate effort and run live `gh api` on every PR (read-heavy against GitHub's branch-protection rate limits).

## What the script checks

`scripts/check_required_checks_sync.py` enforces five invariants. The first four are pure YAML / regex parsing — no network access. The fifth (only enabled when `FLUXION_CHECK_LIVE_PROTECTION=1`) calls `gh api /repos/anchapin/fluxion/branches/develop/protection` and verifies:

1. `required_status_checks.contexts` matches `release_gates.yaml::ci.required_checks` by symmetric set equality.
2. `required_status_checks.strict` is `true`.
3. `required_pull_request_reviews.required_approving_review_count` ≥ 1.
4. `enforce_admins.enabled` is `true`.

When any check fails the script exits 1 and the cron run fails. When all pass, exit 0 and the cron succeeds (silent green).

## Secret rotation policy

**Name:** `FLUXION_BRANCH_PROTECTION_PAT` (org-level secret, available to all repos in the `anchapin` org).

**Scope:** `repo` only — the minimum needed for the `GET /repos/{owner}/{repo}/branches/{branch}/protection` endpoint. **No `admin:org`, no `workflow`, no other scopes** — the PAT is used only for read-only branch-protection reads; the script never writes branch protection.

**Rotation cadence:** **quarterly**, at the end of each calendar quarter (Mar 31, Jun 30, Sep 30, Dec 31). The rotation owner is the current Fluxion release lead; the rotation procedure is:

1. Create a new fine-grained PAT (or a new classic PAT with `repo` scope) in the `anchapin` GitHub org.
2. Overwrite the org secret `FLUXION_BRANCH_PROTECTION_PAT` with the new token value via the org Settings → Secrets page.
3. Trigger `workflow_dispatch` on `.github/workflows/required-checks-sync-cron.yml` to confirm the new token works (the next scheduled run will also exercise it).
4. Revoke the previous PAT from the user's developer settings page.
5. Record the rotation date in the org secrets dashboard and in the next release-notes update.

**Emergency rotation:** If the PAT leaks or is suspected compromised, rotate immediately (out-of-band, do not wait for the next quarterly window) and add an entry to the release notes.

**Audit trail:** Every cron run that touches the `gh api` branch-protection endpoint is logged in GitHub Actions with the workflow run ID and the operator who triggered it (for `workflow_dispatch`) or the scheduled run timestamp (for `schedule`).

## Failure alert path

The script's exit code drives the alert. The cron is designed so a non-zero exit code is the *only* failure signal — no silent "drift found, but I'll let it through" path.

1. The scheduled workflow runs at 06:00 UTC.
2. The script exits 1 when either static YAML drift or live branch-protection drift is detected (see the script's docstring for the four failure categories).
3. The `Annotate failure remediation` step emits a `::error::` annotation with the remediation steps (read drift messages, update YAML or apply via `gh api PUT`, re-run via `workflow_dispatch`).
4. GitHub Actions marks the run as failed.
5. Subscribers to workflow-failure notifications (the Fluxion maintainer on-call rotation) receive the alert via email / Slack / the GitHub mobile app, depending on their subscription preferences.

**No issue is auto-created** on failure. The cron-comments pattern from `.github/workflows/known-issues-stale.yml` is intentionally not used here because drift here is an acute signal — "develop's branch protection lost required checks" — that should be triaged immediately, not tracked in a long-lived tracking issue. The `::error::` annotation + failed run is the entire signal.

## Manual verification

To exercise the cron manually outside of the schedule:

1. Trigger via the Actions tab → "Required Checks Sync (Live Mode)" → "Run workflow".
2. Inspect the run logs:
   - Success: ends with `Live develop branch protection matches release_gates.yaml (contexts, strict, reviews, enforce_admins).`
   - Failure: ends with `DRIFT DETECTED` (static) or `LIVE DRIFT DETECTED` (live) plus a remediation hint.
3. To reproduce locally without `gh auth`, run the script with the env unset:

   ```bash
   python3 scripts/check_required_checks_sync.py
   # exits 0 in static-only mode; never touches the network.
   ```

## References

- **Issue #3123** — this issue (cron-mode wiring + PAT rotation policy).
- **Issue #3116** — added the `FLUXION_CHECK_LIVE_PROTECTION=1` live-mode hook in `scripts/check_required_checks_sync.py` and the canonical-vs-suffix tolerance fix.
- **Issue #2866** — created the original `scripts/check_required_checks_sync.py` static-mode gate.
- **Issue #2526** — least-privilege `GITHUB_TOKEN` defaults across all workflows.
- **`.github/workflows/required-checks-sync-cron.yml`** — the workflow itself.
- **`scripts/check_required_checks_sync.py`** — the verification script (static + live modes).
- **`.github/workflows/scripts-tests.yml`** — PR-triggered static-only invocation (no network access).
