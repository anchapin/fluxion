# Branch Protection Configuration

> **Status:** Reference document. This file describes the GitHub branch
> protection configuration that anchors the `main` branch's "Required
> status checks" list. The configuration is enforced by GitHub and can
> only be modified by a repository administrator in the GitHub web UI
> (Settings → Branches → `main` → Require status checks to pass before
> merging). This file does **not** enforce anything by itself — it is
> the canonical spec that admins and the verification process check
> against.

This document was introduced by **issue #1351** to close the
acceptance gap from **#1297** (cross-platform FP determinism CI gate).
The previous gate had a working `Determinism Check` job inside
`ashrae_validation.yml` but it was an OS-matrix job and was not
listed in the branch-protection "Required status checks" — so a
determinism failure on one OS could merge if the others passed (or
the gate could pass if the in-workflow `compare-hashes` step did not
catch a regression that the upstream `Cross-Platform Determinism CI`
workflow did).

The fix is two-layered:

1. **In-workflow wiring (this repo, in this PR)** — add an explicit,
   non-matrix `Fluxion Determinism Gate` listener job that observes
   the upstream `Cross-Platform Determinism CI` workflow_run and
   fails the ASHRAE 140 CI Gate workflow if the upstream concluded
   `failure` / `cancelled` / `timed_out`.
2. **Branch protection (manual admin step)** — add the listener job
   name to the "Required status checks" list on `main` so the PR
   cannot be merged without that check reporting `success`.

---

## Required status checks on `main`

The `main` branch **must** require all of the following status checks
to report `success` before any PR can be merged:

| # | Status check name (exact)                            | Workflow / job                                              | Source issue |
|---|------------------------------------------------------|-------------------------------------------------------------|--------------|
| 1 | `ASHRAE 140 Strict Energy Gate (Issue #1333)`        | `.github/workflows/ashrae_140_strict_energy_gate.yml`       | #1333        |
| 2 | `Fluxion Determinism Gate (Issue #1351)`             | `.github/workflows/ashrae_validation.yml` (`fluxion-determinism-gate` listener job) | #1351 / #1297 |

The list is mirrored in `release_gates.yaml` under `ci.required_checks`
and `ci.workflow_index`. The two sources **must** stay in sync — any
change to one is a change to the other.

### Why the determinism gate is a *non-matrix* job name

The upstream `Determinism Check` job in `determinism_check.yml` runs
on an OS matrix (`ubuntu-latest`, `windows-latest`, `macos-latest`),
which GitHub exposes as three separate required-check rows. Branch
protection can only reference one canonical name, so the gate wiring
in `ashrae_validation.yml` adds a separate listener job
(`fluxion-determinism-gate`, surfaced as
`Fluxion Determinism Gate (Issue #1351)`) that fires **after** the
upstream `Cross-Platform Determinism CI` workflow concludes on the
same SHA. The listener is the single source of truth that branch
protection references.

### What happens if the listener fails

The listener:

1. Inspects `github.event.workflow_run.conclusion` for the upstream
   `Cross-Platform Determinism CI` run on the same SHA.
2. On `failure` / `cancelled` / `timed_out`, exits non-zero (so the
   listener reports `failure` in the PR's checks list) **and** posts
   a PR comment with the upstream run URL and a triage checklist.
3. On `success` / `neutral` / `skipped`, exits zero.

---

## Manual admin step (one-time)

The exact branch-protection list cannot be edited from a PR (it lives
in GitHub repo settings). A repository administrator must:

1. Open the GitHub repository → **Settings** → **Branches**.
2. Click **Add rule** (or edit the existing `main` rule).
3. Set **Branch name pattern** = `main`.
4. Enable **Require status checks to pass before merging**.
5. Click **Add checks** and search for / type the exact check names
   listed in the table above.
6. (Recommended) Enable **Require branches to be up to date before
   merging** so the listener fires on the merge commit, not the
   branch tip.
7. Save the rule.

The `main` rule's current state is also exposed via the GitHub API
(`GET /repos/{owner}/{repo}/branches/main/protection`); a small
verification script that compares the API's `required_status_checks`
list to the table above lives at
`scripts/check_branch_protection.py` (added in a follow-up to
#1351).

---

## Verification path

The acceptance criterion for #1351 requires an empirical test that
a determinism failure actually blocks the PR. To verify after the
manual admin step is applied:

1. Create a temporary branch off `main` (e.g.
   `chore/1351-verify-determinism-gate`).
2. Modify `tests/case_900_determinism.rs` to compare one of the
   determinism constants to a deliberately wrong value (e.g.
   `assert_eq!(annual_heating_mwh, 0.0, ...)`).
3. Open a PR. The PR's checks list must show
   `Fluxion Determinism Gate (Issue #1351)` (and the in-workflow
   `Compare Hashes` and the upstream `Cross-Platform Determinism CI`)
   all reporting `failure`.
4. **Without** merging, revert the assertion change and either
   re-push (the listener re-fires on the new SHA) or close the PR.
5. The merge button must be disabled while the listener is failing
   — that is the gate blocking the merge.

If the merge button is **not** disabled while the listener is
failing, the manual admin step has not been applied correctly.
Re-check the branch-protection rule's "Required status checks" list
and ensure `Fluxion Determinism Gate (Issue #1351)` is present and
exact (no extra whitespace, correct case).

---

## Out of scope (not changed by this document)

- The independent `Cross-Platform Determinism CI` workflow
  (`.github/workflows/determinism_check.yml`) is kept as the primary
  entry point. Do not remove it.
- The in-workflow `determinism-check` matrix + `compare-hashes` job
  inside `ashrae_validation.yml` are kept as a fast first line of
  defence. They mirror the upstream workflow but the listener is
  the durable single-check anchor for branch protection.
- Physics changes, determinism-failure root-cause fixes, and
  expansion of the determinism test cases are all separate issues
  (F# and B# verticals in the issue batch, see #1297 and #1351).

---

## Linked issues

- **#1297** — original "Establish cross-platform FP determinism CI
  gate" issue. Closed when the upstream `determinism_check.yml`
  workflow landed. The acceptance gap ("the determinism gate must be
  a *required* PR gate, not just an info check") is closed by this
  document + the listener job wiring.
- **#1351** — this issue. Wires the listener job as a required
  status check, documents the manual admin step, and adds the
  canonical `Fluxion Determinism Gate (Issue #1351)` check name.
- **#1357** — Wave 1 follow-up that unblocked the determinism gate
  by fixing the `ort` feature-gate import (PR #1357,
  `fix(ci): gate ort imports + add drift-check permissions + fix
  MultiNodeSolver false-positive`).
- **#1063** — referenced by #1351's "Linked existing issues"
  section. Predecessor determinism discussion.

---

_Last updated: 2026-06-27. Reviewed by: TBD. Maintained by: fluxion
release-gate working group._
