# Branch Protection Strict Mode — Issues #3142, #3809 (ADR-0016 lane reconciliation)

**Issue:** #3142 (path-filter rationale), #3809 (ADR-0016 lane reconciliation), ADR-0016 (three-lane model)
**Date:** 2026-08-29 (initial), 2026-09-16 (ADR-0016 rewrite)
**Status:** Implemented

## Problem

When merging a PR that touches only `scripts/`, `.github/workflows/`, or `docs/`, GitHub displays:

```
GraphQL: At least 1 approving review is required by reviewers with write access.
14 of 23 required status checks are expected. (mergePullRequest)
```

The "14 of 23" is misleading. The 9 checks that "didn't run" are **path-filtered** — their workflows have `paths:` filters that exclude `scripts/`, `.github/workflows/`, and `docs/` changes. GitHub counts them as "expected" because they are listed in the branch-protection required checks, but their workflows never triggered for this PR.

**All 14 checks that ran actually passed.** The 9 omitted checks cannot run on workflow-only PRs by design.

## Root Cause

GitHub branch protection required checks are a flat list. When a required check's workflow has a `paths:` filter that excludes the PR's changed files:

1. The workflow does not run for this PR
2. GitHub still shows the check as "expected" in the PR status
3. The PR shows "X of Y required status checks expected" where Y includes checks that structurally cannot run

This is working as designed — GitHub cannot know that a check is "path-filtered" and therefore should not count against the PR. The confusion arises from the mismatch between "required checks in branch protection" and "checks that can actually run for this PR."

## ADR-0016 Lane Model (Issue #3809)

`release_gates.yaml::ci` classifies every check into one of three lanes per [`docs/adr/0016-fast-lane-gates-nightly-authority.md`](../adr/0016-fast-lane-gates-nightly-authority.md):

| Lane | Authoritative enforcer | Required per-PR? | Path class |
|---|---|---|---|
| **Lane 1 — fast lane** | GitHub branch protection (`ci.required_checks`) | yes (every PR) | always |
| **Lane 2 — path-filtered physics gauntlet** | GitHub branch protection (`ci.required_checks`) | yes when the physics path class changes (deny-list semantics) | `src/**`, `fluxion-core/**`, `fluxion-fluid/**`, `tests/**`, `models/**`, `weather/**` |
| **Lane 3 — nightly authority** | wave-orchestrator merge freeze (`ci.nightly_authority`) | NO — non-blocking per-PR | nightly on `develop` |

Lane 1 (~20–27 min CI pole = `rust-tests.yml`) is required on every PR regardless of changed files. Lane 2 is required only when the PR touches the physics/binding path class; deny-list path filters (`paths-ignore` for docs/deps-only change sets) mean an unexpected file type triggers the gauntlet rather than skipping it — fail-safe by construction. Lane 3 (determinism, perf, coverage, CUDA) is enforced by the wave orchestrator, NOT branch protection — GitHub cannot express "last nightly green" as a required check, so a red nightly freezes merges + new waves (blind window ≤ ~18 h, bounded by one nightly cycle). The local-validation ladder in [`docs/agents/pre-push-checklist.md`](../agents/pre-push-checklist.md) is calibrated to this lane prediction.

## Path-Filtered Lane-2 Checks

The following required checks have workflows with `paths:` filters that exclude workflow-only changes:

| Check | Workflow | Path Filter |
|-------|----------|-------------|
| Docs Hygiene Gate (Issue #2466) | `docs-hygiene.yml` | `docs/**`, `**/*.md`, scripts/**, AGENTS.md, etc. |
| Architecture Drift Detection | `architecture_drift.yml` | `src/**/*.rs`, `ARCHITECTURE.md`, scripts/** |
| Module Size (Issue #2878) | `architecture_drift.yml` | `src/sim/thermal_model_data.rs`, `src/sim/thermal_model_data/**`, `scripts/check_module_size.py`, `tests/reference_data/module_size/**` |
| Crate Size Gate (Issue #2930) | `crate-size.yml` | `Cargo.toml`, `.cargoignore` |
| MSRV Check (Issue #2934) | `msrv.yml` | `**/Cargo.toml`, `**/Cargo.lock` |

For a PR touching only `scripts/`, `.github/workflows/`, or `docs/`:
- `docs-hygiene.yml` **does** run (its path filter includes these paths)
- `architecture_drift.yml`, `crate-size.yml`, `msrv.yml` **do not** run

`Module Size (Issue #2878)` is not a separate workflow — it is a step inside `architecture_drift.yml`'s `check-drift` job (Issue #2878, wired via #3394). The job name remains `Architecture Drift Detection`, but the step emits its own check name and inherits the parent workflow's `paths:` filter, so it is path-filtered for the same reason as the row above.

The Issue #3810 GH-listener pattern mitigates this for the Lane-1 fast lane by adding unconditional listener jobs (`if: always()`) inside path-filtered workflows that emit the required-check name even when the upstream was `skipped` due to paths. This is the Lane-1 floor for `required_checks_workflow_only` (14 checks); branch protection still requires the listener variant name (e.g. `Workspace Check (GH)`, `Energy Conservation (GH)`). Lane-2 physics validation cannot be synthesized this way — a listener cannot certify "ASHRAE validation passed" without the upstream physics work, so the 5 path-filtered Lane-2 checks remain excluded from `required_checks_workflow_only`. Issue #3898 additionally removed the five `paths-ignore` Lane-2 gates (`ASHRAE 140 Strict Energy Gate (Issue #1333)`, `Surrogate ASHRAE 140 MAE Gate (Issue #2924)`, `h_tr_em Regression Gate (LIMIT-13)`, `Tracked-vs-Ignored Gate (Issue #3356)`, `Fast-Math vs IEEE-754 ASHRAE 600/900 Regression Gate (GH)`) from BOTH required lists — their workflows skip docs-only diffs entirely, so a required name could never be satisfied on a docs-only PR (see the fast-math section below).

## Fast-Math Gate: #3358 Promotion, #3898 Demotion

The `Fast-Math vs IEEE-754 ASHRAE 600/900 Regression Gate (GH)` check (#3358, #3326, #3322) was promoted from advisory to required in 2026-09 via the `(GH)` listener pattern. Issue #3898 then REMOVED it from both `required_checks` and `required_checks_workflow_only`: the workflow's `pull_request` trigger carries `paths-ignore` for docs/markdown (deny-list Lane-2 semantics — the workflow is skipped only when every changed file is irrelevant to the physics path class, fail-safe by design), so on a docs-only PR no run — the `fast-math-gh` listener included — is ever created and the required name could never report, making every docs-only PR permanently unmergeable under `enforce_admins: true`.

The listener job in `.github/workflows/fast_math_check.yml` is named `fast-math-gh` and emits the exact check name `"Fast-Math vs IEEE-754 ASHRAE 600/900 Regression Gate (GH)"`; the actual physics assertions (`±0.05%` load agreement + `≤ 1e-5 W` residual ceiling, Issue #3326) live in the upstream `compare` job. The gate itself is UNCHANGED by #3898 — it keeps running as a non-required path-filtered check on every PR class that touches its filter, plus nightly at 04:00 UTC, and remains in `ci.workflow_index`. The wave-orchestrator all-checks-green merge criterion and the ADR-0016 Lane-3 nightly authority re-cover the invariant. Operator procedures (activation / rollback of the live branch-protection context) live in [`docs/ci/fast-math-stability-window.md`](fast-math-stability-window.md).

## Solution

`release_gates.yaml::ci` partitions every required check into the three ADR-0016 lanes above. Branch protection lists the Lane 1 + Lane 2 union as required checks (19 checks). Lane 3 lives in `ci.nightly_authority` and is enforced by the wave-orchestrator merge freeze, not branch protection.

1. **`required_checks`** — Lane 1 + Lane 2 union (19 checks; the 8 ADR-0016 Lane-3 gates were moved out into `ci.nightly_authority`, and Issue #3898 removed the five `paths-ignore` gates that can never report on a docs-only PR). Use this for branch protection configuration on `main`, and for `develop` PRs that touch the physics path class.

2. **`required_checks_workflow_only`** — Lane-1 + Lane-2 items that run on every PR regardless of changed files (14 checks; ADR-0016 moved 8 nightly-authority gates out, and Issue #3898 removed the 5 `paths-ignore` gates — `ASHRAE 140 Strict Energy Gate (Issue #1333)`, `Surrogate ASHRAE 140 MAE Gate (Issue #2924)`, `h_tr_em Regression Gate (LIMIT-13)`, `Tracked-vs-Ignored Gate (Issue #3356)`, `Fast-Math vs IEEE-754 ASHRAE 600/900 Regression Gate (GH)` — from BOTH required lists, because their `paths-ignore` filters mean no workflow run is ever created for a docs-only diff, so a required name can never be satisfied there). This excludes the 5 path-filtered Lane-2 items listed above.

3. **`nightly_authority`** (Lane 3) — `ci.nightly_authority` lists the 8 nightly-only gates (#1351, #1618, #2693, #2772, #2919, #2922, #1932, #1603) that were removed from `required_checks_workflow_only` per ADR-0016. They run nightly on `develop`; their authority is the wave-orchestrator merge freeze, not branch protection.

### Branch Protection Configuration

**For `main` branch:** Use `required_checks` (19 checks; the full Lane 1 + Lane 2 union). Code-changing PRs that touch the physics path class must pass all Lane-1 + Lane-2 gates. Lane 3 is not required per-PR on `main` either — the wave orchestrator's merge freeze covers both branches.

**For `develop` branch:** Use `required_checks_workflow_only` (14 checks; Lane 1 + non-path-filtered Lane 2 items). Workflow-only PRs (docs, CI, scripts) can merge without triggering the 5 path-filtered Lane-2 items that structurally cannot run for them, and without waiting on the five Issue #3898 `paths-ignore` gates that skip docs-only diffs by design. The Lane-3 merge freeze is the develop-side authority for the 8 nightly-authority gates.

### Reviews-Advisory Policy (Issue #3810, PR #3805)

Per Issue #3810 and the solo-PR pattern observed in PR #3805, both `develop` and `main` carry a **reviews-advisory** policy:

- `required_approving_review_count = 0` — operator-author-and-merger workflow; the author can self-merge without an approving review (the GitHub GraphQL error `At least 1 approving review is required by reviewers with write access` is the canonical symptom).
- `enforce_admins = true` — the protection rules above (Lane-1 / Lane-2 gates, Lane-3 merge freeze) still apply to admins; `enforce_admins.enabled=true` is what makes the gate authoritative rather than advisory for admins.

`scripts/apply_branch_protection.py` defaults to `required_approving_review_count=1` (production hardening) and accepts `--required-approving-review-count=0` for the solo-operator workflow. Issue #3807 documented the original hardcoded `1`; the CLI flag in #3829 is the documented escape hatch.

### Alternative: Single Rule with Documentation

If GitHub branch protection only supports one required-checks list, use `required_checks_workflow_only` and document the behavior:

> **Note:** Some required checks (`Docs Hygiene Gate`, `Architecture Drift Detection`, `Module Size`, `Crate Size Gate`, `MSRV Check`) have workflows that only run when specific file patterns are changed. For PRs touching only `scripts/`, `.github/workflows/`, or `docs/`, these checks will not run and are excluded from the required list. The five Issue #3898 `paths-ignore` gates (`ASHRAE 140 Strict Energy Gate`, `Surrogate ASHRAE 140 MAE Gate (Issue #2924)`, `h_tr_em Regression Gate`, `Tracked-vs-Ignored Gate`, `Fast-Math ... (GH)`) are likewise excluded from BOTH required lists — they skip docs-only diffs entirely, so requiring their names made docs-only PRs permanently unmergeable; they keep running as non-required path-filtered checks. The 14 always-run checks provide adequate regression protection for workflow-only changes; the 8 nightly-authority gates (Lane 3) are enforced by the wave-orchestrator merge freeze.

## Implementation Notes

- The path-filtered Lane-2 checks **still run** on PRs that touch the relevant files (e.g., `Cargo.toml` changes trigger `MSRV Check`).
- The path-filtered Lane-2 checks **still block** code-changing PRs that would affect them.
- The `required_checks_workflow_only` list is a subset of `required_checks` — it removes only the 5 path-filtered checks above.
- The five Issue #3898 `paths-ignore` gates are absent from BOTH lists but **still run** as non-required checks on every PR class that touches their filters (any non-docs diff); their invariants are re-covered by the wave-orchestrator all-checks-green merge criterion and the Lane-3 nightly authority.
- Lane 3 nightly authority is enforced by the wave orchestrator (Issue #3286 / `#3286 β-soak` convention in CI comment threads); the merge freeze is the binding signal, not branch protection.
- The local-validation ladder per [`docs/agents/pre-push-checklist.md`](../agents/pre-push-checklist.md) is calibrated to the lane prediction: a docs-only PR triggers Lane 1 (~19–26 min CI); anything touching the physics path class triggers the full Lane-2 gauntlet (~35–40 min).

## See Also

- [`docs/adr/0016-fast-lane-gates-nightly-authority.md`](../adr/0016-fast-lane-gates-nightly-authority.md) — three-lane model, deny-list semantics, nightly authority + merge freeze
- [`docs/agents/pre-push-checklist.md`](../agents/pre-push-checklist.md) — per-lane local validation ladder, fix-push etiquette
- `release_gates.yaml::ci.required_checks`
- `release_gates.yaml::ci.required_checks_workflow_only`
- `release_gates.yaml::ci.nightly_authority` (Lane 3)
- `scripts/check_required_checks_sync.py` (Issue #2866 / #3441)
- `scripts/apply_branch_protection.py` (Issue #3386 / #3829 CLI flag)
- `scripts/check_branch_protection_diff.py` (Issue #3383 diagnostic)
- `.github/workflows/docs-hygiene.yml`
- `.github/workflows/architecture_drift.yml`
- `.github/workflows/crate-size.yml`
- `.github/workflows/msrv.yml`
