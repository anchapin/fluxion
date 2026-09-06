# Branch Protection Strict Mode — Issue #3142

**Issue:** #3142  
**Date:** 2026-08-29  
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

## Path-Filtered Checks

The following required checks have workflows with `paths:` filters that exclude workflow-only changes:

| Check | Workflow | Path Filter |
|-------|----------|-------------|
| Docs Hygiene Gate (Issue #2466) | `docs-hygiene.yml` | `docs/**`, `**/*.md`, scripts/**, AGENTS.md, etc. |
| Architecture Drift Detection | `architecture_drift.yml` | `src/**/*.rs`, `ARCHITECTURE.md`, scripts/** |
| Crate Size Gate (Issue #2930) | `crate-size.yml` | `Cargo.toml`, `.cargoignore` |
| MSRV Check (Issue #2934) | `msrv.yml` | `**/Cargo.toml`, `**/Cargo.lock` |

For a PR touching only `scripts/`, `.github/workflows/`, or `docs/`:
- `docs-hygiene.yml` **does** run (its path filter includes these paths)
- `architecture_drift.yml`, `crate-size.yml`, `msrv.yml` **do not** run

## Workflow-Only Promotion: Fast-Math Gate (Issue #3358)

The `Fast-Math vs IEEE-754 ASHRAE 600/900 Regression Gate (GH)` check
(#3358, #3326, #3322) was promoted from advisory to required in 2026-09
via the `(GH)` listener pattern. It has no `paths:` filter and runs on
every PR + push to `main`/`develop` + nightly at 04:00 UTC, so it is
included in **both** `required_checks` and `required_checks_workflow_only`.

The listener job in `.github/workflows/fast_math_check.yml` is named
`fast-math-gh` and emits the exact check name `"Fast-Math vs IEEE-754
ASHRAE 600/900 Regression Gate (GH)"`; the actual physics assertions
(`±0.05%` load agreement + `≤ 1e-5 W` residual ceiling, Issue #3326) live
in the upstream `compare` job. The listener is an additive change; the
workflow's top-level `name:` field is intentionally unchanged per
Issue #3358 directive ("Do NOT modify fast_math_check.yml's ``name:``
field; the promotion is at the release_gates layer, not the workflow").

The 4-week stability window (remaining acceptance criterion from #3358)
is tracked in the issue comments using the #3286 β-soak convention,
not in code. When the window closes, the live `develop` branch
protection is updated via `gh api --method PATCH` (preserves existing
contexts) to add the `(GH)` listener name to
`required_status_checks.contexts`; the YAML side already emits it.
Full operator procedure in
[`docs/ci/fast-math-stability-window.md`](fast-math-stability-window.md).

## Solution

`release_gates.yaml` now contains two arrays:

1. **`required_checks`** — All checks for code-changing PRs (30 checks). Use this for branch protection configuration on `main`.

2. **`required_checks_workflow_only`** — Only the checks that run on every PR regardless of changed files (26 checks). This excludes the 4 path-filtered checks above. The fast-math listener (#3358) is INCLUDED in this list because it has no `paths:` filter and runs on every PR (including workflow-only PRs like the one that lands this very gate's promotion).

### Branch Protection Configuration

**For `main` branch:** Use `required_checks` (all 23 checks). Code-changing PRs must pass all gates.

**For `develop` branch:** Use `required_checks_workflow_only` (19 checks). Workflow-only PRs (docs, CI, scripts) can merge without triggering the path-filtered checks that structurally cannot run for them.

### Alternative: Single Rule with Documentation

If GitHub branch protection only supports one required-checks list, use `required_checks_workflow_only` and document the behavior:

> **Note:** Some required checks (`Docs Hygiene Gate`, `Architecture Drift Detection`, `Crate Size Gate`, `MSRV Check`) have workflows that only run when specific file patterns are changed. For PRs touching only `scripts/`, `.github/workflows/`, or `docs/`, these checks will not run and are excluded from the required list. The 19 always-run checks provide adequate regression protection for workflow-only changes.

## Implementation Notes

- The path-filtered checks **still run** on PRs that touch the relevant files (e.g., `Cargo.toml` changes trigger `MSRV Check`)
- The path-filtered checks **still block** code-changing PRs that would affect them
- The `required_checks_workflow_only` list is a subset of `required_checks` — it removes only the 4 path-filtered checks

## See Also

- `release_gates.yaml::ci.required_checks`
- `release_gates.yaml::ci.required_checks_workflow_only`
- `.github/workflows/docs-hygiene.yml`
- `.github/workflows/architecture_drift.yml`
- `.github/workflows/crate-size.yml`
- `.github/workflows/msrv.yml`
