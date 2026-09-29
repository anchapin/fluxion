# CI Baseline Maintenance Refactor — Design Doc (Issue #4243)

## Problem

Physics PRs routinely require manual updates to CI enforcement files. The #4155
session (2026-09-29) needed 4 one-time PATs; most were for baseline/ratchet bumps,
not code. The files:

| File | What must be hand-updated | Frequency |
|------|---------------------------|-----------|
| `scripts/check_test_inventory_drift.py` | `BASELINE_LIB_TESTS`, `BASELINE_WORKSPACE_TESTS` constants | Every test-adding PR |
| `tests/reference_data/test_inventory_baseline.json` | Regenerate via `--update-baseline` | Every test-adding PR |
| `scripts/check_module_size.py` | `max_lines` per ratcheted file | When a ratcheted module grows |
| `tests/reference_data/module_size/*.json` | `max_lines` + history entry | When a ratcheted module grows |
| `scripts/check_physics_sim_cycle.py` | `BASELINE_SIM_TO_PHYSICS` | When sim→physics edges grow |
| `AGENTS.md` | Hardcoded test counts (4 locations) | Every test-count change |
| `SCORECARD.md` | Auto-regenerated (no manual work) | — |

## Root causes

1. **Absolute ratchets instead of relative guards.** The checks enforce
   `live <= HARDCODED_MAX`. Every legitimate growth requires a human to bump
   the constant. The alternative — enforce `growth_per_PR <= ALLOWANCE` — needs
   no manual update for legitimate growth.

2. **Constants live in script code, not data.** `BASELINE_LIB_TESTS = 4296` is
   a Python constant inside the check script. Moving all such constants into
   the JSON reference-data files (like module-size ratchets already do) would
   at least keep churn in data, not code — but doesn't eliminate it.

3. **No auto-bump on merge.** SCORECARD.md auto-regenerates via
   `scorecard-drift.yml` (Issue #3128) — the PR author does nothing. The test
   inventory and module-size baselines could follow the same pattern: CI
   auto-bumps the baseline on merge to `develop`, PRs only fail if they
   *shrink* the baseline unexpectedly or exceed a per-PR growth cap.

4. **AGENTS.md counts are hand-synced.** A pre-commit hook or CI job could
   rewrite the four cited figures from `tests/test_inventory.json`.

## Recommended approach

**Phase 1 (low risk, high leverage): auto-bump on merge to develop.**
Extend the scorecard-drift pattern:
- New workflow `baseline-auto-bump.yml` triggers on push to `develop`.
- Regenerates `test_inventory_baseline.json`, bumps `BASELINE_*` in the check
  scripts (via a script, not hand-edit), updates module-size JSON ratchets,
  rewrites AGENTS.md counts.
- Commits directly to `develop` (no PR, no PAT needed — uses `GITHUB_TOKEN`).

PRs then only fail the drift gates if:
- They *reduce* test counts without explanation (shrink guard), or
- A single PR grows a ratcheted module by more than the per-PR allowance
  (e.g. max(5%, 50 lines)), catching accidental bloat while allowing
  legitimate growth.

**Phase 2 (medium risk): relative ratchets.**
Replace absolute `live <= MAX` with `live - baseline_at_branch_point <= ALLOWANCE`.
Requires the check to know the branch point (available via `git merge-base`).
Eliminates the need for *any* manual bump for legitimate growth; the baseline
becomes informational, not enforcement.

**Phase 3 (cleanup): consolidate constants.**
Move all remaining `BASELINE_*` Python constants into JSON reference-data files.
Single source of truth; scripts become logic-only.

## What NOT to change

- Enforcement semantics: the gates must still catch unintended growth. Auto-bump
  on merge preserves this (the bump happens *after* human review of the PR).
- ASHRAE 140 bands/tolerances: out of scope. This is about CI maintenance toil,
  not validation thresholds.
- The `workflow` OAuth scope issue: separate work (vault-backed push, currently
  parked on platform bugs). This refactor reduces the *frequency* of pushes
  needing it but doesn't solve the auth story.

## Estimated impact

- **Before:** ~3-4 manual file updates per physics PR (baseline script, JSON,
  AGENTS.md, sometimes module-size). Each requires a PAT push if workflows are touched.
- **After Phase 1:** zero manual updates for test-count/module-size growth.
  PR authors run no baseline scripts. PAT pushes only for actual workflow changes.
- **After Phase 2:** also eliminates the per-PR allowance check complexity;
  the merge-base diff is the single enforcement point.

## Open questions

1. Does auto-bump on merge risk masking a PR that accidentally deletes tests?
   Mitigation: the shrink guard (fail if `live < baseline - tolerance`).
2. Should the auto-bump commit be signed/attributed to a bot account?
3. Phase 2 needs `git merge-base` in CI — available in `actions/checkout` with
   `fetch-depth: 0`. Confirm no shallow-clone issues.
