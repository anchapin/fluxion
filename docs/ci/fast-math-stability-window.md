# Fast-Math Gate Stability-Window Runbook — Issue #3358

**Issue:** #3358 (Closes), #3286 (template)  
**Date:** 2026-09-06  
**Status:** YAML-side promotion landed (#3358); live branch-protection activation
gates on the 4-week stability window sign-off below.

## Summary

This runbook is the **operator + reviewer companion** to
`docs/ci/fast-math-check.md` (the workflow design) and
`docs/ci/branch-protection-strict-mode.md` (the workflow-only rationale).
The actual promotion is structural — see those two documents for the
gate's design and the required-checks sync discipline.

The four remaining acceptance criteria from Issue #3358 are tracked here:

1. **4 weeks of green nightly runs accumulated** — tracked in the
   Issue #3358 comments using the #3286 β-soak convention (one comment
   per week, summarizing the seven nightly runs, the PR-triggered runs,
   and any Hetzner-overflow fallback events).
2. **GH Action runtime under 30 min confirmed** — recorded in the same
   weekly comment (the `compare` job has a 10-min `timeout-minutes`; the
   probe jobs each have 30-min; total runtime is observed via the
   `fast_math_check.yml` workflow run's `Summary` tab).
3. **Zero red→green-after-retry cycles** — also recorded in the weekly
   comment; any cycle counts against the stability window.
4. **Live `develop` branch-protection activation** — this is the final
   step; see "Activation procedure" below.

The structural YAML promotion (adding the `(GH)` listener to the
workflow + the `ci.required_checks` entry + the `ci.workflow_index`
entry + the `ci.required_checks_workflow_only` entry) was landed in
#3358 itself. Live activation is **gated** on the stability sign-off
so an unstable run cannot block develop merges.

## Why tracking lives in the issue, not in code

Issue #3286 established the convention: stability-window evidence for
required-check promotions lives in the GitHub issue comments (one
weekly summary per week) rather than in committed YAML. Reasons:

- **Append-only history** — weekly comments cannot be edited after the
  fact, so the trail cannot drift retroactively.
- **Cross-referenceable** — the issue link is the single source of
  truth; a future operator can audit it without checking out a commit
  hash.
- **No false signal in CI** — committing a "stability accumulated"
  flag would create a tautological gate (the gate asserts its own
  pass-condition).

## Weekly tracking template (copy-paste into Issue #3358 comments)

```markdown
## Week N stability summary — YYYY-MM-DD → YYYY-MM-DD

- **Nightly runs:** <count> (cron: 0 4 * * *)
- **PR-triggered runs:** <count> on develop
- **Hetzner-overflow fallback events:** <count>
- **Failed runs:** <count> (if 0, write "0 (clean week)")
- **Red→green-after-retry cycles:** <count> (if 0, write "0 (clean week)")
- **Worst-case `compare` runtime:** <minutes>:<seconds> (cap: 10 min)
- **Worst-case total workflow runtime:** <minutes> (cap: 30 min)
- **Max energy-balance residual under `--features fast-math`:** <W> (cap: 1e-5 W)
- **Max per-case load delta:** <%> (cap: 0.05%)

Cumulative status: <N>/4 weeks clean.
```

## Activation procedure (after the 4th clean week)

Once four consecutive clean weekly summaries are posted, the live
`develop` branch-protection can be activated:

1. **Verify the YAML-side promotion is in place on the develop HEAD:**

   ```bash
   git fetch origin develop
   git checkout origin/develop
   python3 scripts/check_required_checks_sync.py
   # Expected: "No drift. 30 required_check(s) and 31 workflow_index entr(ies)..."
   ```

2. **Snapshot the current contexts:**

   ```bash
   gh api /repos/anchapin/fluxion/branches/develop/protection/required_status_checks \
     | jq '.contexts'
   ```

3. **Add the new required check** (preserves existing contexts):

   ```bash
   EXISTING=$(gh api /repos/anchapin/fluxion/branches/develop/protection/required_status_checks \
     | jq -c '.contexts')
   NEW=$(echo "${EXISTING}" | jq '. + ["Fast-Math vs IEEE-754 ASHRAE 600/900 Regression Gate (GH)"]')
   gh api --method PATCH /repos/anchapin/fluxion/branches/develop/protection/required_status_checks \
     -H "Content-Type: application/json" \
     --input - <<EOF
   {"contexts": ${NEW}}
   EOF
   ```

4. **Verify live sync:**

   ```bash
   FLUXION_CHECK_LIVE_PROTECTION=1 python3 scripts/check_required_checks_sync.py
   # Expected: "Live develop branch protection matches release_gates.yaml..."
   ```

5. **Comment on Issue #3358** with the activation confirmation + the
   `gh api` PATCH response hash. Close the issue.

## Rollback procedure (if the gate fires post-activation)

If the fast-math check fires spuriously after activation (and the
red→green-after-retry pattern matches the #3116 false-positive class):

1. **Disable** the required check temporarily:

   ```bash
   CURRENT=$(gh api /repos/anchapin/fluxion/branches/develop/protection/required_status_checks \
     | jq -c '.contexts')
   REDUCED=$(echo "${CURRENT}" | jq 'map(select(. != "Fast-Math vs IEEE-754 ASHRAE 600/900 Regression Gate (GH)"))')
   gh api --method PATCH /repos/anchapin/fluxion/branches/develop/protection/required_status_checks \
     -H "Content-Type: application/json" \
     --input - <<EOF
   {"contexts": ${REDUCED}}
   EOF
   ```

2. **Comment on Issue #3358** with the false-positive diagnosis.
3. **Re-run** the stability window for one more week before re-activating.

## See Also

- `docs/ci/fast-math-check.md` — the workflow design + assertion contract
- `docs/ci/branch-protection-strict-mode.md` — the workflow-only rationale (#3142)
- Issue #3326 — the original advisory gate + acceptance criteria
- Issue #3322 — the `fast-math` feature flag + `fp_algebraic.rs` helper layer
- Issue #3324 — solar / irradiance reductions (first kernel consumer)
- Issue #3325 — AI batch metric reductions (planned consumer)
- Issue #3142 — required-checks sync discipline
- Issue #3286 — β-soak tracking convention (template this runbook mirrors)
- `release_gates.yaml::ci.required_checks` — the canonical required-check list
- `release_gates.yaml::ci.workflow_index` — the workflow-to-check map
- `scripts/check_required_checks_sync.py` — the drift guard (run with
  `FLUXION_CHECK_LIVE_PROTECTION=1` after activation)
