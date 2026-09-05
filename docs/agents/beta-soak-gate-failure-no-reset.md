# β-soak gate-failure protocol — no counter reset (Issue #3354)

The `Nightly ASHRAE 140 Gauge β-Soak` workflow
(`.github/workflows/nightly-ashrae-140-gauge.yml`) has been failing
every scheduled and `workflow_dispatch` run since 2026-09-01
(0 green / 6 failed at this writing). This note records the
**no-counter-reset** decision per the gate-failure protocol from
Issue #3286.

## Root cause

The failure is on **ADR-0007 Criterion 2 — energy conservation**
(`cargo test --locked --features gauge-solver --test
zone_balance_eplus_isolation`). Specifically, two tests fail consistently
across all six failed runs:

1. `test_physics_thermal_model_eplus_case_600_reference_csv`
   (`tests/zone_balance_eplus_isolation.rs:298:5`) — wild step-to-step
   oscillation, `max ΔT = 34.007 °C`.
2. `test_free_floating_case_900ff_isolation`
   (`tests/zone_balance_eplus_isolation.rs:445:5`) — numerical
   divergence to non-finite (`T_min = −∞` / `T_max = +∞`).

Both failures are listed verbatim in `docs/KNOWN_ISSUES.md`
**§LIMIT-21** (Issue #3297) — the "Gauge β-path pre-existing
air-trajectory failure cohort". The §LIMIT-21 entry notes the set is
verified **identical** at `fd7ef13^` = `832b0fe` and at HEAD
`0b54606` on 2026-09-03, i.e. **pre-existing, NOT introduced by any
recent PR**.

The workflow itself is correct: `.github/workflows/nightly-ashrae-140-gauge.yml`
runs all seven ADR-0007 criteria and the streak-computation is
stateless (`gh run list --workflow … --status completed --limit 60`,
uploaded as `beta-soak-state.json`, no commit-back). It fires the
gate-failure protocol exactly as designed.

## Soak counter — NO silent reset

The workflow computes the consecutive-green streak from the run
history; the current value is `0/30` because the most recent run
failed. **No manual reset is being performed and none is required**
— silently resetting the counter to mask a documented regression
would violate AGENTS.md / RULES.md / ADR-0001 (No-Parameter-Tuning
Rule).

## Workflow file changes

**None.** The workflow YAML, runner selection, dependencies, and step
order are correct. Per Issue #3354's scope guard, this PR does not
touch `.github/workflows/nightly-ashrae-140-gauge.yml` or any other
workflow / `release_gates.yaml` field — the workflow is firing the
gate-failure protocol as designed.

## Linked tracking

- Issue #3354 — β-soak workflow failing every run (parent diagnostic).
- Issue #3359 — tracking sub-issue linking nightly criterion-2 failures
  to §LIMIT-21 / §LIMIT-22.
- Issue #3286 — β-soak 30-day soak window tracking (gate contract).
- Issue #3284 — original nightly CI workflow infra.
- Issue #3291 — Phase A8 umbrella (gated on β-soak).
- Issue #3297 — §LIMIT-21 cohort owner (mass-state proxy aftermath).
- `docs/KNOWN_ISSUES.md` §LIMIT-21 — pre-existing air-trajectory cohort.
- `docs/KNOWN_ISSUES.md` §LIMIT-22 — exact-CN proxy aftermath.
- `docs/adr/0007-gauge-solver-structural-work.md` — ADR-0007 acceptance
  criteria.
- ADR-0001 — No-Parameter-Tuning Rule (why no baseline/constant
  changes).
- PR description (branch `fix/issue-3354-beta-soak-workflow-fix`)
  includes `Closes #3354` and the scope-guard line from Issue #3354.

## What a human operator should do

Until the §LIMIT-21 air-trajectory program lands (#3291 / #1465 /
#1462), the streak will remain at 0. If a human operator wishes to
formally retire this tracking surface — e.g., pause the β-soak
program pending §LIMIT-21 closure — they should close Issue #3359
with rationale and link the closure to a follow-up issue if needed.