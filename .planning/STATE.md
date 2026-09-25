---
gsd_state_version: 1.0
milestone: v1.3
milestone_name: Blind ASHRAE 140 Validation (Physics Only)
current_phase: phase-b-physics-fixes
status: in_progress
stopped_at: "Phase B sub-phases B1a (Issue #3797, PR pending) PHYSICS-01 solar distribution audit (Case 600, §LIMIT-30 in docs/KNOWN_ISSUES.md) shipped 2026-09-17 + B1b (Issue #3798 / PR #3847) release-gates cohort registration already merged + B2a (Issue #3799 / PR #3845) PHYSICS-02 thermal-mass τ characterization (Case 900, §LIMIT-29) already merged + B2b (Issue #3800 / PR #3841) already merged + B3a (Issue #3801 / PR #3838) PHYSICS-03 free-floating deviation audit already merged + B3b (Issue #3802 / PR #3835) §LIMIT-28 FF cohort entry already merged; GaugeSolver production-path switchover shipped 2026-09-07 (Phase A8, Issue #3291 / PR #3482); β-soak gate at 0/30 nights green (Issue #3286); ≥80% blind-validation coverage is the next milestone gate"
last_updated: "2026-09-25"
progress:
  total_phases: 5
  completed_phases: 0
  total_plans: 0
  completed_plans: 0
  notes: "v1.3 is structured as the A–E phase rollup (A baseline stripping, B physics fixes, C benchmark correction, D blind validation pass, E sustained CI/validation). Phase B sub-phases B.1 (PHYSICS-01 solar distribution, B1a audit + B1b release-gates), B.2 (PHYSICS-02 thermal-mass τ, B2a audit + B2b release-gates), B.3 (PHYSICS-03 free-floating temperature, B3a audit + B3b LIMIT-28) all shipped in docs as of 2026-09-17; structural fix routed to GaugeSolver production-path work coordinated by #1465 / #1462 (production-path staged via #3291 / PR #3482 — Phase A8 default flip, gated on §LIMIT-21 β-soak closure). Per `AGENTS.md:12` and `docs/adr/0007-gauge-solver-structural-work.md`."
---

# Fluxion Project State

**Milestone:** v1.3 Blind ASHRAE 140 Validation (Physics Only)
**Last Updated:** 2026-09-25
**Current Phase:** phase-b-physics-fixes
**Decision:** Planning artifacts synchronized from shipped reality (Phase A8 / Issue #3291 / PR #3482 closed 2026-09-07; Phase B sub-phases B.1 / B.2 / B.3 audit + release-gates shipped 2026-09-16/17 — B1a #3797, B1b #3798, B2a #3799, B2b #3800, B3a #3801, B3b #3802 all closed at the docs level with §LIMIT-29 / §LIMIT-30 structural LIMIT entries); frontmatter and body refreshed to v1.3 per issue #3632 and the Phase B wave (PRs #3835 / #3838 / #3841 / #3845 / #3847 / pending B1a).

## Live validation snapshot (2026-09-07 run; mirror of `SCORECARD.md`)

| Metric | Current | Release-gate target | Status |
|--------|---------|---------------------|--------|
| Pass rate (metric-level) | **14.1%** (11/84) | ≥ 60% | ❌ Fail |
| Mean Absolute Error (MAE) | **49.82%** | ≤ 50% | ✅ Pass |
| Cases fully passing | 0/18 (0.0%) | — | ❌ |
| Strict ±15% annual-energy gate (Cases 600/900 heating) | passes | passes | ✅ |
| Strict ±15% Cases 600/900 cooling (annual) | fails structurally | passes | ❌ |

`SCORECARD.md` (auto-generated — regenerate with `python3 scripts/generate_scorecard.py`, never hand-edit) is the single source of truth for these numbers; `docs/ASHRAE140_RESULTS.md` carries the detailed breakdown. This section is a planning-artifact mirror and may lag the canonical snapshot by one regeneration cycle.

## Progress Summary

| Status | Count |
|--------|-------|
| ✅ Phase A8 shipped (GaugeSolver production-path switchover) | 1 |
| ✅ Phase B sub-phases shipped (B.1 / B.2 / B.3 audit + release-gates) | 6 |
| 🚧 Executing (β-soak gate, Issue #3286) | 1 |
| 📋 Planning (Phases C / D / E) | 3 |

**Overall:** Phase A8 shipped on 2026-09-07 (Issue #3291 / PR #3482, ADR-0007, commit `e811df6`); production-path switchover is unconditional with the `gauge-solver` cargo feature on, gated behind the β-soak gate (Issue #3286, currently 0/30 nights green) for unconditional-feature-on default per `AGENTS.md:12`. Phase B sub-phases all shipped at the docs level as of 2026-09-17: B.1 PHYSICS-01 (B1a audit Issue #3797, B1b release-gates Issue #3798 / PR #3847), B.2 PHYSICS-02 (B2a audit Issue #3799 / PR #3845 + §LIMIT-29, B2b release-gates Issue #3800 / PR #3841), B.3 PHYSICS-03 (B3a audit Issue #3801 / PR #3838, B3b LIMIT-28 Issue #3802 / PR #3835). The structural fix is uniformly routed to the GaugeSolver production-path work coordinated by #1465 / #1462 (per §LIMIT-21 β-soak closure).

## Phase status

- ✅ **phase-a8-ashrae-140-blind-validation (GaugeSolver production-path switchover)** — Shipped 2026-09-07 (Issue #3291 / PR #3482, ADR-0007, commit `e811df6`). Cross-references: `AGENTS.md:12`, `ARCHITECTURE.md:716-735` and `:1267-1269`, `docs/KNOWN_ISSUES.md` §LIMIT-21 / §LIMIT-22, `docs/adr/0007-gauge-solver-structural-work.md`, `src/sim/thermal_selector.rs`.
- ✅ **Phase B.1 — Solar distribution (PHYSICS-01)** — Shipped 2026-09-17 at the docs level. B1a audit (Issue #3797) per-tilt / per-azimuth incident-energy deviation table vs the analytical cos(θᵢ) reference + per-surface distribution metrics ranked (`solar_distribution_to_air = 0.30` vs ASHRAE 140 expectation 0.0, Δ +0.30, worst) + per-surface Case 600 5R1C h_tr_is / h_tr_ms conductance deviations vs hand-calc (−86.8 % / −78.0 %) + three mechanism hypotheses + module-isolation suites green (read-only) → documented as `docs/ASHRAE140_RESULTS.md` §"Phase B1a (PHYSICS-01) solar distribution audit (Case 600, Issue #3797)" and `docs/KNOWN_ISSUES.md` §LIMIT-30. B1b release-gates registration (Issue #3798 / PR #3847) already merged in `release_gates.yaml` lines 83–110 (Case 600 series annotation, "B1b solar-distribution cohort, Issue #3798"). Cross-references: `docs/KNOWN_ISSUES.md` §LIMIT-05 / §LIMIT-13 / §LIMIT-16 / §LIMIT-17 / §LIMIT-21 / §LIMIT-22 / §LIMIT-29 / §LIMIT-30, SOLAR-01 / Issue #274, #3072 (aggressive-baseline cohort), Issue #1323-#1337 (fixture-data lineage).
- ✅ **Phase B.2 — Thermal-mass time constant (PHYSICS-02)** — Shipped 2026-09-17 at the docs level. B2a audit (Issue #3799 / PR #3845) Case 900 τ characterization + §LIMIT-29 + three mechanism hypotheses + module-isolation suites green → documented as `docs/ASHRAE140_RESULTS.md` §"Phase B2a (PHYSICS-02) thermal-mass time-constant characterization (Case 900, post-#3770)". B2b release-gates registration (Issue #3800 / PR #3841) in `release_gates.yaml` lines 64–81 (Case 900 annotation, "B2b thermal-mass cohort, Issue #3800"). Cross-references: §LIMIT-05 / §LIMIT-05 UPDATE (#2453), §LIMIT-13, §LIMIT-16 / #3059, §LIMIT-17 / #3058 / ADR-0011, §LIMIT-21, §LIMIT-22, #3770 (mass-node 141°C regression blocking the B2a `test_thermal_mass_temperature_damping` un-quarantine).
- ✅ **Phase B.3 — Free-floating temperature (PHYSICS-03)** — Shipped 2026-09-16/17 at the docs level. B3a audit (Issue #3801 / PR #3838) per-case FF deviation audit (600FF / 900FF / 950FF) + ranking → documented as `docs/ASHRAE140_RESULTS.md` §"Phase B3a (PHYSICS-03) free-floating temperature deviation audit (600FF/900FF/950FF)". B3b LIMIT-28 (Issue #3802 / PR #3835) FF cohort (600FF / 650FF / 900FF / 950FF) bidirectional diurnal-swing gap → documented as `docs/ASHRAE140_RESULTS.md` §"Free-Floating cohort (600FF/650FF/900FF/950FF) bidirectional diurnal-swing gap" + `docs/KNOWN_ISSUES.md` §LIMIT-28. Cross-references: §LIMIT-17 / #3058 / ADR-0011, §LIMIT-24 / #3551, §LIMIT-16 / #3059, §LIMIT-05 UPDATE (#2453), §LIMIT-15 / #3060, §FREE-01 / §FREE-02 / §FREE-03, §LIMIT-21, #1465 / #1462 / #3291 / #3286.
- 🚧 **Issue #3286 β-soak gate** — Production-path switchover waits on `#3286 β-soak` reaching 30/30 nights green (currently 0/30) per `AGENTS.md` and `docs/agents/beta-soak-criterion-2-tracker.md`.
- 📋 **Phase C — Benchmark correction** (BENCH-01): replace "calibrated for 5R1C" ranges with true ASHRAE reference values from EnergyPlus / ESP-r / TRNSYS.
- 📋 **Phase D — Blind validation pass** (VALIDATE-01): run the full blind suite targeting ≥80% case coverage at the ±15%/±10%/±15%/±1.0°C gates.
- 📋 **Phase E — Sustained validation** (SUSTAIN-01 / SUSTAIN-02): CI gate prevents merges below 80% pass rate + annual re-validation against the latest ASHRAE reference data.

## Key decisions recorded for v1.3

| Decision | Rationale | Status |
|----------|-----------|--------|
| Blind ASHRAE 140 validation methodology (v1.3) | Replace "informed" validation with physics-only blind methodology; no calibration factors, no case-ID hints, true ASHRAE reference values | 🚧 in progress |
| GaugeSolver is the production-path default (Phase A8 / ADR-0007) | Structural fix for the 5R1C/CTF high-mass and low-mass peak-load failures — treats solar as geometric curvature rather than per-timestep energy injection | ✅ shipped 2026-09-07 |
| Phase B.1 / B.2 / B.3 audit + release-gates shipped at docs level | Per-axis attribution for Case 600 series (B.1 / §LIMIT-30), Case 900 (B.2 / §LIMIT-29), FF cohort (B.3a / §LIMIT-28); structural fix uniformly routed to GaugeSolver #1465 / #1462 | ✅ shipped 2026-09-17 |
| No parameter tuning (`RULES.md`, `ADR-0001`) | Fix the underlying math, never tune to make tests pass | ✅ held |
| `gauge-solver` cargo feature retained as the production-path gate pending §LIMIT-21 / #3297 closure | β-soak gate (Issue #3286) gates unconditional default-on; `FiveROneC` and `NineRFourC` remain explicit opt-in legacy paths | 🚧 in progress |

## Sync Status

This file was manually refreshed for the v1.3 milestone on 2026-09-17 by issue #3797 (Phase B1a solar distribution audit deliverable).
The previous refresh on 2026-09-09 was the output of issue #3632 (Phase A8 / Issue #3291 / PR #3482 sync); that generation reflected the post-A8 reality. The 2026-09-17 refresh adds Phase B sub-phases B.1 / B.2 / B.3 to the Phase status block — B1a Issue #3797 (this PR's deliverable), B1b Issue #3798 / PR #3847, B2a Issue #3799 / PR #3845 + §LIMIT-29, B2b Issue #3800 / PR #3841, B3a Issue #3801 / PR #3838, B3b Issue #3802 / PR #3835 + §LIMIT-28 — all shipped at the docs level with structural fix uniformly routed to the GaugeSolver production-path work coordinated by #1465 / #1462. The v1.2 historical record is preserved in `.planning/MILESTONES.md` (v1.1/v1.2 entries + the v0.8.0/v0.4/v0.2 rollup), `.planning/v1.2-MILESTONE-VERIFICATION.md`, and `.planning/MILESTONE_v1.2_SUMMARY.md` for traceability.

Last refreshed: 2026-09-25
Refreshing issue: #3999 (scorecard-number sync only — snapshot section refreshed to the verified 2026-09-07 SCORECARD.md numbers); previous refreshes: #3797 (2026-09-17, Phase B1a audit); #3632 (2026-09-09, Phase A8 sync)
