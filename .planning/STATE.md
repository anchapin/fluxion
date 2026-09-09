---
gsd_state_version: 1.0
milestone: v1.3
milestone_name: Blind ASHRAE 140 Validation (Physics Only)
current_phase: phase-a8-ashrae-140-blind-validation
status: in_progress
stopped_at: "GaugeSolver production-path switchover shipped 2026-09-07 (Phase A8, Issue #3291 / PR #3482); β-soak gate at 0/30 nights green (Issue #3286); ≥80% blind-validation coverage is the next milestone gate"
last_updated: "2026-09-09"
progress:
  total_phases: 5
  completed_phases: 0
  total_plans: 0
  completed_plans: 0
  notes: "v1.3 is structured as the A–E phase rollup (A baseline stripping, B physics fixes, C benchmark correction, D blind validation pass, E sustained CI/validation). Current concrete shipped work under v1.3 is Phase A8 (GaugeSolver production-path switchover), tracked under the A3 cohort umbrella. Per `AGENTS.md:12` and `docs/adr/0007-gauge-solver-structural-work.md`."
---

# Fluxion Project State

**Milestone:** v1.3 Blind ASHRAE 140 Validation (Physics Only)
**Last Updated:** 2026-09-09
**Current Phase:** phase-a8-ashrae-140-blind-validation
**Decision:** Planning artifacts synchronized from shipped reality (Phase A8 / Issue #3291 / PR #3482 closed 2026-09-07); frontmatter and body refreshed to v1.3 per issue #3632.

## Live validation snapshot (2026-08-11 run; mirror of `.planning/PROJECT.md`)

| Metric | Current | Release-gate target | Status |
|--------|---------|---------------------|--------|
| Pass rate (metric-level) | **20.3%** (13/64) | ≥ 60% | ❌ Fail |
| Mean Absolute Error (MAE) | **55.09%** | ≤ 50% | ❌ Fail |
| Cases fully passing | 1/18 (5.6%) | — | ❌ |
| Strict ±15% annual-energy gate (Cases 600/900 heating) | passes | passes | ✅ |
| Strict ±15% Cases 600/900 cooling (annual) | fails structurally | passes | ❌ |

The canonical status snapshot lives in `SCORECARD.md` (auto-generated) and `docs/ASHRAE140_RESULTS.md`. This section is the planning-artifact mirror and may lag the canonical snapshot by one regeneration cycle.

## Progress Summary

| Status | Count |
|--------|-------|
| ✅ Phase A8 shipped (GaugeSolver production-path switchover) | 1 |
| 🚧 Executing (β-soak gate, Issue #3286) | 1 |
| 📋 Planning (Phases B / C / D / E) | 4 |

**Overall:** Phase A8 shipped on 2026-09-07 (Issue #3291 / PR #3482, commit `e811df6`); production-path switchover is unconditional with the `gauge-solver` cargo feature on, gated behind the β-soak gate (Issue #3286, currently 0/30 nights green) for unconditional-feature-on default per `AGENTS.md:12`.

## Phase status

- ✅ **phase-a8-ashrae-140-blind-validation (GaugeSolver production-path switchover)** — Shipped 2026-09-07 (Issue #3291 / PR #3482, ADR-0007, commit `e811df6`). Cross-references: `AGENTS.md:12`, `ARCHITECTURE.md:716-735` and `:1267-1269`, `docs/KNOWN_ISSUES.md` §LIMIT-21 / §LIMIT-22, `docs/adr/0007-gauge-solver-structural-work.md`, `src/sim/thermal_selector.rs`.
- 🚧 **Issue #3286 β-soak gate** — Production-path switchover waits on `#3286 β-soak` reaching 30/30 nights green (currently 0/30) per `AGENTS.md` and `docs/agents/beta-soak-criterion-2-tracker.md`.
- 📋 **Phase B — Physics fixes** (B.1 solar distribution / B.2 thermal-mass time constant / B.3 free-floating temperature; PHYSICS-01/02/03) — root-cause path for the Case 600/900 structural failures.
- 📋 **Phase C — Benchmark correction** (BENCH-01): replace "calibrated for 5R1C" ranges with true ASHRAE reference values from EnergyPlus / ESP-r / TRNSYS.
- 📋 **Phase D — Blind validation pass** (VALIDATE-01): run the full blind suite targeting ≥80% case coverage at the ±15%/±10%/±15%/±1.0°C gates.
- 📋 **Phase E — Sustained validation** (SUSTAIN-01 / SUSTAIN-02): CI gate prevents merges below 80% pass rate + annual re-validation against the latest ASHRAE reference data.

## Key decisions recorded for v1.3

| Decision | Rationale | Status |
|----------|-----------|--------|
| Blind ASHRAE 140 validation methodology (v1.3) | Replace "informed" validation with physics-only blind methodology; no calibration factors, no case-ID hints, true ASHRAE reference values | 🚧 in progress |
| GaugeSolver is the production-path default (Phase A8 / ADR-0007) | Structural fix for the 5R1C/CTF high-mass and low-mass peak-load failures — treats solar as geometric curvature rather than per-timestep energy injection | ✅ shipped 2026-09-07 |
| No parameter tuning (`RULES.md`, `ADR-0001`) | Fix the underlying math, never tune to make tests pass | ✅ held |
| `gauge-solver` cargo feature retained as the production-path gate pending §LIMIT-21 / #3297 closure | β-soak gate (Issue #3286) gates unconditional default-on; `FiveROneC` and `NineRFourC` remain explicit opt-in legacy paths | 🚧 in progress |

## Sync Status

This file was manually refreshed for the v1.3 milestone on 2026-09-09 by issue #3632.
The previous frontmatter (milestone `v1.2`, last_updated `2026-04-19`) was the output of `python scripts/sync_planning.py` (DX-01) on 2026-04-19; that generation no longer reflects shipped reality (Phase A8 / Issue #3291 / PR #3482 shipped 2026-09-07, the v1.3 framing is in `AGENTS.md:3`, `README.md:5`, `CHANGELOG.md:5`, `.planning/PROJECT.md:7-9`, `.planning/ROADMAP.md:4`, and `.planning/MILESTONES.md` has a v1.3 section). The v1.2 historical record is preserved in `.planning/MILESTONES.md` (v1.1/v1.2 entries + the v0.8.0/v0.4/v0.2 rollup), `.planning/v1.2-MILESTONE-VERIFICATION.md`, and `.planning/MILESTONE_v1.2_SUMMARY.md` for traceability.

Last refreshed: 2026-09-09
Refreshing issue: #3632
