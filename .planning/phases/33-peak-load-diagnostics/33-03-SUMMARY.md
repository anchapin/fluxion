---
phase: 33-peak-load-diagnostics
plan: 03
subsystem: validation
tags: [diagnostics, heat-balance, root-cause]
requires: [PEAK-04, FLOAT-04]
provides: [final-diagnostic-report]
affects: [src/sim/engine.rs]
tech-stack: [rust, physics-analysis]
key-files: [docs/phases/33-peak-load-diagnostics/33-DIAGNOSTIC-REPORT.md]
decisions:
  - "Identify missing floor/roof thermal mass in Cm as the primary root cause for peak load overestimation."
  - "Recommend unifying envelope conduction (h_tr_ms/h_tr_em) to include all opaque surfaces in Phase 34."
metrics:
  duration: 25m
  completed_date: "2026-04-03"
---

# Phase 33 Plan 03: Final Diagnostic Consolidation Summary

## Substantive One-Liner
Consolidated heat balance component audit identifying missing floor/roof thermal mass and coupling as the root cause for peak load and free-floating temperature failures.

## Accomplishments
- **Heat Balance Audit**: Analyzed Case 900 hourly diagnostic data and identified that the 5R1C network was only including wall mass and conduction, ignoring the heavy concrete floor and roof.
- **Diagnostic Report**: Created a comprehensive report (`docs/phases/33-peak-load-diagnostics/33-DIAGNOSTIC-REPORT.md`) detailing the 10x under-estimation of the building's time constant ($\tau$) and its impact on peak load overestimation (+65% for heating).
- **Physics Correction Roadmap**: Outlined specific corrections for Phase 34, including integrating all envelope elements into $C_m$, $H_{ms}$, and $H_{em}$, and refining surface-to-air coupling.

## Root Cause Summary
1.  **Mass Missing:** `Cm` excludes floor/roof mass (~60% of total mass missing).
2.  **Coupling Weak:** `h_tr_ms` excludes floor/roof area, leading to weak air-mass thermal coupling.
3.  **Under-damping:** Low mass + weak coupling leads to a 4.5h time constant instead of the expected 50h+, causing extreme diurnal swings and high HVAC peaks.

## Deviations from Plan
None - plan executed as written.

## Self-Check: PASSED
- [x] Diagnostic report created and committed
- [x] Root cause for peak overestimation identified
- [x] SUMMARY.md created
- [x] ROADMAP.md/STATE.md updated (via final state update)
