---
phase: 33-peak-load-diagnostics
plan: 01
subsystem: validation
tags: [diagnostics, peak-load, profile-comparison]
requires: [PEAK-03, PEAK-04]
provides: [hourly-diagnostic-export, peak-load-quantification]
affects: [src/sim/engine.rs, src/validation/diagnostics.rs]
tech-stack: [rust, python, numpy]
key-files: [src/validation/diagnostics.rs, src/sim/engine.rs, tests/case_900_peak_diagnostic.rs, scripts/compare_peak_profiles.py]
decisions:
  - 1. Expand SimulationDiagnostics to record internal states (Zone/Mass/Surface temps, Fluxes)
  - 2. Implement sub-hourly to hourly averaging in the comparison script to handle EnergyPlus reference data
metrics:
  duration: 15m
  completed_date: "2026-04-03"
---

# Phase 33 Plan 01: Case 900 Peak Load Diagnostics Summary

## Substantive One-Liner
High-resolution hourly diagnostic suite for Case 900 peak load analysis, including internal state recording and profile comparison metrics.

## Accomplishments
- **Expanded SimulationDiagnostics**: Added recording for outdoor/ground temperatures, mass temperatures (Tm), surface temperatures (Ts), and detailed load breakdowns (solar, internal, HVAC, inter-zone, infiltration).
- **Engine Integration**: Integrated diagnostics into the `step_physics` method in `src/sim/engine.rs` to capture transient states.
- **Case 900 Peak Diagnostic Test**: Created an integration test that runs Case 900 for a full year and exports data to `case_900_peak_hourly.csv`.
- **Profile Comparison Script**: Developed `scripts/compare_peak_profiles.py` to quantify peak overestimation and timing shifts against EnergyPlus reference data.

## Peak Load Findings
The initial diagnostic run for Case 900 shows significant overestimation of peak heating:
- **Peak Heating**: Fluxion 4437 W vs Reference 2687 W (+65.1%)
- **Peak Cooling**: Fluxion 3415 W vs Reference 3041 W (+12.3%)

The cooling peak is relatively close, but the heating peak remains significantly overestimated. This provides the baseline for the physics fixes in Phase 34.

## Deviations from Plan
- **Rule 3 - Blocking Issue**: The EnergyPlus reference JSON contained sub-hourly data (10-minute intervals). Updated the comparison script to handle this by averaging to hourly intervals to match Fluxion's output.
- **Rule 3 - Blocking Issue**: The environment lacked `matplotlib`. Modified the comparison script to make plotting optional while still providing quantitative metrics.

## Self-Check: PASSED
- [x] Task 1 complete (Test created, CSV exported)
- [x] Task 2 complete (Comparison script created)
- [x] SUMMARY.md created
- [x] ROADMAP.md/STATE.md updated (via state advance-plan)
