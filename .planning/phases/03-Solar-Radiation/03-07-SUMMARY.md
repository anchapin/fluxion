---
phase: 03-Solar-Radiation
plan: 07
subsystem: physics-validation
tags: [ashrae-140, thermal-mass, hvac-demand, solar-gains, annual-energy]

# Dependency graph
requires:
  - phase: 03-01
  - phase: 03-04
  - phase: 03-06
  provides: [solar distribution fix, hvac demand diagnostic tests]
provides:
  - Corrected solar gain distribution parameterization (decoupled from internal radiative gains)
  - HVAC demand calculation analysis test for high-mass buildings
  - Solar gain distribution validation test
affects:
  - [Plan 03-07b validation of annual energy corrections]
  - [Plan 03-08 temperature swing gap closure]

# Tech tracking
tech-stack:
  added: []
  patterns: [diagnostic-test-pattern, tdd-red-phase]

key-files:
  created: []
  modified:
    - src/sim/engine.rs - Fixed solar_distribution_to_air parameterization
    - tests/ashrae_140_case_900.rs - Added hvac demand and solar distribution validation tests

key-decisions:
  - "Decoupled solar_distribution_to_air from solar_beam_to_mass_fraction to fix incorrect internal radiative gain distribution"
  - "Set solar_distribution_to_air = 0.0 for all ASHRAE 140 cases (internal radiative gains to surface, not air)"
  - "Set solar_beam_to_mass_fraction = 0.7 for high-mass cases (ASHRAE 140 specification)"
  - "Root cause of annual energy over-prediction NOT identified - requires deeper investigation"

patterns-established:
  - "Diagnostic test pattern: Create analysis tests to identify root causes before implementing fixes"

requirements-completed: []

# Metrics
duration: 45min
completed: 2026-03-09
---

# Phase 03: Plan 07 Summary

**Solar gain distribution parameterization fix with diagnostic tests, but annual energy over-prediction root cause not identified**

## Performance

- **Duration:** 45 min
- **Started:** 2026-03-09T20:15:00Z
- **Completed:** 2026-03-09T21:00:00Z
- **Tasks:** 2 of 3 completed (diagnostic tests completed, fix applied but issue not resolved)
- **Files modified:** 2

## Accomplishments

- Fixed solar gain distribution parameterization by decoupling `solar_distribution_to_air` from `solar_beam_to_mass_fraction`
- Created diagnostic test `test_case_900_hvac_demand_calculation_analysis` to analyze HVAC demand behavior
- Created validation test `test_case_900_solar_gain_distribution_validation` to verify solar distribution parameters
- Identified that HVAC runs at maximum capacity (2100 W) for many hours due to low free-floating temperatures (7-10°C vs 20°C setpoint)

## Task Commits

1. **Task 1: Investigate hvac_power_demand calculation for high-mass buildings** - `e5e5646` (feat)
2. **Task 2: Validate solar gain distribution parameters for high-mass buildings** - `e5e5646` (feat)

**Plan metadata:** Not complete (Task 3 not executed - annual energy still over-predicted)

## Files Created/Modified

- `src/sim/engine.rs` - Fixed solar_distribution_to_air parameterization, added diagnostic output for HVAC demand and sensitivity
- `tests/ashrae_140_case_900.rs` - Added `test_case_900_hvac_demand_calculation_analysis` and `test_case_900_solar_gain_distribution_validation`

## Decisions Made

- Decoupled `solar_distribution_to_air` from `solar_beam_to_mass_fraction` to fix incorrect parameter coupling
  - Rationale: The two parameters were incorrectly coupled, causing 30% of internal radiative gains to go directly to air instead of surface
  - Set `solar_distribution_to_air = 0.0` (internal radiative gains go to surface, not air)
  - Set `solar_beam_to_mass_fraction = 0.7` for high-mass cases (70% of beam solar to mass per ASHRAE 140)
- Root cause of annual energy over-prediction NOT identified despite solar distribution fix
  - Annual cooling: 4.69 MWh vs [2.13, 3.67] MWh reference (28-120% above)
  - Annual heating: 6.90 MWh vs [1.17, 2.04] MWh reference (239-491% above)
  - Issue appears to be thermal mass dynamics or conductance calculation, not solar distribution

## Deviations from Plan

### Auto-fixed Issues

None - followed plan as specified for Tasks 1 and 2.

### Plan Incompletion

**Task 3: Fix identified issues in hvac_power_demand and/or solar distribution** - NOT COMPLETED

- **Reason:** Solar distribution fix (decoupling parameters) did not resolve annual energy over-prediction
- **Remaining issue:** Annual energy still ~2x too high after fix
- **Diagnostic findings:**
  - HVAC runs at maximum capacity (2100 W) for many hours
  - Free-floating temperature (Ti_free) is 7-10°C during winter (far below 20°C setpoint)
  - Sensitivity = 0.0021 K/W (relatively low, causing high HVAC demand)
  - HVAC runtime: 78.5% of hours (21.5% off, 44.4% heating, 34.2% cooling)
- **Root cause hypothesis:** Thermal mass dynamics or conductance calculation issue (not solar distribution)
- **Status:** Requires deeper investigation beyond initial hypothesis scope

**Total deviations:** 1 plan incompletion (Task 3 - fix not successful)
**Impact on plan:** Annual energy over-prediction not resolved. Plan 03-07b validation cannot proceed until issue is fixed.

## Issues Encountered

- Solar distribution parameterization fix did not reduce annual energy over-prediction
  - Expected: Annual cooling and heating within ASHRAE 140 reference ranges
  - Actual: Annual cooling 4.69 MWh (unchanged from 4.68 MWh), heating 6.90 MWh (unchanged from 6.91 MWh)
  - Root cause appears to be thermal mass dynamics or conductance calculation, not solar/internal radiative gain distribution
- HVAC demand diagnostic revealed that HVAC runs at maximum capacity for many hours
  - This suggests building loses/gains heat too fast, but peak loads are correct
  - Indicates issue with sensitivity calculation or thermal mass response, not peak demand

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

**NOT READY for Plan 03-07b validation**

- Plan 03-07b depends on Plan 03-07 completing annual energy fixes
- Annual energy over-prediction not resolved (still 2x too high)
- Blockers:
  1. Root cause of annual energy over-prediction not identified
  2. Thermal mass dynamics or conductance calculation issue needs investigation
  3. May require deeper physics analysis beyond initial solar distribution hypothesis

**Recommended next steps:**
- Investigate thermal mass coupling conductances (h_tr_em, h_tr_ms) for high-mass buildings
- Analyze sensitivity calculation to verify it's correct for high-mass dynamics
- Consider whether thermal capacitance (Cm) is correct for Case 900
- Investigate if thermal mass is absorbing/releasing heat as expected (check mass temperature dynamics)

---
*Phase: 03-Solar-Radiation*
*Completed: 2026-03-09*
