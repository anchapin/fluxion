---
phase: 15-hvac-equipment-modeling
plan: 06
subsystem: hvac-equipment
tags: [hvac, equipment, variable-capacity, predictive-control, cycling-loss, economizer, thermal-network]

# Dependency graph
requires:
  - phase: 15-04 (predictive control, cycling tracker, economizer)
    provides: PredictiveController, CyclingTracker, EconomizerMode fields in ThermalModel
provides:
  - hvac_equipment field integration for dynamic equipment attachment
  - VariableCapacityEquipment.calculate_power() integration in physics loop
  - PredictiveController.calculate_modulation() integration for thermal inertia control
  - CyclingTracker.calculate_cycling_loss() integration for efficiency penalties
  - Economizer.is_economizer_active() integration for free cooling
affects:
  - 15-07 (HVAC integration tests - needs equipment integration to test)
  - Phase 16 (Psychrometrics - economizer enthalpy mode needs psychrometrics)
  - Future phases requiring realistic HVAC simulation

# Tech tracking
tech-stack:
  added: []
  patterns:
  - Enum wrapper pattern for dynamic trait object dispatch (AnyEquipment)
  - Thermal inertia-based predictive control with dT/dt calculation
  - Backward compatibility fallback (IdealHVACController when no equipment attached)

key-files:
  created:
    - src/sim/hvac/equipment.rs (AnyEquipment enum wrapper)
  modified:
    - src/sim/engine.rs (hvac_equipment field, integration in step_physics_5r1c)
    - src/sim/hvac/mod.rs (AnyEquipment export, calculate_free_cooling_capacity export)

key-decisions:
  - "AnyEquipment enum wrapper: Used enum instead of dyn trait to enable Clone compatibility with ThermalModel for batch parallelism"
  - "Thermal demand vs electrical power: Returned thermal demand from equipment.calculate_power() to preserve thermal network physics, with electrical power calculated internally for energy tracking"
  - "Backward compatibility: Preserved IdealHVACController fallback when hvac_equipment is None to maintain ASHRAE 140 test compatibility"

patterns-established:
  - "Pattern: Enum wrapper for Clone trait objects - When trait requires Clone, wrap all concrete types in enum and implement trait for enum"
  - "Pattern: Type alias for import conflicts - Use 'as' keyword to rename conflicting enum imports (HVACMode as EquipmentHVACMode)"
  - "Pattern: Separation of thermal and electrical power - HVAC equipment consumes electrical power to provide thermal power; track both for accurate simulation"

requirements-completed: [HVAC-01, HVAC-02, HVAC-03, HVAC-04, HVAC-05, HVAC-07, HVAC-08, HVAC-09]

# Metrics
duration: 12min
completed: 2026-03-13T21:38:00Z
---

# Phase 15: Plan 6 - HVAC Equipment Integration Summary

**Variable capacity HVAC equipment integration with predictive control, cycling losses, and economizer free cooling in thermal physics loop**

## Performance

- **Duration:** 12 min
- **Started:** 2026-03-13T21:26:05Z
- **Completed:** 2026-03-13T21:38:00Z
- **Tasks:** 4
- **Files modified:** 3

## Accomplishments

- **Added hvac_equipment field to ThermalModel** - Enables dynamic equipment attachment with Clone compatibility for batch parallelism
- **Created AnyEquipment enum wrapper** - Solves dyn trait Clone limitation by wrapping all equipment types (Chiller, Boiler, VAVTerminal, CAVSystem, HeatPump) in enum
- **Integrated predictive control** - Calls predictive_controller.calculate_modulation() to determine HVAC mode and capacity modulation based on thermal inertia
- **Integrated equipment power calculation** - Replaces IdealHVACController with variable capacity equipment when attached, including efficiency curves, cycling losses, and economizer free cooling
- **Updated previous_temperatures tracking** - Stores temperatures at end of each timestep for accurate dT/dt calculation in predictive control

## Task Commits

Each task was committed atomically:

1. **Task 1: Add hvac_equipment field to ThermalModel** - `0d84a7d` (feat)
2. **Task 2: Integrate predictive control into solve_timesteps** - `9caf19b` (feat)
3. **Task 3: Integrate equipment power calculation into solve_timesteps** - `af16865` (feat)
4. **Task 4: Update previous_temperatures each timestep** - `6929532` (feat)

**Plan metadata:** (none - each task committed separately)

## Files Created/Modified

- `src/sim/hvac/equipment.rs` - Added AnyEquipment enum wrapper implementing VariableCapacityEquipment trait for all equipment types
- `src/sim/engine.rs` - Added hvac_equipment field, integrated predictive control, equipment calculation, economizer, cycling losses, and previous_temperatures update in step_physics_5r1c
- `src/sim/hvac/mod.rs` - Exported AnyEquipment and calculate_free_cooling_capacity for use in engine.rs

## Decisions Made

- **AnyEquipment enum wrapper pattern**: VariableCapacityEquipment trait requires Clone for ThermalModel compatibility with rayon batch parallelism, but dyn VariableCapacityEquipment is not object-safe. Solution: Wrap all concrete equipment types in enum and implement trait for enum.
- **Thermal demand preservation**: Equipment.calculate_power() returns electrical power consumption, but thermal network requires thermal power input. Decision: Calculate electrical power internally for energy tracking, but return thermal demand to physics loop to preserve network dynamics.
- **Backward compatibility fallback**: When hvac_equipment is None, fall back to IdealHVACController calculation. This preserves existing ASHRAE 140 test compatibility while enabling new equipment-based simulation when equipment is attached.
- **Free cooling integration**: Check economizer activation and calculate free_cooling_capacity when in cooling mode, reducing required thermal load from equipment.

## Deviations from Plan

None - plan executed exactly as written. All four tasks completed without deviations from specified tasks.

## Issues Encountered

- **Type conflict: HVACMode enum duplication** - engine.rs and equipment.rs both define HVACMode enum. Fixed by importing equipment::HVACMode as EquipmentHVACMode to avoid conflict.
- **Trait object Clone limitation** - Cannot use Box<dyn VariableCapacityEquipment> because trait requires Clone, but dyn Clone is not object-safe. Fixed with AnyEquipment enum wrapper pattern.
- **calculate_free_cooling_capacity signature mismatch** - Function takes 6 arguments (mode, outdoor_temp, outdoor_enthalpy, zone_temp, zone_enthalpy, cooling_setpoint), not 3. Fixed by calling with correct arguments and None for enthalpy parameters (not available until Phase 16).
- **calculate_cycling_loss return type** - Returns tuple (startup_penalty, efficiency_multiplier), not single value. Fixed by destructuring tuple and using efficiency_multiplier.
- **Generic type conversion** - self.temperatures is generic type T, not VectorField. Fixed by converting via as_ref().to_vec() and VectorField::new().

## Verification

**Build verification:**
- `cargo build --release` - Compiled successfully with only pre-existing warnings
- No new compilation errors introduced

**Test verification:**
- `cargo test --package fluxion --lib sim::hvac` - All 36 HVAC tests passed
- `cargo test --package fluxion --lib ashrae` - 42 passed, 1 pre-existing failure unrelated to changes (test_validator_multireference_enrichment)

**Integration verification:**
- Confirmed predictive_controller.calculate_modulation() called at line 2620
- Confirmed equipment.calculate_power() called at line 2688
- Confirmed cycling_tracker.calculate_cycling_loss() called at line 2690
- Confirmed is_economizer_active() called at line 2642
- Confirmed equipment.update_state() called at line 2684
- Confirmed previous_temperatures updated at line 2906
- Confirmed hvac_equipment field exists at line 363
- Confirmed hvac_equipment cloned in ThermalModel::clone() at line 638

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

**HVAC equipment integration complete and functional.**

**Ready for:**
- Plan 15-07 (HVAC integration tests) - Equipment integration provides test targets for end-to-end validation
- Phase 16 (Psychrometrics) - Economizer enthalpy mode can be implemented once psychrometrics module available
- Future diagnostic cases (800-810) - Realistic HVAC simulation enables equipment validation against ASHRAE reference cases

**Known limitations:**
- Ventilation airflow hardcoded to 10000 m³/s in calculate_free_cooling_capacity - TODO comment added for building spec integration
- Electrical power energy tracking not yet separate from thermal power - Future enhancement needed for accurate energy cost calculation
- Enthalpy mode disabled until Phase 16 psychrometrics available

---
*Phase: 15-hvac-equipment-modeling*
*Completed: 2026-03-13*

## Self-Check: PASSED

**Created files:**
- ✓ src/sim/hvac/equipment.rs (AnyEquipment enum wrapper)
- ✓ src/sim/engine.rs (hvac_equipment field and integration)
- ✓ src/sim/hvac/mod.rs (export updates)
- ✓ .planning/phases/15-hvac-equipment-modeling/15-06-SUMMARY.md

**Commits:**
- ✓ 0d84a7d (Task 1: Add hvac_equipment field)
- ✓ 9caf19b (Task 2: Integrate predictive control)
- ✓ af16865 (Task 3: Integrate equipment power calculation)
- ✓ 6929532 (Task 4: Update previous_temperatures)
