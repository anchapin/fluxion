---
phase: 15
plan: 01
subsystem: HVAC Equipment Modeling
tags: [hvac, variable-capacity, trait, equipment-models]
dependency_graph:
  requires: []
  provides: [HVAC-01, HVAC-02, HVAC-03]
  affects: [15-02, 15-03]
tech_stack:
  added:
    - VariableCapacityEquipment trait
    - HVACMode enum
    - Chiller and Boiler equipment models
    - VAV, CAV, and HeatPump trait implementations
  patterns:
    - Unified trait interface for all equipment types
    - Part-load ratio (PLR) tracking for cycling loss calculations
    - Temperature-dependent capacity and efficiency
    - Send + Sync for thread-safe BatchOracle parallel evaluation
key_files:
  created:
    - path: src/sim/hvac/equipment.rs
      description: VariableCapacityEquipment trait and equipment implementations
      lines: 528
      purpose: Unified interface for all variable-capacity HVAC equipment
    - path: tests/hvac_equipment.rs
      description: Integration tests for VariableCapacityEquipment trait
      lines: 257
      purpose: Validate trait behavior for all equipment types
  modified:
    - path: src/sim/hvac/mod.rs
      description: HVAC module with equipment exports
      changes: Added equipment module, updated exports, added current_plr field to VAV, CAV, and HeatPump
      lines: 295
decisions:
  - "Use reheat_capacity (W) instead of airflow (m³/s) for VAV rated capacity to maintain consistent units with thermal load calculations"
  - "Heat pump capacity defaults to heating mode when Off, allowing capacity calculation without mode update"
  - "PLR clamping implemented at [0.0, 1.0] to prevent invalid ratios during overload or negative load scenarios"
  - "VAV efficiency: COP 0.8 (fan + electric reheat), COP 3.0 (fan + cooling coil)"
  - "CAV efficiency: COP 0.85 (fan + heating), COP 3.2 (fan + cooling), constant fan power included"
  - "Heat pump efficiency uses existing linear degradation methods (will be replaced with polynomial curves in Plan 15-03)"
metrics:
  duration: 507
  completed_date: 2026-03-13T20:48:17Z
  tasks_completed: 6
  files_created: 2
  files_modified: 1
  lines_added: 785
  lines_removed: 1
  tests_added: 9
  tests_passed: 9
  tests_failed: 0
---

# Phase 15 Plan 01: Variable Capacity Equipment Trait Summary

Create VariableCapacityEquipment trait and enhance existing VAV, CAV, and HeatPump equipment with trait implementation, enabling unified variable capacity control across all equipment types.

## One-Liner

VariableCapacityEquipment trait provides unified interface for HVAC equipment with continuous modulation, part-load ratio tracking, and temperature-dependent efficiency for VAV, CAV, HeatPump, Chiller, and Boiler equipment types.

## Implementation Summary

### Core Deliverables

1. **VariableCapacityEquipment Trait** (`src/sim/hvac/equipment.rs`)
   - 7 core methods for unified equipment interface
   - Send + Sync bounds for thread-safe BatchOracle parallel evaluation
   - Clone bound for equipment copying in parallel simulations
   - Methods:
     - `calculate_capacity(plr, outdoor_temp)`: Actual capacity (W) with temperature degradation
     - `calculate_efficiency(plr, outdoor_temp, mode)`: COP/EER at current conditions
     - `calculate_power(load, outdoor_temp, mode)`: Electrical power consumption (W)
     - `rated_capacity()`: Maximum capacity at design conditions (W)
     - `rated_efficiency(mode)`: Rated COP/EER at design conditions
     - `current_plr()`: Current part-load ratio (0.0 to 1.0)
     - `update_state(current_load, outdoor_temp, mode)`: Update PLR and mode

2. **Equipment Implementations**
   - **VAVTerminal**: Airflow-based capacity (0.5 m³/s max), reheat coil for heating (COP 0.8), cooling coil (COP 3.0)
   - **CAVSystem**: Constant fan power (500 W/m³/s), thermal output modulation, heating (COP 0.85), cooling (COP 3.2)
   - **HeatPump**: Temperature-sensitive capacity and efficiency, heating capacity 12kW, cooling capacity 10kW, COP 3.5 heating, EER 3.0 cooling
   - **Chiller** (placeholder): Cooling-only equipment, 100kW capacity, COP 4.5, temperature limits 5-45°C
   - **Boiler** (placeholder): Heating-only equipment, 100kW capacity, 85% efficiency, minimum -20°C

3. **Struct Enhancements**
   - Added `current_plr: f64` field to VAVTerminal, CAVSystem, and HeatPump
   - Initialized to 0.0 in constructors
   - Clamped to [0.0, 1.0] in `update_state()` method

4. **Module Exports** (`src/sim/hvac/mod.rs`)
   - Added `pub mod equipment;`
   - Re-exported: `VariableCapacityEquipment`, `HVACMode`, `Chiller`, `Boiler`
   - VAVTerminal, CAVSystem, HeatPump already exported (defined in mod.rs)

5. **Comprehensive Testing** (`tests/hvac_equipment.rs`)
   - 9 integration tests validating trait behavior
   - Test coverage:
     - Chiller capacity, efficiency, power, PLR tracking, temperature limits
     - Boiler capacity, efficiency, power, PLR tracking, temperature sensitivity
     - VAV capacity, efficiency, power, PLR tracking
     - CAV capacity, efficiency, power, PLR tracking (includes fan power)
     - HeatPump capacity, efficiency, power, PLR tracking, mode synchronization
     - PLR clamping to [0.0, 1.0] for overload and negative load scenarios
   - All tests passing (9/9)

## Deviations from Plan

None - plan executed exactly as written.

## Design Decisions

### Capacity Units
- **VAV**: Changed from airflow (m³/s) to reheat capacity (W) to maintain consistent units with thermal load calculations
- **Rationale**: VariableCapacityEquipment trait methods expect thermal load in Watts, not airflow rate

### Heat Pump Mode Default
- **Decision**: Use heating mode as default when calculating capacity in Off state
- **Rationale**: Allows capacity calculation without requiring mode update first, simplifies API usage

### PLR Tracking
- **Implementation**: Store PLR as struct field (`current_plr`), not as method parameter
- **Rationale**: Enables cycling loss calculations that need PLR history from previous timesteps

### Efficiency Models
- **Current**: Linear degradation with temperature (existing HeatPump methods)
- **Future**: Polynomial curves with PLR + temperature inputs (Plan 15-03)
- **Rationale**: Placeholder implementation allows immediate trait usage while preparing for advanced models

## Technical Details

### Temperature Sensitivity
- **Heat Pump**: Capacity degrades ~1% per degree from design temperature, minimum 30% of rated
- **Chiller**: Capacity degrades 0.5% per degree, minimum 30% at extreme temps (<5°C or >45°C)
- **Boiler**: Capacity degrades 0.1% per degree, minimum 50% at extreme cold (<-20°C)
- **VAV/CAV**: Minimal temperature sensitivity (primary variation is PLR, not temperature)

### Efficiency Values
- **VAV Heating**: COP 0.8 (fan + electric reheat coil)
- **VAV Cooling**: COP 3.0 (fan + cooling coil)
- **CAV Heating**: COP 0.85 (fan + heating coil)
- **CAV Cooling**: COP 3.2 (fan + cooling coil)
- **Heat Pump Heating**: COP 3.5 at -5°C, degrades at colder temps
- **Heat Pump Cooling**: EER 3.0 at 35°C, degrades at hotter temps
- **Chiller**: COP 4.5 at 35°C, degrades at extreme temps
- **Boiler**: 85% efficiency (AFUE), minimal degradation with temperature

### Thread Safety
- All trait implementations include `Send + Sync` bounds
- Enables safe parallel evaluation in BatchOracle with rayon
- Equipment structs are `Clone`, allowing lightweight copies for each parallel configuration

## Validation Results

### Compilation
- Library compiles without errors
- 20 warnings (mostly unused variables - pre-existing, not introduced by this plan)

### Test Results
- All 9 integration tests passing
- Tests validate:
  - Capacity calculations at design and off-design conditions
  - Efficiency degradation with temperature
  - Power calculation (including constant fan power for CAV)
  - PLR tracking and clamping
  - Mode synchronization for HeatPump
  - Temperature limits for Chiller and Boiler

### Integration
- HVAC module exports all required types
- Trait implementations accessible from integration tests
- No breaking changes to existing VAV, CAV, or HeatPump APIs

## Next Steps

### Plan 15-02: Chiller and Boiler Implementation
- Full implementation with polynomial efficiency curves
- AHRI reference data integration
- Temperature-dependent performance validation

### Plan 15-03: Efficiency Curves
- Replace linear degradation with polynomial curves
- PLR + temperature as curve inputs
- AHRI coefficient library integration

### Plan 15-04: Cycling Losses
- Implement startup penalties (on/off transitions)
- Minimum runtime constraints
- PLR degradation at low loads (e.g., +20% at 30% PLR)

## Files Modified

### Created
- `src/sim/hvac/equipment.rs` (528 lines)
- `tests/hvac_equipment.rs` (257 lines)

### Modified
- `src/sim/hvac/mod.rs` (295 lines) - added equipment module, exports, current_plr fields

### Commit
- `a0f8042` - feat(15-01): implement VariableCapacityEquipment trait for VAV, CAV, and HeatPump

## Self-Check: PASSED

- [x] VariableCapacityEquipment trait defined with 7 core methods
- [x] VAVTerminal, CAVSystem, HeatPump implement trait
- [x] Equipment models support continuous 0-100% modulation
- [x] Unit tests validate trait behavior (9/9 passing)
- [x] HVAC module exports equipment types
- [x] Library compiles without errors
- [x] Commit created with proper format
- [x] SUMMARY.md created with substantive content
