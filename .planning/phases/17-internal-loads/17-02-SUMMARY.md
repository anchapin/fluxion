---
phase: 17-internal-loads
plan: 02
subsystem: equipment-load-modeling
tags: [equipment, trait, heat-gains, mass-coupling, validation]
dependency_graph:
  requires: []
  provides: [Equipment, ComputerEquipment, ServerRack, GenericEquipment]
  affects: [ThermalModel, solve_timesteps]
tech_stack:
  added: [serde:Serialize, serde:Deserialize]
  patterns: [trait-abstraction, mass-coupled-radiative, convective-radiative-split]
key_files:
  created:
    - path: src/sim/equipment.rs
      size: 253 lines
      description: Equipment trait and implementations
  modified:
    - path: src/sim/mod.rs
      changes: Added equipment module export
decisions:
  - id: EQ-01
    title: Trait-based equipment abstraction
    context: Followed existing codebase patterns (ContinuousTensor, PsychrometricCalculations)
    rationale: Provides consistent API across equipment types, supports mixed equipment lists via Vec<Box<dyn Equipment>>
    impact: Enables flexible equipment modeling with thermal characteristics

  - id: EQ-02
    title: Mass-coupled radiative heat distribution
    context: Equipment radiative heat splits between air and thermal mass based on mass_coupling_factor
    rationale: More accurate 5R1C physics accounting for equipment placement and thermal characteristics
    impact: HVAC demand calculations will account for equipment-specific coupling factors

  - id: EQ-03
    title: Default thermal characteristics per equipment type
    context: Research indicated different equipment types have different radiative/convective splits and mass coupling
    rationale: ComputerEquipment (0.3/0.7/0.2), ServerRack (0.5/0.5/0.8), GenericEquipment (0.5/0.5/0.5)
    impact: Realistic heat gain modeling for office and data center scenarios

metrics:
  duration: 298s (4m 58s)
  completed_date: 2026-03-14
  tasks_completed: 3
  files_created: 1
  files_modified: 1
  lines_added: 287
  tests_added: 4
  tests_passing: 13
---

# Phase 17 Plan 02: Equipment trait and implementations Summary

Equipment load modeling with trait-based abstraction and mass-coupled radiative heat distribution for realistic office equipment heat gain modeling.

## Implementation Overview

Created `src/sim/equipment.rs` module with:
- **Equipment trait**: Provides consistent API for power and heat gain calculations across all equipment types
- **ComputerEquipment**: Desktop computers, laptops, monitors (0.3 radiative, 0.7 convective, 0.2 mass coupling)
- **ServerRack**: Data center servers (0.5 radiative, 0.5 convective, 0.8 mass coupling)
- **GenericEquipment**: Generic equipment with configurable thermal characteristics
- **Validation**: `validate()` method ensures fractions sum to 1.0 and coupling factor in [0, 1]

## Key Features

### Trait-Based Abstraction
Equipment trait provides consistent API:
- `id(&self) -> &str`: Equipment identifier
- `power_at_hour(&self, hour_of_year: usize) -> f64`: Power calculation with schedule
- `convective_gains(&self, hour_of_year: usize) -> f64`: Convective heat to air
- `radiative_gains(&self, hour_of_year: usize) -> f64`: Radiative heat to thermal mass
- `mass_coupling_factor(&self) -> f64`: Fraction of radiative heat absorbed by mass

### Mass-Coupled Radiative Heat Distribution
Equipment radiative heat splits between air and thermal mass:
- `radiative_to_mass = radiative_gains * mass_coupling_factor`
- `radiative_to_air = radiative_gains * (1.0 - mass_coupling_factor)`

This provides more accurate 5R1C physics by accounting for equipment placement and thermal characteristics.

### Time-Varying Schedules
All equipment types use `DailySchedule` for time-varying power:
- ComputerEquipment: Off by default (schedule can be configured)
- ServerRack: 24/7 operation by default (constant 1.0 schedule)
- GenericEquipment: Off by default (schedule can be configured)

### Convective/Radiative Heat Split
Each equipment type has configurable fractions:
- ComputerEquipment: 0.3 radiative, 0.7 convective
- ServerRack: 0.5 radiative, 0.5 convective
- GenericEquipment: 0.5 radiative, 0.5 convective

Fractions must sum to 1.0 (validated by `validate()` method).

## Test Coverage

Added 4 comprehensive tests (13 total equipment tests):
1. **test_equipment_trait**: Verifies trait implementation for all equipment types
2. **test_equipment_power_at_hour**: Verifies power and heat gain calculations
   - Tests: `power = rated_power * count * schedule`
   - Tests: `convective_gains = power * convective_fraction`
   - Tests: `radiative_gains = power * radiative_fraction`
   - Tests: `total_power = convective + radiative`
3. **test_mass_coupled_radiative**: Verifies mass-coupled radiative heat distribution
   - Tests: `radiative_to_mass = radiative * coupling_factor`
   - Tests: `radiative_to_air = radiative * (1.0 - coupling_factor)`
   - Tests: `total_radiative = to_mass + to_air`
4. **test_server_rack_24_7**: Verifies 24/7 constant schedule operation
   - Tests: ServerRack uses constant 1.0 schedule by default
   - Tests: Power output is constant across day and night hours

All tests passing: ✅

## Deviations from Plan

None - plan executed exactly as written.

## Integration Points

### Schedule Module Integration
Equipment uses `DailySchedule` from `src/sim/schedule.rs`:
- Time-varying power via `schedule.value(hour_of_year % 24)`
- Supports both daily and weekly schedules (future-ready)

### ThermalModel Integration (Future)
Equipment will integrate with `ThermalModel::solve_timesteps()` in future plans:
- Equipment passed as `Option<&[Box<dyn Equipment>]>` to solve_timesteps
- Equipment radiative heat splits between air and mass based on `mass_coupling_factor()`
- Supports mixed equipment lists via `Vec<Box<dyn Equipment>>`

## Code Quality

- Follows existing codebase patterns (trait-based abstraction)
- Consistent with `LightingSchedule` and `OccupancyProfile` patterns
- Comprehensive test coverage (13 tests passing)
- Clean commit history with conventional commit messages
- All pre-commit hooks passing (fmt, check, batch-oracle-pattern, rust-doc-check, audit)

## Future Work

Next steps in Phase 17:
- **Plan 17-03**: Integrate equipment loads with ThermalModel
- **Plan 17-04**: Add weekly schedule support for equipment
- **Phase 17 completion**: Building profiles with default equipment configurations

## Performance Considerations

- Zero-cost abstraction: Trait dispatch optimized by compiler
- No allocations in power calculation inner loop
- Compatible with BatchOracle pattern (single-level parallelism)
- Equipment lists passed as references to enable cloning of ThermalModel

## Self-Check: PASSED

- [x] All tasks executed (3/3)
- [x] Each task committed individually with proper format
- [x] All tests passing (13/13)
- [x] Equipment trait defined with required methods
- [x] Three equipment types implement Equipment trait
- [x] Power calculation uses schedule values correctly
- [x] Heat gains split between convective and radiative components
- [x] Mass coupling factor determines radiative heat distribution
- [x] Validation ensures fractions sum to 1.0 and coupling factor in [0, 1]
- [x] Module added to src/sim/mod.rs
- [x] No deviations from plan

## References

- Plan: `.planning/phases/17-internal-loads/17-02-PLAN.md`
- Context: `.planning/phases/17-internal-loads/17-CONTEXT.md`
- Research: `.planning/phases/17-internal-loads/17-RESEARCH.md`
- Schedule module: `src/sim/schedule.rs`
- Lighting module: `src/sim/lighting.rs` (pattern reference)
- Occupancy module: `src/sim/occupancy.rs` (pattern reference)
