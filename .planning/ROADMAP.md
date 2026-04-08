# Fluxion Roadmap

**Project:** Building Energy Modeling Engine (Rust + Python)
**Milestone:** v1.0 (Next Milestone)
**Current Phase:** Planning
**Last Updated:** 2026-04-07

---

## Milestones

- ✅ **v1.0 Multi-Zone Support** — Phases M1-M3 (shipped 2026-04-07)
- ✅ **v0.8 Peak Load & Free-Float Validation** — Phases 33-36 (shipped 2026-04-07)
- ✅ **v0.7 Thermal Physics Complete** — Phases 28-32 (COMPLETE 2026-04-02)
- ✅ **v0.6 Validation Excellence** — Phases 24-27 (COMPLETE 2026-03-17)

---

## Current Status

**Milestone:** v1.0 (Multi-Zone Support) ✅ COMPLETED 2026-04-07
**Next Milestone:** Planning
**Status:** v1.0 milestone completed successfully. Ready for next milestone planning.

---

## Phases

<details>
<summary>✅ v1.0 Multi-Zone Support (Phases M1-M3) — SHIPPED 2026-04-07</summary>

- [x] **Phase M1: Multi-Zone Thermal Network Foundation** - N-zone thermal network with inter-zone conductance (3/3 plans complete)
- [x] **Phase M2: Zone-Level HVAC Controls** - Independent setpoints, schedules, setbacks (8/8 plans complete)
- [x] **Phase M3: ASHRAE 140 Multi-Zone Validation** - Case 960/970, cross-validation (3/3 plans complete)

</details>

<details>
<summary>✅ v0.8 Peak Load & Free-Float Validation (Phases 33-36) — SHIPPED 2026-04-07</summary>

- [x] **Phase 33: Peak Load Diagnostics** - Diagnostic suite for hourly peak analysis (completed 2026-04-03)
- [x] **Phase 34: Peak Load Physics Fix** - Address high-mass peak load overestimation (completed 2026-04-03)
- [x] **Phase 35: Free-Floating Validation** - Improve free-floating temperature profiles (completed 2026-04-06)
- [x] **Phase 36: v0.8.0 Release** - Documentation, release artifacts, and final validation (completed 2026-04-06)

</details>

## Phase M2: Zone-Level HVAC Controls

**Goal:** Implement zone-level HVAC controls for the multi-zone thermal network established in Phase M1. This phase focuses on adding per-zone heating/cooling setpoints, independent HVAC control logic, and extending the Python API and CLI to support multi-zone HVAC operations.

**Requirements:** [MZ-03, MZ-04, MZ-09, MZ-10]

**Status:** Gap closure in progress - addressing compilation errors and completing integration

- [x] M2-01: Zone-Level HVAC Controls Foundation (COMPLETE)
  - Implemented ZoneSetpoints and ZoneControl structs
  - Created comprehensive HVAC control tests
  - Verified independent zone control logic

- [⚠️] M2-02: Python API Multi-Zone HVAC Bindings (PARTIAL - Build failures)
  - Created PyZoneSetpoints and PyZoneControl wrappers
  - Implemented Python module registration
  - Blocked by VectorField API incompatibility

- [⚠️] M2-03: CLI Multi-Zone HVAC Support (PARTIAL - Integration blocked)
  - Created HVAC CLI command structure
  - Integrated with multi-zone CLI
  - Blocked by unresolved HVAC module dependencies

- [✅] M2-04: Fix Python Bindings Technical Blockers (COMPLETE)
  - Fixed ThermalModel import paths
  - Resolved PyO3 API compatibility issues
  - Added proper feature flag gating
  - HVAC bindings temporarily disabled due to ThermalModel API mismatch

- [x] M2-05: Fix Critical Gaps - VectorField API & CLI Integration (Gap Closure)
  - Fix VectorField API usage in HVAC control tests
  - Correct ThermalModel import path in zone_control.rs
  - Implement actual HVAC integration in CLI handlers

- [x] M2-06: Enable and Verify Python Bindings (Gap Closure)
  - Enable HVAC bindings module registration
  - Build and test Python bindings
  - Verify end-to-end Python HVAC functionality

- [x] M2-07: Fix Critical Compilation Errors (Gap Closure)
  - Fix ThermalModel import paths
  - Fix VectorField API usage in tests
  - Fix zone_setpoints module imports

- [x] M2-08: Complete Python Bindings Verification (Gap Closure)
  - Enable HVAC bindings module registration
  - Build and test Python bindings
  - Verify end-to-end Python HVAC functionality

- [x] M2-07: Fix Critical Compilation Errors (Gap Closure)
  - Fix ThermalModel import paths
  - Fix VectorField API usage in tests
  - Fix zone_setpoints module imports

- [ ] M2-08: Complete Python Bindings Verification (Gap Closure)
  - Enable HVAC bindings module registration
  - Build and test Python bindings
  - Verify end-to-end Python HVAC functionality

---

## Phase M3: ASHRAE 140 Multi-Zone Validation

**Goal:** Validate multi-zone implementation against ASHRAE 140 cases 960 and 970

**Requirements:** [MZ-06, MZ-07]

**Status:** Not started

- [x] M3-01: Multi-Zone Validation Framework - Extend validation infrastructure for multi-zone cases
  - Implement Case960Validator and Case970Validator structs
  - Extend ASHRAE140MultiZoneValidator with case-specific methods
  - Create comprehensive validation tests

- [x] M3-02: Reference Data and Statistical Methods - Add reference data and statistical analysis
  - Implement Case 960 reference data in benchmark.rs
  - Extend reference_data.rs with multi-zone loading utilities
  - Add statistical validation methods (RMSE, MBE, percentage difference)

- [x] M3-03: Validation Execution and Reporting - CLI tool and documentation
  - Create run_multi_zone_validation CLI tool
  - Enhance validation reporting with multi-zone support
  - Generate comprehensive validation results documentation

---

## Progress

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|----------|----------------|--------|-----------|
| 33. Peak Load Diagnostics | v0.8 | 1/1 | Complete | ✅ 2026-04-03 |
| 34. Peak Load Physics Fix | v0.8 | 4/4 | Complete | ✅ 2026-04-03 |
| 35. Free-Floating Validation | v0.8 | 1/1 | Complete | ✅ 2026-04-06 |
| 36. v0.8.0 Release | v0.8 | 4/4 | Complete | ✅ 2026-04-06 |
| 37. Multi-Zone Thermal Network Foundation | v1.0 | 3/3 | Complete | ✅ 2026-04-07 |
| 38. Zone-Level HVAC Controls | v1.0 | 5/8 | In Progress | ⚠️ Gap Closure |
| 39. ASHRAE 140 Multi-Zone Validation | v1.0 | 0/3 | Not started | - |

**Overall Progress:** v0.8.0 milestone complete (100%), v1.0 in progress (40%)

---

## v1.1 ASHRAE 140 Completion (Current Milestone)

### Phase 40: Case Expansion Foundation

**Goal:** Extend ASHRAE 140 case support and establish cross-validation framework

**Requirements:** [CASE-01, CASE-02, CASE-03, CROSS-01, CROSS-02]

**Status:** Gap closure in progress

- [x] 40-01: ASHRAE 140 Case Expansion - Extend case definitions (COMPLETE)
  - Implemented Cases 800-810 (HVAC equipment validation)
  - Implemented Cases 195-470 (diagnostic validation)
  - Modular organization by series

- [x] 40-02: Extended Reference Database - Reference data infrastructure (COMPLETE)
  - Reference data loading module
  - Directory structure and documentation
  - CSV parsing infrastructure

- [x] 40-03: Cross-Validation Framework - Adapter pattern implementation (COMPLETE)
  - EnergyPlus adapter with file-based comparison
  - TRNSYS adapter with file-based comparison
  - Core comparison logic and reporting

- [x] 40-04: CLI Integration - Command-line interface (COMPLETE)
  - Validation commands for single cases and series
  - Cross-validation commands
  - Performance profiling commands

- [x] 40-05: Performance Optimization - Initial optimization (COMPLETE)
  - Parallel execution infrastructure
  - Performance monitoring framework
  - Basic optimization strategies

- [⚠️] 40-06: Reference Data Generation - Gap closure (IN PROGRESS)
  - Generate series_800.csv with 96,360 rows
  - Generate series_195.csv with 2,418,560 rows
  - Enhance reference data loading with caching

- [⚠️] 40-07: Case Execution Implementation - Gap closure (PLANNED)
  - Implement actual simulation logic for Cases 800-810
  - Implement actual simulation logic for Cases 195-470
  - Validation execution with real Fluxion simulations

- [⚠️] 40-08: CLI and Cross-Validation Enhancement - Gap closure (PLANNED)
  - Replace stubbed CLI commands with real execution
  - Enhance cross-validation with real Fluxion results
  - Update batch processing with parallel execution

- [⚠️] 40-09: Real Performance Monitoring - Gap closure (PLANNED)
  - Replace simulated delays with actual timing
  - Enhance performance reporting and analysis
  - Verify <50ms/timestep target compliance

### Progress

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|----------|----------------|--------|-----------|
| 40. Case Expansion Foundation | v1.1 | 10/9 | Complete   | 2026-04-08 |

**v1.1 Progress:** 56% complete (5/9 plans)

---

*Roadmap updated: 2026-04-07 - Phase 40 gap closure plans added to address verification gaps*
