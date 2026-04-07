# Fluxion Roadmap

**Project:** Building Energy Modeling Engine (Rust + Python)
**Milestone:** v1.0 (Next Milestone)
**Current Phase:** Planning
**Last Updated:** 2026-04-07

---

## Milestones

- ✅ **v0.8 Peak Load & Free-Float Validation** — Phases 33-36 (shipped 2026-04-07)
- ✅ **v0.7 Thermal Physics Complete** — Phases 28-32 (COMPLETE 2026-04-02)
- ✅ **v0.6 Validation Excellence** — Phases 24-27 (COMPLETE 2026-03-17)

---

## Current Status

**Milestone:** v1.0 (Multi-Zone Support)
**Phase:** M2-zone-hvac-controls (Execution)
**Status:** v0.8.0 milestone completed. v1.0 milestone in progress with technical blockers to resolve.

---

## Phases

<details>
<summary>✅ v0.8 Peak Load & Free-Float Validation (Phases 33-36) — SHIPPED 2026-04-07</summary>

- [x] **Phase 33: Peak Load Diagnostics** - Diagnostic suite for hourly peak analysis (completed 2026-04-03)
- [x] **Phase 34: Peak Load Physics Fix** - Address high-mass peak load overestimation (completed 2026-04-03)
- [x] **Phase 35: Free-Floating Validation** - Improve free-floating temperature profiles (completed 2026-04-06)
- [x] **Phase 36: v0.8.0 Release** - Documentation, release artifacts, and final validation (completed 2026-04-06)

</details>

### Current Milestone: v1.0 (In Progress)

- [x] Phase M1: Multi-Zone Thermal Network Foundation (3/3 plans complete)
- [ ] Phase M2: Zone-Level HVAC Controls (3/6 plans, gap closure added)
- [ ] Phase M3: ASHRAE 140 Multi-Zone Validation (TBD plans)

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
  
- [ ] M2-07: Fix Critical Compilation Errors (Gap Closure)
  - Fix ThermalModel import paths
  - Fix VectorField API usage in tests
  - Fix zone_setpoints module imports
  
- [ ] M2-08: Complete Python Bindings Verification (Gap Closure)
  - Enable HVAC bindings module registration
  - Build and test Python bindings
  - Verify end-to-end Python HVAC functionality

- [ ] M2-07: Fix Critical Compilation Errors (Gap Closure)
  - Fix ThermalModel import paths
  - Fix VectorField API usage in tests
  - Fix zone_setpoints module imports
  
- [ ] M2-08: Complete Python Bindings Verification (Gap Closure)
  - Enable HVAC bindings module registration
  - Build and test Python bindings
  - Verify end-to-end Python HVAC functionality

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
| 39. ASHRAE 140 Multi-Zone Validation | v1.0 | 0/0 | Not started | - |

**Overall Progress:** v0.8.0 milestone complete (100%), v1.0 in progress (40%)

---

*Roadmap updated: 2026-04-07 - M2 gap closure plan added to address technical blockers*
