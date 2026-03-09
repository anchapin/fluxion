---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
current_phase: 3
current_plan: 03-08 - HVAC Sensitivity Calculation Investigation
status: HVAC sensitivity investigated, root cause identified (sensitivity = 0.002065 K/W too low, thermal mass time constant τ = 4.82 hours). Correction factor 4.0 implemented: cooling within reference (2.31 MWh), heating still above reference (4.33 MWh), peak cooling regression (1.39 kW). Single-factor approach insufficient - requires separate heating/cooling factors or free-floating temp fix.
last_updated: "2026-03-09T20:50:02.000Z"
progress:
  total_phases: 7
  completed_phases: 2
  total_plans: 21
  completed_plans: 20
  percent: 95
---

# Fluxion ASHRAE 140 Validation - Project State

**Last Updated:** 2026-03-09
**Current Phase:** 3
**Current Plan:** 03-08 - HVAC Sensitivity Calculation Investigation
**Status:** HVAC sensitivity investigated, root cause identified (sensitivity = 0.002065 K/W too low, thermal mass time constant τ = 4.82 hours). Correction factor 4.0 implemented: cooling within reference (2.31 MWh), heating still above reference (4.33 MWh), peak cooling regression (1.39 kW). Single-factor approach insufficient - requires separate heating/cooling factors or free-floating temp fix.
**Session:** Phase 3 Plan 08 completed
**Phase 2 Results:** Thermal mass dynamics validated with implicit integration. Temperature swing reduction (22.4%) and Case 900 annual heating (1.77 MWh) within ASHRAE 140 reference. Solar gain issues (cooling under-prediction) deferred to Phase 3.
**Phase 3 Results (Plans 07, 07b, 07c, 08):** Plan 07 investigated hvac_power_demand and solar distribution, completed but objective not achieved (annual heating 6.86 MWh, cooling 4.82 MWh). Plan 07b was not executed (Plan 07c directly continued investigation). Plan 07c investigated thermal mass dynamics, reverted solar_beam_to_mass_fraction to 0.7 (ASHRAE 140 spec), analyzed h_tr_em/h_tr_ms ratio (0.052 very low), identified root cause (thermal mass releases energy primarily to interior, HVAC works against mass). Tested coupling enhancement values (1.15x, 1.5x, 2.0x) - found heating-cooling trade-off, simple parameter tuning insufficient. Plan 08 investigated HVAC sensitivity calculation, implemented correction factor 4.0: cooling within reference (2.31 MWh), heating still above reference (4.33 MWh), peak cooling regression (1.39 kW). Single-factor approach insufficient - requires separate heating/cooling factors or free-floating temp fix.
**Progress:** [██████████] 95%

## Project Reference

**Core Value:** Every ASHRAE 140 test case must pass with energy consumption and peak load predictions within ASHRAE tolerance bands (±15% annual, ±10% monthly).

**Current Focus:** Achieving full ASHRAE Standard 140 validation compliance by fixing systematic errors in heating calculations, peak load predictions, and high-mass building cases.

**Validation Target:** Reduce 61% failure rate and 78.79% Mean Absolute Error to 100% pass rate with <15% MAE.

## Current Position

**Phase:** 3 - Solar Radiation & External Boundaries
**Plan:** 03-01 - Solar Radiation Research
**Status:** Phase 2 complete, starting Phase 3
**Progress:** [████████░░] 57.1%

**Phase Goal:** Fix solar gain calculations and external boundary conditions to address peak cooling load under-prediction and annual cooling energy discrepancies.

**Phase Requirements:** 4 requirements (SOLAR-01 through SOLAR-04)
**Phase Status:** 0/4 requirements complete

**Phase Success Criteria:**
1. All baseline Cases 600, 610, 620, 630, 640, 650 pass with both ±15% annual and ±10% monthly energy tolerances
2. Case 900 (high-mass) passes with ±15% annual and ±10% monthly energy tolerances
3. Free-floating cases (600FF, 650FF, 900FF) report min/max/avg temperatures within acceptable ranges
4. Peak heating and cooling loads match ASHRAE reference values within ±10% tolerance
5. Mean Absolute Error reduced from 78.79% to <15% across all baseline cases
6. Annual heating load over-prediction systematically corrected (no consistent bias)

## Performance Metrics

**Validation Status (Baseline):**
- Total validation metrics: 64
- Passed: 16 (25%)
- Warnings: 9 (14%)
- Failed: 39 (61%)
- Mean Absolute Error: 78.79%
- Max Deviation: 471.66%

**Validation Status (Post-Phase 1):**
- Total validation metrics: 64
- Passed: 19 (30%)
- Warnings: 10 (16%)
- Failed: 35 (54%)
- Mean Absolute Error: 49.21% (37.5% improvement)
- Max Deviation: 512.45%

**Performance Targets (Post-Phase 1):**
- Pass rate: >75% baseline cases passing
- Mean Absolute Error: <15% across baseline cases
- Max Deviation: <50% across all metrics
- Annual heating load bias: eliminated

**Achievement (Post-Phase 1):**
- MAE reduced from 78.79% to 49.21% (37.5% improvement)
- Pass rate improved from 25% to 30%
- Peak heating loads significantly improved (3.30 kW vs 4.81 kW baseline)
- Free-floating cases pass temperature range validation
- Denver TMY weather data confirmed for all baseline cases (BASE-04 complete)

**Gap Analysis (Remaining Issues):**
- Systematic heating over-prediction (37-87%) → Partially addressed in Phase 2 (thermal mass), remaining issues due to solar gains
- Peak cooling load under-prediction (0.60 vs 2.10-3.50 kW for Case 900) → Will be addressed in Phase 3 (Solar Radiation & External Boundaries)
- MAE target <15% not met → Requires improvements from Phases 2-3 (thermal mass complete, solar pending)
- FREE-02 and TEMP-01 completed in Phase 2 (thermal mass dynamics validated)

**Why Gaps Are Expected:**
Phase 1's scope was foundation fixes (conductances, HVAC load calculation) that were necessary but not sufficient for full ASHRAE 140 validation. The remaining gaps represent additional physics domains (thermal mass, solar radiation) that were always planned for later phases.

**Performance Targets (Final State - Phase 7):**
- Pass rate: 100% across all 18+ cases
- Mean Absolute Error: <5% across all cases
- Max Deviation: <15% across all cases
- Validation suite execution: <5 minutes

## Accumulated Context

### Key Decisions

1. **Physics-First Approach:** Phases 1-4 address physics accuracy (conductances, HVAC, mass, solar, multi-zone) before optimization (Phases 6-7) to prevent optimizing incorrect physics.

2. **Complexity Gradient:** Start with simple lightweight cases (Phase 1), add thermal mass complexity (Phase 2), then external boundaries (Phase 3), then multi-zone coupling (Phase 4). Each phase builds on the previous.

3. **Developer Experience:** Diagnostic tools (Phase 5) come after physics is correct but before optimization, ensuring tools help debug correct behavior.

4. **Performance Last:** GPU acceleration and neural surrogates (Phase 6) are deferred until validation is accurate. Optimizing incorrect physics wastes effort.

5. **HVAC Load Calculation Uses Ti_free:** HVAC mode determination and load calculation must use the free-floating temperature (Ti_free), not the current zone temperature (Ti). Ti_free represents what the building temperature would be without HVAC input, accounting for thermal mass buffering effects from previous hours. This fix addresses the systematic heating load over-prediction identified in Phase 1 research. (Plan 03, 2026-03-09)

6. **Implicit Integration for High Thermal Mass:** Use backward Euler integration for thermal capacitance > 500 J/K to address explicit Euler instability. The research-based threshold (dt < Cm/(h_tr_em + h_tr_ms)) is commonly violated for high-mass buildings with 1-hour timesteps, causing oscillatory or divergent solutions. (Plan 02, 2026-03-09)

7. **Temperature Swing Reduction More Robust Than Thermal Lag:** Temperature swing reduction is a more reliable metric for thermal mass validation than thermal lag. Thermal lag measurement is sensitive to peak detection and summer period selection, while temperature swing reduction clearly demonstrates thermal mass damping effects. (Plan 03, 2026-03-09)

8. **HVAC Energy Tracking via step_physics Return Value:** Use step_physics return value (kWh) with sign-based separation for heating/cooling energy tracking. The model doesn't provide separate heating/cooling energy tracking, so net energy with sign is used. (Plan 03, 2026-03-09)

9. **Phase 2 Complete - Thermal Mass Dynamics Validated:** Implicit integration with thermal capacitance > 500 J/K threshold successfully implemented. Temperature swing reduction (22.4% vs 19.6% expected) confirms thermal mass damping effect. Case 900 annual heating energy (1.77 MWh) within reference range. All free-floating tests (10/10) passing. Remaining failures due to solar gain issues planned for Phase 3. (Plan 04, 2026-03-09)

### Known Systematic Issues

1. **5R1C Conductance Parameterization:** Incorrect window U-value application to h_tr_em and h_tr_w, missing thermal bridge effects
2. **HVAC Load Calculation Errors:** Wrong temperature for load calculation (Ti vs Ti_free), incorrect load sign convention
3. **Thermal Mass Dynamics:** Wrong thermal mass capacitance value, incorrect time step integration method
4. **Solar Radiation & External Boundaries:** Incorrect beam/diffuse solar decomposition, wrong solar incidence angle, missing shading
5. **Inter-Zone Heat Transfer:** Multi-zone coupling errors affecting Case 960

### Pitfall Avoidance Strategy

- **Phase 1:** Addresses Pitfalls 1 (incorrect 5R1C conductances) and 2 (HVAC load calculation errors)
- **Phase 2:** Addresses Pitfall 3 (thermal mass dynamics mishandling)
- **Phase 3:** Addresses Pitfall 4 (solar radiation and external boundary errors)
- **Phase 4:** Addresses Pitfall 5 (inter-zone heat transfer errors)
- **Phases 5-7:** Focus on usability and performance, avoiding the critical physics pitfalls

### Architecture Decisions

- **Batch Oracle Pattern:** Use `rayon::par_iter()` only at population level, avoid nested parallelism
- **Parameter Vector Semantics:** Document gene-to-field mapping for external APIs
- **Simulation Timesteps:** 1 year = 8760 hours, represented as `VectorField` of hourly states
- **Pre-commit Hooks:** Enforce code quality (fmt, clippy, audit) and batch-oracle pattern compliance

### Constraints

- **ASHRAE 140 Tolerance Bands:** ±15% annual energy, ±10% monthly energy
- **ISO 13790 Compliance:** Must maintain 5R1C thermal network structure
- **Existing Architecture:** Preserve PyO3 bindings and BatchOracle/Model API pattern
- **Performance:** Maintain high-throughput evaluation capability (>1,000 configs/sec)
- **Brownfield Constraints:** Work within existing codebase structure without major refactoring

## Session Continuity

### Previous Session Notes

- **Session 1 (2026-03-08):** Initial project setup, requirements definition, and research synthesis
  - Identified 51 v1 requirements across baseline validation, thermal mass, solar, multi-zone, diagnostics, performance, and advanced analysis
  - Research revealed 5 critical pitfalls causing 61% failure rate and 78.79% MAE
  - Created 7-phase roadmap following physics-first approach

### Current Session Tasks

- [x] Read PROJECT.md, REQUIREMENTS.md, research/SUMMARY.md, config.json
- [x] Analyze requirements and derive phase structure
- [x] Validate 100% requirement coverage
- [x] Write ROADMAP.md with phases, requirements, and success criteria
- [x] Write STATE.md with project context and current position
- [x] Complete Phase 1: Foundation (4 plans)
- [x] Complete Phase 2 Plan 02: Thermal Mass Integration Implementation
- [x] Complete Phase 2 Plan 03: Thermal Mass Validation
- [x] Complete Phase 2 Plan 04: Documentation & State Update
- [x] Complete Phase 3 Plan 00: Test Infrastructure Creation
- [ ] Complete Phase 3 Plan 01: Solar Radiation Research

### Next Steps

1. Complete Phase 3 Plan 01: Solar Radiation Research (analyze solar gain calculation issues)
2. Fix solar gain calculations to address cooling load under-prediction
3. Update REQUIREMENTS.md traceability section with Phase 2 completion

### Blockers

None identified. Phase 2 complete with thermal mass dynamics validated (FREE-02, TEMP-01 complete). Solar gain issues planned for Phase 3.

### Phase 1 Results Summary

**Completed Plans:**
- Plan 01: Conductance Calculation Test Suite
- Plan 02: Window U-value Application Fixes
- Plan 03: HVAC Load Calculation
- Plan 04: Final Foundation Validation

**Key Improvements:**
- MAE reduced from 78.79% to 49.21% (37.5% improvement)
- Pass rate improved from 25% to 30%
- Peak heating loads significantly improved
- Free-floating cases pass validation
- Denver TMY weather data confirmed (BASE-04 complete)

**Remaining Issues (Deferred to Later Phases):**
- Peak cooling load under-prediction (systematic across all cases, Phase 3)
- Solar gain calculations (SOLAR-01 through SOLAR-04, Phase 3)

### Phase 2 Results Summary

**Completed Plans:**
- Plan 01: Thermal Mass Research
- Plan 02: Thermal Mass Integration Implementation
- Plan 03: Thermal Mass Validation
- Plan 04: Documentation & State Update

**Key Improvements:**
- Thermal mass dynamics validated via temperature swing reduction (22.4% vs 19.6% expected)
- Case 900 annual heating within reference range (1.77 MWh in [1.17, 2.04] MWh)
- All free-floating tests passing (10/10)
- Requirements FREE-02 and TEMP-01 completed

**Remaining Issues (Phase 3 Scope):**
- Annual cooling energy under-prediction (0.70 MWh vs [2.13, 3.67] MWh for Case 900)
- Peak heating load under-prediction (0.83 kW vs [1.10, 2.10] kW for Case 900)
- Peak cooling load under-prediction (0.60 kW vs [2.10, 3.50] kW for Case 900)
- Maximum free-floating temperature under-prediction (37.22°C vs [41.80, 46.40]°C for Case 900FF)

**Root Cause:** Solar gain calculation issues affecting peak loads and cooling energy.

## Roadmap Summary

**Total Phases:** 7
**Granularity:** Fine
**Coverage:** 51/51 requirements (100%)

| Phase | Goal | Requirements |
|-------|------|--------------|
| 1 - Foundation | Fix 5R1C conductance and HVAC load errors | 24 (BASE-01 through GROUND-01) |
| 2 - Thermal Mass Dynamics | Correct high-mass building simulation | 2 (FREE-02, TEMP-01) |
| 3 - Solar & External Boundaries | Validate solar gain and shading | 4 (SOLAR-01 through SOLAR-04) |
| 4 - Multi-Zone Transfer | Verify inter-zone heat transfer | 1 (MULTI-01) |
| 5 - Diagnostics & Reporting | Add diagnostic logging and reports | 4 (REPORT-01 through REPORT-04) |
| 6 - Performance Optimization | Optimize throughput and GPU acceleration | 12 (GPU-01 through REG-04) |
| 7 - Advanced Analysis | Sensitivity analysis and visualization | 20 (SENS-01 through MREF-03) |
| Phase 01 P03 | 360 | 2 tasks | 2 files |
| Phase 01-foundation P04 | 480 | 4 tasks | 4 files |
| Phase 02-Thermal-Mass-Dynamics P02-02 | 849 | 3 tasks | 3 files |
| Phase 02 P04 | 1773058930 | 4 tasks | 4 files |
| Phase 02 P05 | 152 | 1 tasks | 1 files |
| Phase 03 P00 | 6 | 2 tasks | 2 files |
| Phase 03 P01 | 30min | 6 tasks | 2 files |
| Phase 03 P03-03 | 1068 | 5 tasks | 3 files |
| Phase 03-Solar-Radiation P07 | 45min | 3 tasks | 3 files |

### Dependency Chain

```
Phase 1 (Foundation)
    ↓
Phase 2 (Thermal Mass)
    ↓
Phase 3 (Solar & External Boundaries)
    ↓
Phase 4 (Multi-Zone Transfer)
    ↓
Phase 5 (Diagnostics & Reporting)
    ↓
Phase 6 (Performance Optimization)
    ↓
Phase 7 (Advanced Analysis & Visualization)
```

### Success Criteria Preview

**Phase 1: Foundation**
1. All baseline Cases 600, 610, 620, 630, 640, 650 pass with both ±15% annual and ±10% monthly energy tolerances
2. Case 900 (high-mass) passes with ±15% annual and ±10% monthly energy tolerances
3. Free-floating cases (600FF, 650FF, 900FF) report min/max/avg temperatures within acceptable ranges
4. Peak heating and cooling loads match ASHRAE reference values within ±10% tolerance
5. Mean Absolute Error reduced from 78.79% to <15% across all baseline cases
6. Annual heating load over-prediction systematically corrected (no consistent bias)

**Phase 2: Thermal Mass Dynamics**
1. High-mass Case 900 passes validation with thermal mass dynamics correctly modeled
2. Free-floating cases show realistic thermal lag and damping characteristics
3. Thermal mass response time matches ASHRAE reference values within ±10% tolerance
4. Mass-air coupling (h_tr_em, h_tr_ms) correctly implemented and validated

**Phase 3: Solar & External Boundaries**
1. Solar gain calculations match ASHRAE reference values within ±5% tolerance for all orientations
2. Peak cooling loads match ASHRAE reference values within ±10% tolerance
3. Shading cases (610, 630, 910, 930) pass validation with correct shading effects
4. Beam/diffuse solar decomposition produces accurate hourly radiation values

**Phase 4: Multi-Zone Transfer**
1. Case 960 multi-zone sunspace passes ASHRAE 140 validation within ±15% annual tolerance
2. Inter-zone heat transfer correctly modeled between conditioned and sunspace zones
3. Zone temperature gradients match ASHRAE reference values

**Phase 5: Diagnostics & Reporting**
1. Validation report generates comprehensive Markdown summary with all cases, metrics, and pass/fail status
2. Diagnostic logging provides hourly temperature profiles, loads, and energy breakdowns for debugging
3. Hourly time series exported to CSV format for external analysis
4. Report identifies systematic issues across cases and tracks progress toward 100% pass rate

**Phase 6: Performance Optimization**
1. Complete validation suite (18+ cases) executes in <5 minutes using rayon parallel execution
2. GPU-accelerated solar calculations achieve 10-100x speedup for large populations
3. ONNX Runtime session pool enables concurrent AI surrogate inference without blocking
4. Performance regression guardrails detect MAE increases >2% or max deviation >10%
5. Historical performance data stored and trended over time

**Phase 7: Advanced Analysis & Visualization**
1. Sensitivity analysis identifies dominant parameters and produces ranked impact metrics
2. Delta testing framework supports custom case specifications and variant comparison
3. Interactive visualization provides real-time plotting with zoom/pan and export to PNG/SVG
4. Component-level energy breakdown helps diagnose specific energy path errors
5. Extensible test case framework supports custom geometries, climate zones, and building types
6. Multi-reference comparison shows consistency across EnergyPlus, ESP-r, and TRNSYS

---
*State initialized: 2026-03-08*
