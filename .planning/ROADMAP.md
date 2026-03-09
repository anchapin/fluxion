# Fluxion ASHRAE 140 Validation Roadmap

**Created:** 2026-03-08
**Granularity:** Fine
**Total Phases:** 7
**Coverage:** 51/51 requirements (100%)

## Phases

- [ ] **Phase 1: Foundation - Core Validation Fixes** - Fix critical 5R1C conductance and HVAC load calculation errors causing 61% failure rate
- [ ] **Phase 2: Thermal Mass Dynamics** - Correct high-mass building simulation with proper mass-air coupling and integration
- [ ] **Phase 3: Solar Radiation & External Boundaries** - Validate solar gain calculations, beam/diffuse decomposition, and shading effects
- [ ] **Phase 4: Multi-Zone Inter-Zone Transfer** - Verify and correct inter-zone heat transfer calculations for Case 960
- [ ] **Phase 5: Diagnostic Tools & Reporting** - Add comprehensive diagnostic logging, hourly CSV export, and validation reports
- [ ] **Phase 6: Performance Optimization** - Optimize batch validation throughput and add GPU-accelerated calculations
- [ ] **Phase 7: Advanced Analysis & Visualization** - Implement sensitivity analysis, delta testing, and interactive visualization

## Phase Details

### Phase 1: Foundation - Core Validation Fixes

**Goal**: Correct fundamental 5R1C thermal network parameterization and HVAC load calculations to reduce 61% failure rate and 78.79% MAE.

**Depends on**: Nothing (first phase)

**Requirements**:
- BASE-01: All Cases 600, 610, 620, 630, 640, 650 pass with ±15% annual energy tolerance
- BASE-02: All Cases 600, 610, 620, 630, 640, 650 pass with ±10% monthly energy tolerance
- BASE-03: Case 900 passes with ±15% annual energy tolerance and ±10% monthly energy tolerance
- BASE-04: Baseline cases (600/900) use Denver TMY weather data in EPW format
- FREE-01: Cases 600FF, 650FF, 900FF pass free-floating mode validation
- FREE-02: Free-floating mode tests thermal mass dynamics independently of HVAC
- COND-01: Case 195 validates envelope heat transfer independently
- METRIC-01: Validation produces annual heating/cooling energy values (MWh) for all cases
- METRIC-02: Validation produces peak heating/cooling loads (kW) for all cases
- REF-01: All cases compare results to ASHRAE 140 reference ranges and show pass/fail within ±5% tolerance band
- TEMP-01: Free-floating cases report min/max/avg temperatures (°C)
- WEATHER-01: All cases use Denver TMY weather data in EPW format
- THERM-01: All non-FF cases use dual setpoints (heating <20°C, cooling >27°C)
- THERM-02: Thermostat control validates setpoint logic and heating/cooling mode switching
- LAYER-01: Layer-by-layer R-value calculation for wall/roof/floor assemblies
- LAYER-02: ASHRAE film coefficients applied to window properties correctly
- WINDOW-01: U-value, SHGC, normal transmittance, and glass type parameters set correctly
- WINDOW-02: Double clear, low-E, and other glazing properties applied per case specifications
- INFIL-01: Air change rate (ACH) modeled correctly for baseline cases
- INTERNAL-01: Continuous internal gains (200W typical) modeled for occupied hours
- INTERNAL-02: Convective/radiative split applied to internal gains correctly
- GROUND-01: Ground boundary condition uses constant soil temperature (10°C)

**Success Criteria** (what must be TRUE):
1. All baseline Cases 600, 610, 620, 630, 640, 650 pass with both ±15% annual and ±10% monthly energy tolerances
2. Case 900 (high-mass) passes with ±15% annual and ±10% monthly energy tolerances
3. Free-floating cases (600FF, 650FF, 900FF) report min/max/avg temperatures within acceptable ranges
4. Peak heating and cooling loads match ASHRAE reference values within ±10% tolerance
5. Mean Absolute Error reduced from 78.79% to <15% across all baseline cases
6. Annual heating load over-prediction systematically corrected (no consistent bias)

**Plans**: TBD

### Phase 2: Thermal Mass Dynamics

**Goal**: Correct thermal mass dynamics for high-mass building cases with proper implicit/semi-implicit integration and mass-air coupling.

**Depends on**: Phase 1

**Requirements**:
- FREE-02: Free-floating mode tests thermal mass dynamics independently of HVAC
- TEMP-01: Free-floating cases report min/max/avg temperatures (°C) to validate thermal mass response

**Success Criteria** (what must be TRUE):
1. High-mass Case 900 passes validation with thermal mass dynamics correctly modeled
2. Free-floating cases show realistic thermal lag and damping characteristics
3. Thermal mass response time matches ASHRAE reference values within ±10% tolerance
4. Mass-air coupling (h_tr_em, h_tr_ms) correctly implemented and validated

**Plans**: TBD

### Phase 3: Solar Radiation & External Boundaries

**Goal**: Validate solar gain calculations, beam/diffuse decomposition, and shading geometry to fix cooling load under-prediction.

**Depends on**: Phase 2

**Requirements**:
- SOLAR-01: Hourly DNI/DHI solar radiation values calculated for all building orientations
- SOLAR-02: Solar incidence angle effects modeled for all orientations
- SOLAR-03: Window transmittance (SHGC) and normal transmittance values applied correctly
- SOLAR-04: Solar radiation modeling supports beam/diffuse decomposition

**Success Criteria** (what must be TRUE):
1. Solar gain calculations match ASHRAE reference values within ±5% tolerance for all orientations
2. Peak cooling loads match ASHRAE reference values within ±10% tolerance
3. Shading cases (610, 630, 910, 930) pass validation with correct shading effects
4. Beam/diffuse solar decomposition produces accurate hourly radiation values

**Plans**: TBD

### Phase 4: Multi-Zone Inter-Zone Transfer

**Goal**: Verify and correct inter-zone heat transfer calculations for multi-zone Case 960.

**Depends on**: Phase 3

**Requirements**:
- MULTI-01: Case 960 passes with inter-zone heat transfer validation

**Success Criteria** (what must be TRUE):
1. Case 960 multi-zone sunspace passes ASHRAE 140 validation within ±15% annual tolerance
2. Inter-zone heat transfer correctly modeled between conditioned and sunspace zones
3. Zone temperature gradients match ASHRAE reference values

**Plans**: TBD

### Phase 5: Diagnostic Tools & Reporting

**Goal**: Add comprehensive diagnostic logging, hourly CSV export, and validation report generation to accelerate debugging.

**Depends on**: Phase 4

**Requirements**:
- REPORT-01: Validation produces human-readable Markdown summary with pass/fail status
- REPORT-02: Validation provides detailed error breakdown by metric
- REPORT-03: Validation includes case-by-case comparison tables
- REPORT-04: Validation shows systematic issues identified and addressed

**Success Criteria** (what must be TRUE):
1. Validation report generates comprehensive Markdown summary with all cases, metrics, and pass/fail status
2. Diagnostic logging provides hourly temperature profiles, loads, and energy breakdowns for debugging
3. Hourly time series exported to CSV format for external analysis
4. Report identifies systematic issues across cases and tracks progress toward 100% pass rate

**Plans**: TBD

### Phase 6: Performance Optimization

**Goal**: Optimize batch validation throughput to achieve <5 minute execution time for all 18+ cases and add GPU-accelerated calculations.

**Depends on**: Phase 5

**Requirements**:
- GPU-01: ONNX Runtime integrated with CUDA backend for parallel solar calculations
- GPU-02: Batch inference optimization for neural surrogates with GPU kernel acceleration
- GPU-03: GPU memory management for large population evaluations
- SURR-01: ONNX Runtime session pool for concurrent AI surrogate inference
- SURR-02: Batched surrogate inference with rayon for population-level parallelism
- SURR-03: Neural surrogates trained and integrated for expensive physics calculations
- BATCH-01: All 18+ ASHRAE 140 cases executed in parallel with rayon
- BATCH-02: Aggregated validation results collected and summarized automatically
- BATCH-03: Complete validation suite execution time <5 minutes
- REG-01: Mean Absolute Error (MAE) tracked and alert generated when >2%
- REG-02: Max Deviation tracked and alert generated when >10%
- REG-03: Pass rate trends monitored over time to detect performance regression
- REG-04: Historical performance data stored for long-term trend analysis

**Success Criteria** (what must be TRUE):
1. Complete validation suite (18+ cases) executes in <5 minutes using rayon parallel execution
2. GPU-accelerated solar calculations achieve 10-100x speedup for large populations
3. ONNX Runtime session pool enables concurrent AI surrogate inference without blocking
4. Performance regression guardrails detect MAE increases >2% or max deviation >10%
5. Historical performance data stored and trended over time

**Plans**: TBD

### Phase 7: Advanced Analysis & Visualization

**Goal**: Implement sensitivity analysis, delta testing, and interactive visualization for research and optimization workflows.

**Depends on**: Phase 6

**Requirements**:
- SENS-01: Parameter perturbation studies to measure impact on energy consumption
- SENS-02: Case variant comparison to isolate individual effects
- SENS-03: Sensitivity metrics calculated (NMBE, CVRMSE, percentage change per parameter)
- SENS-04: Sensitivity analysis results exported to CSV for external analysis
- DELTA-01: Case variant comparison implemented
- DELTA-02: Delta test framework supports custom case specifications
- DELTA-03: Delta test results show isolated parameter effects
- VIZ-01: Real-time plotting of temperature profiles and HVAC demand curves
- VIZ-02: Interactive visualization supports zooming and pan for detailed inspection
- VIZ-03: Visualization plots exported to PNG/SVG format for documentation
- VIZ-04: Time series animation for understanding thermal dynamics
- COMP-01: Diagnostic reports include energy breakdown by component
- COMP-02: Component-level data exported to CSV for detailed analysis
- COMP-03: Component breakdown helps diagnose over/under-prediction in specific energy paths
- SWING-01: Min/max/avg free-floating temperatures calculated and reported
- SWING-02: Temperature swing range (max - min) quantified and validated
- SWING-03: Swing analysis identifies thermal mass effectiveness and passive cooling/heating potential
- EXT-01: Builder pattern supports custom case specifications beyond ASHRAE 140
- EXT-02: Custom climate zones supported
- EXT-03: Custom building geometries supported
- EXT-04: Extensible framework documented for future case additions
- MREF-01: Validation results compared to EnergyPlus, ESP-r, and TRNSYS simultaneously
- MREF-02: Multi-reference comparison tables generated showing side-by-side results
- MREF-03: Consistency checks performed across all reference programs

**Success Criteria** (what must be TRUE):
1. Sensitivity analysis identifies dominant parameters and produces ranked impact metrics
2. Delta testing framework supports custom case specifications and variant comparison
3. Interactive visualization provides real-time plotting with zoom/pan and export to PNG/SVG
4. Component-level energy breakdown helps diagnose specific energy path errors
5. Extensible test case framework supports custom geometries, climate zones, and building types
6. Multi-reference comparison shows consistency across EnergyPlus, ESP-r, and TRNSYS

**Plans**: TBD

## Progress Table

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Foundation | 0/0 | Not started | - |
| 2. Thermal Mass Dynamics | 0/0 | Not started | - |
| 3. Solar Radiation & External Boundaries | 0/0 | Not started | - |
| 4. Multi-Zone Inter-Zone Transfer | 0/0 | Not started | - |
| 5. Diagnostic Tools & Reporting | 0/0 | Not started | - |
| 6. Performance Optimization | 0/0 | Not started | - |
| 7. Advanced Analysis & Visualization | 0/0 | Not started | - |

## Dependencies

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

## Research Flags

**Phases requiring deeper research during planning:**
- Phase 3 (Solar Radiation & External Boundaries): Solar radiation modeling with shading geometry requires research into beam/diffuse decomposition algorithms, incidence angle calculations, and ASHRAE film coefficient correlations
- Phase 6 (Performance Optimization): GPU-accelerated thermal calculations and ONNX Runtime integration need research into CUDA kernel optimization, memory management strategies, and async inference patterns

**Phases with standard patterns (skip research-phase):**
- Phase 1 (Foundation): Well-documented 5R1C thermal network and HVAC control patterns in ASHRAE 140 standard and ISO 13790
- Phase 2 (Thermal Mass): Standard implicit/semi-implicit integration methods for thermal mass
- Phase 4 (Multi-Zone): Multi-zone heat transfer uses established coupled differential equation solvers
- Phase 5 (Diagnostics): Diagnostic logging and CSV export follow standard Rust patterns
- Phase 7 (Advanced Analysis): Sensitivity analysis, delta testing, and visualization have well-established libraries (scikit-learn, matplotlib, pandas)

## Coverage Validation

**v1 Requirements Mapped:** 51/51 (100%)

**Requirements by Phase:**
- Phase 1: 24 requirements (BASE-01 through GROUND-01)
- Phase 2: 2 requirements (FREE-02, TEMP-01)
- Phase 3: 4 requirements (SOLAR-01 through SOLAR-04)
- Phase 4: 1 requirement (MULTI-01)
- Phase 5: 4 requirements (REPORT-01 through REPORT-04)
- Phase 6: 12 requirements (GPU-01 through REG-04)
- Phase 7: 20 requirements (SENS-01 through MREF-03)

**Orphaned Requirements:** 0 ✓
**Duplicate Requirements:** 0 ✓
**Gaps:** None identified ✓

---
*Roadmap created: 2026-03-08*
