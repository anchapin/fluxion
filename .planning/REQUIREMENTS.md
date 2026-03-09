# Requirements: Fluxion ASHRAE 140 Validation Fixes

**Defined:** 2026-03-08
**Core Value:** Every ASHRAE 140 test case must pass with energy consumption and peak load predictions within ASHRAE tolerance bands (±15% annual, ±10% monthly).

## v1 Requirements

### Baseline Test Cases (600/900 Series)

- [ ] **BASE-01**: All Cases 600, 610, 620, 630, 640, 650 pass with ±15% annual energy tolerance (currently passing: 600, 610, 620, 630)
- [ ] **BASE-02**: All Cases 600, 610, 620, 630, 640, 650 pass with ±10% monthly energy tolerance (currently passing: 600, 610, 620, 630, 640)
- [ ] **BASE-03**: Case 900 passes with ±15% annual energy tolerance and ±10% monthly energy tolerance (currently fails with 78.79% MAE)
- [ ] **BASE-04**: Baseline cases (600/900) use Denver TMY weather data in EPW format

### Free-Floating Mode

- [ ] **FREE-01**: Cases 600FF, 650FF, 900FF pass free-floating mode validation (min/max/avg temperatures within acceptable range)
- [ ] **FREE-02**: Free-floating mode tests thermal mass dynamics independently of HVAC (Cases 195FF, 600FF, 650FF, 900FF)

### Multi-Zone Capability

- [ ] **MULTI-01**: Case 960 passes with inter-zone heat transfer validation (currently passes)

### Special Conduction Cases

- [ ] **COND-01**: Case 195 validates envelope heat transfer independently of solar/load effects (solid conduction, no windows/loads)

### Annual/Peak Load Metrics

- [ ] **METRIC-01**: Validation produces annual heating/cooling energy values (MWh) for all cases
- [ ] **METRIC-02**: Validation produces peak heating/cooling loads (kW) for all cases

### Reference Range Comparison

- [ ] **REF-01**: All cases compare results to ASHRAE 140 reference ranges (EnergyPlus, ESP-r, TRNSYS) and show pass/fail within ±5% tolerance band

### Temperature Metrics (FF Cases)

- [ ] **TEMP-01**: Free-floating cases report min/max/avg temperatures (°C) to validate thermal mass response

### Weather Data Integration

- [ ] **WEATHER-01**: All cases use Denver TMY weather data in EPW format (hourly dry-bulb, wet-bulb temperatures)

### Solar Radiation Modeling

- [ ] **SOLAR-01**: Hourly DNI/DHI solar radiation values calculated for all building orientations
- [ ] **SOLAR-02**: Solar incidence angle effects modeled for all orientations
- [ ] **SOLAR-03**: Window transmittance (SHGC) and normal transmittance values applied correctly
- [ ] **SOLAR-04**: Solar radiation modeling supports beam/diffuse decomposition

### Thermostat Control

- [ ] **THERM-01**: All non-FF cases use dual setpoints (heating <20°C, cooling >27°C)
- [ ] **THERM-02**: Thermostat control validates setpoint logic and heating/cooling mode switching

### Multi-Layer Construction

- [ ] **LAYER-01**: Layer-by-layer R-value calculation for wall/roof/floor assemblies
- [ ] **LAYER-02**: ASHRAE film coefficients applied to window properties correctly

### Window Properties

- [ ] **WINDOW-01**: U-value, SHGC, normal transmittance, and glass type parameters set correctly for all cases
- [ ] **WINDOW-02**: Double clear, low-E, and other glazing properties applied per case specifications

### Infiltration Modeling

- [ ] **INFIL-01**: Air change rate (ACH) modeled correctly for baseline cases (typically 0.5 ACH)

### Internal Loads

- [ ] **INTERNAL-01**: Continuous internal gains (200W typical) modeled for occupied hours
- [ ] **INTERNAL-02**: Convective/radiative split applied to internal gains correctly

### Ground Boundary Condition

- [ ] **GROUND-01**: Ground boundary condition uses constant soil temperature (10°C per ASHRAE 140 specification)

### Validation Report Generation

- [ ] **REPORT-01**: Validation produces human-readable Markdown summary with pass/fail status and tolerance bands
- [ ] **REPORT-02**: Validation provides detailed error breakdown by metric (heating, cooling, peaks)
- [ ] **REPORT-03**: Validation includes case-by-case comparison tables
- [ ] **REPORT-04**: Validation shows systematic issues identified and addressed

## v2 Requirements

### Diagnostic Logging & Debugging Tools

- [ ] **DIAG-01**: Comprehensive diagnostic logging with hourly temperature profiles, loads, and energy breakdowns
- [ ] **DIAG-02**: Diagnostic data exported to CSV format for external analysis
- [ ] **DIAG-03**: Peak load timing identification with hourly timestamps
- [ ] **DIAG-04**: Energy component breakdown (conduction, infiltration, solar, internal gains) for debugging
- [ ] **DIAG-05**: Environment variable configuration (`ASHRAE_140_DEBUG`, `ASHRAE_140_VERBOSE`) for diagnostic output toggling

### Sensitivity Analysis

- [ ] **SENS-01**: Parameter perturbation studies to measure impact on energy consumption
- [ ] **SENS-02**: Case variant comparison to isolate individual effects (e.g., with/without shading)
- [ ] **SENS-03**: Sensitivity metrics calculated (NMBE, CVRMSE, percentage change per parameter)
- [ ] **SENS-04**: Sensitivity analysis results exported to CSV for external analysis

### Automated CI/CD Integration

- [ ] **CI-01**: GitHub Actions workflow runs ASHRAE 140 validation on every commit with pass/fail thresholds
- [ ] **CI-02**: Blocking merge on regressions (MAE increase >2%, max deviation >10%)
- [ ] **CI-03**: Automated batch validation of all 18+ cases with rayon for comprehensive results

### GPU-Accelerated Calculations

- [ ] **GPU-01**: ONNX Runtime integrated with CUDA backend for parallel solar calculations
- [ ] **GPU-02**: Batch inference optimization for neural surrogates with GPU kernel acceleration
- [ ] **GPU-03**: GPU memory management for large population evaluations (>10,000 configs/sec)

### Neural Surrogate Integration

- [ ] **SURR-01**: ONNX Runtime session pool for concurrent AI surrogate inference
- [ ] **SURR-02**: Batched surrogate inference with rayon for population-level parallelism
- [ ] **SURR-03**: Neural surrogates trained and integrated for expensive physics calculations

### Batch Validation

- [ ] **BATCH-01**: All 18+ ASHRAE 140 cases executed in parallel with rayon
- [ ] **BATCH-02**: Aggregated validation results collected and summarized automatically
- [ ] **BATCH-03**: Complete validation suite execution time <5 minutes

### Hourly CSV Export

- [ ] **EXPORT-01**: Full hourly time series exported to CSV format (temperature, load, energy, HVAC)
- [ ] **EXPORT-02**: CSV export includes component-level breakdowns for external analysis

### Environment Variable Configuration

- [ ] **ENV-01**: `ASHRAE_140_DEBUG` flag enables detailed diagnostic logging
- [ ] **ENV-02**: `ASHRAE_140_VERBOSE` flag enables verbose simulation output
- [ ] **ENV-03**: Environment variables validated at startup and documented

### Peak Load Timing Validation

- [ ] **PEAK-01**: Peak heating occurs in winter months (Dec-Jan) and peak cooling in summer months (Jun-Aug)
- [ ] **PEAK-02**: Peak identification uses hourly average, not instantaneous, to match ASHRAE methodology
- [ ] **PEAK-03**: Peak units verified as kW (not W) to match reference values

### Component-Level Energy Breakdown

- [ ] **COMP-01**: Diagnostic reports include energy breakdown by component (conduction, infiltration, solar, internal gains)
- [ ] **COMP-02**: Component-level data exported to CSV for detailed analysis
- [ ] **COMP-03**: Component breakdown helps diagnose over/under-prediction in specific energy paths

### Free-Floating Temperature Swing Analysis

- [ ] **SWING-01**: Min/max/avg free-floating temperatures calculated and reported
- [ ] **SWING-02**: Temperature swing range (max - min) quantified and validated
- [ ] **SWING-03**: Swing analysis identifies thermal mass effectiveness and passive cooling/heating potential

### Regression Guardrails

- [ ] **REG-01**: Mean Absolute Error (MAE) tracked and alert generated when >2%
- [ ] **REG-02**: Max Deviation tracked and alert generated when >10%
- [ ] **REG-03**: Pass rate trends monitored over time to detect performance regression
- [ ] **REG-04**: Historical performance data stored for long-term trend analysis

### Delta Testing

- [ ] **DELTA-01**: Case variant comparison implemented (e.g., 600 vs 610 shading)
- [ ] **DELTA-02**: Delta test framework supports custom case specifications
- [ ] **DELTA-03**: Delta test results show isolated parameter effects

### Extensible Test Case Framework

- [ ] **EXT-01**: Builder pattern supports custom case specifications beyond ASHRAE 140 standard
- [ ] **EXT-02**: Custom climate zones supported (extensible framework)
- [ ] **EXT-03**: Custom building geometries supported (rectangular, L-shaped, etc.)
- [ ] **EXT-04**: Extensible framework documented for future case additions

### Multi-Reference Comparison

- [ ] **MREF-01**: Validation results compared to EnergyPlus, ESP-r, and TRNSYS simultaneously
- [ ] **MREF-02**: Multi-reference comparison tables generated showing side-by-side results
- [ ] **MREF-03**: Consistency checks performed across all reference programs

### Thermal Mass Response Analysis

- [ ] **MASS-01**: Thermal mass time constants extracted from simulation data
- [ ] **MASS-02**: Phase shift and damping characteristics analyzed for high-mass cases
- [ ] **MASS-03**: Mass response quality metrics calculated (overshoot, settling time)
- [ ] **MASS-04**: Mass analysis results exported for building optimization guidance

### Sensitivity Analysis

- [ ] **SENS-V1-01**: Global sensitivity analysis identifies dominant parameters
- [ ] **SENS-V1-02**: Sensitivity heat maps generated showing parameter impact zones
- [ ] **SENS-V1-03**: Sensitivity ranking helps prioritize parameter accuracy improvements

### Interactive Visualization

- [ ] **VIZ-01**: Real-time plotting of temperature profiles and HVAC demand curves
- [ ] **VIZ-02**: Interactive visualization supports zooming and pan for detailed inspection
- [ ] **VIZ-03**: Visualization plots exported to PNG/SVG format for documentation
- [ ] **VIZ-04**: Time series animation for understanding thermal dynamics

### Delta Testing

- [ ] **DEL-V1-01**: Delta test variants tracked and compared systematically
- [ ] **DEL-V1-02**: Delta test framework supports sensitivity analysis workflows
- [ ] **DEL-V1-03**: Delta test results integrated with regression guardrails

## Out of Scope

| Feature | Reason |
|----------|--------|
| 6R2C thermal model | Deferred to future - maintain ISO 13790 compliance for this validation cycle |
| FMI 3.0 co-simulation | Deferred to future - Phase 3 milestone scope |
| RL policy integration | Deferred to future - Phase 6 milestone scope |
| Custom tolerance bands per case | Out of scope - use standard ASHRAE 140 ±5% tolerance for all cases |
| Real-time validation during simulation | Out of scope - violates "collect data once" principle, breaks batching |
| Adaptive timestep in validation | Out of scope - breaks reproducibility and reference program compatibility |
| Custom tolerance bands per case | Out of scope - use standard ASHRAE 140 ±5% tolerance for all cases |
| Random design variations | Out of scope - reference programs don't test random cases, makes comparison impossible |
| Excessive diagnostic output by default | Out of scope - enabled via environment variables when debugging |
| Reference data embedding in source code | Out of scope - external CSV/JSON files in `benchmarks/` directory |
| Manual result verification | Out of scope - not scalable, error-prone, impossible for large validation suites |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| BASE-01 | Phase 1 | Pending |
| BASE-02 | Phase 1 | Pending |
| BASE-03 | Phase 1 | Pending |
| BASE-04 | Phase 1 | Pending |
| FREE-01 | Phase 1 | Pending |
| FREE-02 | Phase 1 | Pending |
| MULTI-01 | Phase 4 | Pending |
| COND-01 | Phase 1 | Pending |
| METRIC-01 | Phase 1 | Pending |
| METRIC-02 | Phase 1 | Pending |
| REF-01 | Phase 1 | Pending |
| TEMP-01 | Phase 1 | Pending |
| WEATHER-01 | Phase 1 | Pending |
| SOLAR-01 | Phase 3 | Pending |
| SOLAR-02 | Phase 3 | Pending |
| SOLAR-03 | Phase 3 | Pending |
| SOLAR-04 | Phase 3 | Pending |
| THERM-01 | Phase 1 | Pending |
| THERM-02 | Phase 1 | Pending |
| LAYER-01 | Phase 1 | Pending |
| LAYER-02 | Phase 1 | Pending |
| WINDOW-01 | Phase 1 | Pending |
| WINDOW-02 | Phase 1 | Pending |
| INFIL-01 | Phase 1 | Pending |
| INTERNAL-01 | Phase 1 | Pending |
| INTERNAL-02 | Phase 1 | Pending |
| GROUND-01 | Phase 1 | Pending |
| REPORT-01 | Phase 5 | Pending |
| REPORT-02 | Phase 5 | Pending |
| REPORT-03 | Phase 5 | Pending |
| REPORT-04 | Phase 5 | Pending |

**Coverage:**
- v1 requirements: 51 total
- Mapped to phases: 0 (will be created in ROADMAP.md)
- Unmapped: 51 ⚠️

---
*Requirements defined: 2026-03-08*
*Last updated: 2026-03-08 after research*
