# Architecture Research: ASHRAE 140 Full Compliance

**Domain:** Building Energy Modeling - ASHRAE Standard 140 Validation
**Researched:** 2026-03-13
**Overall confidence:** MEDIUM
**Confidence breakdown:** Stack: HIGH | Features: MEDIUM | Architecture: HIGH | Pitfalls: MEDIUM

## Executive Summary

Fluxion implements a 5R1C thermal network model (ISO 13790 compliant) with optional 6R2C extension. The existing architecture is well-structured for ASHRAE 140 validation, with comprehensive test infrastructure, multi-reference validation, and advanced analysis tools. However, achieving **full** ASHRAE 140 compliance requires addressing several gaps:

**Critical gaps identified:**
1. **High-mass annual energy accuracy:** 5R1C model over-predicts by 229-322% on 900 series cases; 6R2C evaluation showed no improvement (2026-03-13 decision: keep 5R1C as default)
2. **Diagnostic cases 195-470, 800-810:** Not yet implemented (only basic cases 600-960, 195 exist)
3. **HVAC equipment modeling:** Uses ideal HVAC controller; no equipment efficiency curves, part-load ratios, or capacity limits
4. **Weather data refinement:** Has EPW parsing and sky temperature calculations, but lacks psychrometric computations (dew point, humidity ratio, enthalpy)
5. **Statistical validation framework:** Has pass/fail validation but lacks statistical acceptance criteria (confidence intervals, NMBE, CVRMSE)

**Key architectural insight:** The existing codebase is **extendable without breaking changes**. The ThermalModelType enum pattern, modular validation framework, and CTA (Continuous Tensor Abstraction) provide clean extension points. The BatchOracle pattern with time-first loop optimization is maintained.

**Primary recommendation:** Extend existing 5R1C architecture rather than adopting 6R2C as default. Focus on thermal mass corrections, HVAC equipment modeling extensions, psychrometric module addition, and statistical validation framework.

---

## Key Findings

**Stack:** Rust 2021 with faer/ndarray for linear algebra, rayon for data parallelism, PyO3 for Python bindings
**Architecture:** 5R1C thermal network with optional 6R2C (experimental), modular validation with multi-reference comparison, BatchOracle pattern for high-throughput optimization
**Critical pitfall:** High-mass annual energy over-prediction is a structural 5R1C limitation; 6R2C provides no accuracy improvement per Phase 12 validation

---

## Recommended Architecture

### Core Framework

**Existing 5R1C Architecture (Keep as Default)**

```rust
pub struct ThermalModel<T: ContinuousTensor<f64>> {
    // State variables (CTA VectorFields)
    pub temperatures: T,           // Zone air temperatures
    pub mass_temperatures: T,      // Thermal mass temperatures
    pub loads: T,                  // Total thermal loads

    // 5R1C Parameters (CTA VectorFields)
    pub h_tr_em: T,  // Transmission: Exterior -> Mass
    pub h_tr_ms: T,  // Transmission: Mass -> Surface
    pub h_tr_is: T,  // Transmission: Surface -> Interior
    pub h_tr_w: T,   // Transmission: Exterior -> Interior (Windows)
    pub h_ve: T,     // Ventilation: Exterior -> Interior

    // Thermal model type (5R1C or 6R2C)
    pub thermal_model_type: ThermalModelType,

    // HVAC equipment extensions (NEW)
    pub hvac_equipment: Option<HVACEquipment>,
}
```

**Rationale:** 5R1C is ISO 13790 compliant, validated for low-mass cases, and meets performance targets (2,575 configs/sec). 6R2C evaluation showed no accuracy improvement with 1.5-2x performance penalty (docs/6R2C_DECISION.md).

**New Component: HVAC Equipment Module**

```rust
// src/hvac/equipment.rs
pub struct HVACEquipment {
    pub equipment_type: EquipmentType,  // Boiler, Chiller, Heat Pump, DX Coil
    pub heating_capacity: f64,         // Watts
    pub cooling_capacity: f64,         // Watts
    pub heating_efficiency_curve: EfficiencyCurve,  // Part-load efficiency
    pub cooling_efficiency_curve: EfficiencyCurve,
    pub part_load_ratio: f64,        // Current PLR (0-1)
    pub minimum_turn_down: f64,        // Minimum operating point (0-1)
    pub cycling_losses: f64,           // Cycling penalty factor
}

pub enum EquipmentType {
    IdealHVAC,      // Current model (infinite capacity, perfect control)
    Boiler,          // Condensing/non-condensing boilers
    Chiller,         // Vapor compression chillers
    HeatPump,       // Air-source heat pump
    DXCoil,         // Direct expansion cooling coil
}

pub struct EfficiencyCurve {
    pub coefficients: Vec<f64>,  // Polynomial coefficients for PLR vs efficiency
    pub reference_conditions: ReferenceConditions,
}
```

**Integration Point:** Extend `ThermalModel::solve_timesteps` to call `hvac_equipment.compute_output(load_demand)` instead of ideal controller.

---

### Component Boundaries

| Component | Responsibility | Communicates With |
|-----------|---------------|-------------------|
| **ThermalModel** (src/sim/engine.rs) | 5R1C/6R2C thermal network solving, state updates | WeatherSource, HVACEquipment, SurrogateManager, SimulationDiagnostics |
| **HVACEquipment** (src/hvac/equipment.rs) | Equipment efficiency curves, part-load ratios, cycling losses | ThermalModel, Psychrometrics (for enthalpy calculations) |
| **Psychrometrics** (src/hvac/psychrometrics.rs) | Dew point, humidity ratio, enthalpy calculations | WeatherSource, HVACEquipment |
| **WeatherInterpolator** (src/weather/interpolator.rs) | Sub-hourly weather interpolation, radiation smoothing | WeatherSource, ThermalModel |
| **StatisticalValidator** (src/validation/statistical.rs) | NMBE, CVRMSE, confidence intervals | ASHRAE140Validator, ValidationReportGenerator |
| **DiagnosticCaseBuilder** (src/validation/diagnostics/builder.rs) | Cases 195-470, 800-810 construction | ASHRAE140Case, ThermalModel |
| **ASHRAE140Validator** (src/validation/ashrae_140_validator.rs) | Multi-reference validation, pass/fail determination | MultiReferenceDB, StatisticalValidator |
| **BatchOracle** (src/lib.rs) | High-throughput population evaluation | ThermalModel (cloned per config), SurrogateManager |

---

### Data Flow

**Existing Time-First Loop (Maintain for Performance)**

```
for timestep in 0..8760 {
    // 1. Collect all zone temperatures from all configurations
    let all_temps: Vec<Vec<f64>> = configs.iter().map(|c| c.temperatures.as_ref().to_vec()).collect();

    // 2. Single batched inference for all configurations
    let all_loads = surrogates.predict_loads_batched(&all_temps)?;

    // 3. Distribute loads and run physics in parallel with rayon
    configs.par_iter_mut().for_each(|config| {
        config.set_loads(&all_loads[config_index]);
        config.step_physics(timestep, outdoor_temp);
    });

    // 4. Collect results
    let results: Vec<f64> = configs.iter().map(|c| c.hvac_energy).collect();
}
```

**New HVAC Equipment Integration Point**

```
In ThermalModel::step_physics:
    let hvac_demand = calculate_ideal_hvac_demand(ti_free, ti_setpoint_heating, ti_setpoint_cooling);

    // NEW: Apply equipment constraints
    let hvac_output = if let Some(equipment) = &self.hvac_equipment {
        equipment.compute_output(hvac_demand, self.temperatures.as_ref()[zone_idx])
    } else {
        // Fallback to ideal controller
        ideal_hvac_controller(hvac_demand)
    };
```

**New Weather Interpolation Flow**

```
ThermalModel::solve_timesteps:
    for step in 0..num_steps {
        let sub_step_fraction = step % sub_steps_per_hour;
        let weather = weather_interpolator.interpolate(step, sub_step_fraction);

        self.step_physics(step, weather.dry_bulb_temp);
    }
```

**New Statistical Validation Flow**

```
ASHRAE140Validator::validate_analytical_engine:
    let results = self.run_all_cases();

    // NEW: Statistical analysis
    let stats = StatisticalValidator::analyze(&results, &reference_db);
    let nmbes = stats.nmbes_by_case();
    let cvrmses = stats.cvrmse_by_case();
    let confidence_intervals = stats.confidence_intervals(0.95);

    // Generate report with statistical metrics
    report.add_statistics(nmbes, cvrmses, confidence_intervals);
```

---

## Recommended Project Structure

```
src/
├── sim/
│   ├── engine.rs              # ThermalModel with 5R1C/6R2C branching
│   └── thermal_model.rs      # ThermalModelBuilder for case construction
├── hvac/                     # NEW: HVAC equipment module
│   ├── equipment.rs           # HVACEquipment, EfficiencyCurve, EquipmentType
│   └── psychrometrics.rs     # Psychrometrics, dew point, enthalpy
├── weather/
│   ├── mod.rs                # WeatherSource trait, HourlyWeatherData
│   ├── epw.rs                # EPW file parsing
│   └── interpolator.rs       # NEW: Weather interpolation
├── validation/
│   ├── ashrae_140_cases.rs  # ASHRAE 140 test case specifications
│   ├── ashrae_140_validator.rs  # Validation framework with comparison logic
│   ├── diagnostics/          # NEW: Diagnostic case builders
│   │   ├── mod.rs
│   │   ├── builder.rs        # Case 195-470, 800-810 builders
│   │   └── specs.rs          # Diagnostic case specifications
│   ├── statistical.rs         # NEW: Statistical validation
│   └── report.rs             # ValidationReport, compute_status
└── physics/
    └── cta.rs                # Continuous Tensor Abstraction (VectorField)

tests/
├── test_hvac_equipment.rs       # NEW: HVAC equipment tests
├── test_psychrometrics.rs       # NEW: Psychrometric tests
├── test_weather_interpolator.rs  # NEW: Weather interpolation tests
├── test_diagnostic_cases.rs     # NEW: Diagnostic case validation
├── test_statistical_validation.rs  # NEW: Statistical validation tests
└── test_6r2c_model.rs      # Existing 6R2C tests (11 tests)

docs/
├── 6R2C_IMPLEMENTATION.md  # Existing 6R2C design documentation
├── 6R2C_DECISION.md         # Existing 6R2C adoption decision
└── KNOWN_LIMITATIONS.md        # 5R1C limitations and 6R2C investigation results
```

### Structure Rationale

- **sim/engine.rs:** Core physics engine; maintains 5R1C/6R2C branching pattern
- **hvac/** (NEW): Isolate HVAC complexity from physics engine; clean separation of concerns
- **weather/interpolator.rs** (NEW): Keep weather processing modular; easy to test and optimize independently
- **validation/diagnostics/** (NEW): Separate diagnostic case builders from baseline cases; maintain clear organization
- **validation/statistical.rs** (NEW): Statistical validation as separate module; easy to extend with new metrics
- **tests/**: Co-locate tests with implementation for easy discovery; maintain existing test organization

---

## Architectural Patterns

### Pattern 1: Thermal Model Type Switching (Existing - Keep)

**What:** Runtime selection between 5R1C and 6R2C physics solvers based on ThermalModelType enum
**When to use:** When ThermalModel::step_physics is called, branch to appropriate solver
**Trade-offs:** Pros: Backward compatible, minimal overhead (single branch check). Cons: Dual code paths to maintain.

**Example:**
```rust
// Source: src/sim/engine.rs:2460-2465
pub fn step_physics(&mut self, timestep: usize, outdoor_temp: f64) -> f64 {
    if self.is_6r2c_model() {
        self.step_physics_6r2c(timestep, outdoor_temp)
    } else {
        self.step_physics_5r1c(timestep, outdoor_temp)
    }
}
```
**Why keep:** Proven pattern, maintains backward compatibility, minimal overhead (single branch check)

---

### Pattern 2: Modular Validation with Multi-Reference Comparison (Existing - Extend)

**What:** Load reference data from JSON, compare against EnergyPlus, ESP-r, TRNSYS ranges
**When to use:** In ASHRAE140Validator::validate_case and add_result_with_multi
**Trade-offs:** Pros: Comprehensive validation coverage, easy to add new reference programs. Cons: JSON dependency, needs reference data maintenance.

**Example:**
```rust
// Source: src/validation/report.rs:106-134
pub fn compute_status(value: f64, ref_min: f64, ref_max: f64) -> ValidationStatus {
    let ref_mid = (ref_min + ref_max) / 2.0;
    let percent_error = if ref_mid != 0.0 {
        ((value - ref_mid) / ref_mid.abs()) * 100.0
    } else {
        0.0
    };

    let tolerance_min = ref_min * 0.95;
    let tolerance_max = ref_max * 1.05;

    if value >= ref_min && value <= ref_max {
        if percent_error.abs() >= 10.0 {
            ValidationStatus::Warning
        } else {
            ValidationStatus::Pass
        }
    } else if value >= tolerance_min && value <= tolerance_max {
        ValidationStatus::Warning
    } else {
        ValidationStatus::Fail
    }
}
```
**Why extend:** Provides comprehensive validation coverage, easy to add new reference programs, statistical analysis fits naturally into this framework

---

### Pattern 3: Optional Feature Pattern (Apply to HVAC Equipment)

**What:** Use Option<T> for optional components, default behavior preserved when None
**When to use:** Adding HVAC equipment, psychrometrics, or other non-core features
**Trade-offs:** Pros: Backward compatible, easy to enable/disable per test case, no breaking changes. Cons: Additional runtime checks (unwrap_or_else).

**Example:**
```rust
pub struct ThermalModel<T: ContinuousTensor<f64>> {
    pub hvac_equipment: Option<HVACEquipment>,  // NEW: Optional equipment model
    pub psychrometrics: Option<Psychrometrics>, // NEW: Optional psychrometrics
}
```
**Why use:** Backward compatible, easy to enable/disable per test case, no breaking changes to existing validation

---

### Pattern 4: Builder Pattern for Case Construction (Existing - Extend)

**What:** CaseBuilder for constructing ASHRAE 140 test cases with fluent API
**When to use:** Creating diagnostic cases 195-470, 800-810
**Trade-offs:** Pros: Clean API, easy to add new case types, type-safe. Cons: More boilerplate than direct struct initialization.

**Example:**
```rust
// Source: src/validation/ashrae_140_cases.rs
let custom_spec = CaseBuilder::new()
    .low_mass_construction()
    .with_dimensions(8.0, 6.0, 2.7)
    .with_south_window(12.0)
    .with_hvac_setpoints(20.0, 27.0)
    .build()?;
```
**Why extend:** Proven pattern for case construction, easy to add new case types (diagnostic, equipment modeling)

---

### Pattern 5: Continuous Tensor Abstraction (Existing - Maintain)

**What:** VectorField abstraction for tensor operations (+, *, /, gradient, integrate)
**When to use:** All physics calculations use CTA operations instead of raw Vec<f64>
**Trade-offs:** Pros: Enables future GPU acceleration, clean physics code, vectorized operations already optimized. Cons: Additional abstraction layer, learning curve for new contributors.

**Example:**
```rust
// Source: src/physics/cta.rs
impl<T: ContinuousTensor<f64>> Add<VectorField<T>> for VectorField<T> {
    type Output = VectorField<T>;
    fn add(self, other: VectorField<T>) -> Self::Output {
        // Element-wise addition
    }
}
```
**Why maintain:** Enables future GPU acceleration, clean physics code, vectorized operations already optimized

---

## Data Flow

### Request Flow

```
User Action (BatchOracle::evaluate_population or Model::simulate)
    ↓
ThermalModel (cloned per config) → SurrogateManager (batched inference)
    ↓              ↓
WeatherSource (hourly data) → ThermalModel::step_physics
    ↓                           ↓
HVACEquipment (optional) ← ThermalModel::solve_timesteps → ValidationReport
    ↓                           ↓
Psychrometrics (optional) ← HVACEquipment::compute_output
    ↓
ASHRAE140Validator::validate_case → StatisticalValidator (optional)
    ↓
ValidationReportGenerator (HTML/CSV export)
```

### Key Data Flows

1. **Batch Processing (BatchOracle):** Population vectors → ThermalModel clones → rayon par_iter → batched ONNX inference → parallel physics → energy results
2. **Weather Data:** WeatherSource → HourlyWeatherData → WeatherInterpolator (sub-hourly) → ThermalModel::step_physics
3. **HVAC Energy:** Thermal demand → HVACEquipment::compute_output (with psychrometrics) → HVAC energy → ValidationReport
4. **Statistical Validation:** Simulation results → StatisticalValidator::analyze (NMBE, CVRMSE, CI) → ValidationReport

---

## Scaling Considerations

| Scale | Architecture Adjustments |
|-------|--------------------------|
| 100 users | ~17 cases/sec (current validation throughput) is sufficient |
| 10K users | ~17 cases/sec (current validation throughput) is sufficient; add caching for equipment efficiency curves |
| 1M users | Horizontal scaling: distribute validation jobs across workers, use batch validation; GPU acceleration for CTA operations if GPU inference enabled |

### Scaling Priorities

1. **First bottleneck:** Validation throughput (~17 cases/sec) - Already optimized with time-first loop; can parallelize across multiple workers for large-scale validation
2. **Second bottleneck:** HVAC equipment calculations - Add caching for equipment efficiency curves, precompute polynomial evaluations
3. **Third bottleneck:** Psychrometric calculations - Cache humidity ratio for repeated timesteps, use SIMD for polynomial evaluations
4. **Fourth bottleneck:** Weather interpolation - Use GPU acceleration for vectorized interpolation, lazy evaluation (only interpolate when needed)

---

## Anti-Patterns

### Anti-Pattern 1: Breaking 5R1C Default Behavior

**What people do:** Changing ThermalModel::new() to default to 6R2C
**Why it's wrong:** 6R2C provides no accuracy improvement (docs/6R2C_DECISION.md), introduces 1.5-2x performance penalty
**Do this instead:** Keep 5R1C as default, offer 6R2C as opt-in via configure_6r2c_model()

---

### Anti-Pattern 2: Nested Parallelism in step_physics

**What people do:** Adding rayon::par_iter() inside step_physics or HVAC equipment calculations
**Why it's wrong:** Breaks BatchOracle pattern, causes thread pool contention, violates pre-commit hook
**Do this instead:** Only use rayon::par_iter() at population level in BatchOracle::evaluate_population

---

### Anti-Pattern 3: Hardcoding Psychrometric Constants

**What people do:** Defining atmospheric pressure, water vapor constants, etc., directly in equations
**Why it's wrong:** Reduces testability, makes altitude corrections difficult
**Do this instead:** Create Psychrometrics struct with configurable standard atmospheric pressure, methods for dew point, humidity ratio

---

### Anti-Pattern 4: Monolithic Validation Status Calculation

**What people do:** Computing pass/fail status inline during case simulation
**Why it's wrong:** Violates separation of concerns, hard to add statistical validation
**Do this instead:** Collect all results, then compute status with compute_status and StatisticalValidator::analyze

---

### Anti-Pattern 5: Ignoring Thermal Mass Correction in Equipment Modeling

**What people do:** HVAC equipment model uses ideal load demand without considering thermal mass energy storage
**Why it's wrong:** Over-predicts HVAC energy, violates energy balance in high-mass buildings
**Do this instead:** Use existing thermal_mass_energy_accounting flag, subtract mass energy change from HVAC demand before computing equipment output

---

## Integration Points

### External Services

| Service | Integration Pattern | Notes |
|---------|---------------------|-------|
| ONNX Runtime (for surrogates) | Load ONNX models at runtime, batched inference via SessionPool | Already implemented; ensure thread safety for concurrent inference |
| EPW weather files | EpwWeatherSource::from_file(path) parses TMY data | Already implemented; add interpolation support |
| Python PyO3 bindings | BatchOracle, Model classes exposed via #[pymodule] | Already implemented; maintain backward compatibility |

### Internal Boundaries

| Boundary | Communication | Notes |
|----------|---------------|-------|
| ThermalModel ↔ HVACEquipment | Direct method calls (ThermalModel calls HVACEquipment::compute_output) | Keep coupling minimal; use Option<HVACEquipment> for optional integration |
| HVACEquipment ↔ Psychrometrics | Direct method calls (HVACEquipment calls Psychrometrics::enthalpy) | Psychrometrics is pure functions, easy to test independently |
| WeatherInterpolator ↔ ThermalModel | Direct method calls (ThermalModel calls WeatherInterpolator::interpolate) | Ensure interpolation preserves energy conservation (integrated values should match hourly totals) |
| ASHRAE140Validator ↔ StatisticalValidator | Direct method calls (Validator calls StatisticalValidator::analyze) | Statistical analysis is additive, can run after all cases validated |

---

## Sources

### Primary (HIGH confidence)

**Existing codebase:**
- `src/sim/engine.rs` - ThermalModel struct, 5R1C/6R2C implementation, step_physics methods
- `src/validation/ashrae_140_validator.rs` - ASHRAE140Validator, validate_analytical_engine, multi-reference validation
- `src/validation/ashrae_140_cases.rs` - ASHRAE140Case enum, CaseBuilder pattern, all case specifications
- `src/validation/report.rs` - ValidationStatus, compute_status, BenchmarkReport, BenchmarkData
- `src/validation/diagnostics.rs` - SimulationDiagnostics, hourly data collection, CSV export
- `src/weather/mod.rs` - WeatherSource trait, HourlyWeatherData struct, validation methods
- `src/weather/epw.rs` - EpwWeatherSource, EPW file parsing
- `src/physics/cta.rs` - ContinuousTensor trait, VectorField, tensor operations

**Existing documentation:**
- `docs/6R2C_IMPLEMENTATION.md` - Comprehensive 6R2C design, thermal mass energy accounting, configuration methods
- `docs/6R2C_DECISION.md` - Phase 12 validation results, 6R2C vs 5R1C comparison, adoption decision (keep 5R1C as default)
- `docs/ARCHITECTURE.md` - BatchOracle pattern, ThermalModel structure, physics engine overview
- `docs/ASHRAE140_RESULTS.md` - Current validation status (18/18 passing), systematic issues identified
- `CLAUDE.md` - Project instructions, BatchOracle pattern, critical conventions, build commands
- `docs/KNOWN_LIMITATIONS.md` - 5R1C high-mass annual energy error (229-322%), peak load accuracy

**Phase research:**
- `.planning/phases/12-Model-Exploration/12-RESEARCH.md` - 6R2C evaluation, Phase 12 requirements, build sequence, pitfalls
- `.planning/phases/12-Model-Exploration/12-01-SUMMARY.md` - 6R2C validation results, decision criteria
- `.planning/REQUIREMENTS.md` - v0.3 maintenance release requirements, MODEL6R2C-01..05 tasks

### Secondary (MEDIUM confidence)

**Validation framework:**
- `src/validation/multi_reference.rs` - MultiReferenceDB, ProgramRange, per-program validation
- `src/validation/reporter.rs` - ValidationReportGenerator, systematic issue tracking, HTML/CSV export
- `src/validation/benchmark.rs` - BenchmarkData structure, get_benchmark_data function

**Analysis tools:**
- `src/analysis/sensitivity.rs` - OAT and Sobol sensitivity analysis
- `src/analysis/delta.rs` - Delta testing framework for variant comparison
- `src/analysis/components.rs` - Component breakdown (conduction, convection, radiation)

### Tertiary (LOW confidence)

**External sources (web search unavailable):**
- ASHRAE Standard 140 official document - Not accessible via web search (likely paywall)
- ASHRAE Handbook - Fundamentals (psychrometrics chapter) - Not accessible via web search
- EnergyPlus source code - Not accessed, would provide equipment model reference
- ESP-r documentation - Not accessed, would provide thermal network reference
- TRNSYS documentation - Not accessed, would provide validation reference

**Note:** Web search tool returned empty results for all ASHRAE 140 queries. Research is based primarily on existing codebase and documentation. External references to ASHRAE 140 specifications, equipment models and statistical criteria would strengthen confidence.

---

## Build Sequence Recommendation

### Phase 1: Thermal Mass Corrections (High Priority)
**Rationale:** Addresses root cause of high-mass annual energy errors; foundational for all other work

1. **Implement advanced thermal mass correction**
   - Extend thermal_mass_correction_factor to be case-specific (not just sqrt(C/C_ref))
   - Add time-constant-based corrections for high-mass buildings
   - Validate against Cases 900, 940 heating (current: 229-322% error)
   - **Dependencies:** None (extend existing ThermalModel)
   - **Estimated effort:** 2-3 weeks

2. **Add thermal mass energy accounting validation**
   - Validate that mass energy change is correctly tracked and subtracted from HVAC energy
   - Add diagnostic logging for thermal mass energy flows
   - Test with Cases 900FF, 950FF (free-floating) to verify energy conservation
   - **Dependencies:** Existing thermal_mass_energy_accounting flag
   - **Estimated effort:** 1 week

---

### Phase 2: HVAC Equipment Modeling (High Priority)
**Rationale:** ASHRAE 140 requires equipment validation; enables more realistic simulations

1. **Implement HVAC equipment module**
   - Create HVACEquipment struct with efficiency curves, part-load ratios
   - Add equipment types: Boiler, Chiller, Heat Pump, DX Coil
   - Implement compute_output(load_demand, zone_temp) method
   - **Dependencies:** Psychrometrics module (for enthalpy)
   - **Estimated effort:** 3-4 weeks

2. **Integrate equipment with ThermalModel**
   - Add hvac_equipment: Option<HVACEquipment> field
   - Modify step_physics to use equipment when available
   - Ensure thermal_mass_energy_accounting works with equipment
   - **Dependencies:** HVAC equipment module, Phase 1 thermal mass corrections
   - **Estimated effort:** 2 weeks

3. **Add equipment validation test cases**
   - Create ASHRAE 140 cases with equipment specifications
   - Validate against reference equipment performance data
   - Test part-load behavior, cycling losses, minimum turn-down
   - **Dependencies:** HVAC equipment integration
   - **Estimated effort:** 2 weeks

---

### Phase 3: Psychrometrics Module (Medium Priority)
**Rationale:** Required for accurate enthalpy calculations, equipment efficiency curves

1. **Implement psychrometric calculations**
   - Create Psychrometrics struct with standard atmospheric pressure
   - Add methods: dew_point, humidity_ratio, enthalpy, wet_bulb
   - Validate against ASHRAE Fundamentals reference values
   - **Dependencies:** None (new module)
   - **Estimated effort:** 2 weeks

2. **Integrate with weather and HVAC modules**
   - Modify HourlyWeatherData to include computed psychrometric properties
   - Use enthalpy in HVAC equipment efficiency calculations
   - Add psychrometric validation tests (e.g., dew point should be <= dry bulb)
   - **Dependencies:** Psychrometrics module, HVAC equipment module
   - **Estimated effort:** 1 week

---

### Phase 4: Diagnostic Cases (Medium Priority)
**Rationale:** ASHRAE 140 requires diagnostic cases 195-470, 800-810 for in-depth validation

1. **Implement diagnostic case builders**
   - Create Case195, Case200, Case470 specifications in ASHRAE140Case enum
   - Add Case800, Case810 specifications
   - Use CaseBuilder pattern for construction
   - **Dependencies:** None (extend existing case builders)
   - **Estimated effort:** 2-3 weeks

2. **Validate diagnostic cases**
   - Run full validation suite for diagnostic cases
   - Compare against EnergyPlus, ESP-r, TRNSYS reference results
   - Generate diagnostic reports highlighting discrepancies
   - **Dependencies:** Diagnostic case builders, Phase 1-3 improvements
   - **Estimated effort:** 1-2 weeks

---

### Phase 5: Weather Data Refinement (Medium Priority)
**Rationale:** Sub-hourly interpolation and radiation smoothing improve accuracy for thermal mass dynamics

1. **Implement weather interpolation**
   - Create WeatherInterpolator struct with cubic spline interpolation
   - Support 15-minute, 30-minute timesteps
   - Add radiation smoothing (moving average for DNI, DHI)
   - **Dependencies:** None (new module)
   - **Estimated effort:** 2 weeks

2. **Validate interpolation accuracy**
   - Compare interpolated weather against high-resolution TMY data
   - Verify energy conservation (interpolated solar gains should sum to hourly totals)
   - Test impact on thermal mass dynamics (Cases 900, 940)
   - **Dependencies:** Weather interpolation module
   - **Estimated effort:** 1 week

---

### Phase 6: Statistical Validation Framework (Low Priority)
**Rationale:** ASHRAE 140 statistical acceptance criteria (NMBE, CVRMSE) required for full compliance

1. **Implement statistical validator**
   - Create StatisticalValidator struct with NMBE, CVRMSE, confidence intervals
   - Add methods: analyze(results, references), compute_nmbes, compute_cvrmes
   - Validate against ASHRAE 140 statistical criteria (NMBE < 10%, CVRMSE < 30%)
   - **Dependencies:** Existing validation framework
   - **Estimated effort:** 2 weeks

2. **Integrate with validation reporting**
   - Add statistical metrics to ValidationReportGenerator
   - Update HTML/CSV export to include NMBE, CVRMSE, confidence intervals
   - Generate statistical summary tables by case type (low-mass, high-mass, diagnostic)
   - **Dependencies:** Statistical validator module
   - **Estimated effort:** 1 week

---

### Phase 7: Performance Optimization (Low Priority)
**Rationale:** Additional features (equipment, psychrometrics, interpolation) may impact throughput

1. **Profile new components**
   - Use criterion benchmarks to measure HVAC equipment calculation overhead
   - Profile psychrometric calculations per timestep
   - Measure weather interpolation impact on total simulation time
   - **Dependencies:** All previous phases implemented
   - **Estimated effort:** 1 week

2. **Optimize bottlenecks**
   - Cache equipment efficiency curve evaluations (polynomial coefficients)
   - SIMD-accelerate psychrometric calculations
   - Lazy evaluation for weather interpolation (only when needed)
   - Ensure BatchOracle pattern maintained (no nested parallelism)
   - **Dependencies:** Performance profiling results
   - **Estimated effort:** 2-3 weeks

---

## Phase Ordering Rationale

**Critical dependencies:**
1. **Phase 1 (Thermal Mass)** must be first because:
   - High-mass annual energy error is the largest validation gap (229-322%)
   - HVAC equipment modeling (Phase 2) depends on correct thermal mass energy accounting
   - Weather interpolation (Phase 5) impacts thermal mass dynamics

2. **Phase 2 (HVAC Equipment)** and **Phase 3 (Psychrometrics)** are tightly coupled:
   - HVAC equipment efficiency curves require enthalpy calculations (psychrometrics)
   - Both phases can proceed in parallel with interface stubs
   - Recommended: Start Phase 3 first (simpler, no dependencies), then Phase 2

3. **Phase 4 (Diagnostic Cases)** depends on Phase 1-3:
   - Diagnostic cases test equipment behavior, thermal mass dynamics
   - Need accurate thermal mass corrections before validating diagnostic cases
   - Psychrometrics needed for equipment-based diagnostic cases

4. **Phase 5 (Weather Interpolation)** can proceed independently after Phase 1:
   - Sub-hourly weather improves thermal mass dynamics accuracy
   - Can be developed in parallel with Phase 2-3
   - Recommended: Start after Phase 1 to focus on thermal mass first

5. **Phase 6 (Statistical Validation)** requires all validation results:
   - Needs comprehensive case set (diagnostic cases from Phase 4)
   - Statistical analysis benefits from larger dataset (all cases validated)
   - Recommended: Start after Phase 4 to have full case set

6. **Phase 7 (Performance Optimization)** must be last:
   - Cannot optimize until all features are implemented
   - Profiling results depend on complete feature set
   - Recommended: Start after Phase 6 to optimize final codebase

**Parallelization opportunities:**
- **Wave 1:** Phase 1 (Thermal Mass) + Phase 3 (Psychrometrics) can start in parallel
- **Wave 2:** After Phase 1 completes, Phase 2 (HVAC Equipment) + Phase 5 (Weather Interpolation) can start in parallel
- **Wave 3:** After Phase 2-4 complete, Phase 6 (Statistical Validation) + Phase 7 (Profiling) can start in parallel

**Estimated total effort:** 20-30 weeks across all phases (5-6 months with parallelization)

---

## Research Flags for Phases

### Phase 1: Thermal Mass Corrections
**Likely needs deeper research:**
- Are time-constant-based corrections sufficient, or need spatial thermal mass distribution?
- Should correction factors be case-specific or parameterized by construction type?
- How to validate thermal mass corrections without ASHRAE 140 reference for corrected cases?

### Phase 2: HVAC Equipment Modeling
**Likely needs deeper research:**
- What efficiency curve coefficients should be used for each equipment type?
- Are ASHRAE 140 equipment reference data publicly available?
- How to handle equipment sizing (capacity limits vs. load matching)?

### Phase 3: Psychrometrics Module
**Standard patterns, unlikely to need research:**
- Psychrometric equations are well-documented in ASHRAE Fundamentals
- Existing implementation patterns in EnergyPlus, TRNSYS can be referenced
- Recommended: Use ASHRAE Fundamentals 2021 as reference

### Phase 4: Diagnostic Cases
**Likely needs deeper research:**
- Are ASHRAE 140 diagnostic case specifications (195-470, 800-810) publicly available?
- What construction assemblies, HVAC schedules, internal loads are specified for these cases?
- How to obtain reference results from EnergyPlus, ESP-r, TRNSYS for diagnostic cases?

### Phase 5: Weather Data Refinement
**Likely needs deeper research:**
- What interpolation algorithm (linear, cubic spline, Hermite) is best for weather data?
- Should radiation be smoothed, or use raw sub-hourly values?
- Are high-resolution TMY datasets publicly available for Denver and other ASHRAE 140 locations?

### Phase 6: Statistical Validation Framework
**Standard patterns, unlikely to need research:**
- NMBE, CVRMSE, confidence interval formulas are standard statistics
- ASHRAE 140 statistical criteria are well-defined
- Recommended: Use ASHRAE Guideline 14 or 140 for statistical acceptance criteria

### Phase 7: Performance Optimization
**Standard patterns, unlikely to need research:**
- Profiling tools (criterion, flamegraph) are well-established
- Optimization patterns (caching, SIMD, lazy evaluation) are standard
- Recommended: Profile first, then optimize based on data

---

## Gaps to Address

### Areas where research was inconclusive:

1. **ASHRAE 140 Diagnostic Case Specifications**
   - Cases 195-470, 800-810 specifications not found in codebase
   - Likely behind ASHRAE paywall or in appendices not publicly indexed
   - **Recommendation:** Purchase ASHRAE Standard 140 document or request from research institution

2. **HVAC Equipment Efficiency Curve Coefficients**
   - Part-load efficiency curves for boilers, chillers, heat pumps not documented
   - Reference implementations (EnergyPlus, ESP-r) may have default curves
   - **Recommendation:** Review EnergyPlus source code (open source) for equipment model coefficients

3. **Weather Interpolation Best Practices**
   - No clear guidance on sub-hourly interpolation method (linear vs. cubic spline)
   - Radiation smoothing approach (moving average vs. no smoothing) unclear
   - **Recommendation:** Review EnergyPlus weather interpolation module for reference implementation

4. **High-Mass Thermal Mass Correction Strategy**
   - 6R2C evaluation showed no improvement, but root cause of annual energy error unclear
   - May need more sophisticated corrections than time-constant-based approach
   - **Recommendation:** Consult academic literature on thermal mass modeling in high-mass buildings

5. **Statistical Acceptance Criteria for ASHRAE 140**
   - NMBE, CVRMSE thresholds not specified in available documentation
   - Confidence interval requirements (95% vs. 99%) unclear
   - **Recommendation:** Review ASHRAE 140 appendices for statistical validation criteria

---

## Appendix: 5R1C vs 6R2C Decision Summary

**Decision (2026-03-13): Keep 5R1C as default for v0.3**

**Rationale:**
1. **No accuracy improvement on high-mass cases:** Both 5R1C and 6R2C predict 5.35 MWh heating vs reference 1.17-2.04 MWh (229-322% error) for Case 900
2. **Significant performance penalty:** 6R2C is 1.5-2x slower (~150-200ms vs ~100ms per config), throughput drops from 2,575 to ~1,200-1,500 configs/sec
3. **No breaking changes required:** Keeping 5R1C as default maintains backward compatibility and performance targets

**Evidence:**
- 11/11 6R2C unit tests pass (implementation is correct)
- ASHRAE 140 validation shows no accuracy improvement (docs/6R2C_DECISION.md)
- Performance benchmarks confirm 1.5-2x slowdown (benches/engine_bench.rs)
- 6R2C throughput ~1,200-1,500 configs/sec vs 5R1C ~2,575 configs/sec

**Recommendation for full ASHRAE 140 compliance:**
- Use 5R1C as default model
- Focus on thermal mass corrections (Phase 1) to address high-mass annual energy error
- Keep 6R2C as opt-in for future research or special cases
- Document that 6R2C does not improve ASHRAE 140 validation accuracy

---

## Metadata

**Research date:** 2026-03-13
**Valid until:** 2026-04-13 (30 days - stable domain, existing implementation well-documented)
**Research mode:** Ecosystem (architecture integration for ASHRAE 140 full compliance)
**Confidence sources:** Existing codebase (HIGH), Phase 12 validation (HIGH), Web search (unavailable - LOW)
