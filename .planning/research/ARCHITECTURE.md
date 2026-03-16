# Architecture Research

**Domain:** ASHRAE 140 Validation Framework for Building Energy Modeling (BEM) Engines
**Researched:** 2026-03-08
**Confidence:** MEDIUM

## Standard Architecture

### System Overview

ASHRAE 140 validation frameworks in BEM engines typically follow a layered architecture with clear separation between test case definitions, simulation execution, result comparison, and reporting.

```
┌─────────────────────────────────────────────────────────────┐
│                   Test Orchestration Layer                │
├─────────────────────────────────────────────────────────────┤
│  ┌────────────────┐  ┌────────────────┐  ┌───────────┐  │
│  │ Test Runner   │  │ CI Pipeline   │  │ CLI Interface│  │
│  │ (cargo test)  │  │ (GitHub Actions)│  │ (fluxion CLI)│  │
│  └────────┬───────┘  └───────┬───────┘  └─────┬─────┘  │
├───────────┴───────────────────┴───────────────┴─────────┤
│                   Validation Engine Layer                │
├─────────────────────────────────────────────────────────────┤
│  ┌────────────────────────────────────────────────────┐   │
│  │          ASHRAE140Validator                      │   │
│  │  - validate_analytical_engine()                  │   │
│  │  - validate_with_diagnostics()                   │   │
│  │  - DiagnosticCollector integration                 │   │
│  └────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────┤
│                   Test Case Definitions                  │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────┐  │
│  │ Case Specs   │  │ Benchmark    │  │ Case        │  │
│  │ (geometry,   │  │ Data        │  │ Builders    │  │
│  │  materials,  │  │ (ref values) │  │ (custom     │  │
│  │  HVAC,       │  │              │  │  configs)   │  │
│  │  shading)    │  │              │  │             │  │
│  └──────────────┘  └──────────────┘  └─────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                   Physics Simulation Layer                │
├─────────────────────────────────────────────────────────────┤
│  ┌────────────────────────────────────────────────────┐   │
│  │              ThermalModel (5R1C)                  │   │
│  │  - solve_timesteps()                             │   │
│  │  - apply_parameters()                            │   │
│  │  - IdealHVACController                          │   │
│  └────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────┤
│                   Result Processing & Reporting           │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────┐  │
│  │ Comparison   │  │ Report       │  │ Export      │  │
│  │ Engine       │  │ Generation   │  │ (Markdown,  │  │
│  │ (tolerance   │  │ (summary,    │  │  JSON, CSV)│  │
│  │  checking)   │  │  detailed)   │  │             │  │
│  └──────────────┘  └──────────────┘  └─────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility | Typical Implementation |
|-----------|----------------|------------------------|
| **Test Runner** | Executes validation test suite, orchestrates test execution | Rust's `cargo test` framework, test modules in `tests/` directory |
| **Validator** | Core validation logic, coordinates test execution and result collection | `ASHRAE140Validator` struct with validation methods |
| **Case Specifications** | Defines building geometry, materials, HVAC, weather, and control strategies | `CaseSpec` struct with builder pattern for customization |
| **Benchmark Data** | Reference values from EnergyPlus, ESP-r, TRNSYS, DOE2 | `BenchmarkData` struct with min/max ranges per metric |
| **Physics Engine** | Simulates building thermal behavior using 5R1C thermal network | `ThermalModel` with ISO 13790-compliant 5R1C implementation |
| **Diagnostic Collector** | Captures detailed simulation data for debugging | `DiagnosticCollector` with configurable output (hourly, energy breakdown, peak timing) |
| **Comparison Engine** | Validates simulation results against reference ranges | Tolerance-based comparison with Pass/Warning/Fail status |
| **Report Generator** | Produces human-readable validation summaries | Markdown, JSON, CSV export formats |
| **CI Pipeline** | Automated validation on every commit/PR | GitHub Actions workflow with threshold checks |

## Recommended Project Structure

```
src/
├── validation/                    # Validation framework core
│   ├── mod.rs                    # Public API exports
│   ├── ashrae_140_validator.rs   # Main validator orchestration
│   ├── ashrae_140_cases.rs      # Case specifications (600/900/FF series)
│   ├── benchmark.rs              # Reference data from EnergyPlus/ESP-r/TRNSYS
│   ├── diagnostic.rs             # Diagnostic output and debugging tools
│   ├── physics_validator.rs      # Physics law validation (energy balance, temp bounds)
│   ├── cross_validator.rs        # Surrogate vs analytical comparison
│   ├── report.rs                # Validation report generation
│   ├── fdd.rs                   # Fault detection and diagnostics
│   ├── ml_data_collector.rs     # ML training data collection
│   ├── thermal_mass.rs           # Thermal mass validation helpers
│   └── ashrae_140/             # Per-case implementations
│       ├── mod.rs
│       └── case_600.rs         # Example: Case-specific setup
├── sim/
│   ├── engine.rs                # ThermalModel (5R1C physics engine)
│   └── construction.rs          # Building assembly definitions
├── physics/
│   └── cta.rs                  # Continuous Tensor Abstraction
├── weather/
│   └── denver.rs               # Denver TMY2 weather data
└── lib.rs                      # PyO3 bindings (BatchOracle, Model)

tests/                          # Integration tests
├── ashrae_140_validation.rs     # Main validation test suite
├── ashrae_140_diagnostic_test.rs  # Diagnostic-specific tests
├── ashrae_140_integration.rs   # Integration tests for specific cases
└── test_case_195_*.rs          # Per-case test files

.github/workflows/
└── ashrae_140_validation.yml   # CI/CD pipeline

tools/data_gen/
├── ashrae_140_generator.py      # Generate test case data
└── test_ashrae_140_generator.py

docs/
├── ASHRAE140_VALIDATION.md      # Validation overview
├── ASHRAE140_RESULTS.md        # Current validation results
├── ASHRAE140_MILESTONES.md     # Progress tracking
└── ASHRAE_140_*.md           # Additional documentation
```

### Structure Rationale

- **`src/validation/`**: Centralized validation framework with clear separation of concerns (cases, benchmarks, reporting, diagnostics)
- **`src/sim/engine.rs`**: Physics engine is independent of validation framework—can be used by both validation and production code
- **`tests/`**: Integration tests for validation ensure test suite execution is correct
- **`src/validation/ashrae_140/`**: Per-case implementation files allow for case-specific logic without cluttering main validator
- **`tools/data_gen/`**: Python scripts for generating test data leverage Python's data manipulation capabilities
- **`.github/workflows/`**: CI automation ensures validation runs on every commit/PR

## Architectural Patterns

### Pattern 1: Builder Pattern for Test Cases

**What:** Enables flexible construction of test case specifications with optional parameters

**When to use:** When you need to create many similar test cases with variations (e.g., different window orientations, shading configurations)

**Trade-offs:**
- **Pros:** Clean API, readable configuration, easy to add new variants
- **Cons:** More boilerplate than direct struct initialization

**Example:**
```rust
use fluxion::validation::ashrae_140_cases::{ASHRAE140Case, CaseBuilder};

// Use predefined case
let spec = ASHRAE140Case::Case600.spec();

// Build custom case with builder
let custom_spec = CaseBuilder::new()
    .low_mass_construction()
    .with_dimensions(8.0, 6.0, 2.7)
    .with_south_window(12.0)
    .with_hvac_setpoints(20.0, 27.0)
    .build()
    .unwrap();
```

### Pattern 2: Diagnostic Collector

**What:** Captures detailed simulation data during execution for post-hoc analysis

**When to use:** When debugging validation failures or understanding simulation behavior

**Trade-offs:**
- **Pros:** Flexible output configuration, can enable/disable granular data collection, minimal performance impact when disabled
- **Cons:** Additional memory overhead when enabled, adds complexity to simulation loop

**Example:**
```rust
use fluxion::validation::diagnostic::DiagnosticConfig;

// Configure diagnostics
let config = DiagnosticConfig {
    enabled: true,
    output_hourly: true,
    hourly_output_path: Some("hourly_output.csv".to_string()),
    output_energy_breakdown: true,
    output_peak_timing: true,
    output_temperature_profiles: true,
    verbose: true,
};

let validator = ASHRAE140Validator::with_diagnostics(config);
```

### Pattern 3: Tolerance-Based Validation

**What:** Compares simulation results against reference ranges with configurable tolerance bands

**When to use:** For all ASHRAE 140 validation—standard practice in BEM engines

**Trade-offs:**
- **Pros:** Matches ASHRAE 140 methodology, clear pass/fail criteria, accounts for reference program variability
- **Cons:** Requires calibration of tolerance bands, may mask small systematic errors

**Example:**
```rust
pub enum ValidationStatus {
    Pass,    // Within 5% of reference range
    Warning, // Within reference range but >2% deviation
    Fail,    // Outside 5% tolerance band
}

// Validation logic
let within_range = value >= ref_min && value <= ref_max;
let deviation = ((value - ref_midpoint) / ref_midpoint).abs();

let status = if deviation < 0.05 {
    ValidationStatus::Pass
} else if within_range {
    ValidationStatus::Warning
} else {
    ValidationStatus::Fail
};
```

## Data Flow

### Request Flow

```
User/CI Command (fluxion validate --all or cargo test)
    ↓
ASHRAE140Validator::validate_analytical_engine()
    ↓
For each test case:
    ├─→ Load CaseSpec from ASHRAE140Case enum
    ├─→ Load BenchmarkData (reference ranges)
    ├─→ Configure ThermalModel from CaseSpec
    ├─→ Solve with model.solve_timesteps(8760, surrogates, use_ai=false)
    ├─→ Collect results (annual heating/cooling, peak loads)
    ├─→ Compare against BenchmarkData with tolerance checking
    └─→ Store ValidationResult
    ↓
Generate BenchmarkReport (summary statistics, per-case results)
    ↓
Export to Markdown/JSON/CSV
    ↓
Return report (CI: check pass rate threshold)
```

### Diagnostic Data Flow

```
Validation with Diagnostics Enabled
    ↓
DiagnosticCollector initialized with DiagnosticConfig
    ↓
During simulation:
    ├─→ Hourly temperatures collected
    ├─→ Hourly loads collected
    ├─→ Energy breakdown accumulated
    └─→ Peak timing tracked
    ↓
DiagnosticReport generated:
    ├─→ HourlyData (8760 points per metric)
    ├─→ EnergyBreakdown (heating, cooling, solar, internal)
    ├─→ PeakTiming (when peaks occur)
    └─→ TemperatureProfile (min/max over simulation)
    ↓
Export to CSV files (if configured)
    ↓
Validation report includes diagnostic summary
```

### Key Data Flows

1. **Case Specification → Model Configuration:** `CaseSpec` (geometry, materials, HVAC, weather) → `ThermalModel::new()` with parameters set via `apply_parameters()`
2. **Simulation → Results:** `ThermalModel::solve_timesteps()` → cumulative energy consumption, peak loads, temperature traces
3. **Results → Comparison:** Simulation values compared to `BenchmarkData` ranges → `ValidationStatus` (Pass/Warning/Fail)
4. **All Cases → Report:** `Vec<ValidationResult>` → `BenchmarkReport` with statistics (MAE, pass rate, max deviation)
5. **Report → Export:** `BenchmarkReport::to_markdown()`, `to_json()`, `to_csv()` → files for CI/PR comments

## Scaling Considerations

| Scale | Architecture Adjustments |
|-------|--------------------------|
| 1-10 cases | Single-threaded execution is fine, run all tests sequentially |
| 10-100 cases | Parallel test execution with `cargo test --test-threads=N`, collect results in memory |
| 100+ cases | Consider test case grouping, incremental validation (only run changed cases), parallel CI jobs |

### Scaling Priorities

1. **First bottleneck:** Test execution time (8760 hours simulated per case)
   - **Fix:** Parallel test execution, cache simulation results, skip unchanged cases
2. **Second bottleneck:** Diagnostic data collection (writing hourly CSV files)
   - **Fix:** Optional diagnostic output, binary serialization instead of CSV, batch writes

## Anti-Patterns

### Anti-Pattern 1: Hardcoded Reference Values

**What people do:** Embed reference values directly in test assertions
```rust
assert_eq!(heating_mwh, 6.5); // Bad: Magic number, not in source of truth
```

**Why it's wrong:** Reference values should be centralized in `benchmark.rs`, making it easy to update when ASHRAE releases new reference data

**Do this instead:**
```rust
let benchmark = get_benchmark_data("600");
assert!(heating_mwh >= benchmark.annual_heating_min);
assert!(heating_mwh <= benchmark.annual_heating_max);
```

### Anti-Pattern 2: Tight Coupling Between Test Cases and Physics Engine

**What people do:** Put case-specific logic directly in `ThermalModel`
```rust
pub fn solve_case_600(&self) { ... } // Bad: Physics engine knows about test cases
```

**Why it's wrong:** Violates separation of concerns—physics engine should be independent of validation framework

**Do this instead:**
```rust
// In validation code
let spec = ASHRAE140Case::Case600.spec();
let mut model = ThermalModel::new(spec.num_zones);
model.apply_parameters(&spec.params);
model.solve_timesteps(8760, &surrogates, false);
```

### Anti-Pattern 3: Missing Diagnostic Data

**What people do:** Only capture aggregate results (annual energy, peak loads) without hourly data

**Why it's wrong:** Impossible to debug why a test case failed—can't see temperature profiles, peak timing, or energy breakdowns

**Do this instead:**
```rust
let config = DiagnosticConfig::full();
let validator = ASHRAE140Validator::with_diagnostics(config);
let report = validator.validate_analytical_engine();
// Generates hourly_output.csv, energy_breakdown.csv, etc.
```

### Anti-Pattern 4: Ignoring Free-Floating Cases

**What people do:** Focus only on HVAC-controlled cases, skip free-floating variants

**Why it's wrong:** Free-floating cases validate thermal envelope physics without HVAC complications—critical for catching envelope heat transfer bugs

**Do this instead:**
```rust
// Include both controlled and free-floating variants
let cases = vec![
    ASHRAE140Case::Case600,  // HVAC-controlled
    ASHRAE140Case::Case600FF, // Free-floating
];
```

### Anti-Pattern 5: Skipping High-Mass Cases

**What people do:** Only validate low-mass (600 series) cases, skip high-mass (900 series)

**Why it's wrong:** High-mass cases test thermal mass dynamics, which is where 5R1C models often fail

**Do this instead:**
```rust
// Validate both low and high mass cases
let all_cases = [
    // Low mass
    "600", "610", "620", "630", "640", "650",
    // High mass
    "900", "910", "920", "930", "940", "950",
];
```

## Integration Points

### External Services

| Service | Integration Pattern | Notes |
|---------|---------------------|-------|
| **GitHub Actions CI** | Workflow runs `cargo test --test ashrae_140_validation`, extracts results with regex, generates Markdown report | Check pass rate threshold (currently 12.5%) |
| **ASHRAE Standard 140** | Reference data from EnergyPlus, ESP-r, TRNSYS, DOE2 in `benchmark.rs` | Tolerance bands: ±15% annual energy, ±10% monthly energy |

### Internal Boundaries

| Boundary | Communication | Notes |
|----------|---------------|-------|
| **Validator ↔ ThermalModel** | Direct method calls (`solve_timesteps()`) | Validator configures model, runs simulation, collects results |
| **Validator ↔ DiagnosticCollector** | Event-driven data collection | Simulation passes data to collector, collector aggregates into reports |
| **Validator ↔ Report Generator** | Structured data (`BenchmarkReport`) | Report formats data for export (Markdown/JSON/CSV) |
| **Test Cases ↔ Case Specs** | Enum variant → `CaseSpec` | Each test case has a predefined specification that can be customized |

## Component Boundaries for Fluxion

### Layered Architecture

```
┌─────────────────────────────────────────────────────────┐
│         PyO3 Bindings Layer (lib.rs)                │
│  - BatchOracle: High-throughput optimization          │
│  - Model: Single-building detailed analysis          │
└─────────────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────────┐
│         Validation Layer (validation/)                 │
│  - ASHRAE140Validator: Test orchestration             │
│  - Case Specs: Test case definitions                  │
│  - Benchmark Data: Reference values                   │
│  - Diagnostic Collector: Debug data capture           │
│  - Report Generator: Result formatting               │
└─────────────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────────┐
│         Physics Engine Layer (sim/engine.rs)          │
│  - ThermalModel: ISO 13790 5R1C thermal network     │
│  - IdealHVACController: HVAC logic                   │
│  - Construction: Building assemblies                  │
└─────────────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────────┐
│         Mathematical Layer (physics/cta.rs)           │
│  - Continuous Tensor Abstraction: VectorField ops      │
└─────────────────────────────────────────────────────────┘
```

### Build Order Implications

1. **Foundational Layer:** `physics/cta.rs` → Required by physics engine
2. **Physics Layer:** `sim/engine.rs` → Depends on CTA, used by validation
3. **Validation Data:** `validation/benchmark.rs` → Reference values, no dependencies
4. **Validation Core:** `validation/ashrae_140_cases.rs` → Depends on physics
5. **Validator:** `validation/ashrae_140_validator.rs` → Depends on cases, benchmarks, diagnostic
6. **Tests:** `tests/ashrae_140_validation.rs` → Depends on validator
7. **CI Pipeline:** `.github/workflows/ashrae_140_validation.yml` → Depends on tests

**Critical dependency:** Cannot build validation without physics engine working correctly—validation failures may indicate physics bugs, not just validation framework issues.

## Sources

- [Fluxion codebase analysis](https://github.com/your-org/fluxion) (HIGH confidence - direct code inspection)
- [ASHRAE Standard 140 methodology](https://www.ashrae.org/technical-resources/bookstore/standard-140) (MEDIUM confidence - general knowledge, web search failed)
- [Building Energy Simulation validation practices](https://energy.gov/eere/buildings/building-energy-modeling) (LOW confidence - web search failed, based on general knowledge)

---
*Architecture research for: ASHRAE 140 Validation Framework in Building Energy Modeling Engines*
*Researched: 2026-03-08*
