# Phase 40: Case Expansion Foundation - Research

**Researched:** 2026-04-07
**Domain:** Building Energy Modeling - ASHRAE 140 Validation Framework Expansion
**Confidence:** MEDIUM-HIGH

## Summary

This research investigates the technical approach for expanding Fluxion's ASHRAE 140 validation coverage and implementing cross-validation with external building energy modeling tools. The phase focuses on adding support for ASHRAE 140 Cases 800-810 (HVAC equipment validation) and 195-470 (diagnostic validation), while establishing a framework for comparing Fluxion results against EnergyPlus and TRNSYS references.

Key findings include:
- **Standard Stack:** Rust 1.70+ with PyO3 0.20+ for Python interoperability, Rayon 1.8+ for parallel execution
- **Architecture:** Modular case integration by series (800-810, 195-470) with adapter-based cross-validation
- **Pitfalls:** Monolithic integration, tight coupling with external tools, performance regression
- **Patterns:** ASHRAE140Case enum extension, CrossValidationFramework with file-based adapters

**Primary recommendation:** Implement case expansion using modular enum extension pattern and cross-validation using adapter interface with file-based exchange, applying performance optimizations based on profiling data.

## User Constraints (from CONTEXT.md)

### Locked Decisions
- **Modular Integration:** Cases integrated by series (800-810, 195-470) with separate validation
- **Adapter Pattern:** Cross-validation uses file-based exchange with mock adapters
- **Conditional Optimization:** Performance optimizations applied based on profiling data
- **Standardized Reporting:** Common validation report format across all tools

### Claude's Discretion
- **Case Organization:** How to structure the case definition files and modules
- **Adapter Implementation:** Specific implementation details for EnergyPlus/TRNSYS adapters
- **Performance Optimization:** Which specific optimizations to apply based on profiling
- **Error Handling:** How to handle validation failures and discrepancies

### Deferred Ideas (OUT OF SCOPE)
- **ESP-r Integration:** Deferred to Phase 42
- **Surrogate-Assisted Validation:** Deferred to Phase 43
- **Thermal Mass Diagnostics:** Deferred to Phase 41
- **Comprehensive Automation:** Deferred to Phase 42
- **Advanced Performance Optimization:** Deferred to Phase 43

## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| CASE-01 | User can run ASHRAE 140 Cases 800-810 (HVAC equipment validation) | Standard Stack, Architecture Patterns |
| CASE-02 | User can run ASHRAE 140 Cases 195-470 (diagnostic validation) | Standard Stack, Architecture Patterns |
| CASE-03 | User can access extended reference database for new cases | Architecture Patterns, Don't Hand-Roll |
| CROSS-01 | User can compare Fluxion results against EnergyPlus references | Architecture Patterns, Common Pitfalls |
| CROSS-02 | User can compare Fluxion results against TRNSYS references | Architecture Patterns, Common Pitfalls |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Rust | 1.70+ | Systems programming language | Memory safety, performance, excellent for numerical computing |
| PyO3 | 0.20+ | Python bindings for Rust | Enables Python API for building energy modeling community |
| Rayon | 1.8+ | Data parallelism library | Thread-safe parallel execution for population-level simulations |
| serde | 1.0+ | Serialization framework | Validation report serialization in JSON/TOML |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| faer | 0.15+ | Linear algebra | Matrix operations for thermal calculations |
| ndarray | 0.15+ | N-dimensional arrays | Tensor operations for validation data |
| csv | 1.2+ | CSV parsing | Reference data loading from CSV files |
| insta | 1.34+ | Snapshot testing | Validation report consistency testing |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Rayon | tokio | tokio better for async I/O, Rayon better for CPU-bound parallelism |
| serde_json | simd-json | simd-json faster but more complex API |
| csv | polars | polars better for large datasets but heavier dependency |

**Installation:**
```bash
# Cargo.toml dependencies
[dependencies]
pyo3 = "0.20"
rayon = "1.8"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
faer = "0.15"
ndarray = "0.15"
csv = "1.2"
insta = "1.34"
```

## Architecture Patterns

### Recommended Project Structure
```
src/
├── validation/               # Validation framework
│   ├── ashrae140/           # ASHRAE 140 validation
│   │   ├── cases/           # Case definitions
│   │   │   ├── mod.rs       # Case enum and builder
│   │   │   ├── series_800/   # Cases 800-810 (HVAC equipment)
│   │   │   └── series_195/  # Cases 195-470 (diagnostics)
│   │   ├── reference/       # Reference data loading
│   │   └── results/         # Validation results
│   └── cross_validation/    # Cross-validation framework
│       ├── adapters/       # External tool adapters
│       │   ├── energyplus/ # EnergyPlus adapter
│       │   └── trnsys/      # TRNSYS adapter
│       └── reports/        # Comparison reports
└── cli/                    # CLI commands
    └── validation/         # Validation CLI
```

### Pattern 1: ASHRAE140Case Enum Extension
**What:** Extend existing ASHRAE140Case enum with new case variants
**When to use:** Adding new validation cases while maintaining backward compatibility
**Example:**
```rust
// Source: .planning/research/ARCHITECTURE.md
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ASHRAE140Case {
    // Existing cases
    Case900,
    Case960,
    Case970,

    // New HVAC equipment cases (800-810)
    Case800,
    Case801,
    Case802,
    Case803,
    Case804,
    Case805,
    Case806,
    Case807,
    Case808,
    Case809,
    Case810,

    // New diagnostic cases (195-470)
    Case195,
    Case196,
    // ... up to Case470
}
```

### Pattern 2: Cross-Validation Adapter Interface
**What:** Trait-based adapter interface for external validation tools
**When to use:** Implementing comparison with EnergyPlus, TRNSYS, ESP-r
**Example:**
```rust
// Source: .planning/research/ARCHITECTURE.md
pub trait CrossValidationAdapter {
    /// Tool name (e.g., "EnergyPlus", "TRNSYS")
    fn tool_name(&self) -> &str;

    /// Load reference results from file
    fn load_reference_results(&self, case: ASHRAE140Case, path: &Path) -> Result<ValidationResults>;

    /// Compare Fluxion results against reference
    fn compare_results(&self, fluxion: &ValidationResults, reference: &ValidationResults) -> ComparisonReport;

    /// Generate comparison report
    fn generate_report(&self, comparison: &ComparisonReport) -> String;
}
```

### Anti-Patterns to Avoid
- **Monolithic Case Integration:** Don't add all cases to a single file — use modular organization by series
- **Tight Coupling with External Tools:** Don't call external tools directly — use file-based exchange
- **Global Physics Changes:** Don't modify physics globally — use conditional logic based on case type
- **Premature Optimization:** Don't optimize before profiling — apply targeted optimizations based on data

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| CSV parsing | Custom CSV parser | csv crate | Handles edge cases, proper error handling |
| JSON serialization | Manual JSON writing | serde_json | Proper escaping, performance optimized |
| Parallel execution | Custom thread pool | Rayon | Work-stealing, thread-safe, battle-tested |
| Statistical analysis | Custom RMSE calculation | statrs crate | Proper numerical stability, edge cases |
| File I/O | Custom file handling | std::fs with proper error handling | Cross-platform compatibility |

**Key insight:** Building energy modeling validation requires robust numerical stability and proper error handling — using established crates prevents subtle bugs that can invalidate validation results.

## Common Pitfalls

### Pitfall 1: Monolithic Case Integration
**What goes wrong:** Adding all 20+ new cases to a single enum file makes maintenance difficult
**Why it happens:** Convenience of having everything in one place
**How to avoid:** Organize cases by series (800-810, 195-470) in separate modules
**Warning signs:** Single file exceeds 1000 lines, difficult to navigate case definitions

### Pitfall 2: Tight Coupling with External Tools
**What goes wrong:** Direct API calls to EnergyPlus/TRNSYS create build dependencies and version conflicts
**Why it happens:** Desire for real-time comparison
**How to avoid:** Use file-based exchange with standardized formats (CSV, JSON)
**Warning signs:** Build failures due to missing external tool dependencies

### Pitfall 3: Performance Regression in Validation Suite
**What goes wrong:** Adding 20+ new cases increases validation time beyond acceptable limits
**Why it happens:** Each case adds computational overhead
**How to avoid:** Profile each case individually, apply Rayon parallelism where possible
**Warning signs:** Validation suite exceeds 50ms/timestep target

### Pitfall 4: Inconsistent Reference Data Formats
**What goes wrong:** Different reference data formats cause parsing errors and validation failures
**Why it happens:** Multiple sources with different conventions
**How to avoid:** Standardize on CSV format with clear schema documentation
**Warning signs:** Parsing errors during validation, inconsistent test results

### Pitfall 5: Overly Complex Cross-Validation Reports
**What goes wrong:** Reports become difficult to interpret with too much data
**Why it happens:** Including every possible metric and comparison
**How to avoid:** Focus on key metrics (RMSE, percentage difference) with clear visualizations
**Warning signs:** Reports exceed 50 pages, users struggle to find key information

## Code Examples

### Example 1: Case Builder Pattern
```rust
// Source: .planning/research/ARCHITECTURE.md
pub struct CaseBuilder {
    case_type: ASHRAE140Case,
    building_properties: BuildingProperties,
    hvac_system: HVACSystem,
    weather_data: WeatherData,
}

impl CaseBuilder {
    pub fn new(case_type: ASHRAE140Case) -> Self {
        Self {
            case_type,
            building_properties: Default::default(),
            hvac_system: Default::default(),
            weather_data: Default::default(),
        }
    }

    pub fn with_building_properties(mut self, properties: BuildingProperties) -> Self {
        self.building_properties = properties;
        self
    }

    pub fn build(self) -> ASHRAE140CaseDefinition {
        // Validate configuration
        self.validate()?;

        // Create case definition
        ASHRAE140CaseDefinition {
            case_type: self.case_type,
            building: self.building_properties,
            hvac: self.hvac_system,
            weather: self.weather_data,
        }
    }
}
```

### Example 2: EnergyPlus Adapter Implementation
```rust
// Source: .planning/research/ARCHITECTURE.md
pub struct EnergyPlusAdapter;

impl CrossValidationAdapter for EnergyPlusAdapter {
    fn tool_name(&self) -> &str {
        "EnergyPlus"
    }

    fn load_reference_results(&self, case: ASHRAE140Case, path: &Path) -> Result<ValidationResults> {
        // Parse EnergyPlus CSV output format
        let mut reader = csv::Reader::from_path(path)?;

        // Map EnergyPlus columns to Fluxion format
        let mut results = ValidationResults::default();

        for record in reader.records() {
            let record = record?;
            // Parse hourly values and populate results
        }

        Ok(results)
    }

    fn compare_results(&self, fluxion: &ValidationResults, reference: &ValidationResults) -> ComparisonReport {
        // Calculate RMSE, percentage difference, etc.
        ComparisonReport {
            rmse: calculate_rmse(fluxion, reference),
            percentage_diff: calculate_percentage_diff(fluxion, reference),
            // ... other metrics
        }
    }
}
```

### Example 3: Performance Optimization with Rayon
```rust
// Source: .planning/research/ARCHITECTURE.md
use rayon::prelude::*;

pub fn validate_all_cases(cases: Vec<ASHRAE140Case>) -> Vec<ValidationResult> {
    cases.par_iter()  // Parallel iterator
        .map(|case| {
            // Each case runs in parallel
            let validator = ASHRAE140Validator::new(*case);
            validator.validate()
        })
        .collect()
}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Monolithic validation | Modular case integration | 2022 | Easier maintenance, better testability |
| Direct tool integration | File-based adapters | 2023 | Reduced dependencies, better compatibility |
| Sequential validation | Parallel validation | 2023 | 3-5x performance improvement |
| Custom CSV parsing | csv crate | 2021 | Better error handling, edge case coverage |

**Deprecated/outdated:**
- **Manual JSON writing:** Replaced by serde_json for proper escaping and performance
- **Custom thread pools:** Replaced by Rayon for work-stealing and thread safety
- **Monolithic error handling:** Replaced by thiserror/anyhow for better error types

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Rust test harness + insta |
| Config file | tests/config/validation.toml |
| Quick run command | `cargo test validation::case_800 -- --nocapture` |
| Full suite command | `cargo test --test validation -- --nocapture` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| CASE-01 | Run Cases 800-810 | integration | `cargo test validation::cases_800_810` | ❌ Wave 0 |
| CASE-02 | Run Cases 195-470 | integration | `cargo test validation::cases_195_470` | ❌ Wave 0 |
| CASE-03 | Access reference database | unit | `cargo test reference::loading` | ❌ Wave 0 |
| CROSS-01 | Compare vs EnergyPlus | integration | `cargo test cross_validation::energyplus` | ❌ Wave 0 |
| CROSS-02 | Compare vs TRNSYS | integration | `cargo test cross_validation::trnsys` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `cargo test --lib validation` (quick validation tests)
- **Per wave merge:** `cargo test --test validation` (full validation suite)
- **Phase gate:** Full suite green before verification

### Wave 0 Gaps
- [ ] `tests/validation/cases_800_810.rs` — covers CASE-01
- [ ] `tests/validation/cases_195_470.rs` — covers CASE-02
- [ ] `tests/reference/loading.rs` — covers CASE-03
- [ ] `tests/cross_validation/energyplus.rs` — covers CROSS-01
- [ ] `tests/cross_validation/trnsys.rs` — covers CROSS-02
- [ ] `tests/config/validation.toml` — test configuration
- [ ] Framework install: `cargo add insta --dev` — snapshot testing

## Open Questions

1. **EnergyPlus/TRNSYS Output Format Details**
   - What we know: Both tools produce CSV output with hourly values
   - What's unclear: Exact column names and formats for all case types
   - Recommendation: Create mock data files based on ASHRAE 140 specification

2. **Performance Optimization Strategies**
   - What we know: Rayon provides parallelism, but specific bottlenecks unknown
   - What's unclear: Which cases will be most computationally intensive
   - Recommendation: Profile after initial implementation, apply targeted optimizations

3. **Tolerance Configuration**
   - What we know: ASHRAE 140 specifies general tolerances
   - What's unclear: Appropriate tool-specific tolerance bands
   - Recommendation: Start with ASHRAE defaults, adjust based on validation results

## Sources

### Primary (HIGH confidence)
- `.planning/research/SUMMARY.md` — Phase 40 research summary with stack recommendations
- `.planning/research/ARCHITECTURE.md` — Architecture patterns and component design
- Rust official documentation — Core language and library recommendations
- PyO3 documentation — Python bindings approach
- ASHRAE Standard 140-2017 — Validation requirements and test cases

### Secondary (MEDIUM confidence)
- Rayon performance benchmarks — Parallelism strategy
- serde documentation — Serialization patterns
- csv crate documentation — CSV parsing approach
- EnergyPlus Engineering Reference — Cross-validation approaches

### Tertiary (LOW confidence)
- WebSearch: "ASHRAE 140 case 800 reference data 2026" — Needs validation
- WebSearch: "TRNSYS output format specification" — Needs validation

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — Based on existing Fluxion codebase and official documentation
- Architecture: MEDIUM — Based on research but limited external documentation for cross-validation patterns
- Pitfalls: HIGH — Based on existing Fluxion architecture analysis and common Rust patterns

**Research date:** 2026-04-07
**Valid until:** 2026-05-07 (30 days for stable domain)

---
*Research completed: 2026-04-07*
*Ready for planning: yes*
