# Phase 18: Diagnostic Cases - Research

**Researched:** 2026-03-14
**Domain:** Building Energy Modeling (BEM) Validation - ASHRAE 140 Diagnostic Cases
**Confidence:** MEDIUM

## Summary

Phase 18 focuses on implementing comprehensive diagnostic case coverage for ASHRAE 140 validation. The phase builds on Phase 15 (HVAC equipment) and Phase 17 (internal loads) to validate the full engine with Cases 195-470 (in-depth diagnostics), Cases 800-810 (HVAC equipment), and non-residential diagnostic variants.

The research reveals that Fluxion already has a robust validation framework with ASHRAE140Validator, multi-reference database support, and existing diagnostic test infrastructure (Case 195 solid conduction, Cases 800-810 stubs). The phase requires extending the existing ASHRAE140Case enum with diagnostic case variants, implementing case specifications, populating the multi-reference database with reference ranges, and creating consolidated validation tests.

**Primary recommendation:** Use the hybrid structure from CONTEXT.md (consolidated validation logic in `tests/ashrae_140/diagnostics.rs` with public case spec functions in `src/validation/ashrae_140_cases.rs`) to balance maintainability with accessibility.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
**Case File Organization:**
- Hybrid structure with consolidated logic and public case specs
- Create `tests/ashrae_140/diagnostics.rs` module containing consolidated validation logic for all diagnostic cases
- Keep case specification functions public in `src/validation/ashrae_140_cases.rs`
- Each diagnostic case (195-470, 800-810, variants) has its own spec function
- Case spec data loaded from `docs/ashrae_140_references.json` (multi-reference DB)

**Case Specification Format:**
- Multi-reference database (docs/ashrae_140_references.json)
- Case parameters loaded from external JSON files via `MultiReferenceDB`
- Case specs reference: `docs/ashrae_140_references.json` (Phase 7 multi-reference integration)
- Diagnostic cases query multi-ref DB for reference ranges (when ASHRAE 140 official specs available)
- Fallback to sensible defaults when official specs not available
- Follows existing pattern from Phase 17 (building_profiles.json)

**Validation Strategy:**
- Auto-discovery for baseline cases (600-960): `fluxion validate` auto-runs complete baseline suite
- Targeted re-run for affected case ranges when diagnostics added: `fluxion validate-case 195-470` or `fluxion validate-case 800-810`
- Diagnostic-aware validation: Validator tracks which cases have been added and only re-runs affected ranges
- Full re-run option available: `fluxion validate --full` for comprehensive validation

**CLI Integration:**
- Auto-discovery for baseline Cases 600-960: `fluxion validate` automatically discovers and runs all diagnostic cases in ranges
- Explicit invocation for new diagnostic ranges: `fluxion validate-case` allows running specific cases (e.g., `fluxion validate-case 800`)
- Subcommands: `fluxion validate 195-470`, `fluxion validate 800-810` for specific diagnostic ranges
- Consistent with existing pattern: Matches `fluxion validate-600` pattern (baseline validation)

### Claude's Discretion
- Exact JSON schema for multi-reference DB structure (field names, types)
- Diagnostic module organization details (module structure, helper function naming)
- Smart validation re-run trigger thresholds (how many cases before auto-discovery re-runs baseline)
- CLI subcommand design (validate-case flags, range argument format)
- Test framework patterns (property tests, integration tests, validation structure)

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope. All decisions relate to case organization, specification format, validation strategy, and CLI integration as defined in Phase 18 requirements (DIAG-01 through DIAG-05).
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| DIAG-01 | Implement ASHRAE 140 Cases 195-470 (in-depth diagnostics) | Existing ASHRAE140Case enum and CaseSpec pattern; Case 195 already implemented as reference; multi-reference DB pattern established |
| DIAG-02 | Implement ASHRAE 140 Cases 800-810 (HVAC equipment) | HVAC equipment complete from Phase 15; Cases 800-810 test stubs exist in `tests/ashrae_140_cases_800_810.rs` |
| DIAG-03 | Implement non-residential cases from ASHRAE 140 | Building profiles pattern from Phase 17 provides template; can extend ASHRAE140Case with commercial building variants |
| DIAG-04 | Implement solid conduction test variants | Case 195 solid conduction already implemented; can extend with variants (different construction, no loads/solar) |
| DIAG-05 | Implement solar gain diagnostic variants | Case 195 pattern (no windows) provides zero-solar baseline; can add variants with different SHGC/albedo values |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Rust Cargo Test Framework | Built-in | Unit/integration testing | Standard Rust testing with `cargo test` |
| Serde | ^1.0 | JSON serialization/deserialization | De facto standard for JSON in Rust ecosystem |
| Rayon | ^1.0 | Parallel test execution | Already used in codebase; efficient parallel validation |
| Assert Macros | Built-in | Test assertions | Standard Rust assertion macros |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| MultiReferenceDB | Custom (Phase 7) | Load ASHRAE 140 reference ranges from JSON | For all diagnostic case validation against multi-program references |
| ASHRAE140Validator | Custom (Phase 5+) | Validation framework with tolerance checking | For all ASHRAE 140 validation logic |
| DiagnosticCollector | Custom (Phase 5) | Detailed diagnostic output (hourly data, energy breakdown) | For comprehensive diagnostic reporting |
| ThermalModel | Custom (Core) | Physics engine simulation | For all case simulation logic |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| MultiReferenceDB | Hardcoded reference constants | MultiReferenceDB provides version-controlled, external JSON; hardcoded requires recompilation for updates |
| ASHRAE140Validator | Custom validation loop | ASHRAE140Validator provides standardized tolerances, multi-program comparison; custom would duplicate logic |
| Consolidated diagnostics.rs | Separate test files per case | Consolidated module reduces file count and centralizes helper functions; separate files provide isolation at cost of duplication |

**Installation:**
```bash
# All dependencies are already in Cargo.toml
cargo build --release  # Build with optimizations for validation performance
cargo test  # Run all tests including diagnostic cases
```

## Architecture Patterns

### Recommended Project Structure
```
tests/ashrae_140/
├── diagnostics.rs              # NEW: Consolidated validation logic for diagnostic cases
│   ├── Helper functions for running cases 195-470
│   ├── Helper functions for running cases 800-810
│   ├── Helper functions for solid conduction variants
│   ├── Helper functions for solar gain variants
│   └── Integration tests for all diagnostic ranges
├── ashrae_140_case_195_470.rs    # NEW: Test for Cases 195-470 range
├── ashrae_140_case_800_810.rs    # EXISTING: Test for Cases 800-810 range (extend)
├── ashrae_140_case_195_solid_conduction.rs  # EXISTING: Test for solid conduction variant
├── ashrae_140_case_600_960.rs    # NEW: Integration test for baseline + diagnostics
└── ... (existing test files)

src/validation/ashrae_140_cases.rs
├── Extend ASHRAE140Case enum with:
│   ├── Case196 through Case470 (in-depth diagnostics)
│   ├── Case800, Case810 (HVAC equipment)
│   ├── Case800Series through Case810Series (equipment variants)
│   └── NonResidential variants (office, retail, etc.)
├── Add spec() methods for all new cases
├── Add number() methods for all new cases
├── Add description() methods for all new cases
└── Implement CaseBuilder methods for new cases

docs/ashrae_140_references.json
├── Extend with cases "196" through "470"
├── Extend with cases "800", "801", ..., "810"
├── Add reference ranges for EnergyPlus, ESP-r, TRNSYS
└── Add diagnostic variant specifications (solid conduction, solar gain)
```

### Pattern 1: ASHRAE140Case Enum Extension
**What:** Extend existing ASHRAE140Case enum with diagnostic case variants
**When to use:** Adding new ASHRAE 140 diagnostic cases (195-470, 800-810)
**Example:**
```rust
// Source: /home/alex/Projects/fluxion/src/validation/ashrae_140_cases.rs
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ASHRAE140Case {
    // ... existing cases (600 series, 900 series, 960, 195) ...

    // Diagnostic cases 196-470 (in-depth diagnostics)
    Case196, Case197, ..., Case470,

    // HVAC equipment cases 800-810
    Case800, Case801, ..., Case810,

    // Non-residential variants
    Office, Retail, Hospital, ...
}

impl ASHRAE140Case {
    pub fn spec(&self) -> CaseSpec {
        match self {
            // ... existing cases ...
            ASHRAE140Case::Case196 => CaseBuilder::case_196(),
            // ... more cases ...
            ASHRAE140Case::Case800 => CaseBuilder::case_800(),
            ASHRAE140Case::Case810 => CaseBuilder::case_810(),
            ASHRAE140Case::Office => CaseBuilder::office_building(),
        }
    }

    pub fn number(&self) -> &'static str {
        match self {
            // ... existing cases ...
            ASHRAE140Case::Case196 => "196",
            ASHRAE140Case::Case800 => "800",
            ASHRAE140Case::Office => "OFFICE",
        }
    }
}
```

### Pattern 2: MultiReferenceDB Extension
**What:** Extend `docs/ashrae_140_references.json` with diagnostic case reference ranges
**When to use:** Adding ASHRAE 140 reference data for diagnostic cases
**Example:**
```json
// Source: /home/alex/Projects/fluxion/docs/ashrae_140_references.json
{
  "version": "2025-01",
  "source": "ASHRAE 140",
  "cases": {
    "600": { ... },  // existing
    "196": {
      "annual_heating": {
        "EnergyPlus": {"min": 4.0, "max": 5.0},
        "ESP-r": {"min": 3.8, "max": 5.2},
        "TRNSYS": {"min": 3.9, "max": 5.1}
      },
      "annual_cooling": {
        "EnergyPlus": {"min": 6.0, "max": 8.0}
      },
      "peak_heating": { ... },
      "peak_cooling": { ... }
    },
    "800": {
      "annual_heating": {
        "EnergyPlus": {"min": 8.0, "max": 10.0}
      }
      // ... HVAC equipment metrics
    }
  }
}
```

### Pattern 3: Consolidated Validation Module
**What:** Create `tests/ashrae_140/diagnostics.rs` with helper functions shared across diagnostic ranges
**When to use:** Running multiple diagnostic cases with shared validation logic
**Example:**
```rust
// Source: Pattern from Phase 17 internal loads and existing test files
//! Consolidated validation logic for ASHRAE 140 diagnostic cases

use fluxion::validation::ASHRAE140Case;
use fluxion::validation::ASHRAE140Validator;

/// Validates a range of diagnostic cases and returns pass/fail summary
fn validate_diagnostic_range(
    start: u32,
    end: u32,
    validator: &mut ASHRAE140Validator,
) -> DiagnosticRangeResult {
    let mut results = Vec::new();

    for case_num in start..=end {
        let case_id = case_num.to_string();
        let result = validator.validate_case(&case_id);
        results.push((case_id, result));
    }

    DiagnosticRangeResult {
        range: format!("{}-{}", start, end),
        total_cases: results.len(),
        passed: results.iter().filter(|(_, r)| r.passed).count(),
        results,
    }
}

/// Helper: Run Cases 195-470 diagnostic suite
pub fn run_cases_195_470() -> DiagnosticRangeResult {
    let mut validator = ASHRAE140Validator::new();
    validate_diagnostic_range(195, 470, &mut validator)
}

/// Helper: Run Cases 800-810 HVAC equipment suite
pub fn run_cases_800_810() -> DiagnosticRangeResult {
    let mut validator = ASHRAE140Validator::new();
    validate_diagnostic_range(800, 810, &mut validator)
}
```

### Pattern 4: Test Stub to Full Implementation
**What:** Replace placeholder assertions with full ASHRAE 140 reference validation
**When to use:** Implementing Cases 800-810 with proper reference data
**Example:**
```rust
// Source: /home/alex/Projects/fluxion/tests/ashrae_140_cases_800_810.rs
#[test]
fn test_ashrae_800() {
    // TODO: Implement after ASHRAE 140 Case 800 specifications are researched
    // For now, create a simple heat pump simulation with realistic parameters

    // EXISTING STUB:
    let mut model = ThermalModel::<VectorField>::new(1);
    model.heating_setpoint = 20.0;
    model.cooling_setpoint = 27.0;

    // Set up heat pump equipment
    let heatpump = HeatPump::new(
        "HP-800".to_string(),
        12000.0, // 12kW heating
        10000.0, // 10kW cooling
        3.5,     // COP 3.5
        3.0,     // EER 3.0
    );

    model.hvac_equipment = Some(AnyEquipment::HeatPump(heatpump));

    let surrogates = SurrogateManager::new().expect("Failed to create SurrogateManager");
    let total_energy = model.solve_timesteps(8760, &surrogates, false, None, None, None);

    // PLACEHOLDER ASSERTIONS:
    assert!(total_energy > 0.0);
    assert!(total_energy.is_finite());

    // FULL IMPLEMENTATION (after ASHRAE 140 specs available):
    // let spec = ASHRAE140Case::Case800.spec();
    // let mut model = ThermalModel::<VectorField>::from_spec(&spec);
    // let result = simulate_year(model);
    // assert_within_reference!(result.annual_heating_mwh, HEATING_MIN, HEATING_MAX);
    // assert_within_reference!(result.annual_cooling_mwh, COOLING_MIN, COOLING_MAX);
}
```

### Anti-Patterns to Avoid
- **Hardcoding reference constants in test files**: Use MultiReferenceDB for version-controlled reference data
- **Creating duplicate validation logic**: Consolidate shared validation in `tests/ashrae_140/diagnostics.rs` helper functions
- **Mixing diagnostic cases with baseline tests**: Keep diagnostic cases in separate files/ranges for targeted re-runs
- **Ignoring multi-program reference ranges**: Always validate against EnergyPlus, ESP-r, TRNSYS ranges when available
- **Forgetting to extend ASHRAE140Case enum**: All new cases must be added to the enum and implement `spec()`, `number()`, `description()`

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Reference data management | Custom JSON parsing logic | MultiReferenceDB (Phase 7) | Already handles JSON loading, versioning, program-specific ranges |
| Validation tolerance checking | Custom tolerance logic | ASHRAE140Validator (Phase 5+) | Standardized tolerances (±15% annual, ±10% monthly, ±1°C free-float) with pass/warning/fail |
| Diagnostic output collection | Custom logging logic | DiagnosticCollector (Phase 5) | Provides hourly data, energy breakdown, peak timing, temperature profiles |
| Case specification | Manual parameter setting | CaseBuilder pattern | Consistent case creation with validation, follows existing pattern |
| Parallel test execution | Custom threading | Rayon par_iter() | Already used in codebase, efficient parallel validation |

**Key insight:** Fluxion's validation framework (Phase 5-7) provides all necessary infrastructure for diagnostic case implementation. Don't duplicate ASHRAE140Validator, DiagnosticCollector, or MultiReferenceDB logic.

## Common Pitfalls

### Pitfall 1: Missing ASHRAE 140 Specifications for Cases 195-470, 800-810
**What goes wrong:** Attempting to implement diagnostic cases without official ASHRAE 140 specifications leads to incorrect reference ranges and validation failures
**Why it happens:** ASHRAE 140 standard is paywalled; diagnostic case specifications (195-470, 800-810) are not publicly available in full detail
**How to avoid:** Use EnergyPlus/ESP-r/TRNSYS open-source implementations as reference sources; fall back to sensible defaults when official specs unavailable; document assumptions clearly in case spec functions
**Warning signs:** Large energy deviations (>20%) from reference programs, inconsistent results across programs, no documentation for parameter choices

### Pitfall 2: Incorrect MultiReferenceDB JSON Structure
**What goes wrong:** JSON parsing errors when loading reference data; cases not found in database
**Why it happens:** MultiReferenceDB expects specific JSON structure (version, source, cases map, CaseRefs structure with HashMaps for each metric)
**How to avoid:** Follow existing `docs/ashrae_140_references.json` pattern; use ProgramRange struct for min/max values; validate JSON with `MultiReferenceDB::from_file()` before committing
**Warning signs:** `Failed to load reference data` warnings, `cannot find case` errors, `key not found` panics

### Pitfall 3: Not Extending ASHRAE140Case Enum Completely
**What goes wrong:** Compilation errors when using new cases; `match` arms missing for new variants
**Why it happens:** Adding case spec functions without updating enum variants or forgetting to add `spec()`/`number()`/`description()` match arms
**How to avoid:** Always update enum, add all three method implementations (spec, number, description), add to CaseBuilder if applicable
**Warning signs:** `non-exhaustive patterns` compilation error, `unreachable pattern` warnings

### Pitfall 4: Test Stubs Not Replaced with Full Implementation
**What goes wrong:** Cases 800-810 remain TODO stubs with placeholder assertions; no actual ASHRAE 140 validation
**Why it happens:** Starting with stubs but not replacing with full implementation after specifications are available
**How to avoid:** Track stubs with TODO markers; prioritize replacing stubs with full implementations; use reference ranges from MultiReferenceDB
**Warning signs:** Tests only assert `total_energy > 0.0`, no reference range comparisons, TODO comments in test files

### Pitfall 5: Diagnostic Cases Not Integrated with Validation CLI
**What goes wrong:** `fluxion validate` doesn't run diagnostic cases; users must manually call test functions
**Why it happens:** Adding test files without updating ASHRAE140Validator to include new cases in validation runs
**How to avoid:** Extend `validate_analytical_engine()` to include diagnostic case ranges; add CLI subcommands for targeted re-runs; update validation reports
**Warning signs:** `fluxion validate` output doesn't include diagnostic cases, no way to run specific diagnostic ranges

### Pitfall 6: Conflicting Case Numbers with Baseline Cases
**What goes wrong:** Case 195 conflicts with baseline numbering (600-960); confusion about which cases are diagnostic vs baseline
**Why it happens:** Diagnostic cases use different numbering scheme (195-470, 800-810) but baseline cases are 600-960
**How to avoid:** Clearly document case ranges: baseline (600-960), diagnostics (195-470, 800-810); use descriptive names in ASHRAE140Case enum
**Warning signs:** Users confused about which cases to run, overlapping case numbers in documentation

## Code Examples

Verified patterns from existing codebase:

### Adding New Case to ASHRAE140Case Enum
```rust
// Source: /home/alex/Projects/fluxion/src/validation/ashrae_140_cases.rs (lines 219-250)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ASHRAE140Case {
    // ... existing cases ...
    Case195,  // Example: Solid conduction case
}

impl ASHRAE140Case {
    pub fn spec(&self) -> CaseSpec {
        match self {
            // ... existing cases ...
            ASHRAE140Case::Case195 => CaseBuilder::case_195_solid_conduction(),
        }
    }

    pub fn number(&self) -> String {
        match self {
            // ... existing cases ...
            ASHRAE140Case::Case195 => "195".to_string(),
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            // ... existing cases ...
            ASHRAE140Case::Case195 => "Solid conduction test case",
        }
    }
}
```

### Loading MultiReferenceDB
```rust
// Source: /home/alex/Projects/fluxion/src/validation/multi_reference.rs (lines 34-39)
impl MultiReferenceDB {
    pub fn from_file(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let content = fs::read_to_string(path)?;
        let db: MultiReferenceDB = serde_json::from_str(&content)?;
        Ok(db)
    }
}

// Usage in ASHRAE140Validator (lines 76-91):
let default_multi_ref_path = Path::new("docs/ashrae_140_references.json");
if default_multi_ref_path.exists() {
    match MultiReferenceDB::from_file(default_multi_ref_path) {
        Ok(db) => {
            validator.multi_ref = Some(db);
        }
        Err(e) => {
            eprintln!("Warning: Failed to load multi-reference data from {}: {}",
                      default_multi_ref_path.display(), e);
        }
    }
}
```

### Running Validation with Diagnostics
```rust
// Source: /home/alex/Projects/fluxion/tests/ashrae_140_case_600.rs (lines 30-59)
#[test]
fn test_case_600_baseline_ashrae_140_reference() {
    let mut model = Case600Model::new();
    let result = model.simulate_year();

    println!("\n=== ASHRAE 140 Case 600 Results ===");
    println!(
        "Annual Heating: {:.2} MWh (reference: {:.2}-{:.2} MWh)",
        result.annual_heating_mwh,
        reference::ANNUAL_HEATING_MIN,
        reference::ANNUAL_HEATING_MAX
    );
    println!("=== End ===\n");

    // Assertions against reference ranges
    assert!(
        result.annual_heating_mwh + result.annual_cooling_mwh > 0.0,
        "Total HVAC energy should be positive, got {} MWh",
        result.annual_heating_mwh + result.annual_cooling_mwh
    );
}
```

### Diagnostic Test Pattern (Case 195 Solid Conduction)
```rust
// Source: /home/alex/Projects/fluxion/tests/ashrae_140_case_195_solid_conduction.rs (lines 30-57)
fn simulate_case_195() -> (f64, f64, f64) {
    let spec = ASHRAE140Case::Case195.spec();
    let mut model = ThermalModel::<VectorField>::from_spec(&spec);
    let weather = DenverTmyWeather::new();

    let mut annual_heating_joules = 0.0;
    let mut annual_cooling_joules = 0.0;
    let mut peak_heating_watts: f64 = 0.0;

    for step in 0..8760 {
        let weather_data = weather.get_hourly_data(step).unwrap();
        let hvac_kwh = model.step_physics(step, weather_data.dry_bulb_temp);

        if hvac_kwh > 0.0 {
            annual_heating_joules += hvac_kwh * 3.6e6;
            peak_heating_watts = peak_heating_watts.max(hvac_kwh * 1000.0);
        } else {
            annual_cooling_joules += (-hvac_kwh) * 3.6e6;
        }
    }

    (
        annual_heating_joules / 3.6e9,  // Convert to MWh
        annual_cooling_joules / 3.6e9,
        peak_heating_watts / 1000.0,
    )
}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Hardcoded reference constants in test files | MultiReferenceDB with external JSON | Phase 7 | Version-controlled reference data, no recompilation needed for updates |
| Manual validation loop with custom tolerances | ASHRAE140Validator with standardized tolerances | Phase 5 | Consistent validation across all cases, pass/warning/fail criteria |
| Separate test files per case with duplicated logic | Consolidated diagnostics.rs with helper functions | Phase 18 (planned) | Reduced duplication, easier maintenance, targeted re-runs |
| No CLI integration for validation | `fluxion validate --all`, `fluxion validate-case 195-470` | Phase 18 (planned) | User-friendly validation, selective testing without manual test invocation |

**Deprecated/outdated:**
- Hardcoded reference ranges in test files: Replaced by MultiReferenceDB (Phase 7)
- Manual tolerance checking: Replaced by ASHRAE140Validator (Phase 5+)
- No diagnostic output: Replaced by DiagnosticCollector with hourly data, energy breakdown (Phase 5)

## Open Questions

1. **ASHRAE 140 Specifications for Cases 195-470**
   - What we know: These cases exist and are "in-depth diagnostics" for testing specific components
   - What's unclear: Full specifications (geometry, HVAC, loads) are paywalled in ASHRAE 140 standard
   - Recommendation: Use EnergyPlus/ESP-r/TRNSYS open-source implementations as reference; document assumptions; fall back to sensible defaults

2. **ASHRAE 140 Specifications for Cases 800-810**
   - What we know: These cases test HVAC equipment performance and control strategies; test stubs exist with placeholder assertions
   - What's unclear: Exact equipment specifications (capacity, efficiency curves, control logic)
   - Recommendation: Research EnergyPlus HVAC equipment test files; use Phase 15 equipment implementation as baseline; document assumptions

3. **Non-Residential Case Specifications**
   - What we know: ASHRAE 140 includes non-residential variants; building_profiles.json pattern from Phase 17 provides template
   - What's unclear: Which non-residential building types are included (office, retail, hospital, etc.)
   - Recommendation: Start with office building type using Phase 17 profile; extend to other types as specifications become available

4. **Solid Conduction and Solar Gain Variants**
   - What we know: Case 195 solid conduction exists as baseline; solar gain variants should test different SHGC/albedo values
   - What's unclear: Exact variant specifications (which SHGC values, which albedo values)
   - Recommendation: Use Case 195 as zero-solar baseline; add variants with SHGC=0.3, 0.6, 0.9 and albedo=0.1, 0.5, 0.9

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Cargo Test Framework (built-in) |
| Config file | None — uses Cargo.toml configuration |
| Quick run command | `cargo test test_ashrae_140_case_800 --lib` (single case) |
| Full suite command | `cargo test test_ashrae --lib` (all ASHRAE 140 cases) |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| DIAG-01 | Cases 195-470 produce validation results | integration | `cargo test test_ashrae_140_case_195_470 --lib` | ❌ Wave 0 |
| DIAG-02 | Cases 800-810 validate equipment efficiency | integration | `cargo test test_ashrae_140_case_800_810 --lib` | ⚠️ Partial (stubs exist) |
| DIAG-03 | Non-residential cases extend validation | integration | `cargo test test_ashrae_140_non_residential --lib` | ❌ Wave 0 |
| DIAG-04 | Solid conduction variants expose edge cases | integration | `cargo test test_ashrae_140_solid_conduction_variants --lib` | ⚠️ Partial (Case 195 exists) |
| DIAG-05 | Solar gain variants validate specific components | integration | `cargo test test_ashrae_140_solar_gain_variants --lib` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `cargo test test_ashrae_140 --lib` (quick smoke test of ASHRAE 140 cases)
- **Per wave merge:** `cargo test --lib` (full test suite including all diagnostic cases)
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/ashrae_140_case_195_470.rs` — covers DIAG-01
- [ ] `tests/ashrae_140_case_non_residential.rs` — covers DIAG-03
- [ ] `tests/ashrae_140_solid_conduction_variants.rs` — covers DIAG-04 (Case 195 exists but variants don't)
- [ ] `tests/ashrae_140_solar_gain_variants.rs` — covers DIAG-05
- [ ] `tests/ashrae_140/diagnostics.rs` — consolidated validation logic module
- [ ] Framework install: None — Cargo test framework is built-in, no installation needed

## Sources

### Primary (HIGH confidence)
- [Fluxion source code](/home/alex/Projects/fluxion) - Examined existing validation framework, ASHRAE140Case enum, MultiReferenceDB, ASHRAE140Validator, DiagnosticCollector
- [Case 195 solid conduction test](/home/alex/Projects/fluxion/tests/ashrae_140_case_195_solid_conduction.rs) - Reference implementation for diagnostic case pattern
- [Cases 800-810 test stubs](/home/alex/Projects/fluxion/tests/ashrae_140_cases_800_810.rs) - Existing placeholder implementations showing what needs to be completed
- [MultiReferenceDB implementation](/home/alex/Projects/fluxion/src/validation/multi_reference.rs) - Pattern for loading ASHRAE 140 reference data from JSON
- [ASHRAE140Validator implementation](/home/alex/Projects/fluxion/src/validation/ashrae_140_validator.rs) - Validation framework with tolerance checking
- [Building profiles JSON](/home/alex/Projects/fluxion/data/building_profiles.json) - Phase 17 pattern for external JSON configuration

### Secondary (MEDIUM confidence)
- [ASHRAE 140 documentation](https://www.ashrae.org/technical-resources/bookstore/standard-140) - ASHRAE 140 standard (paywalled, specifications not publicly available)
- [Phase 17 internal loads](/home/alex/Projects/fluxion/.planning/phases/17-internal-loads/) - Building profiles pattern, JSON configuration approach
- [Phase 15 HVAC equipment](/home/alex/Projects/fluxion/.planning/phases/15-hvac-equipment-modeling/) - Equipment implementation that Cases 800-810 should validate

### Tertiary (LOW confidence)
- Web search for ASHRAE 140 diagnostic case specifications - No results found (likely due to paywall protection)
- EnergyPlus/ESP-r/TRNSYS open-source implementations - Not yet examined (potential source for reference data)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - Cargo test framework is built-in Rust standard; all dependencies already in codebase
- Architecture: HIGH - Examined existing ASHRAE140Case enum, MultiReferenceDB, ASHRAE140Validator patterns
- Pitfalls: MEDIUM - Some pitfalls are based on common Rust/JSON errors, but ASHRAE 140 specification unavailability adds uncertainty

**Research date:** 2026-03-14
**Valid until:** 2026-04-13 (30 days - codebase patterns stable, ASHRAE 140 specifications unlikely to change)
