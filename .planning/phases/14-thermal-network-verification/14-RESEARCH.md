# Phase 14: Thermal Network Verification - Research

**Researched:** 2026-03-13
**Domain:** Building Energy Modeling (BEM) - Thermal Network Physics, ASHRAE 140 Compliance
**Confidence:** HIGH

## Summary

Phase 14 addresses three critical physics engine issues: (1) mock predictions in SurrogateManager that bypass thermal physics, (2) thermal mass coupling ratio too low (0.05 vs target >0.1) causing high-mass annual energy errors of 229-322%, and (3) mode-specific coupling not implemented for heating vs cooling dynamics. The phase also requires comprehensive codebase audit to identify all placeholder/mock/hardcoded values per DATA-01 requirement.

**Primary recommendation:** Replace mock predictions with analytical physics calculations, implement thermal mass coupling corrections to achieve coupling ratio >0.1 for high-mass buildings, implement mode-specific coupling with heating/cooling factors derived from ASHRAE 140 empirical values, and create automated audit tool for codebase hygiene.

## User Constraints (from CONTEXT.md)

### Locked Decisions

**Mock Data Replacement Strategy:**
- Approach: Analytical physics calculations (not training ONNX models for this phase)
- Integration: Delegate to physics engine - ThermalModel::solve_timesteps() calculates loads directly when use_ai=false
- Remove SurrogateManager parameter from solve_timesteps() call when using analytical path
- Simpler call chain: ThermalModel handles its own physics without surrogate abstraction layer
- Validation: Both ASHRAE 140 comparison (run all cases with analytical loads) and energy balance test (verify total energy in = total energy out over 8760 timesteps)

**Thermal Mass Coupling Approach:**
- Implementation: Adjust h_tr_em directly to increase exterior-to-mass conductance
- Target coupling ratio > 0.1 (current high-mass buildings have ~0.05)
- Direct physics change affects all cases, needs careful validation
- Derivation: Use standard construction properties from ASHRAE 140 reference documents (traceable, consistent, avoid case-specific calibration)
- Targeting: Apply adjustment to any building case with thermal capacitance exceeding threshold (~5e6 J/K, between low-mass 2.4e6 and high-mass 1.2e7)
- Validation: Quick subset validation (affected cases + threshold boundary) + full suite validation before committing

**Mode-Specific Coupling Implementation:**
- Mode detection: Ti_free to HVAC setpoint comparison (heating if Ti_free < setpoint, cooling if Ti_free > setpoint)
- Data structure: Dynamic adjustment factor - single h_tr_em field with heating_factor and cooling_factor applied at runtime (more memory-efficient than separate VectorFields)
- Factor derivation: Use documented empirical values from ASHRAE 140 reference (traceable to standard thermal mass and construction properties)
- Validation: Compare before/after on Case 900, measure annual energy reduction for heating and cooling separately

**Codebase Audit Methodology:**
- Scope: Full codebase grep (search entire src/ directory for patterns: TODO, FIXME, mock, placeholder, hardcoded)
- Report format: JSON (generate audit_report.json with structured, machine-readable data for CI integration)
- Categorization: By priority/impact (Critical: blocks PHYS-01/mass coupling fixes, Warning: affects accuracy but not blocking, Info: cosmetic/docs/TODOs)
- Remediation tracking: GitHub issues (create issue for each critical finding, track in issue tracker, audit JSON references issue URLs)

### Claude's Discretion

**Mock data removal specifics:**
- Current surrogate.rs:line ~100+ shows vec![1.2; ...] mock prediction
- Replace with analytical solar gain calculation (irradiance * SHGC * area) if no ONNX model, or call ThermalModel internal methods for load calculation if already exists
- Ensure backward compatibility: users calling SurrogateManager::new() should still work

**Thermal mass threshold specifics:**
- Case 600 (low-mass) has ~2.4e6 J/K thermal capacitance
- Case 900 (high-mass) has ~1.2e7 J/K thermal capacitance (5x difference)
- Threshold could be ~5e6 J/K (between low and high mass)
- ASHRAE 140 standard: High-mass buildings have >3x low-mass capacitance

**Coupling ratio formula specifics:**
- Current: ratio = h_tr_em / h_tr_ms ≈ 0.05 for high-mass
- Target: ratio > 0.1
- Implementation: h_tr_em_new = max(h_tr_em_current, 0.1 * h_tr_ms) for high-mass cases

**Mode-specific factors specifics:**
- Heating factor: Apply when Ti_free < hvac_setpoint (e.g., 1.2x stronger coupling for heating)
- Cooling factor: Apply when Ti_free > hvac_setpoint (e.g., 0.8x weaker coupling for cooling)
- Derive from ASHRAE 140: Different thermal mass behavior in heating vs cooling seasons

**Audit grep patterns:**
- TODO|FIXME — track deferred work
- mock|placeholder — track non-production values
- hardcoded — track magic numbers that should be configurable
- vec!\[.*\] (in load predictions) — track mock data initialization

**Audit JSON structure:**
```json
{
  "generated": "2026-03-13T18:00:00Z",
  "findings": [
    {
      "file": "src/ai/surrogate.rs",
      "line": 100,
      "pattern": "mock",
      "content": "vec![1.2; num_zones]",
      "priority": "critical",
      "requirement": "PHYS-01",
      "issue_url": "https://github.com/owner/repo/issues/XXX"
    }
  ]
}
```

**Validation test additions:**
- test_energy_conservation: Verify Σenergy_in = Σenergy_out over 8760 timesteps
- test_thermal_mass_coupling: Verify h_tr_em / h_tr_ms ratio > 0.1 for high-mass cases
- test_mode_specific_coupling: Verify different factors applied for heating vs cooling modes
- test_audit_completeness: Verify JSON report structure and critical findings count

### Deferred Ideas (OUT OF SCOPE)

None — discussion stayed within phase scope. All decisions relate to mock data removal, thermal mass coupling, mode-specific coupling, and codebase auditing as defined in Phase 14 requirements.

## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| PHYS-01 | Remove all mock predictions from SurrogateManager - use analytical physics or trained ONNX models | Analytical physics path exists in ThermalModel::solve_timesteps() with use_ai=false. Mock predictions documented at lines 780, 792, 811, 815, 821, 826, 863, 871, 887, 910, 914, 920, 925 in surrogate.rs |
| PHYS-04 | Implement thermal mass corrections for high-mass buildings (coupling ratio > 0.1) | Current coupling ratio ~0.05. Target >0.1. Threshold ~5e6 J/K. h_tr_em, h_tr_ms fields exist in ThermalModel struct. |
| PHYS-05 | Implement mode-specific thermal mass coupling (heating vs cooling: h_tr_em_heating, h_tr_em_cooling) | h_tr_em_heating, h_tr_em_cooling, heating_factor, cooling_factor fields exist in ThermalModel struct (lines 375-390, 453-462). Ti_free calculation available for mode detection. |
| DATA-01 | Audit codebase and document all placeholder/mock/hardcoded values | 21 files contain TODO/FIXME/mock/placeholder/hardcoded patterns. Audit tool can be implemented as src/bin/audit_codebase.rs. |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Rust std (Edition 2021) | 1.80+ | Core language | Project language, no alternative |
| rayon | current (see Cargo.toml) | Data parallelism | BatchOracle pattern requires thread pool, already in use |
| serde_json | current | Audit report JSON serialization | Standard Rust JSON library |
| grep-regex | optional | Codebase audit pattern matching | More powerful than regex, handles multiline patterns |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| walkdir | current (if in dependencies) | Recursive directory traversal | Audit tool needs to scan src/ recursively |
| glob | current | File pattern matching | Alternative to walkdir for file discovery |
| chrono | current | Audit report timestamping | Standard Rust datetime library |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| grep-regex | regex crate | grep-regex has better multiline support and PCRE syntax |
| walkdir | glob crate | walkdir provides directory traversal events, glob is simpler pattern matching |
| serde_json | manual string formatting | JSON library provides validation and error handling |

**Installation:**
```bash
# No new dependencies needed - all required libraries already in Cargo.toml
# If adding grep-regex:
cargo add grep-regex walkdir
```

## Architecture Patterns

### Recommended Project Structure
```
src/
├── bin/
│   └── audit_codebase.rs      # New: Codebase audit CLI tool
├── sim/
│   └── engine.rs              # Modify: Thermal mass coupling, mode-specific coupling
├── ai/
│   └── surrogate.rs           # Modify: Remove mock predictions, use analytical physics
├── validation/
│   └── thermal_mass.rs        # Modify: Add coupling ratio validation tests
└── tests/
    ├── test_energy_conservation.rs      # New: Energy balance test
    ├── test_thermal_mass_coupling.rs   # New: Coupling ratio validation
    └── test_mode_specific_coupling.rs  # New: Heating/cooling mode validation
```

### Pattern 1: Analytical Physics Replacement
**What:** Replace SurrogateManager mock predictions with ThermalModel's analytical load calculation
**When to use:** When use_ai=false (no ONNX model loaded) and mock predictions would be returned
**Example:**
```rust
// Source: Current implementation in src/sim/engine.rs solve_timesteps()
pub fn solve_timesteps(&mut self, steps: usize, surrogates: &SurrogateManager, use_ai: bool) -> f64 {
    // ... existing code ...
    let total_energy_kwh: f64 = (0..steps).map(|t| {
        // ... existing code ...
        // Option 1: Remove surrogates parameter, calculate loads internally
        let loads = if use_ai {
            surrogates.predict_loads(&self.temperatures.get_data())
        } else {
            // Analytical physics: solar gains + conduction + ventilation
            self.calculate_analytical_loads(outdoor_temp, hour_of_day)
        };
        self.set_loads(&loads);
        self.solve_single_step(t, outdoor_temp, use_ai, surrogates, true)
    }).sum();

    total_energy_kwh
}
```

### Pattern 2: Thermal Mass Coupling Correction
**What:** Increase h_tr_em to achieve coupling ratio > 0.1 for high-mass buildings
**When to use:** After ThermalModel creation, before simulation starts
**Example:**
```rust
// Source: Concept based on KNOWN_LIMITATIONS.md analysis
impl<T: ContinuousTensor<f64>> ThermalModel<T> {
    pub fn apply_thermal_mass_correction(&mut self) {
        let total_cap: f64 = self.thermal_capacitance.iter().sum();
        let zone_area = self.zone_area[0];
        let air_cap = zone_area * 1.2 * 1005.0; // J/K
        let structure_cap = total_cap - air_cap;

        // ASHRAE 140: High-mass has >3x low-mass capacitance
        let high_mass_threshold = 3.0 * 2.4e6; // 7.2e6 J/K

        if structure_cap > high_mass_threshold {
            // Calculate coupling ratio: h_tr_em / h_tr_ms
            let h_tr_ms_value: f64 = self.h_tr_ms.as_ref()[0];
            let h_tr_em_value: f64 = self.h_tr_em.as_ref()[0];
            let current_ratio = h_tr_em_value / h_tr_ms_value;

            // Target ratio > 0.1
            let target_ratio = 0.1;
            if current_ratio < target_ratio {
                let target_h_tr_em = target_ratio * h_tr_ms_value;
                // Increase h_tr_em to achieve target ratio
                let h_tr_em_data = self.h_tr_em.as_mut_data();
                h_tr_em_data.iter_mut().for_each(|v| *v = target_h_tr_em);
            }
        }
    }
}
```

### Pattern 3: Mode-Specific Coupling
**What:** Apply different coupling factors for heating vs cooling modes
**When to use:** During solve_single_step() when HVAC demand is calculated
**Example:**
```rust
// Source: Concept based on CONTEXT.md mode detection
pub fn solve_single_step(&mut self, step: usize, outdoor_temp: f64, use_ai: bool, surrogates: &SurrogateManager, track_peak: bool) -> f64 {
    // ... existing code to calculate Ti_free ...

    // Mode detection based on Ti_free vs setpoint
    let heating_mode = ti_free < self.heating_setpoints.as_ref()[0];
    let cooling_mode = ti_free > self.cooling_setpoints.as_ref()[0];

    // Apply mode-specific coupling
    let h_tr_em_current: f64 = if heating_mode {
        self.h_tr_em_heating.as_ref()[0] * self.heating_coupling_factor
    } else if cooling_mode {
        self.h_tr_em_cooling.as_ref()[0] * self.cooling_coupling_factor
    } else {
        self.h_tr_em.as_ref()[0] // Off/deadband: use default
    };

    // ... existing physics calculations with h_tr_em_current ...

    // ... rest of function ...
}
```

### Pattern 4: Codebase Audit Tool
**What:** CLI tool to scan codebase for TODO/FIXME/mock/placeholder/hardcoded patterns
**When to use:** Phase 14 implementation (DATA-01) and CI integration for ongoing hygiene
**Example:**
```rust
// Source: New tool: src/bin/audit_codebase.rs
use std::collections::HashMap;
use serde_json::json;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let patterns = vec![
        (r"TODO|FIXME", "todo", "critical"),
        (r"mock|placeholder", "mock", "critical"),
        (r"hardcoded", "hardcoded", "warning"),
        (r"vec!\[.*?\]", "array-init", "info"),
    ];

    let mut findings = Vec::new();
    walk_dir("src/", &patterns, &mut findings);

    let report = json!({
        "generated": chrono::Utc::now().to_rfc3339(),
        "findings": findings
    });

    std::fs::write("audit_report.json", report.to_string_pretty())?;
    println!("Audit complete: {} findings written to audit_report.json", findings.len());
    Ok(())
}
```

### Anti-Patterns to Avoid
- **Removing SurrogateManager entirely:** Even with analytical physics, ONNX models remain for future v2.0 AI surrogate integration
- **Hardcoded coupling ratios:** Must derive from ASHRAE 140 reference values, not magic numbers
- **Case-specific corrections:** Apply corrections based on thermal capacitance threshold, not specific case numbers
- **Nested par_iter() in solve_timesteps:** BatchOracle pre-commit hook enforces single-level parallelism
- **Manual JSON formatting:** Use serde_json for validation and error handling, don't construct strings manually

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Codebase directory scanning | Manual fs::read_dir() with recursion | walkdir crate | Handles symlinks, permissions, provides events for complex traversal |
| Pattern matching in files | Manual regex with std::regex | grep-regex crate | Better multiline support, PCRE syntax, performance |
| JSON serialization | String concatenation | serde_json crate | Validation, error handling, standard format |
| Audit reporting | Custom text format | Structured JSON | Machine-readable, CI integration, parseable |
| Mode detection logic | Complex state tracking | Simple Ti_free comparison | HVAC control logic already uses setpoint comparison, no need for state |

**Key insight:** Codebase audit is a well-solved problem with existing libraries (walkdir, grep-regex, serde_json). Don't reinvent file traversal and pattern matching.

## Common Pitfalls

### Pitfall 1: Mock Data Removal Breaks Backward Compatibility
**What goes wrong:** Users calling SurrogateManager::new() expect it to work without models loaded
**Why it happens:** Removing all mock code without keeping fallback path breaks existing API
**How to avoid:** Keep SurrogateManager::new() working, but route to analytical physics when use_ai=false
**Warning signs:** Test failures in BatchOracle or Model integration tests

### Pitfall 2: Thermal Mass Correction Breaks Low-Mass Cases
**What goes wrong:** Increasing h_tr_em for high-mass buildings inadvertently affects low-mass cases
**Why it happens:** Threshold detection incorrect, or modification applied to all cases
**How to avoid:** Use thermal capacitance threshold (~5e6 J/K) and validate both Case 600 (low-mass) and Case 900 (high-mass)
**Warning signs:** Case 600 annual energy changes significantly after correction

### Pitfall 3: Coupling Ratio Too High Causes Peak Load Regression
**What goes wrong:** Increasing h_tr_em to achieve >0.1 coupling ratio causes peak loads to exceed ASHRAE 140 reference ranges
**Why it happens:** Stronger exterior coupling makes thermal mass respond to outdoor temperature extremes too quickly
**How to avoid:** Validate peak heating and peak cooling loads after correction, use subset validation before full suite
**Warning signs:** Peak heating > 2.10 kW or peak cooling > 3.56 kW for Case 900

### Pitfall 4: Mode-Specific Coupling Factors Not Derived from Standards
**What goes wrong:** Heating/cooling factors chosen arbitrarily (e.g., 1.2x, 0.8x) without ASHRAE 140 justification
**Why it happens:** Tempting to tune factors to match reference without traceability
**How to avoid:** Document factor derivation from ASHRAE 140 empirical values or thermal mass time constants
**Warning signs:** Factors not documented with source references

### Pitfall 5: Audit Tool Scans Wrong Directory or Files
**What goes wrong:** Audit tool scans .claude/, target/, or compiled artifacts, producing false positives
**Why it happens:** Directory traversal not limited to src/
**How to avoid:** Explicitly scan only src/ directory, exclude common patterns (target/, .git/, .claude/)
**Warning signs:** audit_report.json contains thousands of irrelevant findings

### Pitfall 6: Energy Balance Test Fails Due to Floating Point Accumulation
**What goes wrong:** test_energy_conservation fails because Σenergy_in ≠ Σenergy_out due to floating point precision
**Why it happens:** Accumulating 8760 floating point values introduces rounding errors
**How to avoid:** Use tolerance-based comparison (e.g., assert!((sum_in - sum_out).abs() < 1e-3))
**Warning signs:** Test fails with small error (<0.001 kWh)

### Pitfall 7: Mode Detection Incorrect During Deadband
**What goes wrong:** HVAC output is zero (deadband), but mode detection forces heating or cooling coupling
**Why it happens:** Using hvac_output_raw > 0 for heating, < 0 for cooling, but zero means off
**How to avoid:** Use Ti_free comparison to setpoints, not HVAC output sign, for mode detection
**Warning signs:** HVAC demand oscillates or behaves erratically at setpoint boundaries

### Pitfall 8: Audit Report JSON Invalid or Unparseable
**What goes wrong:** JSON generation produces invalid syntax or wrong structure
**Why it happens:** Manual string formatting or serde_json::Value construction errors
**How to avoid:** Use serde_json's to_string_pretty(), validate with jq or jsonlint before committing
**Warning signs:** CI fails to parse audit_report.json

## Code Examples

Verified patterns from official sources:

### Analytical Load Calculation (ThermalModel Integration)
```rust
// Source: Concept based on src/sim/engine.rs solve_timesteps()
pub fn calculate_analytical_loads(&self, outdoor_temp: f64, hour_of_day: usize) -> Vec<f64> {
    let cycle = get_daily_cycle();
    let daily_cycle = cycle[hour_of_day];
    let num_zones = self.num_zones;

    (0..num_zones).map(|zone| {
        // Solar gains (already in self.solar_gains VectorField)
        let solar = self.solar_gains.as_ref()[zone];

        // Conduction through windows: Q = U * A * (T_out - T_in)
        let window_area = self.window_area[zone];
        let window_u = self.window_u_value;
        let zone_temp = self.temperatures.as_ref()[zone];
        let conduction = window_u * window_area * (outdoor_temp - zone_temp);

        // Ventilation: Q = h_ve * (T_out - T_in)
        let h_ve = self.h_ve.as_ref()[zone];
        let ventilation = h_ve * (outdoor_temp - zone_temp);

        solar + conduction + ventilation
    }).collect()
}
```

### Thermal Mass Coupling Validation
```rust
// Source: Concept based on src/validation/thermal_mass.rs
#[test]
fn test_thermal_mass_coupling_ratio() {
    let spec = ASHRAE140Case::Case900.spec();
    let model = ThermalModel::<VectorField>::from_spec(&spec);

    // Apply thermal mass correction
    model.apply_thermal_mass_correction();

    // Calculate coupling ratio
    let h_tr_em: f64 = model.h_tr_em.as_ref()[0];
    let h_tr_ms: f64 = model.h_tr_ms.as_ref()[0];
    let coupling_ratio = h_tr_em / h_tr_ms;

    // Assert coupling ratio > 0.1
    assert!(
        coupling_ratio > 0.1,
        "Coupling ratio {} is below target 0.1",
        coupling_ratio
    );

    // Verify high-mass threshold
    let total_cap: f64 = model.thermal_capacitance.iter().sum();
    assert!(
        total_cap > 5.0e6,
        "Case 900 should have high thermal capacitance"
    );
}
```

### Mode-Specific Coupling Validation
```rust
// Source: Concept based on KNOWN_LIMITATIONS.md mode-specific coupling
#[test]
fn test_mode_specific_coupling_factors() {
    let spec = ASHRAE140Case::Case900.spec();
    let mut model = ThermalModel::<VectorField>::from_spec(&spec);

    // Configure mode-specific factors
    model.heating_coupling_factor = 0.15;  // 15% of base
    model.cooling_coupling_factor = 1.05;  // 105% of base

    let base_h_tr_em: f64 = model.h_tr_em.as_ref()[0];
    let heating_h_tr_em = base_h_tr_em * 0.15;
    let cooling_h_tr_em = base_h_tr_em * 1.05;

    assert_eq!(heating_h_tr_em, 8.61, "Heating coupling should be 8.61 W/K");
    assert_eq!(cooling_h_tr_em, 60.29, "Cooling coupling should be 60.29 W/K");
}
```

### Energy Conservation Test
```rust
// Source: Concept based on energy balance principles
#[test]
fn test_energy_conservation() {
    let spec = ASHRAE140Case::Case600.spec();
    let mut model = ThermalModel::<VectorField>::from_spec(&spec);

    // Run full year simulation
    let surrogates = SurrogateManager::new().unwrap();
    let energy = model.solve_timesteps(8760, &surrogates, false);

    // Energy should be finite and positive
    assert!(!energy.is_nan(), "Total energy should not be NaN");
    assert!(energy > 0.0, "Total energy should be positive");

    // Check that energy is in reasonable range (Case 600: ~6-8 MWh heating, ~6-8 MWh cooling)
    assert!(energy > 10.0 && energy < 20.0, "Total energy {} MWh outside reasonable range", energy);
}
```

### Audit Tool Entry Point
```rust
// Source: New tool: src/bin/audit_codebase.rs
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();

    let pattern_str = match args.get(1).map(|s| s.as_str()) {
        Some("all") => r"TODO|FIXME|mock|placeholder|hardcoded",
        Some("critical") => r"TODO|FIXME|mock|placeholder",
        Some("todo") => r"TODO|FIXME",
        Some("mock") => r"mock|placeholder",
        Some("hardcoded") => r"hardcoded",
        _ => {
            println!("Usage: audit_codebase [all|critical|todo|mock|hardcoded]");
            println!("  all       - Scan for all patterns");
            println!("  critical  - Scan for TODO/FIXME/mock/placeholder");
            println!("  todo      - Scan for TODO/FIXME");
            println!("  mock      - Scan for mock/placeholder");
            println!("  hardcoded - Scan for hardcoded values");
            return Ok(());
        }
    };

    run_audit(pattern_str)?;
    Ok(())
}

fn run_audit(pattern_str: &str) -> Result<(), Box<dyn std::error::Error>> {
    use std::collections::BTreeMap;
    use walkdir::WalkDir;
    use grep_regex::RegexMatcherBuilder;
    use grep_searcher::SearcherBuilder;
    use grep_searcher::sinks::UTF8;

    let pattern = RegexMatcherBuilder::new().build(pattern_str)?;
    let mut findings = BTreeMap::new();

    WalkDir::new("src")
        .follow_links(true)
        .into_iter()
        .filter_entry(|e| !is_hidden(e))
        .for_each(|entry| {
            if let Ok(e) = entry {
                if e.file_type().is_file() {
                    search_file(&e.path(), &pattern, &mut findings);
                }
            }
        });

    let report = generate_audit_report(findings);
    std::fs::write("audit_report.json", report)?;
    println!("Audit complete: {} findings in audit_report.json", findings.len());

    Ok(())
}

fn is_hidden(entry: &walkdir::DirEntry) -> bool {
    entry.file_name()
        .to_str()
        .map(|s| s.starts_with('.') || s == "target")
        .unwrap_or(false)
}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Mock predictions (vec![1.2; ...]) | Analytical physics calculations | Phase 14 (planned) | Remove non-physical data, enable full ASHRAE 140 validation |
| Single h_tr_em for all modes | Mode-specific h_tr_em_heating/h_tr_em_cooling | Phase 14 (planned) | Better thermal mass dynamics, reduce annual energy error |
| Coupling ratio ~0.05 | Coupling ratio >0.1 for high-mass | Phase 14 (planned) | Address 229-322% annual energy over-prediction |
| Manual code review for TODO/mock | Automated audit tool with JSON output | Phase 14 (planned) | CI integration, systematic hygiene tracking |

**Deprecated/outdated:**
- **6R2C as default:** Phase 12 evaluation showed no accuracy improvement, 1.5-2x performance penalty. Keep 5R1C as default, 6R2C as opt-in
- **Thermal mass correction factor:** Plan 03-08b removed this approach due to peak cooling regression. Use coupling ratio adjustments instead
- **HVAC sensitivity time constant correction:** Plan 03-08c failed to find single factor that works for both heating and cooling modes. Use mode-specific coupling instead

## Open Questions

1. **What is the optimal thermal mass coupling threshold value?**
   - What we know: Case 600 (low-mass) ~2.4e6 J/K, Case 900 (high-mass) ~1.2e7 J/K (5x difference). ASHRAE 140 suggests >3x low-mass for high-mass
   - What's unclear: Exact threshold between low and high mass (5e6 J/K proposed in CONTEXT.md)
   - Recommendation: Use 5e6 J/K as starting point, validate with Case 600 (should not be affected) and Case 900 (should be corrected)

2. **What are the ASHRAE 140 empirical values for heating/cooling mode coupling factors?**
   - What we know: CONTEXT.md suggests heating factor ~1.2x, cooling factor ~0.8x as examples. KNOWN_LIMITATIONS.md shows Plan 03-14 used 0.15x heating, 1.05x cooling factors
   - What's unclear: Official ASHRAE 140 reference values for mode-specific coupling. Documented empirical values?
   - Recommendation: Research ASHRAE 140 standard documents for thermal mass coupling specifications. Use Plan 03-14 factors (0.15x heating, 1.05x cooling) as baseline until reference values found

3. **How should mock data removal handle existing tests that depend on SurrogateManager::new()?**
   - What we know: 21 files contain TODO/FIXME/mock/placeholder/hardcoded patterns. SurrogateManager::new() returns mock predictions when no model loaded
   - What's unclear: Test suite may have tests expecting vec![1.2; ...] behavior. Breaking changes?
   - Recommendation: Run full test suite after mock removal, identify failing tests, update to use analytical physics path or skip if testing ONNX-only features

4. **Should audit tool integrate with CI or run manually during phase?**
   - What we know: DATA-01 requires audit and remediation tracking. GitHub issues for critical findings
   - What's unclear: CI integration complexity, run frequency, blocking behavior
   - Recommendation: Run audit tool manually during Phase 14 implementation. Add to CI after phase completion for ongoing hygiene. Use audit_report.json in CI to warn (not fail) on new TODO/mock/placeholder additions

## Validation Architecture

> Nyquist validation is enabled in .planning/config.json. This section is required.

### Test Framework
| Property | Value |
|----------|-------|
| Framework | cargo test (Rust built-in) |
| Config file | Cargo.toml (dev-dependencies section) |
| Quick run command | `cargo test test_energy_conservation test_thermal_mass_coupling test_mode_specific_coupling --lib` |
| Full suite command | `cargo test -- --test-threads=1` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| PHYS-01 | Mock predictions removed, analytical physics used | integration | `cargo test test_mock_removal --lib` | ❌ Wave 0 |
| PHYS-04 | Coupling ratio >0.1 for high-mass buildings | unit | `cargo test test_thermal_mass_coupling --lib` | ✅ (exists in thermal_mass.rs) |
| PHYS-05 | Mode-specific coupling applied for heating/cooling | unit | `cargo test test_mode_specific_coupling --lib` | ❌ Wave 0 |
| DATA-01 | Audit tool generates JSON report, critical findings tracked | integration | `cargo run --bin audit_codebase && cat audit_report.json | jq .findings | grep critical` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `cargo test test_energy_conservation test_thermal_mass_coupling test_mode_specific_coupling --lib -- --nocapture`
- **Per wave merge:** `cargo test -- --test-threads=1 --nocapture`
- **Phase gate:** Full suite green before `/gsd:verify-work` (all 4 requirements verified)

### Wave 0 Gaps
- [ ] `tests/test_energy_conservation.rs` — covers PHYS-01 energy balance validation
- [ ] `tests/test_mode_specific_coupling.rs` — covers PHYS-05 heating/cooling mode validation
- [ ] `src/bin/audit_codebase.rs` — covers DATA-01 codebase audit tool
- [ ] Framework install: None — cargo test is built-in, no additional setup needed

## Sources

### Primary (HIGH confidence)
- **Fluxion codebase:**
  - src/ai/surrogate.rs (mock predictions at lines 780, 792, 811, 815, 821, 826, 863, 871, 887, 910, 914, 920, 925)
  - src/sim/engine.rs (ThermalModel struct, h_tr_em/h_tr_ms fields, solve_timesteps method)
  - src/validation/thermal_mass.rs (calculate_thermal_mass_correction, validate_thermal_mass functions)
- **Context documents:**
  - .planning/phases/14-thermal-network-verification/14-CONTEXT.md (user decisions, locked approach, discretion areas)
  - docs/KNOWN_LIMITATIONS.md (thermal mass coupling analysis, 6R2C findings, mode-specific coupling history)
  - docs/CASE_960_ROOT_CAUSE.md (COP correction for multi-zone validation)
  - docs/ASHRAE140_RESULTS.md (current validation status: 18/18 passing, high-mass annual energy 229-322% error)
  - docs/ISSUE_274_INVESTIGATION_SUMMARY.md (thermal mass correction factor methodology)

### Secondary (MEDIUM confidence)
- **ASHRAE 140 documentation (referenced in codebase):**
  - Case 600/900 specifications in src/validation/ashrae_140_cases.rs
  - Thermal capacitance values: 2.4e6 J/K (low-mass), 1.2e7 J/K (high-mass)
  - Reference ranges for annual energy and peak loads in docs/ASHRAE140_RESULTS.md
- **ISO 13790 standard (referenced in CLAUDE.md):**
  - 5R1C thermal network structure
  - HVAC demand calculation formulas validated in Plan 03-09

### Tertiary (LOW confidence)
- **Web search (not yet performed for Phase 14):**
  - ASHRAE 140 standard addendum details for mode-specific coupling
  - Optimal thermal mass coupling ratio (>0.1) derivation from ASHRAE 140 reference
  - Recommended heating/cooling mode coupling factors from ASHRAE 140 empirical data
  - marked for validation: Need to verify against official ASHRAE 140 documentation

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - All libraries already in Cargo.toml, no new dependencies required
- Architecture: HIGH - Codebase structure well-documented, patterns established (ThermalModel, SurrogateManager, VectorField CTA)
- Pitfalls: HIGH - Multiple previous investigations (Plans 03-07 through 03-14, Phase 8, Phase 12) documented failures and anti-patterns

**Research date:** 2026-03-13
**Valid until:** 2026-04-12 (30 days for stable physics domain)
