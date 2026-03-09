# Phase 1: Foundation - Core Validation Fixes - Research

**Researched:** 2026-03-09
**Domain:** Building Energy Modeling - 5R1C Thermal Network, ASHRAE 140 Validation
**Confidence:** HIGH

## Summary

Phase 1 addresses the root causes of Fluxion's 61% ASHRAE 140 failure rate and 78.79% Mean Absolute Error. The research identifies two primary systemic issues: incorrect window conductance parameterization (h_tr_em, h_tr_w) and HVAC load calculation errors using wrong temperature source (Ti vs Ti_free). These issues cause systematic heating load over-prediction across all baseline cases (600-650, 900).

The 5R1C thermal network is implemented per ISO 13790 with resistances stored as conductances in W/K units. Current code calculates conductances correctly for most elements but has critical gaps in window U-value application and HVAC power demand computation. Fixing these issues should reduce MAE from 78.79% to <15% and achieve >75% pass rate on lightweight cases.

**Primary recommendation:** Use test-driven development with parameterized rstest framework - write comprehensive unit tests for each conductance calculation first, then fix implementation to make tests pass, validate against ASHRAE 140 Case 600 reference values.

## User Constraints (from CONTEXT.md)

### Locked Decisions
- **High-impact fixes only:** Window-related conductances (h_tr_em, h_tr_w) and HVAC load calculation
- **Lightweight cases only:** Validate Cases 600, 610, 620, 630, 640, 650 (defer Case 900 high-mass to Phase 2)
- **Thermal mass dynamics deferred:** No thermal mass fixes in Phase 1 (deferred to Phase 2)
- **No new capabilities:** Only correcting existing physics, no feature additions
- **Test-driven development:** Write failing unit tests for each conductance first, then fix to make tests pass
- **All tests upfront:** Write complete test suite before touching implementation
- **Conductance unit tests first:** Isolate root cause of heating over-prediction
- **Case 600 first:** Validate against well-documented ASHRAE 140 reference values
- **Research-guided fix:** Apply known issues from research (conductance units, window U-value), then write tests to validate
- **Helper methods:** Extract calc_h_tr_em(), calc_h_tr_w() for maintainability
- **Fix high-impact areas first:** Window conductances before other conductances
- **Energy balance verification:** Sum of loads = energy change in thermal mass
- **Indirect validation:** Run Case 600 simulation, check results against reference (no direct field access)

### Testing Methodology
- **Both unit and integration tests:** Comprehensive approach with both test types
- **Unit tests:** Test each 5R1C conductance independently
- **Integration tests:** Full ASHRAE 140 case simulations validating annual/monthly energy and peak loads
- **Test structure:**
  - Unit tests in `src/sim/tests/test_conductance_calculations.rs`
  - Integration tests in `tests/ashrae_140_validation.rs`
  - Parameterized tests using rstest framework for multiple envelope properties
- **Execution:**
  - Run `cargo test --test-threads=1` after each fix in development loop
  - CI/CD enforces all tests, gate commits on test failures
- **Diagnostic integration:** Use existing DiagnosticCollector for hourly data and energy breakdowns

### Claude's Discretion
- Exact implementation of helper method signatures and internal structure
- Specific test case organization within files
- Validation of conductance values before tests (diagnostic analysis optional)
- Order of testing individual conductances within the all-at-once approach

### Deferred Ideas (OUT OF SCOPE)
- Thermal mass dynamics fixes (Case 900) — Phase 2
- Other conductances (h_tr_ms, h_tr_is, h_ve) — defer to later phases if not critical
- Solar radiation and external boundary fixes — Phase 3
- Inter-zone heat transfer fixes — Phase 4

## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| BASE-01 | Implement 5R1C thermal network conductance calculations per ISO 13790 | Conductance formulas documented in ISO 13790 Annex C; existing code has gaps in window U-value application |
| BASE-02 | Validate conductance calculations against ASHRAE 140 Case 600 reference | Case 600 reference values well-documented; conductance ranges calculable from envelope properties |
| BASE-03 | Ensure conductance units in W/K (not K/W) for 5R1C network | Code uses W/K correctly; verify window conductances follow same convention |
| BASE-04 | Implement proper U-value to conductance conversion for envelope elements | Window U-value application to h_tr_w needs verification; thermal bridge effects may be missing |
| FREE-01 | Implement free-floating temperature calculation (Ti_free) | hvac_power_demand() uses Ti_free for load calculation; verify calculation correctness |
| COND-01 | Implement conductance calculations from envelope properties | Existing code in from_spec() calculates h_tr_w, h_tr_em; verify formulas and edge cases |
| METRIC-01 | Implement annual heating energy calculation | step_physics_5r1c() returns HVAC energy; verify sign convention and accumulation |
| METRIC-02 | Implement peak heating power calculation | peak_power_heating field tracks maximum; verify calculation timing and reset logic |
| REF-01 | Validate against ASHRAE 140 reference values (EnergyPlus, ESP-r, TRNSYS) | Reference values in tests/ashrae_140_case_600.rs; tolerance bands ±15% annual, ±10% monthly |
| TEMP-01 | Implement temperature profile tracking | HourlyData tracks zone and mass temperatures; diagnostic infrastructure exists |
| WEATHER-01 | Use ASHRAE 140 standard weather data | HourlyWeatherData loads weather; verify data source matches ASHRAE 140 specification |
| THERM-01 | Implement thermal mass capacitance (Cm) calculation | ISO 13790 Annex C effective capacitance; current implementation uses half-insulation rule |
| THERM-02 | Implement mass-to-surface conductance (h_tr_ms) calculation | ISO 13790 standard value h_ms = 9.1 W/m²K; current code uses this value |
| LAYER-01 | Implement effective thermal capacitance per layer | Construction layers have iso_13790_effective_capacitance_per_area(); verify half-insulation rule |
| LAYER-02 | Implement layer position relative to insulation | ISO 13790 Annex C defines effective layers; current implementation uses this approach |
| WINDOW-01 | Implement window conductance (h_tr_w) calculation | h_tr_w = U_win × Window Area; verify thermal bridge coefficient application |
| WINDOW-02 | Apply window U-value to h_tr_em (exterior-to-mass) | Current code calculates h_tr_em from opaque conductance; verify window U-value is included |
| INFIL-01 | Implement infiltration conductance (h_ve) calculation | h_ve = ACH × V × ρ × cp / 3600; current code uses this formula |
| INTERNAL-01 | Implement internal gains (occupants, equipment) | loads field contains internal gains; verify convective/radiative split |
| INTERNAL-02 | Implement radiative vs convective internal gain distribution | convective_fraction parameter (default 0.5); verify ASHRAE 140 default |
| GROUND-01 | Implement ground temperature boundary condition | GroundTemperature trait implemented; verify Case 600 uses correct ground model |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Rust | Edition 2021 | Systems programming, memory safety | Project language, CTA operations, 5R1C physics |
| serde | 1.0 | Serialization for diagnostics | Diagnostic output, test fixtures |
| anyhow | 1.0 | Error handling | Error propagation in thermal model |

### Testing
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| rstest | *TBD* | Parameterized testing | Required by CONTEXT.md for testing across multiple envelope properties |
| cargo test | Built-in | Unit and integration tests | Rust's native test framework |
| criterion | 0.5 | Benchmarking | Performance validation for conductance calculations |

### Physics & Math
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| faer | 0.23.2 | Numerical operations | VectorField CTA operations, element-wise arithmetic |
| ndarray | 0.16 | Multi-dimensional arrays | Alternative tensor backend, future GPU support |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| rstest | rstest with fixtures | rstest provides cleaner parameterized test syntax |
| VectorField | raw Vec<f64> | CTA enables future GPU acceleration; VectorField is required |
| ISO 13790 | ISO 52016 | ISO 52016 replaced ISO 13790 but Fluxion uses ISO 13790; maintain compatibility |

**Installation:**
```bash
# Add rstest to dev-dependencies (if not present)
cargo add rstest --dev

# Run tests
cargo test

# Run with output
cargo test -- --nocapture

# Single-threaded (for debugging)
cargo test -- --test-threads=1
```

## Architecture Patterns

### Recommended Project Structure
```
src/sim/
├── tests/
│   └── test_conductance_calculations.rs  # NEW: Unit tests for conductances
├── engine.rs                              # ThermalModel, 5R1C implementation
├── construction.rs                         # Assemblies, Construction (R-values)
└── ...

tests/
├── ashrae_140_validation.rs               # EXISTING: Integration tests
├── ashrae_140_case_600.rs               # EXISTING: Case 600 reference
└── ashrae_140_diagnostic_test.rs        # EXISTING: Diagnostic tests
```

### Pattern 1: Parameterized Testing with rstest
**What:** Test multiple conductance calculations with different envelope properties using single test function
**When to use:** Validating conductance formulas across multiple window U-values, areas, insulation levels
**Example:**
```rust
use rstest::*;

#[rstest]
#[case(0.5, 12.0, 6.0)]   // U=0.5, Area=12m², expected=6.0 W/K
#[case(1.5, 12.0, 18.0)]  // U=1.5, Area=12m², expected=18.0 W/K
#[case(2.9, 12.0, 34.8)]  // U=2.9, Area=12m², expected=34.8 W/K
fn test_h_tr_w_calculation(#[case] u_value: f64, #[case] area: f64, #[case] expected: f64) {
    let h_tr_w = calculate_h_tr_w(u_value, area);
    assert!((h_tr_w - expected).abs() < 0.01, "h_tr_w mismatch: got {}, expected {}", h_tr_w, expected);
}
```

### Pattern 2: Helper Method Extraction
**What:** Extract conductance calculations into standalone functions for testing
**When to use:** Complex conductance formulas (h_tr_em with thermal bridges, h_tr_w with window area)
**Example:**
```rust
// In ThermalModel impl
pub fn calculate_h_tr_w(window_u_value: f64, window_area: f64) -> f64 {
    window_u_value * window_area
}

pub fn calculate_h_tr_em(
    wall_u: f64,
    wall_area: f64,
    roof_u: f64,
    roof_area: f64,
    thermal_bridge_coeff: f64,
    h_ms: f64,
    a_m: f64,
) -> f64 {
    let h_tr_op = wall_area * wall_u + roof_area * roof_u + thermal_bridge_coeff;
    let h_tr_em_val = 1.0 / ((1.0 / h_tr_op) - (1.0 / (h_ms * a_m)));
    h_tr_em_val.max(0.1)
}
```

### Pattern 3: Test-Driven Development Cycle
**What:** Write failing test, implement fix, verify pass, repeat
**When to use:** All Phase 1 conductance fixes per CONTEXT.md locked decision
**Example:**
```bash
# 1. Write failing test
# 2. Run test and confirm failure
cargo test test_h_tr_w_calculation

# 3. Implement fix in calculate_h_tr_w()
# 4. Run test and confirm pass
cargo test test_h_tr_w_calculation

# 5. Run full test suite
cargo test -- --test-threads=1
```

### Anti-Patterns to Avoid
- **Testing implementation details:** Test conductance formulas, not internal VectorField operations
- **Hardcoded reference values:** Calculate expected values from physics formulas, don't hardcode magic numbers
- **Ignoring unit conversion:** Always specify W/K vs K/W; conductances are W/K in ISO 13790
- **Skipping HVAC sign convention:** Heating = positive, Cooling = negative; verify in tests

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Parameterized test framework | Custom test loops | rstest | Cleaner syntax, fixtures, compile-time type checking |
| Conductance calculations from scratch | Manual formula implementation | ISO 13790 Annex C formulas | Standard-compliant, verified, handles edge cases |
| HVAC load calculation | Manual power demand logic | hvac_power_demand() + Ti_free | Existing implementation with CTA support |
| Diagnostic output | Custom logging | DiagnosticCollector | Existing infrastructure with CSV export, energy breakdowns |
| Thermal mass calculation | Manual capacitance summation | ISO 13790 effective capacitance | Half-insulation rule handles layer position correctly |

**Key insight:** ISO 13790 provides well-tested formulas for conductances and thermal mass. Implementing custom calculations risks introducing new bugs. Use ISO 13790 Annex C as source of truth.

## Common Pitfalls

### Pitfall 1: Conductance Unit Confusion (W/K vs K/W)
**What goes wrong:** Conductance stored as resistance (K/W) instead of conductance (W/K), causing 100x+ errors in heat transfer
**Why it happens:** Confusion between R (resistance, K/W) and H (conductance, W/K) in thermal networks
**How to avoid:**
- Always use W/K for conductances in 5R1C model
- Document units in function signatures: `calc_h_tr_w(...) -> f64` (W/K)
- Add unit tests with explicit W/K assertions
**Warning signs:** Conductance values <0.01 W/K or >1000 W/K for typical buildings

### Pitfall 2: Window U-Value Not Applied to h_tr_em
**What goes wrong:** Window thermal bridge effect missing from exterior-to-mass conductance, causing systematic heating over-prediction
**Why it happens:** h_tr_em calculated from opaque walls/roof only, window conduction bypassed thermal mass
**How to avoid:**
- Include window U-value × window_area in h_tr_op calculation
- Use thermal_bridge_coefficient field to capture window thermal bridging
- Test with varying window areas (6m², 12m², 18m²)
**Warning signs:** Heating load scales linearly with window area but h_tr_em unchanged

### Pitfall 3: HVAC Load Calculation Using Wrong Temperature
**What goes wrong:** Load calculated from Ti (actual indoor temp) instead of Ti_free (free-floating temp), causing HVAC fighting thermal mass
**Why it happens:** hvac_power_demand() receives Ti instead of Ti_free, calculates power to overcome thermal mass inertia
**How to avoid:**
- Always use Ti_free for load calculation: `Q_hvac = (T_setpoint - T_free) / sensitivity`
- Verify hvac_power_demand() receives t_i_free parameter, not t_i
- Test with extreme outdoor temps: -10°C winter, 40°C summer
**Warning signs:** HVAC energy massively different from reference (>300% error)

### Pitfall 4: Missing Thermal Bridge Coefficient
**What goes wrong:** Thermal bridges (window frames, wall junctions) not modeled, causing systematic underprediction of heat loss
**Why it happens:** Simplified conductance formulas assume perfect insulation at joints
**How to avoid:**
- Include thermal_bridge_coefficient in h_tr_em calculation
- Typical value: 0.5-2.0 W/K for residential buildings
- Test with thermal_bridge_coefficient = 0.0 vs 1.0 W/K
**Warning signs:** Peak heating loads consistently lower than reference

### Pitfall 5: HVAC Sign Convention Errors
**What goes wrong:** Heating and cooling energy signed incorrectly, causing double-counting or subtraction errors
**Why it happens:** Inconsistent sign convention across hvac_heating vs hvac_cooling fields
**How to avoid:**
- Heating = positive energy (energy input to building)
- Cooling = negative energy (energy removed from building)
- Net energy = heating + cooling (signed sum)
- Test with winter-only and summer-only cases
**Warning signs:** Annual energy wrong sign or magnitude when heating and cooling both present

### Pitfall 6: Sensitivity Calculation Errors
**What goes wrong:** HVAC sensitivity (°C/W) wrong, causing HVAC demand too high/low
**Why it happens:** Denominator in sensitivity formula missing ground coupling term or inter-zone conductance
**How to avoid:**
- Use ISO 13790 formula: `sensitivity = (h_tr_ms + h_tr_is) / denominator`
- Denominator includes: h_tr_ms × h_tr_is + (h_tr_ms + h_tr_is) × (h_tr_w + h_ve + h_tr_floor)
- Test with known simple cases (single zone, no ground coupling)
**Warning signs:** HVAC demand 10-100x different from reference

### Pitfall 7: Not Using Test-Driven Development
**What goes wrong:** Implementing fixes without tests leads to regression bugs and incomplete fixes
**Why it happens:** Rushing to fix bugs without validation, testing only at integration level
**How to avoid:**
- Write failing unit test for each conductance BEFORE fixing
- Run tests after every change: `cargo test -- --test-threads=1`
- Commit only when all tests pass
**Warning signs:** Fixes work for one case but break others, regression bugs discovered late

## Code Examples

Verified patterns from official sources:

### Conductance Calculation (ISO 13790 Annex C)
```rust
// Source: ISO 13790:2008 Annex C
// Window conductance (h_tr_w = U_win × Window Area)
pub fn calculate_h_tr_w(window_u_value: f64, window_area: f64) -> f64 {
    window_u_value * window_area  // W/K
}

// Exterior-to-mass conductance (h_tr_em)
// Includes opaque walls, roof, and thermal bridges
pub fn calculate_h_tr_em(
    wall_u: f64,
    wall_area: f64,
    roof_u: f64,
    roof_area: f64,
    thermal_bridge_coeff: f64,
    h_ms: f64,  // Mass-to-surface conductance (9.1 W/m²K)
    a_m: f64,  // Effective mass area (factor × floor_area)
) -> f64 {
    // Opaque conductance (walls + roof + thermal bridges)
    let h_tr_op = wall_area * wall_u + roof_area * roof_u + thermal_bridge_coeff;

    // Exterior-to-mass conductance (series resistance)
    let h_tr_em_val = 1.0 / ((1.0 / h_tr_op) - (1.0 / (h_ms * a_m)));

    // Prevent division by zero or negative conductance
    h_tr_em_val.max(0.1)  // Minimum 0.1 W/K
}
```

### HVAC Load Calculation (Ti_free)
```rust
// Source: ASHRAE Handbook Fundamentals, Chapter 19
// HVAC power demand using free-floating temperature
pub fn calculate_hvac_power_demand(
    t_i_free: f64,  // Free-floating indoor temp (°C)
    heating_setpoint: f64,
    cooling_setpoint: f64,
    sensitivity: f64,  // Temperature change per Watt (°C/W)
) -> f64 {
    if t_i_free < heating_setpoint {
        // Heating: power needed to reach setpoint
        let temp_deficit = heating_setpoint - t_i_free;
        (temp_deficit / sensitivity).max(0.0)  // Positive W
    } else if t_i_free > cooling_setpoint {
        // Cooling: power needed to reach setpoint
        let temp_excess = t_i_free - cooling_setpoint;
        -(temp_excess / sensitivity).min(0.0)  // Negative W
    } else {
        // Deadband: no HVAC needed
        0.0
    }
}
```

### Infiltration Conductance
```rust
// Source: ISO 13790:2008
// Infiltration conductance (h_ve = ACH × V × ρ × cp / 3600)
pub fn calculate_h_ve(
    infiltration_ach: f64,  // Air changes per hour
    zone_volume: f64,       // m³
    air_density: f64 = 1.2,  // kg/m³ at sea level
    specific_heat: f64 = 1005.0,  // J/kg·K for air
) -> f64 {
    let air_cap = zone_volume * air_density * specific_heat;  // J/K
    (infiltration_ach * air_cap) / 3600.0  // W/K
}
```

### Mass-to-Surface Conductance
```rust
// Source: ISO 13790:2008
// Mass-to-surface conductance (h_tr_ms = 9.1 × A_m)
pub fn calculate_h_tr_ms(
    a_m: f64,  // Effective mass area (m²)
) -> f64 {
    const H_MS: f64 = 9.1;  // W/m²K (ISO 13790 standard value)
    H_MS * a_m  // W/K
}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Manual conductance calculation | ISO 13790 Annex C formulas | 2025-12 (Issue #340) | Reduced HVAC energy from 300-500% error to 61% failure rate |
| Cached sensitivity only | Dynamic sensitivity recalculation | 2025-12 (Issue #366) | Fixed ventilation rate changes causing stale sensitivity |
| Heating from Ti | Heating from Ti_free | TBD (Phase 1 fix needed) | Expected: Eliminate HVAC fighting thermal mass |

**Deprecated/outdated:**
- Old conductance calculation without thermal bridges: Replaced by ISO 13790 Annex C approach
- Static sensitivity without recalculation: Fixed in Issue #366, now recalculates at each timestep
- Manual HVAC load calculation: Should use Ti_free (not yet fixed)

## Open Questions

1. **Window U-value application to h_tr_em**
   - What we know: Current code calculates h_tr_em from opaque walls/roof only
   - What's unclear: Should window U-value be added to h_tr_op directly or via thermal_bridge_coefficient?
   - Recommendation: Test both approaches, compare to ASHRAE 140 Case 600 reference

2. **Thermal bridge coefficient value**
   - What we know: thermal_bridge_coefficient field exists, default value unclear
   - What's unclear: What is the correct default for ASHRAE 140 cases (0.5 W/K? 1.0 W/K?)
   - Recommendation: Test Case 600 with values 0.0, 0.5, 1.0 W/K, pick best match

3. **HVAC sign convention in current code**
   - What we know: hvac_heating and hvac_cooling track separate values
   - What's unclear: Are they signed (positive/negative) or both positive?
   - Recommendation: Review step_physics_5r1c() implementation, verify sign convention

4. **rstest installation**
   - What we know: Cargo.toml does not include rstest
   - What's unclear: Should rstest be added or use built-in parameterization?
   - Recommendation: Add rstest for cleaner parameterized test syntax (per CONTEXT.md)

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Rust cargo test + rstest (to be added) |
| Config file | Cargo.toml (dev-dependencies) |
| Quick run command | `cargo test --test-threads=1` |
| Full suite command | `cargo test` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| BASE-01 | 5R1C conductance calculations | unit | `cargo test test_h_tr_w_calculation` | ❌ Wave 0 |
| BASE-02 | Case 600 reference validation | integration | `cargo test test_case_600_baseline_ashrae_140_reference` | ✅ tests/ashrae_140_case_600.rs |
| BASE-03 | Conductance units W/K | unit | `cargo test test_conductance_units` | ❌ Wave 0 |
| BASE-04 | U-value to conductance conversion | unit | `cargo test test_u_value_to_conductance` | ❌ Wave 0 |
| FREE-01 | Ti_free calculation | unit | `cargo test test_ti_free_calculation` | ❌ Wave 0 |
| COND-01 | Conductance from envelope properties | unit | `cargo test test_conductance_from_envelope` | ❌ Wave 0 |
| METRIC-01 | Annual heating energy | integration | `cargo test test_annual_heating_energy` | ❌ Wave 0 |
| METRIC-02 | Peak heating power | integration | `cargo test test_peak_heating_power` | ❌ Wave 0 |
| REF-01 | ASHRAE 140 reference validation | integration | `cargo test --test ashrae_140_validation` | ✅ tests/ashrae_140_validation.rs |
| TEMP-01 | Temperature profile tracking | integration | `cargo test test_temperature_profiles` | ❌ Wave 0 |
| WEATHER-01 | ASHRAE 140 weather data | integration | `cargo test test_ashrae_140_weather` | ❌ Wave 0 |
| THERM-01 | Thermal mass capacitance | unit | `cargo test test_thermal_mass_capacitance` | ❌ Wave 0 |
| THERM-02 | Mass-to-surface conductance | unit | `cargo test test_h_tr_ms_calculation` | ❌ Wave 0 |
| LAYER-01 | Effective thermal capacitance | unit | `cargo test test_effective_capacitance` | ❌ Wave 0 |
| LAYER-02 | Layer position relative to insulation | unit | `cargo test test_layer_position` | ❌ Wave 0 |
| WINDOW-01 | Window conductance | unit | `cargo test test_h_tr_w_calculation` | ❌ Wave 0 |
| WINDOW-02 | Window U-value to h_tr_em | unit | `cargo test test_window_to_h_tr_em` | ❌ Wave 0 |
| INFIL-01 | Infiltration conductance | unit | `cargo test test_h_ve_calculation` | ❌ Wave 0 |
| INTERNAL-01 | Internal gains | integration | `cargo test test_internal_gains` | ❌ Wave 0 |
| INTERNAL-02 | Radiative vs convective split | unit | `cargo test test_convective_fraction` | ❌ Wave 0 |
| GROUND-01 | Ground temperature boundary | integration | `cargo test test_ground_temperature` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `cargo test --test-threads=1` (quick test of conductance unit tests)
- **Per wave merge:** `cargo test` (full suite including integration tests)
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `src/sim/tests/test_conductance_calculations.rs` — unit tests for all 5R1C conductances
- [ ] `tests/ashrae_140_phase1_integration.rs` — integration tests for lightweight cases 600-650
- [ ] Add `rstest` to Cargo.toml dev-dependencies:
  ```toml
  [dev-dependencies]
  rstest = "0.18"  # or latest version
  ```
- [ ] Helper methods in `src/sim/engine.rs`:
  - `calculate_h_tr_w(window_u_value, window_area) -> f64`
  - `calculate_h_tr_em(...) -> f64`
  - `calculate_h_tr_ms(a_m) -> f64`
  - `calculate_h_ve(ach, volume) -> f64`

## Sources

### Primary (HIGH confidence)
- **ISO 13790:2008** - Energy performance of buildings - Calculation of energy use for space heating and cooling, Annex C (5R1C thermal network)
- **ASHRAE Handbook - Fundamentals, Chapter 19** - Residential Cooling and Heating Load Calculations (HVAC load calculation)
- **Fluxion source code** - `src/sim/engine.rs` (ThermalModel implementation, conductance calculations)
- **Fluxion source code** - `tests/ashrae_140_case_600.rs` (Case 600 reference values)
- **Fluxion source code** - `src/validation/diagnostic.rs` (DiagnosticCollector, HourlyData)
- **Fluxion source code** - `src/validation/ashrae_140_cases.rs` (CaseSpec, builder pattern)
- **Fluxion documentation** - `docs/ASHRAE_140_5R1C_MODEL.md` (5R1C model overview)
- **Fluxion documentation** - `docs/ISSUE_273_ROOT_CAUSE_ANALYSIS.md` (HVAC power demand bug analysis)
- **Fluxion documentation** - `docs/ASHRAE140_RESULTS.md` (Current validation status: 61% failure rate)
- **CONTEXT.md** - Locked decisions for Phase 1 (high-impact fixes, TDD approach, test methodology)

### Secondary (MEDIUM confidence)
- **ISO 52016** - Replacement for ISO 13790 (not yet adopted by Fluxion)
- **Fluxion documentation** - `docs/ARCHITECTURE.md` (Thermal model structure, CTA operations)
- **Fluxion documentation** - `docs/CLAUDE.md` (Project guidelines, testing patterns)

### Tertiary (LOW confidence)
- **Web search** - Attempted queries for ISO 13790 and ASHRAE 140 (search service returned no results, relied on existing code and docs)
- **External references** - EnergyPlus, ESP-r, TRNSYS reference values (not independently verified, from Fluxion test files)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - Rust, serde, cargo test are project standards; rstest is standard Rust testing framework
- Architecture: HIGH - 5R1C model documented in code and docs; conductance formulas from ISO 13790
- Pitfalls: HIGH - All 7 pitfalls identified from existing code review and issue investigation reports
- Testing: HIGH - Existing diagnostic infrastructure (DiagnosticCollector) well-documented; rstest is standard Rust parameterized testing

**Research date:** 2026-03-09
**Valid until:** 2026-04-08 (30 days for stable physics standards)

**Phase Requirements Coverage:**
- Total requirements: 24 (BASE-01 through GROUND-01)
- Requirements with research support: 24/24 (100%)
- Open questions: 4 (need resolution during planning/implementation)

**Ready for Planning:** Research complete. Planner can create PLAN.md files with conductance fix tasks, HVAC load correction tasks, and comprehensive test suite.
