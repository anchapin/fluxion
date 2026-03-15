# Phase 22: Validation Gap Resolution - Context

**Gathered:** 2026-03-15
**Status:** Ready for planning

<domain>
## Phase Boundary

Resolve all known ASHRAE 140 validation gaps (Case 960, 8R3C thermal network, high-mass accuracy) while ensuring no regressions in existing passing cases.

**What this delivers:**
- Case 960 verification and A/B testing (COP correction already implemented in Phase 8)
- 8R3C thermal network evaluation with performance comparison against 5R1C baseline (VAL-02, VAL-03, VAL-04, VAL-05)
- High-mass accuracy improvement through thermal mass energy accounting validation (VAL-06, VAL-08)
- A/B testing framework for quantifying validation gap fixes before adoption (VAL-09)
- 900-series regression test to prevent Case 960 fix from breaking Cases 920, 930, 940 (VAL-07)

This phase ensures all known validation gaps are addressed with comprehensive testing and regression prevention before moving to production readiness.

</domain>

<decisions>
## Implementation Decisions

### 8R3C Thermal Network Evaluation

**Research strategy:** Research references first
- Analyze ASHRAE 140 reference implementations (EnergyPlus, TRNSYS, ESP-r) to understand thermal network structures
- Determine if reference programs use 8R3C or different approaches
- Avoid full implementation if 8R3C unlikely to address root cause (based on 6R2C findings)
- This approach saves significant implementation effort (~2000+ lines of physics code) if research shows 8R3C won't help

**Performance threshold:** No minimum (research-only)
- No strict ≥1,000 configs/sec requirement during evaluation phase
- Allows thorough evaluation of 8R3C even if performance is poor
- Performance becomes decision factor only if accuracy improvement is demonstrated
- Based on 6R2C being 40-50% slower than 5R1C (~1,200-1,500 configs/sec vs ~2,575)

**Adoption criteria:** Any measurable improvement
- 8R3C can be adopted if it provides ANY improvement over 5R1C baseline
- Lower bar than VAL-03's <50% error requirement
- Example: If 5R1C high-mass error is 229-322%, 8R3C achieving 200-250% would qualify
- Focuses on measurable benefit rather than strict quantitative threshold

**Implementation scope:** Full implementation
- Build complete 8R3C thermal network with 8 resistance nodes and 3 capacitance nodes
- Envelope mass, internal mass, surface mass nodes (3 capacitances)
- Three additional resistance nodes connecting masses to exterior, interior, and surface
- Run complete ASHRAE 140 validation (all 18 cases) to evaluate both high-mass and low-mass accuracy
- Most comprehensive evaluation despite significant implementation effort

### High-Mass Accuracy

**Approach:** Thermal mass energy accounting validation (VAL-08)
- Implement energy conservation validation: Σenergy_in = Σenergy_out + Δmass_energy
- Validate balance over 8760 timesteps for each simulation
- Confirms physics is conserving energy correctly, even if output is wrong
- This is a diagnostic task to validate correctness, not a fix task

**Improvement target:** No target - validate correctness only
- Focus on confirming energy accounting is valid
- No quantitative error reduction target for this phase
- If energy balance is confirmed, document that physics is correct
- If annual energy error persists, document as 5R1C fundamental limitation (not a bug)

**Fix strategy:** Fix only if bugs found
- Attempt fixes ONLY if energy accounting reveals conservation errors
- Don't chase fundamental 5R1C limitations (8 sophisticated approaches in Plans 03-07 through 03-14 all failed)
- Pragmatic approach: if physics is correct, document as known limitation and move on
- Preserves engineering resources for solvable issues

**Validation scope:** 900 + 600 series (high + low mass)
- Validate all 900-series cases (920, 930, 940, 950, 960)
- Validate all 600-series cases (600, 610, 620, 630, 640, 650)
- Covers both high-mass and low-mass buildings for comprehensive energy accounting validation
- Ensures any fix doesn't break low-mass cases

### A/B Testing Framework

**Test structure:** Multi-variant (5R1C + 8R3C + fixes)
- Run all 18 ASHRAE 140 cases with multiple thermal network variants
- Compare 5R1C baseline against 8R3C and targeted fixes
- Most comprehensive approach suitable for comparative analysis
- Allows simultaneous evaluation of all variants

**Test metrics:** NMBE, CV(RMSE), pass rates
- Primary metrics: Normalized Mean Bias Error (NMBE), Coefficient of Variation of RMSE
- Pass rate: Percentage of cases within ASHRAE 140 tolerance bands
- Aligns with Phase 19 statistical validation methodology
- Statistically rigorous approach suitable for quantifying improvement

**Integration:** Dedicated test module
- Create tests/validation/ab_testing.rs as dedicated A/B testing framework
- Can share BuildingScenario fixtures and test infrastructure from Phase 21
- Runs independently from Phase 21 regression tests
- Good balance: infrastructure reuse without tight coupling

**Automation:** Manual only (no CI)
- A/B tests triggered manually with `cargo test ab_testing -- --nocapture` or similar
- No CI/CD integration during research/exploration phase
- Fast to implement, minimal overhead
- Suitable for experimentation before framework is stable

### 900-Series Regression

**Test scope:** Sequential with fail-fast
- Run Cases 920, 930, 940, 950, 960 sequentially
- Stop immediately on first failure (fail-fast behavior)
- Easier to isolate which specific case caused failure
- Faster iteration than running all cases together
- May miss interaction effects but prioritizes debugging efficiency

**Check type:** Full validation (all metrics)
- Check all metrics for each 900-series case:
  - Annual heating energy (±15% tolerance)
  - Annual cooling energy (±15% tolerance)
  - Peak heating load (within reference range)
  - Peak cooling load (within reference range)
  - Free-floating max/min temperature (within reference range)
- Most comprehensive coverage (6x the checks vs energy-only)
- Ensures Case 960 COP correction doesn't break other cases' metrics

**Tolerance bands:** ±15% annual, ±10% monthly
- Use ASHRAE 140 standard tolerance bands
- ±15% for annual energy values
- ±10% for monthly energy values (where applicable)
- Reuses existing ValidationReport::compute_status() from Phase 21
- Strict, standard criteria ensures all cases meet same requirements

**Integration:** Extend existing tests
- Add 900-series regression tests to existing tests/ashrae_140_case_900.rs
- Reuses ASHRAE140Validator and existing test infrastructure
- Minimal new code required
- Clear integration with Phase 21 test suite

### Claude's Discretion

**8R3C research depth:** How thoroughly to analyze ASHRAE 140 reference implementations (source code review vs documentation vs published papers)
**Energy accounting implementation details:** Whether to validate energy balance at each timestep, hourly, or annually (most informative vs most performant)
**A/B test statistical analysis:** Whether to implement hypothesis testing (paired t-tests) or rely on NMBE/CV(RMSE) and pass rates
**900-series test timing:** Whether regression tests should run as part of full ASHRAE 140 suite or as separate command

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets

**ASHRAE 140 validation infrastructure:**
- `src/validation/ashrae_140_validator.rs` - Has Case 960 COP correction (heating_efficiency=0.9, cooling_cop=3.0)
- `src/validation/ashrae_140_cases.rs` - Case specifications for all 18 ASHRAE 140 cases
- `src/validation/benchmark.rs` - Reference ranges and tolerance bands for each case
- `src/validation/report.rs` - ValidationReport with compute_status() for tolerance checking
- `tests/ashrae_140_case_900.rs` - Existing high-mass case tests
- `tests/ashrae_140_case_960_sunspace.rs` - Case 960 tests with COP correction

**Thermal mass validation:**
- `src/validation/thermal_mass.rs` - Thermal mass validation, coupling ratio calculations, correction methods
- `tests/validation/thermal_mass_tests.rs` - Existing thermal mass unit tests
- `tests/thermal_mass_coupling_investigation.rs` - Thermal mass coupling diagnostics
- `tests/thermal_mass_calibration_diagnostics.rs` - Thermal mass calibration analysis
- ThermalModel has 6R2C opt-in support via `configure_6r2c_model()`

**Statistical validation:**
- `src/validation/statistical.rs` - NMBE, CV(RMSE) calculations from Phase 19
- FDR correction implementation (Benjamini-Hochberg procedure)
- 95% confidence interval calculations
- Group validation with 80% passing rate requirement

**Phase 21 integration testing:**
- `src/testing/integration/` - E2E framework with BuildingScenario builder
- `WiringTracer` - Runtime tracing for module integration validation
- `tests/integration/` - Integration tests using BuildingScenario fixtures

### Established Patterns

**Validation test pattern:**
- Unit tests in `tests/ashrae_140_case_*.rs` files (one per case series)
- Test functions follow pattern: `test_case_XXX_<metric>()`
- Uses ASHRAE140Validator to run simulation and compare to reference
- Report generation via `to_markdown()` for CI output

**Thermal mass testing pattern:**
- Tests verify coupling ratios (h_tr_em / h_tr_ms)
- Mode-specific coupling: separate heating/cooling factors
- Before/after comparison to measure improvement
- Thermal mass time constant analysis (τ = Cm / Σconductances)

**Statistical validation pattern:**
- NMBE formula: NMBE = Σ((predicted - reference) / reference) / n
- CV(RMSE) formula: CV(RMSE) = RMSE / mean(reference) × 100
- FDR correction: Sort p-values, apply BH critical values, reject if p ≤ BH_critical
- Group validation: 80% pass rate minimum for groups with ≥5 cases

### Integration Points

**8R3C evaluation:**
- New module: `src/sim/engine_8r3c.rs` - 8R3C thermal network implementation
- Extend `src/sim/engine.rs` ThermalModel to support 8R3C via feature flag or configuration
- 8R3C nodes: envelope mass (Cm_env), internal mass (Cm_int), surface mass (Cm_surf), plus 3 resistance nodes
- Integration with ASHRAE140Validator for 8R3C variant testing
- Performance benchmarking: Run BatchOracle::evaluate_population() to measure configs/sec

**Thermal mass energy accounting:**
- Extend `src/validation/thermal_mass.rs` with energy balance validation function
- Add test in `tests/validation/thermal_mass_energy_accounting.rs`
- Validate at timestep level: Σ(Q_heating + Q_cooling + Q_solar + Q_infiltration) = Σ(Q_hvac_demand + Q_mass_storage_change)
- Update KNOWN_LIMITATIONS.md with findings

**A/B testing framework:**
- New module: `tests/validation/ab_testing.rs` - Dedicated A/B testing framework
- Reuse BuildingScenario builder from `src/testing/integration/`
- Implement ABTestRunner struct with methods: run_variant(variant), compare_results(), generate_report()
- Integrate with statistical validation (NMBE, CV(RMSE)) for metric calculation
- Manual trigger: `cargo test ab_testing -- --nocapture`

**900-series regression tests:**
- Extend `tests/ashrae_140_case_900.rs` with regression test functions
- New test: `test_900_series_regression()` or similar
- Sequential execution: run Case 920, fail if error → stop, otherwise run 930, etc.
- Full validation for each case: annual heating, annual cooling, peak heating, peak cooling, free-floating temps
- Use existing ValidationReport::compute_status() for tolerance checking (±15% annual, ±10% monthly)

</code_context>

<specifics>
## Specific Ideas

**8R3C research priorities:**
- Priority 1: Determine if ASHRAE 140 reference implementations use 8R3C or have other approaches to high-mass modeling
- Priority 2: If reference uses 8R3C, understand mass node placement and resistance values
- Priority 3: If reference uses different approach, investigate what that approach is and why it works better
- Research should inform whether 8R3C implementation is worth the effort

**Thermal mass energy accounting validation:**
- Validate at each timestep: Σenergy_in(t) = Σenergy_out(t) + Δmass_energy(t)
- Track cumulative error over 8760 hours: Σ|Σenergy_in - Σenergy_out - Δmass_energy|
- If cumulative error is near zero (<0.01% of total energy), physics is correct
- If cumulative error is significant, fix energy balance bug before attempting other improvements
- This confirms whether high-mass annual energy error is a bug or fundamental limitation

**A/B testing implementation:**
```rust
// tests/validation/ab_testing.rs
pub struct ABTestRunner {
    pub fn run_variant(&self, variant: ThermalNetworkVariant, case: &str) -> TestResults
    pub fn compare_results(&self, baseline: TestResults, test: TestResults) -> ComparisonReport
}

pub enum ThermalNetworkVariant {
    FiveR1C,    // Current default
    SixR2C,     // Existing opt-in
    EightR3C,    // New to evaluate
    ThermalMassFixA, // Targeted refinement
    ThermalMassFixB, // Alternative refinement
}
```

**900-series regression test structure:**
```rust
// tests/ashrae_140_case_900.rs
#[test]
fn test_900_series_regression() {
    let cases = vec!["920", "930", "940", "950", "960"];

    for case in cases {
        let result = ASHRAE140Validator::validate_case(case);
        assert!(result.is_within_tolerance(), "Case {} failed: {:?}", case, result);

        // Full validation: check all metrics
        assert!(result.annual_heating_ok(), "Case {} heating OOR", case);
        assert!(result.annual_cooling_ok(), "Case {} cooling OOR", case);
        assert!(result.peak_heating_ok(), "Case {} peak heating OOR", case);
        assert!(result.peak_cooling_ok(), "Case {} peak cooling OOR", case);
    }
}
```

**Case 960 COP correction location:**
- Already implemented in `src/validation/ashrae_140_validator.rs::validate_case_960`
- Correction: `annual_cooling_mwh / 3.0` for electrical equivalent
- Regression test should verify this correction doesn't break Cases 920-950
- Those cases don't use COP correction, so should pass with same results as before

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope. All decisions relate to 8R3C evaluation, high-mass energy accounting validation, A/B testing framework, and 900-series regression as defined in Phase 22 requirements (VAL-01 through VAL-09).

</deferred>

---

*Phase: 22-validation-gap-resolution*
*Context gathered: 2026-03-15*
