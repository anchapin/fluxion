---
phase: A-baseline-stripping
plan: 02
type: execute
wave: 2
depends_on: ["A-01"]
files_modified:
  - tests/ashrae_140_blind_validation.rs
  - data/ashrae_140_true_reference/
autonomous: false
requirements:
  - BASELINE-01
  - BASELINE-03

must_haves:
  truths:
    - "True baseline failure state is measured and documented"
    - "Every failing case has a magnitude-of-failure entry in the baseline report"
  artifacts:
    - path: "tests/ashrae_140_blind_validation.rs"
      provides: "Deterministic blind validation test harness"
    - path: ".planning/baseline/BLIND_BASELINE_RESULTS.md"
      provides: "Complete baseline failure measurements"
---

<objective>
Establish the true baseline: run the blind validation harness and measure exactly how much each case fails. This is the foundation for all subsequent physics fixes — we must know the starting point to measure progress.
</objective>

<execution_context>
@/home/alex/.agents/get-shit-done/workflows/execute-plan.md
</execution_context>

<context>
@.planning/phases/A-baseline-stripping/A-01-SUMMARY.md
@src/validation/ashrae_140_validator.rs (validate_blind method from A-01)
@src/validation/benchmark.rs (current calibrated ranges)

From A-01: ValidationMode::Blind is implemented. validate_blind(case_spec) takes only CaseSpec.

From ASHRAE140_RESULTS.md:
- Current pass rate WITH corrections: 9.4% (6/64)
- Current pass rate WITHOUT corrections: Expected ~0%
- 900 series failures: heating ÷ 4.0, cooling × 0.35-0.50 corrections applied
- Free-floating cases show extreme failures (125°C vs 41-46°C reference for 900FF)
</context>

<tasks>

<task type="auto">
  <name>Task 1: Create blind validation test file</name>
  <files>tests/ashrae_140_blind_validation.rs</files>
  <action>
Create tests/ashrae_140_blind_validation.rs that:
1. Loads all ASHRAE 140 case definitions (reuse CaseSpec from existing validator)
2. Runs each case through validate_blind() with corrections disabled
3. Compares against current benchmark ranges (since true reference data doesn't exist yet)
4. Outputs structured results: case, metric, simulated value, reference range, pass/fail, % error

Use the same test infrastructure pattern as existing tests/ directory.

The test MUST be deterministic: same input → same output every run.

Key implementation:
```rust
#[test]
fn test_blind_validation_all_cases() {
    let validator = Ashrae140Validator::new_blind();  // No corrections
    let cases = ashrae_140_cases::all_cases();  // Load all standard cases

    for case in cases {
        let result = validator.validate_blind(&case.spec());
        // Compare and record
    }

    // Generate report
    let report = generate_blind_validation_report(results);
    println!("{}", report);
}
```

Include tests for:
- 600 series (low-mass): 600, 610, 620, 630, 640, 650
- 900 series (high-mass): 900, 910, 920, 930, 940, 950
- Free-floating: 600FF, 650FF, 900FF, 950FF
- Special: 960, 195

At minimum: 18 cases that can be run headlessly.
  </action>
  <verify>
    cargo test --test ashrae_140_blind_validation 2>&1 | tail -40
  </verify>
  <done>Blind validation test file exists and runs successfully</done>
</task>

<task type="auto">
  <name>Task 2: Run baseline measurement</name>
  <files>tests/ashrae_140_blind_validation.rs, .planning/baseline/</files>
  <action>
Run the blind validation test and capture the full output:

```bash
cargo test --test ashrae_140_blind_validation -- --nocapture 2>&1 | tee /tmp/blind_baseline.log
```

Analyze the results and create .planning/baseline/BLIND_BASELINE_RESULTS.md with:

### Summary Statistics
- Total cases: N
- Passed: N (X%)
- Failed: N (X%)
- Mean Absolute Error by metric type

### Per-Case Results Table
| Case | Annual Heating | Annual Cooling | Peak Heating | Peak Cooling | Free-Float Min/Max | Status |
|------|----------------|----------------|--------------|--------------|---------------------|--------|
| 600 | X% error | X% error | X% error | X% error | N/A | FAIL |
| 900 | X% error | X% error | X% error | X% error | X% error | FAIL |

### Failure Magnitude Analysis
For each failing case, document:
- Which metrics fail and by how much
- Error magnitude as % of reference mean
- Pattern analysis: do 600 series fail differently than 900 series?

### Key Observations
- Solar distribution: do heavy-mass cases show systematic cooling over-prediction?
- Thermal mass: do time-constant related metrics (peak, diurnal swing) show patterns?
- Free-floating: how extreme are the temperature failures?
  </action>
  <verify>
    File exists: .planning/baseline/BLIND_BASELINE_RESULTS.md
    Contains: Summary statistics, per-case table, failure magnitude analysis
  </verify>
  <done>Baseline failure state documented with per-case, per-metric error magnitudes</done>
</task>

<task type="checkpoint:human-verify" gate="blocking">
  <what-built>Blind validation harness and baseline measurement</what-built>
  <how-to-verify>
    1. Review .planning/baseline/BLIND_BASELINE_RESULTS.md
    2. Confirm 0% pass rate (or near-zero) for blind mode
    3. Review failure magnitude patterns — are they consistent with known issues?
    4. Check that free-floating temperature failures are extreme (expected: 125°C vs 41-46°C)

    Commands to verify:
    - cargo test --test ashrae_140_blind_validation -- --nocapture
    - cat .planning/baseline/BLIND_BASELINE_RESULTS.md
  </how-to-verify>
  <resume-signal>
    Type "approved" to confirm baseline is documented correctly, or describe what needs adjustment
  </resume-signal>
</task>

</tasks>

<verification>
Run: cargo test --test ashrae_140_blind_validation 2>&1

Verify:
- Test executes without panics
- Baseline results are captured in .planning/baseline/BLIND_BASELINE_RESULTS.md
</verification>

<success_criteria>
- Blind validation harness runs successfully (deterministic, no crashes)
- True baseline failure state is documented
- Per-case, per-metric error magnitudes are recorded
- Failure patterns are identified (solar, thermal mass, free-float categories)
</success_criteria>

<output>
After completion, create `.planning/phases/A-baseline-stripping/A-02-SUMMARY.md`
</output>