---
phase: D-blind-validation-pass
plan: D-01
type: execute
wave: 1
depends_on: ["C-01"]
files_modified:
  - tests/ashrae_140_blind_validation.rs
autonomous: false
requirements:
  - VALIDATE-01

must_haves:
  truths:
    - "80%+ of ASHRAE 140 cases pass all tolerance bands in blind mode"
  artifacts:
    - path: ".planning/validation/BLIND_VALIDATION_RESULTS.md"
      provides: "Full validation results with pass/fail breakdown"
---

<objective>
Run the complete ASHRAE 140 blind validation suite and achieve 80%+ pass rate. This is the final validation gate — all physics fixes from Phase B and benchmark corrections from Phase C must result in ≥80% of cases passing all tolerance bands.
</objective>

<context>
From Phase A: Baseline was 0% pass rate without corrections.
From Phase B: Solar, thermal mass, and free-float fixes should have improved performance.
From Phase C: Benchmark now uses true reference values.

The target is 80%+ pass rate — meaning at least 46 of 58 cases must pass all metrics.
</context>

<tasks>

<task type="auto">
  <name>Task 1: Run full blind validation suite</name>
  <files>tests/ashrae_140_blind_validation.rs</files>
  <action>
Run the complete blind validation suite:

```bash
cargo test --test ashrae_140_blind_validation -- --nocapture 2>&1 | tee /tmp/blind_validation_results.log
```

Capture:
- Per-case results (passed/failed per metric)
- Overall pass rate
- Mean Absolute Error by metric type
- Failure patterns — which cases still fail and by how much

Generate .planning/validation/BLIND_VALIDATION_RESULTS.md with full results table:

| Case | Annual Heating | Annual Cooling | Peak Heating | Peak Cooling | Free-Float | Status |
|------|----------------|----------------|--------------|--------------|------------|--------|
| 600 | ±X% | ±X% | ±X% | ±X% | N/A | PASS/FAIL |
| ... | ... | ... | ... | ... | ... | ... |

Summary:
- Total cases: N
- Passed: N (X%)
- Failed: N (X%)
- MAE by metric
  </action>
  <verify>
    File exists: .planning/validation/BLIND_VALIDATION_RESULTS.md
    Contains: Full results table, pass rate, MAE breakdown
  </verify>
  <done>Full blind validation results documented</done>
</task>

<task type="checkpoint:human-verify" gate="blocking">
  <what-built>Full ASHRAE 140 blind validation results</what-built>
  <how-to-verify>
    Review .planning/validation/BLIND_VALIDATION_RESULTS.md:

    1. Check overall pass rate — must be ≥80% for success
    2. Review failing cases — are they consistent with known limitations?
    3. Check if failures are in categories (solar, thermal mass, free-float) or scattered
    4. Confirm the validation is truly blind (no case-ID hints in output)

    If pass rate < 80%:
    - Document remaining failure patterns
    - Identify if additional physics fixes are needed (return to Phase B)
    - Or document architectural limitation if physics cannot be fixed
  </how-to-verify>
  <resume-signal>
    If ≥80%: Type "approved" to proceed to Phase E
    If <80%: Type "continue to Phase B fixes" with specific cases to address
  </resume-signal>
</task>

</tasks>

<verification>
Run: cargo test --test ashrae_140_blind_validation

Verify:
- All cases executed (no crashes)
- Results are deterministic (run twice, same results)
- Pass rate computed correctly
</verification>

<success_criteria>
- Full suite executed without crashes
- Pass rate ≥ 80% achieved OR documented reason why not achievable
- Failing cases documented with root cause analysis
</success_criteria>

<output>
After completion, create `.planning/phases/D-blind-validation-pass/D-01-SUMMARY.md`
</output>