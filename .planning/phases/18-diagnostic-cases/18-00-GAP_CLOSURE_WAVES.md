# Phase 18 Gap Closure Wave Analysis

**Created:** 2026-03-14
**Status:** Gap closure plans created

## Wave Structure

### Wave 4: Energy Calculation & Equipment Specification Fixes (Parallel)

Plans in this wave can run in parallel as they target different issues:

**18-12: Fix Cases 802-810 Energy Calculation Method**
- Target: Lines 166, 228, 284, 340, 396, 452, 514, 576, 638 in tests/ashrae_140_cases_800_810.rs
- Action: Replace `get_heating_energy_kwh() + get_cooling_energy_kwh()` with `get_electrical_energy_kwh()`
- Dependencies: 18-11 (electrical energy calculation bug fix)
- Duration: ~30 minutes (9 tasks, straightforward code changes)
- Autonomous: Yes (no checkpoints)

**18-13: Fix Case 801 Equipment Specification**
- Target: Case 801 COP issue (3.0 outside expected 3.2-4.2 range)
- Action: Update equipment specification in src/validation/ashrae_140_cases.rs
- Dependencies: 18-11 (electrical energy calculation bug fix)
- Duration: ~45 minutes (4 tasks, includes decision checkpoint)
- Autonomous: Yes (decision checkpoint handled)
- Risk: May require ASHRAE 140 specification research (Option C)

### Wave 5: Thermal Load Investigation (Sequential)

Plan in this wave depends on Wave 4 completion:

**18-14: Investigate Thermal Load Calculation Bug**
- Target: calc_analytical_loads() returning 155 MW instead of ~7.5 kW
- Action: Investigate root cause and either fix or document for Phase 20
- Dependencies: 18-12, 18-13 (energy calculation fixes must be in place first)
- Duration: ~60 minutes (5 tasks, includes investigation and decision checkpoints)
- Autonomous: Yes (decision checkpoints handled)
- Outcome: Either fix in Phase 18 or defer to Phase 20 with documentation

## Dependency Graph

```
Wave 4 (Parallel):
  18-11 (electrical energy fix) ──┬──> 18-12 (Cases 802-810 energy method)
                                 └──> 18-13 (Case 801 COP issue)

Wave 5 (Sequential):
  18-12 + 18-13 ──> 18-14 (thermal load investigation)
```

## Execution Strategy

### Recommended Execution Order

1. **Wave 4 (Parallel):**
   - Execute 18-12 and 18-13 in parallel
   - 18-12 is straightforward code change (no decisions)
   - 18-13 has one decision checkpoint (Task 1) - may need user input
   - Expected total time: ~45 minutes (if 18-13 decision is quick)

2. **Wave 5 (Sequential):**
   - Execute 18-14 after Wave 4 completes
   - 18-14 has two decision checkpoints (Tasks 1, 3)
   - Outcome uncertain - may fix bug or defer to Phase 20
   - Expected total time: ~60 minutes

### Parallel Execution Command

To execute Wave 4 plans in parallel (if 18-13 decision is resolved quickly):

```bash
# Terminal 1: Execute 18-12
/gsd:execute-phase 18 --plan 12

# Terminal 2: Execute 18-13 (if decision is Option A or B, not C)
/gsd:execute-phase 18 --plan 13
```

### Sequential Execution Command

To execute plans sequentially (safer, handles decision checkpoints):

```bash
/gsd:execute-phase 18 --plan 12
/gsd:execute-phase 18 --plan 13
/gsd:execute-phase 18 --plan 14
```

## Gap Closure Progress

### Before Gap Closure Plans

**Status from 18-VERIFICATION.md (2026-03-14T19:30:00Z):**
- Score: 4/5 must-haves verified (80% completion)
- Case 800: PASSING
- Case 801: FAILING (COP 3.0 outside 3.2-4.2)
- Cases 802-810: FAILING (wrong energy calculation method, thermal load bug)

### Expected After Plan 18-12

**Cases 802-810 Energy Calculation Fixed:**
- Cases 802-810 will use `get_electrical_energy_kwh()` instead of thermal energy sum
- Expected improvement: Tests may still fail due to thermal load bug (65 MWh vs 10-20 MWh)
- Score: 4/5 must-haves verified (still 80% - thermal load bug remains)

### Expected After Plan 18-13

**Case 801 Equipment Specification Fixed:**
- Case 801 COP will fall within 3.2-4.2 range (if Option A selected)
- Case 801 will have higher efficiency than Case 800 (two-stage vs single-stage)
- Expected improvement: Case 801 passes if COP fix is correct
- Score: 4.5/5 must-haves verified (90% - only thermal load bug remains)

### Expected After Plan 18-14

**Thermal Load Bug: Either Fixed or Documented**

**Option A: Fixed in Phase 18**
- calc_analytical_loads() returns realistic values (~7.5 kW average)
- Cases 802-810 electrical energy falls within 10-20 MWh range
- All Cases 800-810 pass
- Score: 5/5 must-haves verified (100% - Phase 18 complete)

**Option B: Documented and Deferred to Phase 20**
- Thermal load bug comprehensively documented in docs/KNOWN_ISSUES.md
- ROADMAP.md Phase 20 updated to include thermal load fix
- Cases 802-810 still fail (thermal load bug not fixed)
- Score: 4.5/5 must-haves verified (90% - DIAG-02 partially complete)
- Recommendation: Move to Phase 19 (Statistical Validation) and return in Phase 20

## Risk Assessment

### Plan 18-12 (Low Risk)
- **Risk:** Code change is straightforward, already done for Cases 800-801
- **Mitigation:** Follow exact pattern from Cases 800-801 (lines 50, 104)
- **Confidence:** HIGH - simple find-and-replace operation

### Plan 18-13 (Medium Risk)
- **Risk:** May require ASHRAE 140 specification research (Option C)
- **Mitigation:** If Option C selected, document assumption and proceed with reasonable values
- **Confidence:** MEDIUM - depends on decision in Task 1

### Plan 18-14 (High Risk - Outcome Uncertain)
- **Risk:** Thermal load bug may be complex thermal network issue
- **Mitigation:** Comprehensive investigation in Task 2 before deciding fix vs defer
- **Confidence:** LOW - root cause unknown until investigation

## Success Criteria

### Phase 18 Complete (All 5 must-haves verified)

**Truths verified:**
1. ASHRAE 140 Cases 195-470 implemented and produce validation results
2. ASHRAE 140 Cases 800-810 validate equipment efficiency and control strategies
3. Non-residential cases extend validation beyond residential buildings
4. Solid conduction and solar gain variants expose edge cases
5. CLI integration allows validation of diagnostic cases

**Requirements satisfied:**
- DIAG-01: Cases 195-470 ✅
- DIAG-02: Cases 800-810 ✅ (after gap closure)
- DIAG-03: Non-residential cases ✅
- DIAG-04: Solid conduction variants ✅
- DIAG-05: Solar gain variants ✅

### Phase 18 Partial Complete (4/5 must-haves verified)

**If thermal load bug deferred to Phase 20:**
- DIAG-02 remains partially satisfied (Cases 800-801 pass, 802-810 fail)
- Recommendation: Proceed to Phase 19, return to thermal load in Phase 20
- Justification: Thermal load bug is deep physics issue, better addressed in data quality phase

## Next Steps

### After Wave 4 Completion (18-12, 18-13)
1. Run HVAC equipment tests: `cargo test --test ashrae_140_cases_800_810`
2. Verify Cases 800-801 pass, Cases 802-810 may still fail (thermal load bug)
3. Review 18-12 and 18-13 SUMMARY files
4. Proceed to Wave 5 (18-14)

### After Wave 5 Completion (18-14)
1. If thermal load fixed: Run full validation, verify all Cases 800-810 pass
2. If thermal load deferred: Review KNOWN_ISSUES.md, update ROADMAP.md Phase 20
3. Run phase verification: `/gsd:verify-work 18`
4. If 5/5 must-haves verified: Proceed to Phase 19
5. If 4/5 must-haves verified: Decision on whether to proceed to Phase 19 or continue Phase 18

---

*Wave analysis created: 2026-03-14*
*Total gap closure plans: 3 (18-12, 18-13, 18-14)*
