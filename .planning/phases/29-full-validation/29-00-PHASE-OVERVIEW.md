# Phase 29: Full ASHRAE 140 Validation

**Milestone:** v0.7 Thermal Physics Complete
**Phase Number:** 29
**Status:** READY TO EXECUTE
**Created:** 2026-03-18
**Dependencies:** Phase 28 COMPLETE (CTF/FD Solver Integration)

---

## Executive Summary

Phase 29 validates all 18 ASHRAE 140 test cases with the new CTF/FD solver integration from Phase 28. The goal is to achieve ±15% tolerance for annual energy across all cases, with special focus on improving high-mass accuracy (Case 900 heating: currently ~31% error with 5R1C).

**9 Plans organized into 3 waves:**
- **Wave 1 (Days 1-5):** High-mass validation (900-series cases)
- **Wave 2 (Days 6-10):** Regression testing (600-series, 800-series, free-float)
- **Wave 3 (Days 11-15):** Performance benchmarking and reporting

---

## Phase Goal

Validate all 18 ASHRAE 140 cases with CTF/FD solver integration. Achieve ±15% tolerance for annual energy across all cases while maintaining ≥800 configs/sec throughput.

---

## Success Criteria

**All 6 must be TRUE for Phase 29 to be COMPLETE:**

1. ✅ **Case 900 annual heating:** 1.17-2.04 MWh (±15% of reference ~1.60 MWh)
2. ✅ **Case 900 annual cooling:** 2.13-3.67 MWh (±15% of reference ~2.90 MWh)
3. ✅ **All 900-series cases:** Within ±15% tolerance (9 cases total)
4. ✅ **No regression on 600-series:** Maintain ±15% tolerance (8 cases)
5. ✅ **No regression on 800-series:** Maintain ±15% tolerance (8 cases)
6. ✅ **Performance ≥800 configs/sec:** Throughput with CTF/FD for high-mass

---

## Plans Summary

| Plan | Name | Wave | Effort | Status |
|------|------|------|--------|--------|
| [29-01](./29-01-PLAN.md) | 900-Series Validation Suite | 1 | 2-3 days | READY |
| [29-02](./29-02-PLAN.md) | Case 900 Deep Dive | 1 | 2-3 days | READY |
| [29-03](./29-03-PLAN.md) | Case 960 Multi-Zone Validation | 1 | 2-3 days | READY |
| [29-04](./29-04-PLAN.md) | 600-Series Regression Suite | 2 | 1-2 days | READY |
| [29-05](./29-05-PLAN.md) | 800-Series Regression Suite | 2 | 1-2 days | READY |
| [29-06](./29-06-PLAN.md) | Free-Float Cases Validation | 2 | 1-2 days | READY |
| [29-07](./29-07-PLAN.md) | Performance Benchmarking | 3 | 1-2 days | READY |
| [29-08](./29-08-PLAN.md) | Validation Report Generation | 3 | 1-2 days | READY |
| [29-09](./29-09-PLAN.md) | Human Verification & Documentation | 3 | 1 day | READY |

---

## Execution Strategy

### Wave 1: High-Mass Validation (Days 1-5)

**Focus:** Run all 900-series cases with CTF/FD enabled

**Plans:**
- 29-01: 900-Series Validation Suite (all 8 cases)
- 29-02: Case 900 Deep Dive (heating/cooling analysis)
- 29-03: Case 960 Multi-Zone Validation (complex sunspace case)

**Parallel Execution:** All 3 plans can run simultaneously

**Deliverables:**
- Results table for all 900-series cases
- Case 900 analysis (5R1C vs CTF vs FD)
- Case 960 multi-zone validation

**Success:** Case 900 within ±15%, all 900-series passing

---

### Wave 2: Regression Testing (Days 6-10)

**Focus:** Ensure no regression on low-mass and internal load cases

**Plans:**
- 29-04: 600-Series Regression Suite (low-mass)
- 29-05: 800-Series Regression Suite (internal loads)
- 29-06: Free-Float Cases Validation (no HVAC)

**Parallel Execution:** All 3 plans can run simultaneously

**Deliverables:**
- Results tables for 600-series, 800-series, free-float
- Comparison vs v0.6 baseline
- Regression analysis

**Success:** All cases maintain ±15% tolerance (no regression)

---

### Wave 3: Performance & Reporting (Days 11-15)

**Focus:** Benchmark performance and generate comprehensive report

**Plans:**
- 29-07: Performance Benchmarking (throughput, latency)
- 29-08: Validation Report Generation (visualizations, analysis)
- 29-09: Human Verification & Documentation (final review)

**Sequential Execution:** 29-07 → 29-08 → 29-09

**Deliverables:**
- Performance benchmark results
- Comprehensive validation report (Markdown + plots)
- Phase 29 summary document
- Updated project state files

**Success:** Report complete, state files updated, Phase 29 marked COMPLETE

---

## Requirements Traceability

This phase satisfies v0.6 requirements:

| Requirement | Description | Plans | Status |
|-------------|-------------|-------|--------|
| **VAL-10** | Case 900 annual heating within ±15% | 29-01, 29-02 | PENDING |
| **VAL-11** | Case 900 annual cooling within ±15% | 29-01, 29-02 | PENDING |
| **VAL-12** | No regression on low-mass cases | 29-04, 29-05 | PENDING |
| **VAL-13** | Performance maintained (≥800 configs/sec) | 29-07 | PENDING |
| **VAL-14** | All 18 ASHRAE 140 cases pass validation | 29-01, 29-04, 29-05, 29-06 | PENDING |
| **VAL-15** | Performance benchmarking complete | 29-07, 29-08 | PENDING |

---

## Risk Management

### HIGH RISK: CTF/FD doesn't improve accuracy

**Probability:** Medium
**Impact:** High (phase goal not achieved)

**Mitigation:**
- Fall back to 5R1C with thermal mass correction
- Document limitations, recommend ±20% tolerance
- Investigate alternative physics (ML correction, hybrid models)

**Contingency:**
- Adjust success criteria to ±20% for high-mass cases
- Focus on monthly energy accuracy (not just annual)
- Document as known limitation for v1.0

---

### MEDIUM RISK: Performance below 800 configs/sec

**Probability:** Medium
**Impact:** Medium (throughput target not met)

**Mitigation:**
- Optimize CTF coefficient precomputation
- Reduce FD mesh resolution
- Profile and optimize hot paths

**Contingency:**
- Accept 500-800 configs/sec if accuracy achieved
- Document performance vs accuracy tradeoff
- Recommend 5R1C for optimization workflows

---

### MEDIUM RISK: Case 960 multi-zone complexity

**Probability:** Medium
**Impact:** Medium (single case fails)

**Mitigation:**
- Test each zone independently first
- Verify inter-zone coupling logic
- Simplify sunspace control if needed

**Contingency:**
- Document Case 960 as known limitation
- Focus on single-zone cases for v0.7
- Plan multi-zone improvements for v1.0

---

### LOW RISK: Regression on low-mass cases

**Probability:** Low
**Impact:** High (breaks existing accuracy)

**Mitigation:**
- Automatic method selection ensures 5R1C for τ < 2h
- Comprehensive regression testing (Plans 29-04, 29-05)
- Compare vs v0.6 baseline

**Contingency:**
- Fix method selection logic immediately
- Disable CTF/FD for low-mass cases
- Prioritize regression fix over high-mass improvements

---

## Deliverables

### Code & Tests
- [ ] Validation test results (all 18 cases)
- [ ] Performance benchmark results
- [ ] Regression test suite (updated)

### Documentation
- [ ] 29-01-SUMMARY.md through 29-07-SUMMARY.md (plan summaries)
- [ ] 29-08-VALIDATION-REPORT.md (comprehensive report)
- [ ] 29-00-PHASE-SUMMARY.md (phase summary)
- [ ] Visualizations (4+ plots in docs/)

### Project State
- [ ] STATE.md updated (Phase 29 marked COMPLETE)
- [ ] REQUIREMENTS.md updated (VAL-10 through VAL-15 complete)
- [ ] ROADMAP.md updated (Phase 29 marked complete)

---

## Exit Criteria

**Phase 29 is COMPLETE when:**

1. ✅ All 9 plans executed and verified
2. ✅ All 6 success criteria met (or gaps documented)
3. ✅ Validation report generated and reviewed
4. ✅ Phase 29 summary document written
5. ✅ Project state files updated (STATE.md, REQUIREMENTS.md, ROADMAP.md)
6. ✅ Ready to plan Phase 30 (Documentation & Release)

**Next Phase:** [Phase 30](../30-documentation-release/) — Documentation & Release

---

## Resources

### Reference Documents
- [ASHRAE 140 Standard](../../docs/ASHRAE140_VALIDATION.md)
- [CTF/FD Solver Integration (Phase 28)](../28-ctf-fd-solver-integration/28-00-PHASE-SUMMARY.md)
- [v0.6 Validation Results](../../ASHRAE_140_PROGRESS.md)

### Test Commands
```bash
# Run all ASHRAE 140 cases
cargo test --release -- test_ashrae_140_comprehensive_regression --nocapture

# Run specific case
cargo test --release -- ashrae_140_case_900 --nocapture

# Run performance benchmarks
cargo bench --release -- benchmark
```

### Key Files
- Validator: `src/validation/ashrae_140_validator.rs`
- Test suite: `tests/integration/test_ashrae_140_regression.rs`
- Case definitions: `src/validation/ashrae_140_cases.rs`
- CTF solver: `src/physics/ctf_solver.rs`
- FD solver: `src/physics/fd_solver.rs`

---

*Phase overview created: 2026-03-18*
*Ready to execute: YES*
*Next action: /gsd:execute-phase 29*
