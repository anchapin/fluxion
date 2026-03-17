# Phase 25: Alternative Physics Implementation - Summary

**Status:** PLANNING COMPLETE
**Plans Created:** 7 plans (25-00 through 25-06)
**Requirements Addressed:** ALT-01, ALT-02, ALT-03, ALT-04, ALT-05
**Ready for Execution:** Yes

---

## Phase Goal

Implement and test alternative physics approaches for high-mass building thermal modeling. Achieve ±15% annual energy accuracy for Case 900 (currently 229-322% error with 5R1C).

---

## Plans Overview

| Plan | Name | Type | Effort | Status |
|------|------|------|--------|--------|
| 25-00 | State-of-the-Art Literature Review | Research | 2-3 days | Pending |
| 25-01 | EnergyPlus SDK Installation and Data Generation | Infrastructure | 3-4 days | Pending |
| 25-02 | Adaptive Timestep Implementation | Implementation | 1-2 weeks | Pending |
| 25-03 | Finite Difference Method Implementation | Implementation | 2-3 weeks | Pending |
| 25-04 | Conduction Transfer Functions (CTF) Implementation | Implementation | 3-4 weeks | Pending |
| 25-05 | Hybrid RC + ML Correction | Implementation | 2 weeks | Pending |
| 25-06 | Comparative Evaluation of All Approaches | Evaluation | 1 week | Pending |

**Total Estimated Effort:** 10-14 weeks (sequential) / 6-8 weeks (parallel waves)

---

## Execution Strategy

### Wave 1: Foundation (Week 1-2)
**Parallel execution:**
- Plan 25-00: Literature review (research)
- Plan 25-01: EnergyPlus data generation (infrastructure)

**Gate:** Both plans complete before Wave 2

### Wave 2: Implementation (Week 2-6)
**Parallel execution:**
- Plan 25-02: Adaptive timestep (quick win)
- Plan 25-03: Finite difference (core physics)
- Plan 25-04: CTF (core physics)
- Plan 25-05: ML correction (surrogate)

**Gate:** All implementations functional before Wave 3

### Wave 3: Evaluation (Week 6-8)
**Sequential:**
- Plan 25-06: Comparative evaluation and recommendation

**Gate:** Recommendation approved before Phase 26

---

## Requirements Traceability

| Requirement | Plans Addressing | Status |
|-------------|------------------|--------|
| ALT-01: Finite difference method | 25-00, 25-03, 25-06 | Pending |
| ALT-02: Conduction Transfer Functions | 25-00, 25-04, 25-06 | Pending |
| ALT-03: Adaptive timestep | 25-00, 25-02, 25-06 | Pending |
| ALT-04: Frequency-domain analysis | 25-00 (literature) | Pending |
| ALT-05: Hybrid RC + ML | 25-00, 25-01, 25-05, 25-06 | Pending |

---

## Success Criteria (Phase-Level)

- [ ] ✅ Literature review completed (5 documents, 15+ sources)
- [ ] ✅ EnergyPlus data generated (20+ simulations, 100+ MB data)
- [ ] ✅ Adaptive timestep implemented and validated
- [ ] ✅ Finite difference implemented and validated
- [ ] ✅ CTF implemented and validated
- [ ] ✅ ML correction implemented and validated
- [ ] ✅ Case 900 accuracy: at least one approach achieves ±15%
- [ ] ✅ Comprehensive evaluation completed (all approaches compared)
- [ ] ✅ Clear recommendation for Phase 26

---

## Key Deliverables

### Research
- `docs/literature/CTF_STATE_OF_THE_ART.md`
- `docs/literature/FINITE_DIFFERENCE_STATE_OF_THE_ART.md`
- `docs/literature/STATE_SPACE_ADMITTANCE_REVIEW.md`
- `docs/literature/ENERGYPLUS_CONDUCTION_ANALYSIS.md`
- `docs/literature/THERMAL_MODELING_COMPARISON.md`

### Infrastructure
- EnergyPlus v23.1+ installed
- `tests/energyplus_data/` (Case 900 IDF + results)
- `tests/energyplus_data/case_900_mass_sweep.csv`
- `tests/energyplus_data/case_900_timestep_sweep.csv`
- `tests/energyplus_data/case_900_construction_sweep.csv`
- `tests/energyplus_data/case_900_hourly_profiles.csv`

### Implementation
- `src/sim/adaptive_timestep.rs`
- `src/physics/fd_discretization.rs`, `src/physics/fd_solver.rs`
- `src/physics/ctf_coefficients.rs`, `src/physics/ctf_solver.rs`
- `src/ml/residual_correction.rs` + trained model

### Validation
- `tests/test_adaptive_timestep.rs`
- `tests/test_fd_case_900.rs`
- `tests/test_ctf_case_900.rs`
- `tests/test_ml_corrected_case_900.rs`
- `docs/ACCURACY_COMPARISON_CASE_900.md`
- `docs/PHASE_25_RECOMMENDATION.md`

---

## Technical Approach Summary

### 1. Adaptive Timestep (25-02)
**Idea:** Reduce timestep for high-mass buildings (1hr → 6min) to improve numerical integration accuracy.

**Expected:**
- Accuracy: 50-100% error (improvement from 229-322%, but not ±15%)
- Performance: ~400-600 configs/sec (4-5× slower)
- Complexity: Low (minimal physics changes)

**Role:** Baseline fallback, quick validation that timestep contributes to error

---

### 2. Finite Difference (25-03)
**Idea:** Replace lumped capacitance with 1D heat conduction through wall layers.

**Expected:**
- Accuracy: ±5-10% (target met)
- Performance: ~500-800 configs/sec (3-5× slower)
- Complexity: High (new physics implementation)

**Role:** High-accuracy option, physically interpretable

---

### 3. CTF - Conduction Transfer Functions (25-04)
**Idea:** Precompute frequency-domain response coefficients (EnergyPlus approach).

**Expected:**
- Accuracy: ±5-10% (target met, EnergyPlus-proven)
- Performance: ~800-1,200 configs/sec (2-3× slower)
- Complexity: Very High (complex coefficient calculation)

**Role:** Best accuracy + performance balance, industry standard

---

### 4. Hybrid RC + ML Correction (25-05)
**Idea:** Keep fast 5R1C physics, train ML to predict residual error.

**Expected:**
- Accuracy: ±15-25% (partial target)
- Performance: ~2,000-2,300 configs/sec (minimal slowdown)
- Complexity: Medium (ML training + integration)

**Role:** Fastest option, good for population optimization

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| CTF coefficient calculation fails | Medium | High | Fallback to finite difference |
| ML model doesn't generalize | Medium | Medium | Train on diverse data, validate on unseen cases |
| Performance too slow (<300 configs/sec) | Low | Medium | Optimize hot paths, reduce node count |
| None achieve ±15% accuracy | Low | High | Combine approaches (e.g., CTF + ML correction) |
| EnergyPlus installation issues | Low | Low | Use OpenStudio SDK alternative |

---

## Decision Gates

### Gate 1: After Plan 25-01 (EnergyPlus Data)
**Decision:** Is EnergyPlus data quality sufficient for training/validation?
- **Yes:** Proceed to Wave 2
- **No:** Fix data generation issues before continuing

### Gate 2: After Plan 25-06 (Evaluation)
**Decision:** Which approach to use for Phase 26?
- **Option A:** CTF (recommended if accuracy + performance balance)
- **Option B:** Finite difference (if CTF fails or robustness prioritized)
- **Option C:** ML correction (if speed prioritized over ±15% accuracy)
- **Option D:** Hybrid (CTF + ML correction for maximum accuracy)

### Gate 3: Phase 26 Go/No-Go
**Decision:** Proceed to Phase 26 full validation or iterate on Phase 25?
- **Go:** At least one approach achieves ±15% for Case 900
- **No-Go:** Iterate on best approach (tune parameters, fix issues)

---

## Phase 26 Preview

**Goal:** Comprehensive validation of recommended approach against all 18 ASHRAE 140 cases.

**Plans (tentative):**
- 26-01: Low-mass validation (600-series, 800-series) - no regression check
- 26-02: High-mass validation (900-series) - accuracy check
- 26-03: Monthly energy validation - ±10% tolerance
- 26-04: Performance benchmarking - ≥1,000 configs/sec
- 26-05: Documentation and migration guide

**Success Criteria:**
- All 18 cases within ±15% annual energy
- Low-mass cases unchanged (±5% from 5R1C baseline)
- Performance ≥1,000 configs/sec
- Complete documentation

---

## Notes

- **User Priority:** Accuracy first, performance second
- **Strategy:** Implement multiple approaches in parallel, select best via data-driven comparison
- **Timeline:** 6-8 weeks for full Phase 25 (parallel waves)
- **Next Step:** Execute Wave 1 (25-00, 25-01) to establish foundation

---

*Phase Summary created: 2026-03-17*
*Plans created: 7 (25-00 through 25-06)*
*Status: PLANNING COMPLETE - Ready for execution*
