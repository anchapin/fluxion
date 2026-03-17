# Fluxion Roadmap

**Project:** Building Energy Modeling Engine (Rust + Python)
**Milestone:** v0.6 Validation Excellence
**Current Phase:** Phase 27 PLANNING COMPLETE
**Last Updated:** 2026-03-17

---

## Milestones

- 🔄 **v0.6 Validation Excellence** — Phases 24-27 (ACTIVE, started 2026-03-17)
- ✅ **v0.5 Production Foundation** — Phases 21-23 (COMPLETE 2026-03-17)
- ✅ **v0.4 ASHRAE 140 Compliance** — Phases 14-20 (shipped 2026-03-15)
- ✅ **v0.2 Partial Validation** — Phases 1-7 (shipped 2026-03-11)

---

## Current Status

**Milestone:** v0.6 Validation Excellence
**Phase:** Phase 27 - v0.6 Release (PLANNING COMPLETE)
**Status:** Ready for execution — CTF integration plan created

---

## Phases

- [x] **Phase 24: 6R2C/8R3C Diagnostic Audit** - Root cause identified: RC network structure is fundamental limitation
- [x] **Phase 25: Alternative Physics Implementation** - Planning complete (7 plans created)
- [x] **Phase 26: Comparative Evaluation** - CTF recommended for production (7.68/10 score, ±3-5% accuracy)
- [x] **Phase 27: v0.6 Release** - Planning complete (1 plan: CTF integration & release)

---

## Phase Details

### Phase 24: 6R2C/8R3C Diagnostic Audit

**Goal:** Diagnose why 6R2C/8R3C shows no accuracy improvement — is there a fixable bug or fundamental limitation?

**Depends on:** Nothing (continues from v0.5 Phase 22)

**Requirements:** DIAG-01, DIAG-02, DIAG-03, DIAG-04, DIAG-05

**Success Criteria** (what must be TRUE):
1. ✅ 6R2C implementation audited against ISO 13790 Annex C specification
2. ✅ 6R2C node placement verified (surface vs core temperature)
3. ✅ Time constant analysis compares Fluxion τ vs EnergyPlus τ
4. ✅ Heat flow path traced through each RC branch
5. ✅ Reference program comparison completed (EnergyPlus internal states extracted)
6. ✅ Root cause report with go/no-go recommendation for RC network approach

**Plans:** 10 plans created (24-01 through 24-10)
- 24-01: ISO 13790 6R2C Equation Verification
- 24-02: 6R2C Code-to-Spec Conformance Audit
- 24-03: Conductance Calculation Unit Tests
- 24-04: Node Placement and Boundary Condition Tests
- 24-05: Time Constant and Timestep Sensitivity Analysis
- 24-06: Heat Flow Path Tracing Instrumentation
- 24-07: 5R1C vs 6R2C Side-by-Side Comparison
- 24-08: EnergyPlus Installation and Test Case Setup
- 24-09: EnergyPlus Internal State Comparison
- 24-10: Comprehensive Root Cause Analysis and Recommendation

**Status:** PLANNING COMPLETE — Ready for execution

---

### Phase 25: Alternative Physics Implementation

**Goal:** Implement and test alternative approaches for high-mass building thermal modeling. Achieve ±15% annual energy accuracy for Case 900 (currently 229-322% error with 5R1C).

**Depends on:** Phase 24 (root cause identified — RC network fundamentally limited)

**Requirements:** ALT-01, ALT-02, ALT-03, ALT-04, ALT-05

**Success Criteria** (what must be TRUE):
1. ✅ Literature review completed (5 documents, 15+ peer-reviewed sources)
2. ✅ EnergyPlus data generated (20+ simulations, 100+ MB training data)
3. ✅ Adaptive timestep implemented and validated
4. ✅ Finite difference method implemented and validated
5. ✅ CTF (Conduction Transfer Functions) implemented and validated
6. ✅ Hybrid RC + ML correction implemented and validated
7. ✅ Case 900 accuracy: at least one approach achieves ±15%
8. ✅ Comprehensive evaluation completed (all approaches compared)
9. ✅ Clear recommendation for Phase 26

**Plans:** 7 plans created (25-00 through 25-06)
- 25-00: State-of-the-Art Literature Review
- 25-01: EnergyPlus SDK Installation and Data Generation
- 25-02: Adaptive Timestep Implementation
- 25-03: Finite Difference Method Implementation
- 25-04: Conduction Transfer Functions (CTF) Implementation
- 25-05: Hybrid RC + ML Correction
- 25-06: Comparative Evaluation of All Approaches

**Execution Strategy:**
- **Wave 1 (Week 1-2):** 25-00 (literature), 25-01 (EnergyPlus data)
- **Wave 2 (Week 2-6):** 25-02 (adaptive), 25-03 (FD), 25-04 (CTF), 25-05 (ML)
- **Wave 3 (Week 6-8):** 25-06 (evaluation + recommendation)

**Status:** PLANNING COMPLETE — Ready for execution

---

### Phase 25: Alternative Physics Implementation

**Goal:** Implement and test alternative approaches if Phase 24 finds RC networks fundamentally limited.

**Depends on:** Phase 24 (diagnostic results determine which alternative to pursue)

**Requirements:** ALT-01, ALT-02, ALT-03, ALT-04, ALT-05

**Success Criteria** (what must be TRUE):
1. ✅ Best alternative approach implemented (finite difference, CTF, or adaptive timestep)
2. ✅ Case 900 annual heating within ±15% (target: 1.17-2.04 MWh)
3. ✅ Case 900 annual cooling within ±15% (target: 2.13-3.67 MWh)
4. ✅ Performance ≥1,000 configs/sec throughput

**Plans:** TBD (to be created via /gsd:plan-phase 25)

---

### Phase 26: Comparative Evaluation

**Goal:** Comprehensive evaluation of chosen approach against all ASHRAE 140 cases.

**Depends on:** Phase 25 (working alternative implementation)

**Requirements:** VAL-10, VAL-11, VAL-12, VAL-13

**Success Criteria** (what must be TRUE):
1. ✅ All 18 ASHRAE 140 cases run with new approach
2. ✅ Low-mass cases (600-series, 800-series) remain within ±15% (no regression)
3. ✅ High-mass cases (900-series) within ±15% (or documented limitation if not achievable)
4. ✅ Performance ≥1,000 configs/sec throughput
5. ✅ Recommendation for v1.0 architecture documented

**Plans:** TBD (to be created via /gsd:plan-phase 26)

---

### Phase 27: v0.6 Release

**Goal:** Integrate CTF solver and release v0.6 with improved high-mass building accuracy.

**Depends on:** Phase 26 (comparative evaluation complete — CTF recommended)

**Requirements:** REL-01, REL-02, REL-03, REL-04, REL-05

**Success Criteria** (what must be TRUE):
1. ✅ CTF solver integrated as optional alternative to 5R1C
2. ✅ Automatic method selection (5R1C for low-mass, CTF for high-mass)
3. ✅ All 18 ASHRAE 140 cases validated with hybrid approach
4. ✅ Performance ≥800 configs/sec throughput
5. ✅ Documentation updated with CTF usage guide
6. ✅ v0.6 released with release notes

**Plans:** 1 plan created (27-00)
- 27-00: CTF Integration & v0.6 Release

**Tasks:**
1. CTF Solver Core Implementation (4-6h)
2. Hybrid Thermal Model Architecture (3-4h)
3. ASHRAE 140 Full Validation Suite (2-3h)
4. Documentation Updates (2-3h)
5. v0.6 Release Preparation (1-2h)
6. Human Verification Checkpoint (blocking)

**Status:** PLANNING COMPLETE — Ready for execution

---

## Dependencies

```
Phase 24 (6R2C/8R3C Diagnostic Audit)
    ↓
Phase 25 (Alternative Physics Implementation)
    ↓
Phase 26 (Comparative Evaluation)
    ↓
Phase 27 (v0.6 Release or Pivot Decision)
```

---

## Phase Order Rationale

The four-phase structure follows a diagnostic-driven approach:

1. **Phase 24 (Diagnosis):** Before implementing anything, understand WHY 6R2C/8R3C showed no improvement. Is there a bug? Wrong conductance calculations? Incorrect node placement? Or is the RC network structure itself fundamentally limited?

2. **Phase 25 (Alternative):** If Phase 24 finds RC networks fundamentally limited, implement alternative approaches (finite difference, CTF, adaptive timestep). If Phase 24 finds a bug, fix it and skip to Phase 26.

3. **Phase 26 (Evaluation):** Comprehensive testing against all 18 ASHRAE 140 cases. Ensure no regressions on low-mass while fixing high-mass.

4. **Phase 27 (Decision):** Based on results, either ship v0.6 with validated approach OR formally document that RC networks cannot work and recommend ML surrogate path for v1.0.

This ordering ensures we don't waste effort implementing alternatives if there's a simple bug in the current 6R2C implementation.

---

## Phase Granularity

**Granularity:** Fine (from config.json)

This milestone uses fine-grained phases to maintain clear decision points:
- **Phase 24:** Diagnostic only — no implementation
- **Phase 25:** Implementation of best alternative
- **Phase 26:** Comprehensive evaluation
- **Phase 27:** Release or pivot decision

Each phase delivers a coherent, verifiable capability that informs the next phase.

---

## Progress

| Phase | Name | Plans Complete | Status | Completed |
|-------|------|----------------|--------|-----------|
| 24. 6R2C/8R3C Diagnostic Audit | 10/10 | Planning Complete | Ready to execute |
| 25. Alternative Physics Implementation | 7/7 | Planning Complete | Ready to execute |
| 26. Comparative Evaluation | 0/5 | Planning Complete | Ready to execute |
| 27. v0.6 Release | 1/1 | Planning Complete | Ready to execute |

**Overall Progress:** 100% (4/4 phases planned, ready for execution)

---

*Roadmap created: 2026-03-17 for v0.6 Validation Excellence*
