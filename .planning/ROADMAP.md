# Fluxion Roadmap

**Project:** Building Energy Modeling Engine (Rust + Python)
**Milestone:** v0.6 Validation Excellence (ACTIVE)
**Current Phase:** Phase 24 (Pending)
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
**Phase:** Phase 24 - 6R2C/8R3C Diagnostic Audit (PENDING)
**Status:** Milestone initialized, planning complete

---

## Phases

- [ ] **Phase 24: 6R2C/8R3C Diagnostic Audit** - Root cause analysis of why adding RC nodes shows no improvement
- [ ] **Phase 25: Alternative Physics Implementation** - Test finite difference, CTF, adaptive timestep approaches
- [ ] **Phase 26: Comparative Evaluation** - Run all 18 ASHRAE 140 cases, compare accuracy vs performance
- [ ] **Phase 27: v0.6 Release or Pivot Decision** - Ship validated approach OR formal proof + ML surrogate recommendation

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

**Plans:** TBD (to be created via /gsd:plan-phase 24)

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

### Phase 27: v0.6 Release or Pivot Decision

**Goal:** Ship v0.6 with validated approach OR formally document that RC networks cannot work and recommend ML surrogate path.

**Depends on:** Phase 26 (comprehensive evaluation results)

**Requirements:** TBD (based on outcome)

**Success Criteria** (what must be TRUE):

**If Successful:**
1. ✅ v0.6 shipped with new physics approach
2. ✅ All documentation updated
3. ✅ Migration guide (v0.5 → v0.6) provided
4. ✅ CI/CD updated with new validation thresholds

**If Pivot Required:**
1. ✅ Formal proof RC networks cannot achieve ±15% for high-mass
2. ✅ ML surrogate feasibility documented
3. ✅ Recommendation for v1.0/v2.0 roadmap

**Plans:** TBD (to be created via /gsd:plan-phase 27)

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

| 24. 6R2C/8R3C Diagnostic Audit | 0/6 | Pending     | -          |
## Progress

| Phase | Name | Plans Complete | Status | Completed |
|-------|------|----------------|--------|-----------|
| 24. 6R2C/8R3C Diagnostic Audit | 0/6 | Pending | - |
| 25. Alternative Physics Implementation | 0/4 | Pending | - |
| 26. Comparative Evaluation | 0/4 | Pending | - |
| 27. v0.6 Release or Pivot Decision | 0/2 | Pending | - |

**Overall Progress:** 0% (0/4 phases started)

---

*Roadmap created: 2026-03-17 for v0.6 Validation Excellence*
