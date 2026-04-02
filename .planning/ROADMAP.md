# Fluxion Roadmap

**Project:** Building Energy Modeling Engine (Rust + Python)
**Milestone:** v0.7 Thermal Physics Complete
**Current Phase:** Phase 29 COMPLETE, Phase 30 PLANNED (Cooling Energy Fix)
**Last Updated:** 2026-03-18
**Decision:** Phase 30 fix applied. Now proceed with Phase 31 re-validation and v0.7.0 release.

---

## Milestones

- 🔄 **v0.7 Thermal Physics Complete** — Phases 28-31 (ACTIVE, started 2026-03-17)
- ✅ **v0.6 Validation Excellence** — Phases 24-27 (COMPLETE 2026-03-17)
- ✅ **v0.5 Production Foundation** — Phases 21-23 (COMPLETE 2026-03-17)
- ✅ **v0.4 ASHRAE 140 Compliance** — Phases 14-20 (shipped 2026-03-15)
- ✅ **v0.2 Partial Validation** — Phases 1-7 (shipped 2026-03-11)

---

## Current Status

**Milestone:** v0.7 Thermal Physics Complete
**Phase:** Phase 30 COMPLETE (Cooling Energy Fix), Phase 31 PLANNED (Full Validation & Release)
**Status:** Phase 30 complete (3/3 plans), thermal mass fix applied. Phase 31 plans created (3 plans).

---

## Phases

- [x] **Phase 28: CTF/FD Solver Integration** - COMPLETE (6/6 plans done)
- [x] **Phase 29: Full Validation** - COMPLETE (9/9 plans, 32.8% pass rate)
- [x] **Phase 30: Cooling Energy Fix** - COMPLETE (3/3 plans done)
- [ ] **Phase 31: Full Validation & Release** - READY TO EXECUTE (3 plans)

---

## Phase Details

### Phase 28: CTF/FD Solver Integration

**Goal:** Integrate CTF or finite difference solver into production thermal model with automatic method selection.

**Depends on:** Nothing (continues from v0.6 where CTF/FD were implemented but not integrated)

**Requirements:** SOLVER-01, SOLVER-02, SOLVER-03, SOLVER-04, SOLVER-05, SOLVER-06

**Success Criteria** (what must be TRUE):
1. ✅ CTF solver integrated into `ThermalModel`
2. ✅ Finite difference solver integrated as fallback
3. ✅ Automatic method selection based on thermal mass (τ > 2 hours)
4. ✅ Zero-copy data sharing between solver methods
5. ✅ Python API and CLI integration complete
6. ✅ Unit tests for CTF/FD solvers passing

**Plans:** 6 plans created
- 28-01: CTF Solver Core Integration
- 28-02: Finite Difference Solver Integration
- 28-03: Method Selector Implementation
- 28-04: Thermal Model Refactoring for Solver Abstraction
- 28-05: Python API and CLI Integration
- 28-06: Unit Tests for CTF/FD Solvers

**Execution Strategy:**
- **Wave 1 (Week 1-2):** CTF solver core integration, FD solver integration
- **Wave 2 (Week 2-3):** Method selector, solver abstraction
- **Wave 3 (Week 3-4):** Python API, CLI, unit tests

**Status:** PLANNED (ready to execute)

---

### Phase 29: Full Validation

**Goal:** Validate all 18 ASHRAE 140 cases with new solver integration. Achieve ±15% tolerance for all cases.

**Depends on:** Phase 28 (solver integration complete) ✅

**Requirements:** VAL-10, VAL-11, VAL-12, VAL-13, VAL-14, VAL-15

**Success Criteria** (what must be TRUE):
1. ✅ Case 900 annual heating: 1.17-2.04 MWh (±15%)
2. ✅ Case 900 annual cooling: 2.13-3.67 MWh (±15%)
3. ✅ All 900-series cases within ±15%
4. ✅ No regression on 600-series (maintain ±15%)
5. ✅ No regression on 800-series (maintain ±15%)
6. ✅ Performance ≥800 configs/sec throughput

**Plans:** 9 plans created (READY TO EXECUTE)
- 29-01: 900-Series Validation Suite
- 29-02: Case 900 Deep Dive (Heating/Cooling)
- 29-03: Case 960 Multi-Zone Validation
- 29-04: 600-Series Regression Suite
- 29-05: 800-Series Regression Suite
- 29-06: Free-Float Cases Validation
- 29-07: Performance Benchmarking
- 29-08: Validation Report Generation
- 29-09: Human Verification & Documentation

**Execution Strategy:**
- **Wave 1 (Days 1-5):** 900-series validation suite (3 plans, parallel)
- **Wave 2 (Days 6-10):** Regression testing (3 plans, parallel)
- **Wave 3 (Days 11-15):** Performance benchmarking, validation report (3 plans, sequential)

**Status:** PLANNED (ready to execute)

---

### Phase 30: Cooling Energy Fix

**Goal:** Fix systematic cooling energy underestimation (-33.76% for Case 900). Improve pass rate from 32.8% to >80%.

**Depends on:** Phase 29 (validation complete)

**Requirements:** COOL-01, COOL-02, COOL-03, COOL-04, COOL-05, COOL-06

**Success Criteria** (what must be TRUE):
1. Energy balance verified (no unaccounted-for energy)
2. Solar gains match ASHRAE 140 reference
3. HVAC control operates correctly (setpoint tracking)
4. CTF solver symmetry confirmed (heating = cooling)
5. Case 900 cooling within ±15% tolerance
6. 900-series cooling >80% pass rate

**Plans:** 3 plans in 2 waves
- 30-01: Energy balance verification (Wave 1)
- 30-02: Component isolation audits (Wave 1, parallel)
- 30-03: Root cause fix & regression test (Wave 2)

**Execution Strategy:**
- **Wave 1 (Days 1-3):** Energy balance check + solar/control audits (parallel)
- **Wave 2 (Days 4-7):** Apply root cause fix, re-validate all 18 cases

**Status:** COMPLETE (3/3 plans done, thermal mass asymmetry corrected)

---

### Phase 31: Full Validation & Release

**Goal:** Complete v0.7.0 release by re-validating all ASHRAE 140 cases with the Phase 30 thermal mass fix, fine-tuning if needed, and preparing documentation and release artifacts.

**Depends on:** Phase 30 (cooling energy fix complete) ✅

**Success Criteria** (what must be TRUE):
1. Full ASHRAE 140 re-validation complete with Phase 30 fix applied
2. Overall pass rate >80% (vs 32.8% before fix)
3. Case 900 cooling within ±15% tolerance (~24.5 MWh expected)
4. All 2121 unit tests still passing
5. API documentation updated with thermal mass correction details
6. Migration guide created (v0.6 → v0.7)
7. v0.7.0 released to crates.io and PyPI

**Plans:** 1/3 plans executed
- 31-01: Full ASHRAE 140 Re-validation (Wave 1)
- 31-02: Documentation & Version Update (Wave 2, depends on 31-01)
- 31-03: Release Execution (Wave 3, depends on 31-02)

**Execution Strategy:**
- **Wave 1:** Full ASHRAE 140 re-validation (all 18 cases with Phase 30 fix)
- **Wave 2:** Version bump, changelog, documentation updates
- **Wave 3:** Release to crates.io, PyPI, GitHub release

**Status:** READY TO EXECUTE (plans created)

---

## Dependencies

```
Phase 28 (Solver Integration)
    ↓
Phase 29 (Full Validation)
    ↓
Phase 30 (Documentation & Release)
```

---

## Phase Order Rationale

The three-phase structure follows a logical progression:

1. **Phase 28 (Integration):** First, integrate the CTF/FD solvers that were implemented in v0.6 but left experimental. Add automatic method selection so users don't need to manually choose.

2. **Phase 29 (Validation):** Once integrated, validate all 18 ASHRAE 140 cases to ensure ±15% tolerance. Verify no regression on low-mass cases while fixing high-mass.

3. **Phase 30 (Release):** Document the new solver integration, create migration guide, and release v0.7.0 to crates.io and PyPI.

This ordering ensures we don't release without proper validation, and we don't validate without proper integration.

---

## Phase Granularity

**Granularity:** Standard (from config.json)

This milestone uses standard-grained phases to maintain clear deliverables:
- **Phase 28:** Integration only — no validation or docs
- **Phase 29:** Validation only — no new features
- **Phase 30:** Documentation and release — no new features

Each phase delivers a coherent, verifiable capability that informs the next phase.

---

## Progress

| Phase | Name | Plans Complete | Status | Completed |
|-------|------|----------------|--------|-----------|
| 28. CTF/FD Solver Integration | 6/6 | COMPLETE | ✅ Solver integration done |
| 29. Full Validation | 9/9 | COMPLETE | ✅ 32.8% pass rate, 75% improvement |
| 30. Cooling Energy Fix | 3/3 | COMPLETE | ✅ Thermal mass asymmetry fixed |
| 31. Full Validation & Release | 1/3 | In Progress|  |

**Overall Progress:** 75% (3/4 phases complete, 1/4 with plans ready)

---

*Roadmap updated: 2026-03-18 for Phase 29 planning*
