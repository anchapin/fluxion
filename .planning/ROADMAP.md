# Fluxion Roadmap

**Project:** Building Energy Modeling Engine (Rust + Python)
**Milestone:** v0.7 Thermal Physics Complete
**Current Phase:** Phase 28 COMPLETE, Phase 29 PLANNED
**Last Updated:** 2026-03-18

---

## Milestones

- 🔄 **v0.7 Thermal Physics Complete** — Phases 28-30 (ACTIVE, started 2026-03-17)
- ✅ **v0.6 Validation Excellence** — Phases 24-27 (COMPLETE 2026-03-17)
- ✅ **v0.5 Production Foundation** — Phases 21-23 (COMPLETE 2026-03-17)
- ✅ **v0.4 ASHRAE 140 Compliance** — Phases 14-20 (shipped 2026-03-15)
- ✅ **v0.2 Partial Validation** — Phases 1-7 (shipped 2026-03-11)

---

## Current Status

**Milestone:** v0.7 Thermal Physics Complete
**Phase:** Phase 28 COMPLETE (Solver Integration), Phase 29 PLANNED (Full Validation)
**Status:** Phase 28 complete (6/6 plans), Phase 29 planned (9 plans ready) — Ready to execute Phase 29

---

## Phases

- [x] **Phase 28: CTF/FD Solver Integration** - COMPLETE (6/6 plans done)
- [ ] **Phase 29: Full Validation** - PLANNED (2-3 weeks, 9 plans ready)
- [ ] **Phase 30: Documentation & Release** - PLANNED (1-2 weeks)

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

### Phase 30: Documentation & Release

**Goal:** Complete documentation and release v0.7.0.

**Depends on:** Phase 29 (validation complete)

**Requirements:** PROD-14, PROD-15, PROD-16, PROD-17, PROD-18, TEST-01 through TEST-06

**Success Criteria** (what must be TRUE):
1. ✅ CTF/FD solver documentation complete
2. ✅ Migration guide (v0.6 → v0.7) complete
3. ✅ Performance optimization guide complete
4. ✅ API documentation updated
5. ✅ All tests passing (unit, integration, regression)
6. ✅ v0.7.0 released to crates.io and PyPI

**Plans:** TBD (to be created via `/gsd:plan-phase 30`)

**Execution Strategy:**
- **Wave 1 (Week 1-1):** Solver documentation, migration guide
- **Wave 2 (Week 1-2):** API docs, performance guide
- **Wave 3 (Week 2):** Test suite, release preparation, human verification

**Status:** PLANNED

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
| 29. Full Validation | 0/9 | PLANNED | - |
| 30. Documentation & Release | 0/7 | PLANNED | - |

**Overall Progress:** 33% (1/3 phases complete, 0/3 planned)

---

*Roadmap updated: 2026-03-18 for Phase 29 planning*
