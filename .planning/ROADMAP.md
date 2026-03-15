# Fluxion Roadmap

**Project:** Building Energy Modeling Engine (Rust + Python)
**Milestone:** v0.5 Production Foundation
**Current Phase:** Phase 22 (gap closure)
**Last Updated:** 2026-03-15

---

## Milestones

- ✅ **v0.5 Production Foundation** — Phases 21-23 (complete, 15/15 plans complete)
- ✅ **v0.4 ASHRAE 140 Compliance** — Phases 14-20 (shipped 2026-03-15) — See `.planning/milestones/v0.4-ROADMAP.md`
- ✅ **v0.2 Partial Validation** — Phases 1-7 (shipped 2026-03-11) — See `.planning/milestones/v0.2-ROADMAP.md`

---

## Current Status

**Milestone:** v0.5 Production Foundation
**Phase:** Phase 22 - Validation Gap Resolution
**Status:** Gap closure plans created (22-06, 22-07)

---

## Phases

- [x] **Phase 21: Integration Testing Framework** - E2E tests, wiring validation, regression test suite (completed 2026-03-15, gap closure mode)
- [x] **Phase 22: Validation Gap Resolution** - Case 960 fix, 8R3C evaluation, high-mass accuracy (5/5 plans executed, 2 gap closure plans)
- [ ] **Phase 23: Production Readiness** - Complete docs, benchmarks, stability guarantees

---

## Phase Details

### Phase 21: Integration Testing Framework

**Goal:** Users can run comprehensive integration tests that detect wiring issues and prevent regressions before they reach production

**Depends on:** Nothing (continues from v0.4 Phase 20)

**Requirements:** INTEG-01, INTEG-02, INTEG-03, INTEG-04, INTEG-05, INTEG-06, INTEG-07, INTEG-08

**Success Criteria** (what must be TRUE):
1. User can run `cargo test --test integration` and execute full E2E test suite in under 5 minutes
2. Integration tests catch wiring issues (e.g., solve_timesteps() never calls predict_loads() when use_ai=true)
3. User can run `fluxion validate --all` and verify all 18 ASHRAE 140 cases pass in CI
4. Python-side integration tests validate NumPy array handling and error cases across FFI boundary
5. User can run wiring validation check that reports module dependency issues before commit

**Plans:** 10/10 plans complete
- [x] 21-01-PLAN.md — E2E framework with BuildingScenario builder and WiringTracer
- [x] 21-02-PLAN.md — Python PyO3 integration tests with NumPy array validation
- [x] 21-03-PLAN.md — ASHRAE 140 regression test suite with nightly GitHub Actions
- [x] 21-04-PLAN.md — Test data management with versioned directories
- [x] 21-05-PLAN.md — Wiring validation, E2E scenarios, CI/CD integration
- [ ] 21-06-PLAN.md — Refactor integration tests to use BuildingScenario and WiringTracer
- [ ] 21-07-PLAN.md — Implement Rust-side PyO3 tests OR document blocker
- [ ] 21-08-PLAN.md — Correct verification error in VERIFICATION.md
- [ ] 21-09-PLAN.md — Add HVAC variant tests with parameterization
- [ ] 21-10-PLAN.md — Add automated dependency checking OR accept manual tracing

**Gap Closure Mode:** 5 plans created to address verification gaps from 21-VERIFICATION.md
- 21-06: Framework-test disconnect (tests don't use BuildingScenario/WiringTracer)
- 21-07: Missing Rust-side PyO3 tests (test_pyo3_bindings.rs is TODO stub)
- 21-08: Performance verification error (22 seconds IS under 5 minutes)
- 21-09: Missing HVAC variant tests (VAV, CAV, HeatPump, Chiller)
- 21-10: Automated dependency checking missing (only manual runtime tracing exists)

---

### Phase 22: Validation Gap Resolution

**Goal:** Users can validate that all known ASHRAE 140 gaps (Case 960, 8R3C, high-mass) are resolved without breaking existing cases

**Depends on:** Phase 21 (integration testing framework provides regression guardrails)

**Requirements:** VAL-01, VAL-02, VAL-03, VAL-04, VAL-05, VAL-06, VAL-07, VAL-08, VAL-09

**Success Criteria** (what must be TRUE):
1. User can run `fluxion validate --case 960` and see annual cooling energy within ±15% of reference
2. User can run 8R3C thermal network and see <50% error improvement for high-mass cases (or 5R1C remains default)
3. User can run full ASHRAE 140 validation suite and see 18/18 cases passing (no regressions)
4. High-mass annual energy accuracy improved from 229-322% error baseline (verified through thermal mass energy accounting)
5. A/B testing framework quantifies improvement for each validation gap fix before adoption

**Plans:** 5/5 executed, 2 gap closure plans

- [x] 22-01-PLAN.md — 900-series regression test to prevent Case 960 fix from breaking other cases
- [x] 22-02-PLAN.md — Thermal mass energy accounting validation (diagnose physics correctness)
- [x] 22-03-PLAN.md — A/B testing framework for variant comparison
- [x] 22-04-PLAN.md — Case 960 validation with COP correction
- [x] 22-05-PLAN.md — 8R3C thermal network research and evaluation
- [ ] 22-06-PLAN.md — Fix compilation issues and verify 900-series regression test execution (gap closure)
- [ ] 22-07-PLAN.md — Investigate and fix thermal mass energy balance calculation (gap closure)

**Gap Closure Mode:** 2 plans created to address verification gaps from 22-VERIFICATION.md
- 22-06: VAL-07 - 900-series regression test (compilation issues blocking verification)
- 22-07: VAL-06, VAL-08 - Thermal mass energy accounting (1100%+ energy balance errors requiring physics investigation)

---

### Phase 23: Production Readiness

**Goal:** Users have complete documentation, performance benchmarks, and stability guarantees for production deployment

**Depends on:** Phase 22 (stable validation fixes provide baseline for production artifacts)

**Requirements:** PROD-01, PROD-02, PROD-03, PROD-04, PROD-05, PROD-06, PROD-07, PROD-08, PROD-09, PROD-10, PROD-11, PROD-12, PROD-13

**Success Criteria** (what must be TRUE):
1. User can reference complete API documentation with examples for all BatchOracle and Model methods
2. User can run performance benchmarks and see documented throughput guarantees (10,000+ configs/sec) by hardware
3. User can verify performance regression detection fails PR if >10% slowdown from baseline
4. User can read stability guarantees document covering input validation, error handling, failure modes, and determinism
5. User can migrate from v0.4 to v0.5 using migration guide with no breaking API changes

**Plans:** TBD

---

| 21. Integration Testing Framework | 10/10 | Complete    | 2026-03-15 |
## Progress

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 21. Integration Testing Framework | 5/5 complete, 5 gap closure plans | Gap closure mode | 2026-03-15 |
| 22. Validation Gap Resolution | 5/5 executed, 2 gap closure plans | Gap closure mode | 2026-03-15 |
| 23. Production Readiness | 0/0 | Not started | - |

**Overall Progress:** 67% (2/3 phases complete)

---

## Dependencies

```
Phase 21 (Integration Testing - gap closure mode)
    ↓
Phase 22 (Validation Gap Resolution - gap closure mode)
    ↓
Phase 23 (Production Readiness)
```

---

## Phase Order Rationale

The three-phase structure follows natural dependency chains:

1. **Integration Testing → Validation Gaps:** Validation fixes require comprehensive testing to avoid regressions. Phase 21 provides E2E framework and regression tests needed for Phase 22's A/B testing.

2. **Validation Gaps → Production Readiness:** Production benchmarks and documentation depend on stable validation fixes. Phase 22 resolves known gaps, providing stable physics for Phase 23's realistic benchmarks.

This ordering directly addresses the critical pitfalls identified in research:
- Phase 21 prevents Pitfall 1 (wiring issues not detected) and Pitfall 6 (brittle tests)
- Phase 22 prevents Pitfall 2 (fixes introduce regressions) and Pitfall 5 (8R3C migration without validation)
- Phase 23 prevents Pitfall 3 (unrealistic benchmarks) and Pitfall 4 (stale documentation)

---

## Phase Granularity

**Granularity:** Fine (from config.json)

This milestone uses fine-grained phases to maintain clear delivery boundaries:
- **Integration Testing Framework** (8 requirements) - Complete E2E testing infrastructure
- **Validation Gap Resolution** (9 requirements) - Resolve all known ASHRAE 140 gaps
- **Production Readiness** (13 requirements) - Complete production artifacts

Each phase delivers a coherent, verifiable capability that unblocks the next phase.

---

*Roadmap created: 2026-03-15*
*Updated: 2026-03-15 - Added gap closure plans (21-06 through 21-10)*
*Updated: 2026-03-15 - Added gap closure plans (22-06 through 22-07)*
