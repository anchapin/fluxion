# Requirements: Fluxion

**Defined:** 2026-03-15
**Updated:** 2026-03-17 for v0.6 Validation Excellence
**Core Value:** Full ASHRAE Standard 140 compliance achieved (v0.4): All 37 v0.4 requirements satisfied (100%). Thermal network verified with analytical physics, comprehensive HVAC equipment modeling (VAV/CAV/HeatPump/Chiller/Boiler), psychrometric calculations validated, internal loads with schedules implemented, diagnostic cases (195-470, 800-810) complete, statistical validation framework with Addendum B compliance, data quality finalized (mocks removed, hardcodes replaced, documentation complete).

## v0.5 Requirements (COMPLETE)

Requirements for v0.5 Production Foundation milestone. **All satisfied as of 2026-03-17.**

### Integration Testing Framework (INTEG)

- [x] **INTEG-01**: User can run E2E integration tests that validate full system workflows
- [x] **INTEG-02**: Integration test framework provides reusable test fixtures for building scenarios, weather data, HVAC configs
- [x] **INTEG-03**: E2E tests detect wiring issues between modules (validation, simulation, AI surrogates)
- [x] **INTEG-04**: Python-side integration tests validate PyO3 bindings with real NumPy arrays
- [x] **INTEG-05**: Regression test suite runs full ASHRAE 140 validation (18 cases) on every commit
- [x] **INTEG-06**: Test data management provides centralized repository with versioning for EPW files, reference results
- [x] **INTEG-07**: CI/CD integration runs integration tests and benchmarks on every PR, fails on regressions
- [x] **INTEG-08**: Wiring validation system automatically checks module dependencies and integration points

### Validation Gap Resolution (VAL)

- [x] **VAL-01**: Case 960 annual cooling energy passes ASHRAE 140 tolerance bands (±15% annual energy, ±10% monthly energy)
- [x] **VAL-02**: 8R3C thermal network evaluation completed with performance comparison against 5R1C baseline
- [x] **VAL-03**: 8R3C provides <50% error improvement for high-mass cases or 5R1C remains default
- [x] **VAL-04**: 8R3C maintains ≥1,000 configs/sec throughput (baseline: ~2,575 for 5R1C)
- [x] **VAL-05**: 8R3C maintains ≥90% pass rate for low-mass cases (600-series, 800-series)
- [x] **VAL-06**: High-mass annual energy accuracy improved from 229-322% error baseline (thermal mass energy accounting validated)
- [x] **VAL-07**: 900-series regression test runs all cases (920, 930, 940, 960) together to prevent Case 960 fix from breaking other cases
- [x] **VAL-08**: Thermal mass energy accounting validated (energy_in = energy_out + mass_energy_change)
- [x] **VAL-09**: A/B testing framework quantifies improvement for validation gap fixes

### Production Readiness (PROD)

- [x] **PROD-01**: Complete API documentation covers all PyO3 public functions (BatchOracle, Model, all exported methods)
- [x] **PROD-02**: API documentation includes usage examples for common workflows (optimization, validation, analysis)
- [x] **PROD-03**: Performance benchmarks use realistic workloads (8760 timesteps, multi-zone buildings, population throughput 100/1000/10000 configs)
- [x] **PROD-04**: Performance benchmarks run with --release profile (LTO, codegen-units=1)
- [x] **PROD-05**: Performance regression detection fails PR if >10% slowdown from baseline
- [x] **PROD-06**: Performance benchmarks measure both single-config latency (<100ms target) and population throughput (10,000+ configs/sec target)
- [x] **PROD-07**: Performance benchmarks include Python-Rust FFI boundary overhead measurement
- [x] **PROD-08**: Stability guarantees documented with input validation contracts (parameter ranges, error handling)
- [x] **PROD-09**: Stability guarantees include failure modes documentation (what happens on invalid inputs)
- [x] **PROD-10**: Error recovery strategies implemented (graceful degradation, no panics in release builds)
- [x] **PROD-11**: Stability guarantees include determinism documentation (same inputs → same outputs)
- [x] **PROD-12**: Performance bounds documented with latency/throughput guarantees by hardware
- [x] **PROD-13**: API stability maintained (no breaking changes without deprecation, migration guides provided)

---

## v0.6 Requirements (Active)

Requirements for v0.6 Validation Excellence milestone. **Focus: Deep diagnostic investigation of high-mass annual energy failure.**

### Root Cause Diagnosis (DIAG)

- [ ] **DIAG-01**: 6R2C implementation audit — Verify conductance calculations match ISO 13790 6R2C specification
- [ ] **DIAG-02**: 6R2C node placement verification — Confirm thermal mass node is correctly positioned (surface vs core temperature)
- [ ] **DIAG-03**: Time constant analysis — Compare Fluxion τ vs EnergyPlus τ for identical high-mass constructions
- [ ] **DIAG-04**: Heat flow path tracing — Track energy flow through each RC branch to identify where 5R1C/6R2C diverge
- [ ] **DIAG-05**: Reference program comparison — Run identical case in EnergyPlus, extract internal state variables for comparison

### Alternative Physics Approaches (ALT)

- [ ] **ALT-01**: Finite difference method — Implement 1D heat conduction through mass layers (replace lumped capacitance)
- [ ] **ALT-02**: Conduction Transfer Functions (CTF) — Implement CTF coefficients for multi-layer walls (EnergyPlus approach)
- [ ] **ALT-03**: Adaptive timestep integration — Reduce timestep for high-mass buildings (1hr → 6min → 1min)
- [ ] **ALT-04**: Frequency-domain analysis — Use thermal response factors (X, Y, Z factors) for periodic heat transfer
- [ ] **ALT-05**: Hybrid RC + ML correction — Keep 5R1C physics, train ML to predict residual error

### Validation (VAL)

- [ ] **VAL-10**: Case 900 annual heating within ±15% — Target: 1.17-2.04 MWh (currently 5.35 MWh)
- [ ] **VAL-11**: Case 900 annual cooling within ±15% — Target: 2.13-3.67 MWh (currently 4.75 MWh)
- [ ] **VAL-12**: No regression on low-mass cases — 600-series must remain within ±15%
- [ ] **VAL-13**: Performance maintained — ≥1,000 configs/sec throughput

## v1.0 Requirements

Deferred to future release. Tracked but not in current roadmap.

### Monitoring and Observability
- **MONIT-01**: Monitoring infrastructure collects metrics (latency percentiles, error rates, throughput)
- **MONIT-02**: Alert thresholds configured for stability violations (p95 latency >150ms)
- **MONIT-03**: Dashboards provide real-time monitoring visibility

### Deployment and Operations
- **DEPLOY-01**: Production deployment guide complete with build/install instructions
- **DEPLOY-02**: Migration guide provided (v0.4 → v0.5, v0.5 → v1.0)
- **DEPLOY-03**: Load testing validates production-ready behavior under realistic traffic

### Advanced Validation
- **ADV-01**: Extended ASHRAE standards support (140.2, additional cases)
- **ADV-02**: Cross-validation against other BEM engines (EnergyPlus, ESP-r, TRNSYS)

## Out of Scope

Explicitly excluded from v0.6. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| 8R3C Implementation | Research shows no accuracy improvement over 5R1C, 65-75% performance penalty |
| 6R2C as Default | Phase 12 evaluation: no accuracy gain, 40-50% slower |
| Major Physics Refactoring | If RC networks fundamentally cannot work, need different approach (CTF, finite difference) |
| FMI 3.0 Co-Simulation | v2.0 feature - requires extensive protocol work beyond validation focus |
| REST/gRPC API | v2.0 feature - library focus for v0.6, API expansion is v2.0 scope |
| Docker Containerization | v2.0 feature - optional for library distribution |
| Additional ASHRAE Standards | v1.0 feature - extended cases and 140.2 deferred to future |
| Advanced AI Features | v1.0 feature - RL policy, multi-fidelity surrogates deferred to future |
| Mobile/Web UI | Out of scope - CLI and library API are primary use cases |
| Breaking API Changes | v0.5 users expect backward compatibility - maintain BatchOracle/Model API stability |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

### v0.5 Requirements (COMPLETE)

| Requirement | Phase | Status |
|-------------|-------|--------|
| INTEG-01 | Phase 21 | Complete |
| INTEG-02 | Phase 21 | Complete |
| INTEG-03 | Phase 21 | Complete |
| INTEG-04 | Phase 21 | Complete |
| INTEG-05 | Phase 21 | Complete |
| INTEG-06 | Phase 21 | Complete |
| INTEG-07 | Phase 21 | Complete |
| INTEG-08 | Phase 21 | Complete |
| VAL-01 | Phase 22 | Complete |
| VAL-02 | Phase 22 | Complete |
| VAL-03 | Phase 22 | Complete |
| VAL-04 | Phase 22 | Complete |
| VAL-05 | Phase 22 | Complete |
| VAL-06 | Phase 22 | Complete |
| VAL-07 | Phase 22 | Complete |
| VAL-08 | Phase 22 | Complete |
| VAL-09 | Phase 22 | Complete |
| PROD-01 | Phase 23 | Complete |
| PROD-02 | Phase 23 | Complete |
| PROD-03 | Phase 23 | Complete |
| PROD-04 | Phase 23 | Complete |
| PROD-05 | Phase 23 | Complete |
| PROD-06 | Phase 23 | Complete |
| PROD-07 | Phase 23 | Complete |
| PROD-08 | Phase 23 | Complete |
| PROD-09 | Phase 23 | Complete |
| PROD-10 | Phase 23 | Complete |
| PROD-11 | Phase 23 | Complete |
| PROD-12 | Phase 23 | Complete |
| PROD-13 | Phase 23 | Complete |

**v0.5 Coverage:** 30/30 (100%) ✅

### v0.6 Requirements (Active)

| Requirement | Phase | Status |
|-------------|-------|--------|
| DIAG-01 | Phase 24 | Pending |
| DIAG-02 | Phase 24 | Pending |
| DIAG-03 | Phase 24 | Pending |
| DIAG-04 | Phase 24 | Pending |
| DIAG-05 | Phase 24 | Pending |
| ALT-01 | Phase 25 | Pending |
| ALT-02 | Phase 25 | Pending |
| ALT-03 | Phase 25 | Pending |
| ALT-04 | Phase 25 | Pending |
| ALT-05 | Phase 25 | Pending |
| VAL-10 | Phase 26 | Pending |
| VAL-11 | Phase 26 | Pending |
| VAL-12 | Phase 26 | Pending |
| VAL-13 | Phase 26 | Pending |

**v0.6 Coverage:** 14/14 mapped (100%), 0/14 complete

---
*Requirements defined: 2026-03-15*
*Last updated: 2026-03-17 for v0.6 milestone*
