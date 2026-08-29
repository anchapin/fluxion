# ADR-0014: Bottom-Up Testing Architecture

- **Status:** Proposed
- **Date:** 2026-08-28
- **Deciders:** Fluxion maintainers
- **Supersedes:** None
- **Depends on:** ADR-0001 (No-Parameter-Tuning Rule), ADR-0003 (5R1C High-Mass Limitations)
- **Source:** Bottom-up testing PRD — `docs/implementation-plans/bottom-up-testing-prd.md`

---

## Executive Summary

This ADR captures the architectural decisions governing Fluxion's bottom-up unit and integration testing strategy for ASHRAE 140 compliance. It defines the module dependency order for gap-filling, the coverage metric target, the expected-value derivation strategy, and the relationship between unit and integration tests.

---

## Context

Fluxion's ASHRAE 140 validation fails at 14.3% (0/18 cases fully passing). The root cause is undiagnosed: components may be untested, may fail in isolation but not in integration, or may be tested against the wrong code paths. A systematic bottom-up testing approach is required to answer this question.

The existing test suite already contains bottom-up isolation tests for individual components (conduction vs EnergyPlus, HVAC vs analytical formulas, ventilation vs EnergyPlus, etc.). This ADR governs how new tests are added and how the existing tests are structured.

---

## Decision 1: Module Dependency Order for Gap-Filling

**Decision:** Audit and fill test gaps in the order: `weather → solar → physics/conduction → sim/ventilation → physics/hvac → sim/thermal_model → validation → cli`.

**Rationale:** Each module's outputs are inputs to the next in the diagnostic chain:

```
weather → solar → conduction → ventilation → zone thermal model → HVAC → validator
```

Testing upstream first establishes a known-good signal source before checking downstream integration. If we test `solar/` before `weather/`, we cannot verify that the solar outputs are correct for the weather data actually used in production.

If a downstream module's test fails, the upstream module is re-audited first to determine whether the upstream output is correct.

**Alternatives considered:**
- Random order — would require re-testing downstream modules when upstream tests are added; inefficient
- Downstream-first — would catch integration failures but not identify which upstream component is responsible; produces longer debugging cycles

---

## Decision 2: Line Coverage as the Primary Metric

**Decision:** Use **line coverage** (not function or branch coverage) as the 80–90% target.

**Rationale:**
- Function coverage is too coarse — a function with one passing test counts as 100% even if only 20% of its lines are exercised
- Branch coverage requires LLVM ProfData and is harder to interpret for physics code where certain error-handling branches are intentionally rarely executed
- Line coverage is the standard output of `cargo-llvm-cov`, is easy to bucket per module, and maps directly to "untested code" in IDEs

**Per-module targets:**
- Overall: ≥ 80% line coverage
- Per module: ≥ 70% line coverage (to allow some modules to be more thoroughly tested than others)

**Alternatives considered:**
- Branch coverage — already used by `fluxion-core` in `release_gates.yaml`; too granular for this PRD's scope
- MC/DC — required for DO-178C avionics but excessive for building physics; toolchain complexity unjustified

---

## Decision 3: Analytical Expected Values, Not Just EnergyPlus Reference

**Decision:** For each untested function, compute expected values using **(a) fundamental physics equations executed via Python in the test** and **(b) EnergyPlus reference CSV** where available.

**Rationale:**
- EnergyPlus reference data may itself contain errors (see KNOWN_ISSUES.md §REF-01 — monthly reference data is "documented-shape" not direct E+ output)
- Physics equations provide a first-principles check independent of both Fluxion's implementation AND E+'s implementation
- Using both provides cross-validation: if both agree, the expected value is trustworthy

**Rule (per RULES.md §Must-Always-0):** All numerical reasoning must be done by writing and executing Python code via `ctx_execute`, never by mental arithmetic.

**Implication for tests:** A test that only checks Fluxion vs E+ without an independent analytical calculation is insufficient for untested components. The analytical calculation is the primary expected value; E+ is a secondary cross-check.

**Alternatives considered:**
- E+ only — risk of propagating E+ bugs into Fluxion tests; no independent verification
- Analytical only — not always feasible for complex multi-variable functions (e.g., multi-node thermal networks)

---

## Decision 4: Dead Code Removal Precedes Coverage Baseline

**Decision:** Remove dead code **before** measuring coverage and setting the CI baseline.

**Rationale:**
- Dead code inflates the denominator (lines to cover) without any benefit
- Removing dead code before baseline means the baseline represents only the live code that needs to be tested
- If dead code is removed after baseline, coverage % would artificially jump, confusing trend tracking

**Stub exception:** Code that is stubbed out for future work (marked `TODO`, `unimplemented`, feature-gated placeholder) is **not** dead code and must not be removed. Removing such code would require re-adding it later, which violates the "don't remove something we'll need to add back" principle.

**Alternatives considered:**
- Measure first, remove later — dead code pollutes the baseline and the first coverage report; adds noise to trend data
- Remove dead code only if it doesn't affect test coverage — too complex a rule; easier to just remove it

---

## Decision 5: One Integration Test Per Diagnostic-Chain Wiring Edge

**Decision:** For each wiring edge in the diagnostic chain (Weather→Solar, Solar→Conduction, Conduction→Ventilation, Ventilation→Zone, HVAC→Zone, Zone→Validator), there must be **at least one integration test** that exercises that edge using the actual code path invoked by the ASHRAE 140 validator.

**Rationale:**
- Unit test coverage of 80% does not guarantee that components are wired correctly
- The ASHRAE validator uses a specific code path (`from_spec` → `step_physics`) that may differ from the code path exercised by a unit test that calls a function directly
- One integration test per wiring edge is the minimum to catch wiring regressions

**This does not replace per-component unit tests.** Unit tests remain the primary coverage mechanism; integration tests are a safety net for wiring correctness.

**Wire edges requiring integration tests:**

| Edge | Integration Test |
|------|-----------------|
| Weather → Solar | `tests/weather_solar_integration.rs` |
| Solar → Conduction | `tests/solar_conduction_wiring.rs` |
| Conduction → Zone thermal model | `tests/conduction_zone_integration.rs` |
| Ventilation → Zone thermal model | `tests/ventilation_zone_integration.rs` |
| HVAC → Zone thermal model | `tests/hvac_zone_integration.rs` |
| Zone thermal model → ASHRAE validator | `tests/ashrae_140_case_<NNN>.rs` |

---

## Decision 6: Reference Data Is Never Tuned to Pass a Test

**Decision:** Reference data (EnergyPlus CSVs, analytical expected values) is **never adjusted** to make a failing test pass. If a test fails against reference data, the reference data or the test setup is investigated, not the reference values.

**Rationale:** Per ADR-0001 (No-Parameter-Tuning Rule), Fluxion must not tune outputs to pass tests. This applies equally to reference data. If the reference data is wrong, the generator is fixed or the physics is corrected — never the expected values.

**Exception:** If a test's expected values were computed against an **incorrect** implementation (i.e., the expected value itself was produced by buggy code), the expected value must be corrected using independently-verified physics equations or a corrected EnergyPlus run.

**Distinction:**
- Wrong expected value because E+ has a bug → fix expected value using physics equations
- Wrong expected value because Fluxion is wrong → fix Fluxion, keep expected value

---

## Decision 7: Integration Test Tolerance Equals Unit Test Tolerance

**Decision:** Integration tests use the **same tolerance** as unit tests: ±0.5 °C for temperature, ±1% relative or ±1 W/m² absolute for heat flux/power, ±2% relative for energy.

**Rationale:** Tolerance should reflect the intrinsic accuracy of the physics model, not whether the test is unit or integration. Using tighter tolerances for integration tests would implicitly allow larger errors in component outputs, which is the opposite of what a bottom-up testing strategy aims to catch.

---

## Consequences

### Positive
- Clear dependency order prevents re-testing downstream modules when upstream tests are added
- Line coverage as target is straightforward to measure and interpret
- Analytical + E+ expected values provide cross-validation
- One integration test per wiring edge ensures wiring correctness is not overlooked

### Negative
- Running modules in dependency order means later modules (validation, cli) wait for earlier modules to complete — longer overall timeline for Phase 2
- Analytical expected values require more work per test than simply copying E+ output — slower test writing
- 70% per-module coverage target may allow some modules to remain less well-tested than others

### Neutral
- Dead code removal as a prerequisite adds one phase before coverage measurement — slightly longer Phase 0
- Reference data provenance headers add overhead to each new CSV — more documentation work per reference file

---

## Review History

| Date | Reviewer | Notes |
|------|----------|-------|
| 2026-08-28 | Initial draft | Proposed; pending maintainer review |
