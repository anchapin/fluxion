# Bottom-Up Unit & Integration Testing Plan for ASHRAE 140 Compliance

**Status**: DRAFT
**Date**: 2026-08-28
**Target**: 80–90% line coverage on ASHRAE-relevant modules; achieve ASHRAE 140 pass rate improvement over 14.3% baseline
**Scope**: Modules directly involved in ASHRAE 140 test execution (`weather/`, `solar/`, `physics/`, `sim/`, `thermal/`, `validation/`, `cli/`)
**Out of scope**: `ai/`, `api/`, `orchestration/`, `twin/`, `measures/`, `quantum/` unless dead code removal surfaces a dependency

---

## 1. Problem Statement

Fluxion's ASHRAE 140 validation pass rate is **14.3%** (12/84 metrics). The root cause is undiagnosed:

- **(a)** Some sub-components are untested and produce incorrect outputs
- **(b)** Components are tested in isolation but fail when wired together
- **(c)** Integration paths do not exercise the same code paths used by the ASHRAE 140 validator

This plan defines a systematic, exhaustive bottom-up testing approach to answer that question and achieve 80–90% sub-component line coverage.

---

## 2. Current Baseline

| Metric | Value | Source |
|--------|-------|--------|
| ASHRAE 140 pass rate | 14.3% (12/84 metrics) | `SCORECARD.md` |
| Cases fully passing | 0/18 | `SCORECARD.md` |
| Mean Absolute Error | 51.03% | `SCORECARD.md` |
| Max single-case deviation | 470.11% | `SCORECARD.md` |

Existing bottom-up tests cover: conduction vs EnergyPlus, HVAC vs analytical formulas, ventilation vs EnergyPlus, solar position vs EnergyPlus, weather vs EnergyPlus, zone balance trait isolation.

Yet ASHRAE 140 validation still fails at 14.3%, suggesting either wiring issues, untested edge cases, or reference data gaps.

---

## 3. Scope

### 3.1 In-Scope Modules

| Module | Rationale |
|--------|-----------|
| `src/weather/` | Weather data (TMY3) feeds solar and thermal models |
| `src/solar/` | Solar position and surface irradiance |
| `src/physics/` | Conduction solvers (5R1C, CTF, FD), thermal mass, HVAC equipment |
| `src/sim/` | Thermal model (5R1C/9R4C network), ventilation, solar gain distribution, shading, occupancy, lighting, interzone, equipment |
| `src/thermal/` | Thermal integration |
| `src/validation/` | ASHRAE 140 validator, ashrae_140_cases |
| `src/cli/` | CLI commands that invoke the ASHRAE validation chain |

### 3.2 Module Dependency Order

Modules are audited and filled in dependency order (upstream first):

```
weather → solar → physics/conduction → sim/ventilation → physics/hvac → sim/thermal_model → validation → cli
```

Testing upstream first establishes a known-good signal source before checking downstream integration.

---

## 4. Success Criteria

| # | Criterion | Target |
|---|-----------|--------|
| 1 | Dead code removed | 0 dead code warnings on `cargo clippy --all-targets` |
| 2 | Line coverage (in-scope modules) | ≥ 80% overall, ≥ 70% per module |
| 3 | Bottom-up isolation tests passing | 100% of existing + new tests pass |
| 4 | Integration tests per wiring edge | ≥ 1 passing test per diagnostic-chain edge |
| 5 | ASHRAE 140 pass rate | Document improvement over 14.3% baseline |

---

## 5. Workback Schedule

### Phase 0 — Dead Code Removal (est. 1–2 days)

**Step 0.1:** Run `cargo clippy --all-targets -- -D dead_code`

**Step 0.2:** Audit each flagged item:
- Truly dead (no callers in tests, no external usage) → remove
- Stub for future work (marked `TODO`, `unimplemented`, feature-gated) → leave
- Used only in `#[cfg(test)]` blocks → keep

**Step 0.3:** Commit dead code removal: `chore(dead-code): remove unused ...`

**Step 0.4:** Verify 0 new warnings on subsequent `cargo clippy --all-targets`

**Gate:** No new dead code warnings on subsequent runs.

---

### Phase 1 — Coverage Baseline Setup (est. 1–2 days)

**Step 1.1:** Install `cargo-llvm-cov` if not present (`cargo install cargo-llvm-cov`)

**Step 1.2:** Configure `coverage.toml`:
- 80% overall line coverage target
- Per-module bucket targets (≥ 70% per module)
- Ignore patterns for generated fixture code, `#[ignore]` tests, test data files

**Step 1.3:** Run `cargo llvm-cov --lcov --output-path coverage_baseline.json`

**Step 1.4:** Commit baseline JSON; enable CI gate in `release_gates.yaml` (Issue #1932)

**Gate:** Baseline JSON committed; CI fails if coverage drops below threshold on new code.

---

### Phase 2 — Audit & Gap Filling (est. 2–4 weeks, iterative per module)

Each module goes through three phases before moving to the next:

#### Phase 2A — Audit (per module)

1. List all public and internal functions using `cargo doc --document-private-items`
2. Cross-reference with existing tests in `tests/`
3. Run existing bottom-up tests; record pass/fail per test file
4. Flag untested functions (no test coverage at all)
5. Flag failing tests; classify failure as:
   - **Real bug** → file issue, fix bug
   - **Incorrect reference data** → fix test
   - **Wrong code path** (test exercises different code than ASHRAE validator) → write new integration test

#### Phase 2B — Unit Test Gap Filling

For each untested function:
- Call function directly with known inputs
- Compute expected output analytically (from physics equations) or from EnergyPlus reference CSV
- Assert `actual ≈ expected` within tolerance
- Use `proptest` for property-based testing where inputs are numeric
- Store E+ reference CSVs in `tests/reference_data/` (per existing convention)

**Tolerance standards:**
- Temperature: ±0.5 °C
- Heat flux / power: ±1% relative or ±1 W/m² absolute
- Energy (kWh): ±2% relative
- Dimensionless ratios (U-value, SHGC): ±1% relative

#### Phase 2C — Integration Test Gap Filling

For each wiring edge in the diagnostic chain:
- Wire two or more components together (e.g., `Weather → SolarGainDistribution → ConductionSolver`)
- Use a known ASHRAE case input (Case 600 spec) as the test fixture
- Compute expected output from EnergyPlus simulation or analytical chain
- Assert wired output matches expected within tolerance

**Key wiring edges to test:**
| Edge | Components |
|------|------------|
| Weather → Solar | Weather TMY3 → solar position/irradiance |
| Solar → Conduction | Surface irradiance → per-surface heat flux |
| Conduction → Zone thermal model | Per-surface flux → 5R1C/9R4C node network |
| Ventilation → Zone thermal model | ACH / infiltration → air-side heat transfer |
| HVAC equipment → Zone thermal model | Heating/cooling load → zone demand |
| Zone thermal model → ASHRAE validator | Annual energy, peak loads → pass/fail |

**Gate per module:** Module reaches ≥ 80% line coverage AND all bottom-up paths to ASHRAE cases have ≥ 1 passing integration test before moving to the next module.

---

### Phase 3 — Final Validation (est. 1 day)

**Step 3.1:** `cargo test --workspace` — all tests pass

**Step 3.2:** `cargo llvm-cov` — overall ≥ 80%

**Step 3.3:** ASHRAE 140 validation run — document pass rate vs 14.3% baseline

**Step 3.4:** `python scripts/generate_scorecard.py` — regenerate SCORECARD.md

---

## 6. Test Naming Conventions

| Type | Pattern | Example |
|------|---------|---------|
| Unit (component) | `tests/physics/<component>_isolation.rs` | `tests/physics/five_r1c_solver_isolation.rs` |
| Unit (analytical) | `tests/<module>/<component>_analytical.rs` | `tests/solar/solar_position_analytical.rs` |
| Integration (wiring) | `tests/<module>_<module>_wiring.rs` | `tests/weather_solar_integration.rs` |
| E+ reference | `tests/<module>_<component>_vs_energyplus.rs` | `tests/conduction_ctf_step_response_vs_energyplus.rs` |
| ASHRAE case | `tests/ashrae_140_case_<NNN>.rs` | `tests/ashrae_140_case_900.rs` |

---

## 7. Reference Data Management

- EnergyPlus reference CSVs live in `tests/reference_data/` (existing convention)
- Each new reference CSV must include a provenance header:
  - EnergyPlus version used
  - IDF file used
  - Date generated
  - Method (hourly output, which output variable)
- Python generators in `tests/reference_data/` must have unit tests in `scripts/ci/`
- **Never tune reference data to pass a test** — if the reference data is wrong, fix the generator, not the test

---

## 8. Open Questions

| # | Question | Decision Needed |
|---|---------|----------------|
| 1 | Tolerance per physics domain | 1% relative for energy/power; ±0.5 °C for temperature — accept or adjust? |
| 2 | Property-based testing | Use `proptest` for all numeric inputs or only critical functions? |
| 3 | EnergyPlus version | Use same version as existing reference data for consistency? |
| 4 | CI timeout strategy | Split test suite into parallel CI jobs if runtime exceeds threshold? |

---

## 9. Risks & Mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Reference CSV data is incorrect | Medium | Independently compute expected values from physics equations; compare E+ output against analytical before using as golden |
| Gap between "test passes in isolation" and "component wired correctly" | High | Write integration tests per wiring edge that exercise the actual `from_spec` → `step_physics` path used by ASHRAE validator |
| Coverage tool requires complex ignore patterns | Medium | Use bucket-per-module approach; ignore `tests/` subdirs except known test files |
| E+ not available in CI environment | Low | E+ reference data is pre-generated and committed; tests run offline |

---

## 10. Deliverables Summary

| # | Deliverable | Type |
|---|-------------|------|
| 1 | Dead code removed from in-scope modules | Commit |
| 2 | Coverage baseline JSON + CI gate configured | Commit |
| 3 | Per-module test audit document | `docs/bottom_up_testing_audit.md` |
| 4 | New unit tests for untested sub-components | Test files in `tests/` |
| 5 | New integration tests for wiring paths | Test files in `tests/` |
| 6 | Updated SCORECARD with new pass rate | Commit |

---

## 11. Related Documents

- `RULES.md` — no parameter tuning rule, numerical reasoning via code
- `AGENTS.md` — diagnostic chain: Weather → Solar → Conduction → Ventilation → Zone Balance
- `docs/adr/0014-bottom-up-testing-architecture.md` — architectural decisions (this plan's ADRs)
- `docs/KNOWN_ISSUES.md` — LIMIT-05, LIMIT-16, LIMIT-17, LIMIT-18, LIMIT-20 (structural failures requiring GaugeSolver)
- `SCORECARD.md` — current 14.3% baseline pass rate
