# Bottom-Up Testing PRD — Handoff Prompt

**Created**: 2026-08-28
**Status**: IN_PROGRESS
**Author**: Claude Code session

---

## Current Repository State

### What's Done

1. **Coverage baseline exists** at `validation/coverage_baseline.json` (dated 2026-08-10):
   - Overall: **79.8% line**, **63.8% branch**
   - `weather_solar`: 97.1% line / **61.1% branch** (below 63% floor)
   - `weather_ventilation`: 92.7% line / 88.5% branch
   - `conduction_zone`: 87.7% line / 64.7% branch
   - `hvac`: 95.1% line / 74.2% branch
   - `sim`: 81.6% line / 62.7% branch
   - `validation`: 89.9% line / 67.6% branch

2. **No `coverage.toml`** exists in project root (the PRD references this but it was never created)

3. **No dead code** errors on `cargo clippy --all-targets -- -D dead_code` — only unused import warnings exist

4. **ASHRAE 140 validation passes**: 3/3 tests pass with `FLUXION_EPW_DIR` set

5. **Existing test suites** (all passing with proper env):
   - `cargo test -p fluxion --lib`: 4066 passed
   - `cargo test --test zone_balance_eplus_isolation`: 19 passed
   - `cargo test --test ashrae_140_validation`: 3 passed
   - `cargo test --test ashrae_140_blind_validation`: 17 passed, 7 ignored
   - `cargo test --test ashrae_140_solid_conduction_variants`: 3 passed, 2 ignored
   - `cargo test --test invariant_checker_test`: 9 passed, 1 ignored

### Critical Prerequisites Before Starting

**Set FLUXION_EPW_DIR for all test runs:**
```bash
export FLUXION_EPW_DIR=/home/alex/Projects/fluxion/assets/weather
```

Without this, ~40 tests fail with "EPW file not found".

---

## What the PRD Prescribes vs. Reality

| PRD Step | Reality |
|----------|---------|
| Phase 0: Dead code removal | **Not needed** — 0 dead_code errors |
| Phase 1: Coverage baseline + `coverage.toml` | **Baseline exists** but `coverage.toml` was never created |
| Phase 2: Module audit + gap filling | **Not started** — PRD is DRAFT |
| Phase 3: Final validation | **Not started** |

---

## Recommended Next Steps (Priority Order)

### 1. Create `coverage.toml` (Phase 1 completion)

The coverage baseline JSON exists but there's no `coverage.toml` to configure targets. Create one:
```toml
# coverage.toml
[defaults]
output = { "terminal" = ["text"], "html" = ["html"] }

[[targets]]
name = "in-scope"
modules = ["weather", "solar", "physics", "sim", "thermal", "validation", "cli"]
line-coverage-threshold = 80.0
branch-coverage-threshold = 70.0
```

### 2. Run `cargo llvm-cov` to get current coverage

```bash
cargo install cargo-llvm-cov  # if not present
cargo llvm-cov --lcov --output-path coverage_current.json
```

Compare against baseline at `validation/coverage_baseline.json`.

### 3. Phase 2A — Module Audit

Start with `weather` module (upstream first per PRD dependency order):
```bash
cargo doc --document-private-items -p fluxion --no-deps 2>/dev/null
# List untested functions: cross-reference src/weather/ with tests/
```

Key audit questions per PRD:
- Are there functions with **zero test coverage**?
- Are there **failing tests** that indicate real bugs vs. reference data issues?
- Are there **code paths** exercised by ASHRAE validator but not by unit tests?

### 4. Phase 2B/2C — Gap Filling

**High-value targets** (based on current ASHRAE 140 failure patterns):
- `weather` → `solar` wiring edge (Weather TMY3 → solar irradiance)
- `solar` → `conduction` wiring edge (irradiance → per-surface flux)
- The `step_physics_5r1c` path in `gauge_zone_solver.rs` (recently fixed but needs regression coverage)

### 5. Phase 3 — Final Validation

```bash
FLUXION_EPW_DIR=$(pwd)/assets/weather cargo test --workspace
cargo llvm-cov  # verify 80% overall
cargo test --test ashrae_140_validation
python scripts/generate_scorecard.py
```

---

## Key Context for Working in This Codebase

### Diagnostic Chain (per AGENTS.md)
```
Weather → Solar → Conduction → Ventilation → Zone Balance
```

### Critical Files for ASHRAE Validation
- `src/physics/gauge_zone_solver.rs` — 5R1C transient coupling (recently fixed at line 621)
- `src/validation/ashrae_140_validator.rs` — main validator
- `src/sim/thermal_model.rs` — `ThermalModelTrait` swap point
- `src/sim/ventilation.rs` — `VentilationSchedule` swap point

### Testing Commands
```bash
# All tests with EPW
FLUXION_EPW_DIR=$(pwd)/assets/weather cargo test -p fluxion --lib

# Specific test
FLUXION_EPW_DIR=$(pwd)/assets/weather cargo test --test ashrae_140_validation

# Coverage
cargo llvm-cov --open  # opens HTML report
```

### LIMIT-* Structural Issues (Known Failures)
These require **GaugeSolver rework (#1465/#1462)** and won't be fixed by unit testing alone:
- LIMIT-14: Case 960 inter-zone
- LIMIT-16: Cases 610/630/650 peak cooling
- LIMIT-17: Case 950FF night-vent
- LIMIT-18: Case 960 Blind heating_max
- LIMIT-19: InvariantChecker artificial gain
- LIMIT-20: Case 195 HighMass returns 0 kWh

---

## Open Questions from PRD (Section 8)

| # | Question | Recommended Answer |
|---|----------|-------------------|
| 1 | Tolerance per physics domain | Accept the PRD's defaults: ±0.5°C temp, ±1% relative energy/power |
| 2 | Property-based testing | Use `proptest` only for critical numeric functions (not blanket) |
| 3 | EnergyPlus version | Same as existing reference data |
| 4 | CI timeout strategy | Split jobs if total runtime > 15 min |

---

## Files to Create/Modify

| File | Action |
|------|--------|
| `coverage.toml` | Create — configure coverage targets |
| `docs/bottom_up_testing_audit.md` | Create — per-module audit results |
| `tests/weather_solar_integration.rs` | Create — wire Weather→Solar |
| `tests/solar_conduction_integration.rs` | Create — wire Solar→Conduction |
| `docs/implementation-plans/bottom-up-testing-prd.md` | Update status DRAFT→IN_PROGRESS |

---

## Verification Commands

```bash
# Dead code check (should be clean)
cargo clippy --all-targets -- -D dead_code

# All lib tests
FLUXION_EPW_DIR=$(pwd)/assets/weather cargo test -p fluxion --lib

# ASHRAE validation
FLUXION_EPW_DIR=$(pwd)/assets/weather cargo test --test ashrae_140_validation

# Coverage report
cargo llvm-cov --open
```
