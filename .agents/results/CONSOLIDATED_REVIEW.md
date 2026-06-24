# Fluxion Multi-Agent Review: Consolidated Report

**Review Date**: 2026-06-22
**Agents**: BEM Engineer, Architecture Reviewer, QA Specialist, Performance Engineer, Code Quality Reviewer
**Focus**: Core building energy simulation accuracy

---

## Executive Summary

Fluxion has a well-designed architecture and solid foundation, but **3 critical bugs block Phase 1 completion**:

| Priority | Issue | Impact | Module |
|----------|-------|--------|--------|
| 🔴 **CRITICAL** | 5R1C `step()` is steady-state only — mass never updated | Zone cooling off by ~90% | Conduction |
| 🔴 **CRITICAL** | `WeatherDependentVentilation::get_ach()` ignores weather | Contract violation | Ventilation |
| 🔴 **CRITICAL** | 4 of 6 conduction CSVs are synthetic (not E+) | False validation confidence | Testing |

**Module Health**:
- Weather: ✅ PASS
- Solar: ✅ PASS  
- Conduction: 🔴 FAIL (transient tests ignored)
- Ventilation: ⚠️ FAIL (contract violation)
- Zone Balance: ⚠️ PARTIAL (cooling off by ~90%)

---

## Critical Issues (Must Fix)

### 1. 🔴 5R1C Transient Solver Bug

**Location**: `src/physics/five_r1c_solver.rs` — `step()` method

**Root Cause**: The solver computes only `Q = ΔT / R_total` using initial `T_mass`. The mass temperature is **never updated** between timesteps.

```rust
// CURRENT (BUG):
let Q = (T_ext - T_mass[i]) / R_total;  // T_mass never changes
// energy_storage_rate() returns 0.0
```

**Required Fix** (ISO 13790 5R1C transient):
```rust
let dT_mass = (Q_ext + Q_int + Q_solar - Q_to_air) / C_mass;
T_mass[i] += dT_mass * dt;
let Q_to_air = (T_mass[i] - T_air) / R_1;
```

**Impact**:
- 6 transient tests are `#[ignore]` in `tests/conduction_5r1c_isolation.rs`
- Zone cooling underestimates by ~90%
- Night setback has no effect
- Thermal lag is non-existent

**File References**:
- `src/physics/five_r1c_solver.rs:180-220`
- `tests/conduction_5r1c_isolation.rs:459,540,598,674,744,809`

---

### 2. 🔴 WeatherDependentVentilation Contract Violation

**Location**: `src/sim/ventilation.rs:323-329`

**Issue**: `get_ach(&self, hour: usize)` ignores all weather parameters (outdoor temp, indoor temp, wind speed, zone volume) and returns `base_ach` regardless.

```rust
// Current (broken):
fn get_ach(&self, _hour: usize) -> f64 { self.base_ach }
```

**Fix**: Either add weather parameters to trait method, or rename to `get_base_ach()` and promote `get_ach_weather()` to trait level.

---

### 3. 🔴 Synthetic Data Masquerading as EnergyPlus Reference

**Location**: `tests/reference_data/conduction/`

| File | Status |
|------|--------|
| `step_response_200mm_concrete.csv` | ✅ Real E+ 25.2.0 |
| `step_response_fixed_zone_20c.csv` | ✅ Real E+ 25.2.0 |
| `step_response_lightweight.csv` | ⚠️ SYNTHETIC |
| `step_response_roof.csv` | ⚠️ SYNTHETIC |
| `step_response_floor.csv` | ⚠️ SYNTHETIC |
| `step_response_composite.csv` | ⚠️ SYNTHETIC |

**Impact**: 67% of conduction validation passes against fabricated data. Steady-state tests give false confidence.

---

## Architecture Issues

### 4. Module 3 ≠ Zone Solver (ADR-002 Confusion)

**Files**: `src/physics/solver_trait.rs` vs `src/sim/thermal_model_core.rs`

ARCHITECTURE.md line 212 implies `HeatConductionSolver` is the zone-level 5R1C solver. **This is false**:

| Path | Location | Purpose |
|------|----------|---------|
| `FiveR1CSolver` | `physics/five_r1c_solver.rs` | Per-surface, **steady-state only** |
| Zone network | `thermal_model_core.rs` | **Actual** zone solver (5R1C/9R4C) |

**Fix**: Update ARCHITECTURE.md to explicitly document this split.

---

### 5. 9R4C Routing via String Matching

**Location**: `src/sim/thermal_model_core.rs:1865-1869`

```rust
if spec.case_id.starts_with("9") && spec.case_id != "960" { ... }
```

**Fix**: Use `ThermalModelType` enum throughout for type-level routing.

---

## Testing Infrastructure Issues

### 6. CI Release Gates Are No-Ops

**Location**: `release_gates.yaml`

```yaml
min_pass_rate: 4.0    # 96% failure = PASS
max_mae: 300.0        # 300% error = PASS
```

### 7. Module Isolation Tests Not in CI

The CI gate only runs ASHRAE 140 tests — none of the 5 module isolation tests are included.

### 8. 33 `#[ignore]` Tests

Many tests are blocked with "TODO: Fix" comments but no issue tracking.

---

## Performance Opportunities

### 9. Solar Position Recalculation (HIGH)

**Location**: `src/sim/thermal_model_physics/physics_impl.rs:1878,2098`

Solar position recalculated per-surface per-timestep despite being deterministic:
- 5 surfaces × 8760 timesteps = 43,800 calls/year
- Only 8,760 unique datetime values
- **Recommendation**: Cache by `(year, month, day, hour)` — 5× speedup potential

### 10. Trig Value Redundancy (MEDIUM)

**Location**: `src/solar/solar_position.rs:99-196`

`gamma.cos()`, `gamma.sin()`, `lat_rad.sin()`, etc. recomputed each call despite being constants or day-of-year dependent.

---

## Code Quality Notes

| Pattern | Assessment |
|---------|------------|
| Error handling (`thiserror`) | ✅ Good |
| Trait objects (`Box<dyn>`) | ✅ Appropriate |
| Unsafe blocks (7 total) | ⚠️ Needs audit + safety comments |
| `OnceLock` in `thermal_model_core.rs` | ⚠️ Limits testability |
| `thermal_model_core.rs` size (~1100+ lines) | ⚠️ Consider splitting |

---

## Prioritized Action Items

### Immediate (Blocker)
1. **Fix 5R1C transient bug** — Implement ISO 13790 mass temperature update
2. **Fix WeatherDependentVentilation** — Honor weather parameters in `get_ach()`
3. **Regenerate synthetic conduction CSVs** or relabel as test fixtures

### Short-term
4. Update ARCHITECTURE.md Module 3 vs Module 5 documentation
5. Replace `case_id` string matching with `ThermalModelType` enum
6. Add module isolation tests to CI gate
7. Fix release gate thresholds (`min_pass_rate: 4.0` → `min_pass_rate: 60.0`)

### Medium-term
8. Implement solar position caching
9. Cache declination by day-of-year
10. Add solar gain distribution isolation test
11. Triage 33 `#[ignore]` tests — create issues or remove

---

## Validation Status

| Module | Isolation Tests | E+ Reference | Tolerance | CI Gate |
|--------|----------------|--------------|-----------|---------|
| Weather | ✅ 19 tests | ✅ Real | 1% | ❌ Not in CI |
| Solar | ✅ 7 tests | ✅ Real | 0.5°/1% | ❌ Not in CI |
| Conduction | ⚠️ 15 pass, 6 ignored | ⚠️ 67% synthetic | 0.1-2% | ❌ Not in CI |
| Ventilation | ✅ Tests pass | ⚠️ No E+ ref for wind+stack | 1% | ❌ Not in CI |
| Zone Balance | ⚠️ All ignored | ⚠️ ±15% (vs 1% spec) | ±15% | ✅ In CI |

---

## Appendix: Agent Reports

Full reports available at:
- `.agents/results/result-bem.md` — Physics accuracy
- `.agents/results/result-architecture.md` — Trait architecture  
- `.agents/results/result-qa.md` — Testing infrastructure
- `.agents/results/result-backend.md` — Performance optimization
- `.agents/results/architecture/fluxion-trait-architecture-review.md` — Detailed architecture