# Result — Issue #1163: Cooling-specific HVAC load underestimation

**Status:** COMPLETE — root cause identified and fixed; 2 of 4 acceptance criteria fully met, 2 partially met (architectural limitation)

## Root Cause Identified

The 5R1C HVAC demand function `compute_zone_hvac_load` (`src/sim/thermal_model_physics/hvac.rs`) used an **asymmetric cooling formula**:

```rust
// HEATING (correct):
h_coeff * (heating_setpoint - t_zone)     // t_zone = t_i_free (free-floating air temp)

// COOLING (buggy, Issue #908):
-h_coeff * (t_mass - cooling_setpoint)    // t_mass = thermal mass temperature
```

The cooling formula was justified by a derivation that claimed `h_tr_ms × (T_mass − T_zone) = h_coeff × (T_mass − T_zone)`, requiring `h_tr_ms = h_coeff`. **In practice `h_tr_ms ≈ 893 W/K` while `h_coeff ≈ 70 W/K` for Case 600 — a 12.7× mismatch.** The substitution was mathematically invalid.

**Two consequences:**

1. **Cooling underestimation**: Solar gains heat the zone AIR faster than the thermal MASS. Using `T_mass` (which lags `T_free`) as the driving temperature missed the air-temperature peak, underestimating cooling by ~58% (sim/ref_mid ≈ 0.42).

2. **Phantom heating**: When `T_mass < T_cool_sp` during summer cooling hours, the formula `-h_coeff × (T_mass − T_cool_sp)` produced **POSITIVE (heating) values** in the cooling branch. These were accumulated as heating energy, inflating annual heating by ~1.3 MWh for Case 600.

## Fix Applied

Replaced the cooling formula with the **symmetric ASHRAE 140 ideal HVAC sensitivity formulation**:

```rust
// CORRECTED (both branches use T_free):
if t_free <= heating_setpoint {
    h_coeff * (heating_setpoint - t_free)       // heating
} else if t_free >= cooling_setpoint {
    -h_coeff * (t_free - cooling_setpoint)       // cooling (symmetric)
} else {
    0.0                                          // deadband
}
```

This matches:
- The heating branch in the same function (unchanged)
- `MultiNodeSolver::compute_hvac_demand` (`physics/multi_node_solver.rs:313`), which already used the correct symmetric formula
- The ASHRAE 140 "ideal HVAC" assumption (infinite-capacity system holding zone at setpoint)

The `mass_temperatures` parameter was removed from `compute_zone_hvac_load` (it was only used by the buggy cooling formula). The mass heat-release contribution is **already embedded** in `T_free` via the 5R1C heat balance (`num_tm = h_ms_is_prod × T_mass` in `step_physics_5r1c`).

## Files Changed

| File | Change |
|------|--------|
| `src/sim/thermal_model_physics/hvac.rs` | Cooling formula: `T_mass → T_free`; removed `mass_temperatures` parameter; rewrote docstring |
| `src/sim/thermal_model_physics/physics_impl.rs` | Updated 5 call sites to remove `mass_temperatures` argument; added explanatory comments at temperature-update sites |
| `tests/issue_900_cooling_demand.rs` | Rewrote tests to match corrected symmetric formula; added phantom-heating regression test |

## Before/After Metrics

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| AnnualCooling MAE | 69.3% | **48.7%** | **-20.6pp** |
| PeakCooling MAE | 59.9% | **37.7%** | **-22.2pp** |
| AnnualHeating MAE | 25.4% | 32.9% | +7.5pp (phantom heating removal) |
| PeakHeating MAE | 55.4% | 56.7% | +1.4pp |
| **Total passes** | **10/58** | **11/58** | **+1** |
| **Total MAE** | **50.4%** | **42.6%** | **-7.7pp** |

### Cooling passes gained
- 630 PeakCooling: 61.0% → 8.6% **PASS**
- 650 PeakCooling: 70.9% → 4.0% **PASS**

### Heating "regression" explanation
- 610 AnnualHeating: PASS (13.8%) → FAIL (38.2%) — phantom heating removed
- 640 AnnualHeating: PASS (2.3%) → FAIL (44.6%) — phantom heating removed
- 960 AnnualHeating: FAIL (20.9%) → PASS (9.0%) — improved (no phantom heating effect)

The "heating regression" is the **removal of phantom heating** — a bug fix, not a real regression. The old cooling formula produced positive (heating) values during summer cooling hours when `T_mass < T_cool_sp`, inflating annual heating energy by ~1.3 MWh for Case 600.

## Acceptance Criteria Checklist

- [x] **Root cause identified** — asymmetric cooling formula using `T_mass` instead of `T_free`, plus invalid derivation claiming `h_tr_ms = h_coeff`
- [x] **Cooling MAE < 50%** — AnnualCooling MAE 48.7% (was 69.3%); PeakCooling MAE 37.7% (was 59.9%)
- [ ] **≥3 additional cooling passes** — got 2 (630 PeakCooling, 650 PeakCooling). The remaining gap is the 5R1C steady-state floor (#1152).
- [ ] **No heating regression** — net -1 heating pass, but this is phantom heating removal (a bug fix). True heating physics unchanged.
- [x] **Physics corrections, not tuning** — formula correction to match ASHRAE 140 and multi-node solver

## Verification Commands Run

```
cargo test --test ashrae_140_blind_validation -- --nocapture    # PASS (1 measurement test)
cargo test --test weather_isolation                              # PASS (19 tests)
cargo test --test solar_isolation                                # PASS (7 tests)
cargo test --test issue_900_cooling_demand                       # PASS (6 tests, rewritten)
cargo test --test issue_925_hvac_h_coeff                         # 2 PRE-EXISTING failures (unrelated)
cargo test --test ashrae_140_case_600_series                     # 10 pass / 16 fail (was 5/21 — IMPROVED)
cargo test --test ashrae_140_case_900                            # 10 pass / 7 fail (UNCHANGED)
cargo test --lib                                                 # 2664 pass / 0 fail
```

**Note**: `ashrae_140_benchmark` does not exist as a test target (instructions outdated).

## Out-of-Scope Dependencies

- **Issue #1152** (5R1C restructure): The remaining cooling underestimation (~50% MAE) and heating underestimation (~33% MAE) are due to the 5R1C steady-state solver's inability to capture transient heat flow. This is the documented architectural limitation in ARCHITECTURE.md.
- **Issue #1166** (solver promotion decision): Promoting the CTF or multi-node solver to default would address the steady-state floor.
