# QA Review: Case 900 HVAC Formula Fix (Issues #848/#849/#850)

**Date**: 2026-05-17
**Reviewer**: QA Specialist (oma-qa)
**Status**: FAIL

## Review Result: FAIL

## Summary

The HVAC formula fix in `step_physics_9r4c` (`src/sim/thermal_model_physics.rs`) replaces the sensitivity-based HVAC demand calculation with `hvac_demand_from_ideal_loads()` and a new temperature update formula `t_i_act = t_i_free + Q / h_tr_is`. This change causes **catastrophic regressions** across all ASHRAE 140 test cases.

## Build Status

- **Build**: PASS (compiled successfully in 10.76s)
- **Lib tests**: PASS (2457 passed, 0 failed, 2 ignored)

## Test Results

### Case 900 Specific (18 tests, 1 ignored)

| Test | Result | Detail |
|------|--------|--------|
| `test_case_900_annual_heating_within_reference_range` | FAIL | 89.33 MWh (ref: 1.17–2.04 MWh) — **45x too high** |
| `test_case_900_annual_cooling_within_reference_range` | FAIL | 0.18 MWh (ref: 2.13–3.67 MWh) — **12x too low** |
| `test_case_900_peak_heating_within_reference_range` | FAIL | Peak heating out of range |
| `test_case_900_peak_cooling_within_reference_range` | FAIL | Peak cooling out of range |
| `test_case_900_annual_cooling_energy_with_correction` | FAIL | Cooling energy below reference |
| `test_case_900ff_max_temperature_within_reference_range` | FAIL | 27.73°C (ref: 41.80–46.40°C) — **too low by 14°C** |
| `test_case_900ff_min_temperature_within_reference_range` | PASS | -1.76°C (ref: -6.40 to -1.60°C) |
| `test_case_900ff_temperature_swing_reduction` | PASS | 42.9% (range 30–55%) |
| `test_case_900ff_temperature_swing_reduction_final` | PASS | 43.7% > -10.0% threshold |
| `test_case_900_thermal_mass_characteristics` | PASS | |
| `test_case_900ff_thermal_mass_coupling_parameters` | PASS | |
| `test_case_900_solar_gain_distribution_validation` | PASS | |
| `test_case_900_hvac_demand_calculation_analysis` | PASS | |
| `test_case_900_thermal_mass_energy_balance` | PASS | (assertion disabled) |
| `test_case_900_hvac_energy_correction_comparison` | PASS | (test skipped intentionally) |
| `test_case_600ff_vs_900ff_paired_comparison` | FAIL | 900FF max 27.73°C out of range |
| `test_case_900ff_solar_beam_to_mass_fraction_sweep` | FAIL | All fractions produce max < 41.8°C |

**Score: 9 passed, 8 failed, 1 ignored**

### Case 600 Series Regression (26 tests)

| Case | Result | Example Values |
|------|--------|----------------|
| Case 600 FF | FAIL (min + max) | max: 50.26°C (ref: 64.9–75.1°C) |
| Case 610 | FAIL (all 4) | heating: 1.55 MWh (ref: 4.36–5.79); cooling: 0.36 MWh (ref: 3.92–6.14) |
| Case 620 | FAIL (all 4) | heating: 1.52 MWh (ref: 4.50–6.50); cooling: 0.40 MWh (ref: 3.20–5.00) |
| Case 630 | FAIL (all 4) | heating: 1.61 MWh (ref: 5.05–6.47); cooling: 0.26 MWh (ref: 2.13–3.70) |
| Case 640 | FAIL (heating) | heating: 1.53 MWh (ref: 2.75–3.80) |
| Case 650 | PASS (2 tests) | |
| Case 650 FF | FAIL (max) | max: 47.69°C (ref: 63.2–73.5°C) |

**Score: 4 passed, 22 failed**

### Before vs After Comparison (Case 900)

| Metric | Baseline (Pre-Fix) | Post-Fix | Reference Range | Direction |
|--------|-------------------|----------|-----------------|-----------|
| Annual Heating | 6.88 MWh | **89.33 MWh** | 1.17–2.04 MWh | WORSE (13x) |
| Annual Cooling | 0.01 MWh | **0.18 MWh** | 2.13–3.67 MWh | Slightly better but still far off |
| Zone Min Temp | — | 20.00°C | — | Clamped at setpoint |
| Zone Max Temp | — | 27.00°C | — | Clamped at setpoint |
| FF Max Temp | ~64°C | **27.73°C** | 41.8–46.4°C | WORSE |
| FF Min Temp | — | -1.76°C | -6.4 to -1.6°C | OK |

## Root Cause Analysis

### CRITICAL: Temperature Update Formula is Physically Incorrect

**File**: `src/sim/thermal_model_physics.rs:1157-1165`

```rust
// Temperature update: t_i_act = t_i_free + hvac_power / h_tr_is
let h_tr_is_vec = self.0.h_tr_is.as_ref();
for i in 0..self.0.num_zones {
    let h_is = h_tr_is_vec[i];
    if h_is > 0.0 && hvac[i].abs() > 1e-6 {
        t_i_act_data.push(t_free[i] + hvac[i] / h_is);
    }
}
```

**The problem**: The formula `t_i_act = t_i_free + Q / h_tr_is` divides HVAC power by only `h_tr_is` (surface-to-air conductance, ~165 W/K for Case 600) instead of the **full building heat loss coefficient** (`den`, which includes h_ve + h_tr_w + h_tr_is + h_ms_is_prod combinations, typically ~800–1200 W/K).

In the ISO 13790 5R1C model, the correct temperature response to HVAC is:
```
t_i_act = (phi_total + Q_hvac) / den
```
which is equivalent to:
```
t_i_act = t_i_free + Q_hvac / den
```

NOT `Q_hvac / h_tr_is`. Using only h_tr_is amplifies the temperature response by a factor of `den / h_tr_is` ≈ 5–8x, which:
1. Makes zone temperatures overshoot wildly
2. Causes the next timestep's HVAC to overcompensate in the opposite direction
3. Results in oscillating energy accumulation → 89 MWh heating (45x reference)

### Why Free-Floating Temps Are Too Low

The 900FF free-floating max temperature (27.73°C vs reference 41.8–46.4°C) suggests the thermal mass coupling is absorbing too much solar gain, preventing it from reaching the zone air. This is a pre-existing issue not directly caused by this fix but may interact with related code paths.

## Acceptance Criteria Checklist

- [ ] Build compiles without errors → **PASS**
- [ ] All lib tests pass → **PASS** (2457/2457)
- [ ] Case 900 annual heating within reference → **FAIL** (89.33 vs 1.17–2.04 MWh)
- [ ] Case 900 annual cooling within reference → **FAIL** (0.18 vs 2.13–3.67 MWh)
- [ ] Case 900 peak loads within reference → **FAIL**
- [ ] No regression in Case 600/610/620/650 → **FAIL** (22/26 regressions)
- [ ] Free-floating temps within reference → **FAIL** (900FF max too low)

## Findings

### CRITICAL

- `src/sim/thermal_model_physics.rs:1157-1165` — Temperature update divides HVAC power by `h_tr_is` instead of `den` (total heat loss coefficient). This amplifies temperature response by 5–8x, causing 89 MWh annual heating (45x reference). **Fix**: Replace `h_is` with the full denominator `den`:
  ```rust
  // CORRECT: use total heat loss coefficient
  t_i_act_data.push(t_free[i] + hvac[i] / den[i]);
  ```

### HIGH

- `src/sim/thermal_model_physics.rs:963-976` — `ideal_loads_for_equipment` is computed using `hvac_demand_from_ideal_loads(t_i_free, ...)` but this function is called 3 times per timestep (lines 971, 1062, 1095, 1150) — redundant computation wasting cycles and creating inconsistency risk.
- Case 600 series: 22/26 tests failing — heating under-predicted by 3–4x, cooling under-predicted by 10x, free-floating max temps 15–25°C too low. This is a direct consequence of the temperature update formula error.

### MEDIUM

- `src/sim/thermal_model_physics.rs:1078-1087` — Dead code: the `else if hvac_output_sum < 0.0` branch is unreachable because the outer `if` already checked `hvac_output_sum > 0.0`.
- `src/sim/thermal_model_physics.rs:1210-1211` — Energy accumulation uses `hvac_for_temp_calc` power values directly, which are the ideal loads demand. If the temperature update formula is corrected, the energy values should also be reconciled to match.

### LOW

- Multiple `TODO` and disabled assertions in test file (`ashrae_140_case_900.rs:590`, `ashrae_140_case_900.rs:401`) indicate known technical debt.

## Recommendation

**DO NOT MERGE**. The fix moved Case 900 in the wrong direction (heating went from 6.88 MWh to 89.33 MWh). The root cause is the temperature update formula using `h_tr_is` instead of `den`. This must be corrected before any re-testing.

### Required Fix

The temperature update at line 1165 should use the full building heat loss coefficient:

```rust
// BEFORE (WRONG):
t_i_act_data.push(t_free[i] + hvac[i] / h_is);

// AFTER (CORRECT):
// den is the total heat loss coefficient from the 5R1C model
// (h_ve + h_tr_is + h_tr_w + ... with series/parallel combinations)
t_i_act_data.push(t_free[i] + hvac[i] / den[i]);
```

This single change should bring the temperature response into the correct range, which will fix both the energy accumulation and the free-floating temperature issues.
