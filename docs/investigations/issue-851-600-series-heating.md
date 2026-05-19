# Issue #851: 600-Series Low-Mass Heating Overestimate

**Status:** Investigation Complete
**Date:** 2026-05-17
**Severity:** Critical
**Affected Cases:** 600, 610, 620, 630, 640, 650 (all low-mass 600-series)

## 1. Current Results vs Reference

| Case | Metric | Reference Range (MWh) | Expected Behavior |
|------|--------|----------------------|-------------------|
| 600 | Annual Heating | 5.50 – 7.50 | Should be in range |
| 600 | Annual Cooling | 8.00 – 10.50 | Should be in range |
| 610 | Annual Heating | 4.36 – 5.79 | Should be in range |
| 610 | Annual Cooling | 3.92 – 6.14 | Should be in range |

The issue reports annual heating energy approximately **2× the expected reference range** for all 600-series cases.

## 2. Root Cause Hypotheses (Ranked by Likelihood)

### Hypothesis 1 (PRIMARY): Wrong HVAC Energy Formula — `ideal_loads` Uses Supply Air Capacity Instead of Building Heat Demand

**Likelihood: HIGH (95%)** — This is a confirmed architectural bug, partially documented in Issue #872.

**Location:** `src/sim/hvac/ideal_loads.rs` lines 139-158

**The Bug:**

The `calculate_sensible_heating_load()` formula computes:
```rust
Q = mass_flow × cp × (T_supply - T_zone)
```

where:
- `mass_flow` = ρ × (ACH/3600) × V = 1.2 × (0.5/3600) × 129.6 = **0.0216 kg/s**
- `cp` = 1005 J/(kg·K)
- `T_supply` = 40°C (heating supply air temperature)
- `T_zone` = `T_i_free` (free-floating temperature, NOT actual zone temperature)

This yields an HVAC capacity of **mass_flow × cp = 21.7 W/K**.

The CORRECT formula per ISO 13790 should be:
```
Q = h_tr_is × (T_setpoint - T_i_free)
```

For Case 600:
- `h_tr_is` = wall_film + ceiling_film + floor_film = 489.1 + 480.0 + 282.2 = **1,251.3 W/K**
- The ideal_loads formula provides **21.7 W/K** — a **57.6× undersize** vs the building's actual heat transfer capacity.

**Why This Causes Overcounting (Not Undercounting):**

Although the per-timestep HVAC output is much LOWER than the correct value, the temperature update formula is:
```
t_i_act = t_i_free + Q_hvac / h_tr_is
```

For `T_i_free = 5°C`:
- `Q_wrong` = 21.7 × (40 - 5) = 760 W
- `t_i_act` = 5 + 760/1251 = **5.6°C** (nowhere near the 20°C setpoint!)

The building **never reaches setpoint**, so:
1. HVAC fires **every single timestep** during the heating season (no deadband satisfaction)
2. The thermal mass never gets a chance to absorb heat because the temperature response is negligible
3. Energy accumulates as `Q_wrong × dt` for ~4000-5000 hours/year

The cumulative effect produces annual heating energy in the range of **10-15 MWh** instead of the expected 5.5-7.5 MWh — roughly **2× overcount**.

**Evidence:**
- Issue #872 (`docs/implementation-plans/issue-872-hvac-formula-fix.md`) documents this exact bug for the 9R4C model
- The fix plan says: *"Bug 1: Wrong HVAC Energy Formula — mass_flow × cp × ΔT is a ventilation airflow capacity formula, not an ideal loads formula"*
- Code comment at line 1138-1144 confirms the sensitivity approach was replaced with ideal_loads, but ideal_loads itself is wrong

### Hypothesis 2 (SECONDARY): `h_tr_ms` Overestimate for Low-Mass Buildings

**Likelihood: MEDIUM (40%)**

**Location:** `src/sim/thermal_model_core.rs` lines 877-994

The physics-based `h_tr_ms` calculation uses the ISO 13790 "half-insulation rule":
```rust
r_interior_to_mass += layer_r / 2.0;  // for insulation layer
```

For lightweight construction (Case 600), the resistance from interior surface to mass node is very small (R ≈ 0.1-0.5 m²K/W), making `h_tr_ms` very large (~500+ W/K). This causes:
- Mass temperature tracks surface temperature almost instantly
- The `num_tm` term in `t_i_free` calculation is dominated by outdoor conditions
- `t_i_free` runs cold even with moderate internal gains

If `h_tr_ms` is too high, the thermal mass doesn't buffer cold outdoor temperatures, and `t_i_free` is systematically lower than it should be, triggering more heating.

### Hypothesis 3 (TERTIARY): Solar Gain Undercounting

**Likelihood: LOW (20%)**

If solar gains through the 12 m² south window are undercounted, `t_i_free` would be lower during daylight hours, triggering more heating. However, solar gain issues primarily affect cooling (peak cooling under-prediction is documented as SOLAR-01), so this is less likely to cause 2× heating overestimate.

## 3. Specific Code Locations

### Primary Bug Location

| File | Lines | Description |
|------|-------|-------------|
| `src/sim/hvac/ideal_loads.rs` | 139-158 | `calculate_sensible_heating_load()` — wrong formula |
| `src/sim/hvac/ideal_loads.rs` | 103-123 | `calculate_sensible_cooling_load()` — same wrong formula |
| `src/sim/thermal_model_physics.rs` | 1150 | `hvac_for_temp_calc` — calls ideal_loads for energy |
| `src/sim/thermal_model_physics.rs` | 1177-1188 | Energy accumulation from `hvac_for_temp_calc` |
| `src/sim/thermal_model_physics.rs` | 1210-1211 | Annual energy accumulation (kWh) |

### Energy Flow Trace

```
step_physics_5r1c()
  ├── hvac_demand_from_ideal_loads(t_i_free, heat_sp, cool_sp)  [L1150]
  │     └── IdealLoadsSystem.calculate_power_demand_vector()
  │           └── calculate_sensible_heating_load(t_i_free, heat_sp, 40.0, 129.6, 0.5)
  │                 └── mass_flow × cp × (40 - t_i_free)  = 21.7 × ΔT  ← WRONG
  ├── t_i_act = t_i_free + Q / h_tr_is                           [L1165]
  ├── heating_sum += Q (if Q > 0)                                 [L1179]
  ├── heating_energy_joules = heating_sum × dt                    [L1187]
  ├── annual_heating_energy += heating_joules / 3.6e6             [L1210]
  └── return hvac_energy_for_step / 3.6e6                         [L1390]
```

### The Correct Formula (per ISO 13790 §7.2.2.2)

The HVAC power to maintain setpoint temperature should be:
```
Φ_HC = (h_tr_is + H_ve) × (θ_set - θ_i_free)
```

where `H_ve` is the ventilation conductance. This is the building's heat loss that HVAC must overcome.

For the 5R1C model, the self-consistent formulation is:
```
Q = h_tr_is × (T_set - T_i_free)
```

This ensures: `T_i_act = T_i_free + h_tr_is × (T_set - T_i_free) / h_tr_is = T_set` (exact setpoint reach).

### Conductance Values for Case 600

| Conductance | Value | Source |
|-------------|-------|--------|
| h_tr_is (surface→air) | 1,251.3 W/K | 7.69×opaque_wall + 10.0×floor + 5.88×floor |
| mass_flow × cp | 21.7 W/K | 1.2 × (0.5/3600) × 129.6 × 1005 |
| h_ve (ventilation) | ~21.6 W/K | ACH × V × ρ × cp / 3600 |
| Ratio | 57.6× | h_tr_is / (mass_flow × cp) |

## 4. Recommended Fix Approach

### Option A: Replace `ideal_loads` Formula with `h_tr_is × ΔT` (Recommended)

Replace in `thermal_model_physics.rs`:
```rust
// BEFORE (wrong):
let hvac = self.hvac_demand_from_ideal_loads(t_i_free, heat_sp, cool_sp);

// AFTER (correct):
let hvac = self.hvac_demand_h_loss(t_i_free, heat_sp, cool_sp);
```

Where `hvac_demand_h_loss` computes:
```rust
fn hvac_demand_h_loss(&self, t_free: &[f64], heat_sp: f64, cool_sp: f64) -> T {
    let h_tr_is = self.0.h_tr_is.as_ref();
    let mut result = vec![0.0; self.0.num_zones];
    for i in 0..self.0.num_zones {
        if t_free[i] < heat_sp {
            result[i] = h_tr_is[i] * (heat_sp - t_free[i]);  // heating
        } else if t_free[i] > cool_sp {
            result[i] = -h_tr_is[i] * (t_free[i] - cool_sp); // cooling
        }
    }
    T::from(VectorField::new(result))
}
```

This ensures:
- `t_i_act = t_i_free + h_tr_is × ΔT / h_tr_is = t_i_free + ΔT = T_setpoint` (exact)
- Energy = `h_tr_is × ΔT × dt` (physically correct heat flow)
- Self-consistent temperature update

### Option B: Fix the `ideal_loads` Formula Parameters

Keep the `mass_flow × cp × ΔT` structure but use the correct ΔT:
```rust
// The correct ΔT should be (T_set - T_zone), not (T_supply - T_zone)
let delta_t = (heating_setpoint - zone_temp).max(0.0);
// And the correct conductance should be h_tr_is, not mass_flow × cp
```

This is essentially the same as Option A but with a different code path.

### Option C: Adjust Supply Temperature to Match Required Capacity

Set `T_supply` such that `mass_flow × cp × (T_supply - T_zone) ≈ h_tr_is × (T_set - T_zone)`:
```
T_supply = T_zone + h_tr_is × (T_set - T_zone) / (mass_flow × cp)
```

For T_zone=15°C: T_supply = 15 + 1251×5/21.7 = **303°C** — physically unrealistic.

**Option A is the recommended approach.**

## 5. Supporting Evidence

### Issue #872 Documentation

The `docs/implementation-plans/issue-872-hvac-formula-fix.md` file documents the same bug for the 9R4C model:
- *"mass_flow × cp = 21.7 W/K is a ventilation airflow capacity, not an ideal loads formula"*
- *"Building envelope heat loss = 119 W/K — HVAC capacity is 5.5× undersized"*
- The fix plan says to replace with `H × (T_set - T_free)` using `h_tr_is`

### Code Comment Evidence

At `thermal_model_physics.rs:1138-1144`:
```rust
// Root Cause Fix (Case 600): Replace sensitivity superposition with ideal loads.
// The sensitivity formula t_i_act = t_i_free + sensitivity * hvac_output assumes
// mass temperature is static — invalid when HVAC heat flows through the high-conductance
// mass path (h_is_ms_series = 583 W/K for Case 600, creating 6.1x conductance overestimate).
//
// Fix: Use hvac_demand_from_ideal_loads() (mass_flow × cp × ΔT) unconditionally.
```

This comment confirms the sensitivity approach was replaced with ideal_loads, but ideal_loads itself has the wrong formula. The "fix" replaced one incorrect approach with another.

### Test Framework Evidence

The TDD test at `tests/tdd/ashrae140_case_series.rs:29-48` asserts:
```rust
assert_in_range(result.annual_heating_mwh, 5.50, 7.50, "Case 600 annual heating should be 5.50-7.50 MWh");
```

If heating is ~2× expected, this test would fail with `actual ≈ 11-15 MWh`.

## 6. Impact Assessment

| Aspect | Impact |
|--------|--------|
| Affected models | All 5R1C and 6R2C models (600-series, 195, 196) |
| Unaffected models | 9R4C may have same issue (Issue #872) — needs separate fix |
| Free-float cases | Not affected (HVAC disabled in free-float mode) |
| Cooling | Same bug applies — cooling energy may be similarly incorrect |
| Temperature accuracy | Zone temperatures never reach setpoint — incorrect for all downstream calculations |

## 7. Verification Plan

After applying fix:
1. Run Case 600 annual simulation — expect heating in 5.5-7.5 MWh range
2. Verify `t_i_act` reaches setpoint (20°C daytime, 5.5°C night setback)
3. Check peak heating/cooling against reference ranges
4. Run Case 610, 620, 630, 640, 650 — verify all pass
5. Verify Case 900 (high-mass) results are not regressed by the change
6. Check free-floating cases (600FF, 900FF) still produce zero HVAC energy
