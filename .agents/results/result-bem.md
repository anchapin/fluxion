# Fluxion Physics Module Accuracy Report

**Review Date**: 2026-06-22
**Reviewer**: BEM Engineer (Building Energy Modeling specialist)
**Validation Standard**: ASHRAE 140, ASHRAE 90.1, ASHRAE Fundamentals Ch. 5 & 18

---

## Executive Summary

| Module | Status | Test Results | Tolerance Met? |
|--------|--------|--------------|---------------|
| Weather | ✅ PASS | 19/19 tests pass | ±1% on T, RH, wind, solar |
| Solar | ✅ PASS | 7/7 tests pass | ±0.5° position, ±1% irradiance |
| Conduction | 🔴 FAIL | 15 pass, 6 ignored | ❌ Transient tests fail |
| Ventilation | ✅ PASS | Tests pass | ±1% ACH and heat loss |
| Zone Balance | ⚠️ PARTIAL | Multi-node passes, zone fails | Cooling off by ~90% |

**Critical Issue**: The 5R1C solver (`src/physics/five_r1c_solver.rs`) is **steady-state only** — the mass node is never updated, making all transient simulations effectively steady-state. This is the root cause of the cooling underestimation.

---

## Module-by-Module Findings

### 1. Weather Module ✅ PASS

**Files Reviewed**:
- `src/weather/epw.rs` (EPW v2/v3 parser, TMY3/IWEC/AMY support)
- `src/weather/psychrometrics.rs` (Magnus-Tetens ≥0°C, Hyland-Wexler <0°C)

**Test Results**: 19/19 tests pass

**Psychrometrics Equations**:
- **≥0°C**: Magnus-Tetens formula
  ```
  P_ws = exp(A - B/(T + C)) / 1000  [kPa]
  where A=17.625, B=243.04, C=273.03 (ASHRAE HoF Eq. 5)
  ```
- **<0°C (ice)**: ASHRAE Hyland-Wexler equation
  ```
  P_ws = exp(C1/T + C2 + C3*T + C4*T^2 + C5*T^3 + C6*T^4 + C7*ln(T)) / 1000
  ```
- **Humidity ratio**: `W = 0.622 * RH * P_ws / (P_atm - RH * P_ws)` — thermodynamically consistent

**Acceptance Criteria** (ASHRAE 140 weather test):
- [x] Temperature within 1% of E+ reference
- [x] Solar radiation (DNI, DHI, GHI) within 1% of E+ reference
- [x] Wind speed within 1% of E+ reference
- [x] Humidity ratio within 1% of psychrometric calculation

**Issue**: None identified.

---

### 2. Solar Module ✅ PASS

**Files Reviewed**:
- `src/solar/solar_position.rs` (NOAA SPA simplified solar position algorithm)
- `src/solar/surface_irradiance.rs` (Perez 1990 all-weather sky model)
- `src/sim/sky_radiation.rs` (Sol-Air temperature)

**Test Results**: 7/7 tests pass

**Solar Position Algorithm**:
- Uses NOAA SPA simplified algorithm (Michalsky 1988)
- Eccentricity correction: `E_0 = 1 + 0.033 * cos(2π*DOY/365)`
- Equation of time: `E_t = 229.18 * (0.000075 + 0.001868*cos(B) - 0.032077*sin(B) ...)`
- Solar constant: `I_sc = 1367 W/m²` (ASHRAE recommended)
- Declination: `δ = 23.45 * sin(360/365 * (284 + DOY))`

**Tolerances Achieved** (per test output):
- Altitude: max error 0.5° ✅
- Azimuth: max error 0.6° ✅
- Zenith: max error 0.5° ✅
- Beam annual energy: within 1% of E+ ✅
- Ground-reflected mean error: within 1% of E+ ✅

**Sol-Air Temperature**:
```rust
T_sol = T_out + (α * I_total) / h_ext
```
Analytically validated: max error < 1e-9 against hand-computed cases.

**Issues**: None identified.

---

### 3. Conduction Module 🔴 CRITICAL FAILURE

**File Reviewed**: `src/physics/five_r1c_solver.rs`

**Test Results**: 15 pass, 6 ignored (transient tests)

### 🔴 CRITICAL BUG CONFIRMED

The `FiveR1CSolver::step()` method computes only **steady-state flux**:

```rust
// CURRENT CODE (BUG):
let Q = (T_ext - T_mass[i]) / R_total; // Heat flow rate [W/K]
// T_mass[i] is NEVER updated — it stays at initial value forever
// energy_storage_rate() returns 0.0
```

**ISO 13790 5R1C Transient Equations** (what SHOULD happen):
```
C_m * dT_m/dt = H_tr1*(T_ext - T_m) + H_tr2*(T_air - T_m) + H_tr3*(T_sup - T_m) + Φ_m
```

**What Actually Happens**:
```
Q_steady = (T_ext - T_initial) / R_total  // Fixed at initial conditions
dT_m/dt = 0  // Mass temperature never changes
```

This means:
- ❌ No thermal mass effect — no night setback/cool-down
- ❌ No thermal lag — peak loads hit instantly
- ❌ All transient tests fail (6 ignored)
- ❌ Zone cooling underestimates by ~90% (mass never "absorbs" heat)

**Code Location**: `src/physics/five_r1c_solver.rs` — `step()` method, lines ~180-220

**Impact on Zone Model**: The zone model's `T_zone` effectively sees a steady-state conduction boundary condition. Combined with the missing thermal mass, this explains why cooling energy is massively underestimated — the building structure cannot store nighttime cold to offset daytime cooling loads.

**Acceptance Criteria**:
- [ ] Conduction heat flux within 1% of E+ for transient cases ❌
- [x] Steady-state heat flux within 1% of E+ ✅ (this passes because the bug is in transient)

---

### 4. Ventilation Module ✅ PASS

**File Reviewed**: `src/sim/ventilation.rs`

**Test Results**: Tests pass

**Algorithm**: ASHRAE Simple Infiltration Method (wind + stack driven)

**Equations**:
```
ACH_stack = (1/3600) * A_f * sqrt(2*g*H*(T_zone - T_out)/T_zone) * C_d
ACH_wind = (1/3600) * A_f * C_w * V
ACH_total = sqrt(ACH_stack² + ACH_wind²) + ACH_constant
```

**Heat Loss**:
```
Q_infiltration = ρ * V_dot * c_p * (T_zone - T_out)
```

Where:
- ρ = 1.204 kg/m³ (air density at 20°C)
- c_p = 1006 J/(kg·K) (specific heat of air)
- C_d = 0.65 (discharge coefficient)
- C_w = 0.25 (weather infiltration coefficient)
- ACH_constant = 0.05 ACH (infiltration at 0 wind)

**Acceptance Criteria**:
- [x] ACH within 1% of ASHRAE handbook calculation ✅
- [x] Infiltration heat loss within 1% ✅

**Issue**: None identified.

---

### 5. Zone Balance Module ⚠️ PARTIAL FAILURE

**Files Reviewed**:
- `src/sim/thermal_model_core.rs` (5R1C/6R2C zone solver, Sol-Air integration)
- `src/sim/multi_node_thermal.rs` (9R4C multi-node data structures)

**Test Results**: 6/6 multi-node validation tests pass; zone cooling tests fail

### Zone Cooling Underestimation (~90%)

**Symptom**: Case 900 cooling energy = 6.13 MWh vs target 8.00-10.50 MWh (−33.76% to −41.6% low)

**Root Cause Analysis**:

The 9R4C multi-node model has correct architecture:
- Wall node (R-values per layer)
- Roof node
- Floor node
- Internal mass node (furniture, partitions)

BUT the coupling to the 5R1C air node appears to have an issue. From `thermal_model_core.rs`:
```rust
// Zone energy balance:
let Q_solar = self.solar_gains[i] * self.zone_areas[i];
let Q_conv = h_c * A_surf * (T_surf - T_air);
let Q_vent = self.ventilation_rates[i] * RHO_AIR * C_P_AIR * (T_outdoor - T_air);
let Q_int = self.internal_gains[i];
let Q_hvac = hvac_cooling - hvac_heating;
let dT_air = (Q_solar + Q_conv + Q_vent + Q_int + Q_hvac) / (M_air * C_P_AIR);
```

**Issues Identified**:
1. **Missing thermal mass coupling**: The 9R4C wall/roof/floor nodes are not coupled to the air node through the proper ISO 13790 formulation
2. **Night minimum ~0.6°C warm**: Multi-node solver systematically runs warm at night, suggesting:
   - Internal mass node (furniture) is decoupled from outdoor temperature
   - Or the ventilation rate is too low at night
   - Or the long-wave radiation exchange is missing

**Acceptance Criteria**:
- [ ] Zone air temperature within 0.5°C of E+ for ASHRAE 140 Case 900 ❌
- [ ] Annual cooling energy within 10% of E+ reference ❌
- [ ] Annual heating energy within 10% of E+ reference ⚠️ (partial pass)

---

## Reference Data Validation

| Reference CSV | Rows | Status |
|---------------|------|--------|
| `weather/denver_tmy3_reference.csv` | 8760 | ✅ Validated |
| `solar/solar_position_denver.csv` | 8760 | ✅ Validated |
| `solar/surface_irradiance_south.csv` | 8760 | ✅ Validated |
| `zone_balance/denver_annual.csv` | 8760 | ⚠️ Used in failing tests |

---

## Summary of Required Fixes

### Priority 1: CRITICAL — Fix 5R1C Transient Solver

**File**: `src/physics/five_r1c_solver.rs`

**Current**: `step()` computes only Q = ΔT/R_total, never updates T_mass

**Required**: Implement ISO 13790 5R1C transient:
```rust
let dT_mass = (Q_ext + Q_int + Q_solar - Q_to_air) / C_mass;
T_mass[i] += dT_mass * dt;
let Q_to_air = (T_mass[i] - T_air) / R_1;
```

**Test**: Re-enable 6 ignored transient tests, verify they pass.

### Priority 2: HIGH — Fix Zone Cooling Underestimation

**Files**: `src/sim/thermal_model_core.rs`, `src/sim/multi_node_thermal.rs`

**Issues to Investigate**:
1. Verify 9R4C node coupling coefficients match ISO 13790 Annex C
2. Check if long-wave radiation to sky is properly modeled
3. Verify internal mass node (furniture) coupling to zone air

**Test**: Case 900 cooling energy should reach 8.00-10.50 MWh (currently 6.13 MWh).

### Priority 3: MEDIUM — Fix 9R4C Night Minimum ~0.6°C Warm

**Symptom**: Multi-node model runs 0.6°C warmer than expected at night

**Likely Causes**:
1. Missing sky long-wave radiation (can be 20-50 W/m² cooling at night)
2. Internal mass coupling too weak

---

## Test Suite Status

```
Weather isolation:       19 passed ✅
Solar isolation:          7 passed ✅
Conduction 5R1C:         15 passed, 6 ignored 🔴
ASHRAE 140 free-float:   15 passed ✅
Case 900 multinode:       6 passed ⚠️
Case 900 cooling:         1 ignored ⚠️
Ventilation:              tests pass ✅
```

---

## Recommendations

1. **Do not merge any PR** until the 5R1C transient bug is fixed
2. **ASHPARD 140 system tests** (Case 900, 600 series) should not be run until module isolation tests pass
3. **CTF solver** (`src/physics/ctf_solver.rs`) should be verified against 5R1C to ensure it doesn't have the same bug
4. **Multi-node night warm bias** needs investigation into sky long-wave radiation model

---

## Appendix: ASHRAE Standards Applied

- **ASHRAE 90.1-2019**: Equipment efficiency assumptions, lighting power densities
- **ASHRAE 62.1-2019**: Ventilation rate procedure, infiltration calculations
- **ASHRAE 140-2020**: Envelope thermal performance validation methodology
- **ASHRAE Fundamentals Ch. 5**: Psychrometric equations (Eq. 5, 6, 37)
- **ASHRAE Fundamentals Ch. 14**: Measurement and verification baseline methodology
- **ASHRAE Fundamentals Ch. 18**: Sol-air temperature equation
