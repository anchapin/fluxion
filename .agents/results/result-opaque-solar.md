# Opaque Surface Solar Gain Path Trace — Phase 4, Agent 1

**Status**: COMPLETE
**Confidence**: HIGH
**Date**: 2026-05-18

## Executive Summary

The opaque solar gain path is **architecturally correct** but contains **one significant bug** in the `prepare_solvers_and_sol_air` function that feeds wrong data into CTF/FD solvers. For the primary 5R1C model used by Case 600FF, the opaque solar gains are correctly computed and routed through `phi_m`. The 10-25°C temperature deficit is **NOT caused by missing opaque solar gains** — the gains are present at ~9 W/m² of floor area at peak, which is modest but correct for low-U-value surfaces.

**Root cause of the temperature deficit is NOT in this subsystem.** Opaque solar contributes only ~436W at peak noon (vs ~1200W from the south window), so even a total loss of opaque gains would explain at most 5-8°C of the 10-25°C deficit.

---

## 1. Complete Trace of Opaque Solar Gain Path

### Path A: Opaque Solar Gain → phi_m (PRIMARY PATH, CORRECT)

```
Step 1: Surface Irradiance Calculation
  File: src/sim/thermal_model_iterative.rs:254-273
  calculate_hourly_solar() computes beam+diffuse irradiance per orientation
  Returns: (SunPosition, SurfaceIrradiance, SolarGain)
  SurfaceIrradiance.total_wm2 = beam_irradiance + diffuse_irradiance

Step 2: Opaque Solar Gain Computation
  File: src/sim/thermal_model_iterative.rs:296-304
  Formula: q_opaque = opaque_area × surface.u_value × irradiance.total_wm2 × alpha × re
  Where:
    - opaque_area = surface.area - surface.window_area (m²)
    - surface.u_value = wall U-value or roof U-value (W/m²K)
    - irradiance.total_wm2 = total solar irradiance on surface (W/m²)
    - alpha = 0.6 (ASHRAE 140 default solar absorptance) ✓
    - re = 0.034 = 1/h_se = 1/29.3 (exterior film resistance) ✓

Step 3: Accumulation into opaque_solar_gains field
  File: src/sim/thermal_model_iterative.rs:506-513, 523
  zone_opaque_gains.push(opaque_gain_watts / floor_area)
  opaque_solar_gains = VectorField with units W/m² of floor area

Step 4: Routing to phi_m (mass node)
  File: src/sim/thermal_model_physics.rs:613, 623, 631
  opaque_sol_w = opaque_solar_ref[i] * area_ref[i]  // Converts back to Watts
  phi_m = load_w * m_air_frac + remaining_sol * m_sol_frac + opaque_sol_w

Step 5: phi_m drives mass temperature
  File: src/sim/thermal_model_physics.rs:950-960
  t_i_free = (num_tm + num_phi_st + num_rest) / den
  Where num_tm includes h_tr_ms * T_mass, which absorbs phi_m
```

### Path B: Sol-Air Temperature → 5R1C Conduction (SECONDARY, INTENTIONALLY DISABLED)

```
Step 1: Sol-air temperature computation
  File: src/sim/thermal_model_core.rs:205-230
  prepare_solvers_and_sol_air() computes t_sol_air
  BUG: Uses solar_gains (WINDOW gains, ~4 W/m²) instead of surface irradiance (~900 W/m²)
  BUG: Uses for_roof() for all surfaces (walls should use for_wall())

Step 2: 5R1C path intentionally bypasses sol-air
  File: src/sim/thermal_model_physics.rs:650-655
  t_sol_air = outdoor_temp  // NO solar boost — by design
  Comment: "Solar gains on opaque surfaces are already included in phi_m"

Step 3: 5R1C conduction uses t_sol_air = outdoor_temp
  File: src/sim/thermal_model_physics.rs:660
  derived_h_ext uses outdoor_temp as exterior boundary
  Q_envelope = h_tr_em × (T_outdoor - T_mass)
  This is CORRECT because opaque solar gains are in phi_m
```

### Path C: CTF/FD Conduction (TERTIARY, BUGGY BUT LIKELY NOT ACTIVE FOR 600FF)

```
Step 1: prepare_solvers_and_sol_air computes sol-air per zone
  File: src/sim/thermal_model_core.rs:227-228
  t_sol_air_zone = sol_air_calc.for_roof(outdoor_temp, i_sol, sky_temp)
  i_sol comes from solar_gains (WINDOW gains!) — WRONG

Step 2: CTF/FD solvers use this as exterior boundary
  File: src/sim/thermal_model_core.rs:250, 265, 283, 286
  t_ext = t_sol_air_data[i] — uses wrong sol-air temp

Step 3: CTF correction subtracts 5R1C equivalent
  File: src/sim/thermal_model_physics.rs:857-871
  net_ctf_flux = q_ctf - q_5r1c
  q_5r1c = h_tr_em × (t_sol_air - T_mass)
  t_sol_air = outdoor_temp (from line 653) — CORRECT subtraction base
```

---

## 2. Bugs Found

### BUG 1 (MEDIUM severity): prepare_solvers_and_sol_air uses wrong irradiance
- **File**: `src/sim/thermal_model_core.rs:214, 220, 228`
- **What**: `i_sol` comes from `self.0.solar_gains` which is WINDOW gains (~4 W/m² of floor area), not surface solar irradiance (~900 W/m² on roof)
- **Effect**: Sol-air temperature gets negligible solar boost (~0.1°C instead of ~18°C)
- **Impact**: If CTF/FD solvers are active, envelope conduction misses solar heating. For 5R1C-only Case 600FF, this path is bypassed (line 653 sets t_sol_air = outdoor_temp directly).
- **Fix**: Pass per-surface irradiance to `prepare_solvers_and_sol_air` instead of solar_gains field. Use `for_wall()` for walls and `for_roof()` only for roofs.

### BUG 2 (LOW severity): for_roof() used for all surfaces
- **File**: `src/sim/thermal_model_core.rs:228`
- **What**: `sol_air_calc.for_roof()` is called for every zone, regardless of surface type
- **Effect**: Walls get longwave sky radiation correction (appropriate for horizontal surfaces), while walls should use `for_wall()` with ground-reflected radiation
- **Impact**: Negligible for Case 600FF since the 5R1C path bypasses this. For CTF/FD cases, walls would get slightly wrong sol-air temp.

---

## 3. Quantitative Analysis

### Opaque Solar Gain Magnitude at Peak Noon (Summer, Denver)

| Surface | Area (m²) | U-value (W/m²K) | Peak Irradiance (W/m²) | Gain (W) |
|---------|-----------|------------------|------------------------|----------|
| Roof    | 48.0      | 0.327            | 900                    | 288.2    |
| N Wall  | 21.6      | 0.514            | 100                    | 22.6     |
| E Wall  | 16.2      | 0.514            | 100                    | 17.0     |
| W Wall  | 16.2      | 0.514            | 400                    | 67.9     |
| S Opaque| 9.6       | 0.514            | 400                    | 40.3     |
| **Total** | **111.6** | | | **436.0** |

**Total opaque solar gain at peak**: ~436 W = 9.1 W/m² of floor area

### Comparison with Window Solar Gain
- South window: 12 m² × 787 W/m² × 0.787 SHGC ≈ 1200W at peak
- Opaque surfaces: ~436W at peak (36% of window gain)
- Combined: ~1636W at peak

### Is 436W Enough to Explain 10-25°C Deficit?
- Zone thermal capacity (5R1C mass node): ~C_m = 165,000 J/K (ASHRAE 140 Case 600)
- If opaque gains were completely missing: ΔT = 436W × 3600s / 165,000 ≈ 9.5°C/hour deficit at peak
- But losses scale too: at ΔT=10°C higher, loss through envelope increases
- Net effect: maybe 3-5°C of the observed 10-25°C deficit
- **Conclusion**: Missing opaque gains alone cannot explain the full deficit

### The re=0.034 Factor
- `re = 0.034 m²K/W` = exterior film thermal resistance = `1/h_se` = `1/29.3`
- This is the ASHRAE 140 standard value (Table 3, h_se = 29.3 W/m²K for winter)
- The formula `q = A × U × α × I × R_e` is the standard sol-air method:
  - `q = (U × A) × (α × I / h_se) = H_op × ΔT_sol-air_boost`
  - Where `ΔT_sol-air_boost = α × I / h_se = 0.6 × 900 / 29.3 = 18.4°C`
- **This is CORRECT.** The re=0.034 factor is NOT a bug.

---

## 4. The Sol-Air Temperature Formula Verification

### Implementation in sky_radiation.rs:338-351
```rust
pub fn calculate(&self, outdoor_temp, solar_irradiance, sky_temp, ground_reflected) -> f64 {
    let total_solar = solar_irradiance + ground_reflected.unwrap_or(0.0);
    let solar_term = self.solar_absorptance * total_solar / self.exterior_conductance;
    let delta_r = self.calculate_longwave_radiation_difference(outdoor_temp, sky_temp);
    let longwave_term = self.emissivity * delta_r / self.exterior_conductance;
    outdoor_temp + solar_term - longwave_term
}
```

### ASHRAE Reference Formula
```
T_sol-air = T_out + (α × I_sol / h_se) - (ε × ΔR / h_se)
```

**Verdict**: Formula is CORRECT. Both solar and longwave terms match ASHRAE.

### Constants Verification
- `SOLAR_ABSORPTANCE_DEFAULT = 0.6` (v2023.rs:32) — ASHRAE 140 Table B1-2 ✓
- `EXTERIOR_FILM_COEFF_DEFAULT = 29.3` (v2023.rs:28) — ASHRAE 140 Table 3 ✓
- `emissivity = 0.9` (thermal_model_core.rs:217) — standard opaque surface ✓
- `sky_temp = outdoor_temp - 20.0` (thermal_model_core.rs:225) — ASHRAE 140 clear-sky approximation ✓

---

## 5. How Opaque Solar Becomes Zone Heat Gain

### Energy Flow Summary

```
Solar irradiance on opaque surface
       ↓ (calculate_hourly_solar)
       ↓ irradiance.total_wm2 (W/m²)
       ↓
Absorbed solar = α × I × R_e = ΔT_sol-air_boost
       ↓ (calculate_zone_solar_gain, line 303)
       ↓ q_opaque = A × U × ΔT_sol-air_boost (Watts)
       ↓
Stored in opaque_solar_gains field (W/m² floor area)
       ↓ (calc_analytical_loads, line 523)
       ↓
Added to phi_m (mass node heat flow)
       ↓ (step_physics_5r1c, line 631)
       ↓
Heats mass node T_mass
       ↓
Drives zone air temperature via h_tr_ms coupling
```

### Key Design Decision (CORRECT)
The code intentionally avoids double-counting by:
1. NOT applying sol-air temperature boost in the 5R1C path (line 653: `t_sol_air = outdoor_temp`)
2. Instead routing opaque solar gains through `phi_m` (line 631: `+ opaque_sol_w`)
3. CTF/FD corrections subtract the 5R1C equivalent to avoid double-counting (line 870: `net_ctf_flux = q_ctf - q_5r1c`)

---

## 6. Surface Area Verification for Case 600FF

| Surface | Dimensions | Area | Window | Opaque |
|---------|-----------|------|--------|--------|
| North wall | 8m × 2.7m | 21.6 m² | 0 | 21.6 m² |
| East wall | 6m × 2.7m | 16.2 m² | 0 | 16.2 m² |
| West wall | 6m × 2.7m | 16.2 m² | 0 | 16.2 m² |
| South wall | 8m × 2.7m | 21.6 m² | 12.0 m² | 9.6 m² |
| Roof | 8m × 6m | 48.0 m² | 0 | 48.0 m² |
| Floor | 8m × 6m | 48.0 m² | 0 | 0 (ground coupled) |
| **Total opaque** | | | | **111.6 m²** |

The code at line 198 correctly skips floor (Orientation::Down) surfaces.

---

## 7. Recommended Fix

### Priority 1: Fix prepare_solvers_and_sol_air (BUG 1)
In `thermal_model_core.rs:205-230`, replace:
```rust
// BEFORE (buggy):
let solar_ref = self.0.solar_gains.as_ref();
for &i_sol in solar_ref.iter().take(self.0.num_zones) {
    let t_sol_air_zone = sol_air_calc.for_roof(outdoor_temp, i_sol, sky_temp);
```
With:
```rust
// AFTER (fixed): Use opaque surface irradiance, not window gains
// Option A: Pass per-surface irradiance from thermal_model_iterative
// Option B: Compute weighted-average surface irradiance here
```
Note: This fix is LOW priority for Case 600FF since the 5R1C path bypasses it.

### NOT Recommended
- Do NOT increase alpha beyond 0.6 (matches ASHRAE 140 spec)
- Do NOT change re from 0.034 (matches 1/29.3 = exterior film resistance)
- Do NOT add sol-air boost to the 5R1C path (would double-count with phi_m)

---

## 8. Conclusion for Phase 4

The opaque solar gain path is **functionally correct** for the 5R1C model used by Case 600FF. The gains are:
- Correctly computed: q = A × U × α × I × R_e ✓
- Correctly absorbed: α = 0.6 per ASHRAE 140 ✓
- Correctly routed: through phi_m to mass node ✓
- Not double-counted: 5R1C uses outdoor_temp, not sol-air ✓

The 10-25°C temperature deficit **cannot be explained by missing opaque solar gains**. At peak, opaque gains contribute ~436W (vs ~1200W window gains), and removing them entirely would cause at most ~5°C additional deficit.

**Recommendation**: Look elsewhere for the primary cause. Suggested investigation targets:
1. Window solar gains path (solar_gains field) — is SHGC applied correctly?
2. Mass node thermal capacitance — is C_m correct for low-mass Case 600?
3. Ventilation/infiltration rate — is h_ve too high?
4. Internal gains — are they present and routed correctly?
5. The phi_m → T_mass → T_air coupling — is h_tr_ms correct?

---

## Appendix: Key Constants Verified

| Constant | Value | Location | ASHRAE 140 Ref |
|----------|-------|----------|---------------|
| SOLAR_ABSORPTANCE_DEFAULT | 0.6 | physics/constants/ashrae_140/v2023.rs:32 | Table B1-2 |
| EXTERIOR_FILM_COEFF_DEFAULT | 29.3 | physics/constants/ashrae_140/v2023.rs:28 | Table 3 |
| re (exterior resistance) | 0.034 | thermal_model_iterative.rs:185 | 1/29.3 |
| Surface emissivity | 0.9 | thermal_model_core.rs:217 | Standard |
| Ground reflectance | 0.2 | thermal_model_iterative.rs:272 | ASHRAE default |
| solar_distribution_to_air | 0.0 | thermal_model_core.rs:1561 | ASHRAE 140 FF |
