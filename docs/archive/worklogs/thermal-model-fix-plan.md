# Thermal Model Fix Plan — ASHRAE 140 HVAC & Case 900 Heating

## Diagnosis Summary

### Pre-existing Failures (on `main` before any thermal model changes)
1. **HVAC cooling detection fails** — 0 kWh cooling output despite 35°C outdoor + 1000 W/m² solar
2. **Case 900 heating: 92% error** — 3.19 MWh predicted vs 1.66 MWh reference
3. **Case 900 cooling: 54% error** — 1.14 MWh predicted vs 2.49 MWh reference

### Root Causes

#### 1. HVAC Cooling Detection Failure
- **Root cause:** `h_tr_em` (opaque envelope-to-exterior conductance, ~27.5 W/K for Case 900) was completely missing from `derived_h_ext` in `update_optimization_cache()`
- **Effect:** The HVAC sensitivity formula `sensitivity = 1/h_total` used `h_total = h_tr_w + h_ve` (58.2 W/K) instead of the correct `h_total = h_tr_em + h_tr_w + h_ve` (85.7 W/K)
- **Impact:** Sensitivity was too low (0.012 vs 0.009 K/W), causing the controller to underestimate the temperature rise per watt of HVAC power. At 35°C outdoor, the free-float temperature was miscalculated, preventing cooling mode from triggering
- **Fix:** Add `h_tr_em` to `derived_h_ext`

#### 2. Case 900 Heating/Cooling Error (92%)
- **Root cause:** The 6R2C thermal network uses a lumped `h_tr_is` (369 W/K) for ALL surfaces including the south wall. But the south wall has insulation (k=0.04 W/mK) in the middle, creating a weak thermal coupling from interior to exterior (~25 W/K effective)
- **Effect:** The model predicts the south wall temperature closely tracks interior air temperature (large h_tr_is), causing excessive heat exchange. The actual wall is well-insulated and the interior surface temperature is dominated by exterior conditions, not interior air
- **Impact:** The mass node receives wrong thermal signals, steady-state temperatures are incorrect, leading to wrong HVAC energy predictions
- **Fix:** Requires restructuring the thermal network to separate opaque wall path from window/floor/ceiling paths

#### 3. Side Effect of h_tr_em Fix
- Adding `h_tr_em` to `derived_h_ext` creates a bypass around the thermal mass for the wall's thermal path. The wall's exterior conductance (h_tr_em=27.5) now directly affects interior temperature, bypassing the mass node's buffering
- **Impact:** Worsens Case 900 heating unless compensated
- **Compensation:** Reducing `a_int` from 0.5 to 0.1 weakens the furniture mass coupling, partially offsetting the bypass effect
- **Remaining error:** ~198% (vs baseline 92%)

## Fix Strategy

### Immediate Fix (addresses HVAC cooling, minimizes regression)
1. **Add `h_tr_em` to `derived_h_ext`** — Fixes HVAC sensitivity
2. **Reduce `a_int` from 0.5 to 0.1** — Minimizes Case 900 heating regression
3. **Do NOT add `h_tr_me` to `derived_term_rest_1`** — Makes Case 900 heating worse (+5.5%)

**Expected result:**
- HVAC cooling: PASS ✅
- Case 900 heating: ~198% (pre-existing 92% + new ~100% from h_tr_em side effect)
- Case 900 cooling: ~75% (pre-existing 54% + new ~20%)

### Root Fix (addresses Case 900 heating at the source)
The 6R2C thermal network needs restructuring to properly represent the south wall's thermal behavior. Key changes:

#### A. Separate Wall and Window Paths
Currently, all surfaces share the same `h_tr_is`. Instead, compute separate heat transfer paths:
- **Window path:** h_tr_w (interior air → window → exterior, direct)
- **Floor/ceiling path:** Uses existing h_tr_is
- **Opaque wall path:** Use `1/(1/h_tr_is_wall + 1/h_tr_em)` series combination instead of lumped h_tr_is

For the south wall with insulation (foam k=0.04):
- h_tr_is_south: interior film + radiation to interior surface
- h_tr_em: wall conduction from interior surface to exterior
- Effective wall coupling: h_wall_eff = 1/(1/h_tr_is_south + 1/h_tr_em)

This correctly represents that the wall's interior surface temperature is dominated by exterior conditions (through the insulation), not by interior air temperature.

#### B. Separate Solar Gain Distribution
Currently, all solar gains from all surfaces are lumped. Instead:
- Window solar gains: absorbed directly by interior air (high transmission)
- Opaque wall solar gains: absorbed by the wall's thermal mass (high thermal capacity)

For Case 900 south wall:
- Window (12m²): most solar transmits to interior → immediate air heating
- Opaque wall: solar absorbed by concrete block layer → slow release

#### C. Update derived_h_ext for Proper Wall Representation
```rust
// Window path (direct)
let h_window_path = h_tr_w + h_ve;

// Wall path (series: interior film + wall + exterior film)
// Use h_tr_is_series = 1/(1/h_tr_is_wall + 1/h_tr_em) for each wall
let h_wall_path = ...; // sum of wall effective conductances

// Total
derived_h_ext = h_window_path + h_wall_path;
```

This eliminates the bypass created by directly adding h_tr_em to derived_h_ext.

## Implementation Steps

### Phase 1: Immediate Fix (Minimal Risk)
```
File: src/sim/thermal_model_solvers.rs
- Add h_tr_em to derived_h_ext in update_optimization_cache()

File: src/sim/thermal_model_core.rs
- Reduce a_int from 0.5 to 0.1

Files: None (do NOT add h_tr_me to derived_term_rest_1)
```

### Phase 2: Thermal Network Restructure (Higher Risk, Requires Validation)
```
File: src/sim/thermal_model_core.rs
- Add per-surface-type h_tr_is computation (separate wall from floor/ceiling)
- Add h_tr_is_wall vector field
- Modify h_tr_em computation to use per-wall effective conductance

File: src/sim/thermal_model_solvers.rs
- Update derived_h_ext to use proper series + parallel combination
- Update derived_term_rest_1 to include h_tr_me for wall path
- Update sensitivity calculation with restructured network

File: src/sim/thermal_model_physics.rs
- Update solar gain distribution for wall vs window separation
- Update mass node energy balance for restructured paths

File: src/validation/ashrae_140_cases.rs
- Verify window parameters match ASHRAE 140 Table 7-27:
  * Case 600 (low-mass): single_clear_glass U=5.8, SHGC=0.86
  * Case 900 (high-mass): double_clear_glass U=2.10, SHGC=0.77
  (These are currently correct after ff8982a)
```

### Phase 3: Validation
- Run full ASHRAE 140 test suite
- Compare free-floating temperatures (Case 600FF, 900FF) with reference
- Verify HVAC heating/cooling mode detection across all cases
- Check sensitivity values match expected physics

## Verification Checklist

### HVAC Cooling Test (Case 600 scenario at 35°C outdoor)
- [ ] Free-float temperature reaches appropriate level for cooling trigger
- [ ] HVAC cooling mode activates
- [ ] Cooling energy > 0 kWh

### Case 900 Annual Energy
- [ ] Heating error < 15% (reference: 1.66 MWh)
- [ ] Cooling error < 15% (reference: 2.49 MWh)
- [ ] Free-float max temperature 40-50°C (summer peak, not 16°C)

### Case 600/650 Annual Energy
- [ ] Heating error < 15%
- [ ] Cooling error < 15%
- [ ] All HVAC mode transitions correct

## Known Pre-Existing Issues (Not Caused by Current Changes)
1. Case 900 heating error (92% before h_tr_em fix) — due to 6R2C wall representation
2. Case 900 cooling error (54% before any changes) — same root cause
3. Case 600 free-float temperature (14.87°C at noon in summer) — may indicate solar gain issues
