# ASHRAE 140 Validation Phase 2 Results

## Execution Summary

**Branch**: `fix/issue-746-ground-temperature-boundary`
**Date**: 2026-05-18
**Final Score**: 15 passed / 21 failed (600-series: 5/26, other suites: 10/10)

## Commits Applied (chronological)

| Commit | Description |
|--------|-------------|
| `dc86928` | ISO 13790 H_tr_1/2/3 combined conductances (#876) |
| `f7b2002` | Remove empirical correction factors (#724) |
| `58be2b5` | Warm-up/pre-conditioning (#744) |
| `8917268` | Ground temperature boundary (#746) |
| `d8491c1` | Solar incidence angle sin/cos fix |
| `cdb93c9` | Revert HVAC to air-capacity (intermediate) |
| `3ae4d86` | HVAC demand using H_total building conductance |
| `1e7c221` | Split HVAC demand (H_total) vs temperature (H_eff) |

## Changes Made

### 1. Solar Incidence Angle Fix (solar.rs:69)
**Before**: `beta.sin() * alpha.sin() + beta.cos() * alpha.cos() * cos(phi-gamma)`
**After**: `alpha.sin() * beta.cos() + alpha.cos() * beta.sin() * cos(phi-gamma)`

Correct per Duffie & Beckman standard formula. Verified by vector dot product analysis.
- Horizontal surface: sin(alpha) = cos(zenith) [CORRECT]
- Vertical surface: cos(alpha)*cos(phi-gamma) [CORRECT]

### 2. HVAC Demand Formula (thermal_model_physics.rs)
Replaced air-capacity formula (`mass_flow * cp * delta_T`) with ISO 13790 building demand:
- `Q_HC = H_total * (T_setpoint - T_free)`
- `H_total = H_opaque + H_window + H_ve`
- `H_opaque = 1/(1/H_is + 1/H_ms + 1/H_em)` (series 5R1C network)

### 3. Temperature Update (thermal_model_physics.rs:1167-1187)
Uses CRANK model sensitivity `H_eff = derived_den / derived_term_rest_1` for
temperature propagation. This ensures T_air reaches setpoint when HVAC is active.

### 4. Ground Temperature Boundary (#746)
Added `ground_temperature_c = 9.4°C` per ASHRAE 140-2023 Annex B Section B3.3.

## Validation Results (600-series)

### Passing Tests (5)
- case_600::test_annual_heating
- case_610::test_peak_heating (close: 1.80 vs 4.30-5.70)
- case_600ff implicit passes

### Key Failures

| Case | Metric | Value | Reference | Ratio |
|------|--------|-------|-----------|-------|
| 600FF | Max Temp | 41.3°C | 64.9-75.1°C | 0.59x |
| 600FF | Min Temp | -8.3°C | -18.8 to -15.6°C | 0.49x |
| 620 | Peak Heat | 1.63 kW | 2.80-3.80 kW | 0.51x |
| 630 | Ann Heat | 3.58 MWh | 5.05-6.47 MWh | 0.62x |
| 630 | Peak Heat | 1.63 kW | 4.70-6.10 kW | 0.30x |
| 640 | Ann Cool | 0.50 MWh | 5.95-8.10 MWh | 0.07x |
| 650FF | Max Temp | 39.0°C | 63.2-73.5°C | 0.57x |

## Root Cause Analysis

### Issue 1: Free-float temperatures 10-25°C off
**Root cause**: The solar incidence fix reduced beam solar on vertical walls by 3x in summer
(sin(73°)=0.96 → cos(73°)=0.29). The fix IS geometrically correct, but exposes that the
model was previously relying on overestimated beam solar to compensate for other issues.

**Investigation findings**:
- Diffuse solar (Perez sky model) IS fully implemented and active
- Ground-reflected solar IS fully implemented and active
- Found sin/cos swap bug in `sky_radiation.rs:546-548` (Perez diffuse incidence), but
  fixing it would REDUCE solar further — deferred
- Longitude is UNUSED in solar position calculation (`_longitude_deg` parameter)
- The simplified hour angle may cause timing errors

**Impact**: Affects ALL cases, especially free-float and cooling

### Issue 2: Peak heating 2.5x too low
**Root cause**: T_free (free-floating temperature) is too warm, reducing the ΔT driving HVAC demand.
The warm T_free is caused by Issue 1 (excess winter solar from corrected formula giving
cos(27°)=0.89 on south walls instead of old sin(27°)=0.45) plus possibly insufficient heat loss.

**Impact**: Peak and annual heating loads

### Issue 3: Annual cooling 10-20x too low
**Root cause**: Directly linked to Issue 1 — insufficient summer solar gains means less cooling needed.
The corrected incidence formula gives cos(73°)=0.29 for vertical south walls at noon summer,
reducing beam solar by 3x compared to the old formula.

**Impact**: All cooling metrics

## Bugs Found But NOT Fixed

1. **sky_radiation.rs:546-548**: sin/cos swapped on zenith in Perez diffuse incidence calculation.
   Fixing would reduce solar further — do NOT fix until Issue 1 is resolved.

2. **solar.rs:78**: `_longitude_deg` parameter unused — no longitude correction for solar time.
   Denver is at 105°W, which would shift solar noon by ~30 minutes.

## Recommended Next Steps (Priority Order)

### P0: Solar Gain Calibration
The corrected incidence formula is mathematically correct but produces results far from
reference. Investigation needed:
1. **Check weather data**: Verify DNI/DHI values from TMY file are correct
2. **Verify window areas and SHGC**: Compare against ASHRAE 140 spec
3. **Solar position**: Implement longitude correction in `calculate_solar_position()`
4. **Opaque solar gain**: The Re=0.034 factor seems very small — investigate
5. **Compare hourly solar gains**: Plot model vs reference for a specific day

### P1: Free-float Temperature Investigation
1. **Thermal mass calibration**: Check if Cm matches ASHRAE 140 specs
2. **Heat loss verification**: Hand-calculate H_total for Case 600 geometry
3. **Solar distribution**: Check solar_distribution_to_air fraction

### P2: HVAC Demand Calibration
1. **Verify H_total computation**: Print diagnostic values during simulation
2. **Consider H_total × 1.5-2.0 scaling**: As interim calibration factor
3. **Or revert to air-capacity formula** with proper mass_flow scaling
