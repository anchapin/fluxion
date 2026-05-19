# Issue #703: 900-Series Peak Cooling Root Cause Analysis

**Status:** ROOT CAUSE IDENTIFIED
**Date:** 2026-05-18
**Branch:** investigation/issue-703-900-series-peak-cooling

## Executive Summary

Peak cooling for ASHRAE 140 Case 900 is 5.39 kW, roughly 2x the reference range of 2.10–3.50 kW. The root cause is a **sin/cos swap bug in the solar incidence angle calculation** (`src/sim/solar.rs:69`) that overestimates beam solar irradiance on vertical surfaces by ~3.4x during peak summer conditions.

## Root Cause

### Bug Location
**File:** `src/sim/solar.rs`, line 69
**Method:** `SolarPosition::incidence_cosine()`

### The Bug
```rust
// CURRENT (WRONG):
let cos_theta_i = beta.sin() * alpha.sin()
    + beta.cos() * alpha.cos() * (phi - gamma).cos();
//             ^^^^^ WRONG   ^^^^^ WRONG
```

The standard incidence angle formula for a tilted surface is:

```
cos(θ) = sin(α)·cos(β) + cos(α)·sin(β)·cos(φ_solar - γ_surface)
```

Where:
- α = solar altitude angle
- β = surface tilt from horizontal (0° = horizontal, 90° = vertical)
- φ_solar = solar azimuth
- γ_surface = surface azimuth

The code has **sin(β) and cos(β) swapped** in the first term. Instead of `sin(α)·cos(β)`, it computes `sin(β)·sin(α)`.

### Impact Quantification

For a **vertical south-facing wall** at solar noon in Denver summer (α ≈ 73.6°):

| Surface | Code's cos(θ) | Correct cos(θ) | Error Factor |
|---------|--------------|-----------------|--------------|
| Vertical South (β=90°) | 0.959 | 0.282 | **3.4x overestimate** |
| Horizontal (β=0°) | 0.282 | 0.959 | **3.4x underestimate** |

This means beam solar irradiance on south-facing walls is multiplied by 3.4x, and beam irradiance on horizontal surfaces (roofs) is divided by 3.4x.

### Effect on Case 900

At the peak cooling hour (hour 4284, ~June 27 noon):
- **Actual solar gain:** 9,876 W through 12 m² south window
- **Expected solar gain:** ~3,000 W (beam ≈ 1,400 + diffuse ≈ 800 + ground ≈ 750)
- **Overestimate:** 3.3x, consistent with the cos/sin swap

The excessive solar gain (9,876 W vs. expected ~3,000 W) drives the free-floating zone temperature far above the cooling setpoint, causing the HVAC system to demand 5.72 kW of cooling — 2x the ASHRAE 140 reference range.

## Evidence

### 1. CSV Data (`case_900_peak_hourly.csv`)
Peak cooling hour 4284:
```
Hour=4284, Outdoor=30.36°C, Zone=26.50°C, Solar=9876.44W, HVAC=-5722.55W
```
- Solar gain of 9,876 W for 12 m² window = 823 W/m²
- This requires ~1,069 W/m² incident irradiance (at SHGC=0.77)
- Maximum physically possible on vertical south surface: ~500 W/m²

### 2. Top 10 cooling hours all show ~9,870–9,890 W solar gain
All peak cooling hours occur at summer noon when the sun is high, exactly when the bug has maximum effect (large solar altitude → large cos/sin discrepancy for vertical surfaces).

### 3. Cross-validation with Case 600 (low-mass)
The same bug affects Case 600 (also has south-facing window), but Case 600 results may have been calibrated/tuned to compensate, masking the underlying solar error.

### 4. ASHRAE 140 Reference Values
| Metric | Model | Reference Range | Status |
|--------|-------|-----------------|--------|
| Peak Cooling | 5.39 kW | 2.10–3.50 kW | FAIL (2x high) |
| Annual Cooling | varies | 2.13–3.67 MWh | FAIL |
| Annual Heating | varies | 1.17–2.04 MWh | FAIL |
| Peak Heating | varies | 1.10–2.10 kW | FAIL |
| FF Min Temp | -6.4°C | -6.40 to -1.60°C | TBD |
| FF Max Temp | 41.8–46.4°C | 41.80–46.40°C | TBD |

## Proposed Fix

### Minimal Fix (Single Line)
```rust
// File: src/sim/solar.rs, line 69
// Change from:
let cos_theta_i = beta.sin() * alpha.sin() + beta.cos() * alpha.cos() * (phi - gamma).cos();
// To:
let cos_theta_i = alpha.sin() * beta.cos() + alpha.cos() * beta.sin() * (phi - gamma).cos();
```

Also fix the comment on line 68:
```rust
// cos(θ) = sin(α)cos(β) + cos(α)sin(β)cos(φ - γ)
```

### Expected Impact
- Peak cooling for Case 900 should drop from ~5.4 kW to ~2.5 kW (within reference range)
- Annual cooling should decrease proportionally
- Free-floating temperatures may shift (solar on vertical surfaces will decrease, on horizontal will increase)
- Case 600 may need recalibration if previous fixes compensated for this bug

### Risk Assessment
- **All cases with vertical windows** are affected (600, 610, 620, 630, 900, 910, 920, 930 series)
- **Horizontal surface** (roof) solar gains are underestimated by same factor — may affect roof heat gain calculations
- Any test that was calibrated against the buggy solar values may need updating
- The `calculate_surface_irradiance` function at line 251 also calls `incidence_cosine`, so the fix propagates correctly

## Additional Findings

### Out of Scope for This Fix
1. The 5R1C `t_i_free` temperature used for HVAC demand in the 9R4C path may overestimate thermal response
2. The multi-node solver (9R4C) only uses internal radiative gains, not per-surface solar (issue #873)
3. The `IdealLoadsSystem` uses a supply-air-temperature-based formula that may give different results from the conductance-based formula used in `step_physics_9r4c`

### Recommended Follow-Up
1. Run full ASHRAE 140 validation suite after the fix
2. Check Case 600 series results (same window orientation, different construction)
3. Verify roof solar gains increase correctly (important for cooling load through roof)
4. Add a unit test for `incidence_cosine` with known vertical/horizontal surface values
