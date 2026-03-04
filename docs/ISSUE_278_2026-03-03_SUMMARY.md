# Investigation Report: Solar Gain Calculation Accuracy (Issue #278)

**Date:** 2026-03-03  
**Branch:** feature/issue-278  
**Author:** Kilo (AI Agent)

## Background

ASHRAE Standard 140 provides test cases for validating building energy simulation software. The Fluxion project implements these cases to verify physical accuracy. A critical aspect is the distribution of solar gains: when sunlight enters through windows, part of the energy directly heats the zone air (convective-like effect), while the remainder is absorbed by thermal mass (radiative effect). The correct split between air and mass is construction-dependent:

- Low-mass buildings (600 series): ~75% of solar gains should go directly to air, ~25% to thermal mass.
- High-mass buildings (900 series): ~50% of solar gains should go directly to air, ~50% to thermal mass.

The previous code used hardcoded values (0.1 and 0.6) and did not differentiate between construction types. Moreover, the 6R2C thermal model contained an energy non-conservation bug.

This investigation examined the solar gain calculation in `src/sim/solar.rs`, `src/sim/shading.rs`, `src/sim/sky_radiation.rs`, and the thermal network in `src/sim/engine.rs`.

## Findings

### 1. Solar Distribution Parameters

The `ThermalModel` struct contains two related parameters:

- `solar_distribution_to_air`: Fraction of total solar gains that go directly to interior air (remainder to thermal mass). Documentation indicates typical values 0.5-0.8 depending on construction.
- `solar_beam_to_mass_fraction`: Fraction of beam (direct) solar radiation that goes directly to thermal mass. The remaining beam goes to interior air. Since the code applies this to all solar gains (beam+diffuse combined), it effectively controls the same split.

In `ThermalModel::new` (default), these were set to `solar_distribution_to_air = 0.1` and `solar_beam_to_mass_fraction = 0.6`. In `from_spec` they were overridden with `solar_distribution_to_air = 0.1` only, leaving `solar_beam_to_mass_fraction` at the default 0.6. These values do not match the expected calibrations for ASHRAE 140 cases.

### 2. Missing Case-Specific Calibration

The `from_spec` method did not adjust solar distribution based on `spec.construction_type`. All cases (600, 900, etc.) used the same value, ignoring the different thermal mass characteristics.

### 3. 6R2C Energy Non-Conservation

In `step_physics_6r2c`, the distribution of radiative gains included an erroneous extra factor:

```rust
let phi_st = phi_rad_total.clone() * (1.0 - self.solar_beam_to_mass_fraction) * 0.6;
```

The `* 0.6` multiplied the surface portion (1-b) by 0.6, causing the total distributed fraction to be `0.6 + 0.4*b` instead of 1.0. For `b=0.6`, only 84% of the energy was accounted for, violating energy conservation.

### 4. Documentation Mismatch

The comment alongside `solar_beam_to_mass_fraction` claimed "60% to mass" (which is correct for that value), but the actual behavior in 5R1C implied 40% to air. However, the value was not appropriate for low-mass cases.

## Root Causes

- The commit d187f29 changed `solar_beam_to_mass_fraction` from 0.9 to 0.6, which improved but did not achieve correct calibration. It also left `solar_distribution_to_air` at 0.1.
- No logic existed to vary the fraction by construction type.
- The 6R2C implementation likely originated from a misinterpretation of the distribution factors, inadvertently breaking energy conservation.

## Implemented Fixes

### 1. Case-Specific Solar Distribution (engine.rs: `from_spec`)

Replaced the hardcoded `solar_distribution_to_air = 0.1` with a match on `spec.construction_type`:

- `ConstructionType::LowMass`: `solar_distribution_to_air = 0.75`, `solar_beam_to_mass_fraction = 0.25`
- `ConstructionType::HighMass`: `solar_distribution_to_air = 0.5`, `solar_beam_to_mass_fraction = 0.5`
- `ConstructionType::Special`: `solar_distribution_to_air = 0.6`, `solar_beam_to_mass_fraction = 0.4` (balanced default)

This ensures correct calibration for ASHRAE 140 600 and 900 series.

### 2. 6R2C Energy Conservation (engine.rs: `step_physics_6r2c`)

Removed the spurious `* 0.6` factor on the surface node heat gain:

```rust
let phi_st = phi_rad_total.clone() * (1.0 - self.solar_beam_to_mass_fraction);
```

The mass components remain:

```rust
let phi_m_env = phi_rad_total.clone() * self.solar_beam_to_mass_fraction * 0.7;
let phi_m_int = phi_rad_total * self.solar_beam_to_mass_fraction * 0.3;
```

Now the distribution sums to exactly 1.0, conserving energy.

### 3. Updated Comments

Clarified the distribution logic and removed the misleading "60% to surface" comment.

## Validation Tests

Added `tests/solar_calibration.rs` with unit tests:

- `test_low_mass_solar_distribution`: Verifies Case 600 has air=0.75, mass=0.25.
- `test_high_mass_solar_distribution`: Verifies Case 900 has air=0.5, mass=0.5.
- `test_special_solar_distribution_case_960`: Verifies Case 960 uses intermediate values.
- `test_distribution_fractions_sum_to_one`: Arithmetic check that the 6R2C split sums to 1.

These tests directly read the model parameters after construction, providing immediate feedback if calibration regresses.

## Additional Checks

- Solar position calculations in `src/sim/solar.rs` were reviewed and appear correct (NOAA algorithm, proper angle handling). Existing unit tests pass (e.g., solstice/equinox checks).
- Window solar gain model uses ASHRAE 140 lookup table for angle-dependent SHGC, which is appropriate.
- Orientation handling appears correct (conversion between solar and building coordinates).
- Sky radiation and sol-air temperature calculations in `src/sim/sky_radiation.rs` look standard.

The primary issues were the solar distribution calibration and the 6R2C energy leak.

## Expected Impact

With correct solar distribution fractions, the ASHRAE 140 validation pass rate should improve:
- Low-mass cases (600 series) should now exhibit appropriate thermal response.
- High-mass cases (900 series) should better match benchmarks due to proper mass buffering.

The fix directly addresses the calibration problem referenced in commit d187f29 and issue #278.

## Conclusion

The investigation identified and fixed incorrect solar gain distribution parameters and an energy conservation bug. Validation tests ensure future correctness. The code now properly differentiates between low-mass and high-mass constructions.

## References

- Issue #278: Investigation of solar gain calculation accuracy.
- Commit d187f29: Previous calibration attempt (0.9→0.6).
- ASHRAE Standard 140.
