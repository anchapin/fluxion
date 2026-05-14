# PR #821 Investigation Findings: 600FF/650FF/900FF/950FF Temperature Gap

## Summary

All four free-float (FF) cases underpredict peak summer temperatures by 10-21°C:

| Case | Actual Max | Reference Max | Gap | Hours to Peak |
|------|-----------|---------------|-----|---------------|
| 600FF | 54.61°C (STILL FAILING) | 64.9-75.1°C | **-10.3 to -20.5°C LOW** | Hour 17 (Jul 17) |
| 650FF | ~52°C (STILL FAILING) | 63.2-73.5°C | **-11 to -21°C LOW** | Hour 17 |
| 900FF | ~26°C (STILL FAILING) | 41.8-46.4°C | **-16 to -20°C LOW** | Hour 17 |
| 950FF | TBD | TBD | TBD | TBD |

## FIX ATTEMPTED - DID NOT WORK

**Fix Applied:** Commented out lines 68-72 in `thermal_model_solvers.rs:68-72` which were overwriting thermal_capacitance with hardcoded structure_cap.

**Result:** Test STILL FAILS with identical temperature (54.61°C before and after fix).

**Conclusion:** The thermal_capacitance double-assignment was NOT the root cause, OR there are multiple issues.

## Next Investigation Direction

Need to investigate OTHER potential root causes:
1. **h_tr_ms calculation** - Physics-based h_tr_ms may not match reference program
2. **Solar gain magnitude** - May be too low for lightweight constructions
3. **Construction layer definitions** - Low-mass construction may be too light

## Root Cause: t_i_free Calculation

The zone air temperature for free-float cases is calculated in `thermal_model_physics.rs:864-867`:

```rust
let mut t_i_free = num_tm;          // h_tr_ms * T_mass
t_i_free.add_assign(&num_phi_st);   // h_tr_is * T_surface (but phi_st=0 for FF)
t_i_free.add_assign(&num_rest_with_iz); // rest of heat balance
t_i_free.div_assign(&den);          // denominator
```

Where:
```
num_tm = h_tr_ms * T_mass
num_phi_st = h_tr_is * T_surface (0 for free-float since HVAC off)
num_rest = h_tr_is * T_ext + h_ext * T_ext + h_tr_floor * T_g + phi_ia
den = h_tr_ms * h_tr_is + h_tr_is * (h_ext + h_tr_floor)
```

## Debug Data Captured

### 600FF Peak Temperature Analysis (timestep 3688, hour 16, July 17)

```
DEBUG_MAX t=3688 (hour 16) T_air=50.86°C T_mass=107.54°C t_i_free=51.00°C
den=383063.8844 h_ext=149.2945 h_tr_ms=121.7178 h_tr_is=1251.3240 h_tr_floor=8.8176
t_g=10.00 phi_ia=0.00W phi_st=0.00W phi_m=14202.01W solar=12920.14W outdoor=13.90°C
```

### Key Observations

1. **HVAC is correctly OFF:**
   - phi_ia = 0.00W (no internal convective gains for free-float)
   - phi_st = 0.00W (no surface convective gains for free-float)
   - Free-float mode properly active

2. **Solar gains going to mass (phi_m):**
   - phi_m = 14,202 W (internal mass gains, including solar to mass)
   - solar = 12,920 W (window solar transmission)
   - This is CORRECT - solarDistributionToAir=0.0 per ASHRAE 140 means ALL solar goes to surfaces/mass

3. **Thermal mass is excessively hot:**
   - T_mass = 107.54°C (unphysical for building interior)
   - T_air = 50.86°C (should be 65-75°C per reference)
   - Temperature differential: 107.54 - 50.86 = **56.68°C**

4. **h_tr_ms = 121.72 W/K:**
   - Breakdown: h_ms_physics=70.667, h_ms_roof=32.877, h_ms_floor=18.174
   - This is physics-calculated (not case-specific tuning)
   - Leads to: τ = Cm / h_tr_ms ≈ (wall + roof + floor thermal mass) / 121.72

## Heat Flow Analysis at Peak (600FF, hour 16)

```
Zone Air Heat Balance:
================ GAINS ================  ========== LOSSES ==========
h_tr_ms * (T_mass - T_air) = 121.72 * (107.54 - 50.86) = 6,899 W → MASS to AIR
h_tr_is * (T_surface - T_air) = 1251 * small_diff ≈ 0 W (surface ≈ air)
h_ext * (T_outdoor - T_air) = 149.29 * (13.9 - 50.9) = -5,524 W → LOSS to exterior
h_tr_floor * (T_g - T_air) = 8.82 * (10 - 50.9) = -361 W → LOSS to ground
phi_ia = 0 W (HVAC off)
phi_st = 0 W (HVAC off)

Net balance doesn't make sense: gains should equal losses for steady state
```

**Key insight:** The physics calculation seems correct but T_mass = 107°C is unphysical. This suggests the thermal mass is storing too much solar energy and releasing it back to the zone air more slowly than expected.

## Possible Root Causes

### 1. Thermal Mass Time Constant Too Long
- τ = Cm / h_tr_ms for 600FF should be ~1-2 hours (lightweight construction)
- But the mass temperature reaching 107°C suggests it's accumulating heat over many hours
- The 5R1C model may not properly handle the rapid charging/discharging of lightweight mass

### 2. h_tr_ms May Be Too Low (Not Too High)
- If h_tr_ms is too low, the mass doesn't release heat fast enough to zone air
- But the value 121.72 W/K is physics-calculated from construction layers
- Need to verify against ASHRAE 140 reference implementations

### 3. Solar Gain Partitioning to Mass
- phi_m = 14,202 W includes ALL solar gains (12,920 W) + internal gains
- But ASHRAE 140 may specify different distribution
- Need to check Case 600FF construction and compare with EnergyPlus reference

### 4. CTF vs Full Physics for 900FF
- 900FF has ctf_primary=true (CTF enabled)
- 600FF does NOT (uses full 5R1C)
- Both show large temperature gaps - the problem isn't CTF-specific

## Files to Investigate

- `src/sim/thermal_model_physics.rs:864-867` - t_i_free calculation for free-float
- `src/sim/thermal_model_core.rs:877-960` - h_tr_ms physics-based calculation
- `src/sim/thermal_model_core.rs:1486-1491` - solarDistributionToAir setting
- `src/sim/thermal_integration.rs:73` - mass node energy balance equation

## Current Status: Fix Applied But Test Still Failing

The fix (removing the hardcoded structure_cap overwrite) was applied but the 600FF test still fails with the same max temperature (~54.61°C vs reference 64.9-75.1°C).

This suggests the thermal_capacitance overwrite in `update()` was NOT the root cause, OR there are other issues that need investigation.

### What Was Done
1. Identified potential issue: `thermal_model_solvers.rs:68-72` was overwriting correct `thermal_capacitance` with hardcoded value
2. Applied fix: Commented out lines 68-73 to preserve correct Cm from `from_spec()`
3. Build succeeds
4. Test still fails

### Remaining Investigation Needed

1. **Verify Cm is actually correct**: Add debug output in `from_spec()` to confirm thermal_cap_vec values for 600FF
2. **Check if update() is called for FF cases**: Trace how ThermalModel is created and whether `update()` runs
3. **Other possible root causes**:
   - h_tr_ms might be wrong
   - Solar gain distribution might be wrong
   - Ground temperature might be affecting results
   - The physics model (5R1C) might be fundamentally limited for lightweight construction

## Root Cause Identified (PARTIAL): Double Assignment of thermal_capacitance + Wrong Value

### Summary

There are TWO issues with thermal_capacitance:

1. **Double Assignment**: `thermal_capacitance` is set correctly in `from_spec()` using actual construction, but then **overwritten** in `update()` with a hardcoded value.

2. **Wrong Value**: Even if not overwritten, the hardcoded value `zone_area * 200_000.0` is **~15x too high** for low-mass construction.

### Issue 1: Double Assignment (CRITICAL)

In `thermal_model_core.rs:1416`:
```rust
model.thermal_capacitance = VectorField::new(thermal_cap_vec);  // CORRECT: from construction
```

But then in `thermal_model_solvers.rs:71-72`:
```rust
let structure_cap = self.0.zone_area.clone() * 200_000.0;  // WRONG: hardcoded 200 kJ/m²K
self.0.thermal_capacitance = air_cap + structure_cap;  // OVERWRITES correct value!
```

This `update()` is called at the END of `from_spec()`, so the correct `thermal_cap_vec` is discarded.

### Issue 2: Wrong Hardcoded Value

The hardcoded value `200,000 J/m²K` is ~15x too high for low-mass construction:

| Construction | Actual Cm (J/m²K) | Hardcoded | Overestimate |
|--------------|------------------|-----------|--------------|
| Low-mass wall | ~12,900 | 200,000 | **15.5x** |
| Low-mass roof | ~10,000 | 200,000 | **20x** |
| Low-mass floor | ~25,000 | 200,000 | **8x** |

### Impact on Time Constant

Time constant τ = Cm / h_tr_ms

For 600FF (floor area = 48 m², h_tr_ms = 121.72 W/K):
- **Correct**: τ = 624,000 J/K / 121.72 W/K = **5,100 seconds = 1.4 hours**
- **Current (overwritten)**: τ = 9,600,000 J/K / 121.72 W/K = **78,900 seconds = 21.9 hours**

The time constant is **15x too long**, meaning:
1. Mass charges too slowly during the day
2. Mass doesn't release heat fast enough to zone air
3. Zone air temperature stays too low (51°C vs 65-75°C reference)

## Proposed Fix (APPLIED - BUT TEST STILL FAILING)

In `src/sim/thermal_model_solvers.rs`, REMOVE or COMMENT OUT lines 68-72:

```rust
// REMOVED Issue #821 FIX: thermal_capacitance was being overwritten with wrong value
// The correct thermal_capacitance is calculated in from_spec() using actual construction layers.
// This overwrite (200,000 J/m²K hardcoded) was causing 15x overestimate of thermal mass,
// leading to time constants 15x too long and temperatures 10-20°C too low for FF cases.
// let structure_cap = self.0.zone_area.clone() * 200_000.0;
// self.0.thermal_capacitance = air_cap + structure_cap;
```

The fix was applied, but the test still fails. This means either:
1. The thermal_capacitance overwrite was NOT the root cause, OR
2. There's another issue also affecting the results, OR
3. The fix needs additional changes

## Files to Fix

- `src/sim/thermal_model_solvers.rs:68-72` - REMOVE hardcoded structure_cap assignment (DONE)
- ADDITIONAL INVESTIGATION NEEDED: The fix alone didn't resolve the issue
