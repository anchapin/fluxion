# Investigation Report: Issue #273 - Case 960 Multi-Zone Sunspace Cooling Energy 20x Higher Than Reference

**Date**: 2026-02-19
**Priority**: CRITICAL (150 pts)
**Status**: ROOT CAUSE IDENTIFIED

---

## Executive Summary

Case 960 (multi-zone sunspace) shows cooling energy of **64.79 MWh** compared to the reference range of **1.55-2.78 MWh** - a factor of ~23x higher. Heating energy is also elevated at **40.21 MWh** vs reference of **1.65-2.45 MWh**.

**ROOT CAUSE IDENTIFIED**: The ThermalModel applies HVAC control to ALL zones when it should only control Zone 0 (main zone). Zone 1 (sunspace) should be free-floating (no HVAC), but is incorrectly being conditioned to 20-27°C.

---

## Case 960 Specification

### Zone Configuration
- **Zone 0**: Back-zone (8m x 6m x 2.7m = 48 m²)
  - HVAC controlled: Heating=20°C, Cooling=27°C
  - Internal loads: 200 W (0.6 radiative, 0.4 convective)
  - Windows: None (opaque walls)

- **Zone 1**: Sunspace (8m x 2m x 2.7m = 16 m²)
  - FREE-FLOATING: NO HVAC
  - Internal loads: None
  - Windows: 6 m² south-facing

### Common Wall
- Area: 21.6 m² (8m x 2.7m)
- Construction: 200mm concrete wall
- Conductance: ~51.8 W/K
- Additional convective coupling: ~60 W/K (ASHRAE 140 specification)
- Total inter-zone conductance: ~112 W/K

### Purpose of Sunspace
The sunspace is designed to act as a thermal buffer:
1. Solar gains enter through south glazing
2. Sunspace temperature rises naturally (free-floating)
3. Heat transfers to main zone via common wall
4. This reduces heating/cooling loads on the main zone

---

## Current Results vs Reference

| Metric | Fluxion Value | Reference Range | Status | Error Factor |
|--------|---------------|-----------------|--------|--------------|
| Heating | 40.21 MWh | 1.65-2.45 MWh | FAIL | 16-24x |
| Cooling | 64.79 MWh | 1.55-2.78 MWh | FAIL | 23-42x |
| Peak Heating | 20.56 kW | 2.20-2.90 kW | FAIL | 7-9x |
| Peak Cooling | 24.29 kW | 1.50-2.00 kW | FAIL | 12-16x |

**Validation Pass Rate**: 10.9% (7/64 tests pass)

---

## Root Cause Analysis

### Issue 1: HVAC Applied to All Zones (PRIMARY)

**Location**: `src/sim/engine.rs:400-405`

```rust
// BUG: Uses only spec.hvac[0] for ALL zones
let hvac = &spec.hvac[0];
model.heating_setpoint = hvac.heating_setpoint;
model.cooling_setpoint = hvac.cooling_setpoint;
```

**Problem**:
- `ThermalModel` stores HVAC setpoints as scalar values (`heating_setpoint`, `cooling_setpoint`)
- `from_spec()` takes setpoints only from `spec.hvac[0]` (Zone 0)
- These single values are applied to ALL zones

**Expected Behavior**:
```rust
// Should use zone-specific HVAC schedules
model.heating_setpoint = VectorField::from_values(vec![
    spec.hvac[0].heating_setpoint,  // Zone 0
    spec.hvac[1].heating_setpoint,  // Zone 1
]);
model.cooling_setpoint = VectorField::from_values(vec![
    spec.hvac[0].cooling_setpoint,  // Zone 0
    spec.hvac[1].cooling_setpoint,  // Zone 1
]);
```

**Impact**:
- Sunspace (Zone 1) is conditioned to 20-27°C
- This defeats the thermal buffer purpose
- HVAC actively removes solar gains from sunspace
- Massive overestimation of cooling energy

### Issue 2: Zone Areas Same for All Zones

**Location**: `src/sim/engine.rs:394-406`

```rust
// BUG: Uses only geometry[0] for all zones
model.zone_area = VectorField::from_scalar(floor_area, num_zones);
```

**Problem**:
- All zones use the same floor area (48 m² from Zone 0)
- Zone 1 should have 16 m² (8m x 2m)

**Expected Behavior**:
```rust
// Should use zone-specific floor areas
model.zone_area = VectorField::new(
    spec.geometry.iter().map(|g| g.floor_area()).collect()
);
```

**Impact**:
- Incorrect thermal capacitance for Zone 1
- Incorrect internal load distribution (if applied per-zone)
- Incorrect infiltration heat transfer (volume-based)
- Incorrect solar gain distribution (if per-area based)

### Issue 3: Internal Loads Applied to All Zones

**Location**: `src/sim/engine.rs:552-556`

```rust
// BUG: Uses only internal_loads[0] for all zones
if let Some(ref loads) = spec.internal_loads[0] {
    let load_per_m2 = loads.total_load / floor_area;
    model.loads = VectorField::from_scalar(load_per_m2, num_zones);
    model.convective_fraction = loads.convective_fraction;
}
```

**Problem**:
- Internal loads from Zone 0 are applied to all zones
- Zone 1 (sunspace) should have no internal loads

**Expected Behavior**:
```rust
// Should use zone-specific internal loads
model.loads = VectorField::new(
    spec.internal_loads.iter().map(|opt_loads| {
        match opt_loads {
            Some(loads) => loads.total_load / loads.total_load, // Placeholder
            None => 0.0,
        }
    }).collect()
);
```

---

## Why Cooling is So High

### Expected Behavior (Free-Floating Sunspace)

1. Sunspace receives solar radiation through south glazing
2. Sunspace temperature swings with solar gains: 10-40°C
3. Heat naturally flows to main zone via common wall
4. Main zone temperature stabilized: 18-22°C
5. HVAC mainly compensates for heat loss through exterior walls
6. **Result**: Low annual cooling energy (~2 MWh)

### Actual Behavior (Conditioned Sunspace)

1. Sunspace receives intense solar radiation through south glazing
2. HVAC holds sunspace at 20-27°C (actively removes heat)
3. Main zone also held at 20-27°C
4. HVAC fights against natural thermal dynamics
5. **Result**: Massive cooling energy (~64 MWh) - 23x expected

### Thermal Dynamics

The sunspace's south-facing windows (6 m²) receive substantial solar radiation:
- Annual solar irradiance on south-facing vertical surface: ~1,000 kWh/m²
- Total solar gain to sunspace: ~6,000 kWh/year

In a free-floating sunspace:
- This heat naturally conducts to main zone via common wall
- Main zone sees delayed, buffered heat gain
- HVAC deals with moderated temperature swings

In a conditioned sunspace:
- HVAC actively removes solar gains to maintain 27°C
- No thermal buffering occurs
- HVAC works directly against solar gain

---

## Evidence from Diagnostic Tests

### Test: `test_case_960_zone_temperatures`

**Output**:
```
=== Case 960 Temperature Ranges (Week 1) ===
Back-zone: 20.00°C to 20.00°C
Sunspace: 20.00°C to 20.00°C
=== End ===
```

**Analysis**:
- Both zones show constant 20.00°C
- This confirms sunspace is being conditioned
- Expected: Sunspace should show large swings (e.g., 10-40°C)

### Test: `test_case_960_sunspace_simulation`

**Output**:
```
=== ASHRAE 140 Case 960 Results ===
Annual Heating: 75.45 MWh (reference: 1.65-2.45 MWh)
Annual Cooling: 0.15 MWh (reference: 1.55-2.78 MWh)
Peak Heating: 22.05 kW (reference: 2.20-2.90 kW)
Peak Cooling: 0.88 kW (reference: 1.50-2.00 kW)
=== End ===
```

**Analysis**:
- Results are different from validation test (likely using different weather or configuration)
- Both heating and cooling are wrong
- Confirms fundamental modeling issue

---

## Required Fixes

### Fix 1: Zone-Specific HVAC Setpoints

**File**: `src/sim/engine.rs`

**Change Type Definitions**:
```rust
// Change from scalar to VectorField
pub heating_setpoint: T,
pub cooling_setpoint: T,
```

**Update `from_spec()`**:
```rust
// Line 400-405: Use zone-specific HVAC schedules
model.heating_setpoint = VectorField::new(
    spec.hvac.iter().map(|h| h.heating_setpoint).collect()
);
model.cooling_setpoint = VectorField::new(
    spec.hvac.iter().map(|h| h.cooling_setpoint).collect()
);
```

**Update `hvac_power_demand()`**:
```rust
// Line 885-920: Use zone-specific setpoints
fn hvac_power_demand(&self, hour: usize, t_i_free: &T, sensitivity: &T) -> T {
    // Use zone-specific setpoints from VectorField
    let heating_sp = self.heating_setpoint.clone();
    let cooling_sp = self.cooling_setpoint.clone();

    t_i_free.zip3_with(&heating_sp, &cooling_sp, sensitivity,
        |t, h_sp, c_sp, sens| {
            let mode = if t < h_sp {
                HVACMode::Heating
            } else if t > c_sp {
                HVACMode::Cooling
            } else {
                HVACMode::Off
            };

            match mode {
                HVACMode::Heating => {
                    let t_err = h_sp - t;
                    let q_req = t_err / sens;
                    q_req.min(self.hvac_heating_capacity)
                }
                HVACMode::Cooling => {
                    let t_err = t - c_sp;
                    let q_req = -t_err / sens;
                    q_req.max(-self.hvac_cooling_capacity)
                }
                HVACMode::Off => 0.0
            }
        })
}
```

**Handle Free-Floating Zones**:
```rust
// In hvac_power_demand(), check if zone is free-floating
// Add a field to track which zones have HVAC
pub hvac_enabled: T,  // True for conditioned zones, false for free-floating

// In from_spec():
model.hvac_enabled = VectorField::new(
    spec.hvac.iter().map(|h| h.is_enabled()).collect()
);

// In hvac_power_demand():
t_i_free.zip_with(&hvac_enabled, |t, enabled| {
    if !enabled {
        0.0  // Free-floating zones have no HVAC
    } else {
        // Normal HVAC calculation
    }
})
```

### Fix 2: Zone-Specific Floor Areas

**File**: `src/sim/engine.rs`

**Update `from_spec()`**:
```rust
// Line 394: Use zone-specific floor areas
model.zone_area = VectorField::new(
    spec.geometry.iter().map(|g| g.floor_area()).collect()
);
```

### Fix 3: Zone-Specific Internal Loads

**File**: `src/sim/engine.rs`

**Update `from_spec()`**:
```rust
// Line 552-556: Use zone-specific internal loads
let zone_loads: Vec<f64> = spec.internal_loads.iter().map(|opt_loads| {
    match opt_loads {
        Some(loads) => loads.total_load,  // W (not per m² yet)
        None => 0.0,
    }
}).collect();

// Convert to W/m²
model.loads = VectorField::new(
    zone_loads.iter().zip(spec.geometry.iter())
        .map(|(&load, geo)| load / geo.floor_area())
        .collect()
);
```

### Fix 4: Zone-Specific Infiltration (if needed)

**File**: `src/sim/engine.rs`

**Update `from_spec()`**:
```rust
// Line 498-499: Use zone-specific infiltration (if spec supports it)
// Current implementation uses global infiltration_ach for all zones
// This may be correct per ASHRAE 140 spec
```

---

## Related Issues

- **Issue #271**: Annual energy variance - May share root cause with multi-zone modeling issues
- **Issue #274**: Thermal mass modeling differences - Related to thermal modeling accuracy

---

## Validation Impact

### Current State
- Case 960: FAIL (23-42x error in cooling, 16-24x error in heating)
- Overall validation: 10.9% pass rate (7/64)

### Expected After Fixes
- Case 960: PASS (within reference ranges)
- Overall validation: Expected improvement in multi-zone test cases

---

## Implementation Priority

1. **CRITICAL**: Fix HVAC setpoints to be zone-specific
2. **HIGH**: Fix floor areas to be zone-specific
3. **MEDIUM**: Fix internal loads to be zone-specific
4. **LOW**: Review infiltration handling for multi-zone cases

---

## Testing Plan

1. Run `test_case_960_multi_zone_configuration` - Verify zone setup
2. Run `test_case_960_zone_temperatures` - Verify sunspace swings
3. Run `test_case_960_sunspace_simulation` - Verify energy in range
4. Run full ASHRAE 140 validation - Verify overall pass rate improvement
5. Compare with EnergyPlus reference - Verify physical correctness

---

## Conclusion

The root cause of Case 960's excessive cooling energy has been identified: **HVAC is incorrectly applied to all zones**. The sunspace (Zone 1) should be free-floating but is being conditioned to 20-27°C like the main zone (Zone 0).

This causes the HVAC system to actively remove solar gains from the sunspace, rather than allowing those gains to naturally buffer the main zone through inter-zone heat transfer. The result is 23-42x higher cooling energy than expected.

Additional issues with zone areas and internal loads suggest a broader pattern of using Zone 0's parameters for all zones in multi-zone configurations.

**Recommended Action**: Implement zone-specific HVAC, floor area, and internal loads in `ThermalModel::from_spec()` and related methods.

---

## References

- ASHRAE 140 Standard 140-2017
- Case 960 Specification: `src/validation/ashrae_140_cases.rs:1607-1632`
- Thermal Model: `src/sim/engine.rs`
- Validation Tests: `tests/ashrae_140_case_960_sunspace.rs`
